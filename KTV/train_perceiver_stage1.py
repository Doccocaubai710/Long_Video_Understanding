"""
train_perceiver_stage1.py

Giai doan 1 (anh) cua de xuat GOC cua leader -- Perceiver Resampler + Anchor
Loss -- xem idea "Perceiver + Anchor Loss (ban goc cua leader)" tren web.
KHONG lien quan RL-Gating (khac hoan toan co che, xem file do rieng).

Y tuong: 1 Perceiver Resampler nho (query hoc duoc, so luong CO DINH,
mac dinh 64) doc qua token anh goc T1 (tu chinh vision encoder cua Qwen),
tong hop thanh T2 (64 token tom tat). T2 duoc dua vao LLM (Qwen, DONG BANG)
de sinh caption, so voi caption that (COCO) tinh cross-entropy -- day la
tin hieu chinh day Perceiver hoc tom tat co ich. Dong thoi, Anchor Loss ep
moi T2_j phai "gan" it nhat 1 T1_i nao do (cosine similarity), tranh
Perceiver "troi dat" thanh bieu dien la voi chinh Qwen.

loss = CE(caption that | T2) + anchor_weight * L_anchor
L_anchor = 1 - mean_j( max_i cos(T2_j, T1_i) )

CHI Perceiver duoc train. Qwen luon dong bang (torch.no_grad() khi encode
anh va khi chay LLM -- gradient chi chay qua T2 nguoc ve Perceiver).

DA KIEM CHUNG TREN MAY THAT (2026-09-16): `llm.get_image_features(...)` va
`model.config.image_token_id` (=151655) dung ten. Diem can nho: phai lay
`.pooler_output` (3584 chieu, SAU hop nhat/chieu sang khong gian LLM) --
KHONG phai `.last_hidden_state` (1280 chieu, vision_config.hidden_size,
TRUOC hop nhat) -- lay nham cho da gay loi CUDA device-side assert kho doc
o masked_scatter (source qua nho so voi mask) trong lan chay dau tien.

Usage (thu tren val2017 truoc, it anh, it step -- vai phut):
  python train_perceiver_stage1.py \
    image_dir=Datasets/COCO/val2017 \
    captions_file=Datasets/COCO/annotations/captions_val2017.json \
    model_path=Qwen/Qwen2.5-VL-7B-Instruct \
    num_queries=64 num_layers=2 anchor_weight=0.5 \
    lr=1e-4 num_steps=200 log_every=10 ckpt_every=50 \
    policy_save_path=outputs/perceiver_stage1_smoke.pt
"""

import json
import os
import random

import hydra
import torch
import torch.nn as nn
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from PIL import Image
from qwen_vision_process_fastvid import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MIN_PIXELS = 16 * 28 * 28
SYSTEM_PROMPT = "You are a helpful assistant."
CAPTION_PROMPT = "Describe this image in one short sentence."


def resolve_path(path):
    if path is None:
        return None
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    return to_absolute_path(path)


def load_coco_captions(captions_file):
    """Tra ve list[{file_name, caption}] -- 1 dong/caption (1 anh co the
    xuat hien nhieu lan voi caption khac nhau, giu nguyen nhu vay de tan
    dung het du lieu thay vi chi lay 1 caption/anh)."""
    d = json.load(open(captions_file, "r", encoding="utf-8"))
    img_by_id = {im["id"]: im["file_name"] for im in d["images"]}
    rows = []
    for ann in d["annotations"]:
        fname = img_by_id.get(ann["image_id"])
        if fname is None:
            continue
        rows.append({"file_name": fname, "caption": ann["caption"].strip()})
    return rows


def load_recaptioned_jsonl(captions_file):
    """Doc file do recaption_coco_detailed.py sinh ra -- 1 dong/ANH (khong
    phai 1 dong/caption nhu load_coco_captions), dinh dang
    {"file_name":..., "caption":...} -- caption o day la ban DA GOP 5
    caption goc thanh 1 caption chi tiet, on dinh (khong doi qua tung buoc
    train nua)."""
    rows = []
    with open(captions_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append({"file_name": row["file_name"], "caption": row["caption"].strip()})
    return rows


def load_captions_auto(captions_file):
    """Tu nhan dien dinh dang qua duoi file: .jsonl = file da recaption
    (load_recaptioned_jsonl), .json = file goc COCO (load_coco_captions)."""
    if str(captions_file).endswith(".jsonl"):
        return load_recaptioned_jsonl(captions_file)
    return load_coco_captions(captions_file)


class PerceiverResampler(nn.Module):
    """num_queries token hoc duoc, moi lop: SELF-attention (cac query "ban
    bac" voi nhau truoc, kieu Q-Former -- THEM MOI so voi ban truoc, ly do
    xem duoi) roi moi CROSS-attention (query doc T1) + FFN, co residual+
    LayerNorm chuan. Output luon co dung num_queries token bat ke T1 dai
    ngan bao nhieu -- day la diem mau chot lam T2 CO DINH so luong.

    VI SAO THEM SELF-ATTENTION (2026-09-23): ban truoc KHONG co buoc nay --
    64 query cross-attention sang T1 HOAN TOAN DOC LAP voi nhau, khong
    "biet" cac query khac dang chu y vung nao. visualize_perceiver_attention.py
    da cho thay dau hieu nhieu query cho ra ban do chu y gan giong het nhau
    (nghi ngo mode collapse) -- giong Q-Former (BLIP-2), them self-attention
    giua cac query truoc cross-attention de chung co the "chia nhau" phu
    trach cac vung/khia canh khac nhau cua anh, giam trung lap."""

    def __init__(self, dim, num_queries=64, num_layers=2, num_heads=8, ffn_mult=4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.self_attns = nn.ModuleList([
            nn.MultiheadAttention(dim, num_heads, batch_first=True) for _ in range(num_layers)
        ])
        self.ln_self = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(dim, num_heads, batch_first=True) for _ in range(num_layers)
        ])
        self.ln_q = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.ln_kv = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.ffns = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * ffn_mult), nn.GELU(),
                          nn.Linear(dim * ffn_mult, dim))
            for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(dim)

    def forward(self, T1):
        """T1: [n_tokens, dim] (1 anh/video, batch=1 -- don gian hoa cho
        vong lap tung-anh-mot giong het train_rl_gating.py)."""
        q = self.query.unsqueeze(0)  # [1, num_queries, dim]
        kv = T1.unsqueeze(0)  # [1, n_tokens, dim]
        for self_attn, ln_s, cross_attn, ln_q, ln_kv, ffn in zip(
            self.self_attns, self.ln_self, self.cross_attns, self.ln_q, self.ln_kv, self.ffns
        ):
            q_norm = ln_s(q)
            self_out, _ = self_attn(q_norm, q_norm, q_norm)
            q = q + self_out
            q_norm2 = ln_q(q)
            kv_norm = ln_kv(kv)
            cross_out, _ = cross_attn(q_norm2, kv_norm, kv_norm)
            q = q + cross_out
            q = q + ffn(q)
        return self.final_ln(q).squeeze(0)  # [num_queries, dim]

    def forward_with_attn(self, T1):
        """Giong forward() het, nhung tra ve them attention weight CROSS
        (query -> T1) cua MOI lop (de visualize) -- khong dung trong luc
        train (chi de xem, nen tach ham rieng thay vi doi forward() chinh,
        tranh anh huong code training dang goi forward() binh thuong)."""
        q = self.query.unsqueeze(0)
        kv = T1.unsqueeze(0)
        attn_maps = []  # moi phan tu: [num_queries, n_tokens], da average qua num_heads
        for self_attn, ln_s, cross_attn, ln_q, ln_kv, ffn in zip(
            self.self_attns, self.ln_self, self.cross_attns, self.ln_q, self.ln_kv, self.ffns
        ):
            q_norm = ln_s(q)
            self_out, _ = self_attn(q_norm, q_norm, q_norm)
            q = q + self_out
            q_norm2 = ln_q(q)
            kv_norm = ln_kv(kv)
            cross_out, attn_w = cross_attn(q_norm2, kv_norm, kv_norm, need_weights=True, average_attn_weights=True)
            attn_maps.append(attn_w.squeeze(0).detach())  # [num_queries, n_tokens]
            q = q + cross_out
            q = q + ffn(q)
        return self.final_ln(q).squeeze(0), attn_maps


def anchor_loss(T2, T1):
    """L_anchor = 1 - mean_j( max_i cos(T2_j, T1_i) ) -- moi token tom tat
    phai gan it nhat 1 token anh goc nao do, khong duoc troi qua xa."""
    T2n = torch.nn.functional.normalize(T2, dim=-1)
    T1n = torch.nn.functional.normalize(T1, dim=-1)
    sim = T2n @ T1n.t()  # [num_queries, n_tokens]
    max_sim, _ = sim.max(dim=-1)
    return 1.0 - max_sim.mean()


def run_one_image(model, processor, perceiver, optimizer, image_path, caption, args, max_pixels, device):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": image_path, "max_pixels": max_pixels, "min_pixels": MIN_PIXELS},
            {"type": "text", "text": CAPTION_PROMPT},
        ]},
    ]
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, _ = process_vision_info(messages, return_video_kwargs=True)
    prompt_inputs = processor(
        text=[prompt_text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt",
    ).to(device)

    llm = model.model if hasattr(model, "model") else model
    image_token_id = model.config.image_token_id  # xem canh bao o dau file neu loi o day

    with torch.no_grad():
        # T1: token anh goc tu chinh vision encoder cua Qwen (dong bang).
        image_feat = llm.get_image_features(
            prompt_inputs["pixel_values"], image_grid_thw=prompt_inputs["image_grid_thw"]
        )  # xem canh bao o dau file neu loi o day
        # DUNG .pooler_output (3584 chieu, SAU hop nhat/chieu sang khong
        # gian LLM), KHONG phai .last_hidden_state (1280 chieu, TRUOC hop
        # nhat -- vision_config.hidden_size) -- da xac nhan qua config
        # (out_hidden_size=3584) va qua loi thuc te (T2 tu Perceiver tren
        # last_hidden_state ra 1280, khong khop text_embeds 3584). Dung quy
        # uoc DA XAC NHAN DUNG voi video (xem run_inference_qwen25vl_dyseg.py:
        # video_feat.pooler_output).
        if hasattr(image_feat, "pooler_output"):
            image_feat = image_feat.pooler_output
        T1 = image_feat if not isinstance(image_feat, (list, tuple)) else torch.cat(image_feat, dim=0)
        T1 = T1.float()

    # ---------- Perceiver (grad ON) ----------
    T2 = perceiver(T1.detach())  # [num_queries, D]
    n_queries = T2.shape[0]

    # ---------- Build LLM input: system+user text KHONG doi, nhung thay
    # placeholder anh (dung so luong token that cua T1, do processor tu
    # chen) bang dung n_queries placeholder de ghep T2 vao. Ghep them
    # caption that lam target (teacher-forced) sau prompt. ----------
    caption_ids = processor.tokenizer(caption, add_special_tokens=False).input_ids
    eos_id = processor.tokenizer.eos_token_id

    full_input_ids = prompt_inputs["input_ids"][0]
    image_mask_full = (full_input_ids == image_token_id)
    n_image_tok_real = int(image_mask_full.sum().item())
    # Placeholder thuc te processor da chen co the KHAC n_queries (vi no
    # chen theo so luong token that cua T1, khong biet truoc Perceiver se
    # nen xuong bao nhieu) -- xay lai chuoi input_ids voi DUNG n_queries
    # token anh, giu nguyen phan text truoc/sau.
    pre_ids = full_input_ids[: image_mask_full.nonzero(as_tuple=True)[0][0]]
    post_ids = full_input_ids[image_mask_full.nonzero(as_tuple=True)[0][-1] + 1:]
    image_placeholder = torch.full((n_queries,), image_token_id, dtype=full_input_ids.dtype, device=device)

    prompt_ids = torch.cat([pre_ids, image_placeholder, post_ids])
    caption_ids_t = torch.tensor(caption_ids + [eos_id], dtype=full_input_ids.dtype, device=device)
    input_ids = torch.cat([prompt_ids, caption_ids_t]).unsqueeze(0)  # [1, seq]
    labels = torch.cat([
        torch.full((prompt_ids.shape[0],), -100, dtype=torch.long, device=device),
        caption_ids_t.clone(),
    ]).unsqueeze(0)

    with torch.no_grad():
        text_embeds = model.get_input_embeddings()(input_ids)  # [1, seq, D]

    n_image_tok_final = int((input_ids[0] == image_token_id).sum().item())
    assert n_image_tok_final == n_queries, (
        f"Lech so luong: input_ids sau khi xay lai co {n_image_tok_final} vi tri image_token_id, "
        f"nhung T2 chi co {n_queries} hang -- se lam masked_scatter loi (CUDA assert kho doc). "
        f"n_image_tok_real (truoc khi xay lai)={n_image_tok_real}, "
        f"pre_ids len={pre_ids.shape[0]}, post_ids len={post_ids.shape[0]}, "
        f"image_placeholder len={image_placeholder.shape[0]}, "
        f"caption_ids len={len(caption_ids)}."
    )
    image_mask = (input_ids[0] == image_token_id).unsqueeze(0).unsqueeze(-1)
    final_embeds = text_embeds.masked_scatter(image_mask, T2.to(text_embeds.dtype))
    attention_mask = torch.ones_like(input_ids)

    # T2 khong theo luoi khong gian/thoi gian goc cua anh nua (la ban tom
    # tat tru tuong do Perceiver tao ra) -- dung position_ids TUAN TU binh
    # thuong (giong cach Qwen xu ly doan text thuan) thay vi co gang tai
    # tao mRoPE 2D theo grid anh that, vi n_queries khong khop image_grid_thw
    # goc nua.
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len, device=device).view(1, 1, seq_len).expand(3, 1, seq_len)

    out = model(
        inputs_embeds=final_embeds, position_ids=position_ids, attention_mask=attention_mask,
        labels=labels, return_dict=True,
    )
    caption_loss = out.loss

    a_loss = anchor_loss(T2, T1.detach())
    anchor_weight = float(getattr(args, "anchor_weight", 0.5))
    loss = caption_loss + anchor_weight * a_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(perceiver.parameters(), max_norm=1.0)
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "caption_loss": float(caption_loss.item()),
        "anchor_loss": float(a_loss.item()),
        "n_image_tok_real": n_image_tok_real,
    }


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    max_pixels_units = int(getattr(args, "max_pixels_units", 1024))
    max_pixels = max_pixels_units * 28 * 28

    model_path = getattr(args, "model_path", "Qwen/Qwen2.5-VL-7B-Instruct")
    print(f"Loading {model_path} ... (frozen, no grad)")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=dtype, device_map="auto")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(model_path, max_pixels=max_pixels, min_pixels=MIN_PIXELS)

    image_dir = resolve_path(args.image_dir)
    captions_file = resolve_path(args.captions_file)
    rows = load_captions_auto(captions_file)
    random.seed(0)
    random.shuffle(rows)
    print(f"Loaded {len(rows)} (image, caption) pairs from {captions_file}")

    # Suy video_dim tu 1 anh mau -- giong cach train_rl_gating.py suy video_dim.
    probe = rows[0]
    probe_path = os.path.join(image_dir, probe["file_name"])
    with torch.no_grad():
        probe_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": probe_path, "max_pixels": max_pixels, "min_pixels": MIN_PIXELS},
                {"type": "text", "text": CAPTION_PROMPT},
            ]},
        ]
        pt = processor.apply_chat_template(probe_messages, tokenize=False, add_generation_prompt=True)
        pi, pv, _ = process_vision_info(probe_messages, return_video_kwargs=True)
        p_inputs = processor(text=[pt], images=pi, videos=pv, padding=True, return_tensors="pt").to(device)
        llm = model.model if hasattr(model, "model") else model
        probe_feat = llm.get_image_features(p_inputs["pixel_values"], image_grid_thw=p_inputs["image_grid_thw"])
        if hasattr(probe_feat, "pooler_output"):
            probe_feat = probe_feat.pooler_output
        probe_T1 = probe_feat if not isinstance(probe_feat, (list, tuple)) else torch.cat(probe_feat, dim=0)
        dim = probe_T1.shape[-1]
    print(f"Inferred image token dim: {dim}")

    num_queries = int(getattr(args, "num_queries", 64))
    num_layers = int(getattr(args, "num_layers", 2))
    perceiver = PerceiverResampler(dim, num_queries=num_queries, num_layers=num_layers).to(device).float()
    lr = float(getattr(args, "lr", 1e-4))
    optimizer = torch.optim.Adam(perceiver.parameters(), lr=lr)

    resume_from = getattr(args, "resume_from", None)
    start_step = 0
    if resume_from:
        ckpt = torch.load(resolve_path(resume_from), map_location=device)
        perceiver.load_state_dict(ckpt["state_dict"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        start_step = int(ckpt.get("step", 0))
        print(f"Resumed from {resume_from}, step={start_step}")

    num_steps = int(getattr(args, "num_steps", 200))
    log_every = int(getattr(args, "log_every", 10))
    ckpt_every = int(getattr(args, "ckpt_every", 50))
    policy_save_path = resolve_path(getattr(args, "policy_save_path", "outputs/perceiver_stage1.pt"))
    os.makedirs(os.path.dirname(policy_save_path), exist_ok=True)
    metrics_log_path = policy_save_path + ".metrics.jsonl"
    metrics_f = open(metrics_log_path, "a", encoding="utf-8")

    def save_ckpt(path, step):
        torch.save({"state_dict": perceiver.state_dict(), "dim": dim, "num_queries": num_queries,
                    "num_layers": num_layers, "step": step, "optimizer_state": optimizer.state_dict()}, path)

    step = start_step
    idx = start_step
    running_loss, running_caption, running_anchor = [], [], []
    saved_step_ckpts = []
    pbar = tqdm(total=num_steps, initial=start_step)
    while step < num_steps:
        row = rows[idx % len(rows)]
        idx += 1
        image_path = os.path.join(image_dir, row["file_name"])
        if not os.path.exists(image_path):
            continue
        try:
            stats = run_one_image(model, processor, perceiver, optimizer, image_path, row["caption"],
                                   args, max_pixels, device)
        except Exception as e:
            print(f"SKIP step {step} ({row['file_name']}): {type(e).__name__}: {e}")
            torch.cuda.empty_cache()
            continue

        running_loss.append(stats["loss"])
        running_caption.append(stats["caption_loss"])
        running_anchor.append(stats["anchor_loss"])
        step += 1
        pbar.update(1)
        metrics_f.write(json.dumps({"step": step, **stats}) + "\n")
        metrics_f.flush()
        if step % log_every == 0:
            print(
                f"step={step} loss={stats['loss']:.4f} caption_loss={stats['caption_loss']:.4f} "
                f"anchor_loss={stats['anchor_loss']:.4f} n_image_tok_real={stats['n_image_tok_real']} "
                f"(running loss={sum(running_loss)/len(running_loss):.4f})"
            )
        if ckpt_every > 0 and step % ckpt_every == 0:
            ckpt_path = policy_save_path + f".step{step}"
            save_ckpt(ckpt_path, step)
            save_ckpt(policy_save_path, step)
            saved_step_ckpts.append(ckpt_path)
            # Perceiver o dim=3584 khien moi checkpoint (state_dict +
            # optimizer Adam) nang ~500MB-1GB -- khong xoa file step cu se
            # lap day o dia sau vai nghin step (giong loi da gap o
            # train_perceiver_stage2.py). Chi giu lai keep_last_n ban step
            # gan nhat lam diem resume an toan.
            keep_last_n = int(getattr(args, "keep_last_n_ckpts", 3))
            while len(saved_step_ckpts) > keep_last_n:
                old_path = saved_step_ckpts.pop(0)
                if os.path.exists(old_path):
                    os.remove(old_path)
            print(f"[checkpoint] step={step} -> {ckpt_path} (and {policy_save_path})")
    pbar.close()
    metrics_f.close()

    save_ckpt(policy_save_path, step)
    print(f"[train_perceiver_stage1] done -> saved policy to {policy_save_path}")
    print(f"Per-step metrics logged to {metrics_log_path}")
    print(f"Final running loss={sum(running_loss)/len(running_loss):.4f}")


@hydra.main(config_path=None, version_base=None)
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
