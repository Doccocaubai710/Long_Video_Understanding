"""
train_rap.py -- train Perceiver co neo (rap_model.py) TRUC TIEP tren video, muc giu 25%.

Khong con giai doan anh rieng. Thay vao do la warm-up kieu SCORE (arXiv
2603.26365): pseudo-video = moi anh lap lai nhieu frame, noi nhieu anh thanh
1 video 128 frame, caption dich = noi caption cua tung anh theo thu tu. Sau
warm-up chuyen sang video that (LLaVA-Video-178K, caption chi tiet).

Dieu kien train khop bo moc danh gia: 128 frame, max_pixels_units=16384,
time_mode=lmms, giu 25% token bang cong thuc linspace cua Uniform.

loss = w_ce * CE(caption | T2) + w_cov * coverage + w_dpp * dpp_hinge + w_kd * KL(dense || T2)
  CE       : Qwen dong bang ta lai video tu 25% token dau ra.
  coverage : moi token goc phai co 1 token dau ra o gan (rap_model.coverage_loss).
  dpp_hinge: dau ra khong duoc kem da dang hon token neo (R_DPP trong dac ta).
  KL       : (tuy chon) phan phoi tu tiep theo khi cho Qwen xem DU 100% token
             lam "thay", ep ban 25% cho ra cung phan phoi.

Usage (chay tu thu muc KTV):
  python train_rap.py --out outputs/rap/runA.pt --pseudo_steps 2000 --num_steps 5000
"""

import argparse
import json
import math
import os
import random
import time

os.environ.setdefault("DECORD_EOF_RETRY_MAX", "20480")

import torch
import torch.nn.functional as F
from PIL import Image
from qwen_vision_process_fastvid import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from rap_model import (AnchoredResampler, coverage_loss, diversity_stats, dpp_hinge_loss, load_resampler,
                       save_resampler, uniform_indices)
from run_inference_qwen25vl_2pass import build_pruned_sequence

MIN_PIXELS = 16 * 28 * 28
SYSTEM_PROMPT = "You are a helpful assistant."
PSEUDO_PROMPT = "This video shows several different scenes one after another. Describe each scene in order."
FRAME_H, FRAME_W = 336, 588  # luoi 12 x 21 token/unit, bang frame 16:9 cua VideoMME o cau hinh bo moc


def load_pseudo_rows(jsonl_path, image_dir):
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                rows.append({"image": os.path.join(image_dir, r["file_name"]), "caption": r["caption"].strip()})
    return rows


def load_real_rows(captions_files, video_root):
    existing = set()
    src_root = os.path.join(video_root, "academic_source")
    for dirpath, _dirs, files in os.walk(src_root):
        rel = os.path.relpath(dirpath, video_root)
        for fn in files:
            existing.add(os.path.join(rel, fn))
    rows = []
    for cf in captions_files:
        for item in json.load(open(cf, "r", encoding="utf-8")):
            if item["video"] in existing and "ego4d" not in item["video"]:  # ego4d: khong decode duoc tren may nay
                conv = item["conversations"]
                rows.append({"video": os.path.join(video_root, item["video"]), "caption": conv[1]["value"].strip(),
                             "prompt": conv[0]["value"].replace("<image>\n", "").strip()})
    return rows


def load_qa_rows(qa_files, video_root):
    """Cau hoi trac nghiem cua LLaVA-Video-178K, gom theo video: {video: [(cau hoi, dap an), ...]}."""
    by_video = {}
    for qf in qa_files:
        if not os.path.exists(qf):
            continue
        for item in json.load(open(qf, "r", encoding="utf-8")):
            if "ego4d" in item["video"]:
                continue
            conv = item["conversations"]
            turns = by_video.setdefault(os.path.join(video_root, item["video"]), [])
            for h, g in zip(conv[0::2], conv[1::2]):
                turns.append((h["value"].replace("<image>\n", "").strip(), g["value"].strip()))
    return by_video


def make_qa_sample(video, turns, max_pixels, max_frames, rng, max_turns):
    """1 video + toi da max_turns cau hoi (hoi lien tiep trong 1 hoi thoai). prompt la list -> che do QA."""
    video_ele = {"type": "video", "video": video, "max_pixels": max_pixels, "min_pixels": MIN_PIXELS,
                 "max_frames": max_frames}
    return video_ele, rng.sample(turns, min(max_turns, len(turns))), None


def make_pseudo_sample(rows, rng, max_frames):
    """128 frame = 64 unit, chia ngau nhien cho 4-8 anh, moi anh >= 2 unit."""
    units = max_frames // 2
    n_img = rng.randint(4, 8)
    cuts = sorted(rng.sample(range(1, units // 2), n_img - 1))
    lens = [2 * (b - a) for a, b in zip([0] + cuts, cuts + [units // 2])]  # so unit moi anh, tong = units
    picks = rng.sample(rows, n_img)
    frames, caps = [], []
    for i, (r, ln) in enumerate(zip(picks, lens)):
        img = Image.open(r["image"]).convert("RGB").resize((FRAME_W, FRAME_H), Image.BICUBIC)
        frames.extend([img] * (2 * ln))
        caps.append(f"Scene {i + 1}: {r['caption']}")
    video_ele = {"type": "video", "video": frames, "resized_height": FRAME_H, "resized_width": FRAME_W}
    return video_ele, PSEUDO_PROMPT, "\n".join(caps)


def make_real_sample(row, max_pixels, max_frames):
    video_ele = {"type": "video", "video": row["video"], "max_pixels": max_pixels, "min_pixels": MIN_PIXELS,
                 "max_frames": max_frames}
    return video_ele, row["prompt"], row["caption"]


@torch.no_grad()
def encode(model, processor, video_ele, prompt, device):
    """Chay vision encoder + dung san chuoi prompt, y het run_inference_qwen25vl_kitoke (time_mode=lmms)."""
    if isinstance(prompt, list):  # QA nhieu luot: dap an nam san trong chuoi (teacher forcing)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for i, (q, a) in enumerate(prompt):
            messages.append({"role": "user", "content": ([video_ele] if i == 0 else []) + [{"type": "text", "text": q}]})
            messages.append({"role": "assistant", "content": [{"type": "text", "text": a}]})
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    else:
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [video_ele, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    _, video_inputs, _ = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(text=[text], images=None, videos=video_inputs, padding=True, return_tensors="pt").to(device)
    inputs["second_per_grid_ts"] = torch.full_like(inputs["second_per_grid_ts"], 1.0)
    llm = model.model
    full_embeds = model.get_input_embeddings()(inputs["input_ids"])
    feat = llm.get_video_features(inputs["pixel_values_videos"], video_grid_thw=inputs["video_grid_thw"]).pooler_output
    if isinstance(feat, (list, tuple)):
        feat = torch.cat(feat, dim=0)
    feat = feat.to(full_embeds.dtype)
    vid = model.config.video_token_id
    vpos = (inputs["input_ids"][0] == vid).nonzero(as_tuple=True)[0]
    position_ids, _ = llm.get_rope_index(
        input_ids=inputs["input_ids"], image_grid_thw=None, video_grid_thw=inputs["video_grid_thw"],
        second_per_grid_ts=inputs.get("second_per_grid_ts"), attention_mask=inputs["attention_mask"],
        mm_token_type_ids=inputs.get("mm_token_type_ids"),
    )
    thw = inputs["video_grid_thw"][0]
    T, H, W = int(thw[0]), int(thw[1]) // 2, int(thw[2]) // 2
    assert feat.shape[0] == T * H * W, f"layout: {feat.shape[0]} != {T}*{H}*{W}"
    return {"inputs": inputs, "full_embeds": full_embeds, "feat": feat, "vid": vid, "vpos": vpos,
            "position_ids": position_ids, "T": T, "H": H, "W": W}


def caption_logits(model, ctx, kept_idx, kept_embeds, cap_ids, cap_embeds):
    """Logit cua cac vi tri du doan caption, cho truoc tap token video da giu.
    Chi ap lm_head len cac vi tri caption (khong tinh logit cho ca chuoi)."""
    fe, pid, am, _ = build_pruned_sequence(ctx["full_embeds"], ctx["inputs"], ctx["vid"], ctx["vpos"],
                                           ctx["position_ids"], kept_idx, kept_embeds)
    L = cap_ids.shape[1]
    P = fe.shape[1]
    fe = torch.cat([fe, cap_embeds], dim=1)
    cap_pos = pid[:, :, -1:] + torch.arange(1, L + 1, device=fe.device).view(1, 1, L)
    pid = torch.cat([pid, cap_pos], dim=-1)
    am = torch.cat([am, am.new_ones((1, L))], dim=1)
    hs = model.model(inputs_embeds=fe, position_ids=pid, attention_mask=am, return_dict=True).last_hidden_state
    return model.lm_head(hs[0, P - 1: P - 1 + L]).float()   # [L, vocab]


def answer_positions(ids, tok):
    """Vi tri cac token dap an (ke ca <|im_end|>) trong chuoi goc: sau moi '<|im_start|>assistant\\n'."""
    im_start, im_end = tok.convert_tokens_to_ids("<|im_start|>"), tok.convert_tokens_to_ids("<|im_end|>")
    asst = tok("assistant", add_special_tokens=False).input_ids[0]
    ids = ids.tolist()
    pos, i = [], 0
    while i < len(ids) - 2:
        if ids[i] == im_start and ids[i + 1] == asst:
            j = i + 3
            while j < len(ids):
                pos.append(j)
                if ids[j] == im_end:
                    break
                j += 1
            i = j
        i += 1
    return pos


def qa_logits(model, ctx, kept_idx, kept_embeds, from_end):
    """Logit du doan cac token dap an; from_end = khoang cach tu cuoi chuoi (khong doi khi cat token video)."""
    fe, pid, am, _ = build_pruned_sequence(ctx["full_embeds"], ctx["inputs"], ctx["vid"], ctx["vpos"],
                                           ctx["position_ids"], kept_idx, kept_embeds)
    hs = model.model(inputs_embeds=fe, position_ids=pid, attention_mask=am, return_dict=True).last_hidden_state
    return model.lm_head(hs[0, fe.shape[1] - from_end - 1]).float()   # [so token dap an, vocab]


def step_losses(model, processor, resampler, sample, args, device, train=True):
    video_ele, prompt, caption = sample
    ctx = encode(model, processor, video_ele, prompt, device)
    tok = processor.tokenizer
    is_qa = isinstance(prompt, list)
    if is_qa:
        ids = ctx["inputs"]["input_ids"][0]
        pos = torch.tensor(answer_positions(ids, tok), device=device)
        assert pos.numel() >= 2 * len(prompt), f"khong tim thay dap an trong chuoi ({pos.numel()} token)"
        cap_ids = ids[pos].unsqueeze(0)
        from_end = ids.shape[0] - pos

        def caption_logits(model, ctx, kept_idx, kept_embeds, _ids, _emb):
            return qa_logits(model, ctx, kept_idx, kept_embeds, from_end)
        cap_embeds = None
        w_ce = args.w_ce_qa
    else:
        caption_logits = globals()["caption_logits"]
        cap_ids = tok(caption, add_special_tokens=False, return_tensors="pt").input_ids[:, : args.max_caption_tokens]
        cap_ids = torch.cat([cap_ids, torch.tensor([[tok.eos_token_id]])], dim=1).to(device)
        with torch.no_grad():
            cap_embeds = model.get_input_embeddings()(cap_ids)
        w_ce = args.w_ce
    T, H, W = ctx["T"], ctx["H"], ctx["W"]
    N = H * W
    kept_idx = uniform_indices(T * N, args.retention, device)
    X = ctx["feat"].float().view(T, N, -1)
    out = {"T": T, "N": N, "n_kept": int(kept_idx.numel()), "cap_len": int(cap_ids.shape[1])}

    if not train:  # validation: CE cua RAP, cua Uniform (token neo) va cua dense
        with torch.no_grad():
            T2, layout = resampler(X, kept_idx, H, W)
            all_idx = torch.arange(T * N, device=device)
            lg_rap = caption_logits(model, ctx, kept_idx, T2, cap_ids, cap_embeds)
            lg_uni = caption_logits(model, ctx, kept_idx, layout["anchors"], cap_ids, cap_embeds)
            t_logp = F.log_softmax(caption_logits(model, ctx, all_idx, ctx["feat"], cap_ids, cap_embeds), dim=-1)
            out["ce_rap"] = float(F.cross_entropy(lg_rap, cap_ids[0]))
            out["ce_uniform"] = float(F.cross_entropy(lg_uni, cap_ids[0]))
            out["ce_dense"] = float(F.nll_loss(t_logp, cap_ids[0]))
            if is_qa:  # ti le token dap an doan dung (chu yeu la chu cai A/B/C/D)
                out["acc_rap"] = float((lg_rap.argmax(-1) == cap_ids[0]).float().mean())
                out["acc_uniform"] = float((lg_uni.argmax(-1) == cap_ids[0]).float().mean())
                out["acc_dense"] = float((t_logp.argmax(-1) == cap_ids[0]).float().mean())
            # KL(dense || .): do lech so voi ban khong nen, khong bi anh huong boi "kieu viet" caption
            out["kl_rap"] = float(F.kl_div(F.log_softmax(lg_rap, dim=-1), t_logp, log_target=True, reduction="batchmean"))
            out["kl_uniform"] = float(F.kl_div(F.log_softmax(lg_uni, dim=-1), t_logp, log_target=True, reduction="batchmean"))
            out.update(diversity_stats(T2, layout, T))
        return None, out

    T2, layout = resampler(X, kept_idx, H, W)
    logits = caption_logits(model, ctx, kept_idx, T2, cap_ids, cap_embeds)
    ce = F.cross_entropy(logits, cap_ids[0])
    loss = w_ce * ce
    out["ce"] = float(ce)
    if args.w_cov > 0:
        cov = coverage_loss(X, T2, layout)
        loss = loss + args.w_cov * cov
        out["cov"] = float(cov)
    if args.w_dpp > 0:
        dpp = dpp_hinge_loss(T2, layout, T)
        loss = loss + args.w_dpp * dpp
        out["dpp"] = float(dpp)
    if args.w_kd > 0:
        with torch.no_grad():
            all_idx = torch.arange(T * N, device=device)
            t_logp = F.log_softmax(caption_logits(model, ctx, all_idx, ctx["feat"], cap_ids, cap_embeds), dim=-1)
        kd = F.kl_div(F.log_softmax(logits, dim=-1), t_logp, log_target=True, reduction="batchmean")
        loss = loss + args.w_kd * kd
        out["kd"] = float(kd)
    with torch.no_grad():
        out["delta_ratio"] = float((T2 - layout["anchors"]).norm(dim=-1).mean() / layout["anchors"].norm(dim=-1).mean())
    out["loss"] = float(loss)
    return loss, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--resume", default="")          # tiep tuc dung run nay (lay ca optimizer, step)
    ap.add_argument("--init_from", default="")       # chi lay trong so, step bat dau tu 0
    ap.add_argument("--retention", type=float, default=0.25)
    ap.add_argument("--max_frames", type=int, default=128)
    ap.add_argument("--max_pixels_units", type=int, default=16384)
    ap.add_argument("--num_steps", type=int, default=5000)
    ap.add_argument("--pseudo_steps", type=int, default=2000)   # warm-up: chi pseudo-video
    ap.add_argument("--pseudo_prob_after", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--w_ce", type=float, default=1.0)    # 0 = chi distill tu ban khong nen (can w_kd > 0)
    ap.add_argument("--w_ce_qa", type=float, default=1.0) # trong so CE cho mau hoi-dap
    ap.add_argument("--qa_prob", type=float, default=0.0)  # sau warm-up: ti le buoc dung mau hoi-dap trac nghiem
    ap.add_argument("--qa_turns", type=int, default=6)
    ap.add_argument("--qa_files", default="0_30_s_academic_mc_v0_1_qa_processed.json,"
                                          "1_2_m_academic_mc_v0_1_qa_processed.json,2_3_m_academic_mc_v0_1_qa_processed.json")
    ap.add_argument("--w_cov", type=float, default=0.1)
    ap.add_argument("--w_dpp", type=float, default=0.05)
    ap.add_argument("--w_kd", type=float, default=0.0)
    ap.add_argument("--d_model", type=int, default=1024)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--window", type=int, default=1)
    ap.add_argument("--max_caption_tokens", type=int, default=384)
    ap.add_argument("--val_every", type=int, default=250)
    ap.add_argument("--n_val", type=int, default=8)   # so mau val MOI loai (pseudo, real)
    ap.add_argument("--ckpt_every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pseudo_jsonl", default="outputs/coco_train2017_recaptioned.jsonl")
    ap.add_argument("--pseudo_image_dir", default="Datasets/COCO/train2017")
    ap.add_argument("--video_root", default="Datasets/LLaVA-Video-178K-probe")
    ap.add_argument("--captions_files", default="0_30_s_academic_v0_1_cap_processed.json,"
                                                "1_2_m_academic_v0_1_cap_processed.json,2_3_m_academic_v0_1_cap_processed.json")
    args = ap.parse_args()

    device = "cuda"
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    max_pixels = args.max_pixels_units * 28 * 28

    print(f"[train_rap] args={vars(args)}", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto")
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    # Gradient checkpointing cua HF chi bat khi module o che do train. Qwen
    # khong co dropout/batchnorm nen train() o LLM khong doi ket qua forward.
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.model.language_model.train()
    processor = AutoProcessor.from_pretrained(args.model_path, max_pixels=max_pixels, min_pixels=MIN_PIXELS)

    pseudo_rows = load_pseudo_rows(args.pseudo_jsonl, args.pseudo_image_dir)
    real_rows = load_real_rows([os.path.join(args.video_root, c) for c in args.captions_files.split(",")], args.video_root)
    random.Random(1234).shuffle(real_rows)
    val_real, real_rows = real_rows[: args.n_val], real_rows[args.n_val:]
    val_pseudo_rows, pseudo_rows = pseudo_rows[:200], pseudo_rows[200:]
    vrng = random.Random(4321)
    val_samples = [("pseudo", make_pseudo_sample(val_pseudo_rows, vrng, args.max_frames)) for _ in range(args.n_val)]
    qa_videos, val_qa = [], []
    if args.qa_prob > 0:
        qa_by_video = load_qa_rows([os.path.join(args.video_root, c) for c in args.qa_files.split(",")], args.video_root)
        held = {r["video"] for r in val_real}
        cand = sorted(v for v in qa_by_video if v not in held and os.path.exists(v))
        random.Random(1234).shuffle(cand)
        val_qa_videos, qa_videos = cand[: args.n_val], cand[args.n_val:]
        real_rows = [r for r in real_rows if r["video"] not in set(val_qa_videos)]
        val_qa = [("qa", make_qa_sample(v, qa_by_video[v], max_pixels, args.max_frames, vrng, args.qa_turns)) for v in val_qa_videos]
        print(f"[train_rap] qa videos={len(qa_videos)} turns={sum(len(qa_by_video[v]) for v in qa_videos)} val_qa={len(val_qa)}", flush=True)
    print(f"[train_rap] pseudo images={len(pseudo_rows)} real videos={len(real_rows)} val={args.n_val}+{len(val_real)}", flush=True)

    dim = model.config.vision_config.out_hidden_size
    step = 0
    if args.resume and os.path.exists(args.resume):
        resampler, ckpt = load_resampler(args.resume, device)
        step = int(ckpt["step"])
    elif args.init_from:
        resampler, ckpt = load_resampler(args.init_from, device)
        ckpt = {}
    else:
        resampler = AnchoredResampler(dim, d_model=args.d_model, num_layers=args.num_layers, window=args.window).to(device).float()
        ckpt = {}
    opt = torch.optim.AdamW(resampler.parameters(), lr=args.lr, weight_decay=0.01)
    if "optimizer" in ckpt:
        opt.load_state_dict(ckpt["optimizer"])
    n_params = sum(p.numel() for p in resampler.parameters())
    print(f"[train_rap] resampler params={n_params/1e6:.1f}M start_step={step}", flush=True)
    for _ in range(step * 3):  # tua rng de resume khong lap lai du lieu da dung
        rng.random()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    mf = open(args.out + ".metrics.jsonl", "a", encoding="utf-8")

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        prog = (s - args.warmup) / max(1, args.num_steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))

    def validate(s):
        resampler.eval()
        agg = {}
        samples = val_samples + [("real", make_real_sample(r, max_pixels, args.max_frames)) for r in val_real] + val_qa
        for kind, smp in samples:
            try:
                _, o = step_losses(model, processor, resampler, smp, args, device, train=False)
            except Exception as e:
                print(f"[val] SKIP {kind}: {type(e).__name__}: {e}", flush=True)
                torch.cuda.empty_cache()
                continue
            for k in ("ce_rap", "ce_uniform", "ce_dense", "kl_rap", "kl_uniform", "acc_rap", "acc_uniform", "acc_dense",
                      "erank_out", "erank_anchor", "cos_out", "cos_anchor"):
                if k in o:
                    agg.setdefault(f"{kind}/{k}", []).append(o[k])
        res = {k: sum(v) / len(v) for k, v in agg.items()}
        for kind in ("pseudo", "real", "qa"):
            if f"{kind}/ce_rap" in res:
                gap = res[f"{kind}/ce_uniform"] - res[f"{kind}/ce_dense"]
                res[f"{kind}/gap_closed"] = (res[f"{kind}/ce_uniform"] - res[f"{kind}/ce_rap"]) / gap if abs(gap) > 1e-6 else 0.0
        mf.write(json.dumps({"step": s, "val": res}) + "\n")
        mf.flush()
        print(f"[val] step={s} " + " ".join(f"{k}={v:.4f}" for k, v in sorted(res.items())), flush=True)
        resampler.train()

    def save(s):
        tmp = args.out + ".tmp"
        save_resampler(tmp, resampler, s, {"optimizer": opt.state_dict(), "args": vars(args)})
        os.replace(tmp, args.out)

    validate(step)
    resampler.train()
    opt.zero_grad()
    t_last, skips = time.time(), 0
    while step < args.num_steps:
        use_pseudo = step < args.pseudo_steps or rng.random() < args.pseudo_prob_after or not real_rows
        try:
            if use_pseudo:
                sample, kind = make_pseudo_sample(pseudo_rows, rng, args.max_frames), "pseudo"
            elif qa_videos and rng.random() < args.qa_prob:
                v = rng.choice(qa_videos)
                sample, kind = make_qa_sample(v, qa_by_video[v], max_pixels, args.max_frames, rng, args.qa_turns), "qa"
            else:
                sample, kind = make_real_sample(rng.choice(real_rows), max_pixels, args.max_frames), "real"
            loss, o = step_losses(model, processor, resampler, sample, args, device, train=True)
            (loss / args.accum).backward()
        except Exception as e:
            skips += 1
            print(f"SKIP step {step}: {type(e).__name__}: {str(e)[:300]}", flush=True)
            opt.zero_grad()
            torch.cuda.empty_cache()
            if skips > 200 and skips > step:
                raise RuntimeError("qua nhieu buoc loi lien tiep, dung train")
            continue
        step += 1
        if step % args.accum == 0:
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            torch.nn.utils.clip_grad_norm_(resampler.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
        o.update({"step": step, "kind": kind, "sec": round(time.time() - t_last, 2),
                  "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)})
        t_last = time.time()
        mf.write(json.dumps(o) + "\n")
        mf.flush()
        if step % 10 == 0:
            print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in o.items()), flush=True)
        if step % 20 == 0:
            torch.cuda.empty_cache()
        if step % args.ckpt_every == 0:
            save(step)
        if step % args.val_every == 0:
            validate(step)
    save(step)
    validate(step)
    print(f"[train_rap] done step={step} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
