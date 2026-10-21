"""
train_perceiver_stage2.py (v3 -- chia doan theo noi dung, thay vi chon unit
dai dien co dinh)

Giai doan 2 (video) cua de xuat GOC leader -- Perceiver + Anchor + POINTING.
Ke thua checkpoint Giai doan 1 (anh, COCO) roi tinh chinh voi VIDEO that
(MSR-VTT).

LICH SU SUA (qua thao luan, xem tab RL-Gating/Perceiver tren web de biet chi
tiet):
  v1 (bug): goi Perceiver DUNG 1 LAN cho ca video, luon ra dung 64 token BAT
    KE video dai ngan -- khong tan dung cau truc thoi gian.
  v2 (bug khac, moi phat hien): chon M=30 unit DAI DIEN co dinh (rai deu),
    moi unit rieng le qua Perceiver. 2 van de: (a) bo qua cac unit khong
    duoc chon (video dai bi bo phan lon), (b) unit la don vi qua NHO va CO
    DINH (2 frame Qwen gop) -- luc train, chuoi noi lai luon NGAN (clip
    MSR-VTT ngan, T_llm<=30) nhung luc infer tren video dai co the phai noi
    RAT NHIEU unit (video 384 unit) -- Perceiver chua bao gio duoc "tap"
    o quy mo chuoi dai nhu vay, ket qua infer khong cai thien du tang do
    phu (da kiem chung: 30 unit va 384 unit cho ra diem VideoMME nhu nhau).
  v3 (ban nay): thay unit CO DINH bang DOAN (segment) DO DAI KHONG CO DINH,
    chia theo NOI DUNG bang dyseg_segments() (tai su dung tu
    run_inference_qwen25vl_dyseg.py, da kiem chung o DySeg+STPrune) --
    doan nao it doi (canh tinh) thi DAI, doan nao doi nhieu (chuyen canh)
    thi NGAN, PHU HET moi unit (khong bo unit nao nhu v2). Moi doan qua
    Perceiver MOT LAN, ra dung 64 token/doan -- Perceiver gio hoc nen dung
    LOAI dau vao (doan dai ngan tuy noi dung) ma no se gap luc infer, thay
    vi hoc tren don vi 2-frame co dinh roi bi ep infer o quy mo khac han.

    LUU Y quan trong: dyseg_segments() CHI nhin thay 384 unit DA DUOC QWEN
    LAY MAU SAN (FPS=2.0, tran 768 frame) -- KHONG phai toan bo video goc.
    Gioi han nay ap dung chung cho MOI phuong phap trong du an (RL-Gating,
    Perceiver ban cu), khong rieng gi ban nay.

LUONG DUNG (v3):
  1. Tinh dac trung tung unit (trung binh N token khong gian/unit) ->
     frame_feats [T_llm, D].
  2. dyseg_segments(frame_feats, c, tau) -> danh sach doan (start, end),
     PHU HET T_llm, do dai khac nhau tuy noi dung.
  3. VOI TUNG doan: gop token cua TAT CA unit trong doan do thanh 1 chuoi
     [(end-start)*N, D], qua Perceiver MOT LAN -> T2_seg = 64 token.
  4. Pointing (nguong cosine CO DINH tau, KHONG hoc): token goc nao trong
     doan co do giong (cosine) >= tau voi it nhat 1 token T2_seg thi GIU
     THEM.
  5. Ghep: moi doan dong gop (64 + so token Pointing giu them CUA DOAN DO)
     token -- noi tat ca doan lai thanh chuoi cuoi cung dua vao LLM. Tong
     so luong BIEN DOI theo video VA THEO DO PHUC TAP NOI DUNG (video it
     doi canh -> it doan -> it token, khong ti le thuan voi do dai video
     nhu truoc).
  6. Anchor Loss tinh THEO TUNG DOAN (neo T2_seg voi chinh token goc cua
     doan do), roi lay trung binh qua moi doan.

Usage:
  python train_perceiver_stage2.py \
    video_dir=Datasets/MSRVTT/video \
    captions_file=Datasets/MSRVTT/msrvtt_train_9k.json \
    model_path=Qwen/Qwen2.5-VL-7B-Instruct \
    resume_from=outputs/perceiver_stage1_selfattn.pt \
    dyseg_c=8 dyseg_tau=0.9 pointing_tau=0.5 \
    lr=2e-5 num_steps=200 log_every=10 ckpt_every=50 \
    policy_save_path=outputs/perceiver_stage2_v3_smoke.pt
"""

import json
import os
import random

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from qwen_vision_process_fastvid import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from run_inference_qwen25vl_dyseg import dyseg_segments
from train_perceiver_stage1 import MIN_PIXELS, SYSTEM_PROMPT, PerceiverResampler

CAPTION_PROMPT = "Describe this video in one short sentence."


def resolve_path(path):
    if path is None:
        return None
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    return to_absolute_path(path)


def load_msrvtt_captions(captions_file):
    """MSR-VTT: moi video co 1 danh sach ~20 caption. Trai phang thanh 1
    dong/caption de dung chung 1 kieu vong lap voi Giai doan 1."""
    d = json.load(open(captions_file, "r", encoding="utf-8"))
    rows = []
    for item in d:
        video_file = item.get("video") or (item["video_id"] + ".mp4")
        for cap in item["caption"]:
            rows.append({"file_name": video_file, "caption": cap.strip()})
    return rows


def run_one_video(model, processor, perceiver, optimizer, video_path, caption, args, max_pixels, device):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "video", "video": video_path, "max_pixels": max_pixels, "min_pixels": MIN_PIXELS},
            {"type": "text", "text": CAPTION_PROMPT},
        ]},
    ]
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, _ = process_vision_info(messages, return_video_kwargs=True)
    prompt_inputs = processor(
        text=[prompt_text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt",
    ).to(device)

    llm = model.model if hasattr(model, "model") else model
    video_token_id = model.config.video_token_id

    with torch.no_grad():
        video_feat = llm.get_video_features(
            prompt_inputs["pixel_values_videos"], video_grid_thw=prompt_inputs["video_grid_thw"]
        )
        if hasattr(video_feat, "pooler_output"):
            video_feat = video_feat.pooler_output
        T1_flat = video_feat if not isinstance(video_feat, (list, tuple)) else torch.cat(video_feat, dim=0)
        T1_flat = T1_flat.float()

    thw = prompt_inputs["video_grid_thw"][0].tolist()
    T_llm, H_llm, W_llm = int(thw[0]), int(thw[1]) // 2, int(thw[2]) // 2
    N = H_llm * W_llm
    assert T1_flat.shape[0] == T_llm * N, f"unit-major layout broken: {T1_flat.shape[0]} != {T_llm}*{N}"
    T1 = T1_flat.view(T_llm, N, -1)  # [T_llm, N, D]

    dyseg_c = int(getattr(args, "dyseg_c", 8))
    dyseg_tau = float(getattr(args, "dyseg_tau", 0.9))
    pointing_tau = float(getattr(args, "pointing_tau", 0.5))

    # Buoc 1+2: chia T_llm unit thanh cac DOAN do dai khac nhau theo noi
    # dung (dyseg_segments chi nhin thay 384 unit da duoc Qwen lay mau san,
    # xem canh bao dau file).
    with torch.no_grad():
        frame_feats = T1.mean(dim=1)  # [T_llm, D] -- dac trung dai dien tung unit
    segments = dyseg_segments(frame_feats, c=dyseg_c, tau=dyseg_tau)  # [(start,end), ...] phu het [0,T_llm)

    # ---------- Buoc 3+4: voi TUNG doan, Perceiver rieng + Pointing rieng ----------
    per_seg_tokens = []   # list cac tensor [64 + k_seg, D], se noi lai
    per_seg_anchor = []   # list scalar anchor_loss/doan, se lay trung binh
    total_pointed = 0
    for start, end in segments:
        T1_seg = T1[start:end].reshape(-1, T1.shape[-1])  # [(end-start)*N, D] -- gop het token cua doan
        T2_seg = perceiver(T1_seg)  # [64, D] -- Perceiver doc CA doan trong 1 lan

        T2n = torch.nn.functional.normalize(T2_seg, dim=-1)
        T1n = torch.nn.functional.normalize(T1_seg.detach(), dim=-1)
        sim = T2n @ T1n.t()  # [64, (end-start)*N]
        max_sim_per_q, _ = sim.max(dim=-1)  # [64] -- cho anchor loss (T2 gan T1 khong)
        per_seg_anchor.append(1.0 - max_sim_per_q.mean())

        max_sim_per_tok, _ = sim.max(dim=0)  # [(end-start)*N] -- cho Pointing
        point_mask = max_sim_per_tok >= pointing_tau
        T1_star = T1_seg[point_mask.detach()]  # [k_seg, D], k_seg bien doi
        total_pointed += T1_star.shape[0]

        per_seg_tokens.append(torch.cat([T2_seg, T1_star], dim=0))

    T2_all = torch.cat(per_seg_tokens, dim=0)  # [tong_token_cuoi, D]
    anchor_loss_mean = torch.stack(per_seg_anchor).mean()
    n_queries = T2_all.shape[0]  # BIEN DOI theo video VA theo do phuc tap noi dung

    # ---------- Buoc 5: ghep vao chuoi LLM (so luong placeholder BIEN DOI) ----------
    caption_ids = processor.tokenizer(caption, add_special_tokens=False).input_ids
    eos_id = processor.tokenizer.eos_token_id

    full_input_ids = prompt_inputs["input_ids"][0]
    video_mask_full = (full_input_ids == video_token_id)
    n_video_tok_real = int(video_mask_full.sum().item())
    pre_ids = full_input_ids[: video_mask_full.nonzero(as_tuple=True)[0][0]]
    post_ids = full_input_ids[video_mask_full.nonzero(as_tuple=True)[0][-1] + 1:]
    video_placeholder = torch.full((n_queries,), video_token_id, dtype=full_input_ids.dtype, device=device)

    prompt_ids = torch.cat([pre_ids, video_placeholder, post_ids])
    caption_ids_t = torch.tensor(caption_ids + [eos_id], dtype=full_input_ids.dtype, device=device)
    input_ids = torch.cat([prompt_ids, caption_ids_t]).unsqueeze(0)
    labels = torch.cat([
        torch.full((prompt_ids.shape[0],), -100, dtype=torch.long, device=device),
        caption_ids_t.clone(),
    ]).unsqueeze(0)

    with torch.no_grad():
        text_embeds = model.get_input_embeddings()(input_ids)

    n_video_tok_final = int((input_ids[0] == video_token_id).sum().item())
    assert n_video_tok_final == n_queries, (
        f"Lech so luong: {n_video_tok_final} vs {n_queries}. n_video_tok_real={n_video_tok_real}."
    )
    video_mask = (input_ids[0] == video_token_id).unsqueeze(0).unsqueeze(-1)
    final_embeds = text_embeds.masked_scatter(video_mask, T2_all.to(text_embeds.dtype))
    attention_mask = torch.ones_like(input_ids)

    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len, device=device).view(1, 1, seq_len).expand(3, 1, seq_len)

    out = model(
        inputs_embeds=final_embeds, position_ids=position_ids, attention_mask=attention_mask,
        labels=labels, return_dict=True,
    )
    caption_loss = out.loss

    anchor_weight = float(getattr(args, "anchor_weight", 0.5))
    loss = caption_loss + anchor_weight * anchor_loss_mean

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(perceiver.parameters(), max_norm=1.0)
    optimizer.step()

    seg_lens = [e - s for s, e in segments]
    return {
        "loss": float(loss.item()),
        "caption_loss": float(caption_loss.item()),
        "anchor_loss": float(anchor_loss_mean.item()),
        "n_video_tok_real": n_video_tok_real,
        "n_segments_used": len(segments),
        "seg_len_avg": sum(seg_lens) / len(seg_lens),
        "seg_len_max": max(seg_lens),
        "n_final_tokens": n_queries,
        "n_pointed_avg": total_pointed / max(1, len(segments)),
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
    # Backward van phai chay xuyen qua toan bo 28 layer cua LLM (dong bang
    # weight, nhung van can graph de lan gradient toi Perceiver) -- doan
    # cang dai (canh tinh, it doi) thi chuoi noi lai cang dai, activation
    # luu lai cho backward cang lon. Gradient checkpointing doi compute
    # lay memory, khong doi ket qua train.
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    processor = AutoProcessor.from_pretrained(model_path, max_pixels=max_pixels, min_pixels=MIN_PIXELS)

    video_dir = resolve_path(args.video_dir)
    captions_file = resolve_path(args.captions_file)
    rows = load_msrvtt_captions(captions_file)
    random.seed(0)
    random.shuffle(rows)
    print(f"Loaded {len(rows)} (video, caption) pairs from {captions_file}")

    resume_from = getattr(args, "resume_from", None)
    assert resume_from, "Giai doan 2 BAT BUOC phai +resume_from=<checkpoint Giai doan 1> -- khong train tu dau."
    ckpt = torch.load(resolve_path(resume_from), map_location=device)
    dim = ckpt["dim"]
    num_queries = ckpt["num_queries"]
    num_layers = ckpt["num_layers"]
    perceiver = PerceiverResampler(dim, num_queries=num_queries, num_layers=num_layers).to(device).float()
    perceiver.load_state_dict(ckpt["state_dict"])
    # Phan biet 2 truong hop: resume LAN DAU tu checkpoint Stage 1 (step cua
    # no la step Stage 1, KHONG lien quan step Stage 2) vs Stage 2 TU RESUME
    # CHINH NO (checkpoint da co field "dyseg_c" do save_ckpt() o file nay
    # tu luu -- danh dau rieng cho ban v3, KHAC voi field "num_units" cua
    # checkpoint ban v2 cu -- tranh nham resume step tu 1 checkpoint v2 cu).
    is_stage2_self_resume = "dyseg_c" in ckpt
    lr = float(getattr(args, "lr", 2e-5))
    optimizer = torch.optim.Adam(perceiver.parameters(), lr=lr)
    start_step = 0
    if is_stage2_self_resume:
        start_step = int(ckpt.get("step", 0))
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        else:
            print("CANH BAO: checkpoint khong co optimizer_state -- Adam momentum se reset.")
        print(f"[Stage 2 (v3) tu resume] tiep tuc tu step={start_step}")
    else:
        print(f"Ke thua Perceiver tu {resume_from} (step nguon={ckpt.get('step', '?')}, "
              f"co the la Stage 1 hoac Stage 2 ban v2 cu -- Stage 2 (v3) bat dau tu step=0)")
    print(f"dyseg_c={getattr(args, 'dyseg_c', 8)} dyseg_tau={getattr(args, 'dyseg_tau', 0.9)} "
          f"pointing_tau={getattr(args, 'pointing_tau', 0.5)}")

    num_steps = int(getattr(args, "num_steps", 200))
    log_every = int(getattr(args, "log_every", 10))
    ckpt_every = int(getattr(args, "ckpt_every", 50))
    policy_save_path = resolve_path(getattr(args, "policy_save_path", "outputs/perceiver_stage2_v3.pt"))
    os.makedirs(os.path.dirname(policy_save_path), exist_ok=True)
    metrics_log_path = policy_save_path + ".metrics.jsonl"
    metrics_f = open(metrics_log_path, "a", encoding="utf-8")

    def save_ckpt(path, step):
        torch.save({"state_dict": perceiver.state_dict(), "dim": dim, "num_queries": num_queries,
                    "num_layers": num_layers, "step": step, "optimizer_state": optimizer.state_dict(),
                    "dyseg_c": int(getattr(args, "dyseg_c", 8)),
                    "dyseg_tau": float(getattr(args, "dyseg_tau", 0.9)),
                    "pointing_tau": float(getattr(args, "pointing_tau", 0.5))}, path)

    step = start_step
    idx = start_step
    running_loss = []
    saved_step_ckpts = []
    # num_steps la muc tieu TUYET DOI, khong phai "chay them bay nhieu step".
    pbar = tqdm(total=num_steps, initial=start_step)
    while step < num_steps:
        row = rows[idx % len(rows)]
        idx += 1
        video_path = os.path.join(video_dir, row["file_name"])
        if not os.path.exists(video_path):
            continue
        try:
            stats = run_one_video(model, processor, perceiver, optimizer, video_path, row["caption"],
                                   args, max_pixels, device)
        except Exception as e:
            print(f"SKIP step {step} ({row['file_name']}): {type(e).__name__}: {e}")
            torch.cuda.empty_cache()
            continue

        running_loss.append(stats["loss"])
        step += 1
        pbar.update(1)
        if step % 20 == 0:
            torch.cuda.empty_cache()  # giam phan manh VRAM qua nhieu step dai
        metrics_f.write(json.dumps({"step": step, **stats}) + "\n")
        metrics_f.flush()
        if step % log_every == 0:
            print(
                f"step={step} loss={stats['loss']:.4f} caption_loss={stats['caption_loss']:.4f} "
                f"anchor_loss={stats['anchor_loss']:.4f} n_segments_used={stats['n_segments_used']} "
                f"seg_len_avg={stats['seg_len_avg']:.1f} seg_len_max={stats['seg_len_max']} "
                f"n_final_tokens={stats['n_final_tokens']} n_pointed_avg={stats['n_pointed_avg']:.1f} "
                f"(running loss={sum(running_loss)/len(running_loss):.4f})"
            )
        if ckpt_every > 0 and step % ckpt_every == 0:
            ckpt_path = policy_save_path + f".step{step}"
            save_ckpt(ckpt_path, step)
            save_ckpt(policy_save_path, step)
            saved_step_ckpts.append(ckpt_path)
            # Perceiver o dim=3584 khien moi checkpoint (state_dict +
            # optimizer Adam) nang ~500MB-1GB -- khong xoa file step cu se
            # lap day o dia sau vai nghin step. Chi giu lai keep_last_n ban
            # step gan nhat lam diem resume an toan.
            keep_last_n = int(getattr(args, "keep_last_n_ckpts", 3))
            while len(saved_step_ckpts) > keep_last_n:
                old_path = saved_step_ckpts.pop(0)
                if os.path.exists(old_path):
                    os.remove(old_path)
            print(f"[checkpoint] step={step} -> {ckpt_path} (and {policy_save_path})")
    pbar.close()
    metrics_f.close()

    save_ckpt(policy_save_path, step)
    print(f"[train_perceiver_stage2] done -> saved policy to {policy_save_path}")
    print(f"Per-step metrics logged to {metrics_log_path}")
    print(f"Final running loss={sum(running_loss)/len(running_loss):.4f}")


@hydra.main(config_path=None, version_base=None)
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
