"""
train_place.py -- hoc vi tri neo cho RAP bang GRPO (place_model.AnchorPlacer).

RAP (checkpoint --rap, vd R10E) DONG BANG; chi AnchorPlacer duoc train.
Moi buoc, voi 1 video + caption chi tiet (LLaVA-Video-178K, giong train_rap.py):
  1. thay = Qwen xem 100% token, log-prob tu tiep theo tren caption (teacher forcing)
  2. lay G bo vi tri neo bang Gumbel top-k (moi unit giu dung so neo cua Uniform)
  3. moi bo: RAP -> Qwen -> reward = -KL(thay || ban nen) tren caption
  Uu tien luoi Uniform giam dan (--prior -> --prior_end) de greedy that su roi khoi luoi
  khi diem hoc duoc du lon; voi prior co dinh 4.0, greedy khong doi neo nao sau 500 buoc.
  4. advantage chuan hoa trong nhom (GRPO, 1 lan cap nhat moi rollout nen khong can clip)
     loss = -mean_g(A_g * logp_g) / so luot chon + w_reg * mean(s^2)
Khong backprop qua LLM (reward khong can gradient), nen re hon train_rap.py.

Usage (tu thu muc KTV):
  python train_place.py --rap outputs/rap/R10E.pt --retention 0.10 --out outputs/rap_place/P10.pt
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
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from place_model import AnchorPlacer, load_placer, place, save_placer, uniform_layout
from rap_model import load_resampler
from train_rap import MIN_PIXELS, caption_logits, encode, load_real_rows, make_real_sample


def kl_to_teacher(logits, t_logp):
    return F.kl_div(F.log_softmax(logits, dim=-1), t_logp, log_target=True, reduction="batchmean")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rap", required=True)
    ap.add_argument("--retention", type=float, required=True)
    ap.add_argument("--model_path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--resume", default="")
    ap.add_argument("--max_frames", type=int, default=128)
    ap.add_argument("--max_pixels_units", type=int, default=16384)
    ap.add_argument("--num_steps", type=int, default=3000)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--prior", type=float, default=4.0)       # uu tien o luoi Uniform luc dau (greedy == Uniform)
    ap.add_argument("--prior_end", type=float, default=1.0)   # giam tuyen tinh ve muc nay ...
    ap.add_argument("--prior_anneal", type=float, default=0.6)  # ... trong 60% so buoc dau; checkpoint luu prior hien tai
    ap.add_argument("--w_reg", type=float, default=1e-3)
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--max_caption_tokens", type=int, default=384)
    ap.add_argument("--val_every", type=int, default=250)
    ap.add_argument("--n_val", type=int, default=8)
    ap.add_argument("--ckpt_every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video_root", default="Datasets/LLaVA-Video-178K-probe")
    ap.add_argument("--captions_files", default="0_30_s_academic_v0_1_cap_processed.json,"
                                                "1_2_m_academic_v0_1_cap_processed.json,2_3_m_academic_v0_1_cap_processed.json")
    args = ap.parse_args()

    device = "cuda"
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    max_pixels = args.max_pixels_units * 28 * 28
    print(f"[train_place] args={vars(args)}", flush=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto")
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    model.config.use_cache = False
    processor = AutoProcessor.from_pretrained(args.model_path, max_pixels=max_pixels, min_pixels=MIN_PIXELS)
    tok = processor.tokenizer
    rap, rap_ckpt = load_resampler(args.rap, device)
    rap.eval()
    for p in rap.parameters():
        p.requires_grad_(False)
    print(f"[train_place] rap={args.rap} retention_trained={rap_ckpt.get('args', {}).get('retention')}", flush=True)

    # cung cach chia val voi train_rap.py (cung seed 1234) -> val la video RAP chua thay khi train
    real_rows = load_real_rows([os.path.join(args.video_root, c) for c in args.captions_files.split(",")], args.video_root)
    random.Random(1234).shuffle(real_rows)
    val_rows, real_rows = real_rows[: args.n_val], real_rows[args.n_val:]
    print(f"[train_place] real videos={len(real_rows)} val={len(val_rows)}", flush=True)

    dim = model.config.vision_config.out_hidden_size
    step, ckpt = 0, {}
    if args.resume and os.path.exists(args.resume):
        placer, ckpt = load_placer(args.resume, device)
        step = int(ckpt["step"])
    else:
        placer = AnchorPlacer(dim, d_model=args.d_model, prior=args.prior).to(device).float()
    opt = torch.optim.AdamW(placer.parameters(), lr=args.lr, weight_decay=0.01)
    if "optimizer" in ckpt:
        opt.load_state_dict(ckpt["optimizer"])
    for _ in range(step):  # tua rng de resume khong lap lai du lieu
        rng.random()
    print(f"[train_place] placer params={sum(p.numel() for p in placer.parameters()) / 1e6:.1f}M start_step={step}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    mf = open(args.out + ".metrics.jsonl", "a", encoding="utf-8")

    def prior_at(s):
        f = min(1.0, s / max(1, args.prior_anneal * args.num_steps))
        return args.prior + (args.prior_end - args.prior) * f

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        prog = (s - args.warmup) / max(1, args.num_steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))

    def prepare(row):
        video_ele, prompt, caption = make_real_sample(row, max_pixels, args.max_frames)
        ctx = encode(model, processor, video_ele, prompt, device)
        cap_ids = tok(caption, add_special_tokens=False, return_tensors="pt").input_ids[:, : args.max_caption_tokens]
        cap_ids = torch.cat([cap_ids, torch.tensor([[tok.eos_token_id]])], dim=1).to(device)
        with torch.no_grad():
            cap_emb = model.get_input_embeddings()(cap_ids)
            T, H, W = ctx["T"], ctx["H"], ctx["W"]
            all_idx = torch.arange(T * H * W, device=device)
            t_logp = F.log_softmax(caption_logits(model, ctx, all_idx, ctx["feat"], cap_ids, cap_emb), dim=-1)
        return ctx, cap_ids, cap_emb, t_logp

    @torch.no_grad()
    def reward(ctx, X, idx, cap_ids, cap_emb, t_logp):
        T2, _ = rap(X, idx, ctx["H"], ctx["W"])
        return -float(kl_to_teacher(caption_logits(model, ctx, idx, T2.to(ctx["feat"].dtype), cap_ids, cap_emb), t_logp))

    def validate(s):
        placer.eval()
        r_uni, r_greedy, moved = [], [], []
        for row in val_rows:
            try:
                ctx, cap_ids, cap_emb, t_logp = prepare(row)
                T, H, W = ctx["T"], ctx["H"], ctx["W"]
                X = ctx["feat"].float().view(T, H * W, -1)
                uidx, umask, counts = uniform_layout(T, H * W, args.retention, device)
                with torch.no_grad():
                    gidx, _, _ = place(placer(X, W) + placer.prior * umask, counts, sample=False)
                r_uni.append(reward(ctx, X, uidx, cap_ids, cap_emb, t_logp))
                r_greedy.append(reward(ctx, X, gidx, cap_ids, cap_emb, t_logp))
                moved.append(1.0 - float(umask.view(-1)[gidx].float().mean()))
            except Exception as e:
                print(f"[val] SKIP: {type(e).__name__}: {e}", flush=True)
                torch.cuda.empty_cache()
        res = {"prior": placer.prior, "kl_uniform_anchor": -sum(r_uni) / max(1, len(r_uni)), "kl_greedy": -sum(r_greedy) / max(1, len(r_greedy)),
               "moved_frac": sum(moved) / max(1, len(moved)), "n": len(r_uni)}
        mf.write(json.dumps({"step": s, "val": res}) + "\n")
        mf.flush()
        print(f"[val] step={s} " + " ".join(f"{k}={v:.4f}" for k, v in res.items()), flush=True)
        placer.train()

    def save(s):
        tmp = args.out + ".tmp"
        save_placer(tmp, placer, s, {"optimizer": opt.state_dict(), "args": vars(args)})
        os.replace(tmp, args.out)

    if step == 0:
        validate(0)
    placer.train()
    t_last, skips = time.time(), 0
    while step < args.num_steps:
        placer.prior = prior_at(step)
        try:
            ctx, cap_ids, cap_emb, t_logp = prepare(rng.choice(real_rows))
            T, H, W = ctx["T"], ctx["H"], ctx["W"]
            X = ctx["feat"].float().view(T, H * W, -1)
            uidx, umask, counts = uniform_layout(T, H * W, args.retention, device)
            s = placer(X, W)
            logits = s + placer.prior * umask
            rewards, logps, moved = [], [], []
            for _ in range(args.group):
                idx, logp, n_pick = place(logits, counts, sample=True, gen=gen)
                rewards.append(reward(ctx, X, idx, cap_ids, cap_emb, t_logp))
                logps.append(logp)
                moved.append(1.0 - float(umask.view(-1)[idx].float().mean()))
            r = torch.tensor(rewards, device=device)
            adv = (r - r.mean()) / (r.std() + 1e-6)
            loss = -(adv * torch.stack(logps)).mean() / n_pick + args.w_reg * s.pow(2).mean()
            opt.zero_grad()
            loss.backward()
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            torch.nn.utils.clip_grad_norm_(placer.parameters(), 1.0)
            opt.step()
            r_uni = reward(ctx, X, uidx, cap_ids, cap_emb, t_logp) if (step + 1) % 10 == 0 else None
        except Exception as e:
            skips += 1
            print(f"SKIP step {step}: {type(e).__name__}: {str(e)[:300]}", flush=True)
            torch.cuda.empty_cache()
            if skips > 200 and skips > step:
                raise RuntimeError("qua nhieu buoc loi, dung train")
            continue
        step += 1
        o = {"step": step, "prior": round(placer.prior, 3), "loss": float(loss), "r_mean": float(r.mean()), "r_std": float(r.std()),
             "r_max": float(r.max()), "moved": sum(moved) / len(moved), "s_abs": float(s.detach().abs().mean()),
             "sec": round(time.time() - t_last, 2), "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
        if r_uni is not None:
            o["r_uniform"] = r_uni
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
    print(f"[train_place] done step={step} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
