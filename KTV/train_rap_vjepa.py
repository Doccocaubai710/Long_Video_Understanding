"""
train_rap_vjepa.py -- Giai doan 1 (tu giam sat) cho RAP: pretrain bang V-JEPA, khong can caption, khong chay LLM.

Y tuong (Giai doan 1 cua VJEPA_REWARD_proposal_v2, nhung dung lam LOSS cho RAP
thay vi lam reward cho bo chon token -- outputs/vjepa_reward_correlation_charades
cho thay reward V-JEPA gan nhu khong tuong quan voi do huu ich ngu nghia khi
dung de CHON token, Spearman 0.069):

  Token nen cua RAP (neo Uniform + delta) phai DU THONG TIN de mot predictor nhe
  dung lai bieu dien V-JEPA cua TOAN BO video, ke ca vung bi bo.

    X (Qwen vision, 100% token) --RAP--> T2 (vd 10% token)
    V-JEPA 2 ViT-L (dong bang) tren cung video --> Z [32 x 16 x 16, 1024]  (target, LN khong affine)
    Predictor: query = mask token + toa do (t, y, x) cua M vi tri V-JEPA ngau nhien,
               cross-attn sang T2 (+ toa do cua tung neo)  --> Z_hat
    loss = L1(Z_hat, sg(Z)) + w_delta * relu(delta_ratio - delta_margin)^2

  Toa do lien tuc trong [0,1]^3 (Fourier features dung chung) nen luoi Qwen
  (64 x H x W) va luoi V-JEPA (32 x 16 x 16) khop nhau theo vi tri that.
  Frame cho V-JEPA duoc resize thang 256x256 (khong center-crop) de phu het khung.

Du lieu: video KHONG NHAN = moi video trong LLaVA-Video-178K-probe (ke ca video
khong co caption; tru ego4d va 8 video val cua train_rap.py) + MSR-VTT (10k).
Thu muc Datasets/NExTQA (benchmark) KHONG dung.

Checkpoint luu bang save_resampler nen Giai doan 2 dung thang:
  python train_rap.py --init_from <ckpt nay> --retention 0.10 ... (cong thuc Run E)

Usage (tu thu muc KTV):
  python train_rap_vjepa.py --out outputs/rap_vjepa/V10_s1.pt --retention 0.10 --num_steps 3000
"""

import argparse
import glob
import json
import math
import os
import random
import time

os.environ.setdefault("DECORD_EOF_RETRY_MAX", "20480")

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor, AutoVideoProcessor, Qwen2_5_VLForConditionalGeneration

from precompute_vjepa_salience import load_video_frames
from local_data import local_mirror
from rap_model import AnchoredResampler, load_resampler, save_resampler, uniform_indices
from train_rap import MIN_PIXELS, encode, load_real_rows

GENERIC_PROMPT = "Describe this video."


def fourier(coords, n_freq=16):
    """coords [..., 3] trong [0,1] -> [..., 6 * n_freq]."""
    freqs = (2.0 ** torch.arange(n_freq, device=coords.device, dtype=coords.dtype)) * math.pi
    ang = coords.unsqueeze(-1) * freqs                      # [..., 3, n_freq]
    return torch.cat([ang.sin(), ang.cos()], dim=-1).flatten(-2)


class VJEPAPredictor(nn.Module):
    """Doc tu token nen T2, du doan dac trung V-JEPA tai cac toa do bat ky."""

    def __init__(self, dim_in, dim_out, d_model=512, num_layers=3, num_heads=8, n_freq=16):
        super().__init__()
        self.n_freq = n_freq
        self.ctx_proj = nn.Sequential(nn.LayerNorm(dim_in), nn.Linear(dim_in, d_model))
        self.pos = nn.Sequential(nn.Linear(6 * n_freq, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.mask_token, std=0.02)
        self.ln_q = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.ln_kv = nn.LayerNorm(d_model)
        self.cross = nn.ModuleList([nn.MultiheadAttention(d_model, num_heads, batch_first=True) for _ in range(num_layers)])
        self.ln_f = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.ffn = nn.ModuleList([nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
                                  for _ in range(num_layers)])
        self.out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, dim_out))

    def forward(self, ctx_tok, ctx_xyz, q_xyz):
        """ctx_tok [K, dim_in], ctx_xyz [K, 3], q_xyz [M, 3] -> [M, dim_out]."""
        kv = self.ln_kv(self.ctx_proj(ctx_tok) + self.pos(fourier(ctx_xyz, self.n_freq))).unsqueeze(0)
        q = (self.mask_token + self.pos(fourier(q_xyz, self.n_freq))).unsqueeze(0)
        for ln_q, ca, ln_f, ffn in zip(self.ln_q, self.cross, self.ln_f, self.ffn):
            q = q + ca(ln_q(q), kv, kv, need_weights=False)[0]
            q = q + ffn(ln_f(q))
        return self.out(q[0])


def qwen_xyz(kept_idx, T, H, W):
    """Toa do [0,1]^3 (t, y, x) tam cua moi token Qwen giu lai."""
    N = H * W
    t, n = kept_idx // N, kept_idx % N
    return torch.stack([(t + 0.5) / T, (n // W + 0.5) / H, (n % W + 0.5) / W], dim=-1).float()


def vjepa_xyz(D, G, device):
    t, y, x = torch.meshgrid(torch.arange(D, device=device), torch.arange(G, device=device),
                             torch.arange(G, device=device), indexing="ij")
    return torch.stack([(t + 0.5) / D, (y + 0.5) / G, (x + 0.5) / G], dim=-1).reshape(-1, 3).float()


def list_videos(args, held_out):
    rows = load_real_rows([os.path.join(args.video_root, c) for c in args.captions_files.split(",")], args.video_root)
    vids = {r["video"] for r in rows}
    for d in ("activitynet", "Charades", "youcook2", "NextQA"):  # ca video co trong probe nhung khong co caption
        for ext in ("mp4", "mkv", "webm"):
            vids.update(glob.glob(os.path.join(args.video_root, "academic_source", d, "**", f"*.{ext}"), recursive=True))
    if args.msrvtt_dir:
        vids.update(glob.glob(os.path.join(args.msrvtt_dir, "**", "*.mp4"), recursive=True))
    return sorted(v for v in vids if v not in held_out and "ego4d" not in v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", default="")
    ap.add_argument("--retention", type=float, required=True)
    ap.add_argument("--model_path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--vjepa_path", default="facebook/vjepa2-vitl-fpc64-256")
    ap.add_argument("--max_frames", type=int, default=128)
    ap.add_argument("--max_pixels_units", type=int, default=16384)
    ap.add_argument("--num_steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--n_targets", type=int, default=2048)  # so vi tri V-JEPA du doan moi buoc (tong 8192)
    ap.add_argument("--w_delta", type=float, default=1.0)
    ap.add_argument("--delta_margin", type=float, default=0.2)  # R10E co delta_ratio ~0.10-0.16
    ap.add_argument("--d_model", type=int, default=1024)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--window", type=int, default=1)
    ap.add_argument("--pred_d_model", type=int, default=512)
    ap.add_argument("--pred_layers", type=int, default=3)
    ap.add_argument("--val_every", type=int, default=250)
    ap.add_argument("--n_val", type=int, default=16)
    ap.add_argument("--ckpt_every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video_root", default="Datasets/LLaVA-Video-178K-probe")
    ap.add_argument("--msrvtt_dir", default="Datasets/MSRVTT/video")
    ap.add_argument("--captions_files", default="0_30_s_academic_v0_1_cap_processed.json,"
                                                "1_2_m_academic_v0_1_cap_processed.json,2_3_m_academic_v0_1_cap_processed.json")
    args = ap.parse_args()
    args.video_root = local_mirror(args.video_root, "academic_source")
    args.msrvtt_dir = local_mirror(args.msrvtt_dir)
    args.model_path = local_mirror(args.model_path)

    device = "cuda"
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    max_pixels = args.max_pixels_units * 28 * 28
    print(f"[train_rap_vjepa] args={vars(args)}", flush=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto")
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path, max_pixels=max_pixels, min_pixels=MIN_PIXELS)
    vproc = AutoVideoProcessor.from_pretrained(args.vjepa_path)
    vjepa = AutoModel.from_pretrained(args.vjepa_path, torch_dtype=torch.bfloat16).to(device).eval()
    for p in vjepa.parameters():
        p.requires_grad_(False)
    G = vjepa.config.crop_size // vjepa.config.patch_size
    Dv = vjepa.config.frames_per_clip // vjepa.config.tubelet_size
    z_xyz = vjepa_xyz(Dv, G, device)

    # 8 video val cua train_rap.py (cung seed 1234) khong duoc dung o day
    cap_rows = load_real_rows([os.path.join(args.video_root, c) for c in args.captions_files.split(",")], args.video_root)
    random.Random(1234).shuffle(cap_rows)
    held = {r["video"] for r in cap_rows[:8]}
    videos = list_videos(args, held)
    random.Random(1234).shuffle(videos)
    val_videos, videos = videos[: args.n_val], videos[args.n_val:]
    print(f"[train_rap_vjepa] videos={len(videos)} val={len(val_videos)} vjepa grid={Dv}x{G}x{G}", flush=True)

    dim = model.config.vision_config.out_hidden_size
    step, ckpt = 0, {}
    if args.resume and os.path.exists(args.resume):
        rap, ckpt = load_resampler(args.resume, device)
        step = int(ckpt["step"])
    else:
        rap = AnchoredResampler(dim, d_model=args.d_model, num_layers=args.num_layers, window=args.window).to(device).float()
    pred = VJEPAPredictor(dim, vjepa.config.hidden_size, d_model=args.pred_d_model, num_layers=args.pred_layers).to(device).float()
    if "predictor" in ckpt:
        pred.load_state_dict(ckpt["predictor"])
    params = list(rap.parameters()) + list(pred.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    if "optimizer" in ckpt:
        opt.load_state_dict(ckpt["optimizer"])
    for _ in range(step):
        rng.random()
    print(f"[train_rap_vjepa] rap params={sum(p.numel() for p in rap.parameters()) / 1e6:.1f}M "
          f"predictor params={sum(p.numel() for p in pred.parameters()) / 1e6:.1f}M start_step={step}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    mf = open(args.out + ".metrics.jsonl", "a", encoding="utf-8")

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        prog = (s - args.warmup) / max(1, args.num_steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))

    @torch.no_grad()
    def targets(video):
        frames = load_video_frames(video, num_frames=vjepa.config.frames_per_clip)
        frames = np.stack([cv2.resize(f, (vjepa.config.crop_size, vjepa.config.crop_size), interpolation=cv2.INTER_AREA) for f in frames])
        px = vproc(frames, return_tensors="pt", do_resize=False, do_center_crop=False)["pixel_values_videos"]
        z = vjepa.get_vision_features(px.to(device, torch.bfloat16))[0].float()   # [Dv*G*G, 1024]
        return F.layer_norm(z, z.shape[-1:])

    def losses(video, n_targets, gen=None):
        video_ele = {"type": "video", "video": video, "max_pixels": max_pixels, "min_pixels": MIN_PIXELS,
                     "max_frames": args.max_frames}
        ctx = encode(model, processor, video_ele, GENERIC_PROMPT, device)
        T, H, W = ctx["T"], ctx["H"], ctx["W"]
        X = ctx["feat"].float().view(T, H * W, -1)
        kept_idx = uniform_indices(T * H * W, args.retention, device)
        z = targets(video)
        T2, layout = rap(X, kept_idx, H, W)
        anchors = layout["anchors"]
        c_xyz = qwen_xyz(kept_idx, T, H, W)
        sel = torch.randperm(z.shape[0], device=device, generator=gen)[:n_targets]
        rec = F.l1_loss(pred(T2, c_xyz, z_xyz[sel]), z[sel])
        ratio = (T2 - anchors).norm(dim=-1).mean() / anchors.norm(dim=-1).mean()
        loss = rec + args.w_delta * F.relu(ratio - args.delta_margin) ** 2
        return loss, {"rec": float(rec), "delta_ratio": float(ratio), "T": T, "n_kept": int(kept_idx.numel())}, (T2, anchors, c_xyz, z)

    def validate(s):
        rap.eval()
        pred.eval()
        rec, rec_anchor, ratio = [], [], []
        gen = torch.Generator(device=device)
        gen.manual_seed(0)
        for v in val_videos:
            try:
                with torch.no_grad():
                    _, o, (T2, anchors, c_xyz, z) = losses(v, z_xyz.shape[0], gen)
                    # cung predictor nhung chi cho token neo (delta = 0): delta mang them bao nhieu thong tin
                    rec_anchor.append(float(F.l1_loss(pred(anchors, c_xyz, z_xyz), z)))
                rec.append(o["rec"])
                ratio.append(o["delta_ratio"])
            except Exception as e:
                print(f"[val] SKIP {v}: {type(e).__name__}: {e}", flush=True)
                torch.cuda.empty_cache()
        n = max(1, len(rec))
        res = {"rec": sum(rec) / n, "rec_anchor_only": sum(rec_anchor) / n, "delta_ratio": sum(ratio) / n, "n": len(rec)}
        mf.write(json.dumps({"step": s, "val": res}) + "\n")
        mf.flush()
        print(f"[val] step={s} " + " ".join(f"{k}={v:.4f}" for k, v in res.items()), flush=True)
        rap.train()
        pred.train()

    def save(s):
        tmp = args.out + ".tmp"
        save_resampler(tmp, rap, s, {"predictor": pred.state_dict(), "optimizer": opt.state_dict(), "args": vars(args)})
        os.replace(tmp, args.out)

    if step == 0:
        validate(0)
    rap.train()
    pred.train()
    opt.zero_grad()
    t_last, skips = time.time(), 0
    while step < args.num_steps:
        try:
            loss, o, _ = losses(rng.choice(videos), args.n_targets)
            (loss / args.accum).backward()
        except Exception as e:
            skips += 1
            print(f"SKIP step {step}: {type(e).__name__}: {str(e)[:300]}", flush=True)
            opt.zero_grad()
            torch.cuda.empty_cache()
            if skips > 200 and skips > step:
                raise RuntimeError("qua nhieu buoc loi, dung train")
            continue
        step += 1
        if step % args.accum == 0:
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad()
        o.update({"step": step, "loss": float(loss), "sec": round(time.time() - t_last, 2),
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
    if step % args.ckpt_every != 0:
        save(step)
    if step % args.val_every != 0:
        validate(step)
    print(f"[train_rap_vjepa] done step={step} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
