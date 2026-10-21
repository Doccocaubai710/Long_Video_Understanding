"""
visualize_rl_gating.py

Quick visual sanity check of a trained RL-Gating policy (train_rl_gating.py
checkpoint) on ONE video: which time-units it decides to keep vs drop, and
how confident it is (xac suat giu) per unit. Same montage style as
visualize_pruning.py, but the unit is the unit of decision here (RL-Gating
prunes whole units, not individual spatial patches within a unit -- so no
patch-level before/after overlay like DySeg/STPrune has).

Query-agnostic by construction: builds a placeholder-question prompt just to
get Qwen's video encoding (policy never reads the question), so this never
calls generate() either.

Usage:
  python visualize_rl_gating.py \
    video_path=Datasets/NExTQA/NExTVideo/1164/3238737531.mp4 \
    policy_path=outputs/rl_gating_policy_exp3.pt \
    out_dir=outputs/viz_rl_gating_exp3

Optional: max_pixels_units=16384 (must match what the policy was TRAINED
with -- a different value changes T_llm/N and the policy's learned
threshold may not transfer), threshold=0.5, min_retention_frac=0.15 (set to
0 to see the RAW policy decision with no floor applied).
"""

import os

import hydra
import torch
from omegaconf import DictConfig
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

import run_inference_qwen25vl_dyseg as dm
from qwen_vision_process_fastvid import fetch_video, process_vision_info
from train_rl_gating import GatingPolicy


def frame_to_pil(frame_chw):
    arr = frame_chw.permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()
    return Image.fromarray(arr)


def prob_to_color(p):
    """Green (kept, high confidence) -> yellow (borderline) -> red (dropped,
    high confidence) -- p is the policy's keep-probability for that unit."""
    if p >= 0.5:
        t = (p - 0.5) / 0.5  # 0..1 as p goes 0.5->1.0
        return (int(255 * (1 - t)), 220, 40)
    else:
        t = (0.5 - p) / 0.5  # 0..1 as p goes 0.5->0.0
        return (220, int(255 * (1 - t)), 40)


def run(args):
    out_dir = dm.resolve_path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    max_pixels_units = int(getattr(args, "max_pixels_units", 16384))
    max_pixels = max_pixels_units * 28 * 28
    threshold = float(getattr(args, "threshold", 0.5))
    min_retention_frac = float(getattr(args, "min_retention_frac", 0.0))

    model_path = getattr(args, "model_path", "Qwen/Qwen2.5-VL-7B-Instruct")
    print(f"Loading {model_path} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=dtype, device_map="auto"
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path, max_pixels=max_pixels, min_pixels=dm.MIN_PIXELS)

    policy_path = dm.resolve_path(args.policy_path)
    ckpt = torch.load(policy_path, map_location=device)
    video_dim = ckpt["video_dim"]
    policy = GatingPolicy(video_dim).to(device).float()
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    print(f"Loaded policy from {policy_path} (video_dim={video_dim}, trained for {ckpt.get('step', '?')} steps)")

    video_path = dm.resolve_path(args.video_path)
    # Placeholder question -- policy never reads it (query-agnostic).
    messages = dm.build_messages(video_path, "placeholder", ["A. a", "B. b", "C. c", "D. d"], max_pixels)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, _ = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to(model.device)

    raw_video = fetch_video({
        "type": "video", "video": video_path,
        "max_pixels": max_pixels, "min_pixels": dm.MIN_PIXELS,
    })  # [n_frames_raw, C, H, W], float, 0-255 range

    video_grid_thw = inputs["video_grid_thw"][0]
    llm = model.model if hasattr(model, "model") else model
    with torch.inference_mode():
        video_feat = llm.get_video_features(inputs["pixel_values_videos"], video_grid_thw=inputs["video_grid_thw"])
        X_pooler = video_feat.pooler_output
        if isinstance(X_pooler, (list, tuple)):
            X_pooler = torch.cat(X_pooler, dim=0)

    T_llm = int(video_grid_thw[0])
    H_llm = int(video_grid_thw[1]) // 2
    W_llm = int(video_grid_thw[2]) // 2
    N = H_llm * W_llm
    print(f"video_grid_thw={video_grid_thw.tolist()} -> T_llm={T_llm} H_llm={H_llm} W_llm={W_llm} N={N} raw_frames={raw_video.shape[0]}")

    with torch.no_grad():
        X_units = X_pooler.view(T_llm, N, -1).mean(dim=1).float()
        delta_units = torch.zeros_like(X_units)
        delta_units[1:] = X_units[1:] - X_units[:-1]
        logits = policy(X_units, delta_units)
        probs = torch.sigmoid(logits).cpu()

    kept_mask = probs >= threshold
    n_before_floor = int(kept_mask.sum().item())
    if min_retention_frac > 0:
        min_units = max(1, int(round(min_retention_frac * T_llm)))
        if n_before_floor < min_units:
            floor_idx = torch.linspace(0, T_llm - 1, steps=min_units).round().long()
            kept_mask[floor_idx] = True
    n_kept = int(kept_mask.sum().item())
    retention = n_kept / T_llm
    print(f"threshold={threshold} min_retention_frac={min_retention_frac}: "
          f"{n_before_floor}/{T_llm} units kept by policy alone, {n_kept}/{T_llm} ({retention:.1%}) after floor")
    print(f"probs: min={probs.min():.3f} max={probs.max():.3f} mean={probs.mean():.3f} std={probs.std():.3f}")

    # Montage: one thumbnail per unit, colored by keep-probability (green =
    # confidently kept, red = confidently dropped, yellow = uncertain near
    # 0.5), annotated with the exact probability and whether it survived
    # (asterisk if only kept because of the floor, not the policy itself).
    thumb_w, thumb_h = 160, 120
    cols = 8
    rows = (T_llm + cols - 1) // cols
    montage = Image.new("RGB", (cols * thumb_w, rows * thumb_h), (15, 15, 15))
    for t in range(T_llm):
        frame_idx = min(2 * t, raw_video.shape[0] - 1)
        img = frame_to_pil(raw_video[frame_idx]).resize((thumb_w, thumb_h))
        p = float(probs[t])
        kept = bool(kept_mask[t])
        floor_only = kept and p < threshold
        if not kept:
            # darken dropped units so the eye goes straight to what's kept
            overlay = Image.new("RGBA", img.size, (0, 0, 0, 160))
            img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(img)
        border = prob_to_color(p) if not floor_only else (120, 120, 220)
        draw.rectangle([0, 0, thumb_w - 1, thumb_h - 1], outline=border, width=4)
        draw.rectangle([0, 0, 68, 16], fill=(0, 0, 0))
        tag = f"u{t}:p={p:.2f}" + ("*" if floor_only else "")
        draw.text((3, 2), tag, fill=(255, 255, 255))
        montage.paste(img, ((t % cols) * thumb_w, (t // cols) * thumb_h))
    montage_path = os.path.join(out_dir, "01_rlgating_units_montage.jpg")
    montage.save(montage_path, quality=90)
    print(f"saved {montage_path}")
    print("  xanh la = policy tu tin GIU, do = policy tu tin BO, vang = policy phan van (~0.5)")
    print("  toi (che den) = unit bi bo; vien mau tim + dau * = chi duoc giu nho san toi thieu, khong phai policy")

    print(f"\nDone -> {out_dir}")


@hydra.main(config_path=None, version_base=None)
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
