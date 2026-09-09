"""
visualize_pruning.py

Quick visual sanity check of the DySeg + STPrune pipeline on ONE video --
which units get grouped into which segment, how many tokens each unit
actually keeps, and (for a few representative units) exactly WHICH spatial
patches survive pruning vs get dropped. No plt.show() anywhere -- everything
is saved as .jpg so it can be checked by copying files off a headless A100
server (no display) to a machine that can view images.

Token selection here is query-agnostic (idea.md Phan 0 -- it never reads the
question), so this script does NOT run generate() or need a real question;
it only runs the vision encoder + DySeg + STPrune and visualizes what gets
kept vs dropped.

Usage:
  conda activate qwen25vl
  python visualize_pruning.py \
    video_path=Datasets/Video-MME/data/data/<ten_video>.mp4 \
    out_dir=outputs/viz_<ten_video>

Optional overrides: max_pixels_units=49152 (default 3x, same as dyseg.yaml),
tokens_num=16384, dtm_ratio=0.4, salience_mode=delta -- same knobs as
run_inference_qwen25vl_dyseg.py.
"""

import os

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

import run_inference_qwen25vl_dyseg as dm
from qwen_vision_process_fastvid import fetch_video, process_vision_info

SEG_COLORS = [
    (230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 190, 190), (240, 50, 230), (210, 200, 60),
]


def frame_to_pil(frame_chw):
    arr = frame_chw.permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()
    return Image.fromarray(arr)


def draw_patch_overlay(img, H_grid, W_grid, kept_flags):
    """kept_flags: list[bool] length H_grid*W_grid, row-major. Dropped
    patches get darkened; kept patches get a thin green outline."""
    img = img.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = img.size
    cell_w, cell_h = w / W_grid, h / H_grid
    for idx, kept in enumerate(kept_flags):
        r, c = idx // W_grid, idx % W_grid
        x0, y0, x1, y1 = c * cell_w, r * cell_h, (c + 1) * cell_w, (r + 1) * cell_h
        if kept:
            draw.rectangle([x0, y0, x1, y1], outline=(50, 220, 50, 255), width=2)
        else:
            draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0, 170))
    return Image.alpha_composite(img, overlay).convert("RGB")


def run(args):
    out_dir = dm.resolve_path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    max_pixels_units = int(getattr(args, "max_pixels_units", 49152))
    max_pixels = max_pixels_units * 28 * 28

    print(f"Loading {args.model_path} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=dtype, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(
        args.model_path, max_pixels=max_pixels, min_pixels=dm.MIN_PIXELS
    )

    video_path = dm.resolve_path(args.video_path)
    # Placeholder question -- selection never reads it (query-agnostic).
    messages = dm.build_messages(video_path, "placeholder", ["A. a", "B. b", "C. c", "D. d"], max_pixels)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, _ = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to(model.device)

    # Independent call to the same fetch_video() used above -- deterministic
    # given the same video_path/max_pixels/min_pixels, so it reproduces the
    # exact same resized frames the processor just fed the model, but as a
    # plain tensor we can turn into viewable images.
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
    X = X_pooler.reshape(T_llm, N, -1)
    print(f"video_grid_thw={video_grid_thw.tolist()} -> T_llm={T_llm} H_llm={H_llm} W_llm={W_llm} N={N} raw_frames={raw_video.shape[0]}")

    mode = getattr(args, "mode", "dyseg")
    all_kept_indices = set()
    segments = [(0, T_llm)]  # default for uniform mode: whole video = 1 "segment" for montage coloring
    if mode == "uniform":
        kept_indices, _, diag = dm.uniform_prune_video_tokens(X_pooler, int(args.tokens_num))
        all_kept_indices = set(kept_indices)
        print(f"UNIFORM mode: {diag}")
    else:
        segments = dm.dyseg_segments(X.mean(dim=1), c=args.dyseg_c, tau=args.dyseg_tau)
        v_s, rc_list, d_list, _ = dm.segment_scores(X, segments, gamma=args.recon_gamma, lam=args.lam)
        B_s_list, H_hat, T_gate = dm.allocate_budget(
            v_s, int(args.tokens_num), int(args.b_min), int(args.b_max),
            T0=args.temperature0, kappa=args.kappa, H0=args.h0,
        )
        print(f"{len(segments)} segments, H_hat={H_hat:.3f} T_gate={T_gate:.3f}")
        salience_mode = getattr(args, "salience_mode", "delta")
        vjepa_sal = None
        if salience_mode == "vjepa":
            cache_dir = dm.resolve_path(getattr(args, "vjepa_cache_dir", "outputs/vjepa_salience_cache"))
            cache_path = os.path.join(cache_dir, os.path.basename(video_path).replace("/", "_") + ".pt")
            vjepa_grid = torch.load(cache_path, map_location="cpu")
            vjepa_sal = dm.map_vjepa_to_qwen_grid(vjepa_grid, T_llm, H_llm, W_llm).reshape(T_llm, N).to(X.device)
        for seg, B_s, v in zip(segments, B_s_list, v_s):
            kept = dm.stprune_segment(
                X, seg, (H_llm, W_llm), B_s, d_ratio=args.dtm_ratio,
                p_anchor=args.p_anchor, alpha=args.dtm_alpha, salience_mode=salience_mode,
                vjepa_sal=vjepa_sal,
            )
            all_kept_indices.update(kept.keys())
            print(f"  seg units[{seg[0]:>3},{seg[1]:>3}) v_s={v:.3f} B_s={B_s:>5} kept={len(kept)}")

    per_unit_kept_count = [0] * T_llm
    for flat_idx in all_kept_indices:
        per_unit_kept_count[flat_idx // N] += 1
    print(f"TOTAL kept ({mode}): {len(all_kept_indices)} / {T_llm*N} ({100*len(all_kept_indices)/(T_llm*N):.1f}%)")

    # 1) Montage: every unit, patch-level pruning overlay baked in (dropped
    #    patches darkened, kept patches outlined green -- same as
    #    draw_patch_overlay used below), PLUS colored border by segment and
    #    "kept/N" annotated. Units with 0 kept tokens get a red border
    #    regardless of segment, so the whole video's post-prune state is
    #    visible at a glance in one image.
    thumb_w, thumb_h = 160, 120
    cols = 8
    rows = (T_llm + cols - 1) // cols
    montage = Image.new("RGB", (cols * thumb_w, rows * thumb_h), (15, 15, 15))
    seg_of_unit = {}
    for seg_i, (s, e) in enumerate(segments):
        for t in range(s, e):
            seg_of_unit[t] = seg_i
    for t in range(T_llm):
        frame_idx = min(2 * t, raw_video.shape[0] - 1)
        img = frame_to_pil(raw_video[frame_idx])
        kept_flags = [(t * N + i) in all_kept_indices for i in range(N)]
        img = draw_patch_overlay(img, H_llm, W_llm, kept_flags).resize((thumb_w, thumb_h))
        draw = ImageDraw.Draw(img)
        n_kept = per_unit_kept_count[t]
        col = (220, 30, 30) if n_kept == 0 else SEG_COLORS[seg_of_unit.get(t, 0) % len(SEG_COLORS)]
        draw.rectangle([0, 0, thumb_w - 1, thumb_h - 1], outline=col, width=4)
        draw.rectangle([0, 0, 60, 16], fill=(0, 0, 0))
        draw.text((3, 2), f"u{t}:{n_kept}/{N}", fill=(255, 255, 255))
        montage.paste(img, ((t % cols) * thumb_w, (t // cols) * thumb_h))
    montage_path = os.path.join(out_dir, f"01_units_montage_{mode}.jpg")
    montage.save(montage_path, quality=90)
    print(f"saved {montage_path}  (patches darkened = pruned, green outline = kept; border color = segment, red border = 0 token kept, 'uT:k/N' = kept count)")

    # 2) Patch-level before/after overlay. Which units to render:
    #    +units=0,212,349 (comma-separated) picks exact units, so the SAME
    #    unit can be rendered once per mode (dyseg vs uniform) and compared
    #    side by side. Default (no override): first unit of each segment.
    units_arg = getattr(args, "units", None)
    if units_arg is not None:
        target_units = [int(u) for u in str(units_arg).split(",")]
    else:
        target_units = [s for s, e in segments]
    for t in target_units:
        if not (0 <= t < T_llm):
            print(f"skip unit {t}: out of range [0,{T_llm})")
            continue
        frame_idx = min(2 * t, raw_video.shape[0] - 1)
        img = frame_to_pil(raw_video[frame_idx])
        kept_flags = [(t * N + i) in all_kept_indices for i in range(N)]
        before_after = Image.new("RGB", (img.width * 2 + 10, img.height), (15, 15, 15))
        before_after.paste(img, (0, 0))
        before_after.paste(draw_patch_overlay(img, H_llm, W_llm, kept_flags), (img.width + 10, 0))
        draw = ImageDraw.Draw(before_after)
        draw.text((4, 4), f"{mode} unit{t}: {sum(kept_flags)}/{N} kept", fill=(255, 255, 0))
        path = os.path.join(out_dir, f"02_unit{t}_{mode}_before_after.jpg")
        before_after.save(path, quality=90)
    print(f"saved 02_unit*_{mode}_before_after.jpg for {len(target_units)} units (trai=truoc/dense, phai=sau/pruned -- xanh=giu, den=bo)")

    print(f"\nDone -> {out_dir}")


@hydra.main(config_path="configs/inference_qwen25vl", config_name="dyseg", version_base=None)
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
