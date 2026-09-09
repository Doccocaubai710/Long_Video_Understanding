"""
analyze_spatial_bias.py

Turns the single-video anecdote ("token bi don o goc") into an aggregate
measurement across many videos: for each kept token, how far is it from the
frame center (normalized 0=center, ~0.707=corner), compared against the
baseline (every spatial position, unweighted) -- broken down by WHICH
mechanism kept it (DTM anchor vs salience pick), since Sec 9.3.3/9.3.4 of
the main doc predict two different, independent reasons a corner bias could
show up (DTM's density-peaks favors large homogeneous background regions;
motion-salience picks up pan/rotation artifacts that are geometrically
stronger near frame edges).

Does NOT modify run_inference_qwen25vl_dyseg.py's stprune_segment (kept
return contract unchanged, zero risk to the already-validated pipeline) --
instead replicates just the anchor-index-selection math locally (same lines,
read-only) purely to tag which flat_index in `kept` came from DTM vs
salience.

Saves one bar chart (mean/median normalized center-distance per group) as
.jpg -- no plt.show(), viewable by copying the file off a headless server.

Usage:
  conda activate qwen25vl
  python analyze_spatial_bias.py \
    video_dir=Datasets/Video-MME/data/data \
    gt_file=playground/gt_qa_files/VideoMME/val_qa.json \
    n_videos=30 out_dir=outputs/spatial_bias
"""

import json
import os
import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image, ImageDraw

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

import run_inference_qwen25vl_dyseg as dm
from qwen_vision_process_fastvid import process_vision_info


def tag_dtm_flat_indices(X, seg, N, B_s, d_ratio, p_anchor):
    """Replicates stprune_segment's anchor-selection math (read-only, no
    merge/assign) purely to know which flat indices are DTM anchors, so
    kept-but-not-anchor can be tagged as "salience" without touching the
    real function's return contract."""
    s, e = seg
    B_s = max(1, int(B_s))
    n_dtm_total = max(0, int(round(d_ratio * B_s)))
    anchor_units = list(range(s, e, p_anchor)) or [s]
    n_dtm_per_anchor = max(1, n_dtm_total // len(anchor_units)) if n_dtm_total > 0 else 0

    dtm_flat = set()
    for t in anchor_units:
        if n_dtm_per_anchor <= 0:
            break
        toks = X[t]
        rho, delta = dm.density_peaks(toks)
        score = rho * delta
        n_pick = min(n_dtm_per_anchor, N)
        anchor_local_idx = torch.topk(score, n_pick).indices
        for a in anchor_local_idx.tolist():
            dtm_flat.add(t * N + a)
    return dtm_flat


def draw_bar_chart(groups, means, medians, baseline_mean, out_path, title=""):
    """Pure-PIL grouped bar chart (mean vs median per group) -- no
    matplotlib, so it works on a server with no internet access to pip
    install anything new. y-axis fixed to [0, 0.71] since values here are
    always normalized distances (0=center, ~0.707=corner)."""
    W, H = 700, 520
    pad_l, pad_r, pad_t, pad_b = 70, 30, 60, 90
    plot_w, plot_h = W - pad_l - pad_r, H - pad_t - pad_b
    y_max = 0.71

    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((pad_l, 15), title, fill=(0, 0, 0))

    def y_to_px(v):
        return pad_t + plot_h - (v / y_max) * plot_h

    # axes + gridlines
    draw.line([(pad_l, pad_t), (pad_l, pad_t + plot_h)], fill=(0, 0, 0), width=2)
    draw.line([(pad_l, pad_t + plot_h), (pad_l + plot_w, pad_t + plot_h)], fill=(0, 0, 0), width=2)
    for tick in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        y = y_to_px(tick)
        draw.line([(pad_l - 5, y), (pad_l + plot_w, y)], fill=(230, 230, 230), width=1)
        draw.text((pad_l - 45, y - 6), f"{tick:.1f}", fill=(0, 0, 0))

    baseline_y = y_to_px(baseline_mean)
    draw.line([(pad_l, baseline_y), (pad_l + plot_w, baseline_y)], fill=(150, 150, 150), width=1)

    n = len(groups)
    slot_w = plot_w / n
    bar_w = slot_w * 0.28
    for i, (g, m, md) in enumerate(zip(groups, means, medians)):
        cx = pad_l + slot_w * i + slot_w / 2
        # mean bar
        x0, x1 = cx - bar_w - 4, cx - 4
        y0 = y_to_px(m)
        draw.rectangle([x0, y0, x1, pad_t + plot_h], fill=(48, 112, 192))
        draw.text(((x0 + x1) / 2 - 12, y0 - 16), f"{m:.3f}", fill=(48, 112, 192))
        # median bar
        x0, x1 = cx + 4, cx + 4 + bar_w
        y0 = y_to_px(md)
        draw.rectangle([x0, y0, x1, pad_t + plot_h], fill=(224, 128, 48))
        draw.text(((x0 + x1) / 2 - 12, y0 - 16), f"{md:.3f}", fill=(224, 128, 48))
        draw.text((cx - 45, pad_t + plot_h + 8), g, fill=(0, 0, 0))

    draw.rectangle([pad_l, H - 30, pad_l + 15, H - 15], fill=(48, 112, 192))
    draw.text((pad_l + 22, H - 30), "mean", fill=(0, 0, 0))
    draw.rectangle([pad_l + 90, H - 30, pad_l + 105, H - 15], fill=(224, 128, 48))
    draw.text((pad_l + 112, H - 30), "median", fill=(0, 0, 0))
    draw.text((pad_l + 200, H - 30), "guong xam = mean baseline", fill=(120, 120, 120))

    img.save(out_path, quality=92)


def center_dist_grid(H, W):
    """[H,W] normalized distance-from-center, 0=center .. ~0.707=corner."""
    hh = (torch.arange(H).float() + 0.5) / H - 0.5
    ww = (torch.arange(W).float() + 0.5) / W - 0.5
    d = torch.sqrt(hh[:, None] ** 2 + ww[None, :] ** 2)
    return d  # [H,W]


def run(args):
    out_dir = dm.resolve_path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    max_pixels_units = int(getattr(args, "max_pixels_units", 49152))
    max_pixels = max_pixels_units * 28 * 28
    n_videos = int(getattr(args, "n_videos", 30))
    seed = int(getattr(args, "seed", 0))

    print(f"Loading {args.model_path} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=dtype, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(
        args.model_path, max_pixels=max_pixels, min_pixels=dm.MIN_PIXELS
    )

    video_dir = dm.resolve_path(args.video_dir)
    gt_file = dm.resolve_path(args.gt_file)
    with open(gt_file, "r", encoding="utf-8") as f:
        gt = json.load(f)
    video_names = sorted({s["video_name"] for s in gt})
    random.Random(seed).shuffle(video_names)
    video_names = video_names[:n_videos]
    print(f"Sampling {len(video_names)} videos (seed={seed})")

    all_dists, dtm_dists, sal_dists = [], [], []

    for video_name in video_names:
        video_path = os.path.join(video_dir, video_name)
        if not os.path.exists(video_path):
            print(f"skip missing: {video_path}")
            continue
        try:
            messages = dm.build_messages(video_path, "placeholder", ["A. a", "B. b", "C. c", "D. d"], max_pixels)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs, _ = process_vision_info(messages, return_video_kwargs=True)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
            ).to(model.device)

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

            dist_grid = center_dist_grid(H_llm, W_llm).reshape(-1)  # [N]
            all_dists.extend(dist_grid.tolist() * T_llm)  # every position, every unit

            segments = dm.dyseg_segments(X.mean(dim=1), c=args.dyseg_c, tau=args.dyseg_tau)
            v_s, _, _, _ = dm.segment_scores(X, segments, gamma=args.recon_gamma, lam=args.lam)
            B_s_list, _, _ = dm.allocate_budget(
                v_s, int(args.tokens_num), int(args.b_min), int(args.b_max),
                T0=args.temperature0, kappa=args.kappa, H0=args.h0,
            )

            salience_mode = getattr(args, "salience_mode", "delta")
            for seg, B_s in zip(segments, B_s_list):
                kept = dm.stprune_segment(
                    X, seg, (H_llm, W_llm), B_s, d_ratio=args.dtm_ratio,
                    p_anchor=args.p_anchor, alpha=args.dtm_alpha, salience_mode=salience_mode,
                )
                dtm_flat = tag_dtm_flat_indices(X, seg, N, B_s, args.dtm_ratio, args.p_anchor)
                for flat_idx in kept:
                    local = flat_idx % N
                    d_val = dist_grid[local].item()
                    if flat_idx in dtm_flat:
                        dtm_dists.append(d_val)
                    else:
                        sal_dists.append(d_val)

            print(f"{video_name}: T_llm={T_llm} segments={len(segments)} kept_dtm={len([1 for f in dtm_dists])}")
        except Exception as e:
            print(f"SKIP {video_name}: {type(e).__name__}: {e}")
            continue

    def summarize(name, vals):
        arr = np.array(vals)
        print(f"{name}: n={len(arr)} mean={arr.mean():.4f} median={np.median(arr):.4f}")
        return arr

    print("\n=== Normalized distance-from-center (0=center, ~0.707=corner) ===")
    all_arr = summarize("ALL positions (baseline)", all_dists)
    dtm_arr = summarize("DTM-kept", dtm_dists)
    sal_arr = summarize("Salience-kept", sal_dists)

    groups = ["Baseline (all)", "DTM-kept", "Salience-kept"]
    means = [all_arr.mean(), dtm_arr.mean(), sal_arr.mean()]
    medians = [np.median(all_arr), np.median(dtm_arr), np.median(sal_arr)]
    fig_path = os.path.join(out_dir, "spatial_bias.jpg")
    draw_bar_chart(groups, means, medians, all_arr.mean(), fig_path,
                    title=f"Spatial bias check ({len(video_names)} videos, salience_mode={getattr(args, 'salience_mode', 'delta')})")
    print(f"\nsaved {fig_path}")
    print("Neu DTM-kept/Salience-kept mean > Baseline mean ro ret -> xac nhan thien vi ve goc/rìa.")


@hydra.main(config_path="configs/inference_qwen25vl", config_name="dyseg", version_base=None)
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
