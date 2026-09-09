"""
precompute_vjepa_salience.py

Precomputes a per-video V-JEPA 2 masked-prediction-error grid, cached to disk,
for use as an alternative Tang 3b salience signal in run_inference_qwen25vl_dyseg.py
(salience_mode=vjepa). See README_DySeg_STPrune.md / main design doc Sec 9 for
why: err(tubelet) = ||predictor_output − encoder_output_at_target|| measures
"how surprising is this region given the rest of the video's context" (semantic,
uses the WHOLE video as context) vs the current ||X[t]-X[t-1]|| (motion-only,
1-step-back). One value per video; cached once, reused across all ~3 questions
for that video (same amortization pattern as DySeg's own {S,f} cache).

Verified against the real HF transformers V-JEPA2 API (docs +
modeling_vjepa2.py source, transformers v4.53.3/v5.x) -- NOT guessed:
  - Checkpoint: facebook/vjepa2-vitl-fpc64-256 (patch_size=16, crop_size=256,
    frames_per_clip=64, tubelet_size=2 -> spatial grid 16x16, temporal grid
    64/2=32, N=32*16*16=8192 patches total, row-major (t,h,w) flat order).
  - VJEPA2Model.forward(pixel_values_videos, context_mask, target_mask):
    context_mask/target_mask are List[torch.Tensor] of patch INDICES (shape
    [batch, n_idx]), not boolean masks.
  - predictor_output.last_hidden_state = predictions AT the target positions.
  - predictor_output.target_hidden_state = encoder's own output at those same
    target positions -- i.e. the ground truth is handed back directly, no
    need to separately re-gather it from the full encoder output.

Multi-pass coverage: V-JEPA pretraining masks ~75-90% of patches per sample.
A single such pass would leave most patches WITHOUT an error value. We split
all 8192 patches into K=4 disjoint random partitions and run 4 forward
passes, each time treating one partition as target (25% masked, 75% context)
and the rest as context -- every patch gets exactly one error value, each
computed from strong (75%) context, close to the pretraining regime.

Usage:
  python precompute_vjepa_salience.py \
    video_dir=Datasets/Video-MME/data/data \
    gt_file=playground/gt_qa_files/VideoMME/val_qa.json \
    cache_dir=outputs/vjepa_salience_cache

Requires: pip install torchcodec av   (video decode for AutoVideoProcessor;
falls back to decord-based frame reading below since torchcodec is not used
elsewhere in this repo)
"""

import json
import os

import decord
import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoModel, AutoVideoProcessor

os.environ.setdefault("DECORD_EOF_RETRY_MAX", "20480")


def resolve_path(path):
    if path is None:
        return None
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    return to_absolute_path(path)


def load_video_frames(video_path, num_frames=64):
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    total = len(vr)
    idx = np.linspace(0, total - 1, num_frames).round().astype(int)
    idx = np.clip(idx, 0, total - 1)
    frames = vr.get_batch(idx).asnumpy()  # [num_frames, H, W, C], uint8
    return frames


def build_partitions(n_patches, k, generator):
    """Split [0, n_patches) into k disjoint random partitions (near-equal size)."""
    perm = torch.randperm(n_patches, generator=generator)
    return [p.clone() for p in perm.chunk(k)]


@torch.inference_mode()
def compute_error_grid(model, processor, frames, device, dtype, grid_depth, grid_size, k_passes, generator):
    """frames: [T,H,W,C] uint8 numpy. Returns err: [grid_depth, grid_size, grid_size] float32 CPU tensor."""
    inputs = processor(frames, return_tensors="pt").to(device)
    pixel_values_videos = inputs["pixel_values_videos"].to(dtype)

    n_patches = grid_depth * grid_size * grid_size
    partitions = build_partitions(n_patches, k_passes, generator)

    err_flat = torch.zeros(n_patches, dtype=torch.float32)
    all_idx = torch.arange(n_patches)

    for target_idx in partitions:
        target_set = set(target_idx.tolist())
        context_idx = all_idx[torch.tensor([i.item() not in target_set for i in all_idx])]

        context_mask = [context_idx.unsqueeze(0).to(device)]
        target_mask = [target_idx.unsqueeze(0).to(device)]

        out = model(
            pixel_values_videos=pixel_values_videos,
            context_mask=context_mask,
            target_mask=target_mask,
        )
        pred = out.predictor_output.last_hidden_state[0].float()      # [n_tgt, D]
        target = out.predictor_output.target_hidden_state[0].float()  # [n_tgt, D]
        e = (pred - target).norm(dim=-1).cpu()                        # [n_tgt]
        err_flat[target_idx] = e

    return err_flat.reshape(grid_depth, grid_size, grid_size)


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    checkpoint = getattr(args, "checkpoint", "facebook/vjepa2-vitl-fpc64-256")
    k_passes = int(getattr(args, "k_passes", 4))

    print(f"Loading {checkpoint} ...")
    processor = AutoVideoProcessor.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint, torch_dtype=dtype, device_map="auto").eval()

    grid_size = model.config.crop_size // model.config.patch_size
    grid_depth = model.config.frames_per_clip // model.config.tubelet_size
    n_patches = grid_depth * grid_size * grid_size
    print(f"grid: T={grid_depth} H=W={grid_size} N={n_patches} patches, k_passes={k_passes}")

    video_dir = resolve_path(args.video_dir)
    gt_file = resolve_path(args.gt_file)
    cache_dir = resolve_path(args.cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    with open(gt_file, "r") as f:
        gt_qa_pairs = json.load(f)
    video_names = sorted({sample["video_name"] for sample in gt_qa_pairs})
    print(f"{len(video_names)} unique videos")

    generator = torch.Generator().manual_seed(0)  # same partitions every run -> reproducible cache

    for video_name in tqdm(video_names):
        cache_path = os.path.join(cache_dir, video_name.replace("/", "_") + ".pt")
        if os.path.exists(cache_path):
            continue
        video_path = os.path.join(video_dir, video_name)
        if not os.path.exists(video_path):
            print(f"Missing video: {video_path}")
            continue
        try:
            frames = load_video_frames(video_path, num_frames=model.config.frames_per_clip)
            err_grid = compute_error_grid(
                model, processor, frames, device, dtype, grid_depth, grid_size, k_passes, generator
            )
        except Exception as e:
            print(f"SKIP {video_name}: {type(e).__name__}: {e}")
            continue
        torch.save(err_grid, cache_path)

    print(f"[V-JEPA salience precompute] done -> {cache_dir}")


@hydra.main(config_path=None, version_base=None)
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
