"""
precompute_frame_dpc.py

"Tang 0" preselection: pick WHICH raw frames of a video get handed to
Qwen2.5-VL, instead of the model's own blind fps=2.0/FPS_MAX_FRAMES=768
uniform sampling (qwen_vision_process_fastvid.smart_nframes). For long
videos the uniform cap can throw away frames covering an entire scene
(see README_DySeg_STPrune.md discussion) -- this script instead:

  1. Samples DENSE candidate frames at 2fps across the WHOLE video (no
     768 cap here -- DINOv2 forward passes are much cheaper than Qwen's).
  2. Extracts a pooled DINOv2 feature per candidate frame (facebook/
     dinov2-large, same model/transform as keyframe_select_new.py).
  3. Runs the EXACT SAME density-peaks ranking already used for DTM in
     run_inference_qwen25vl_dyseg.py (density_peaks(): rho=local density,
     delta=distance to nearest higher-density point), applied here at
     frame level instead of token level. score = rho*delta naturally
     picks ~1 representative per distinct "mode" of content: redundant
     near-duplicate frames (a long static stretch) collapse to a handful
     of representatives, while frames from visually distinct/volatile
     stretches rank highly and get more representatives.
  4. Keeps the top `target_frames` candidates by that score, sorted back
     into chronological order, and caches the resulting RAW frame indices
     to disk -- one JSON file per video, reused across all questions for
     that video (same amortization pattern as the DySeg {S,f} cache and
     the V-JEPA salience cache).

The cached frame_indices are consumed by run_inference_qwen25vl_native.py
via `frame_cache_dir=...` (bypasses fps=2.0 sampling entirely -- see
qwen_vision_process_fastvid._read_video_decord's "frame_indices" branch).

Usage:
  python precompute_frame_dpc.py \
    video_dir=Datasets/Video-MME/data/data \
    gt_file=playground/gt_qa_files/VideoMME/val_qa.json \
    cache_dir=outputs/frame_dpc_cache \
    target_frames=768
"""

import json
import os

import decord
import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoImageProcessor, Dinov2Model

from run_inference_qwen25vl_dyseg import density_peaks

os.environ.setdefault("DECORD_EOF_RETRY_MAX", "20480")

DINO_MODEL_NAME = "facebook/dinov2-large"
FRAME_FACTOR = 2  # keep consistent with qwen_vision_process_fastvid.FRAME_FACTOR

fast_transform = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def resolve_path(path):
    if path is None:
        return None
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    return to_absolute_path(path)


def candidate_indices_2fps(total_frames, video_fps, fps=2.0):
    """Dense, uncapped 2fps candidate frame indices across the WHOLE video
    (mirrors smart_nframes()'s target-count formula but WITHOUT the
    FPS_MAX_FRAMES=768 clamp -- that clamp is exactly what this script
    exists to route around)."""
    n = max(FRAME_FACTOR, round(total_frames / max(video_fps, 1e-6) * fps))
    n = min(n, total_frames)
    return np.linspace(0, total_frames - 1, n).round().astype(int)


@torch.inference_mode()
def extract_dino_features(video_path, indices, model, device, batch_size=64):
    """Decode frames at the given raw indices and extract pooled DINOv2
    features, batched. Returns [len(indices), D] float32 CPU tensor."""
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    feats = []
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        frames = vr.get_batch(batch_idx).asnumpy()  # [b,H,W,C] uint8
        tensors = torch.stack([fast_transform(Image.fromarray(f)) for f in frames]).to(device)
        out = model(pixel_values=tensors)
        feats.append(out.pooler_output.float().cpu())
    return torch.cat(feats, dim=0)


def select_frames_by_density_peaks(features, target_frames):
    """features: [M, D] CPU tensor (M candidate frames). Returns sorted
    (chronological) list of positions into `features`/`indices`, length
    min(target_frames, M) rounded down to FRAME_FACTOR."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rho, delta = density_peaks(features.to(device))
    score = rho * delta
    k = min(target_frames, features.shape[0])
    k = max(FRAME_FACTOR, (k // FRAME_FACTOR) * FRAME_FACTOR)
    topk = torch.topk(score, k).indices.cpu().numpy()
    return sorted(topk.tolist())


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {DINO_MODEL_NAME} ...")
    model = Dinov2Model.from_pretrained(DINO_MODEL_NAME).eval().to(device)

    video_dir = resolve_path(args.video_dir)
    gt_file = resolve_path(args.gt_file)
    cache_dir = resolve_path(args.cache_dir)
    target_frames = int(getattr(args, "target_frames", 768))
    os.makedirs(cache_dir, exist_ok=True)

    with open(gt_file, "r") as f:
        gt_qa_pairs = json.load(f)
    video_names = sorted({sample["video_name"] for sample in gt_qa_pairs})
    print(f"{len(video_names)} unique videos, target_frames={target_frames}")

    for video_name in tqdm(video_names):
        cache_path = os.path.join(cache_dir, os.path.basename(video_name) + ".json")
        if os.path.exists(cache_path):
            continue
        video_path = os.path.join(video_dir, video_name)
        if not os.path.exists(video_path):
            print(f"Missing video: {video_path}")
            continue
        try:
            vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
            total_frames, video_fps = len(vr), vr.get_avg_fps()
            del vr
            candidates = candidate_indices_2fps(total_frames, video_fps)
            feats = extract_dino_features(video_path, candidates, model, device)
            positions = select_frames_by_density_peaks(feats, target_frames)
            frame_indices = [int(candidates[p]) for p in positions]
        except Exception as e:
            print(f"SKIP {video_name}: {type(e).__name__}: {e}")
            continue
        with open(cache_path, "w") as f:
            json.dump({
                "frame_indices": frame_indices,
                "n_candidates_2fps": int(len(candidates)),
                "total_frames": int(total_frames),
                "video_fps": float(video_fps),
            }, f)

    print(f"[frame-DPC precompute] done -> {cache_dir}")


@hydra.main(config_path=None, version_base=None)
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
