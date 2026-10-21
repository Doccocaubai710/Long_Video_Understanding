"""
vidcom2_port.py -- VidCom^2 cho Qwen2.5-VL trong cau hinh danh gia cua du an.

Nguon: Long_Video_Understanding/VidCom2 (commit 0e0159a). Repo goc DA ho tro Qwen2.5-VL
(token_compressor/vidcom2/models/qwen2_5_vl.py), nen o day chi CHEP NGUYEN VAN 5 ham cua
token_compressor/vidcom2/vidcom2.py + _compute_keep_indices cua models/qwen2_5_vl.py.
VidCom^2 chi CHON token (khong gop), truoc LLM. R_RATIO = base_scale = ti le giu.
"""
from typing import List, Tuple

import torch
import torch.nn.functional as F


# ====== BAT DAU khoi chep nguyen van (token_compressor/vidcom2/vidcom2.py) ======
def select_low_var_channels(x: torch.Tensor, ratio: float = 0.5) -> torch.Tensor:
    """Selects least informative channels (lowest variance)."""
    variances = x.var(dim=0, unbiased=False)
    k = int(x.shape[-1] * ratio)
    _, topk_idx = torch.topk(variances, k=k, largest=False)
    return x[:, topk_idx]

def compute_gaussian_scores(x: torch.Tensor, tpf: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes Gaussian similarity to Video and Frame centers."""
    frames = x.view(-1, tpf, x.shape[-1])
    frames = F.normalize(frames, dim=-1)
    
    # Broadcastable centers: (1, 1, C) and (B, 1, C)
    vid_center = frames.mean(dim=(0, 1), keepdim=True) 
    frame_center = frames.mean(dim=1, keepdim=True)
    
    alphas = [2**k for k in range(-3, 2)]
    v_score = _multi_scale_gaussian(frames, vid_center, alphas)
    f_score = _multi_scale_gaussian(frames, frame_center, alphas)
    return v_score, f_score

def _multi_scale_gaussian(x: torch.Tensor, center: torch.Tensor, alphas: List[float]) -> torch.Tensor:
    """Helper: vectorized multi-scale Gaussian kernel."""
    dist_sq = ((x - center) ** 2).sum(dim=-1)
    return sum(torch.exp(-dist_sq / (2 * a)) for a in alphas)

def compute_scales(scores: torch.Tensor, base: float, temp: float = 0.01) -> torch.Tensor:
    """Generates dynamic retention rates based on frame importance."""
    probs = F.softmax((scores - scores.max()) / temp, dim=0)
    scales = base * (1 + probs - probs.mean())
    return scales.clamp(max=1.0)

def select_outlier_indices(scores: torch.Tensor, scales: torch.Tensor, tpf: int) -> List[torch.Tensor]:
    """Selects top-k indices with lowest similarity (largest outliers)."""
    ks = (scales * tpf).round().long().clamp(min=1).tolist()
    batch_indices = []
    for i, k in enumerate(ks):
        # largest=False -> retain most distinct tokens
        _, idx = torch.topk(scores[i], k=k, largest=False, sorted=False)
        batch_indices.append(idx.sort().values) 
    return batch_indices

def _map_linear_offset(indices: List[torch.Tensor], tpf: int) -> torch.Tensor:
    """Standard mapping: adds frame_offset to local indices (Llava-OV / Qwen2-VL)."""
    device = indices[0].device
    offsets = torch.arange(len(indices), device=device) * tpf
    return torch.cat([idx + off for idx, off in zip(indices, offsets)])
# ====== KET THUC khoi chep nguyen van ======


@torch.no_grad()
def vidcom2_select(video_embeds, video_grid_thw, base_scale):
    """video_embeds [T*N, D] (dtype cua inputs_embeds, nhu ban goc) -> (kept_indices tang dan, kept_embeds, diag)."""
    t, h, w = video_grid_thw[0].tolist()
    frame_tokens = (h * w) // 4
    # --- _compute_keep_indices cua models/qwen2_5_vl.py ---
    sel_feat = select_low_var_channels(video_embeds)
    vid_score, frame_score = compute_gaussian_scores(sel_feat, frame_tokens)
    scales = compute_scales(-vid_score.mean(dim=-1), base_scale)
    indices = select_outlier_indices(vid_score + frame_score, scales, frame_tokens)
    keep = torch.sort(_map_linear_offset(indices, frame_tokens)).values
    ks = [int(i.numel()) for i in indices]
    return keep, video_embeds[keep], {"mode": "vidcom2", "k_min": min(ks), "k_max": max(ks)}
