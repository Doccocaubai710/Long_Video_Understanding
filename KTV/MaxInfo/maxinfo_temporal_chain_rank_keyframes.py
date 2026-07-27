# maxinfo_temporal_chain_rank_keyframes.py
"""
MaxInfo + Temporal-Chain: thay Rcluster(K-means) bằng R_maxvol (volume gain).
K thích ứng: dừng khi volume gain < tol thay vì fixed K=12.
Fully query-agnostic, training-free.

Chạy: uv run python maxinfo_temporal_chain_rank_keyframes.py experiment=nextqa
"""

import os, json, pickle, cv2
import numpy as np
import torch
from tqdm import tqdm
from scipy.linalg import qr
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

# ── Reuse các helper từ cluster_and_rank_keyframes ──────────────────────────
from cluster_and_rank_keyframes import (
    load_video_frame_tensor,
    get_tensor_for_video,
    get_original_frame_number,
)


# ── MaxVol: tính volume gain khi thêm frame t vào tập S ─────────────────────

def svd_reduce(features: np.ndarray, s: int) -> np.ndarray:
    """Truncated SVD: (N, D) → (N, s)"""
    s = min(s, features.shape[0], features.shape[1])
    try:
        U, _, _ = np.linalg.svd(features, full_matrices=False)
        return U[:, :s].astype(np.float64)
    except np.linalg.LinAlgError:
        return features[:, :s].astype(np.float64)


def volume_gain(t: int, selected: list, Qs: np.ndarray) -> float:
    if len(selected) == 0:
        return float(np.linalg.norm(Qs[t]))
    AK = Qs[selected]   # (len(selected), s)
    try:
        AK_pinv = np.linalg.pinv(AK)   # (s, len(selected))
    except np.linalg.LinAlgError:
        return 1.0
    c = Qs[t] @ AK_pinv   # (s,) @ (s, len(selected)) → (len(selected),)
    return float(np.sqrt(1.0 + np.dot(c, c)))


# ── Temporal-chain terms ─────────────────────────────────────────────────────

def compute_event_scores(features: np.ndarray, K: int) -> np.ndarray:
    """Revent(t) = ‖ft − ft−Δ‖ + ‖ft+Δ − ft‖"""
    N = len(features)
    delta = max(1, int(np.floor(10 * K / N)))
    scores = np.zeros(N)
    for t in range(N):
        left  = features[max(0, t - delta)]
        right = features[min(N - 1, t + delta)]
        scores[t] = np.linalg.norm(features[t] - left) + np.linalg.norm(right - features[t])
    return scores


def normalize(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def temporal_gap(t: int, selected: list, N: int) -> float:
    """T_gap(t, S) = min_{s∈S} |t − s| (normalized)"""
    if not selected:
        return 1.0
    return float(min(abs(t - s) for s in selected)) / N


def redundancy(t: int, selected: list, features: np.ndarray) -> float:
    """R_red(t, S) = max_{s∈S} cos(ft, fs)"""
    if not selected:
        return 0.0
    ft = features[t]
    norm_t = np.linalg.norm(ft)
    if norm_t < 1e-8:
        return 0.0
    sims = []
    for s in selected:
        fs = features[s]
        norm_s = np.linalg.norm(fs)
        if norm_s > 1e-8:
            sims.append(float(np.dot(ft, fs) / (norm_t * norm_s)))
    return max(sims) if sims else 0.0


# ── Main selection: MaxInfo + Temporal Chain ─────────────────────────────────

def maxinfo_temporal_chain_select(
    features: np.ndarray,
    lambda_event: float = 0.5,
    alpha_gap: float = 0.6,
    beta_redundancy: float = 0.8,
    tol: float = 1.05,         # dừng khi volume gain < tol
    max_K: int = 20,           # safety cap
    min_K: int = 6,            # floor
    svd_s: int = 16,           # số singular vectors cho SVD
) -> list:
    """
    Greedy selection với:
      G(t|S) = R̃_maxvol(t,S) + λ·R̃_event(t) + α·T̃_gap(t,S) − β·R̃_red(t,S)

    Dừng khi max volume gain < tol (K thích ứng) hoặc đạt max_K.
    """
    N = len(features)
    if N <= min_K:
        return list(range(N))

    # SVD reduce
    Qs = svd_reduce(features, svd_s)

    # Pre-compute event scores (dùng max_K như estimate K)
    event_scores = normalize(compute_event_scores(features, max_K))

    selected = []
    candidates = list(range(N))

    while len(candidates) > 0:
        # Điều kiện dừng
        if len(selected) >= max_K:
            break

        # Tính G cho mỗi candidate
        vol_gains   = np.array([volume_gain(t, selected, Qs) for t in candidates])
        tgap_scores = np.array([temporal_gap(t, selected, N)  for t in candidates])
        red_scores  = np.array([redundancy(t, selected, features) for t in candidates])

        # Normalize các term phụ
        vol_norm  = normalize(vol_gains)
        tgap_norm = normalize(tgap_scores)
        red_norm  = normalize(red_scores)
        ev_norm   = event_scores[candidates]

        G = vol_norm + lambda_event * ev_norm + alpha_gap * tgap_norm - beta_redundancy * red_norm

        best_local = int(np.argmax(G))
        best_t     = candidates[best_local]
        best_gain  = vol_gains[best_local]

        # Dừng khi gain nhỏ và đã đủ min_K
        if best_gain < tol and len(selected) >= min_K:
            break

        selected.append(best_t)
        candidates.pop(best_local)

    # Floor
    if len(selected) < min_K:
        remaining = [t for t in range(N) if t not in set(selected)]
        for t in remaining:
            if len(selected) >= min_K:
                break
            selected.append(t)

    return sorted(selected)


# ── Pipeline: đọc pkl → select → map → lưu JSON ─────────────────────────────

def run(
    json_path, video_path, video_frame_tensor_path,
    save_cluster_path, combined_output_path=None,
    lambda_event=0.5, alpha_gap=0.6, beta_redundancy=0.8,
    tol=1.05, max_K=20, min_K=6,
    max_frames_to_extract=5400,
):
    os.makedirs(save_cluster_path, exist_ok=True)
    video_frame_tensor = load_video_frame_tensor(video_frame_tensor_path)

    with open(json_path) as f:
        qa_data = json.load(f)

    combined_results = {}

    for sample in tqdm(qa_data):
        video_name  = sample["video_name"]
        question_id = sample["question_id"]

        output_path = os.path.join(save_cluster_path, f"{question_id}.json")
        if os.path.exists(output_path):
            with open(output_path) as f:
                combined_results.update(json.load(f))
            continue

        tensor = get_tensor_for_video(video_frame_tensor, video_name)
        if tensor is None:
            print(f"[SKIP] missing tensor: {video_name}")
            continue

        features = np.array(tensor, dtype=np.float32)
        N = len(features)

        # MaxInfo + Temporal Chain selection (adaptive K)
        selected_indices = maxinfo_temporal_chain_select(
            features,
            lambda_event=lambda_event,
            alpha_gap=alpha_gap,
            beta_redundancy=beta_redundancy,
            tol=tol,
            max_K=max_K,
            min_K=min_K,
        )

        print(f"[MaxInfo-TC] {video_name}: N={N}, K={len(selected_indices)}, frames={selected_indices}")

        # Map sampled-list index → original frame number
        cap = cv2.VideoCapture(os.path.join(video_path, video_name))
        fps          = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        candidate_frame_indices = [
            get_original_frame_number(
                total_frames, idx, fps=fps,
                max_frames_to_extract=max_frames_to_extract,
            )
            for idx in selected_indices
        ]

        # Rank = temporal order (0 = earliest) — fully agnostic, no CLIP
        key_frame_order = [
            [frame_idx, rank]
            for rank, frame_idx in enumerate(candidate_frame_indices)
        ]

        result = {question_id: key_frame_order}
        combined_results.update(result)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2,
                      default=lambda o: int(o) if isinstance(o, np.integer) else o)

    if combined_output_path:
        os.makedirs(os.path.dirname(combined_output_path) or ".", exist_ok=True)
        with open(combined_output_path, "w") as f:
            json.dump(combined_results, f, indent=2,
                      default=lambda o: int(o) if isinstance(o, np.integer) else o)

    print(f"Done. Total={len(qa_data)}")


def resolve(path):
    if path is None: return None
    path = os.path.expanduser(path)
    return path if os.path.isabs(path) else to_absolute_path(path)


@hydra.main(config_path="configs/keyframe_cluster", config_name="config", version_base=None)
def main(cfg: DictConfig):
    run(
        json_path                = resolve(cfg.json_path),
        video_path               = resolve(cfg.video_path),
        video_frame_tensor_path  = resolve(cfg.video_frame_tensor_path),
        save_cluster_path        = resolve(cfg.save_cluster_path),
        combined_output_path     = resolve(cfg.combined_output_path),
        lambda_event             = cfg.get("lambda_event",    0.5),
        alpha_gap                = cfg.get("alpha_gap",       0.6),
        beta_redundancy          = cfg.get("beta_redundancy", 0.8),
        tol                      = cfg.get("tol",             1.05),
        max_K                    = cfg.get("max_K",           20),
        min_K                    = cfg.get("min_K",           6),
        max_frames_to_extract    = cfg.get("max_frames_to_extract", 5400),
    )

if __name__ == "__main__":
    main()