# event_boundary_clustering.py
"""
AFP (Adaptive Frame-Pruning) — từ paper "Less is More" (CVPR 2025, 2508.03337)
Adapted cho KTV pipeline: dùng DINO features thay vì ResNet+CLIP của paper gốc.

Drop-in replacement cho video_frame_clustering() trong cluster_and_rank_keyframes.py.

Điểm khác với KTV:
  - KTV: K-means fixed K, ignore temporal order
  - AFP: AHC bottom-up, distance = visual + temporal, K tự động từ KDE threshold
"""

import numpy as np
from scipy.stats import gaussian_kde
from sklearn.cluster import AgglomerativeClustering


# ─── Step 1: Distance matrix ────────────────────────────────────────────────

def compute_distance_matrix(features: np.ndarray, beta: float = 0.9) -> np.ndarray:
    """
    D(i,j) = beta * d_cos(i,j) + (1-beta) * d_temp(i,j)

    beta=0.9 theo paper gốc: visual distance chiếm 90%, temporal 10%.
    d_temp = |i-j| / (N-1): normalized timestamp distance.

    Frames giống nhau về visual nhưng cách xa về thời gian → ít bị merge hơn.
    """
    N = len(features)

    # L2 normalise để tính cosine distance
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    feats = features / np.where(norms == 0, 1.0, norms)

    # Cosine distance: 1 - cos_sim, range [0, 2]
    dot       = feats @ feats.T
    d_visual  = np.clip(1.0 - dot, 0.0, 2.0).astype(np.float64)

    # Temporal distance: normalized, range [0, 1]
    idx        = np.arange(N, dtype=np.float64)
    denom      = max(N - 1, 1)
    d_temporal = np.abs(idx[:, None] - idx[None, :]) / denom

    return beta * d_visual + (1.0 - beta) * d_temporal


# ─── Step 2: KDE threshold ──────────────────────────────────────────────────

def compute_kde_threshold(features: np.ndarray,
                          offset_ratio: float = 0.1) -> float:
    """
    Theo paper: fit Gaussian KDE lên phân phối pairwise cosine distances,
    tìm PEAK của density curve, đặt threshold hơi cao hơn peak một chút.

    Peak = khoảng cách phổ biến nhất giữa các frames.
    Threshold hơi trên peak → merge các cặp similar hơn mức trung bình.

    offset_ratio=0.1: threshold = peak + 10% * (max - peak)
    """
    N = len(features)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    feats = features / np.where(norms == 0, 1.0, norms)

    # Lấy upper triangle của cosine distance matrix (bỏ diagonal)
    dot      = feats @ feats.T
    d_visual = np.clip(1.0 - dot, 0.0, 2.0)
    upper    = d_visual[np.triu_indices(N, k=1)]

    if len(upper) < 3:
        return float(np.mean(upper)) if len(upper) > 0 else 0.5

    # Fit KDE
    kde = gaussian_kde(upper, bw_method='silverman')
    x   = np.linspace(upper.min(), upper.max(), 500)
    y   = kde(x)

    # Tìm peak (điểm có density cao nhất)
    peak_idx = int(np.argmax(y))
    peak_val = float(x[peak_idx])

    # Threshold = peak + offset nhỏ
    threshold = peak_val + offset_ratio * (float(upper.max()) - peak_val)
    return threshold


# ─── Step 3: Agglomerative clustering ───────────────────────────────────────

def run_ahc(dist_matrix: np.ndarray,
            threshold: float,
            max_segments: int,
            min_segments: int) -> np.ndarray:
    """
    AHC với average linkage + distance_threshold.
    Nếu K vượt quá max_segments hoặc dưới min_segments thì clamp lại.
    Trả về array labels (N,).
    """
    N = dist_matrix.shape[0]

    # Thử với adaptive threshold trước
    ahc = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric='precomputed',
        linkage='average',
        compute_full_tree=True,
    )
    labels    = ahc.fit_predict(dist_matrix)
    n_cluster = len(set(labels))

    # Clamp vào [min_segments, max_segments]
    if n_cluster > max_segments:
        ahc2   = AgglomerativeClustering(
            n_clusters=max_segments,
            metric='precomputed',
            linkage='average',
        )
        labels = ahc2.fit_predict(dist_matrix)
    elif n_cluster < min_segments:
        actual = min(min_segments, N)
        ahc2   = AgglomerativeClustering(
            n_clusters=actual,
            metric='precomputed',
            linkage='average',
        )
        labels = ahc2.fit_predict(dist_matrix)

    return labels


# ─── Step 4: Pick representative frame ──────────────────────────────────────

def pick_centroid_frames(features: np.ndarray, labels: np.ndarray) -> list:
    """
    Với mỗi cluster, chọn frame gần centroid nhất — giống KTV.
    """
    result     = []
    n_clusters = len(set(labels))
    for cid in range(n_clusters):
        mask    = labels == cid
        indices = np.where(mask)[0]
        seg     = features[indices]
        mean    = seg.mean(axis=0)
        dists   = np.linalg.norm(seg - mean, axis=1)
        result.append(int(indices[np.argmin(dists)]))
    return sorted(result)


# ─── Main entry point ────────────────────────────────────────────────────────

def event_boundary_clustering(frame_features,
                               k_budget: int = 12,
                               beta: float = 0.9,
                               offset_ratio: float = 0.1,
                               max_segments: int = 12,
                               min_segments: int = 6,
                               verbose: bool = True) -> list:
    """
    Drop-in replacement cho video_frame_clustering(frame_features, num_cluster).

    Args:
        frame_features : list hoặc np.ndarray (N, D) — DINO features
        k_budget       : giữ cho backward compat, không dùng trong AHC
        beta           : weight cho visual distance (0.9 theo paper gốc)
        offset_ratio   : threshold = KDE_peak + ratio*(max - peak)
        max_segments   : safety upper cap
        min_segments   : floor để đảm bảo đủ candidates cho CLIP
        verbose        : print thông tin

    Returns:
        List[int] — sampled-frame indices, sorted, length trong [min_segments, max_segments]
    """
    features = np.array(frame_features, dtype=np.float32)
    N        = len(features)

    if N <= min_segments:
        if verbose:
            print(f"[AFP] {N} frames ≤ min={min_segments}, returning all.")
        return list(range(N))

    # 1. Distance matrix
    dist_matrix = compute_distance_matrix(features, beta=beta)

    # 2. KDE threshold (chỉ dùng cosine distances, không temporal)
    threshold = compute_kde_threshold(features, offset_ratio=offset_ratio)

    if verbose:
        print(f"[AFP] {N} frames | KDE threshold={threshold:.4f} "
              f"| beta={beta} | min={min_segments} max={max_segments}")

    # 3. AHC
    labels    = run_ahc(dist_matrix, threshold, max_segments, min_segments)
    n_cluster = len(set(labels))

    if verbose:
        sizes = sorted([int((labels == i).sum()) for i in range(n_cluster)], reverse=True)
        print(f"[AFP] K={n_cluster} clusters | sizes={sizes}")

    # 4. Centroid frame per cluster
    return pick_centroid_frames(features, labels)