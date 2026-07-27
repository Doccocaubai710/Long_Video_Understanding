# tacsum_clustering.py
"""
TAC-SUM (Temporal-Aware Cluster-based SUMmarization) — từ paper 2404.04511
Adapted cho KTV pipeline.

Ý tưởng cốt lõi: KTV dùng K-means nhưng bỏ qua temporal order.
TAC-SUM fix bằng cách:
  1. Chạy K-means trong feature space (giống KTV, K cố định)
  2. Convert cluster labels thành temporal segments LÊN TỤC
  3. Chọn representative frame từ mỗi temporal segment (không phải từ cluster)

Kết quả: số frames có thể > K vì 1 cluster có thể xuất hiện
ở nhiều vị trí không liên tục trong video.

Drop-in replacement cho video_frame_clustering() trong cluster_and_rank_keyframes.py.
"""

import numpy as np
from sklearn.cluster import KMeans


# ─── Step 1 + 2: KMeans → temporal segments ─────────────────────────────────

def kmeans_to_temporal_segments(features: np.ndarray,
                                 n_clusters: int) -> list:
    """
    Chạy K-means trong feature space, sau đó convert labels thành
    các temporal segment LIÊN TỤC.

    Ví dụ:
      Labels theo thời gian: [3, 3, 3, 1, 1, 3, 3, 2, 2, 1, 1, 3]
      Temporal segments:      [0:3, 3:5, 5:7, 7:9, 9:11, 11:12]
                               (cùng cluster 3 nhưng 3 segment khác nhau)

    Returns:
        list of (start, end, cluster_id) tuples
    """
    N = len(features)

    km     = KMeans(n_clusters=n_clusters, random_state=0,
                    init='k-means++', n_init=10)
    labels = km.fit_predict(features)   # (N,) int

    # Tìm các đoạn liên tục cùng label
    segments = []
    start    = 0
    for i in range(1, N):
        if labels[i] != labels[i - 1]:
            segments.append((start, i, int(labels[start])))
            start = i
    segments.append((start, N, int(labels[start])))

    return segments, labels, km.cluster_centers_


# ─── Step 3: Pick representative frame per temporal segment ─────────────────

def pick_segment_representatives(features: np.ndarray,
                                  segments: list) -> list:
    """
    Với mỗi temporal segment, chọn frame gần centroid của segment nhất.
    (Không phải centroid của toàn cluster — đây là điểm khác với KTV.)
    """
    result = []
    for start, end, _ in segments:
        seg_feats = features[start:end]
        mean      = seg_feats.mean(axis=0)
        dists     = np.linalg.norm(seg_feats - mean, axis=1)
        result.append(int(start + np.argmin(dists)))
    return result


# ─── Step 4: Merge short segments (optional, controls max output K) ──────────

def merge_short_segments(segments: list,
                          features: np.ndarray,
                          min_len: int = 2,
                          max_output: int = 20) -> list:
    """
    Merge các segment quá ngắn (< min_len frames) vào neighbour gần nhất.
    Sau đó nếu vẫn vượt max_output thì merge các cặp similar nhất.

    min_len=2: segment chỉ có 1 frame rất ngắn, merge vào cạnh nó.
    """
    # Bước 1: merge segment quá ngắn
    changed = True
    while changed:
        changed = False
        new_segs = []
        i = 0
        while i < len(segments):
            s, e, cid = segments[i]
            if (e - s) < min_len and len(segments) > 1:
                # Merge vào segment trước hoặc sau
                if i == 0:
                    # Merge vào segment sau
                    ns, ne, ncid = segments[i + 1]
                    segments = [(s, ne, ncid)] + segments[i + 2:]
                else:
                    # Merge vào segment trước
                    ps, pe, pcid = new_segs[-1]
                    new_segs[-1] = (ps, e, pcid)
                    i += 1
                changed = True
                break
            else:
                new_segs.append((s, e, cid))
                i += 1
        if changed:
            segments = new_segs + segments[i + 1:] if not changed else segments

    # Bước 2: nếu vẫn > max_output, merge cặp similar nhất
    while len(segments) > max_output:
        # Tính mean feature per segment
        means = []
        for s, e, _ in segments:
            m    = features[s:e].mean(axis=0)
            norm = np.linalg.norm(m)
            means.append(m / norm if norm > 0 else m)

        best_sim, best_i = -2.0, 0
        for i in range(len(segments) - 1):
            sim = float(np.dot(means[i], means[i + 1]))
            if sim > best_sim:
                best_sim, best_i = sim, i

        s1, _, c1   = segments[best_i]
        _, e2, _    = segments[best_i + 1]
        segments    = (segments[:best_i]
                       + [(s1, e2, c1)]
                       + segments[best_i + 2:])

    return segments


# ─── Main entry point ────────────────────────────────────────────────────────

def tacsum_clustering(frame_features,
                      n_clusters: int = 12,
                      min_seg_len: int = 2,
                      max_output: int = 20,
                      min_output: int = 6,
                      verbose: bool = True) -> list:
    """
    TAC-SUM drop-in replacement cho video_frame_clustering().

    Args:
        frame_features : list hoặc np.ndarray (N, D) — DINO features
        n_clusters     : K cho K-means (giống num_keyframes trong KTV)
        min_seg_len    : merge segment ngắn hơn giá trị này
        max_output     : cap tối đa số frame trả về
        min_output     : floor tối thiểu số frame trả về
        verbose        : print thông tin

    Returns:
        List[int] — sampled-frame indices, sorted, length trong [min_output, max_output]
    """
    features = np.array(frame_features, dtype=np.float32)
    N        = len(features)

    if N <= min_output:
        if verbose:
            print(f"[TAC-SUM] {N} frames ≤ min={min_output}, returning all.")
        return list(range(N))

    # 1. K-means → temporal segments
    segments, labels, centers = kmeans_to_temporal_segments(features, n_clusters)

    if verbose:
        sizes = [e - s for s, e, _ in segments]
        print(f"[TAC-SUM] {N} frames | K-means K={n_clusters} "
              f"→ {len(segments)} temporal segments | sizes={sizes[:15]}")

    # 2. Merge segments quá ngắn + clamp
    segments = merge_short_segments(segments, features,
                                    min_len=min_seg_len,
                                    max_output=max_output)

    # 3. Floor: nếu quá ít segment, split segment lớn nhất
    while len(segments) < min_output and any((e - s) >= 2 for s, e, _ in segments):
        sizes_list = [(e - s, i) for i, (s, e, _) in enumerate(segments)]
        _, big_i   = max(sizes_list)
        s, e, cid  = segments[big_i]
        mid        = s + (e - s) // 2
        segments   = (segments[:big_i]
                      + [(s, mid, cid), (mid, e, cid)]
                      + segments[big_i + 1:])

    if verbose:
        sizes = [e - s for s, e, _ in segments]
        print(f"[TAC-SUM] After merge: {len(segments)} segments | sizes={sizes}")

    # 4. Representative frame per temporal segment
    result = pick_segment_representatives(features, segments)

    return sorted(result)