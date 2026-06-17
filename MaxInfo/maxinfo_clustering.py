# maxinfo_clustering.py
"""
MaxInfo — "MaxInfo: A Training-Free Key-Frame Selection Method
Using Maximum Volume for Enhanced Video Understanding" (2502.03183)

Adapted cho KTV pipeline:
  - Paper gốc dùng CLIP [CLS] features
  - Ở đây dùng DINO features đã extract sẵn (không cần chạy thêm encoder)
  - Thuật toán SVD + Rectangular MaxVol giữ nguyên theo paper

Drop-in replacement cho video_frame_clustering() trong cluster_and_rank_keyframes.py.

Algorithm (từ paper, Algorithm 1):
  1. Q = stack DINO features → (n, d)
  2. Truncated SVD: Qs = U[:, :s]  (n, s)
  3. rect_maxvol(Qs) → pivot indices r  (frames maximize geometric volume)
  4. Return r sorted chronologically
"""

import numpy as np
from scipy.linalg import qr


# ─── Rectangular MaxVol ──────────────────────────────────────────────────────

def rect_maxvol(A: np.ndarray,
                tol: float = 1.05,
                maxK: int = None,
                minK: int = None) -> list:
    """
    Rectangular MaxVol algorithm (Goreinov et al.)
    Chọn tối đa maxK rows của A sao cho rect-vol = sqrt(det(A[rows] A[rows]^T))
    là lớn nhất có thể, dừng khi volume gain < tol hoặc đạt maxK.

    Args:
        A    : (n, r) matrix — reduced feature matrix sau SVD
        tol  : minimum volume gain để tiếp tục thêm row (1.05 theo paper)
        maxK : maximum số rows chọn (= num_keyframes)
        minK : minimum số rows chọn (= min_segments)

    Returns:
        list of int — selected row indices, sorted
    """
    n, r = A.shape
    if maxK is None: maxK = n
    if minK is None: minK = min(r, n)
    maxK = min(maxK, n)
    minK = max(minK, 1)

    # ── Khởi tạo: chọn r rows đầu bằng QR pivoting ──────────────────────────
    # QR pivoting cho pivot indices tốt hơn random init
    try:
        _, _, perm = qr(A.T, pivoting=True)
        piv = list(perm[:min(r, n)])
    except Exception:
        piv = list(range(min(r, n)))

    piv_set = set(piv)
    remaining = [i for i in range(n) if i not in piv_set]

    # ── Iterative expansion: thêm row tăng volume nhiều nhất ─────────────────
    # Volume gain khi thêm row i: sqrt(1 + ||C_i||^2)
    # với C_i = A[i] @ pinv(A[piv]) (coefficient của row i trong basis piv)
    for _ in range(maxK - len(piv)):
        if not remaining:
            break

        AK = A[piv]                          # (len(piv), r)
        try:
            AK_pinv = np.linalg.pinv(AK)    # (r, len(piv))
        except np.linalg.LinAlgError:
            break

        # C = A[remaining] @ AK_pinv.T  → (len(remaining), len(piv))
        A_rem    = A[remaining]              # (len(remaining), r)
        C        = A_rem @ AK_pinv.T        # (len(remaining), len(piv))

        # Volume gain per candidate row
        vol_gains = np.sqrt(1.0 + np.sum(C ** 2, axis=1))  # (len(remaining),)

        best_local = int(np.argmax(vol_gains))
        best_gain  = float(vol_gains[best_local])

        # Dừng nếu gain quá nhỏ và đã đủ minK
        if best_gain < tol and len(piv) >= minK:
            break

        best_global = remaining[best_local]
        piv.append(best_global)
        piv_set.add(best_global)
        remaining.pop(best_local)

        if len(piv) >= maxK:
            break

    # Floor: nếu ít hơn minK, thêm từ remaining theo thứ tự temporal
    while len(piv) < minK and remaining:
        piv.append(remaining.pop(0))

    return sorted(piv)


# ─── Main Algorithm (Algorithm 1 từ paper) ───────────────────────────────────

def maxinfo_clustering(frame_features,
                       num_keyframes: int = 12,
                       svd_components: int = None,
                       tol: float = 1.05,
                       min_segments: int = 6,
                       verbose: bool = True) -> list:
    """
    MaxInfo keyframe selection — drop-in replacement cho video_frame_clustering().

    Theo Algorithm 1 trong paper (2502.03183):
      Step 1: Q = stack [CLS] features → (n, d)
      Step 2: Truncated SVD → Qs = U[:, :s]  (n, s)
      Step 3: rect_maxvol(Qs, tol, maxK) → pivot indices
      Step 4: return sorted indices

    Args:
        frame_features  : list hoặc np.ndarray (N, D) — DINO features per frame
        num_keyframes   : số frame muốn chọn (= K trong KTV, upper bound)
        svd_components  : số singular vectors giữ lại (None = auto = num_keyframes)
        tol             : volume gain threshold để tiếp tục thêm frame (1.05 theo paper)
        min_segments    : floor tối thiểu
        verbose         : in thông tin

    Returns:
        List[int] — selected sampled-frame indices, sorted chronologically
    """
    # Step 1: Build feature matrix Q
    Q = np.array(frame_features, dtype=np.float64)  # (n, d)
    n, d = Q.shape

    if n <= min_segments:
        if verbose:
            print(f"[MaxInfo] {n} frames ≤ min={min_segments}, returning all.")
        return list(range(n))

    # Step 2: Truncated SVD → Qs
    # s = svd_components nếu được chỉ định, còn lại = num_keyframes
    s = svd_components if svd_components is not None else num_keyframes
    s = min(s, n, d)

    try:
        U, Sigma, Vt = np.linalg.svd(Q, full_matrices=False)
        Qs = U[:, :s]   # (n, s) — top s left singular vectors
    except np.linalg.LinAlgError:
        # Fallback: dùng features gốc nếu SVD lỗi
        if verbose:
            print("[MaxInfo] SVD failed, using raw features.")
        Qs = Q[:, :min(s, d)]

    if verbose:
        print(f"[MaxInfo] {n} frames | SVD ({n}×{d}) → Qs ({n}×{s}) "
              f"| maxK={num_keyframes} minK={min_segments} tol={tol}")

    # Step 3: Rectangular MaxVol → pivot indices
    pivots = rect_maxvol(
        Qs,
        tol=tol,
        maxK=num_keyframes,
        minK=min_segments,
    )

    if verbose:
        print(f"[MaxInfo] Selected K={len(pivots)} frames: {pivots}")

    # Step 4: Return sorted indices
    return sorted(pivots)