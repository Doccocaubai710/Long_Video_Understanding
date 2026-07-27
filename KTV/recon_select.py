"""
TANG 1 - RECON-Select (query-agnostic, training-free).

Chon tap frame toi thieu tai tao duoc toan bo video, bang greedy QR-pivot
tren feature DINO. Mot dai luong duy nhat (residual) quyet dinh:
  (1) so frame K  -> energy stopping (adaptive theo noi dung)
  (2) frame nao   -> argmax residual
  (3) gia tri v_a -> dung cho token allocation o Tang 2
  (4) do dai clip -> mo rong khi hang xom con residual cao

Chay MOT LAN cho moi video. Ket qua cache lai de Tang 2 dung cho moi cau hoi.
"""

import json
import os

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm

import cluster_and_rank_keyframes as base


# ======================================================================
# 1. PREPROCESS
# ======================================================================
def downsample_indices(n, n_frames_orig, fps, target_fps=1.0, max_keep=1200):
    """Ha mat do frame ve ~target_fps.

    Feature DINO duoc trich rat day (2000-5400 frame/video), khien residual
    bi chi phoi boi nhieu anh sang thay vi su kien that. Ha ve ~1 FPS.
    Tra ve list chi so (trong khong gian feature goc) duoc giu lai.
    """
    if n <= 0:
        return []
    # Uoc luong step: neu biet fps that thi dung, khong thi chia deu
    if fps and fps > 0 and n_frames_orig and n_frames_orig > 0:
        # so feature tuong ung 1 giay
        step = max(1, int(round(n / max(1.0, n_frames_orig / fps) / max(target_fps, 1e-6))))
    else:
        step = 1
    idx = list(range(0, n, max(1, step)))
    if len(idx) > max_keep:  # chan tren de QR khong qua cham
        sel = np.linspace(0, len(idx) - 1, max_keep).astype(int)
        idx = [idx[i] for i in sel]
    return idx


def prepare_matrix(feats, center=True):
    """L2-normalize (va tuy chon tru mean). Greedy QR rat nhay chuan hoa."""
    X = np.asarray(feats, dtype=np.float64)
    if center:
        X = X - X.mean(axis=0, keepdims=True)
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / (nrm + 1e-8)
    return X


# ======================================================================
# 2. GREEDY RECONSTRUCTION (QR-pivot)
# ======================================================================
def recon_greedy(X, gamma=0.90, rho=0.2, k_min=8, k_max=48):
    """Greedy: moi buoc chon frame co residual lon nhat, roi Gram-Schmidt.

    R^(0) = X
    a_t   = argmax_i ||r_i||
    v_at  = ||r_at||
    r_i  <- r_i - <r_i, q> q,   q = r_at / ||r_at||

    Dung khi: max||r|| < tau  HOAC  energy tich luy >= gamma  HOAC |S| >= k_max.

    Returns:
      anchors : list[int]  chi so frame (trong X)
      values  : list[float] v_a = residual ma anchor do giai thich
      R_final : np.ndarray residual con lai (dung cho clip extension)
    """
    n = X.shape[0]
    if n == 0:
        return [], [], np.zeros((0, 0))
    R = X.copy()
    norms0 = np.linalg.norm(X, axis=1)
    tau = rho * float(np.median(norms0)) if norms0.size else 0.0

    anchors, values = [], []
    k_max = int(min(k_max, n))

    for _ in range(k_max):
        rn = np.linalg.norm(R, axis=1)
        a = int(np.argmax(rn))
        v = float(rn[a])
        if v < 1e-10:
            break
        # dieu kien dung 1: residual con lai qua nho
        if len(anchors) >= k_min and v < tau:
            break

        anchors.append(a)
        values.append(v)

        q = R[a] / (v + 1e-12)
        R = R - np.outer(R @ q, q)      # Gram-Schmidt mot buoc
        R[a] = 0.0                       # anchor da duoc giai thich hoan toan

        # dieu kien dung 2: nang luong tich luy >= gamma
        if len(anchors) >= k_min:
            vv = np.asarray(values, dtype=np.float64) ** 2
            total = float(np.sum(np.asarray(values, np.float64) ** 2) + np.sum(np.linalg.norm(R, axis=1) ** 2))
            if total > 0 and float(np.sum(vv)) / total >= gamma:
                break

    return anchors, values, R


def energy_stop_k(values, residual_R, gamma=0.90, k_min=8):
    """K* = min{k : sum_{t<=k} v_t^2 / total_energy >= gamma} (bao cao/kiem tra)."""
    v = np.asarray(values, dtype=np.float64) ** 2
    tail = float(np.sum(np.linalg.norm(residual_R, axis=1) ** 2)) if residual_R.size else 0.0
    total = float(v.sum() + tail)
    if total <= 0:
        return max(k_min, len(values))
    c = np.cumsum(v) / total
    idx = np.searchsorted(c, gamma) + 1
    return int(max(k_min, min(idx, len(values))))


# ======================================================================
# 3. CLIP EXTENSION theo residual
# ======================================================================
def extend_clips(anchors, R_final, values, n, theta=0.4, l_max=4):
    """Mo rong anchor sang hang xom khi hang xom CON residual cao.

    Them j = a +/- 1, +/- 2 ... neu  ||r_j|| > theta * v_a  (vung dong),
    dung khi hang xom da redundant (vung tinh). Gioi han |clip| <= l_max.

    Returns dict[anchor] = sorted list of frame indices (trong khong gian X).
    """
    rn = np.linalg.norm(R_final, axis=1) if R_final.size else np.zeros(n)
    clips, used = {}, set()
    order = sorted(range(len(anchors)), key=lambda t: -values[t])  # anchor manh truoc

    for t in order:
        a = anchors[t]
        cur = [a]
        used.add(a)
        for radius in range(1, l_max):
            if len(cur) >= l_max:
                break
            for j in (a - radius, a + radius):
                if len(cur) >= l_max:
                    break
                if 0 <= j < n and j not in used:
                    if rn[j] > theta * values[t]:   # hang xom con mang thong tin moi
                        cur.append(j)
                        used.add(j)
        clips[a] = sorted(cur)
    return clips


def reconstruction_coverage(X, anchors):
    """RC(S) = 1 - ||X - P_S X||^2 / ||X||^2. Metric danh gia KHONG can VLM."""
    if not anchors:
        return 0.0
    A = X[np.asarray(anchors, dtype=int)]
    Q, _ = np.linalg.qr(A.T)              # co so truc giao cua span(S)
    proj = (X @ Q) @ Q.T
    num = float(np.sum((X - proj) ** 2))
    den = float(np.sum(X ** 2)) + 1e-12
    return float(1.0 - num / den)


# ======================================================================
# 4. CLIP IMAGE EMBEDDING (cache cho Tang 2)
# ======================================================================
def embed_anchor_frames(video_path, orig_frames, device):
    """CLIP image-embed cho cac anchor. Tinh MOT LAN/video, cache lai.

    Tang 2 chi can encode TEXT cau hoi -> rat re.
    """
    base.load_clip_model(device)
    cap = cv2.VideoCapture(video_path)
    imgs, ok_idx = [], []
    for k, fi in enumerate(orig_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, img = cap.read()
        if ok:
            imgs.append(base.preprocess_clip(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))))
            ok_idx.append(k)
    cap.release()

    dim = 768
    out = np.zeros((len(orig_frames), dim), dtype=np.float32)
    if not imgs:
        return out
    with torch.no_grad():
        batch = torch.stack(imgs).to(base.device)
        f = base.model_clip.encode_image(batch)
        f = f / f.norm(dim=-1, keepdim=True)
        f = f.float().cpu().numpy()
    if f.shape[1] != dim:
        out = np.zeros((len(orig_frames), f.shape[1]), dtype=np.float32)
    for k, row in zip(ok_idx, f):
        out[k] = row
    return out


# ======================================================================
# 5. PIPELINE
# ======================================================================
def run(json_path, video_path, video_frame_tensor_path, stage1_dir, dataset,
        gamma=0.90, rho=0.2, k_min=8, k_max=48, theta=0.4, l_max=4,
        target_fps=1.0, max_keep=1200, center=True,
        max_frames_to_extract=5400, device="auto", compute_clip=True,
        stats_path=None):
    os.makedirs(stage1_dir, exist_ok=True)
    vft = base.load_video_frame_tensor(video_frame_tensor_path)
    qa = json.load(open(json_path))

    # danh sach video duy nhat
    seen, videos = set(), []
    for s in qa:
        if s["video_name"] not in seen:
            seen.add(s["video_name"])
            videos.append(s["video_name"])

    stats, missing = [], 0
    for vname in tqdm(videos, total=len(videos), desc="Stage1 RECON"):
        out_json = os.path.join(stage1_dir, f"{vname}.json")
        if os.path.exists(out_json):
            continue

        tensor = base.get_tensor_for_video(vft, vname)
        if tensor is None:
            missing += 1
            continue
        feats_full = np.asarray(tensor, dtype=np.float32)

        cap = cv2.VideoCapture(os.path.join(video_path, vname))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        n_full = feats_full.shape[0]
        if total_frames and total_frames > 0:
            n_full = min(n_full, total_frames, max_frames_to_extract)
            feats_full = feats_full[:n_full]
        if n_full < 4:
            continue

        # (a) ha ve ~1 FPS
        keep = downsample_indices(n_full, total_frames, fps, target_fps, max_keep)
        if len(keep) < 4:
            keep = list(range(n_full))
        feats = feats_full[np.asarray(keep, dtype=int)]

        # (b) greedy reconstruction
        X = prepare_matrix(feats, center=center)
        anchors, values, R = recon_greedy(X, gamma=gamma, rho=rho, k_min=k_min, k_max=k_max)
        if not anchors:
            continue
        K_star = energy_stop_k(values, R, gamma=gamma, k_min=k_min)
        rc = reconstruction_coverage(X, anchors)

        # (c) mo rong clip theo residual
        clips = extend_clips(anchors, R, values, X.shape[0], theta=theta, l_max=l_max)

        # (d) map ve chi so frame GOC cua video
        num_ext = min(total_frames, max_frames_to_extract) if (total_frames and total_frames > 0) else n_full

        def to_orig(i_ds):
            i_full = keep[i_ds] if i_ds < len(keep) else i_ds
            if i_full >= num_ext:
                return None
            try:
                return int(base.get_original_frame_number(
                    total_frames if (total_frames and total_frames > 0) else num_ext,
                    i_full, fps=fps, max_frames_to_extract=max_frames_to_extract))
            except ValueError:
                return None

        rec_clips, rec_values, anchor_orig = [], [], []
        for t, a in enumerate(anchors):
            frames = [to_orig(j) for j in clips.get(a, [a])]
            frames = sorted({f for f in frames if f is not None})
            ao = to_orig(a)
            if not frames or ao is None:
                continue
            rec_clips.append(frames)
            rec_values.append(float(values[t]))
            anchor_orig.append(ao)
        if not rec_clips:
            continue

        # (e) CLIP image embedding cua anchor (cache cho Tang 2)
        if compute_clip:
            emb = embed_anchor_frames(os.path.join(video_path, vname), anchor_orig, device)
            np.savez_compressed(os.path.join(stage1_dir, f"{vname}.npz"), emb=emb)

        json.dump({
            "video_name": vname,
            "K": len(rec_clips),
            "K_star_energy": int(K_star),
            "recon_coverage": float(rc),
            "n_feat_used": int(X.shape[0]),
            "anchors_orig": anchor_orig,
            "values": rec_values,
            "clips_orig": rec_clips,
        }, open(out_json, "w"), indent=2)

        stats.append({"video_name": vname, "K": len(rec_clips),
                      "K_star": int(K_star), "RC": float(rc)})

    if stats_path and stats:
        os.makedirs(os.path.dirname(stats_path) or ".", exist_ok=True)
        K = np.array([s["K"] for s in stats])
        rc = np.array([s["RC"] for s in stats])
        with open(stats_path, "w") as f:
            f.write(f"videos={len(stats)} missing={missing}\n")
            f.write(f"K   mean/med/min/max: {K.mean():.1f}/{np.median(K):.0f}/{K.min()}/{K.max()}\n")
            f.write(f"RC  mean/med/min/max: {rc.mean():.3f}/{np.median(rc):.3f}/{rc.min():.3f}/{rc.max():.3f}\n")
        print(open(stats_path).read())

    print(f"[Stage1] done. videos={len(stats)} missing_tensor={missing} -> {stage1_dir}")


@hydra.main(config_path="configs/keyframe_cluster", config_name="config", version_base=None)
def main(cfg: DictConfig):
    g = lambda k, d: getattr(cfg, k, d)
    run(base.resolve_path(cfg.json_path),
        base.resolve_path(cfg.video_path),
        base.resolve_path(cfg.video_frame_tensor_path),
        base.resolve_path(g("stage1_dir", "outputs/casqta_stage1")),
        cfg.dataset,
        gamma=g("gamma", 0.90), rho=g("rho", 0.2),
        k_min=g("k_min", 8), k_max=g("k_max", 48),
        theta=g("theta", 0.4), l_max=g("l_max", 4),
        target_fps=g("target_fps", 1.0), max_keep=g("max_keep", 1200),
        center=bool(g("center", True)),
        max_frames_to_extract=g("max_frames_to_extract", 5400),
        device=g("device", "auto"),
        compute_clip=bool(g("compute_clip", True)),
        stats_path=base.resolve_path(g("stage1_stats", "outputs/casqta_stage1_stats.txt")))


if __name__ == "__main__":
    main()