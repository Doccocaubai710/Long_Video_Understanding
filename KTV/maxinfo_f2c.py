import json
import os

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig
from scipy.linalg import qr
from tqdm import tqdm

import cluster_and_rank_keyframes as base
from temporal_chain_f2c_flex import (
    compute_event_score, build_clips, clips_to_order, clip_relevance, build_prompt,
)


# ---- MaxInfo: SVD + (QR-pivot xap xi rect_maxvol) chon dinh adaptive ----
def maxinfo_peaks(feats, svd_rank=32, tol_ratio=0.05, max_k=64, min_k=4):
    """Tra ve chi so cac frame 'dinh' (informative) voi SO LUONG ADAPTIVE.
    Xap xi rect_maxvol bang QR co pivot tren Qs^T; dung khi tri rieng giam < tol."""
    feats = np.asarray(feats, dtype=np.float32)
    n = feats.shape[0]
    if n <= min_k:
        return list(range(n))
    s = min(svd_rank, n, feats.shape[1])
    U, S, _ = np.linalg.svd(feats, full_matrices=False)
    Qs = U[:, :s]                       # (n, s) — bien thien chinh giua cac frame
    _, R, piv = qr(Qs.T, pivoting=True)  # pivot xep frame theo do doc lap tuyen tinh
    diag = np.abs(np.diag(R))
    if diag.size == 0:
        return [0]
    r = int(np.sum(diag >= diag[0] * tol_ratio))
    r = max(min_k, min(r, max_k, n))
    return sorted(int(p) for p in piv[:r])


def target_k_from_peaks(num_peaks):
    """So dinh goc -> so cluster muc tieu 8/16/32 (theo yeu cau)."""
    if num_peaks <= 12:
        return 8
    elif num_peaks <= 24:
        return 16
    return 32


def merge_peaks_to_k(peaks, feats, k):
    """Neu qua nhieu dinh -> gop ve k cluster bang k-means tren feature cua dinh,
    moi cluster lay dinh gan tam nhat lam anchor."""
    peaks = list(peaks)
    if len(peaks) <= k:
        return peaks
    P = feats[np.array(peaks, dtype=np.int32)]
    km = base.KMeans(n_clusters=k, random_state=0, init="k-means++", n_init=10).fit(P)
    anchors = []
    for c in range(k):
        idx = np.where(km.labels_ == c)[0]
        if len(idx) == 0:
            continue
        dists = np.linalg.norm(P[idx] - km.cluster_centers_[c], axis=1)
        anchors.append(peaks[idx[int(np.argmin(dists))]])
    return sorted(set(anchors))


def cluster(json_path, video_path, video_frame_tensor_path, save_cluster_path, dataset,
            combined_output_path=None, fixed_clip_len=4, query_aware=False,
            svd_rank=32, tol_ratio=0.05, max_peaks=64, l_max=4, lambda_r=0.5, lambda_l=0.05,
            tokens_num=1872, min_tokens_per_frame=16, max_frames_to_extract=5400, device="auto"):
    os.makedirs(save_cluster_path, exist_ok=True)
    vft = base.load_video_frame_tensor(video_frame_tensor_path)
    qa = json.load(open(json_path))
    combined, missing = {}, 0
    stage1 = {}

    for sample in tqdm(qa, total=len(qa)):
        vname, qid = sample["video_name"], sample["question_id"]
        out_path = os.path.join(save_cluster_path, f"{qid}.json")
        if os.path.exists(out_path):
            combined.update(json.load(open(out_path)))
            continue

        if vname in stage1:
            anchors, r_event, n, total_frames, fps, feats = stage1[vname]
        else:
            tensor = base.get_tensor_for_video(vft, vname)
            if tensor is None:
                missing += 1
                continue
            tensor = np.asarray(tensor, dtype=np.float32)
            cap = cv2.VideoCapture(os.path.join(video_path, vname))
            fps = cap.get(cv2.CAP_PROP_FPS); total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
            n = tensor.shape[0]
            if total_frames and total_frames > 0:
                n = min(n, total_frames, max_frames_to_extract); tensor = tensor[:n]
            # --- MaxInfo: chon dinh adaptive, roi gop ve K=8/16/32 ---
            peaks = maxinfo_peaks(tensor, svd_rank=svd_rank, tol_ratio=tol_ratio, max_k=max_peaks)
            K = target_k_from_peaks(len(peaks))
            anchors = merge_peaks_to_k(peaks, tensor, K)
            r_event = compute_event_score(tensor, max(1, len(anchors)))
            feats = tensor
            stage1[vname] = (anchors, r_event, n, total_frames, fps, feats)

        clip_frames = build_clips(anchors, n, fixed_clip_len, r_event,
                                  feats=feats, l_max=l_max, lambda_r=lambda_r, lambda_l=lambda_l)

        if query_aware:
            anchor_orig = {}
            for a in clip_frames:
                try:
                    anchor_orig[a] = base.get_original_frame_number(
                        total_frames if total_frames and total_frames > 0 else n,
                        a, fps=fps, max_frames_to_extract=max_frames_to_extract)
                except ValueError:
                    anchor_orig[a] = 0
            order = list(clip_frames.keys())
            rel = clip_relevance([anchor_orig[a] for a in order],
                                 os.path.join(video_path, vname), build_prompt(sample), device)
            anchor_score = {a: rel[i] for i, a in enumerate(order)}
            if float(np.sum(list(anchor_score.values()))) == 0.0:
                anchor_score = {a: float(r_event[a]) for a in clip_frames}
        else:
            anchor_score = {a: float(r_event[a]) for a in clip_frames}

        key_frame_order = clips_to_order(
            clip_frames, anchor_score, len(clip_frames), tokens_num,
            total_frames, fps, n, max_frames_to_extract, min_tokens_per_frame)
        if not key_frame_order:
            uni = [int(x) for x in np.linspace(0, float(total_frames or n), 12, endpoint=False)]
            key_frame_order = [[fi, max(min_tokens_per_frame, tokens_num // 12)] for fi in uni]

        result = {qid: key_frame_order}
        combined.update(result)
        json.dump(result, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=4,
                  default=lambda o: int(o) if isinstance(o, np.integer) else o)

    if combined_output_path:
        d = os.path.dirname(combined_output_path)
        if d:
            os.makedirs(d, exist_ok=True)
        json.dump(combined, open(combined_output_path, "w", encoding="utf-8"), ensure_ascii=False,
                  indent=4, default=lambda o: int(o) if isinstance(o, np.integer) else o)
    print(f"Done MaxInfo+F2C. clip_len={fixed_clip_len} query_aware={query_aware} "
          f"missing={missing} total={len(qa)}")


@hydra.main(config_path="configs/keyframe_cluster", config_name="config", version_base=None)
def main(cfg: DictConfig):
    g = lambda k, d: getattr(cfg, k, d)
    cluster(base.resolve_path(cfg.json_path), base.resolve_path(cfg.video_path),
            base.resolve_path(cfg.video_frame_tensor_path), base.resolve_path(cfg.save_cluster_path),
            cfg.dataset, combined_output_path=base.resolve_path(cfg.combined_output_path),
            fixed_clip_len=g("fixed_clip_len", 4), query_aware=g("query_aware", False),
            svd_rank=g("svd_rank", 32), tol_ratio=g("tol_ratio", 0.05), max_peaks=g("max_peaks", 64),
            l_max=g("l_max", 4), lambda_r=g("lambda_r", 0.5), lambda_l=g("lambda_l", 0.05),
            tokens_num=g("tokens_num", 1872), min_tokens_per_frame=g("min_tokens_per_frame", 16),
            max_frames_to_extract=g("max_frames_to_extract", 5400), device=g("device", "auto"))


if __name__ == "__main__":
    main()