import json
import os

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

import cluster_and_rank_keyframes as base


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def minmax_normalize(values, eps=1e-8):
    values = np.asarray(values, dtype=np.float32)
    vmin, vmax = float(np.min(values)), float(np.max(values))
    if vmax - vmin < eps:
        return np.zeros_like(values, dtype=np.float32)
    return (values - vmin) / (vmax - vmin + eps)


def compute_event_score(frame_features, num_keyframes):
    """Query-agnostic eventness: R_event(t)=||f_t-f_{t-d}||+||f_{t+d}-f_t||.

    Stands in for F2C's query relevancy r(f_i)=cos(E(f_i),E(Q)). High = frame
    sits near a scene/action transition.
    """
    n = frame_features.shape[0]
    delta = max(1, n // max(1, (10 * num_keyframes)))
    score = np.zeros(n, dtype=np.float32)
    for t in range(n):
        left = max(0, t - delta)
        right = min(n - 1, t + delta)
        score[t] = np.linalg.norm(frame_features[t] - frame_features[left]) + np.linalg.norm(
            frame_features[right] - frame_features[t]
        )
    return score


# ---------------------------------------------------------------------------
# Stage 1: temporal-chain anchor (clip center) selection -- query-agnostic
# ---------------------------------------------------------------------------
def temporal_chain_select(
    frame_features, num_clips=12, lambda_event=0.5, alpha_gap=0.6, beta_redundancy=0.8
):
    """Select K_anchor = num_clips anchors -> clip centers.

    Greedy: G(t|S)=R_cluster~ + lambda*R_event~ + alpha*T_gap~ - beta*R_red~.
    With num_clips = target K, K_anchor/K = 1.0 (F2C best setting).
    Returns (selected_anchor_indices, r_event[N]).
    """
    frame_features = np.asarray(frame_features, dtype=np.float32)
    n = frame_features.shape[0]
    if n == 0:
        return [], np.zeros(0, dtype=np.float32)
    if n <= num_clips:
        return list(range(n)), compute_event_score(frame_features, max(1, n))

    k = min(num_clips, n)
    kmeans = base.KMeans(n_clusters=k, random_state=0, init="k-means++", n_init=10).fit(
        frame_features
    )
    r_cluster = -np.linalg.norm(frame_features - kmeans.cluster_centers_[kmeans.labels_], axis=1)
    r_event = compute_event_score(frame_features, k)
    norm = frame_features / (np.linalg.norm(frame_features, axis=1, keepdims=True) + 1e-8)

    selected, remaining = [], set(range(n))
    first = int(np.argmax(minmax_normalize(r_cluster) + lambda_event * minmax_normalize(r_event)))
    selected.append(first)
    remaining.remove(first)

    while len(selected) < k and remaining:
        cand = np.array(sorted(remaining), dtype=np.int32)
        c_cluster = minmax_normalize(r_cluster[cand])
        c_event = minmax_normalize(r_event[cand])
        c_gap = minmax_normalize(
            np.array([min(abs(int(t) - int(s)) for s in selected) for t in cand], dtype=np.float32)
        )
        sims = norm[cand] @ norm[np.array(selected, dtype=np.int32)].T
        c_red = minmax_normalize(np.max(sims, axis=1))
        scores = c_cluster + lambda_event * c_event + alpha_gap * c_gap - beta_redundancy * c_red
        best = int(cand[int(np.argmax(scores))])
        selected.append(best)
        remaining.remove(best)
    return selected, r_event


# ---------------------------------------------------------------------------
# Stage 2: F2C adaptive clip-length expansion (query-agnostic Eq.8)
# ---------------------------------------------------------------------------
def f2c_centered_window(center, length, n):
    """F2C window centered on anchor; even lengths biased right (l=4 -> k-1..k+2).
    Shifted inside [0,n-1] near edges so the clip keeps `length` frames."""
    if n <= 0:
        return []
    length = min(length, n)
    start = max(0, min(center - (length - 1) // 2, n - length))
    return [start + i for i in range(length)]


def adaptive_clip_length(center, features, r_event_norm, n, l_max, lambda_r=0.5, lambda_l=0.05):
    """l* = argmax_l [ S_C(l) - lambda_r*R_C(l) + lambda_l*(l/l_max) ], l in [1,l_max].
    S_C = mean eventness over clip (agnostic relevancy); R_C = mean pairwise cosine
    (redundancy); + temporal reward favoring longer clips. s_max=2 -> l_max=4."""
    norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    best_l, best_score = 1, -1e30
    for l in range(1, l_max + 1):
        win = f2c_centered_window(center, l, n)
        if len(win) < l:
            break
        s_rel = float(np.mean(r_event_norm[win]))
        if l > 1:
            sub = norm[win]
            sim = sub @ sub.T
            np.fill_diagonal(sim, 0.0)
            r_red = float(sim.sum() / (l * (l - 1)))
        else:
            r_red = 0.0
        score = s_rel - lambda_r * r_red + lambda_l * (l / l_max)
        if score > best_score:
            best_score, best_l = score, l
    return best_l


def clip_budget_tiers(num_clips, tokens_num):
    """Per-CLIP total token budgets mirroring the 2/8/2 -> 288/144/72 structure,
    scaled to tokens_num. Sum ~= tokens_num. Returned descending."""
    scale = tokens_num / 1872.0
    n_high = max(1, round(num_clips / 6))
    n_low = max(1, round(num_clips / 6))
    n_mid = max(0, num_clips - n_high - n_low)
    return (
        [int(round(288 * scale))] * n_high
        + [int(round(144 * scale))] * n_mid
        + [int(round(72 * scale))] * n_low
    )


def build_clips(anchors, features, r_event, n, l_max, lambda_r, lambda_l):
    """Expand each anchor into an adaptive-length clip, merge overlaps by nearest
    anchor. Returns clip_frames: dict[anchor -> sorted list of feature-frame idx]."""
    r_norm = minmax_normalize(r_event)
    windows = {a: set(f2c_centered_window(a, adaptive_clip_length(a, features, r_norm, n, l_max, lambda_r, lambda_l), n)) for a in anchors}
    all_frames = set().union(*windows.values()) if windows else set()

    clip_frames = {a: [] for a in anchors}
    for f in all_frames:
        owners = [a for a in anchors if f in windows[a]]
        owner = min(owners, key=lambda a: (abs(f - a), -float(r_event[a])))
        clip_frames[owner].append(f)
    return {a: sorted(fs) for a, fs in clip_frames.items() if fs}


# ---------------------------------------------------------------------------
# End-to-end pipeline (query-agnostic, no CLIP)
# ---------------------------------------------------------------------------
def cluster(
    json_path,
    video_path,
    video_frame_tensor_path,
    save_cluster_path,
    dataset,
    combined_output_path=None,
    num_clips=12,
    l_max=4,
    tokens_num=1872,
    lambda_event=0.5,
    alpha_gap=0.6,
    beta_redundancy=0.8,
    lambda_r=0.5,
    lambda_l=0.05,
    min_tokens_per_frame=16,
    max_frames_to_extract=5400,
):
    """
    F2C-faithful, query-agnostic key-CLIP selection.

    Output per question: {qid: [[frame_index, per_frame_token_budget], ...]}.
    - 12 clips (K_anchor=K -> ratio 1.0), each adaptively up to l_max=4 frames.
    - Each CLIP gets a total token budget (288/144/72 tier); each frame in the
      clip receives clip_budget // clip_len tokens => total tokens preserved.
    - The second field is an EXPLICIT per-frame token budget (>= min_tokens_per_frame,
      always > 11), so llava_arch can use it directly while legacy rank files
      (values 0..11) keep using the old branch.
    """
    os.makedirs(save_cluster_path, exist_ok=True)
    video_frame_tensor = base.load_video_frame_tensor(video_frame_tensor_path)
    with open(json_path, "r") as f:
        qa_data = json.load(f)

    combined_results = {}
    missing = 0
    budgets_desc = clip_budget_tiers(num_clips, tokens_num)

    for sample in tqdm(qa_data, total=len(qa_data)):
        video_name = sample["video_name"]
        question_id = sample["question_id"]
        output_path = os.path.join(save_cluster_path, f"{question_id}.json")
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                combined_results.update(json.load(f))
            continue

        full_video_path = os.path.join(video_path, video_name)
        cap = cv2.VideoCapture(full_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        tensor = base.get_tensor_for_video(video_frame_tensor, video_name)
        if tensor is None:
            missing += 1
            continue
        tensor = np.asarray(tensor, dtype=np.float32)
        n = tensor.shape[0]

        # Stage 1: anchors (clip centers).
        anchors, r_event = temporal_chain_select(
            tensor, num_clips=num_clips, lambda_event=lambda_event,
            alpha_gap=alpha_gap, beta_redundancy=beta_redundancy,
        )

        # Stage 2: adaptive clip expansion + merge.
        clip_frames = build_clips(anchors, tensor, r_event, n, l_max, lambda_r, lambda_l)

        # Assign per-CLIP budget by importance (anchor eventness), then split per frame.
        clips_sorted = sorted(clip_frames.items(), key=lambda kv: -float(r_event[kv[0]]))
        frame_budget = {}  # feature-frame idx -> per-frame token budget
        for i, (anchor, frames) in enumerate(clips_sorted):
            clip_budget = budgets_desc[min(i, len(budgets_desc) - 1)]
            per_frame = max(min_tokens_per_frame, clip_budget // max(1, len(frames)))
            for fidx in frames:
                # If a frame is shared, keep the higher (less-pruned) budget.
                frame_budget[fidx] = max(frame_budget.get(fidx, 0), per_frame)

        if total_frames and total_frames > 0:
            num_extracted = min(total_frames, max_frames_to_extract)
        else:
            num_extracted = n  # fallback khi cv2 không đọc được frame count

        # Chỉ giữ feature-index nằm trong vùng hợp lệ, rồi map an toàn.
        pairs = {}
        for fidx in sorted(frame_budget.keys()):
            if fidx >= num_extracted:
                continue  # bỏ frame bung ra ngoài biên trích xuất
            try:
                oidx = base.get_original_frame_number(
                    total_frames if total_frames and total_frames > 0 else num_extracted,
                    fidx,
                    fps=fps,
                    max_frames_to_extract=max_frames_to_extract,
                )
            except ValueError:
                continue  # an toàn tuyệt đối: bỏ qua index lỗi
            pairs[int(oidx)] = max(pairs.get(int(oidx), 0), int(frame_budget[fidx]))

        if not pairs:
            # Fallback: uniform 12 frames, evenly split budget.
            uni = [int(x) for x in np.linspace(0, float(total_frames), 12, endpoint=False)]
            per = max(min_tokens_per_frame, tokens_num // 12)
            key_frame_order = [[fi, per] for fi in uni]
        else:
            key_frame_order = [[fi, bud] for fi, bud in sorted(pairs.items())]

        result = {question_id: key_frame_order}
        combined_results.update(result)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4,
                      default=lambda o: int(o) if isinstance(o, np.integer) else o)

    if combined_output_path:
        d = os.path.dirname(combined_output_path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(combined_output_path, "w", encoding="utf-8") as f:
            json.dump(combined_results, f, ensure_ascii=False, indent=4,
                      default=lambda o: int(o) if isinstance(o, np.integer) else o)

    print(f"Finished F2C clip selection. missing_tensor={missing} total={len(qa_data)}")


@hydra.main(config_path="configs/keyframe_cluster", config_name="config", version_base=None)
def main(cfg: DictConfig):
    cluster(
        base.resolve_path(cfg.json_path),
        base.resolve_path(cfg.video_path),
        base.resolve_path(cfg.video_frame_tensor_path),
        base.resolve_path(cfg.save_cluster_path),
        cfg.dataset,
        combined_output_path=base.resolve_path(cfg.combined_output_path),
        num_clips=getattr(cfg, "num_clips", 12),
        l_max=getattr(cfg, "l_max", 4),
        tokens_num=getattr(cfg, "tokens_num", 1872),
        lambda_event=getattr(cfg, "lambda_event", 0.5),
        alpha_gap=getattr(cfg, "alpha_gap", 0.6),
        beta_redundancy=getattr(cfg, "beta_redundancy", 0.8),
        lambda_r=getattr(cfg, "lambda_r", 0.5),
        lambda_l=getattr(cfg, "lambda_l", 0.05),
        min_tokens_per_frame=getattr(cfg, "min_tokens_per_frame", 16),
        max_frames_to_extract=getattr(cfg, "max_frames_to_extract", 5400),
    )


if __name__ == "__main__":
    main()