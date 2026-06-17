import json
import os

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

import cluster_and_rank_keyframes as base


def minmax_normalize(values, eps=1e-8):
    """
    Normalize a 1D score vector into [0, 1] using min-max normalization.

    Why we need this:
    - The temporal-chain objective combines multiple terms (cluster score,
      event score, temporal gap, redundancy).
    - These terms naturally live on different numeric scales.
    - Min-max normalization makes each term comparable before weighted sum.

    Args:
    - values: Array-like scores for frames.
    - eps: Small constant to avoid division-by-zero.

    Returns:
    - np.ndarray with the same shape as input, scaled to [0, 1].
      If all values are nearly identical, returns all zeros.
    """
    values = np.asarray(values, dtype=np.float32)
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmax - vmin < eps:
        return np.zeros_like(values, dtype=np.float32)
    return (values - vmin) / (vmax - vmin + eps)


def compute_event_score(frame_features, num_keyframes):
    """
    Compute query-agnostic event/transition score per frame.

    Formula:
      R_event(t) = ||f_t - f_{t-delta}||_2 + ||f_{t+delta} - f_t||_2

    Intuition:
    - Frames around scene/action changes differ more from neighbors.
    - Higher score means more likely near a temporal transition.

    Delta rule (adaptive window):
    - delta = max(1, floor(N / (10K)))
    - N: number of sampled feature frames for the video
    - K: target keyframe count

    Args:
    - frame_features: np.ndarray [N, D], DINO features.
    - num_keyframes: K in the formula above.

    Returns:
    - np.ndarray [N], transition score per frame.
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


def temporal_chain_select(
    frame_features,
    num_keyframes=15,
    lambda_event=0.5,
    alpha_gap=0.6,
    beta_redundancy=0.8,
):
    """
    Select K anchor keyframes using greedy temporal-chain scoring.

    This is Stage 1 (anchor generation) and is query-agnostic.
    It replaces "pick nearest frame to each centroid independently" with
    sequential selection that jointly optimizes:
    - Representativeness (cluster-center closeness)
    - Eventness (local temporal change)
    - Temporal coverage (gap from already selected frames)
    - Redundancy suppression (cosine similarity to selected frames)

    Greedy objective at each step:
      G(t|S) = R_cluster~ + lambda*R_event~ + alpha*T_gap~ - beta*R_red~

    Args:
    - frame_features: np.ndarray [N, D], DINO features.
    - num_keyframes: number of anchor frames to return (Kanchor).
    - lambda_event: weight for transition/event term.
    - alpha_gap: weight for temporal coverage term.
    - beta_redundancy: weight for redundancy penalty.

    Returns:
    - selected: list[int] anchor indices in feature-frame index space.
    - r_event: np.ndarray [N], event score per feature frame (reused by F2C).
    """
    frame_features = np.asarray(frame_features, dtype=np.float32)
    n = frame_features.shape[0]
    if n == 0:
        return [], np.zeros(0, dtype=np.float32)
    if n <= num_keyframes:
        # If there are too few frames, keep all of them.
        r_event = compute_event_score(frame_features, max(1, n))
        return list(range(n)), r_event

    k = min(num_keyframes, n)
    # K-means provides global visual grouping; we still use it, but only as
    # one score term, not as final one-shot frame selection.
    kmeans = base.KMeans(
        n_clusters=k, random_state=0, init="k-means++", n_init=10
    ).fit(frame_features)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    # Representativeness: closer to own centroid => larger score.
    r_cluster = -np.linalg.norm(frame_features - centers[labels], axis=1)
    # Eventness: larger local change => larger score.
    r_event = compute_event_score(frame_features, k)

    # Pre-normalize features for cosine similarity in redundancy term.
    norm_features = frame_features / (
        np.linalg.norm(frame_features, axis=1, keepdims=True) + 1e-8
    )

    selected = []
    remaining = set(range(n))

    # First frame bootstraps from representativeness + eventness only,
    # because gap and redundancy are undefined when S is empty.
    base_first = minmax_normalize(r_cluster) + lambda_event * minmax_normalize(r_event)
    first = int(np.argmax(base_first))
    selected.append(first)
    remaining.remove(first)

    while len(selected) < k and remaining:
        candidates = np.array(sorted(remaining), dtype=np.int32)

        c_cluster = minmax_normalize(r_cluster[candidates])
        c_event = minmax_normalize(r_event[candidates])

        # Encourage timeline coverage by preferring frames farther from selected ones.
        c_gap_raw = np.array(
            [min(abs(int(t) - int(s)) for s in selected) for t in candidates],
            dtype=np.float32,
        )
        c_gap = minmax_normalize(c_gap_raw)

        # Redundancy term: max cosine similarity to any already selected frame.
        # High similarity means likely duplicate information -> penalize.
        selected_feats = norm_features[np.array(selected, dtype=np.int32)]
        sims = norm_features[candidates] @ selected_feats.T
        c_red_raw = np.max(sims, axis=1)
        c_red = minmax_normalize(c_red_raw)

        # Greedy temporal-chain score for current step.
        scores = c_cluster + lambda_event * c_event + alpha_gap * c_gap - beta_redundancy * c_red
        best = int(candidates[int(np.argmax(scores))])
        selected.append(best)
        remaining.remove(best)

    return selected, r_event


def f2c_expand_clips(selected_indices, r_event, n, target_total=12, smax=2):
    """
    F2C-style adaptive clip length expansion.

    Given Kanchor anchor frames (selected_indices, |selected| <= 15),
    extend high-R_event anchors into short temporal clips (neighbors),
    while low-R_event anchors stay as single frames. The total number of
    frames after expansion is trimmed/padded to target_total so the
    downstream LLaVA pipeline (num_frames=12 token-budget branch) can be
    reused unchanged.

    Why R_event drives clip length:
    - High R_event => frame sits near a scene/action transition => motion
      cues from temporal neighbors are informative => extend into a clip.
    - Low R_event => frame is in a visually static region => a single
      frame already captures the content => keep as length-1 clip.

    Args:
    - selected_indices: list[int], anchor indices in feature-frame space.
    - r_event: np.ndarray [N], event score per feature frame.
    - n: int, total number of feature frames (N).
    - target_total: int, final number of frames after expansion (default 12,
      matches the num_frames=12 branch in llava_arch.py).
    - smax: int, maximum extension radius per anchor (F2C paper default 2).

    Returns:
    - clip_frames: sorted list[int], feature-frame indices, len <= target_total.
    - anchor_set: set[int], original anchor indices (used for ranking).
    """
    anchor_set = set(selected_indices)
    n_anchors = len(selected_indices)

    # Sort anchors by event score (descending): highest-motion anchors get
    # extended first.
    anchors_sorted = sorted(selected_indices, key=lambda t: -r_event[t])

    clip_frames = set(anchor_set)
    extension_budget = max(0, target_total - n_anchors)

    # Extend radius 1 first (immediate neighbors), then radius 2 (smax),
    # always prioritizing the highest-R_event anchors.
    for radius in range(1, smax + 1):
        if extension_budget <= 0:
            break
        for t in anchors_sorted:
            if extension_budget <= 0:
                break
            for offset in (-radius, radius):
                neighbor = t + offset
                if 0 <= neighbor < n and neighbor not in clip_frames:
                    clip_frames.add(neighbor)
                    extension_budget -= 1
                    if extension_budget <= 0:
                        break

    clip_frames = sorted(clip_frames)

    # If Kanchor itself exceeds target_total (e.g. Kanchor=15 > 12), drop the
    # lowest-R_event frames overall so the final set stays within budget.
    if len(clip_frames) > target_total:
        clip_frames = sorted(
            sorted(clip_frames, key=lambda t: -r_event[t])[:target_total]
        )

    return clip_frames, anchor_set


def assign_f2c_ranks(clip_frames, anchor_set, r_event):
    """
    Map clip_frames to ranks in [0..11], matching the num_frames=12 token
    allocation in llava_arch.py:
      rank 0,1   -> 288 tokens (high-resolution: most informative frames)
      rank 2-9   -> 144 tokens (normal)
      rank 10,11 -> 72  tokens (low-resolution: weakest extension frames)

    Anchors with the highest R_event get rank 0/1 (high-res anchors keep
    fine spatial detail). Non-anchor extension frames with the lowest
    R_event get rank 10/11 (low-res, since they mainly add temporal
    continuity rather than new spatial information).

    Args:
    - clip_frames: list[int], feature-frame indices after F2C expansion.
    - anchor_set: set[int], anchor indices.
    - r_event: np.ndarray [N], event score per feature frame.

    Returns:
    - rank_of: dict[int, int], feature-frame index -> rank in [0..11].
    """
    scored = sorted(clip_frames, key=lambda t: -r_event[t])  # high -> low event score

    # Highest-event frames -> rank 0,1 (prefer anchors first).
    high_slots = [t for t in scored if t in anchor_set][:2]
    if len(high_slots) < 2:
        high_slots += [t for t in scored if t not in high_slots][: 2 - len(high_slots)]

    # Lowest-event frames -> rank 10,11 (prefer extension frames first).
    low_candidates = [t for t in reversed(scored) if t not in high_slots]
    low_slots = [t for t in low_candidates if t not in anchor_set][:2]
    if len(low_slots) < 2:
        low_slots += [t for t in low_candidates if t not in low_slots][: 2 - len(low_slots)]

    mid_slots = [t for t in clip_frames if t not in high_slots and t not in low_slots]

    rank_of = {}
    for i, t in enumerate(high_slots):
        rank_of[t] = i  # 0, 1
    for i, t in enumerate(mid_slots):
        rank_of[t] = 2 + i  # 2..9
    for i, t in enumerate(low_slots):
        rank_of[t] = 10 + i  # 10, 11

    return rank_of


def cluster(
    json_path,
    video_path,
    video_frame_tensor_path,
    save_cluster_path,
    dataset,
    combined_output_path=None,
    num_anchors=15,
    lambda_event=0.5,
    alpha_gap=0.6,
    beta_redundancy=0.8,
    target_total=12,
    smax=2,
    max_frames_to_extract=5400,
):
    """
    End-to-end pipeline:
    1) Read QA items and precomputed DINO features.
    2) Stage 1: temporal-chain anchor selection (query-agnostic, up to
       num_anchors anchors).
    3) F2C: expand high-R_event anchors into short temporal clips, trim to
       target_total frames.
    4) Assign ranks (0..11) based on R_event so the existing num_frames=12
       token-budget branch in llava_arch.py allocates more tokens to
       high-motion anchors and fewer tokens to low-motion extension frames.
    5) Map selected feature indices back to original video frame numbers.
    6) Save per-question JSON and optional combined JSON.

    Output format per question (same as original script):
      {question_id: [[frame_index, rank], ...]}
    where rank determines the token budget for that frame (see
    llava_arch.py, num_frames=12 branch).

    Note: this pipeline is fully query-agnostic. There is no CLIP
    image-text reranking step; ranks are derived purely from the
    R_event-based anchor/clip structure.
    """
    os.makedirs(save_cluster_path, exist_ok=True)
    # Load DINO feature tensors generated beforehand (pickle file or folder).
    video_frame_tensor = base.load_video_frame_tensor(video_frame_tensor_path)

    with open(json_path, "r") as f:
        qa_data = json.load(f)

    combined_results = {}
    missing_tensor_count = 0

    for sample in tqdm(qa_data, total=len(qa_data)):
        # Per QA sample metadata.
        video_name = sample["video_name"]
        question_id = sample["question_id"]

        output_path = os.path.join(save_cluster_path, f"{question_id}.json")
        if os.path.exists(output_path):
            # Resume-friendly behavior: if already processed, reuse stored result.
            with open(output_path, "r", encoding="utf-8") as f:
                combined_results.update(json.load(f))
            continue

        # Read video metadata to map feature-frame indices -> original frame indices.
        full_video_path = os.path.join(video_path, video_name)
        cap = cv2.VideoCapture(full_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        tensor = base.get_tensor_for_video(video_frame_tensor, video_name)
        if tensor is None:
            missing_tensor_count += 1
            continue

        tensor = np.asarray(tensor, dtype=np.float32)
        n = tensor.shape[0]

        # Stage 1: temporal-chain anchor selection (Kanchor <= num_anchors).
        selected_indices, r_event = temporal_chain_select(
            tensor,
            num_keyframes=num_anchors,
            lambda_event=lambda_event,
            alpha_gap=alpha_gap,
            beta_redundancy=beta_redundancy,
        )

        # F2C: expand anchors into clips, trim/pad to target_total frames.
        clip_frames, anchor_set = f2c_expand_clips(
            selected_indices, r_event, n, target_total=target_total, smax=smax
        )

        # Convert feature-list indices back to original video frame IDs.
        candidate_frame_indices = [
            base.get_original_frame_number(
                total_frames,
                index,
                fps=fps,
                max_frames_to_extract=max_frames_to_extract,
            )
            for index in clip_frames
        ]

        # If mapping collisions happen, fallback to uniform temporal spacing.
        # This keeps exactly target_total candidates for downstream inference.
        if len(set(candidate_frame_indices)) != target_total:
            candidate_frame_indices = [
                int(index)
                for index in np.linspace(
                    0, float(total_frames), target_total, endpoint=False
                )
            ]
            # Fallback ranking: simple chronological order, ranks 0..target_total-1
            # clipped into [0,11].
            key_frame_order = [
                [frame_index, min(rank, 11)]
                for rank, frame_index in enumerate(candidate_frame_indices)
            ]
        else:
            # F2C ranking: rank derived from R_event-based anchor/clip structure.
            rank_of = assign_f2c_ranks(clip_frames, anchor_set, r_event)
            key_frame_order = [
                [frame_index, rank_of[feat_idx]]
                for frame_index, feat_idx in zip(candidate_frame_indices, clip_frames)
            ]

        result = {question_id: key_frame_order}
        combined_results.update(result)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                result,
                f,
                ensure_ascii=False,
                indent=4,
                default=lambda o: int(o) if isinstance(o, np.integer) else o,
            )

    if combined_output_path:
        # Optionally aggregate all question outputs into one file.
        combined_output_dir = os.path.dirname(combined_output_path)
        if combined_output_dir:
            os.makedirs(combined_output_dir, exist_ok=True)
        with open(combined_output_path, "w", encoding="utf-8") as f:
            json.dump(
                combined_results,
                f,
                ensure_ascii=False,
                indent=4,
                default=lambda o: int(o) if isinstance(o, np.integer) else o,
            )

    print(
        f"Finished temporal-chain + F2C clustering. "
        f"skipped_missing_tensor={missing_tensor_count} total_questions={len(qa_data)}"
    )


@hydra.main(
    config_path="configs/keyframe_cluster", config_name="config", version_base=None
)
def main(cfg: DictConfig):
    """Hydra entrypoint. No CLIP model needed: F2C ranking is query-agnostic."""
    cluster(
        base.resolve_path(cfg.json_path),
        base.resolve_path(cfg.video_path),
        base.resolve_path(cfg.video_frame_tensor_path),
        base.resolve_path(cfg.save_cluster_path),
        cfg.dataset,
        combined_output_path=base.resolve_path(cfg.combined_output_path),
        num_anchors=getattr(cfg, "num_anchors", 15),
        lambda_event=getattr(cfg, "lambda_event", 0.5),
        alpha_gap=getattr(cfg, "alpha_gap", 0.6),
        beta_redundancy=getattr(cfg, "beta_redundancy", 0.8),
        target_total=getattr(cfg, "target_total", 12),
        smax=getattr(cfg, "smax", 2),
        max_frames_to_extract=getattr(cfg, "max_frames_to_extract", 5400),
    )


if __name__ == "__main__":
    main()