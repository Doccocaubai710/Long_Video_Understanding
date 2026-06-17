import json
import os

import clip
import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

import cluster_and_rank_keyframes as base


def minmax_normalize(values, eps=1e-8):
    """Normalize a 1D score vector into [0, 1] using min-max normalization."""
    values = np.asarray(values, dtype=np.float32)
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmax - vmin < eps:
        return np.zeros_like(values, dtype=np.float32)
    return (values - vmin) / (vmax - vmin + eps)


def compute_event_score(frame_features, num_keyframes):
    """
    R_event(t) = ||f_t - f_{t-delta}|| + ||f_{t+delta} - f_t||
    delta = max(1, floor(N / (10K)))
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


def svd_reduce(features: np.ndarray, s: int) -> np.ndarray:
    """Truncated SVD: (N, D) -> (N, s)"""
    s = min(s, features.shape[0], features.shape[1])
    try:
        U, _, _ = np.linalg.svd(features, full_matrices=False)
        return U[:, :s].astype(np.float64)
    except np.linalg.LinAlgError:
        return features[:, :s].astype(np.float64)


def volume_gain(t: int, selected: list, Qs: np.ndarray) -> float:
    """
    Volume gain of adding frame t to selected set:
      gain = sqrt(1 + ||C_t||^2),  C_t = Qs[t] @ pinv(Qs[selected])
    """
    if len(selected) == 0:
        return float(np.linalg.norm(Qs[t]))
    AK = Qs[selected]
    try:
        AK_pinv = np.linalg.pinv(AK)
    except np.linalg.LinAlgError:
        return 1.0
    c = Qs[t] @ AK_pinv
    return float(np.sqrt(1.0 + np.dot(c, c)))


def temporal_gap(t: int, selected: list, n: int) -> float:
    """T_gap(t,S) = min_{s in S} |t - s| / N"""
    if not selected:
        return 1.0
    return float(min(abs(t - s) for s in selected)) / n


def redundancy(t: int, selected: list, norm_features: np.ndarray) -> float:
    """R_red(t,S) = max_{s in S} cos(f_t, f_s)"""
    if not selected:
        return 0.0
    sims = norm_features[selected] @ norm_features[t]
    return float(np.max(sims))


def maxinfo_temporal_chain_select(
    frame_features,
    lambda_event=0.5,
    alpha_gap=0.6,
    beta_redundancy=0.8,
    tol=1.05,
    max_K=20,
    min_K=6,
    svd_s=16,
):
    """
    Greedy selection with adaptive K:
      G(t|S) = R_maxvol~(t,S) + lambda*R_event~(t) + alpha*T_gap~(t,S) - beta*R_red~(t,S)

    Stops when best volume gain < tol (and len(S) >= min_K), or when
    len(S) reaches max_K.

    Returns list[int]: selected indices in feature-frame index space
    (not yet mapped to original video frame IDs).
    """
    frame_features = np.asarray(frame_features, dtype=np.float32)
    n = frame_features.shape[0]
    if n == 0:
        return []
    if n <= min_K:
        return list(range(n))

    max_K = min(max_K, n)
    min_K = min(min_K, n)

    # SVD-reduced features for volume-gain computation.
    Qs = svd_reduce(frame_features, svd_s)

    # Pre-compute event scores using max_K as the K estimate.
    r_event = compute_event_score(frame_features, max_K)
    ev_norm_full = minmax_normalize(r_event)

    # Pre-normalize features for cosine similarity in redundancy term.
    norm_features = frame_features / (
        np.linalg.norm(frame_features, axis=1, keepdims=True) + 1e-8
    )

    selected = []
    candidates = list(range(n))

    while candidates:
        if len(selected) >= max_K:
            break

        vol_gains = np.array(
            [volume_gain(t, selected, Qs) for t in candidates], dtype=np.float32
        )
        tgap_scores = np.array(
            [temporal_gap(t, selected, n) for t in candidates], dtype=np.float32
        )
        red_scores = np.array(
            [redundancy(t, selected, norm_features) for t in candidates], dtype=np.float32
        )

        vol_norm = minmax_normalize(vol_gains)
        tgap_norm = minmax_normalize(tgap_scores)
        red_norm = minmax_normalize(red_scores)
        ev_norm = ev_norm_full[candidates]

        scores = (
            vol_norm
            + lambda_event * ev_norm
            + alpha_gap * tgap_norm
            - beta_redundancy * red_norm
        )

        best_local = int(np.argmax(scores))
        best_t = candidates[best_local]
        best_gain = float(vol_gains[best_local])

        if best_gain < tol and len(selected) >= min_K:
            break

        selected.append(best_t)
        candidates.pop(best_local)

    if len(selected) < min_K:
        remaining = [t for t in range(n) if t not in set(selected)]
        for t in remaining:
            if len(selected) >= min_K:
                break
            selected.append(t)

    return sorted(selected)


def cluster(
    json_path,
    video_path,
    video_frame_tensor_path,
    save_cluster_path,
    dataset,
    combined_output_path=None,
    lambda_event=0.5,
    alpha_gap=0.6,
    beta_redundancy=0.8,
    tol=1.05,
    max_K=20,
    min_K=6,
    svd_s=16,
    max_frames_to_extract=5400,
):
    """
    End-to-end pipeline:
    1) Read QA items and precomputed DINO features.
    2) Stage 1: MaxInfo + temporal-chain candidate selection (query-agnostic,
       adaptive K via volume-gain stopping rule).
    3) Map selected feature indices back to original video frame numbers.
    4) Stage 2: CLIP question-aware ranking of those candidates.
    5) Save per-question JSON and optional combined JSON.

    Output format per question (same as original script):
      {question_id: [[frame_index, rank], ...]}
    where rank=0 is the most question-relevant candidate (from CLIP sort).
    """
    os.makedirs(save_cluster_path, exist_ok=True)
    video_frame_tensor = base.load_video_frame_tensor(video_frame_tensor_path)

    with open(json_path, "r") as f:
        qa_data = json.load(f)

    combined_results = {}
    missing_tensor_count = 0

    for sample in tqdm(qa_data, total=len(qa_data)):
        video_name = sample["video_name"]
        prompt = sample["question"]
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
            missing_tensor_count += 1
            continue

        # Stage 1: MaxInfo + temporal-chain selection, adaptive K.
        selected_feature_indices = maxinfo_temporal_chain_select(
            tensor,
            lambda_event=lambda_event,
            alpha_gap=alpha_gap,
            beta_redundancy=beta_redundancy,
            tol=tol,
            max_K=max_K,
            min_K=min_K,
            svd_s=svd_s,
        )
        num_selected = len(selected_feature_indices)

        # Convert feature-list indices back to original video frame IDs.
        candidate_frame_indices = [
            base.get_original_frame_number(
                total_frames,
                index,
                fps=fps,
                max_frames_to_extract=max_frames_to_extract,
            )
            for index in selected_feature_indices
        ]

        # If mapping collisions reduce unique frames, fallback to uniform
        # spacing with the same adaptive count.
        if len(set(candidate_frame_indices)) != num_selected:
            candidate_frame_indices = [
                int(index)
                for index in np.linspace(
                    0, float(total_frames), num_selected, endpoint=False
                )
            ]

        frames_cluster, _ = base.load_video(
            full_video_path,
            candidate_frame_indices,
            start=None,
            end=None,
        )
        image_batch = [base.preprocess_clip(frame) for frame in frames_cluster]
        if not image_batch:
            continue

        prompt_words = prompt.split(" ")
        if len(prompt_words) > 40:
            prompt = " ".join(prompt_words[10:50])

        # Stage 2: question-aware ranking using CLIP image-text similarity.
        image_input = torch.stack(image_batch).to(base.device)
        with torch.no_grad():
            image_features = base.model_clip.encode_image(image_input)
            text_input = clip.tokenize([prompt]).to(base.device)
            text_features = base.model_clip.encode_text(text_input)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = 100.0 * image_features @ text_features.T

        top_k_count = min(num_selected, similarity.numel())
        _, top_k_indices = torch.topk(similarity.squeeze(), top_k_count)
        top_keyframes = [candidate_frame_indices[int(index)] for index in top_k_indices]

        key_frame_order = [[frame_index, rank] for rank, frame_index in enumerate(top_keyframes)]

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
        f"Finished MaxInfo temporal-chain clustering. "
        f"skipped_missing_tensor={missing_tensor_count} total_questions={len(qa_data)}"
    )


@hydra.main(
    config_path="configs/keyframe_cluster", config_name="config", version_base=None
)
def main(cfg: DictConfig):
    """Hydra entrypoint. Mirrors temporal_chain_keyframe_and_order.py interface."""
    base.load_clip_model(cfg.device)
    cluster(
        base.resolve_path(cfg.json_path),
        base.resolve_path(cfg.video_path),
        base.resolve_path(cfg.video_frame_tensor_path),
        base.resolve_path(cfg.save_cluster_path),
        cfg.dataset,
        combined_output_path=base.resolve_path(cfg.combined_output_path),
        lambda_event=getattr(cfg, "lambda_event", 0.5),
        alpha_gap=getattr(cfg, "alpha_gap", 0.6),
        beta_redundancy=getattr(cfg, "beta_redundancy", 0.8),
        tol=getattr(cfg, "tol", 1.05),
        max_K=getattr(cfg, "max_K", 20),
        min_K=getattr(cfg, "min_K", 6),
        svd_s=getattr(cfg, "svd_s", 16),
        max_frames_to_extract=getattr(cfg, "max_frames_to_extract", 5400),
    )


if __name__ == "__main__":
    main()