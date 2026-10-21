"""
run_inference_qwen25vl_dyseg.py

Full implementation of idea.md's method: Tang 1 (DySeg dynamic segmentation)
+ Tang 2 (non-uniform budget allocation, the main contribution) + Tang 3
(STPrune: DTM density-peak merging + residual salience, replacing ATS since
Qwen2.5-VL has no [CLS] token).

Rewritten to sit on top of the harness-matched pipeline validated in
run_inference_qwen25vl_native.py (native 63.33% vs SCORE's Vanilla 63.8% --
see that file's docstring for the full investigation): native video-path
input via FastVID's patched vision_process.py (qwen_vision_process_fastvid.py,
keeps total dense video tokens bounded near ~14K-16K regardless of duration,
instead of the old fixed-128-frame extraction), lmms-eval's exact prompt /
system prompt / greedy generation config. The DySeg/allocation/STPrune math
itself (Tang 1-3) is UNCHANGED -- it only consumes video_grid_thw + the
pooled video embeddings, which is agnostic to how the video was sampled.

COMPRESSION MARGIN: MAX_PIXELS below is set to 3x the native-Vanilla value
(16384*28*28 -> 16384*3*28*28), so the dense (pre-prune) encoding now lands
around ~42K-48K tokens/video -- close to SCORE's own true-dense measurement
(~46,080, see run_inference_qwen25vl_native.py's Gate G1 notes) -- instead of
the ~14K-16K range that made the first harness-matched run (61.78%) nearly
indistinguishable from Vanilla (63.33%) with no real room for STPrune to
matter. tokens_num stays at 16384, so this is now a genuine ~35% retention
compression, comparable to the retention ratios FastVID/SCORE actually
report against (25-40%), and the comparison target is still the SAME
Vanilla number (63.33%) -- only DySeg's own input pool got richer, not the
baseline.

MANDATORY FIRST RUN: verify_mode=true (no pruning at all) on a handful of
questions, and diff the output against run_inference_qwen25vl_native.py's
predictions for the SAME questions. The verify_mode check done before this
rewrite (30/30 match) was against the OLD frame-extraction pipeline and is
now stale -- re-run it, since the input-construction path changed even
though the position-preserving gather logic (Phan 4.3's hard constraint)
did not.

Pipeline per question:
  1. Encode video natively to get merged visual tokens X[T,N,D] (T =
     temporal-pair units, N = spatial tokens/unit) via
     model.get_video_features(...).pooler_output.
  2. DySeg: segment the T units by frame-to-frame cosine-similarity
     transitions (idea.md Phan 2.2).
  3. Per segment: score v_s = lambda*v_RC + (1-lambda)*v_delta (idea.md
     Phan 3.2), combine, entropy-gate temperature, allocate B_s tokens
     per segment (idea.md Phan 3.3-3.4).
  4. Per segment: STPrune (DTM anchors + delta-salience) selects which
     B_s tokens to keep (idea.md Phan 4).
  5. CRITICAL correctness step: let the model itself compute position_ids
     for the FULL unpruned sequence (known-correct code path), then GATHER
     only the rows for kept video tokens. This avoids hand-reimplementing
     the mRoPE position formula (idea.md Phan 4.3's "hard constraint").
  6. Manually build inputs_embeds (text embeds + kept video embeds spliced
     in, in original temporal order) and the corresponding gathered
     position_ids, then call model.generate(inputs_embeds=..., position_ids=...).

Requires: pip install qwen-vl-utils decord   (same env as
run_inference_qwen25vl_native.py)
"""

import json
import math
import os

# A handful of VideoMME videos trip decord's EOF-retry limit when seeking
# near the end of file (intermittent -- seen on video #1098 of a run where
# the first 1097 read fine). Must be set before decord is used; doubling
# the default per decord's own suggested fix in the error message.
os.environ.setdefault("DECORD_EOF_RETRY_MAX", "20480")

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from qwen_vision_process_fastvid import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from utils import get_chunk

# Same prompt/system-prompt/generation-config fixes as
# run_inference_qwen25vl_native.py. MAX_PIXELS is overridable via hydra
# (max_pixels_units=<N>, in 28x28-patch units) -- default 3x Vanilla's (see
# docstring "COMPRESSION MARGIN" above) so STPrune has a real dense pool to
# compress from. Override to 16384 (1x) to reproduce the near-100%-retention
# regime where dense pool ~= tokens_num, e.g. for the retention~100% unit
# test in README_DySeg_STPrune.md.
MIN_PIXELS = 16 * 28 * 28
SYSTEM_PROMPT = "You are a helpful assistant."

VIDEOMME_OPTION_PROMPT = (
    "Select the best answer to the following multiple-choice question "
    "based on the video and the subtitles. Respond with only the letter "
    "(A, B, C, or D) of the correct option."
)
VIDEOMME_POST_PROMPT = "\nAnswer with the option's letter from the given choices directly."


def resolve_path(path):
    if path is None:
        return None
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    return to_absolute_path(path)


def load_answered_ids(output_path):
    answered_ids = set()
    if not os.path.exists(output_path):
        return answered_ids
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            answer_id = data.get("id", data.get("question_id"))
            if answer_id is not None:
                answered_ids.add(str(answer_id))
    return answered_ids


def build_messages(video_path, question, candidates, max_pixels):
    option_block = "\n".join(candidates)
    text = f"{VIDEOMME_OPTION_PROMPT}\n{question}\n{option_block}\n{VIDEOMME_POST_PROMPT}"
    content = [
        {"type": "video", "video": video_path, "max_pixels": max_pixels, "min_pixels": MIN_PIXELS},
        {"type": "text", "text": text},
    ]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


# ============================================================
# TANG 1 -- DySeg (idea.md Phan 2.2)
# ============================================================
def dyseg_segments(frame_feats, c=8, tau=0.9):
    """frame_feats: [T, D] L2-normalized per-unit global features.
    Returns list of (start, end) exclusive-end ranges covering [0, T)."""
    T = frame_feats.shape[0]
    if T <= 1:
        return [(0, T)]
    f = frame_feats / (frame_feats.norm(dim=-1, keepdim=True) + 1e-8)
    trans = (f[:-1] * f[1:]).sum(-1)  # [T-1], cos(f_i, f_{i+1})
    trans_np = trans.detach().float().cpu().numpy()

    c_eff = min(max(1, c - 1), T - 1)
    s1 = set(np.argsort(trans_np)[:c_eff].tolist())
    s2 = set(int(i) for i in np.where(trans_np < tau)[0])
    boundary_after = sorted(s1 | s2)  # boundary AFTER index i, i.e. cut between i and i+1

    segments, start = [], 0
    for b in boundary_after:
        end = b + 1
        if end > start:
            segments.append((start, end))
        start = end
    if start < T:
        segments.append((start, T))
    return segments


# ============================================================
# TANG 2a -- segment density score v_s (idea.md Phan 3.2)
# ============================================================
def recon_curve_ratio(F_s, gamma=0.9):
    """Greedy QR-pivot on segment-level frame features F_s [P_s, D].
    Returns k/P_s where k = frames needed to reach gamma reconstruction."""
    P = F_s.shape[0]
    if P <= 1:
        return 1.0
    X = F_s.detach().float().cpu().numpy()
    X = X - X.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / (norm + 1e-8)
    R = X.copy()
    total_energy = float(np.sum(X ** 2)) + 1e-8
    cumulative = 0.0
    k = 0
    for _ in range(P):
        rn = np.linalg.norm(R, axis=1)
        a = int(np.argmax(rn))
        v = rn[a]
        if v < 1e-10:
            k += 1
            break
        q = R[a] / v
        proj = R @ q
        cumulative += float(np.sum(proj ** 2))
        R = R - np.outer(proj, q)
        k += 1
        if cumulative / total_energy >= gamma:
            break
    return k / P


def segment_scores(X, segments, gamma=0.9, lam=0.5):
    """X: [T, N, D] merged visual tokens (post spatial-merge), already the
    per-video full sequence. Returns v_s (normalized, list[float]) aligned
    with `segments`, plus raw components for logging."""
    T = X.shape[0]
    frame_feat = X.mean(dim=1)  # [T, D] global per-unit feature
    delta = torch.zeros_like(frame_feat).unsqueeze(1).expand(-1, X.shape[1], -1).clone()
    delta[1:] = X[1:] - X[:-1]
    delta[0] = X[0]
    delta_energy = delta.norm(dim=-1)  # [T, N]

    rc_list, d_list = [], []
    for (s, e) in segments:
        rc_list.append(recon_curve_ratio(frame_feat[s:e], gamma=gamma))
        d_list.append(float(delta_energy[s:e].mean().item()))

    def minmax(vals):
        vals = np.asarray(vals, dtype=np.float64)
        lo, hi = vals.min(), vals.max()
        if hi - lo < 1e-8:
            return np.ones_like(vals) * 0.5
        return (vals - lo) / (hi - lo)

    rc_n = minmax(rc_list)
    d_n = minmax(d_list)
    v = lam * rc_n + (1 - lam) * d_n
    return v.tolist(), rc_list, d_list, delta_energy


# ============================================================
# TANG 2b -- entropy-gated non-uniform allocation (idea.md Phan 3.3-3.4)
# ============================================================
def allocate_budget(v_s, B_total, b_min, b_max, T0=1.0, kappa=1.0, H0=0.5):
    S = len(v_s)
    v = np.asarray(v_s, dtype=np.float64)
    z0 = v / max(T0, 1e-6)
    p0 = np.exp(z0 - z0.max())
    p0 = p0 / p0.sum()
    H = float(-(p0 * np.log(p0 + 1e-12)).sum())
    H_hat = H / max(math.log(S), 1e-8) if S > 1 else 1.0
    T = T0 * math.exp(kappa * (H_hat - H0))

    z = v / max(T, 1e-6)
    p = np.exp(z - z.max())
    p = p / p.sum()

    b_min_eff = min(b_min, B_total // (2 * S)) if S > 0 else b_min
    free = max(0, B_total - S * b_min_eff)
    B_s = b_min_eff + free * p
    B_s = np.clip(B_s, b_min_eff, b_max)
    # renormalize to hit B_total as closely as possible
    scale = B_total / max(B_s.sum(), 1e-8)
    B_s = np.clip(np.round(B_s * scale), b_min_eff, b_max).astype(int)
    return B_s.tolist(), H_hat, T


# ============================================================
# TANG 3 -- STPrune: DTM (density-peak) + residual salience (idea.md Phan 4)
# replacing ATS with residual salience since Qwen2.5-VL has no [CLS].
# ============================================================
def density_peaks(tokens):
    """tokens: [n, D] torch tensor, kept on its current device (GPU) --
    the original numpy/CPU version round-tripped every anchor unit's 720
    tokens through the CPU with an O(n^2) Python-level loop, which was the
    dominant cost of the whole pipeline (~100s/question). This version does
    the full pairwise distance + density-peak ranking as fused GPU ops.
    Returns (rho, delta) as torch tensors, len n."""
    n = tokens.shape[0]
    device = tokens.device
    if n <= 1:
        return torch.ones(n, device=device), torch.ones(n, device=device)
    x = tokens.detach().float()
    d2 = torch.cdist(x, x, p=2) ** 2  # [n,n], single fused GPU op

    k = max(1, min(8, n - 1))
    knn_d2, _ = torch.topk(d2, k + 1, dim=1, largest=False)  # incl. self (0)
    rho = torch.exp(-knn_d2[:, 1:].mean(dim=1))

    order = torch.argsort(rho, descending=True)
    d2_ordered = d2[order][:, order]
    # for rank r (0-indexed), only columns < r are "higher density" candidates
    future_mask = torch.triu(
        torch.ones(n, n, device=device, dtype=torch.bool), diagonal=0
    )
    d2_masked = d2_ordered.masked_fill(future_mask, float("inf"))
    delta_ordered = d2_masked.min(dim=1).values
    delta_ordered[0] = d2_ordered[0].max()  # highest-density point: no one "higher"
    delta_ordered = delta_ordered.clamp(min=0).sqrt()

    delta = torch.empty(n, device=device)
    delta[order] = delta_ordered
    return rho, delta


def stprune_segment(X, seg_range, t_h_w_shape, B_s, d_ratio=0.4, p_anchor=4, alpha=0.6,
                     salience_mode="delta", vjepa_sal=None):
    """X: [T,N,D] full merged token tensor. seg_range=(s,e) unit indices.
    t_h_w_shape=(H_llm, W_llm) spatial grid per unit.
    salience_mode: "delta" (default, ||X[t]-X[t-1]||, motion-based -- see
    README_DySeg_STPrune.md Sec 3b caveat) | "norm" (||X[t]||, crude
    content-magnitude proxy, no cross-frame diff) | "grid" (ignore any
    per-token score entirely; evenly-spaced flat-index pick within the
    segment's remaining candidates, isolating whether ranking by *any*
    signal is actively harmful vs a content-blind pick) | "vjepa"
    (masked-prediction error from a precomputed V-JEPA 2 grid, mapped onto
    this video's [T,N] token grid -- semantic/whole-context signal instead
    of motion; requires vjepa_sal, a [T,N] tensor, see
    precompute_vjepa_salience.py and map_vjepa_to_qwen_grid()).
    Returns: list of (flat_index_into_X, embed_vector) kept for this segment,
    where flat_index = t*N + spatial_idx (used later to fetch position_ids)."""
    s, e = seg_range
    P = e - s
    H_llm, W_llm = t_h_w_shape
    N = H_llm * W_llm
    B_s = max(1, int(B_s))

    n_dtm_total = max(0, int(round(d_ratio * B_s)))
    n_sal_total = max(0, B_s - n_dtm_total)

    anchor_units = list(range(s, e, p_anchor))
    if not anchor_units:
        anchor_units = [s]
    n_dtm_per_anchor = max(1, n_dtm_total // len(anchor_units)) if n_dtm_total > 0 else 0

    kept = {}  # flat_index -> embed tensor (all torch ops below stay on GPU)

    # --- DTM on each anchor unit's own N tokens ---
    for t in anchor_units:
        toks = X[t]  # [N, D]
        if n_dtm_per_anchor <= 0:
            break
        rho, delta = density_peaks(toks)
        score = rho * delta
        n_pick = min(n_dtm_per_anchor, N)
        anchor_local_idx = torch.topk(score, n_pick).indices  # [n_pick]

        toks_unit = torch.nn.functional.normalize(toks, dim=-1, eps=1e-8)
        sim = toks_unit @ toks_unit[anchor_local_idx].T  # [N, n_pick]
        assign = sim.argmax(dim=1)  # [N], which anchor each token maps to

        for a_pos in range(anchor_local_idx.shape[0]):
            a_local = int(anchor_local_idx[a_pos].item())
            members = (assign == a_pos).nonzero(as_tuple=True)[0]
            if members.numel() == 0:
                merged = toks[a_local]
            else:
                b = toks[members].mean(dim=0)
                merged = alpha * toks[a_local] + (1 - alpha) * b
            flat_idx = t * N + a_local  # position = ANCHOR's original slot
            kept[flat_idx] = merged

    # --- Salience over the WHOLE segment (all units, all tokens) ---
    if n_sal_total > 0:
        seg_tokens = X[s:e].reshape(-1, X.shape[-1])  # [(e-s)*N, D]
        idx_range = torch.arange(seg_tokens.shape[0], device=X.device)
        candidate_flat = (s + idx_range // N) * N + (idx_range % N)

        if salience_mode == "grid":
            # No per-token score at all -- evenly-spaced pick over remaining
            # (not-yet-DTM-kept) candidates. Isolates: is ranking by ANY
            # signal actively harmful, vs a content-blind spread?
            remaining_mask = torch.tensor(
                [int(candidate_flat[i].item()) not in kept for i in range(seg_tokens.shape[0])],
                device=X.device,
            )
            remaining_local = remaining_mask.nonzero(as_tuple=True)[0]
            n_pick = min(n_sal_total, remaining_local.shape[0])
            if n_pick > 0:
                pick_local = torch.linspace(
                    0, remaining_local.shape[0] - 1, n_pick, device=X.device
                ).round().long()
                pick_local = torch.unique(remaining_local[pick_local], sorted=True)
            else:
                pick_local = torch.empty(0, dtype=torch.long, device=X.device)
            for oi in pick_local.tolist():
                flat_idx = int(candidate_flat[oi].item())
                kept[flat_idx] = seg_tokens[oi]
            return kept

        if salience_mode == "vjepa":
            # vjepa_sal: [T,N] precomputed masked-prediction-error grid,
            # already mapped onto this video's own token grid (see
            # map_vjepa_to_qwen_grid()). Semantic/whole-context signal.
            sal = vjepa_sal[s:e].reshape(-1)
        elif salience_mode == "norm":
            # Crude content-magnitude proxy -- no cross-frame temporal diff
            # at all, so camera-shake/motion cannot dominate the ranking.
            sal = seg_tokens.norm(dim=-1)
        else:  # "delta" (default) -- motion/change vs the previous unit
            if s == 0:
                delta_prev = torch.zeros_like(X[s])
            else:
                delta_prev = X[s] - X[s - 1]
            deltas = [delta_prev]
            for t in range(s + 1, e):
                deltas.append(X[t] - X[t - 1])
            seg_delta = torch.cat(deltas, dim=0)  # [(e-s)*N, D]
            sal = seg_delta.norm(dim=-1)  # [(e-s)*N], stays on GPU

        order = torch.argsort(sal, descending=True).tolist()
        picked = 0
        for oi in order:
            if picked >= n_sal_total:
                break
            flat_idx = int(candidate_flat[oi].item())
            if flat_idx in kept:
                continue
            kept[flat_idx] = seg_tokens[oi]
            picked += 1

    return kept  # dict[flat_index] -> embed


# ============================================================
# Main per-question pipeline
# ============================================================
def map_vjepa_to_qwen_grid(vjepa_grid, T_q, H_q, W_q):
    """vjepa_grid: [T_v, H_v, W_v] precomputed error grid (see
    precompute_vjepa_salience.py), a coarser, INDEPENDENTLY-sampled grid
    (V-JEPA's own 64-frame/256px sampling, not synced to Qwen's native
    dynamic frame/resolution choice for this video). Maps by fractional
    position (0..1) along each axis -- robust to the two encoders using
    different frame counts/fps/resolution for the same video, since neither
    needs to know the other's exact sampling. Nearest-neighbor lookup:
    V-JEPA's grid is much coarser than Qwen's, so many Qwen tokens will
    legitimately map to the same V-JEPA cell (idea.md's own note: V-JEPA
    isn't meant to resolve fine spatial detail, only temporal/semantic
    region-level surprise).
    Returns: [T_q, H_q, W_q] tensor, same device/dtype as vjepa_grid input
    (moved to CPU float32 internally, caller can .to(device) after)."""
    T_v, H_v, W_v = vjepa_grid.shape
    t_idx = (torch.arange(T_q).float() / max(T_q, 1) * T_v).long().clamp(0, T_v - 1)
    h_idx = (torch.arange(H_q).float() / max(H_q, 1) * H_v).long().clamp(0, H_v - 1)
    w_idx = (torch.arange(W_q).float() / max(W_q, 1) * W_v).long().clamp(0, W_v - 1)
    mapped = vjepa_grid[t_idx][:, h_idx][:, :, w_idx]  # [T_q, H_q, W_q]
    return mapped


def prune_video_tokens(model, X_pooler, video_grid_thw, args, vjepa_grid=None, budget=None):
    """X_pooler: [n_merged_tokens, D] flat, in (t,h,w) row-major order.
    video_grid_thw: [T_raw, H_raw, W_raw] as reported by the processor.
    vjepa_grid: optional [T_v,H_v,W_v] precomputed V-JEPA error grid for
    this video, required when salience_mode=="vjepa".
    budget: optional per-video B_total (retention mode, see run_one_question);
    None keeps the fixed args.tokens_num budget.
    Returns: sorted list of flat_index (into the ORIGINAL n_merged_tokens
    ordering) that should be kept, in ascending (temporal-then-spatial) order."""
    T_llm = int(video_grid_thw[0])
    H_llm = int(video_grid_thw[1]) // 2
    W_llm = int(video_grid_thw[2]) // 2
    N = H_llm * W_llm
    D = X_pooler.shape[-1]
    X = X_pooler.reshape(T_llm, N, D)

    segments = dyseg_segments(X.mean(dim=1), c=args.dyseg_c, tau=args.dyseg_tau)
    v_s, rc_list, d_list, _ = segment_scores(X, segments, gamma=args.recon_gamma, lam=args.lam)

    B_total = int(args.tokens_num)
    b_min = max(1, int(args.b_min))
    b_max = max(b_min, int(args.b_max))
    if budget is not None:
        # Retention mode: b_min/b_max keep their ratio to the reference budget
        # (tokens_num), and b_max is never allowed to make B_total infeasible
        # (S * b_max < B_total would silently under-spend the budget).
        scale = int(budget) / float(args.tokens_num)
        B_total = int(budget)
        b_min = max(1, int(round(b_min * scale)))
        b_max = max(b_min, int(round(b_max * scale)), math.ceil(B_total / max(1, len(segments))))
    B_s_list, H_hat, T_gate = allocate_budget(
        v_s, B_total, b_min, b_max,
        T0=args.temperature0, kappa=args.kappa, H0=args.h0,
    )

    # alloc_mode (ablation, this _abl copy only): "score" (default) keeps the
    # entropy-gated v_s allocation above; "length" ignores v_s and gives every
    # segment a budget proportional to its number of units (largest-remainder
    # rounding), i.e. the same retention ratio in every segment -- isolates the
    # effect of NON-UNIFORM allocation from the effect of token selection.
    if getattr(args, "alloc_mode", "score") == "length":
        lens = np.asarray([e - s for s, e in segments], dtype=np.float64)
        raw = B_total * lens / lens.sum()
        base = np.floor(raw).astype(int)
        for i in np.argsort(-(raw - base))[: int(B_total - base.sum())]:
            base[i] += 1
        B_s_list = base.tolist()

    # fill_budget (default: ON in retention mode, OFF in the legacy fixed-budget
    # mode): allocate_budget() ignores segment LENGTH, so a short segment can be
    # handed more tokens than it contains ((e-s)*N) and the excess is silently
    # lost in stprune_segment (measured 2026-10-07: 19-22% kept instead of 25%
    # on videos with 8-12 segments). Cap every B_s at its segment's capacity and
    # hand the excess to segments that still have room, proportionally to their
    # original allocation.
    if bool(getattr(args, "fill_budget", budget is not None)):
        caps = [(e - s) * N for s, e in segments]
        target = min(B_total, sum(caps))
        filled = [min(int(b), c) for b, c in zip(B_s_list, caps)]
        deficit = target - sum(filled)
        while deficit > 0:
            open_idx = [i for i in range(len(filled)) if caps[i] > filled[i]]
            if not open_idx:
                break
            weights = [max(int(B_s_list[i]), 1) for i in open_idx]
            w_sum = float(sum(weights))
            given = 0
            for i, w in zip(open_idx, weights):
                add = min(caps[i] - filled[i], int(deficit * w / w_sum))
                filled[i] += add
                given += add
            if given == 0:  # con lai it hon so doan: chia moi doan 1 token
                for i in open_idx:
                    if given >= deficit:
                        break
                    filled[i] += 1
                    given += 1
            deficit -= given
        B_s_list = filled

    salience_mode = getattr(args, "salience_mode", "delta")
    vjepa_sal = None
    if salience_mode == "vjepa":
        assert vjepa_grid is not None, "salience_mode=vjepa requires a loaded V-JEPA cache for this video"
        vjepa_sal = map_vjepa_to_qwen_grid(vjepa_grid, T_llm, H_llm, W_llm).reshape(T_llm, N).to(X.device)

    all_kept = {}
    for seg, B_s in zip(segments, B_s_list):
        kept = stprune_segment(
            X, seg, (H_llm, W_llm), B_s,
            d_ratio=args.dtm_ratio, p_anchor=args.p_anchor, alpha=args.dtm_alpha,
            salience_mode=salience_mode, vjepa_sal=vjepa_sal,
        )
        all_kept.update(kept)

    kept_indices = sorted(all_kept.keys())
    kept_embeds = torch.stack([all_kept[i] for i in kept_indices], dim=0)
    return kept_indices, kept_embeds, dict(
        n_segments=len(segments), H_hat=H_hat, T_gate=T_gate,
        B_s=B_s_list, target_total=B_total, actual_total=len(kept_indices),
    )


# ============================================================
# Uniform baseline -- isolates whether DySeg's non-uniform allocation +
# STPrune selection specifically help, vs. just evenly sampling tokens_num
# tokens from the SAME dense pool (same MAX_PIXELS, same tokens_num), no
# segmentation, no allocation formula, no DTM merging.
# ============================================================
def uniform_prune_video_tokens(X_pooler, B_total):
    """Evenly-spaced flat-index sampling over the whole [n_merged_tokens, D]
    pool -- the fair, same-budget, same-dense-pool comparison point for
    prune_video_tokens(). No segment awareness, no per-token scoring."""
    n_total = X_pooler.shape[0]
    B_total = min(int(B_total), n_total)
    idx = torch.linspace(0, n_total - 1, B_total, device=X_pooler.device).round().long()
    idx = torch.unique(idx, sorted=True)
    kept_indices = idx.tolist()
    kept_embeds = X_pooler[idx]
    return kept_indices, kept_embeds, dict(
        mode="uniform", n_total=n_total, target_total=B_total, actual_total=len(kept_indices),
    )


# ============================================================
# V-CAST-style baseline (arXiv 2603.27650), re-implemented from the paper's
# description, NOT the authors' code: per-unit budget from the curvature of
# the unit-mean feature trajectory, then per-unit top-k by (distance to the
# unit mean + min-max normalized norm). Selection only, no merging, tokens
# keep their original (t,h,w) slot. "Frame" in the paper = temporal-pair unit
# here.
# ============================================================
def vcast_prune_video_tokens(X_pooler, video_grid_thw, B_total, tau=0.7):
    T = int(video_grid_thw[0])
    N = (int(video_grid_thw[1]) // 2) * (int(video_grid_thw[2]) // 2)
    n_total = X_pooler.shape[0]
    B_total = min(int(B_total), n_total)
    X = X_pooler.detach().float().reshape(T, N, -1)

    xbar = X.mean(dim=1)  # [T, D]
    xhat = torch.nn.functional.normalize(xbar, dim=-1, eps=1e-8)
    t_idx = torch.arange(T, device=X.device)
    v_in = xhat - xhat[(t_idx - 1).clamp(min=0)]
    v_out = xhat[(t_idx + 1).clamp(max=T - 1)] - xhat
    kappa = 1.0 - torch.nn.functional.cosine_similarity(v_in, v_out, dim=-1, eps=1e-8)  # [T]

    # p_t ∝ exp(kappa_t / tau) -> integer budgets in [0, N] summing to B_total
    p = torch.softmax(kappa / tau, dim=0).double().cpu().numpy()
    budgets = np.zeros(T, dtype=int)
    remaining = B_total
    while remaining > 0:
        open_idx = np.where(budgets < N)[0]
        if open_idx.size == 0:
            break
        raw = remaining * p[open_idx] / p[open_idx].sum()
        add = np.minimum(np.floor(raw).astype(int), N - budgets[open_idx])
        if add.sum() == 0:  # fewer tokens left than open units: largest remainders first
            for j in np.argsort(-raw)[:remaining]:
                add[j] = 1
        budgets[open_idx] += add
        remaining -= int(add.sum())

    cos_to_mean = torch.nn.functional.cosine_similarity(X, xbar.unsqueeze(1), dim=-1, eps=1e-8)  # [T, N]
    norm = X.norm(dim=-1)  # [T, N]
    lo, hi = norm.min(dim=1, keepdim=True).values, norm.max(dim=1, keepdim=True).values
    score = (1.0 - cos_to_mean) + (norm - lo) / (hi - lo + 1e-8)

    kept = []
    for t in range(T):
        b = int(budgets[t])
        if b > 0:
            kept.append(t * N + torch.topk(score[t], b).indices)
    idx = torch.sort(torch.cat(kept)).values
    return idx.tolist(), X_pooler[idx], dict(
        mode="vcast", n_total=n_total, target_total=B_total, actual_total=int(idx.numel()),
        n_segments=int((budgets > 0).sum()),
    )


def run_one_question(model, processor, video_path, question, candidates, args, verify_mode, max_pixels, cache=None):
    """cache: optional dict reused across consecutive questions of the SAME
    video (decoded frames, video features, kept tokens) -- valid because the
    whole pruning pipeline is query-agnostic. None = recompute everything."""
    if cache is None:
        cache = {}
    if cache.get("video_path") != video_path:
        cache.clear()
        cache["video_path"] = video_path
    messages = build_messages(video_path, question, candidates, max_pixels)
    # max_frames (optional): cap on sampled frames, e.g. 128 to match the
    # fixed-baseline harness of run_inference_qwen25vl_kitoke.py. Unset keeps
    # the old behaviour (fps=2.0 up to FPS_MAX_FRAMES=768).
    max_frames = getattr(args, "max_frames", None)
    if max_frames:
        messages[1]["content"][0]["max_frames"] = int(max_frames)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Deliberately NOT forwarding video_kwargs (fps) into processor() -- see
    # run_inference_qwen25vl_native.py's docstring point 1 for why.
    if "video_inputs" not in cache:
        cache["image_inputs"], cache["video_inputs"], _ = process_vision_info(messages, return_video_kwargs=True)
    image_inputs, video_inputs = cache["image_inputs"], cache["video_inputs"]
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)

    # time_mode: "lmms" (default, project-wide convention since 2026-10-07)
    # pins second_per_grid_ts to 1.0 like the lmms-eval harness (see
    # run_inference_qwen25vl_kitoke.py docstring). "hf_default" leaves the
    # processor value untouched = behaviour of all runs before that date, where
    # transformers 5.x collapses every unit onto one temporal position.
    time_mode = getattr(args, "time_mode", "lmms")
    if time_mode == "lmms":
        inputs["second_per_grid_ts"] = torch.full_like(inputs["second_per_grid_ts"], 1.0)
    elif time_mode != "hf_default":
        raise ValueError(f"unknown time_mode={time_mode!r}, expected 'lmms' or 'hf_default'")

    video_grid_thw = inputs["video_grid_thw"][0]
    llm = model.model if hasattr(model, "model") else model

    with torch.inference_mode():
        # Full, correct position_ids for the UNPRUNED sequence -- known-good
        # code path. We only ever GATHER from this, never recompute by hand.
        full_embeds = model.get_input_embeddings()(inputs["input_ids"])
        if "video_embeds_full" not in cache:
            video_feat = llm.get_video_features(
                inputs["pixel_values_videos"], video_grid_thw=inputs["video_grid_thw"]
            )
            video_embeds_full = video_feat.pooler_output
            if isinstance(video_embeds_full, (list, tuple)):
                video_embeds_full = torch.cat(video_embeds_full, dim=0)
            cache["video_embeds_full"] = video_embeds_full.to(full_embeds.dtype).to(full_embeds.device)
        video_embeds_full = cache["video_embeds_full"]

        video_token_id = model.config.video_token_id
        video_mask = (inputs["input_ids"][0] == video_token_id)
        video_positions_in_seq = video_mask.nonzero(as_tuple=True)[0]  # [n_merged_tokens]
        assert video_positions_in_seq.shape[0] == video_embeds_full.shape[0], (
            f"placeholder count {video_positions_in_seq.shape[0]} != "
            f"video embed count {video_embeds_full.shape[0]}"
        )

        position_ids, _ = llm.get_rope_index(
            input_ids=inputs["input_ids"],
            image_grid_thw=None,
            video_grid_thw=inputs["video_grid_thw"],
            second_per_grid_ts=inputs.get("second_per_grid_ts"),
            attention_mask=inputs["attention_mask"],
            mm_token_type_ids=inputs.get("mm_token_type_ids"),
        )  # [3, 1, seq_len]

        mode = getattr(args, "mode", "dyseg")
        n_total = int(video_embeds_full.shape[0])
        # retention (optional): per-video budget K = floor(retention * n_total),
        # same formula as run_inference_qwen25vl_kitoke.py. Unset keeps the
        # fixed absolute budget args.tokens_num.
        retention = getattr(args, "retention", None)
        budget = max(1, int(float(retention) * n_total)) if retention else None
        if "kept" in cache:
            kept_indices, kept_embeds, diag = cache["kept"]
        elif verify_mode:
            kept_indices = list(range(video_embeds_full.shape[0]))
            kept_embeds = video_embeds_full
            diag = dict(mode="verify_no_prune", actual_total=len(kept_indices))
        elif mode == "uniform":
            kept_indices, kept_embeds, diag = uniform_prune_video_tokens(
                video_embeds_full, args.tokens_num if budget is None else budget
            )
        elif mode == "vcast":
            kept_indices, kept_embeds, diag = vcast_prune_video_tokens(
                video_embeds_full, video_grid_thw, args.tokens_num if budget is None else budget,
                tau=float(getattr(args, "vcast_tau", 0.7)),
            )
        else:
            vjepa_grid = None
            if getattr(args, "salience_mode", "delta") == "vjepa":
                cache_dir = resolve_path(getattr(args, "vjepa_cache_dir", "outputs/vjepa_salience_cache"))
                cache_path = os.path.join(cache_dir, os.path.basename(video_path).replace("/", "_") + ".pt")
                vjepa_grid = torch.load(cache_path, map_location="cpu")
            kept_indices, kept_embeds, diag = prune_video_tokens(
                model, video_embeds_full, video_grid_thw, args, vjepa_grid=vjepa_grid, budget=budget
            )
        if "kept" not in cache:
            diag = dict(diag)
            diag["n_total"] = n_total
            diag["t_pos_unique"] = int(position_ids[0, 0, video_positions_in_seq].unique().numel())
            cache["kept"] = (kept_indices, kept_embeds, diag)

        keep_seq_pos = video_positions_in_seq[kept_indices]  # original seq positions kept

        drop_mask = video_mask.clone()
        drop_mask[keep_seq_pos] = False  # True = drop this position from the sequence
        final_seq_mask = ~drop_mask  # keep everything except dropped video slots

        final_embeds = full_embeds[0][final_seq_mask].unsqueeze(0)
        final_embeds = final_embeds.masked_scatter(
            (inputs["input_ids"][0][final_seq_mask] == video_token_id).unsqueeze(0).unsqueeze(-1),
            kept_embeds.to(final_embeds.dtype),
        )
        final_position_ids = position_ids[:, :, final_seq_mask]
        final_attention_mask = inputs["attention_mask"][0][final_seq_mask].unsqueeze(0)

        generated_ids = model.generate(
            inputs_embeds=final_embeds,
            position_ids=final_position_ids,
            attention_mask=final_attention_mask,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
            do_sample=False,
            num_beams=1,
            max_new_tokens=16,
        )
    output = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    return output, diag


def run_inference(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    max_pixels_units = int(getattr(args, "max_pixels_units", 16384 * 3))
    max_pixels = max_pixels_units * 28 * 28

    video_dir = resolve_path(args.video_dir)
    gt_file = resolve_path(args.gt_file)
    output_dir = resolve_path(args.output_dir)

    print(f"Loading {args.model_path} ... (max_pixels_units={max_pixels_units})")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=dtype, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(
        args.model_path, max_pixels=max_pixels, min_pixels=MIN_PIXELS
    )

    with open(gt_file, "r") as f:
        gt_qa_pairs = json.load(f)
    gt_qa_pairs = get_chunk(gt_qa_pairs, args.num_chunks, args.chunk_idx)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{args.output_name}.json")
    generated_ids_seen = load_answered_ids(output_path)
    output_mode = "a" if os.path.exists(output_path) else "w"

    verify_mode = bool(getattr(args, "verify_mode", False))
    cache = {}
    stats_path = os.path.join(output_dir, f"{args.output_name}.stats.jsonl")

    with open(output_path, output_mode, encoding="utf-8") as ans_file, \
            open(stats_path, "a", encoding="utf-8") as stats_file:
        for sample in tqdm(gt_qa_pairs):
            answer_id = str(sample.get("id", sample.get("question_id")))
            if answer_id in generated_ids_seen:
                continue

            question = sample["question"]
            candidates = sample["candidates"]
            video_path = os.path.join(video_dir, sample["video_name"])
            if not os.path.exists(video_path):
                print(f"Missing video: {video_path}")
                continue

            try:
                output, diag = run_one_question(
                    model, processor, video_path, question, candidates, args, verify_mode, max_pixels,
                    cache=cache,
                )
            except Exception as e:
                # A small number of VideoMME videos fail to decode (decord
                # EOF-retry timeout, corrupt container, etc.) -- skip this
                # question rather than crash the whole run. Not marked as
                # answered, so a later rerun (same output_name) will retry it.
                print(f"SKIP {answer_id} ({video_path}): {type(e).__name__}: {e}")
                cache.clear()
                torch.cuda.empty_cache()
                continue
            print(f"pred={output!r} diag={diag}")
            stats_file.write(json.dumps({
                "id": answer_id, "video_name": sample["video_name"],
                **{k: diag.get(k) for k in ("mode", "n_total", "target_total", "actual_total", "n_segments", "t_pos_unique")},
            }) + "\n")
            stats_file.flush()

            sample_set = {
                "task_name": sample["task_name"],
                "question": question,
                "question_id": str(sample.get("question_id", answer_id)),
                "id": answer_id,
                "answer_number": sample["answer_number"],
                "candidates": candidates,
                "answer": sample["answer"],
                "pred": output,
            }
            ans_file.write(json.dumps(sample_set) + "\n")
            ans_file.flush()
            generated_ids_seen.add(answer_id)

    print(f"[Qwen2.5-VL DySeg+QTA+STPrune, harness-matched] done -> {output_path}")


@hydra.main(
    config_path="configs/inference_qwen25vl", config_name="dyseg", version_base=None
)
def main(cfg: DictConfig):
    run_inference(cfg)


if __name__ == "__main__":
    main()
