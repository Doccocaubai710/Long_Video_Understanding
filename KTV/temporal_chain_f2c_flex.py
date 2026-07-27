import json
import os

import cv2
import clip
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm

import cluster_and_rank_keyframes as base


# ====================== shared helpers ======================
def minmax_normalize(values, eps=1e-8):
    v = np.asarray(values, dtype=np.float32)
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < eps:
        return np.zeros_like(v, dtype=np.float32)
    return (v - lo) / (hi - lo + eps)


def compute_event_score(feats, k):
    n = feats.shape[0]
    d = max(1, n // max(1, 10 * k))
    s = np.zeros(n, dtype=np.float32)
    for t in range(n):
        l = max(0, t - d); r = min(n - 1, t + d)
        s[t] = np.linalg.norm(feats[t] - feats[l]) + np.linalg.norm(feats[r] - feats[t])
    return s


def temporal_chain_select(feats, num_clips=12, lambda_event=0.5, alpha_gap=0.6, beta_redundancy=0.8):
    feats = np.asarray(feats, dtype=np.float32)
    n = feats.shape[0]
    if n == 0:
        return [], np.zeros(0, dtype=np.float32)
    if n <= num_clips:
        return list(range(n)), compute_event_score(feats, max(1, n))
    k = min(num_clips, n)
    km = base.KMeans(n_clusters=k, random_state=0, init="k-means++", n_init=10).fit(feats)
    r_cluster = -np.linalg.norm(feats - km.cluster_centers_[km.labels_], axis=1)
    r_event = compute_event_score(feats, k)
    norm = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    sel, rem = [], set(range(n))
    first = int(np.argmax(minmax_normalize(r_cluster) + lambda_event * minmax_normalize(r_event)))
    sel.append(first); rem.remove(first)
    while len(sel) < k and rem:
        cand = np.array(sorted(rem), dtype=np.int32)
        cc = minmax_normalize(r_cluster[cand]); ce = minmax_normalize(r_event[cand])
        cg = minmax_normalize(np.array([min(abs(int(t) - int(s)) for s in sel) for t in cand], dtype=np.float32))
        cr = minmax_normalize(np.max(norm[cand] @ norm[np.array(sel, dtype=np.int32)].T, axis=1))
        sc = cc + lambda_event * ce + alpha_gap * cg - beta_redundancy * cr
        b = int(cand[int(np.argmax(sc))]); sel.append(b); rem.remove(b)
    return sel, r_event


def f2c_centered_window(center, length, n):
    if n <= 0:
        return []
    length = min(length, n)
    start = max(0, min(center - (length - 1) // 2, n - length))
    return [start + i for i in range(length)]


def adaptive_clip_length(center, feats, r_norm, n, l_max, lambda_r=0.5, lambda_l=0.05):
    norm = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    best_l, best = 1, -1e30
    for l in range(1, l_max + 1):
        win = f2c_centered_window(center, l, n)
        if len(win) < l:
            break
        s_rel = float(np.mean(r_norm[win]))
        if l > 1:
            sub = norm[win]; sim = sub @ sub.T; np.fill_diagonal(sim, 0.0)
            r_red = float(sim.sum() / (l * (l - 1)))
        else:
            r_red = 0.0
        sc = s_rel - lambda_r * r_red + lambda_l * (l / l_max)
        if sc > best:
            best, best_l = sc, l
    return best_l


def build_clips(anchors, n, fixed_clip_len, r_event, feats=None, l_max=4, lambda_r=0.5, lambda_l=0.05):
    """fixed_clip_len>=1 -> ep cung do dai; =0 (hoac None) -> adaptive Eq.8."""
    r_norm = minmax_normalize(r_event)
    windows = {}
    for a in anchors:
        if fixed_clip_len and fixed_clip_len >= 1:
            l = min(int(fixed_clip_len), n)
        else:
            l = adaptive_clip_length(a, feats, r_norm, n, l_max, lambda_r, lambda_l)
        windows[a] = set(f2c_centered_window(a, l, n))
    allf = set().union(*windows.values()) if windows else set()
    clips = {a: [] for a in anchors}
    for f in allf:
        owners = [a for a in anchors if f in windows[a]]
        owner = min(owners, key=lambda a: (abs(f - a), -float(r_event[a])))
        clips[owner].append(f)
    return {a: sorted(fs) for a, fs in clips.items() if fs}


def clip_budget_tiers(num_clips, tokens_num):
    sc = tokens_num / 1872.0
    nh = max(1, round(num_clips / 6)); nl = max(1, round(num_clips / 6))
    nm = max(0, num_clips - nh - nl)
    return [int(round(288 * sc))] * nh + [int(round(144 * sc))] * nm + [int(round(72 * sc))] * nl


def read_frames_as_pil(video_path, orig_indices):
    cap = cv2.VideoCapture(video_path); out = []
    for idx in orig_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, img = cap.read()
        out.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) if ok else None)
    cap.release()
    return out


def build_prompt(sample):
    return f"{sample.get('question','')} {sample.get('candidates','')}".strip()


def clip_relevance(anchor_orig_frames, video_path, prompt, device):
    """Query-aware: cosine(CLIP image, CLIP text). Tra ve score theo thu tu anchor."""
    base.load_clip_model(device)
    pil = read_frames_as_pil(video_path, anchor_orig_frames)
    valid = [(i, im) for i, im in enumerate(pil) if im is not None]
    scores = np.zeros(len(anchor_orig_frames), dtype=np.float32)
    if not valid:
        return scores
    words = prompt.split()
    if len(words) > 40:
        prompt = " ".join(words[10:50])
    batch = torch.stack([base.preprocess_clip(im) for _, im in valid]).to(base.device)
    with torch.no_grad():
        imf = base.model_clip.encode_image(batch)
        tef = base.model_clip.encode_text(clip.tokenize([prompt], truncate=True).to(base.device))
        imf = imf / imf.norm(dim=-1, keepdim=True)
        tef = tef / tef.norm(dim=-1, keepdim=True)
        sim = (imf @ tef.T).squeeze(-1).float().cpu().numpy()
    for (i, _), s in zip(valid, np.atleast_1d(sim)):
        scores[i] = float(s)
    return scores


def clips_to_order(clip_frames, anchor_score, num_clips, tokens_num,
                   total_frames, fps, n, max_frames_to_extract, min_tokens_per_frame):
    """anchor_score[anchor] cao -> clip nhan budget cao (288). per_frame = budget/len."""
    budgets = clip_budget_tiers(num_clips, tokens_num)
    clips_sorted = sorted(clip_frames.items(), key=lambda kv: -float(anchor_score[kv[0]]))
    fb = {}
    for i, (a, frames) in enumerate(clips_sorted):
        cb = budgets[min(i, len(budgets) - 1)]
        per = max(min_tokens_per_frame, cb // max(1, len(frames)))
        for f in frames:
            fb[f] = max(fb.get(f, 0), per)
    num_ext = min(total_frames, max_frames_to_extract) if total_frames and total_frames > 0 else n
    pairs = {}
    for f in sorted(fb.keys()):
        if f >= num_ext:
            continue
        try:
            o = base.get_original_frame_number(
                total_frames if total_frames and total_frames > 0 else num_ext,
                f, fps=fps, max_frames_to_extract=max_frames_to_extract)
        except ValueError:
            continue
        pairs[int(o)] = max(pairs.get(int(o), 0), int(fb[f]))
    return [[fi, b] for fi, b in sorted(pairs.items())]


# ====================== pipeline ======================
def cluster(json_path, video_path, video_frame_tensor_path, save_cluster_path, dataset,
            combined_output_path=None, num_clips=12, fixed_clip_len=1, query_aware=False,
            l_max=4, tokens_num=1872, lambda_event=0.5, alpha_gap=0.6, beta_redundancy=0.8,
            lambda_r=0.5, lambda_l=0.05, min_tokens_per_frame=16, max_frames_to_extract=5400,
            device="auto"):
    os.makedirs(save_cluster_path, exist_ok=True)
    vft = base.load_video_frame_tensor(video_frame_tensor_path)
    qa = json.load(open(json_path))
    combined, missing = {}, 0
    stage1_cache = {}  # video_name -> (anchors, r_event, n, total_frames, fps)

    for sample in tqdm(qa, total=len(qa)):
        vname, qid = sample["video_name"], sample["question_id"]
        out_path = os.path.join(save_cluster_path, f"{qid}.json")
        if os.path.exists(out_path):
            combined.update(json.load(open(out_path)))
            continue

        # ---- Stage 1 (per video, query-agnostic) ----
        if vname in stage1_cache:
            anchors, r_event, n, total_frames, fps = stage1_cache[vname]
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
            anchors, r_event = temporal_chain_select(
                tensor, num_clips=num_clips, lambda_event=lambda_event,
                alpha_gap=alpha_gap, beta_redundancy=beta_redundancy)
            stage1_cache[vname] = (anchors, r_event, n, total_frames, fps)
            stage1_cache[vname + "_feat"] = tensor  # keep for adaptive length if needed

        feats = stage1_cache.get(vname + "_feat")
        clip_frames = build_clips(anchors, n, fixed_clip_len, r_event,
                                  feats=feats, l_max=l_max, lambda_r=lambda_r, lambda_l=lambda_l)

        # ---- Stage 2: choose anchor importance score ----
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
            if float(np.sum(list(anchor_score.values()))) == 0.0:  # CLIP fail -> fallback
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
    print(f"Done. mode=(clip_len={fixed_clip_len}, query_aware={query_aware}) "
          f"missing={missing} total={len(qa)}")


@hydra.main(config_path="configs/keyframe_cluster", config_name="config", version_base=None)
def main(cfg: DictConfig):
    g = lambda k, d: getattr(cfg, k, d)
    cluster(base.resolve_path(cfg.json_path), base.resolve_path(cfg.video_path),
            base.resolve_path(cfg.video_frame_tensor_path), base.resolve_path(cfg.save_cluster_path),
            cfg.dataset, combined_output_path=base.resolve_path(cfg.combined_output_path),
            num_clips=g("num_clips", 12), fixed_clip_len=g("fixed_clip_len", 1),
            query_aware=g("query_aware", False), l_max=g("l_max", 4), tokens_num=g("tokens_num", 1872),
            lambda_event=g("lambda_event", 0.5), alpha_gap=g("alpha_gap", 0.6),
            beta_redundancy=g("beta_redundancy", 0.8), lambda_r=g("lambda_r", 0.5),
            lambda_l=g("lambda_l", 0.05), min_tokens_per_frame=g("min_tokens_per_frame", 16),
            max_frames_to_extract=g("max_frames_to_extract", 5400), device=g("device", "auto"))


if __name__ == "__main__":
    main()