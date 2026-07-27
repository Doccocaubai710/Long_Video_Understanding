"""
TANG 2 - QTA (Query-Aware Token Allocation).

Dung lai tap anchor/clip da chon o Tang 1 (query-agnostic), chi PHAN BO LAI
ngan sach token cho tung cau hoi:

  s_a = cos(CLIP_img(f_a), CLIP_txt(Q))          # relevance voi cau hoi
  w_a = s_norm^alpha * v_norm^beta               # ket hop relevance + importance noi tai
  B_a = B_min + (B - K*B_min) * softmax(w_a / T) # phan bo hinh chop, co san
  token/frame = B_a / |clip(a)|                  # chia budget kieu F2C

Rat re: CLIP image-embed da cache o Tang 1, moi cau hoi chi encode TEXT.
Xuat {question_id: [[frame_index, token_budget], ...]}.
"""

import json
import os

import clip
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

import cluster_and_rank_keyframes as base

TOKEN_LEVELS = [288, 144, 72, 36, 18]   # khop nhanh token cua llava_arch


# ======================================================================
def minmax(v, eps=1e-8):
    v = np.asarray(v, dtype=np.float64)
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < eps:
        return np.ones_like(v) * 0.5
    return (v - lo) / (hi - lo + eps)


def build_query(sample):
    """Gop cau hoi + cac phuong an (MCQA) lam text query."""
    return f"{sample.get('question','')} {sample.get('candidates','')}".strip()


def encode_text(query, device):
    base.load_clip_model(device)
    words = query.split()
    if len(words) > 60:                     # tranh cat mat phuong an
        query = " ".join(words[:60])
    with torch.no_grad():
        t = base.model_clip.encode_text(clip.tokenize([query], truncate=True).to(base.device))
        t = t / t.norm(dim=-1, keepdim=True)
    return t.float().cpu().numpy()[0]


def quantize(b):
    """Lam tron ve bac token gan nhat (khop cls_new_token_sim)."""
    return int(min(TOKEN_LEVELS, key=lambda L: abs(L - b)))


def allocate(values, sims, clip_sizes, B=1872, b_min=18, alpha=1.0, beta=0.5, T=0.5,
             min_tok_per_frame=16):
    """Phan bo token cho tung clip roi chia deu trong clip.

    alpha=0 -> agnostic alloc (chi importance noi tai)
    beta =0 -> thuan query-aware (chi relevance)
    """
    K = len(values)
    if K == 0:
        return []
    # san an toan: B_min khong duoc nuot het ngan sach
    b_min = int(max(0, min(b_min, B // (2 * K)))) if K > 0 else 0

    s_n = minmax(sims) if sims is not None else np.ones(K) * 0.5
    v_n = minmax(values)
    # +eps de tranh 0^alpha triet tieu hoan toan
    w = np.power(s_n + 1e-6, alpha) * np.power(v_n + 1e-6, beta)

    z = w / max(T, 1e-6)
    z = z - z.max()
    p = np.exp(z)
    p = p / (p.sum() + 1e-12)

    free = max(0, B - K * b_min)
    B_clip = b_min + free * p

    out = []
    for a in range(K):
        per = B_clip[a] / max(1, clip_sizes[a])
        per = max(min_tok_per_frame, per)
        out.append(quantize(per))
    return out


# ======================================================================
def run(json_path, stage1_dir, save_dir, combined_output_path,
        B=1872, b_min=18, alpha=1.0, beta=0.5, T=0.5,
        min_tok_per_frame=16, device="auto", use_query=True):
    os.makedirs(save_dir, exist_ok=True)
    qa = json.load(open(json_path))

    cache = {}   # video_name -> (meta, emb)
    combined, missing, tok_stats = {}, 0, []

    for sample in tqdm(qa, total=len(qa), desc="Stage2 QTA"):
        vname, qid = sample["video_name"], sample["question_id"]
        out_path = os.path.join(save_dir, f"{qid}.json")
        if os.path.exists(out_path):
            combined.update(json.load(open(out_path)))
            continue

        # --- nap cache Tang 1 (dung chung cho moi cau hoi cua video) ---
        if vname not in cache:
            meta_p = os.path.join(stage1_dir, f"{vname}.json")
            if not os.path.exists(meta_p):
                missing += 1
                cache[vname] = (None, None)
            else:
                meta = json.load(open(meta_p))
                emb_p = os.path.join(stage1_dir, f"{vname}.npz")
                emb = np.load(emb_p)["emb"] if os.path.exists(emb_p) else None
                cache[vname] = (meta, emb)
        meta, emb = cache[vname]
        if meta is None:
            continue

        values = meta["values"]
        clips = meta["clips_orig"]
        K = len(values)

        # --- relevance theo cau hoi ---
        sims = None
        if use_query and emb is not None and emb.shape[0] == K:
            t = encode_text(build_query(sample), device)
            sims = emb @ t                      # cosine (ca hai da normalize)
            if not np.isfinite(sims).all() or float(np.abs(sims).sum()) == 0.0:
                sims = None                     # CLIP loi -> fallback agnostic

        per_clip_tok = allocate(values, sims, [len(c) for c in clips],
                                B=B, b_min=b_min, alpha=alpha, beta=beta, T=T,
                                min_tok_per_frame=min_tok_per_frame)

        # --- xuat [frame, token_budget] ---
        pairs = {}
        for a in range(K):
            for f in clips[a]:
                pairs[int(f)] = max(pairs.get(int(f), 0), int(per_clip_tok[a]))
        if not pairs:
            continue
        key_frame_order = [[f, b] for f, b in sorted(pairs.items())]
        tok_stats.append(sum(b for _, b in key_frame_order))

        result = {qid: key_frame_order}
        combined.update(result)
        json.dump(result, open(out_path, "w"), indent=2)

    if combined_output_path:
        d = os.path.dirname(combined_output_path)
        if d:
            os.makedirs(d, exist_ok=True)
        json.dump(combined, open(combined_output_path, "w"), indent=2)

    if tok_stats:
        t = np.asarray(tok_stats)
        print(f"[Stage2] total tokens/question  mean={t.mean():.0f} "
              f"med={np.median(t):.0f} min={t.min()} max={t.max()} (target B={B})")
    print(f"[Stage2] done. questions={len(combined)} missing_stage1={missing} "
          f"-> {combined_output_path}")


@hydra.main(config_path="configs/keyframe_cluster", config_name="config", version_base=None)
def main(cfg: DictConfig):
    g = lambda k, d: getattr(cfg, k, d)
    run(base.resolve_path(cfg.json_path),
        base.resolve_path(g("stage1_dir", "outputs/casqta_stage1")),
        base.resolve_path(cfg.save_cluster_path),
        base.resolve_path(cfg.combined_output_path),
        B=g("tokens_num", 1872), b_min=g("b_min", 18),
        alpha=g("alpha", 1.0), beta=g("beta", 0.5), T=g("temperature", 0.5),
        min_tok_per_frame=g("min_tok_per_frame", 16),
        device=g("device", "auto"),
        use_query=bool(g("use_query", True)))


if __name__ == "__main__":
    main()