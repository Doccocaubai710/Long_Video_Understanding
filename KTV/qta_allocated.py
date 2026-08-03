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

TOKEN_LEVELS = [576, 288, 144, 72, 36, 18]   # 576 = full unpruned patch count/frame (CLIP ViT-L/14-336)


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


def cap_to_budget(pairs, B, protected=None):
    """Ep tong token thuc te (sau khi da chia deu trong tung clip) <= B.

    quantize() lam tron ve bac GAN NHAT (co the lam tron LEN), va moi frame
    trong mot clip deu duoc gan cung mot muc da luong tu hoa -> tong thuc te
    theo TUNG FRAME co the vuot xa ngan sach B ma allocate() nham toi (B chi
    dam bao dung cho tong theo ANCHOR, truoc khi chia/lam tron theo frame).
    Ham nay hoat dong tren dict {frame: token} cuoi cung, giam dan frame co
    budget lon nhat xuong bac ke tiep cho den khi tong <= B hoac moi frame
    da o san toi thieu.

    `protected`: tap frame KHONG duoc giam (vd frame thuoc top-M anchor duoc
    dam bao tier cao o allocate()) -- neu khong loai tru, cap_to_budget co
    the vo tinh xoa mat chinh cai loi hua top-M vua tao ra.
    """
    protected = protected or set()
    if not pairs or sum(pairs.values()) <= B:
        return pairs
    levels_desc = sorted(TOKEN_LEVELS, reverse=True)  # [576,288,144,72,36,18]
    total = sum(pairs.values())
    while total > B:
        candidates = {f: v for f, v in pairs.items() if f not in protected}
        if not candidates:
            break  # chi con frame duoc bao ve -> chap nhan vuot nhe, giu loi hua top-M
        f = max(candidates, key=candidates.get)
        cur = pairs[f]
        idx = levels_desc.index(cur)
        if idx == len(levels_desc) - 1:
            break  # moi frame khong-bao-ve da o muc toi thieu, khong the giam them
        new_val = levels_desc[idx + 1]
        total -= (cur - new_val)
        pairs[f] = new_val
    return pairs


def allocate(values, sims, clip_sizes, B=1872, b_min=18, alpha=1.0, beta=0.5, T=0.5,
             min_tok_per_frame=16, top_m=0, top_m_level=144):
    """Phan bo token cho tung clip roi chia deu trong clip.

    alpha=0 -> agnostic alloc (chi importance noi tai)
    beta =0 -> thuan query-aware (chi relevance)

    top_m > 0: DAM BAO CUNG top_m anchor (theo diem w cao nhat) nhan tier
    top_m_level TREN MOI FRAME cua clip do (khong chia nho qua softmax).
    Ly do: mem softmax du sac (T thap) van chia xac suat cho ca K doi thu;
    voi K lon (video dai), khong anchor nao "thang dam" du de vuot nguong
    quantize len tier cao -> moi thu bi keo ve san. top_m tai tao lai co
    che dam bao cua KTV cu (2 frame luon 288, 8 frame luon 144) nhung chon
    anchor bang diem so cua RECON+QTA thay vi rank CLIP co dinh.
    Phan ngan sach CON LAI (sau khi tru top_m) moi chia tiep bang softmax
    cho cac anchor khong duoc bao ve.

    Returns (out, protected_set): out[a] = token/frame cho anchor a,
    protected_set = chi so cac anchor duoc bao ve (de cap_to_budget khong
    vo tinh giam chung).
    """
    K = len(values)
    if K == 0:
        return [], set()

    values = np.asarray(values, dtype=np.float64)
    sims_arr = np.asarray(sims, dtype=np.float64) if sims is not None else None

    s_n_all = minmax(sims_arr) if sims_arr is not None else np.ones(K) * 0.5
    v_n_all = minmax(values)
    w_all = np.power(s_n_all + 1e-6, alpha) * np.power(v_n_all + 1e-6, beta)

    out = [0] * K
    protected = set()
    remaining_B = B

    m = min(max(0, int(top_m)), K)
    if m > 0:
        level = quantize(top_m_level)
        top_idx = np.argsort(-w_all)[:m]
        for a in top_idx:
            a = int(a)
            out[a] = level
            protected.add(a)
            remaining_B -= level * max(1, clip_sizes[a])

    rest_idx = [a for a in range(K) if a not in protected]
    Kr = len(rest_idx)
    if Kr == 0:
        return out, protected

    # san an toan: B_min khong duoc nuot het ngan sach con lai
    B_rest = max(Kr * min_tok_per_frame, remaining_B)
    b_min_r = int(max(0, min(b_min, B_rest // (2 * Kr))))

    # tinh lai minmax RIENG tren nhom con lai, de softmax phan biet tot giua
    # cac doi thu con canh tranh thay vi bi nen sat 0 boi cac anchor top-M
    # da bi loai (chung thuong la max cua toan bo tap).
    s_n = minmax(sims_arr[rest_idx]) if sims_arr is not None else np.ones(Kr) * 0.5
    v_n = minmax(values[rest_idx])
    w = np.power(s_n + 1e-6, alpha) * np.power(v_n + 1e-6, beta)

    z = w / max(T, 1e-6)
    z = z - z.max()
    p = np.exp(z)
    p = p / (p.sum() + 1e-12)

    free = max(0, B_rest - Kr * b_min_r)
    B_clip = b_min_r + free * p

    for i, a in enumerate(rest_idx):
        per = B_clip[i] / max(1, clip_sizes[a])
        per = max(min_tok_per_frame, per)
        out[a] = quantize(per)
    return out, protected


def video_budget(K, B_fixed, b_bar, b_min, b_max):
    """B cho mot video: co dinh (B_fixed) hoac thich ung theo K (Cach A, muc 7).

    b_bar > 0 -> B = clip(K * b_bar, K * b_min, b_max)  (thich ung theo noi dung,
    khong phai hang so; chan tren boi b_max = tran vat ly cua context window).
    b_bar <= 0 -> giu B_fixed (hanh vi cu, tuong thich nguoc).
    """
    if b_bar <= 0:
        return B_fixed
    B_video = int(K * b_bar)
    if b_max > 0:
        B_video = min(B_video, int(b_max))
    return max(B_video, K * b_min)


# ======================================================================
def run(json_path, stage1_dir, save_dir, combined_output_path,
        B=1872, b_min=18, alpha=1.0, beta=0.5, T=0.5,
        min_tok_per_frame=16, device="auto", use_query=True,
        b_bar=0, b_max=0, top_m=0, top_m_level=144):
    os.makedirs(save_dir, exist_ok=True)
    qa = json.load(open(json_path))

    cache = {}   # video_name -> (meta, emb)
    combined, missing, tok_stats = {}, 0, []
    b_video_stats = []

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

        B_video = video_budget(K, B, b_bar, b_min, b_max)
        b_video_stats.append(B_video)
        per_clip_tok, protected_anchors = allocate(
            values, sims, [len(c) for c in clips],
            B=B_video, b_min=b_min, alpha=alpha, beta=beta, T=T,
            min_tok_per_frame=min_tok_per_frame,
            top_m=top_m, top_m_level=top_m_level)

        # --- xuat [frame, token_budget] ---
        pairs = {}
        for a in range(K):
            for f in clips[a]:
                pairs[int(f)] = max(pairs.get(int(f), 0), int(per_clip_tok[a]))
        if not pairs:
            continue
        protected_frames = {int(f) for a in protected_anchors for f in clips[a]}
        pairs = cap_to_budget(pairs, B_video, protected=protected_frames)
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
        bv = np.asarray(b_video_stats)
        mode = f"adaptive b_bar={b_bar} b_max={b_max}" if b_bar > 0 else f"fixed B={B}"
        print(f"[Stage2] token budget mode: {mode}")
        print(f"[Stage2] target B/video        mean={bv.mean():.0f} med={np.median(bv):.0f} "
              f"min={bv.min():.0f} max={bv.max():.0f}")
        print(f"[Stage2] realized tokens/question  mean={t.mean():.0f} "
              f"med={np.median(t):.0f} min={t.min()} max={t.max()}")
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
        use_query=bool(g("use_query", True)),
        b_bar=g("b_bar", 0), b_max=g("b_max", 0),
        top_m=g("top_m", 0), top_m_level=g("top_m_level", 144))


if __name__ == "__main__":
    main()