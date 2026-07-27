"""
Phan tich cau truc cluster cua feature DINO tren VideoMME.
Muc tieu: kiem chung xem fix cung k=12 co phu hop khong, bang cach do
so cluster 'tu nhien' cua tung video theo nhieu tieu chi va nhieu thuat toan.

Moi output deu LUU RA FILE (chay tren ailab, khong can man hinh).
"""

import argparse
import collections
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend khong can display -> chi ghi file
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

import cluster_and_rank_keyframes as base

# Import cac thuat toan cua ban; neu thieu thi bo qua metric tuong ung.
try:
    from MaxInfo.maxinfo_clustering import maxinfo_clustering
    HAS_MAXINFO = True
except Exception as e:
    print(f"[warn] khong import duoc MaxInfo: {e}")
    HAS_MAXINFO = False

try:
    from event_boundary_clustering import event_boundary_clustering
    HAS_EVENT = True
except Exception as e:
    print(f"[warn] khong import duoc event_boundary: {e}")
    HAS_EVENT = False


# ==================================================================
# 1. METRIC CO BAN
# ==================================================================
def l2norm(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


def effective_rank(feats):
    """So chieu thong tin thuc = participation ratio cua singular values:
    (sum s)^2 / sum(s^2). Cao = video giau thong tin -> can nhieu cluster."""
    s = np.linalg.svd(np.asarray(feats, np.float64), compute_uv=False)
    s = s[s > 1e-8]
    if s.size == 0:
        return 0.0
    return float((s.sum() ** 2) / np.sum(s ** 2))


def optimal_k_silhouette(feats, k_min=2, k_max=30, n_init=3):
    """Tim k tot nhat theo silhouette (cao = cluster tach bach ro)."""
    n = feats.shape[0]
    k_max = min(k_max, n - 1)
    if k_max < k_min:
        return 1, {}
    Xn = l2norm(feats)
    scores = {}
    for k in range(k_min, k_max + 1):
        try:
            km = KMeans(n_clusters=k, random_state=0, n_init=n_init).fit(Xn)
            scores[k] = float(silhouette_score(Xn, km.labels_, metric="cosine"))
        except Exception:
            scores[k] = -1.0
    best_k = max(scores, key=scores.get) if scores else 1
    return best_k, scores


def kmeans_inertia_curve(feats, k_min=1, k_max=30, n_init=3):
    """Duong inertia de tim 'elbow'."""
    n = feats.shape[0]
    k_max = min(k_max, n)
    Xn = l2norm(feats)
    inertia = {}
    for k in range(k_min, k_max + 1):
        try:
            inertia[k] = float(KMeans(n_clusters=k, random_state=0, n_init=n_init).fit(Xn).inertia_)
        except Exception:
            inertia[k] = np.nan
    return inertia


def redundancy_stats(feats):
    """Do trung lap: sim frame-ke + ti le cap frame gan trung."""
    Xn = l2norm(feats)
    adj = np.sum(Xn[1:] * Xn[:-1], axis=1) if len(Xn) > 1 else np.array([])
    full = Xn @ Xn.T
    iu = np.triu_indices(len(Xn), k=1)
    pair = full[iu] if len(iu[0]) else np.array([])
    return {
        "adj_sim_mean": float(adj.mean()) if adj.size else 0.0,
        "pairwise_sim_mean": float(pair.mean()) if pair.size else 0.0,
        "frac_highly_similar": float((pair > 0.9).mean()) if pair.size else 0.0,
        "adj_sim_curve": adj,
    }


def safe_maxinfo(feats):
    if not HAS_MAXINFO:
        return []
    try:
        return list(maxinfo_clustering(feats, num_keyframes=64, verbose=False))
    except Exception:
        return []


def safe_event(feats):
    if not HAS_EVENT:
        return []
    try:
        return list(event_boundary_clustering(feats, verbose=False))
    except Exception:
        return []


# ==================================================================
# 2. VISUALIZE 1 VIDEO -> 1 file PNG
# ==================================================================
def visualize_one_video(video_name, feats, out_dir, k_fixed=12, k_max=30):
    feats = np.asarray(feats, np.float32)
    n = feats.shape[0]
    Xn = l2norm(feats)

    best_k, sil = optimal_k_silhouette(feats, k_max=k_max)
    inertia = kmeans_inertia_curve(feats, k_max=k_max)
    red = redundancy_stats(feats)
    eff = effective_rank(feats)
    maxinfo_idx = safe_maxinfo(feats)
    event_idx = safe_event(feats)

    fig, ax = plt.subplots(2, 3, figsize=(18, 10))

    # (1) heatmap similarity
    sim = Xn @ Xn.T
    im = ax[0, 0].imshow(sim, cmap="viridis", aspect="auto")
    ax[0, 0].set_title(f"Frame-frame cosine sim (n={n})")
    ax[0, 0].set_xlabel("frame"); ax[0, 0].set_ylabel("frame")
    fig.colorbar(im, ax=ax[0, 0], fraction=0.046)

    # (2) adjacent-frame similarity (valley = scene cut)
    ax[0, 1].plot(red["adj_sim_curve"])
    ax[0, 1].set_title("Adjacent-frame similarity (valley=cut)")
    ax[0, 1].set_xlabel("frame"); ax[0, 1].set_ylabel("cosine")

    # (3) silhouette vs k
    if sil:
        ks = sorted(sil.keys())
        ax[0, 2].plot(ks, [sil[k] for k in ks], marker="o", ms=3)
        ax[0, 2].axvline(best_k, color="g", ls="--", label=f"best k={best_k}")
        ax[0, 2].axvline(k_fixed, color="r", ls=":", label=f"fixed k={k_fixed}")
        ax[0, 2].legend()
    ax[0, 2].set_title("Silhouette vs k (cao=cluster tot)")
    ax[0, 2].set_xlabel("k")

    # (4) elbow (inertia)
    if inertia:
        kk = sorted(inertia.keys())
        ax[1, 0].plot(kk, [inertia[k] for k in kk], marker="o", ms=3)
        ax[1, 0].axvline(k_fixed, color="r", ls=":", label=f"fixed k={k_fixed}")
        ax[1, 0].legend()
    ax[1, 0].set_title("KMeans inertia (elbow)")
    ax[1, 0].set_xlabel("k"); ax[1, 0].set_ylabel("inertia")

    # (5) PCA 2D, mau theo thoi gian, overlay keyframe cua 3 method
    p = PCA(n_components=2).fit_transform(Xn)
    sc = ax[1, 1].scatter(p[:, 0], p[:, 1], c=np.arange(n), cmap="rainbow", s=12, alpha=0.5)
    if maxinfo_idx:
        mi = [i for i in maxinfo_idx if 0 <= i < n]
        ax[1, 1].scatter(p[mi, 0], p[mi, 1], marker="*", s=180,
                         edgecolor="k", facecolor="none", label=f"MaxInfo ({len(mi)})")
    if event_idx:
        ev = [i for i in event_idx if 0 <= i < n]
        ax[1, 1].scatter(p[ev, 0], p[ev, 1], marker="s", s=90,
                         edgecolor="blue", facecolor="none", label=f"Event ({len(ev)})")
    km12 = KMeans(n_clusters=min(k_fixed, n), random_state=0, n_init=3).fit(Xn)
    c12 = [int(np.argmin(np.linalg.norm(Xn - km12.cluster_centers_[c], axis=1)))
           for c in range(min(k_fixed, n))]
    ax[1, 1].scatter(p[c12, 0], p[c12, 1], marker="x", s=90, color="red",
                     label=f"KMeans k={k_fixed}")
    ax[1, 1].set_title("PCA(2D) theo thoi gian + keyframe da chon")
    ax[1, 1].legend(fontsize=8)
    fig.colorbar(sc, ax=ax[1, 1], fraction=0.046, label="time")

    # (6) text summary
    ax[1, 2].axis("off")
    hop_ly = "CO" if abs(best_k - k_fixed) <= 3 else "KHONG"
    txt = (
        f"video: {video_name}\n"
        f"n frames (feat): {n}\n"
        f"effective rank: {eff:.1f}\n"
        f"best k (silhouette): {best_k}\n"
        f"fixed k: {k_fixed}\n"
        f"MaxInfo picks: {len(maxinfo_idx)}\n"
        f"Event picks: {len(event_idx)}\n\n"
        f"adj_sim_mean: {red['adj_sim_mean']:.3f}\n"
        f"pairwise_sim_mean: {red['pairwise_sim_mean']:.3f}\n"
        f"frac_sim>0.9: {red['frac_highly_similar']:.2%}\n\n"
        f"=> fixed k={k_fixed} phu hop? {hop_ly}"
    )
    ax[1, 2].text(0.02, 0.98, txt, va="top", family="monospace", fontsize=11)

    fig.suptitle(f"Cluster analysis: {video_name}", fontsize=14)
    fig.tight_layout()
    safe = video_name.replace("/", "_").replace(".", "_")
    fig.savefig(os.path.join(out_dir, f"video_{safe}.png"), dpi=110)
    plt.close(fig)

    return {
        "video_name": video_name, "n_frames": n, "effective_rank": eff,
        "best_k_silhouette": best_k, "maxinfo_picks": len(maxinfo_idx),
        "event_picks": len(event_idx),
        "adj_sim_mean": red["adj_sim_mean"],
        "pairwise_sim_mean": red["pairwise_sim_mean"],
        "frac_highly_similar": red["frac_highly_similar"],
    }


def metrics_only(video_name, feats, k_fixed=12, k_max=30):
    """Chi tinh so lieu, khong ve PNG (nhanh hon nhieu)."""
    feats = np.asarray(feats, np.float32)
    best_k, _ = optimal_k_silhouette(feats, k_max=k_max)
    red = redundancy_stats(feats)
    return {
        "video_name": video_name, "n_frames": feats.shape[0],
        "effective_rank": effective_rank(feats),
        "best_k_silhouette": best_k,
        "maxinfo_picks": len(safe_maxinfo(feats)),
        "event_picks": len(safe_event(feats)),
        "adj_sim_mean": red["adj_sim_mean"],
        "pairwise_sim_mean": red["pairwise_sim_mean"],
        "frac_highly_similar": red["frac_highly_similar"],
    }


# ==================================================================
# 3. TONG HOP TOAN TAP
# ==================================================================
CSV_KEYS = ["video_name", "bucket", "n_frames", "effective_rank",
            "best_k_silhouette", "maxinfo_picks", "event_picks",
            "adj_sim_mean", "pairwise_sim_mean", "frac_highly_similar"]


def write_csv(rows, out_dir):
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_KEYS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_KEYS})


def dataset_summary(rows, out_dir, k_fixed=12):
    best_k = np.array([r["best_k_silhouette"] for r in rows])
    n_fr = np.array([r["n_frames"] for r in rows])
    eff = np.array([r["effective_rank"] for r in rows])
    mi = np.array([r["maxinfo_picks"] for r in rows])

    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    ax[0, 0].hist(best_k, bins=range(2, 45), color="steelblue", edgecolor="k")
    ax[0, 0].axvline(k_fixed, color="r", ls="--", label=f"fixed k={k_fixed}")
    ax[0, 0].set_title("Phan bo 'best k' (silhouette) toan tap")
    ax[0, 0].set_xlabel("optimal k"); ax[0, 0].legend()

    ax[0, 1].scatter(n_fr, best_k, s=14, alpha=0.5)
    ax[0, 1].axhline(k_fixed, color="r", ls="--")
    ax[0, 1].set_title("optimal k vs so frame video")
    ax[0, 1].set_xlabel("n frames"); ax[0, 1].set_ylabel("optimal k")

    ax[1, 0].hist(eff, bins=30, color="seagreen", edgecolor="k")
    ax[1, 0].axvline(k_fixed, color="r", ls="--", label=f"fixed k={k_fixed}")
    ax[1, 0].set_title("Phan bo effective rank")
    ax[1, 0].set_xlabel("eff. rank"); ax[1, 0].legend()

    ax[1, 1].scatter(n_fr, mi, s=14, alpha=0.5)
    ax[1, 1].axhline(k_fixed, color="r", ls="--", label=f"fixed k={k_fixed}")
    ax[1, 1].set_title("MaxInfo adaptive picks vs so frame")
    ax[1, 1].set_xlabel("n frames"); ax[1, 1].set_ylabel("picks"); ax[1, 1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "DATASET_summary.png"), dpi=120)
    plt.close(fig)


def bucket_summary(rows, out_dir, k_fixed=12):
    order = ["short", "medium", "long"]
    g = collections.defaultdict(list)
    for r in rows:
        g[r.get("bucket", "unknown")].append(r)

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for j, (key, title) in enumerate([("best_k_silhouette", "Optimal k"),
                                      ("effective_rank", "Effective rank"),
                                      ("maxinfo_picks", "MaxInfo picks")]):
        data = [[r[key] for r in g[b]] for b in order if g[b]]
        labels = [b for b in order if g[b]]
        if data:
            ax[j].boxplot(data, labels=labels, showmeans=True)
        ax[j].axhline(k_fixed, color="r", ls="--", label=f"fixed k={k_fixed}")
        ax[j].set_title(f"{title} theo nhom do dai"); ax[j].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "BUCKET_summary.png"), dpi=120)
    plt.close(fig)

    lines = ["===== THEO NHOM DO DAI (short/medium/long) ====="]
    for b in order:
        if not g[b]:
            continue
        bk = np.array([r["best_k_silhouette"] for r in g[b]])
        mi = np.array([r["maxinfo_picks"] for r in g[b]])
        nf = np.array([r["n_frames"] for r in g[b]])
        lines.append(
            f"[{b:6s}] videos={len(g[b]):3d}  n_frames_med={np.median(nf):.0f}  "
            f"best_k_med={np.median(bk):.0f} (min{bk.min()}-max{bk.max()})  "
            f"maxinfo_med={np.median(mi):.0f}")
    txt = "\n".join(lines)
    with open(os.path.join(out_dir, "BUCKET_summary.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)


def save_txt_summary(rows, out_dir, k_fixed=12):
    best_k = np.array([r["best_k_silhouette"] for r in rows])
    eff = np.array([r["effective_rank"] for r in rows])
    mi = np.array([r["maxinfo_picks"] for r in rows])
    ev = np.array([r["event_picks"] for r in rows])
    n_fr = np.array([r["n_frames"] for r in rows])
    within = float(np.mean(np.abs(best_k - k_fixed) <= 3)) if len(rows) else 0.0
    lines = [
        "===== DATASET CLUSTER ANALYSIS SUMMARY =====",
        f"videos analyzed            : {len(rows)}",
        f"n_frames  mean/med/min/max : {n_fr.mean():.1f} / {np.median(n_fr):.0f} / {n_fr.min()} / {n_fr.max()}",
        f"best_k    mean/med/min/max : {best_k.mean():.1f} / {np.median(best_k):.0f} / {best_k.min()} / {best_k.max()}",
        f"eff.rank  mean/med/min/max : {eff.mean():.1f} / {np.median(eff):.0f} / {eff.min():.0f} / {eff.max():.0f}",
        f"MaxInfo picks mean/med     : {mi.mean():.1f} / {np.median(mi):.0f}",
        f"Event picks   mean/med     : {ev.mean():.1f} / {np.median(ev):.0f}",
        f"fixed k                    : {k_fixed}",
        f"videos where |best_k-{k_fixed}|<=3 : {within:.1%}",
        "",
        "KET LUAN: neu ti le tren THAP va best_k trai rong -> fixed k KHONG phu hop.",
    ]
    txt = "\n".join(lines)
    with open(os.path.join(out_dir, "SUMMARY.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)


# ==================================================================
# 4. CHON VIDEO THEO NHOM short/medium/long
# ==================================================================
def bucket_of(vid_num):
    if 1 <= vid_num <= 300:
        return "short"
    if 301 <= vid_num <= 600:
        return "medium"
    if 601 <= vid_num <= 900:
        return "long"
    return "unknown"


def pick_videos(json_path, per_bucket, buckets):
    """VideoMME chuan: 900 video, 001-300 short / 301-600 medium / 601-900 long.
    Lay so prefix cua question_id lam chi so video."""
    qa = json.load(open(json_path))
    seen = set()
    per = collections.defaultdict(list)
    for s in qa:
        v = s["video_name"]
        if v in seen:
            continue
        seen.add(v)
        try:
            num = int(str(s["question_id"]).split("-")[0])
        except Exception:
            continue
        b = bucket_of(num)
        if b in buckets:
            per[b].append(v)

    video_list, tag = [], {}
    for b in buckets:
        chosen = per[b] if per_bucket <= 0 else per[b][:per_bucket]
        for v in chosen:
            video_list.append(v)
            tag[v] = b
    return video_list, tag, {b: len(per[b]) for b in buckets}


# ==================================================================
# 5. MAIN
# ==================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor_path", default="ktv/save_tensor/Video-MME")
    ap.add_argument("--json_path", default="playground/gt_qa_files/Videomme/val_qa.json")
    ap.add_argument("--out_dir", default="outputs/cluster_analysis")
    ap.add_argument("--per_bucket", type=int, default=10,
                    help="so video moi nhom (0 = lay het moi nhom)")
    ap.add_argument("--buckets", default="short,medium,long",
                    help="nhom can phan tich, vd 'long' hoac 'short,long'")
    ap.add_argument("--per_video_plots", type=int, default=1,
                    help="1 = ve PNG chi tiet moi video; 0 = chi tinh metric")
    ap.add_argument("--k_fixed", type=int, default=12)
    ap.add_argument("--k_max", type=int, default=30,
                    help="k toi da khi quet silhouette/elbow (giam de chay nhanh)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    per_video_dir = os.path.join(args.out_dir, "per_video")
    os.makedirs(per_video_dir, exist_ok=True)

    buckets = [b.strip() for b in args.buckets.split(",") if b.strip()]
    print(f"[load] tensor tu {args.tensor_path}")
    vft = base.load_video_frame_tensor(args.tensor_path)

    video_list, tag, avail = pick_videos(args.json_path, args.per_bucket, buckets)
    print(f"[info] co san moi nhom: {avail}")
    print(f"[info] se phan tich {len(video_list)} video ({buckets}, per_bucket={args.per_bucket})")

    rows, missing = [], 0
    for i, vname in enumerate(video_list):
        tensor = base.get_tensor_for_video(vft, vname)
        if tensor is None:
            missing += 1
            print(f"  [skip] khong co tensor: {vname}")
            continue
        feats = np.asarray(tensor, dtype=np.float32)
        if feats.ndim != 2 or feats.shape[0] < 4:
            print(f"  [skip] tensor bat thuong {vname}: shape={feats.shape}")
            continue
        try:
            if args.per_video_plots:
                row = visualize_one_video(vname, feats, per_video_dir,
                                          k_fixed=args.k_fixed, k_max=args.k_max)
            else:
                row = metrics_only(vname, feats, k_fixed=args.k_fixed, k_max=args.k_max)
            row["bucket"] = tag.get(vname, "unknown")
            rows.append(row)
        except Exception as e:
            print(f"  [error] {vname}: {e}")
        if (i + 1) % 5 == 0:
            print(f"  ...{i + 1}/{len(video_list)} done")

    if not rows:
        print("[fatal] khong phan tich duoc video nao. Kiem tra tensor_path/json_path.")
        return

    write_csv(rows, args.out_dir)
    dataset_summary(rows, args.out_dir, k_fixed=args.k_fixed)
    bucket_summary(rows, args.out_dir, k_fixed=args.k_fixed)
    save_txt_summary(rows, args.out_dir, k_fixed=args.k_fixed)

    print(f"\n[done] missing_tensor={missing}. Ket qua tai: {args.out_dir}")
    print("  per_video/*.png      : phan tich chi tiet moi video")
    print("  DATASET_summary.png  : phan bo toan tap")
    print("  BUCKET_summary.png/.txt : so sanh theo short/medium/long")
    print("  summary.csv          : bang so lieu day du")
    print("  SUMMARY.txt          : ket luan van ban")


if __name__ == "__main__":
    main()