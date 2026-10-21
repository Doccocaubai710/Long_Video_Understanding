"""
analyze_pipeline_by_bucket.py

Phan tich output THUC TE cua RECON (Stage 1) + QTA (Stage 2) tren toan bo
dataset, tach theo short/medium/long -- khac voi analyze_videomme_clusters.py
(chi phan tich feature DINO tho tren mau nho, khong dung ket qua pipeline
that). Khong can VLM/GPU, chi doc lai cac file JSON pipeline da tao ra.

Tra loi cau hoi: "long" van thua co phai vi (a) K (so anchor) khong tang
du theo do dai, (b) clip expansion van lam bung no frame nhieu hon o long,
hay (c) budget/frame van bi bop mong hon o long -- du K da dung.

Usage:
  python analyze_pipeline_by_bucket.py \
      --stage1_dir outputs/casqta_stage1 \
      --stage2_json outputs/vmme_casqta.json \
      --csv playground/gt_qa_files/VideoMME/val_qa.csv
"""

import argparse
import csv
import glob
import json
import os
import statistics as s
from collections import defaultdict


def load_video_bucket(csv_path):
    """videoID (vd 'fFjv93ACGo8') -> duration bucket ('short'/'medium'/'long')."""
    m = {}
    for r in csv.DictReader(open(csv_path)):
        m[r["videoID"]] = r["duration"]
    return m


def load_qid_bucket(csv_path):
    """question_id -> duration bucket."""
    m = {}
    for r in csv.DictReader(open(csv_path)):
        m[r["question_id"]] = r["duration"]
    return m


def stem(video_name):
    return os.path.splitext(video_name)[0]


def print_bucket_table(title, data_by_bucket, fields):
    print(f"\n===== {title} =====")
    header = f"{'bucket':8s} {'n':>6s} " + " ".join(f"{f:>14s}" for f in fields)
    print(header)
    for b in ("short", "medium", "long"):
        vals = data_by_bucket.get(b, {})
        n = vals.get("_n", 0)
        row = f"{b:8s} {n:6d} "
        row += " ".join(f"{vals.get(f, float('nan')):14.1f}" for f in fields)
        print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1_dir", default="outputs/casqta_stage1")
    ap.add_argument("--stage2_json", default="outputs/vmme_casqta.json")
    ap.add_argument("--csv", default="playground/gt_qa_files/VideoMME/val_qa.csv")
    ap.add_argument("--k_max", type=int, default=0,
                     help="neu >0, in them ti le video co K == k_max (dang bi chan tran)")
    args = ap.parse_args()

    vid_bucket = load_video_bucket(args.csv)
    qid_bucket = load_qid_bucket(args.csv)

    # --- Stage 1: mot dong / video (K, RC, n_feat_used) ---
    s1_rows = defaultdict(list)   # bucket -> list of dict
    for fp in glob.glob(os.path.join(args.stage1_dir, "*.json")):
        meta = json.load(open(fp))
        vname = meta.get("video_name", os.path.basename(fp))
        b = vid_bucket.get(stem(vname), "unknown")
        s1_rows[b].append({
            "K": meta.get("K", len(meta.get("values", []))),
            "RC": meta.get("recon_coverage", float("nan")),
            "n_feat_used": meta.get("n_feat_used", float("nan")),
            "clip_frames_total": sum(len(c) for c in meta.get("clips_orig", [])),
        })

    s1_summary = {}
    for b, rows in s1_rows.items():
        s1_summary[b] = {
            "_n": len(rows),
            "K_mean": s.mean(r["K"] for r in rows),
            "RC_mean": s.mean(r["RC"] for r in rows),
            "clip_frames_mean": s.mean(r["clip_frames_total"] for r in rows),
            "avg_clip_size": s.mean(r["clip_frames_total"] / max(1, r["K"]) for r in rows),
        }
    print_bucket_table(
        "STAGE 1 (RECON) - 1 dong/video",
        s1_summary, ["K_mean", "RC_mean", "clip_frames_mean", "avg_clip_size"])

    if args.k_max > 0:
        print(f"\n--- % video co K == k_max ({args.k_max}), tuc la con dang bi chan tran ---")
        for b in ("short", "medium", "long"):
            rows = s1_rows.get(b, [])
            if not rows:
                continue
            n_capped = sum(1 for r in rows if r["K"] >= args.k_max)
            print(f"{b:8s} {n_capped}/{len(rows)} = {100*n_capped/len(rows):.1f}%")

    # --- Stage 2: mot dong / cau hoi (n_frames, tokens, tokens/frame) ---
    if os.path.exists(args.stage2_json):
        combined = json.load(open(args.stage2_json))
        s2_rows = defaultdict(list)
        for qid, pairs in combined.items():
            b = qid_bucket.get(qid, "unknown")
            n_frames = len(pairs)
            total_tok = sum(v for _, v in pairs)
            s2_rows[b].append({
                "n_frames": n_frames,
                "total_tok": total_tok,
                "tok_per_frame": total_tok / max(1, n_frames),
            })
        s2_summary = {}
        for b, rows in s2_rows.items():
            s2_summary[b] = {
                "_n": len(rows),
                "n_frames_mean": s.mean(r["n_frames"] for r in rows),
                "total_tok_mean": s.mean(r["total_tok"] for r in rows),
                "tok_per_frame_mean": s.mean(r["tok_per_frame"] for r in rows),
            }
        print_bucket_table(
            "STAGE 2 (QTA) - 1 dong/cau hoi",
            s2_summary, ["n_frames_mean", "total_tok_mean", "tok_per_frame_mean"])
    else:
        print(f"\n[skip] {args.stage2_json} khong ton tai")


if __name__ == "__main__":
    main()
