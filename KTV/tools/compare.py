"""compare.py -- so cap tung cau giua 2 file du doan VideoMME (McNemar z), chia theo short/medium/long.

Usage (chay tu thu muc KTV):
  python -m tools.compare outputs/preds/pred_videomme_f128_rap_runE_r025.json \
      outputs/preds/pred_videomme_f128_holitom_r025.json [--names RAP HoliTom]

|z| > 1,96 la co y nghia. "chi A" = so cau A dung ma B sai.
"""
import argparse
import csv
import math
import os

from eval.compute_accuracy import load_jsonl, prediction_to_index

CSV = "playground/gt_qa_files/VideoMME/val_qa.csv"


def load(path):
    out = {}
    for r in load_jsonl(path):
        idx = prediction_to_index(r.get("pred"), r.get("candidates", []))
        out[str(r.get("question_id"))] = idx is not None and idx == r.get("answer_number")
    return out


def paired(a, b, ids):
    if not ids:
        return float("nan"), float("nan"), 0, 0, 0.0
    da = 100.0 * sum(a[q] for q in ids) / len(ids)
    db = 100.0 * sum(b[q] for q in ids) / len(ids)
    oa = sum(a[q] and not b[q] for q in ids)
    ob = sum(b[q] and not a[q] for q in ids)
    z = (oa - ob) / math.sqrt(oa + ob) if oa + ob else 0.0
    return da, db, oa, ob, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--names", nargs=2, default=None)
    ap.add_argument("--csv", default=CSV)
    args = ap.parse_args()
    na, nb = args.names or [os.path.basename(p).replace("pred_videomme_", "").replace(".json", "") for p in (args.a, args.b)]
    with open(args.csv, newline="", encoding="utf-8") as f:
        bucket = {r["question_id"]: r["duration"] for r in csv.DictReader(f)}
    a, b = load(args.a), load(args.b)
    common = [q for q in a if q in b]
    print(f"{na} vs {nb}: {len(common)} cau chung (A co {len(a)}, B co {len(b)})")
    print(f"{'nhom':<8} {'n':>5} {na[:14]:>14} {nb[:14]:>14} {'chi A':>6} {'chi B':>6} {'z':>6}")
    for key in ["short", "medium", "long", "overall"]:
        ids = [q for q in common if key == "overall" or bucket.get(q) == key]
        da, db, oa, ob, z = paired(a, b, ids)
        print(f"{key:<8} {len(ids):>5} {da:>14.2f} {db:>14.2f} {oa:>6} {ob:>6} {z:>+6.2f}")


if __name__ == "__main__":
    main()
