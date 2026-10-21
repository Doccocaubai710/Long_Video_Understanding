"""
compute_accuracy_by_bucket.py

Same accuracy logic as compute_accuracy.py, but split by VideoMME duration
bucket (short/medium/long) instead of one aggregate number. Needed because
headroom/gains from a selector are not uniform across bucket (see the
K=12 effective-rank finding: short ~70 -> medium ~150 -> long ~300), so an
aggregate delta can hide a bucket-specific story.

Usage:
  python eval/compute_accuracy_by_bucket.py outputs/pred_oracle.json \
      --csv playground/gt_qa_files/VideoMME/val_qa.csv
"""
import argparse
import csv
import sys
from collections import defaultdict

sys.path.insert(0, ".")
from eval.compute_accuracy import load_jsonl, prediction_to_index


def load_qid_bucket(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        return {r["question_id"]: r["duration"] for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_path")
    ap.add_argument("--csv", default="playground/gt_qa_files/VideoMME/val_qa.csv")
    args = ap.parse_args()

    qid_bucket = load_qid_bucket(args.csv)
    records = load_jsonl(args.pred_path)

    totals = defaultdict(lambda: {"correct": 0, "total": 0, "unparsed": 0})
    for r in records:
        bucket = qid_bucket.get(r.get("question_id"), "unknown")
        idx = prediction_to_index(r.get("pred"), r.get("candidates", []))
        for key in (bucket, "overall"):
            totals[key]["total"] += 1
            if idx is None:
                totals[key]["unparsed"] += 1
            elif idx == r.get("answer_number"):
                totals[key]["correct"] += 1

    print(f"{'bucket':10s} {'n':>6s} {'correct':>8s} {'unparsed':>9s} {'acc':>7s}")
    for key in ("short", "medium", "long", "unknown", "overall"):
        if key not in totals:
            continue
        t = totals[key]
        acc = t["correct"] / t["total"] if t["total"] else 0.0
        print(f"{key:10s} {t['total']:6d} {t['correct']:8d} {t['unparsed']:9d} {acc:7.4f}")


if __name__ == "__main__":
    main()
