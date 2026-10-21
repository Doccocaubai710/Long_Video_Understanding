"""So cap hai file du doan tren cac cau chung (McNemar), cham diem bang ham chinh thuc cua repo.
Usage (tu thu muc KTV): python outputs/rap/compare.py <pred_A.json> <pred_B.json> [ten_A] [ten_B]"""
import csv
import math
import sys

sys.path.insert(0, ".")
from eval.compute_accuracy import load_jsonl, prediction_to_index


def load(path):
    out = {}
    for r in load_jsonl(path):
        idx = prediction_to_index(r.get("pred"), r.get("candidates", []))
        out[str(r.get("question_id"))] = (idx is not None and idx == r.get("answer_number"))
    return out


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    na = sys.argv[3] if len(sys.argv) > 3 else "A"
    nb = sys.argv[4] if len(sys.argv) > 4 else "B"
    with open("playground/gt_qa_files/VideoMME/val_qa.csv", newline="", encoding="utf-8") as f:
        bucket = {r["question_id"]: r["duration"] for r in csv.DictReader(f)}
    common = sorted(set(a) & set(b))
    print(f"{na} vs {nb}: {len(common)} cau chung (A co {len(a)}, B co {len(b)})")
    print(f"{'nhom':8s} {'n':>5s} {na[:14]:>14s} {nb[:14]:>14s} {'chi A':>6s} {'chi B':>6s} {'z':>6s}")
    for key in ("short", "medium", "long", "overall"):
        ids = [q for q in common if key == "overall" or bucket.get(q) == key]
        if not ids:
            continue
        ca = sum(a[q] for q in ids)
        cb = sum(b[q] for q in ids)
        oa = sum(a[q] and not b[q] for q in ids)
        ob = sum(b[q] and not a[q] for q in ids)
        z = (oa - ob) / math.sqrt(oa + ob) if oa + ob else 0.0
        print(f"{key:8s} {len(ids):5d} {100 * ca / len(ids):14.2f} {100 * cb / len(ids):14.2f} {oa:6d} {ob:6d} {z:+6.2f}")


if __name__ == "__main__":
    main()
