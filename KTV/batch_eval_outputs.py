"""
batch_eval_outputs.py

Quet toan bo file .json trong outputs/ (khong phai *_accuracy*.txt, khong
phai thu muc), tinh accuracy tong (dung chung eval/compute_accuracy.py, khong
can file CSV bucket -- hoat dong voi moi dataset/task_name), roi in bang xep
hang giam dan. Muc dich: chon nhanh vai file "tieu bieu" trong hang chuc file
cu, thay vi chay tay tung file.

CANH BAO IN KEM (doc ky truoc khi chon so lieu dua vao slide):
  - File "nextqa_*"                 -> dataset NextQA, KHONG PHAI VideoMME,
                                        khong so truc tiep voi cac file
                                        pred_videomme_*.
  - n < 500                         -> smoke test / chay thu mot phan nho,
                                        khong dai dien, dung sai so lon.
  - Cac file tu truoc 2026-08-14    -> harness CHUA sua Gate G1 (xem README
    (theo ten hoac theo ngay tao)      chinh Sec 9) -- backbone/prompt/
                                        generation config khac hoan toan cac
                                        file *_3x/*_1x gan day. KHONG so
                                        truc tiep accuracy giua hai nhom nay.

Usage:
  python batch_eval_outputs.py outputs
  python batch_eval_outputs.py outputs --min_n 500
"""

import argparse
import glob
import os
import sys

# Resolve relative to THIS script's own location, not the caller's current
# directory -- so `eval/` (inside KTV/) is found even when invoked from
# inside outputs/ (e.g. `python ../batch_eval_outputs.py .`).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval.compute_accuracy import load_jsonl, compute_accuracy


OLD_HARNESS_MARKERS = (
    "resolution_sweep", "budget_f", "oracle", "wide_question", "wide_oracle",
    "casqta", "temporal_chain", "tc_f2c", "maxinfo", "smoke_", "check_",
    "predictions_tc", "vmme_exp", "pred_exp", "speedcheck",
)


def classify(fname):
    tags = []
    if "nextqa" in fname:
        tags.append("NEXTQA-not-VideoMME")
    if any(m in fname for m in OLD_HARNESS_MARKERS):
        tags.append("old-harness/pre-Gate-G1?")
    return tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outputs_dir")
    ap.add_argument("--min_n", type=int, default=0, help="Only show rows with at least this many records.")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.outputs_dir, "*.json")))
    rows = []
    for path in paths:
        fname = os.path.basename(path)
        try:
            records = load_jsonl(path)
        except Exception as e:
            print(f"SKIP (khong doc duoc): {fname}: {type(e).__name__}: {e}")
            continue
        if not records or "answer_number" not in records[0]:
            print(f"SKIP (khong phai file prediction chuan): {fname}")
            continue
        totals = compute_accuracy(records)
        overall = totals.get("overall", {"correct": 0, "total": 0, "unparsed": 0})
        n = overall["total"]
        if n < args.min_n:
            continue
        acc = overall["correct"] / n if n else 0.0
        rows.append((fname, n, acc, overall["unparsed"], classify(fname)))

    rows.sort(key=lambda r: r[2], reverse=True)
    print(f"\n{'file':<60} {'n':>6} {'acc':>8} {'unparsed':>9}  canh_bao")
    print("-" * 110)
    for fname, n, acc, unparsed, tags in rows:
        warn = ", ".join(tags)
        print(f"{fname:<60} {n:>6} {acc*100:>7.2f}% {unparsed:>9}  {warn}")


if __name__ == "__main__":
    main()
