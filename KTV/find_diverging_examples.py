"""
find_diverging_examples.py

Find questions where two prediction files disagree on correctness (one got
it right, the other wrong) -- picks out the most informative videos to run
through visualize_pruning.py, instead of choosing one at random.

Usage:
  python find_diverging_examples.py \
    outputs/pred_videomme_qwen25vl_uniform_3x.json \
    outputs/pred_videomme_qwen25vl_dyseg_3x.json \
    --gt_file playground/gt_qa_files/VideoMME/val_qa.json \
    --limit 15

  # Show every question for ONE specific video (e.g. to caption
  # visualize_pruning.py screenshots for that exact video), not just the
  # diverging ones:
  python find_diverging_examples.py \
    outputs/pred_videomme_qwen25vl_uniform_3x.json \
    outputs/pred_videomme_qwen25vl_dyseg_3x.json \
    --video sDWOsWawxPc.mp4
"""

import argparse
import json
import sys

sys.path.insert(0, ".")
from eval.compute_accuracy import load_jsonl, prediction_to_index


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_path_a")
    ap.add_argument("pred_path_b")
    ap.add_argument("--gt_file", default="playground/gt_qa_files/VideoMME/val_qa.json")
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument("--video", default=None, help="Show every question for this video_name only (skips the diverging-only filter).")
    return ap.parse_args()


def main():
    args = parse_args()

    with open(args.gt_file, "r", encoding="utf-8") as f:
        gt = json.load(f)
    id_to_video = {str(s.get("id", s.get("question_id"))): s["video_name"] for s in gt}

    recs_a = {str(r["id"]): r for r in load_jsonl(args.pred_path_a)}
    recs_b = {str(r["id"]): r for r in load_jsonl(args.pred_path_b)}

    common_ids = set(recs_a) & set(recs_b)
    print(f"{len(common_ids)} question ids in common\n")

    if args.video:
        letters = "ABCDEF"
        target_ids = [qid for qid in common_ids if id_to_video.get(qid) == args.video]
        print(f"=== Toan bo {len(target_ids)} cau hoi cua video={args.video} ===\n")
        for qid in sorted(target_ids):
            ra, rb = recs_a[qid], recs_b[qid]
            gt_idx = ra.get("answer_number")
            idx_a = prediction_to_index(ra.get("pred"), ra.get("candidates", []))
            idx_b = prediction_to_index(rb.get("pred"), rb.get("candidates", []))
            print(f"id={qid}")
            print(f"  question: {ra['question']}")
            for i, c in enumerate(ra.get("candidates", [])):
                print(f"    {c}")
            print(f"  answer_dung={letters[gt_idx] if gt_idx is not None and gt_idx < len(letters) else gt_idx}"
                  f"  A({args.pred_path_a.split('/')[-1]})={letters[idx_a] if idx_a is not None else '?'}"
                  f" ({'DUNG' if idx_a == gt_idx else 'SAI'})"
                  f"  B({args.pred_path_b.split('/')[-1]})={letters[idx_b] if idx_b is not None else '?'}"
                  f" ({'DUNG' if idx_b == gt_idx else 'SAI'})")
            print()
        return

    a_right_b_wrong, b_right_a_wrong = [], []
    for qid in common_ids:
        ra, rb = recs_a[qid], recs_b[qid]
        gt_idx = ra.get("answer_number")
        idx_a = prediction_to_index(ra.get("pred"), ra.get("candidates", []))
        idx_b = prediction_to_index(rb.get("pred"), rb.get("candidates", []))
        correct_a = idx_a == gt_idx
        correct_b = idx_b == gt_idx
        if correct_a and not correct_b:
            a_right_b_wrong.append((qid, ra, idx_a, idx_b, gt_idx))
        elif correct_b and not correct_a:
            b_right_a_wrong.append((qid, ra, idx_a, idx_b, gt_idx))

    def show(label, items):
        print(f"=== {label} ({len(items)} total) ===")
        for qid, r, ia, ib, gt_idx in items[: args.limit]:
            video_name = id_to_video.get(qid, "?")
            letters = "ABCDEF"
            print(f"id={qid}  video={video_name}")
            print(f"  question: {r['question'][:100]}")
            print(f"  answer_dung={letters[gt_idx] if gt_idx is not None and gt_idx < len(letters) else gt_idx}"
                  f"  A_pred={letters[ia] if ia is not None else '?'}"
                  f"  B_pred={letters[ib] if ib is not None else '?'}")
            print()

    show(f"A ({args.pred_path_a}) dung, B ({args.pred_path_b}) sai", a_right_b_wrong)
    show(f"B ({args.pred_path_b}) dung, A ({args.pred_path_a}) sai", b_right_a_wrong)


if __name__ == "__main__":
    main()
