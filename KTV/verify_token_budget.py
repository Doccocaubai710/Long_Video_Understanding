"""
verify_token_budget.py

Pre-flight check for a Stage-2 (qta_allocated.py) keyframe JSON, BEFORE
spending a GPU run on it. Mirrors the exact remain_tokens_num decision
in ktv/llava/model/llava_arch.py (encode_images, ~L204-236) in pure
Python, so it needs no model weights / DINO tensors / video files --
only the {question_id: [[frame_index, order], ...]} JSON that
qta_allocated.py already produced.

It reports, per question and in aggregate:
  - intended tokens (what qta_allocated.py computed)
  - realized tokens (what llava_arch.py will ACTUALLY feed, given the
    num_frames / tokens_num you plan to pass to run_inference_multiple_choice_qa.py)
  - whether the QTA spread got flattened to a single value (the missing
    "order >= 12 -> use as direct budget" patch)
  - whether realized tokens exceed the model's context budget
  - whether the real code would crash with UnboundLocalError

Usage:
  python verify_token_budget.py outputs/vmme_casqta.json \
      --num_frames 6 --tokens_num 4608 \
      --context_len 4096 --rope_scaling_factor 2 --text_reserve 400

  # check one question only, verbose:
  python verify_token_budget.py outputs/vmme_casqta.json \
      --num_frames 6 --tokens_num 4608 --question_id <qid>
"""

import argparse
import json
import statistics
import sys

TOKEN_LEVELS = [288, 144, 72, 36, 18]  # must match qta_allocated.py


def compute_remain_tokens(order, num_frames, tokens_num):
    """Exact mirror of ktv/llava/model/llava_arch.py encode_images(), the
    branch taken when keyframe_order is truthy (post order>=12 patch).

    Returns (remain_tokens_num, unbound). unbound=True means the real
    code has no matching branch -> UnboundLocalError at torch.topk(...).
    """
    if order >= 12:
        # CAS-QTA: order IS the per-frame token budget already.
        return order, False
    if num_frames == 6:
        remain = 288 if order == 0 else (144 if order in (1, 2, 3, 4) else 72)
        if tokens_num == 1872:
            remain = 576 if order == 0 else (288 if order in (1, 2, 3, 4) else 144)
        elif tokens_num == 504:
            remain = 144 if order == 0 else 72
        return remain, False
    elif num_frames == 12:
        remain = 288 if order in (0, 1) else (144 if order in range(2, 10) else 72)
        return remain, False
    else:
        return None, True


def check_qta_format(pairs):
    """Warn if 'order' values look like CAS-QTA token budgets (from
    TOKEN_LEVELS) rather than legacy 0..K-1 cluster ranks -- the two
    formats are silently incompatible in llava_arch.py today."""
    orders = {o for _, o in pairs}
    looks_like_budget = orders and orders.issubset(set(TOKEN_LEVELS))
    looks_like_rank = orders and all(isinstance(o, int) and 0 <= o < 12 for o in orders)
    return looks_like_budget, looks_like_rank


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("key_frame_path", help="path to combined_output_path from qta_allocated.py")
    ap.add_argument("--num_frames", type=int, default=6,
                     help="the num_frames= value you plan to pass to run_inference (branch selector, "
                          "NOT the actual number of frames loaded)")
    ap.add_argument("--tokens_num", type=int, default=1872,
                     help="the tokens_num= value you plan to pass to run_inference")
    ap.add_argument("--context_len", type=int, default=4096, help="base LLM context (Vicuna-7B-v1.5)")
    ap.add_argument("--rope_scaling_factor", type=int, default=1,
                     help="match the rope_scaling_factor= in your inference experiment config")
    ap.add_argument("--text_reserve", type=int, default=400,
                     help="approx tokens reserved for system prompt + question + candidates + generation")
    ap.add_argument("--question_id", default=None, help="only check this one question, verbose")
    args = ap.parse_args()

    data = json.load(open(args.key_frame_path))
    budget = args.context_len * max(1, args.rope_scaling_factor) - args.text_reserve

    items = data
    if args.question_id:
        if args.question_id not in data:
            print(f"question_id {args.question_id!r} not found in {args.key_frame_path}", file=sys.stderr)
            sys.exit(1)
        items = {args.question_id: data[args.question_id]}

    n_overflow = n_unbound = n_flattened = n_format_mismatch = 0
    totals, intended_totals = [], []

    for qid, pairs in items.items():
        frame_to_order = {f: o for f, o in pairs}
        keyframe_order = [frame_to_order[f] for f in sorted(frame_to_order)]
        n_frames = len(keyframe_order)

        looks_like_budget, looks_like_rank = check_qta_format(pairs)
        if looks_like_budget and args.num_frames in (6, 12):
            n_format_mismatch += 1

        realized, unbound = [], False
        for order in keyframe_order:
            remain, ub = compute_remain_tokens(order, args.num_frames, args.tokens_num)
            if ub:
                unbound = True
                break
            realized.append(remain)

        if unbound:
            n_unbound += 1
            print(f"[CRASH] {qid}: num_frames={args.num_frames} tokens_num={args.tokens_num} "
                  f"has no matching branch -> UnboundLocalError in llava_arch.py")
            continue

        total = sum(realized)
        intended = sum(keyframe_order)
        totals.append(total)
        intended_totals.append(intended)

        flattened = len(set(realized)) == 1 and len(set(keyframe_order)) > 1
        if flattened:
            n_flattened += 1
        overflow = total > budget
        if overflow:
            n_overflow += 1

        if flattened or overflow or args.question_id:
            print(f"{qid}: frames={n_frames} intended={intended} realized={total} "
                  f"(budget={budget})"
                  + (" OVERFLOW" if overflow else "")
                  + (f" FLATTENED->{realized[0]}/frame (QTA spread ignored)" if flattened else ""))

    print("\n=== summary ===")
    print(f"questions checked:      {len(items)}")
    print(f"crash (unbound branch): {n_unbound}")
    print(f"QTA spread flattened:   {n_flattened}  "
          f"(order values are CAS-QTA token budgets but llava_arch.py reads them as legacy rank)")
    print(f"context overflow:       {n_overflow}  (realized tokens > budget={budget})")
    if totals:
        print(f"intended tokens/question: mean={statistics.mean(intended_totals):.0f} "
              f"median={statistics.median(intended_totals):.0f}")
        print(f"realized tokens/question: mean={statistics.mean(totals):.0f} "
              f"median={statistics.median(totals):.0f} min={min(totals)} max={max(totals)}")


if __name__ == "__main__":
    main()
