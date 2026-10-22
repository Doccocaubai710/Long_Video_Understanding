#!/bin/bash
# Smoke test BEFORE committing to full runs: does Qwen2.5-VL-7B handle much
# larger frame counts / token budgets without OOM or breaking, on this A100?
# B=1872 (num_frames<=32) is a KTV-legacy artifact -- Qwen3-VL's own "dense"
# video config is ~128 frames / 16384 tokens (~128 tok/frame). This sweep
# extends toward that, holding tok/frame roughly constant (~128-150) and
# scaling frame count -- the same lever (more frames) already shown to have
# a real, significant effect in the earlier resolution sweep, just pushed
# much further than the 1872-token ceiling allowed.
#
# Run in the qwen25vl env. 30 questions each, ~5-10 min total expected --
# if any of these OOM or crash, we find out here, not an hour into a full run.
set -euo pipefail

BUCKET_CSV=playground/gt_qa_files/VideoMME/val_qa.csv

# num_frames tokens_num  (~tok/frame)
CONFIGS=(
  "48 7200"     # ~150 tok/frame
  "64 9600"     # ~150 tok/frame, matches LLaVA-Video's own 64-frame recipe
  "96 14400"    # ~150 tok/frame
  "128 16384"   # ~128 tok/frame, matches Qwen3-VL's own reported dense video config
)

for cfg in "${CONFIGS[@]}"; do
  read -r num_frames tokens_num <<< "${cfg}"
  output_name="smoke_qwen25vl_budget_f${num_frames}_t${tokens_num}"

  echo "=================================================="
  echo "num_frames=${num_frames} tokens_num=${tokens_num} (~$((tokens_num / num_frames)) tok/frame)"
  echo "=================================================="

  python run_inference_qwen25vl.py \
    experiment=videomme \
    output_name="${output_name}" \
    key_frame_path=null \
    num_frames="${num_frames}" \
    tokens_num="${tokens_num}" \
    num_chunks=90 chunk_idx=0

  python eval/compute_accuracy.py "outputs/${output_name}.json"
done

echo "=== Smoke sweep done. Review accuracy/unparsed/timing above before running the full sweep. ==="
