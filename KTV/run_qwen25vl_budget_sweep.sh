#!/bin/bash
# Full 2700-question sweep: does scaling total token budget / frame count
# well past the KTV-legacy 1872 ceiling close the gap toward the
# dense-model performance reported in the field (e.g. Qwen3-VL-8B dense
# ~68.5 avg per the KiToke paper)? Existing reference point already in
# hand: 32 frames / 1872 tokens (58 tok/frame) -> 55.52%.
#
# Run in the qwen25vl env, AFTER reviewing run_qwen25vl_budget_smoke.sh's
# output (no OOM, unparsed~0, sane predictions). Each run writes/resumes
# JSONL incrementally, same as every other script in this project -- safe
# to kill and rerun this script if the host gets recycled again.
set -euo pipefail

BUCKET_CSV=playground/gt_qa_files/VideoMME/val_qa.csv

# num_frames tokens_num  (~tok/frame)
CONFIGS=(
  "48 7200"
  "64 9600"
  "96 14400"
  "128 16384"
)

for cfg in "${CONFIGS[@]}"; do
  read -r num_frames tokens_num <<< "${cfg}"
  output_name="pred_videomme_qwen25vl_budget_f${num_frames}_t${tokens_num}"
  pred_path="outputs/${output_name}.json"

  echo "=================================================="
  echo "num_frames=${num_frames} tokens_num=${tokens_num} (~$((tokens_num / num_frames)) tok/frame)"
  echo "=================================================="

  python run_inference_qwen25vl.py \
    experiment=videomme \
    output_name="${output_name}" \
    key_frame_path=null \
    num_frames="${num_frames}" \
    tokens_num="${tokens_num}"

  python eval/compute_accuracy.py "${pred_path}" | tee "outputs/${output_name}_accuracy.txt"
  python eval/compute_accuracy_by_bucket.py "${pred_path}" --csv "${BUCKET_CSV}" \
    | tee "outputs/${output_name}_accuracy_by_bucket.txt"
done

echo "=== Done. Compare against 32 frames / 1872 tokens (already have: 55.52%, outputs/pred_videomme_qwen25vl_resolution_sweep_f32.json) ==="
