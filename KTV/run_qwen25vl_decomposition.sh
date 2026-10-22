#!/bin/bash
# 3-way decomposition on Qwen2.5-VL-7B, isolating "does smart selection
# beat uniform" from "does a perfect oracle signal beat smart selection":
#
#   (1) Uniform 12 frames                          -- already have: 54.11%
#   (2) Wide diverse, ranked by QUESTION (30->12)   -- deployable strategy
#   (3) Wide diverse, ranked by ANSWER/oracle (30->12) -- upper bound
#
# Needs outputs/videomme_wide_question_keyframe6_order.json (run
# gen_wide_question_keyframes.sh in the (ktv) env first) and
# outputs/videomme_oracle_wide_keyframe6_order.json (already generated
# earlier this session for the old-backbone wide-oracle experiment --
# frame selection is backbone-agnostic, reused as-is).
#
# Run in the qwen25vl env.
set -euo pipefail

BUCKET_CSV=playground/gt_qa_files/VideoMME/val_qa.csv

run_and_eval() {
  local output_name="$1"
  local key_frame_path="$2"
  local pred_path="outputs/${output_name}.json"

  echo "=================================================="
  echo "${output_name}  (key_frame_path=${key_frame_path})"
  echo "=================================================="

  python run_inference_qwen25vl.py \
    experiment=videomme \
    output_name="${output_name}" \
    key_frame_path="${key_frame_path}" \
    num_frames=12 \
    tokens_num=1872

  python eval/compute_accuracy.py "${pred_path}" | tee "outputs/${output_name}_accuracy.txt"
  python eval/compute_accuracy_by_bucket.py "${pred_path}" --csv "${BUCKET_CSV}" \
    | tee "outputs/${output_name}_accuracy_by_bucket.txt"
}

echo "=== (2) Wide diverse, ranked by QUESTION (deployable) ==="
run_and_eval "pred_videomme_qwen25vl_wide_question" \
  "outputs/videomme_wide_question_keyframe6_order.json"

echo "=== (3) Wide diverse, ranked by ANSWER (oracle upper bound) ==="
run_and_eval "pred_videomme_qwen25vl_wide_oracle" \
  "outputs/videomme_oracle_wide_keyframe6_order.json"

echo "=== Done. Compare against uniform-12 (already have: 54.11%, outputs/pred_videomme_qwen25vl_uniform12.json) ==="
