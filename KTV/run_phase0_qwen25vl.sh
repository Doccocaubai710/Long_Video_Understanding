#!/bin/bash
# Phase-0 on Qwen2.5-VL-7B-Instruct: finish the resolution sweep (f12 already
# done -> pred_videomme_qwen25vl_uniform12.json, 54.11%) and reuse the
# existing CLIP-oracle keyframe selection (frame selection is backbone-
# agnostic -- it was computed once against DINO/CLIP, independent of which
# VLM reads the frames) to see whether frame choice still matters on a
# stronger backbone.
#
# Run inside the qwen25vl conda env: `conda activate qwen25vl` first.
set -euo pipefail

TOKENS_NUM="${TOKENS_NUM:-1872}"
NUM_FRAMES_LIST=(${NUM_FRAMES_LIST:-8 16 24 32})   # 12 already done
BUCKET_CSV=playground/gt_qa_files/VideoMME/val_qa.csv

echo "=== Resolution sweep (remaining frame counts) ==="
for num_frames in "${NUM_FRAMES_LIST[@]}"; do
  output_name="pred_videomme_qwen25vl_resolution_sweep_f${num_frames}"
  pred_path="outputs/${output_name}.json"

  echo "--------------------------------------------------"
  echo "num_frames=${num_frames} (~$((TOKENS_NUM / num_frames)) tokens/frame)"
  echo "--------------------------------------------------"

  python run_inference_qwen25vl.py \
    experiment=videomme \
    output_name="${output_name}" \
    key_frame_path=null \
    num_frames="${num_frames}" \
    tokens_num="${TOKENS_NUM}"

  python eval/compute_accuracy.py "${pred_path}" | tee "outputs/${output_name}_accuracy.txt"
  python eval/compute_accuracy_by_bucket.py "${pred_path}" --csv "${BUCKET_CSV}" \
    | tee "outputs/${output_name}_accuracy_by_bucket.txt"
done

echo "=== Oracle (reusing existing CLIP-rank-by-answer keyframes, K=12) ==="
output_name="pred_videomme_qwen25vl_oracle"
pred_path="outputs/${output_name}.json"

python run_inference_qwen25vl.py \
  experiment=videomme \
  output_name="${output_name}" \
  key_frame_path=outputs/videomme_oracle_keyframe6_order.json \
  num_frames=12 \
  tokens_num="${TOKENS_NUM}"

python eval/compute_accuracy.py "${pred_path}" | tee "outputs/${output_name}_accuracy.txt"
python eval/compute_accuracy_by_bucket.py "${pred_path}" --csv "${BUCKET_CSV}" \
  | tee "outputs/${output_name}_accuracy_by_bucket.txt"

echo "=== All done ==="
