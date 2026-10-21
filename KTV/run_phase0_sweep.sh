#!/bin/bash
# Phase-0 frames-vs-resolution sweep at fixed token budget B=1872.
#
# Uniform frame sampling (no keyframe selection) + attention-based token
# pruning down to floor(tokens_num / num_frames) tokens/frame -- this is
# the tokens_num // num_frames branch in llava_arch.py:240, which only
# fires when key_frame_path is unset AND prune_mode is one of
# {cls_new_token_sim, uniform_token}. prune_mode=null (as used by the
# "upper_bound_dense_uniform_frames" variants in run_all_tokens_experiments.sh)
# does NOT apply this budget -- it always keeps the full 576 tokens/frame,
# so it is not usable for this sweep. See conversation notes.
#
# NUM_FRAMES_LIST=(8 12 16 24 32) -> 234/156/117/78/58 tokens/frame at B=1872.
set -euo pipefail

# Defaults to plain `python` (your activated conda env). Override with
# PY="uv run python" if this project uses uv on your machine instead.
PY="${PY:-python}"

DATASET="${DATASET:-videomme}"   # videomme | nextqa
TOKENS_NUM="${TOKENS_NUM:-1872}"
RATE="${RATE:-0.2}"
NUM_FRAMES_LIST=(${NUM_FRAMES_LIST:-8 12 16 24 32})

case "${DATASET}" in
  videomme)
    INFERENCE_EXPERIMENT=videomme_6keyframe
    BUCKET_CSV=playground/gt_qa_files/VideoMME/val_qa.csv
    ;;
  nextqa)
    INFERENCE_EXPERIMENT=nextqa
    BUCKET_CSV=""
    ;;
  *)
    echo "Unknown DATASET=${DATASET}" >&2; exit 1 ;;
esac

for num_frames in "${NUM_FRAMES_LIST[@]}"; do
  tok_per_frame=$(( TOKENS_NUM / num_frames ))
  output_name="pred_${DATASET}_resolution_sweep_f${num_frames}"
  pred_path="outputs/${output_name}.json"

  echo "=================================================="
  echo "num_frames=${num_frames} -> ~${tok_per_frame} tokens/frame (B=${TOKENS_NUM})"
  echo "=================================================="

  ${PY} run_inference_multiple_choice_qa.py \
    experiment="${INFERENCE_EXPERIMENT}" \
    output_name="${output_name}" \
    key_frame_path=null \
    prune_mode=cls_new_token_sim \
    rate="${RATE}" \
    tokens_num="${TOKENS_NUM}" \
    num_frames="${num_frames}"

  ${PY} eval/compute_accuracy.py "${pred_path}" | tee "outputs/${output_name}_accuracy.txt"

  if [[ -n "${BUCKET_CSV}" ]]; then
    ${PY} eval/compute_accuracy_by_bucket.py "${pred_path}" --csv "${BUCKET_CSV}" \
      | tee "outputs/${output_name}_accuracy_by_bucket.txt"
  fi
done
