#!/bin/bash
# Phase-0 oracle upper bound.
#
# Selects the same K=12 candidate frames as the plain question-aware KTV
# selector (cluster_and_rank_keyframes.py), but ranks them by CLIP
# similarity to the ground-truth ANSWER text instead of the question.
# Same downstream token budget/pruning as production (cls_new_token_sim,
# rate=0.2, tokens_num=1872) so the only thing that changed is the
# relevance signal used to pick frames.
#
# Compare pred_videomme_oracle's accuracy against your existing
# question-aware run at the same tokens_num to see how much headroom
# selection quality actually has.
set -euo pipefail

# Defaults to plain `python` (your activated conda env). Override with
# PY="uv run python" if this project uses uv on your machine instead.
PY="${PY:-python}"

DATASET="${DATASET:-videomme}"   # videomme | nextqa -- the actual dataset
VARIANT="${VARIANT:-narrow}"     # narrow (K=12 candidates) | wide (num_candidates=30)
TOKENS_NUM="${TOKENS_NUM:-1872}"
RATE="${RATE:-0.2}"

case "${DATASET}" in
  videomme)
    INFERENCE_EXPERIMENT=videomme_6keyframe
    BUCKET_CSV=playground/gt_qa_files/VideoMME/val_qa.csv
    case "${VARIANT}" in
      narrow)
        CLUSTER_EXPERIMENT=videomme_oracle
        KEYFRAME_JSON=outputs/videomme_oracle_keyframe6_order.json
        ;;
      wide)
        # Phase-0.5: num_candidates=30 (KMeans pool widened), still top-12
        # kept for inference -- same K/token budget/pruning as every other run.
        CLUSTER_EXPERIMENT=videomme_oracle_wide
        KEYFRAME_JSON=outputs/videomme_oracle_wide_keyframe6_order.json
        ;;
      *)
        echo "Unknown VARIANT=${VARIANT}" >&2; exit 1 ;;
    esac
    ;;
  nextqa)
    if [[ "${VARIANT}" != "narrow" ]]; then
      echo "VARIANT=wide has no nextqa config yet -- only narrow is prepared for nextqa" >&2
      exit 1
    fi
    CLUSTER_EXPERIMENT=nextqa_oracle
    INFERENCE_EXPERIMENT=nextqa
    KEYFRAME_JSON=outputs/nextqa_oracle_keyframe6_order.json
    BUCKET_CSV=""   # NExTQA has no duration bucket column
    ;;
  *)
    echo "Unknown DATASET=${DATASET}" >&2; exit 1 ;;
esac

if [[ "${VARIANT}" == "narrow" ]]; then
  # Matches the original run's output name exactly so the already-completed
  # pred_videomme_oracle.json (2700/2700, 43.59%) is recognized and reused,
  # not silently re-run from scratch.
  OUTPUT_NAME="pred_${DATASET}_oracle"
else
  OUTPUT_NAME="pred_${DATASET}_oracle_${VARIANT}"
fi
PRED_PATH="outputs/${OUTPUT_NAME}.json"

echo "=== Stage 1: oracle keyframe selection (${DATASET}, variant=${VARIANT}) ==="
# NOTE: if you've run this experiment name before, the per-question cache
# in save_cluster_path will short-circuit and reuse old files. Delete
# the *_keyframes dir + combined JSON first if you changed anything upstream.
${PY} cluster_and_rank_keyframes.py experiment="${CLUSTER_EXPERIMENT}"

echo "=== Stage 2: inference (same prune_mode/rate/tokens_num as production) ==="
${PY} run_inference_multiple_choice_qa.py \
  experiment="${INFERENCE_EXPERIMENT}" \
  output_name="${OUTPUT_NAME}" \
  key_frame_path="${KEYFRAME_JSON}" \
  prune_mode=cls_new_token_sim \
  rate="${RATE}" \
  tokens_num="${TOKENS_NUM}"

echo "=== Eval (overall) ==="
${PY} eval/compute_accuracy.py "${PRED_PATH}" | tee "outputs/${OUTPUT_NAME}_accuracy.txt"

if [[ -n "${BUCKET_CSV}" ]]; then
  echo "=== Eval (by short/medium/long bucket) ==="
  ${PY} eval/compute_accuracy_by_bucket.py "${PRED_PATH}" --csv "${BUCKET_CSV}" \
    | tee "outputs/${OUTPUT_NAME}_accuracy_by_bucket.txt"
fi
