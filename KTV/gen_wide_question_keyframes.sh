#!/bin/bash
# Generate the missing piece for the 3-way decomposition on Qwen2.5-VL-7B:
# wide candidate pool (num_candidates=30), filtered down to 12 by CLIP
# similarity to the QUESTION (the real, deployable strategy) -- as opposed
# to the ground-truth ANSWER (oracle, already generated as
# outputs/videomme_oracle_wide_keyframe6_order.json).
#
# Run in the (ktv) env (needs `clip`), NOT qwen25vl -- this script has no
# GPU-heavy VLM inference, just KMeans + CLIP ranking, so it's cheap.
set -euo pipefail

PY="${PY:-python}"

${PY} cluster_and_rank_keyframes.py experiment=videomme_wide_question

echo "Done -> outputs/videomme_wide_question_keyframe6_order.json"
