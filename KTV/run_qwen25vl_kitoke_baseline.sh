#!/bin/bash
# Baseline KiToke tren Qwen2.5-VL-7B, VideoMME (khong subtitle), 3 muc giu
# token: 100% (dense), 40%, 25%. Khop setting Table 3 cua paper/kitoke repo
# (ban Qwen3-VL): 128 frame, ~256 token/frame -> 64 unit x 256 ~ 16K token,
# alpha=800 tau_diff=110 tau_delta=70 tau_relative=0.40, pivotal, seed 0.
# Qwen3 dung max_pixels=262144/frame voi patch 16 (=256 token/frame); ben
# Qwen2.5 (patch 14) cung so token do la max_pixels_units=16384 TONG.
#
# Chay trong env qwen25vl, tu thu muc KTV:
#   bash run_qwen25vl_kitoke_baseline.sh                 # full 2700 cau x 3 muc
#   LIMIT=12 bash run_qwen25vl_kitoke_baseline.sh        # smoke
#   RATIOS="0.40 0.25" bash run_qwen25vl_kitoke_baseline.sh
#   MODE=uniform RATIOS="0.40 0.25" bash run_qwen25vl_kitoke_baseline.sh   # doi chung cung ngan sach
# Moi run ghi/resume JSONL tung cau -- kill roi chay lai script la an toan.
set -euo pipefail

PYTHON=${PYTHON:-python}
RATIOS=${RATIOS:-"1.00 0.40 0.25"}
MODE=${MODE:-kitoke}
MAX_FRAMES=${MAX_FRAMES:-128}
MAX_PIXELS_UNITS=${MAX_PIXELS_UNITS:-16384}
SEED=${SEED:-0}
TIME_MODE=${TIME_MODE:-lmms}   # lmms | true | hf_default, xem docstring run_inference_qwen25vl_kitoke.py
LIMIT=${LIMIT:-0}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/kitoke_baseline}
GT_FILE=${GT_FILE:-playground/gt_qa_files/VideoMME/val_qa.json}
BUCKET_CSV=playground/gt_qa_files/VideoMME/val_qa.csv

export DECORD_EOF_RETRY_MAX=131072
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

suffix=""
if [[ "${LIMIT}" != "0" ]]; then suffix="_limit${LIMIT}"; fi
if [[ "${SEED}" != "0" ]]; then suffix="${suffix}_seed${SEED}"; fi
if [[ "${TIME_MODE}" != "lmms" ]]; then suffix="${suffix}_time-${TIME_MODE}"; fi

for ratio in ${RATIOS}; do
  if [[ "${ratio}" == "1.00" ]]; then
    mode=dense
    output_name="pred_videomme_f${MAX_FRAMES}_dense${suffix}"
  else
    mode=${MODE}
    output_name="pred_videomme_f${MAX_FRAMES}_${mode}_r${ratio/./}${suffix}"
  fi
  pred_path="${OUTPUT_DIR}/${output_name}.json"

  echo "=================================================="
  echo "mode=${mode} retention=${ratio} max_frames=${MAX_FRAMES} -> ${pred_path}"
  echo "=================================================="

  "${PYTHON}" run_inference_qwen25vl_kitoke.py \
    +video_dir=Datasets/Video-MME/data/data \
    +gt_file="${GT_FILE}" \
    +model_path=Qwen/Qwen2.5-VL-7B-Instruct \
    +mode="${mode}" +retention="${ratio}" +seed="${SEED}" \
    +max_frames="${MAX_FRAMES}" +max_pixels_units="${MAX_PIXELS_UNITS}" \
    +limit="${LIMIT}" +time_mode="${TIME_MODE}" \
    +output_dir="${OUTPUT_DIR}" +output_name="${output_name}" \
    hydra.run.dir="${OUTPUT_DIR}/hydra/${output_name}"

  "${PYTHON}" eval/compute_accuracy_by_bucket.py "${pred_path}" --csv "${BUCKET_CSV}" \
    | tee "${OUTPUT_DIR}/${output_name}_accuracy_by_bucket.txt"
done

echo "=== Done. Ket qua: ${OUTPUT_DIR}/*_accuracy_by_bucket.txt, thong ke nen: ${OUTPUT_DIR}/*.stats.jsonl ==="
