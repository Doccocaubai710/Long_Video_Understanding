#!/bin/bash
# Card 2 (GPU1): RAP R10E tren MLVU dev (cho bang video dai trong trang trinh bay).
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1
export DECORD_EOF_RETRY_MAX=131072 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1
ST=outputs/hds5/STATUS_bench52_card2.txt
echo "$(date -u '+%Y-%m-%d %H:%M:%S') START mlvu_r10e" >> $ST
timeout 16h /home/hoang/miniconda3/envs/qwen25vl/bin/python -m rap.eval_multi --gt_file playground/gt_qa_files/MLVU_Dev/val_qa_mc.json \
  --video_dir Datasets/MLVU_Dev/MLVU/video --model_path Qwen/Qwen2.5-VL-7B-Instruct --output_dir outputs/mlvu_dev/pred --max_frames 128 \
  --variant R10E_r010=ckpt:outputs/checkpoints/R10E.pt@0.10 >> outputs/hds5/logs/bench52_mlvu_r10e.log 2>&1
echo "$(date -u '+%Y-%m-%d %H:%M:%S') END mlvu_r10e: $(wc -l < outputs/mlvu_dev/pred/R10E_r010.json)" >> $ST
