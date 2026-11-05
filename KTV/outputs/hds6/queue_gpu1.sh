#!/bin/bash
# Card 2 (GPU1), dot hds6. Thu tu:
#  0  (trong luc cho Vript) cham RAP con thieu: ELA 10/25/40%, R40E, Run E tren MLVU dev + LVBench  [~2 h]
#  1  H8cv   = H8ce (5%, CE 0.2, +-3) + 50% Vript      -> doi chung H8ce (noi HDS kem RAP tren MLVU)
#  2  H10ng  = H10ce (40%, CE 0.2, +-1) nhung bo cong (1-r) (--gate 0)   -> y tuong A2 cho muc nen nhe
#  3  H10ce10 (dot hds5, w_ceg 1.0, dang tam dung) chay tiep tu checkpoint
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1
PY=/home/hoang/miniconda3/envs/qwen25vl/bin/python; MP=Qwen/Qwen2.5-VL-7B-Instruct; CK=outputs/checkpoints
export DECORD_EOF_RETRY_MAX=131072 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1
ST=outputs/hds6/STATUS_gpu1.txt; log() { echo "$(date -u '+%Y-%m-%d %H:%M:%S') $*" >> $ST; }
VRIPT="--vript_index Datasets/Vript/vript_index.jsonl --vript_prob 0.5 --vript_scenes 8 --max_caption_tokens 768"
log "QUEUE START gpu1"
RV="R40E_r040=ckpt:$CK/R40E.pt@0.40 runE_r025=ckpt:$CK/runE.pt@0.25 ELA_r040=ckpt:$CK/ELA.pt@0.40 ELA_r025=ckpt:$CK/ELA.pt@0.25 ELA_r010=ckpt:$CK/ELA.pt@0.10"
a=""; for v in $RV; do a="$a --variant $v"; done
log "START eval RAP thieu: MLVU dev"; timeout 10h $PY -m rap.eval_multi --gt_file playground/gt_qa_files/MLVU_Dev/val_qa_mc.json --video_dir Datasets/MLVU_Dev/MLVU/video --model_path $MP --output_dir outputs/mlvu_dev/pred --max_frames 128 $a >> outputs/hds6/logs/eval_rap_extra_mlvu.log 2>&1
log "START eval RAP thieu: LVBench"; timeout 10h $PY -m rap.eval_multi --gt_file playground/gt_qa_files/LVBench/val_qa.json --video_dir Datasets/LVBench/videos --model_path $MP --output_dir outputs/hds/lvbench --max_frames 128 $a >> outputs/hds6/logs/eval_rap_extra_lvb.log 2>&1
log "END eval RAP thieu"; $PY -m tools.mlvu_dev_report >> outputs/hds6/logs/report.log 2>&1
until grep -q "prep xong" outputs/hds6/STATUS.txt 2>/dev/null; do sleep 120; done
bash outputs/hds6/run.sh 1 outputs/hds6 H8cv 0.05 005 --w_ceg 0.2 $VRIPT
bash outputs/hds6/run.sh 1 outputs/hds6 H10ng 0.40 040 --w_ceg 0.2 --window 1 --gate 0
bash outputs/hds6/run.sh 1 outputs/hds5 H10ce10 0.40 040 --w_ceg 1.0 --window 1
log "QUEUE DONE gpu1"
