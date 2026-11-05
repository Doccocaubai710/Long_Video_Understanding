#!/bin/bash
# Dot hds6 (30/10): train tren video dai THAT (Vript). Card 1 = GPU0.
#  H3cv = H3ce (10%, KL + CE-gate 0.2, cua so +-3, seed 0) + 50% mau Vript (128 frame tren toan video, caption 8 canh)
#  -> so sach voi H3ce (chi khac du lieu). Cham Video-MME + LVBench + MLVU dev.
# Chay: setsid nohup bash outputs/hds6/queue.sh > outputs/hds6/logs/queue.out 2>&1 &
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1
PY=/home/hoang/miniconda3/envs/qwen25vl/bin/python
export DECORD_EOF_RETRY_MAX=131072 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=${GPU:-0}
O=outputs/hds6; ST=$O/STATUS.txt; MP=Qwen/Qwen2.5-VL-7B-Instruct; MD=outputs/mlvu_dev/pred
log() { echo "$(date -u '+%Y-%m-%d %H:%M:%S') $*" >> $ST; }
nlines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
DEF="--pseudo_steps 0 --pseudo_prob_after 0 --w_ce 0 --w_kd 1.0 --num_steps 4000 --model_path $MP --video_root Datasets/LLaVA-Video-178K-probe"
CFG_H3="--hds 1 --slider 1 --stage2 1 --bias3d 1 --deform 1 --window 3 --self_mask 1 --gate 1 --s2_start 2000 --s1_lr_after 0.1 --w_cov 0"
VRIPT="--vript_index Datasets/Vript/vript_index.jsonl --vript_prob 0.5 --vript_scenes 8 --max_caption_tokens 768"

log "QUEUE START (GPU $CUDA_VISIBLE_DEVICES)"
until grep -q "DOWNLOAD DONE" Datasets/Vript/download.log; do sleep 120; done
log "Vript tai xong: $(ls Datasets/Vript/videos | wc -l) video"
$PY -m tools.prep_vript --root Datasets/Vript --workers 64 >> $O/logs/prep_vript.log 2>&1
log "prep xong: $(wc -l < Datasets/Vript/vript_index.jsonl) video trong index"

name=H3cv
for a in 1 2 3; do
  grep -q "\[train_rap\] done" $O/logs/$name.train.log 2>/dev/null && break
  log "START train $name (lan $a)"
  timeout 20h $PY -m rap.train_rap_elastic --out $O/train/$name.pt --resume $O/train/$name.pt $DEF --retention 0.10 $CFG_H3 --w_ceg 0.2 $VRIPT >> $O/logs/$name.train.log 2>&1
  sleep 30
done
grep -q "\[train_rap\] done" $O/logs/$name.train.log || { log "FAIL train $name"; exit 1; }
log "DONE train $name"
V="${name}_r010=ckpt:$O/train/$name.pt@0.10"
ev() {  # ev <gt> <video_dir> <out_dir> <tag>
  local n; n=$($PY -c "import json;print(len(json.load(open('$1'))))")
  for a in 1 2 3; do
    [ "$(nlines $3/${name}_r010.json)" -ge "$n" ] && break
    log "START eval $4 (lan $a)"
    timeout 16h $PY -m rap.eval_multi --gt_file $1 --video_dir $2 --model_path $MP --output_dir $3 --max_frames 128 --variant $V >> $O/logs/eval_$4.log 2>&1
    sleep 20
  done
  log "END eval $4: $(nlines $3/${name}_r010.json)/$n"
}
ev playground/gt_qa_files/VideoMME/val_qa.json Datasets/Video-MME/data/data $O/videomme videomme
ev playground/gt_qa_files/LVBench/val_qa.json Datasets/LVBench/videos $O/lvbench lvbench
ev playground/gt_qa_files/MLVU_Dev/val_qa_mc.json Datasets/MLVU_Dev/MLVU/video $MD mlvudev
HDS_DIR=$O $PY -m tools.hds_report >> $O/logs/report.log 2>&1
$PY -m tools.mlvu_dev_report >> $O/logs/report.log 2>&1
log "QUEUE DONE"
