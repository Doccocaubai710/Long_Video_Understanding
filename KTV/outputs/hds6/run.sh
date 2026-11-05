#!/bin/bash
# run.sh <gpu> <dot> <ten> <muc> <ma muc> <tham so train them...> : train (resume duoc) roi cham Video-MME + LVBench + MLVU dev.
# Vd: bash outputs/hds6/run.sh 1 outputs/hds6 H8cv 0.05 005 --w_ceg 0.2 $VRIPT
GPUID=$1; O=$2; name=$3; r=$4; rc=$5; shift 5
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1
PY=/home/hoang/miniconda3/envs/qwen25vl/bin/python
export DECORD_EOF_RETRY_MAX=131072 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPUID
ST=outputs/hds6/STATUS_gpu$GPUID.txt; MP=Qwen/Qwen2.5-VL-7B-Instruct; MD=outputs/mlvu_dev/pred
mkdir -p $O/train $O/videomme $O/lvbench $O/logs
log() { echo "$(date -u '+%Y-%m-%d %H:%M:%S') $*" >> $ST; }
nlines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
DEF="--pseudo_steps 0 --pseudo_prob_after 0 --w_ce 0 --w_kd 1.0 --num_steps 4000 --model_path $MP --video_root Datasets/LLaVA-Video-178K-probe"
CFG_H3="--hds 1 --slider 1 --stage2 1 --bias3d 1 --deform 1 --window 3 --self_mask 1 --gate 1 --s2_start 2000 --s1_lr_after 0.1 --w_cov 0"
extra=""
for a in 1 2 3 4; do
  grep -q "\[train_rap\] done" $O/logs/$name.train.log 2>/dev/null && break
  if [ -z "$extra" ] && grep -qi "out of memory" $O/logs/$name.train.log 2>/dev/null; then extra="--window 1"; log "OOM -> $name chay lai voi --window 1"; fi
  log "START train $name (lan $a): r=$r $*"
  timeout 20h $PY -m rap.train_rap_elastic --out $O/train/$name.pt --resume $O/train/$name.pt $DEF --retention $r $CFG_H3 "$@" $extra >> $O/logs/$name.train.log 2>&1
  sleep 30
done
grep -q "\[train_rap\] done" $O/logs/$name.train.log || { log "FAIL train $name"; exit 1; }
log "DONE train $name"
V="${name}_r$rc=ckpt:$O/train/$name.pt@$r"
ev() {
  local n; n=$($PY -c "import json;print(len(json.load(open('$1'))))")
  for a in 1 2 3; do
    [ "$(nlines $3/${name}_r$rc.json)" -ge "$n" ] && break
    log "START eval $name/$4 (lan $a)"
    timeout 16h $PY -m rap.eval_multi --gt_file $1 --video_dir $2 --model_path $MP --output_dir $3 --max_frames 128 --variant $V >> $O/logs/eval_${4}_$name.log 2>&1
    sleep 20
  done
  log "END eval $name/$4: $(nlines $3/${name}_r$rc.json)/$n"
}
ev playground/gt_qa_files/VideoMME/val_qa.json Datasets/Video-MME/data/data $O/videomme videomme
ev playground/gt_qa_files/LVBench/val_qa.json Datasets/LVBench/videos $O/lvbench lvbench
ev playground/gt_qa_files/MLVU_Dev/val_qa_mc.json Datasets/MLVU_Dev/MLVU/video $MD mlvudev
HDS_DIR=$O $PY -m tools.hds_report >> $O/logs/report.log 2>&1; $PY -m tools.mlvu_dev_report >> $O/logs/report.log 2>&1
log "DONE all $name"
