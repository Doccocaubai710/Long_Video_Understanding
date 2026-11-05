#!/bin/bash
# Cham LVBench + MLVU-Test cho dot hds3 (28/10), moi card 1 run + 1 moc 5% (chua co moc nao o 5% tren video dai).
#  card 1 (GPU 0): H8ce @5%  -> outputs/hds3/{lvbench,mlvu}/H8ce_r005.json,  roi uniform_r005 -> outputs/hds/{lvbench,mlvu}/
#  card 2 (GPU 1): H6ce @2%  -> outputs/hds3/{lvbench,mlvu}/H6ce_r002.json,  roi holitom_r005 -> outputs/hds/{lvbench,mlvu}/
# Moc 2% (uniform, holitom, R02E, H5, H6) da co san trong outputs/hds. Moi buoc tu bo qua cau da co (resume).
# Chay: setsid nohup bash outputs/hds3/eval_long.sh <1|2> >> outputs/hds3/logs/eval_long_card<N>.out 2>&1 &
CARD=${1:?can tham so card: 1 hoac 2}
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1   # thu muc KTV
PY=/home/hoang/miniconda3/envs/qwen25vl/bin/python
GPU=$((CARD - 1))
export DECORD_EOF_RETRY_MAX=131072 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPU
O=outputs/hds3
STATUS=$O/STATUS_card$CARD.txt
MPATH=Qwen/Qwen2.5-VL-7B-Instruct
mkdir -p $O/lvbench $O/mlvu $O/logs outputs/hds/lvbench outputs/hds/mlvu
log() { echo "$(date -u '+%Y-%m-%d %H:%M:%S') [$(hostname)] $*" >> $STATUS; }
nlines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }

do_eval() {  # do_eval <lvbench|mlvu> <output_dir> <ten> <cau hinh>
  local ds=$1 od=$2 name=$3 spec=$4 gt vd n
  case $ds in
    lvbench) gt=playground/gt_qa_files/LVBench/val_qa.json;     vd=Datasets/LVBench/videos ;;
    mlvu)    gt=playground/gt_qa_files/MLVU_Test/val_qa_mc.json; vd=Datasets/MLVU_Test/videos ;;
  esac
  n=$($PY -c "import json;print(len(json.load(open('$gt'))))")
  for attempt in 1 2 3 4; do
    [ "$(nlines $od/$ds/$name.json)" -ge "$n" ] && break
    log "START eval $ds/$name (lan $attempt)"
    timeout 14h $PY -m rap.eval_multi --gt_file $gt --video_dir $vd --model_path $MPATH --output_dir $od/$ds \
      --variant $name=$spec >> $O/logs/eval_${ds}_$name.log 2>&1
    sleep 30
  done
  if [ "$(nlines $od/$ds/$name.json)" -ge "$n" ]; then log "DONE eval $ds/$name"; else log "FAIL eval $ds/$name"; fi
  HDS_DIR=$O $PY -m tools.hds_report >> $O/logs/report.log 2>&1
}

log "EVAL_LONG START card $CARD (pid $$)"
if [ "$CARD" = 1 ]; then
  for ds in lvbench mlvu; do do_eval $ds $O H8ce_r005 ckpt:$O/train/H8ce.pt@0.05; done
  for ds in lvbench mlvu; do do_eval $ds outputs/hds uniform_r005 uniform@0.05; done
else
  for ds in lvbench mlvu; do do_eval $ds $O H6ce_r002 ckpt:$O/train/H6ce.pt@0.02; done
  for ds in lvbench mlvu; do do_eval $ds outputs/hds holitom_r005 holitom@0.05; done
fi
log "EVAL_LONG DONE card $CARD"
