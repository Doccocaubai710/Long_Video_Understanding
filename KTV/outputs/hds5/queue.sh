#!/bin/bash
# Dot hds5 (29/10 -> 30/10, ~20 h): cham baseline tren MLVU dev + LVBench, tuning CE-gate o 25% / 40%. 2 card (card N = GPU N-1).
# Card 1:
#  1  MLVU dev: VidCom2 40/25/10/5/2%, DyCoke 40/25%, H3ce @10, H8ce @5, H6ce @2              [~3 h]
#  2  LVBench:  VidCom2 40/25/10/5/2%, DyCoke 40/25%                                           [~1,5 h]
#  3  H9ce5   = H9ce (25%) nhung --w_ceg 0.5; cham Video-MME + LVBench + MLVU dev               [~8,5 h]
#  4  H9ce10  = H9ce (25%) nhung --w_ceg 1.0                                                    [~8,5 h]
#  5  (lap cho GPU khong ranh) H3ce seed 1 @10%
# Card 2:
#  1  MLVU dev: FastVID 40/25/10/5/2%, Uniform + HoliTom 5/2%                                  [~3 h]
#  2  LVBench:  FastVID 40/25/10/5/2%                                                          [~1,5 h]
#  3  H10ce5  = H10ce (40%) nhung --w_ceg 0.5                                                  [~9 h]
#  4  H10ce10 = H10ce (40%) nhung --w_ceg 1.0                                                  [~9 h]
#  5  (lap cho GPU khong ranh) H3ce seed 2 @10%
# Moi buoc tu tiep tuc (train --resume, cham bo qua cau da co); buoc da xong thi bo qua. OOM khi train -> lan sau --window 1.
# Chay: bash outputs/hds5/start.sh <1|2>   |   Ket qua: outputs/hds5/KET_QUA.txt, outputs/mlvu_dev/KET_QUA.txt
CARD=${1:?can tham so card: 1 hoac 2}
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1   # thu muc KTV
PY=${PY:-/home/hoang/miniconda3/envs/qwen25vl/bin/python}
GPU=$((CARD - 1))
export DECORD_EOF_RETRY_MAX=131072 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPU
O=outputs/hds5
STATUS=$O/STATUS_card$CARD.txt
MPATH=Qwen/Qwen2.5-VL-7B-Instruct
VROOT=Datasets/LLaVA-Video-178K-probe
MD=outputs/mlvu_dev/pred
mkdir -p $O/train $O/videomme $O/lvbench $O/logs $MD outputs/hds/lvbench
log() { echo "$(date -u '+%Y-%m-%d %H:%M:%S') [$(hostname)] $*" >> $STATUS; }
report() {
  HDS_DIR=$O $PY -m tools.hds_report >> $O/logs/report.log 2>&1
  HDS_DIR=outputs/hds $PY -m tools.hds_report >> $O/logs/report.log 2>&1
  $PY -m tools.mlvu_dev_report >> $O/logs/report.log 2>&1
}
nlines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }

DEF="--pseudo_steps 0 --pseudo_prob_after 0 --w_ce 0 --w_kd 1.0 --num_steps 4000 --model_path $MPATH --video_root $VROOT"
M3DBDP="--bias3d 1 --deform 1 --window 3 --self_mask 1 --gate 1"
CFG_H3="--hds 1 --slider 1 --stage2 1 $M3DBDP --s2_start 2000 --s1_lr_after 0.1 --w_cov 0"

wait_gpu() {
  local free
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU 2>/dev/null | head -1)
    [ -n "$free" ] && [ "$free" -ge 32000 ] && return 0
    log "WAIT GPU: chi con ${free:-?} MiB trong, cho 5 phut"; sleep 300
  done
}
do_train() {  # do_train <ten> <tham so ...>
  local name=$1; shift
  local extra=""
  for attempt in 1 2 3 4; do
    grep -q "\[train_rap\] done" $O/logs/$name.train.log 2>/dev/null && return 0
    wait_gpu
    if [ -z "$extra" ] && grep -qi "out of memory" $O/logs/$name.train.log 2>/dev/null; then
      extra="--window 1"; log "OOM o lan truoc -> chay lai $name voi --window 1 tu checkpoint"
    fi
    log "START train $name (lan $attempt)"
    timeout 20h $PY -m rap.train_rap_elastic --out $O/train/$name.pt --resume $O/train/$name.pt $DEF "$@" $extra >> $O/logs/$name.train.log 2>&1
    sleep 30
  done
  grep -q "\[train_rap\] done" $O/logs/$name.train.log || { log "FAIL train $name"; return 1; }
}
gt_of() { case $1 in
  videomme) echo "playground/gt_qa_files/VideoMME/val_qa.json Datasets/Video-MME/data/data" ;;
  lvbench)  echo "playground/gt_qa_files/LVBench/val_qa.json Datasets/LVBench/videos" ;;
  mlvudev)  echo "playground/gt_qa_files/MLVU_Dev/val_qa_mc.json Datasets/MLVU_Dev/MLVU/video" ;;
esac; }
do_eval() {  # do_eval <bo> <thu muc du doan> <tag log> <ten=cau_hinh> ...
  local ds=$1 od=$2 tag=$3; shift 3
  local gt vd n v missing args
  read gt vd <<< "$(gt_of $ds)"
  n=$($PY -c "import json;print(len(json.load(open('$gt'))))")
  for attempt in 1 2 3 4; do
    missing=0; args=""
    for v in "$@"; do [ "$(nlines $od/${v%%=*}.json)" -ge "$n" ] || missing=1; args="$args --variant $v"; done
    [ $missing = 0 ] && break
    wait_gpu
    log "START eval $ds/$tag (lan $attempt)"
    timeout 16h $PY -m rap.eval_multi --gt_file $gt --video_dir $vd --model_path $MPATH --output_dir $od \
      --max_frames 128 $args >> $O/logs/eval_${ds}_$tag.log 2>&1
    sleep 30
  done
  missing=0; for v in "$@"; do [ "$(nlines $od/${v%%=*}.json)" -ge "$n" ] || missing=1; done
  if [ $missing = 0 ]; then log "DONE eval $ds/$tag"; else log "FAIL eval $ds/$tag"; fi
  report
}
ck() { echo "$1=ckpt:$2@$3"; }
run() {  # run <ten> <muc> <ma muc> <tham so train ...>: train roi cham Video-MME + LVBench + MLVU dev
  local name=$1 r=$2 rc=$3; shift 3
  do_train $name --retention $r $CFG_H3 "$@" || return 1
  local v; v=$(ck ${name}_r$rc $O/train/$name.pt $r)
  do_eval videomme $O/videomme $name $v
  do_eval lvbench $O/lvbench $name $v
  do_eval mlvudev $MD $name $v
}

log "QUEUE START card $CARD (pid $$, $($PY -c 'import torch;print(torch.cuda.get_device_name(0))' 2>/dev/null))"
if [ "$CARD" = 1 ]; then
  BA="vidcom2_r040=vidcom2@0.40 vidcom2_r025=vidcom2@0.25 vidcom2_r010=vidcom2@0.10 vidcom2_r005=vidcom2@0.05 vidcom2_r002=vidcom2@0.02
      dycoke_r040=dycoke@0.40 dycoke_r025=dycoke@0.25"
  do_eval mlvudev $MD baseA $BA $(ck H3ce_r010 outputs/hds4/train/H3ce.pt 0.10) \
    $(ck H8ce_r005 outputs/hds3/train/H8ce.pt 0.05) $(ck H6ce_r002 outputs/hds3/train/H6ce.pt 0.02)
  do_eval lvbench outputs/hds/lvbench baseA $BA
  run H9ce5  0.25 025 --w_ceg 0.5 --window 1
  run H9ce10 0.25 025 --w_ceg 1.0 --window 1
  run H3ces1 0.10 010 --w_ceg 0.2 --seed 1
else
  BB="fastvid_r040=fastvid@0.40 fastvid_r025=fastvid@0.25 fastvid_r010=fastvid@0.10 fastvid_r005=fastvid@0.05 fastvid_r002=fastvid@0.02"
  do_eval mlvudev $MD baseB $BB uniform_r005=uniform@0.05 holitom_r005=holitom@0.05 uniform_r002=uniform@0.02 holitom_r002=holitom@0.02
  do_eval lvbench outputs/hds/lvbench baseB $BB
  run H10ce5  0.40 040 --w_ceg 0.5 --window 1
  run H10ce10 0.40 040 --w_ceg 1.0 --window 1
  run H3ces2  0.10 010 --w_ceg 0.2 --seed 2
fi
log "QUEUE DONE card $CARD"
report
