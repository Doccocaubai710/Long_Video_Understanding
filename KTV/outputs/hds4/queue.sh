#!/bin/bash
# Dot hds4 (28/10 -> 29/10): tap trung muc 10% + cham video dai o 40% / 25%. 2 card cung may (card N = GPU N-1).
# Card 1:
#  1  Cham LVBench + MLVU o 40% va 25%: H10ce, H9ce (-> outputs/hds3), Uniform / HoliTom 40% + 25% (-> outputs/hds)  [~5 h]
#  2  H4ce  = H4 (video ghep 3 clip) + CE-gate 0,2 @10%; cham Video-MME + LVBench + MLVU                       [~12 h]
#  3  F256  = 256 frame, cung ~1,6K token (muc 5%): H4s1, H5 (elastic), Uniform; chay thu 20 cau truoc         [~5 h]
# Card 2:
#  1  H12   = H4 nhung ghep 6 clip (caption 768 token) @10%; cham ca 3 bo                                       [~13 h]
#  2  H3ce  = H3 + CE-gate @10% (khong ghep); cham ca 3 bo, LVBench/MLVU cham kem H3s1                         [~7,5 h]
# Moi buoc tu tiep tuc (train --resume, cham bo qua cau da co); buoc da xong thi bo qua. OOM khi train -> lan sau --window 1.
# Chay: bash outputs/hds4/start.sh <1|2>   |   Ket qua: outputs/hds4/KET_QUA.txt
CARD=${1:?can tham so card: 1 hoac 2}
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1   # thu muc KTV
PY=${PY:-/home/hoang/miniconda3/envs/qwen25vl/bin/python}
GPU=$((CARD - 1))
export DECORD_EOF_RETRY_MAX=131072 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPU
O=outputs/hds4
STATUS=$O/STATUS_card$CARD.txt
MPATH=Qwen/Qwen2.5-VL-7B-Instruct
VROOT=Datasets/LLaVA-Video-178K-probe
mkdir -p $O/train $O/videomme $O/lvbench $O/mlvu $O/logs outputs/hds/lvbench outputs/hds/mlvu outputs/hds3/lvbench outputs/hds3/mlvu
log() { echo "$(date -u '+%Y-%m-%d %H:%M:%S') [$(hostname)] $*" >> $STATUS; }
report() { HDS_DIR=$O $PY -m tools.hds_report >> $O/logs/report.log 2>&1; HDS_DIR=outputs/hds3 $PY -m tools.hds_report >> $O/logs/report.log 2>&1; }
nlines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }

DEF="--pseudo_steps 0 --pseudo_prob_after 0 --w_ce 0 --w_kd 1.0 --num_steps 4000 --model_path $MPATH --video_root $VROOT"
M3DBDP="--bias3d 1 --deform 1 --window 3 --self_mask 1 --gate 1"
CFG_H3="--hds 1 --slider 1 --stage2 1 $M3DBDP --s2_start 2000 --s1_lr_after 0.1 --w_cov 0"
CEG="--w_ceg 0.2"
STITCH3="--stitch 3 --stitch_prob 0.5 --max_caption_tokens 576"   # = H4
STITCH6="--stitch 6 --stitch_prob 0.5 --max_caption_tokens 768"

wait_gpu() {
  local free
  # cho phan cham video dai cua dot hds3 tren card nay xong
  while pgrep -f "^bash outputs/hds3/eval_long.sh $CARD" > /dev/null; do sleep 120; done
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
  mlvu)     echo "playground/gt_qa_files/MLVU_Test/val_qa_mc.json Datasets/MLVU_Test/videos" ;;
esac; }
do_eval() {  # do_eval <bo> <thu muc ra> <tag log> <so frame> <ten=cau_hinh> ...
  local ds=$1 od=$2 tag=$3 nf=$4; shift 4
  local gt vd n v missing args
  read gt vd <<< "$(gt_of $ds)"
  n=$($PY -c "import json;print(len(json.load(open('$gt'))))")
  for attempt in 1 2 3 4; do
    missing=0; args=""
    for v in "$@"; do [ "$(nlines $od/$ds/${v%%=*}.json)" -ge "$n" ] || missing=1; args="$args --variant $v"; done
    [ $missing = 0 ] && break
    wait_gpu
    log "START eval $ds/$tag (lan $attempt)"
    timeout 16h $PY -m rap.eval_multi --gt_file $gt --video_dir $vd --model_path $MPATH --output_dir $od/$ds \
      --max_frames $nf $args >> $O/logs/eval_${ds}_$tag.log 2>&1
    sleep 30
  done
  missing=0; for v in "$@"; do [ "$(nlines $od/$ds/${v%%=*}.json)" -ge "$n" ] || missing=1; done
  if [ $missing = 0 ]; then log "DONE eval $ds/$tag"; else log "FAIL eval $ds/$tag"; fi
  report
}
ck() { echo "$1=ckpt:$2@$3"; }
run10() {  # run10 <ten> <tham so train ...>: train o 10% roi cham ca 3 bo
  local name=$1; shift
  do_train $name --retention 0.10 $CFG_H3 "$@" || return 1
  local v; v=$(ck ${name}_r010 $O/train/$name.pt 0.10)
  do_eval videomme $O $name 128 $v
  do_eval lvbench $O $name 128 $v
  do_eval mlvu $O $name 128 $v
}

log "QUEUE START card $CARD (pid $$, $($PY -c 'import torch;print(torch.cuda.get_device_name(0))' 2>/dev/null))"
if [ "$CARD" = 1 ]; then
  # 1. video dai o 40% / 25%
  for ds in lvbench mlvu; do
    do_eval $ds outputs/hds3 C40_25 128 $(ck H10ce_r040 outputs/hds3/train/H10ce.pt 0.40) $(ck H9ce_r025 outputs/hds3/train/H9ce.pt 0.25)
    do_eval $ds outputs/hds base40_25 128 uniform_r040=uniform@0.40 holitom_r040=holitom@0.40 uniform_r025=uniform@0.25 holitom_r025=holitom@0.25
  done
  # 2. H4ce
  run10 H4ce $STITCH3 $CEG
  # 3. 256 frame, muc 5% (~1,6K token = 128 frame o 10%); thu 20 cau truoc
  F256="$(ck F256_H4s1_r005 outputs/hds2/train/H4s1.pt 0.05) $(ck F256_H5_r005 outputs/hds/train/H5.pt 0.05) F256_uniform_r005=uniform@0.05"
  mkdir -p $O/f256_test
  log "START thu 256 frame (20 cau)"
  read gt vd <<< "$(gt_of videomme)"
  if timeout 2h $PY -m rap.eval_multi --gt_file $gt --video_dir $vd --model_path $MPATH --output_dir $O/f256_test --max_frames 256 --limit 20 \
       $(for v in $F256; do echo --variant $v; done) >> $O/logs/f256_test.log 2>&1 && [ "$(nlines $O/f256_test/F256_H5_r005.json)" -ge 20 ]; then
    log "OK thu 256 frame -> cham day du"
    for ds in videomme lvbench mlvu; do do_eval $ds $O F256 256 $F256; done
  else
    log "FAIL thu 256 frame (xem $O/logs/f256_test.log) -> bo qua buoc 3"
  fi
else
  # 1. H12: ghep 6 clip
  run10 H12 $STITCH6
  # 2. H3ce: CE-gate khong ghep; cham kem H3s1 tren video dai
  do_train H3ce --retention 0.10 $CFG_H3 $CEG && {
    do_eval videomme $O H3ce 128 $(ck H3ce_r010 $O/train/H3ce.pt 0.10)
    for ds in lvbench mlvu; do do_eval $ds $O H3ce 128 $(ck H3ce_r010 $O/train/H3ce.pt 0.10) $(ck H3s1_r010 outputs/hds2/train/H3s1.pt 0.10); done
  }
fi
log "QUEUE DONE card $CARD"
report
