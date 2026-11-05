#!/bin/bash
# Dot HDS-MAP 3D o muc nen thap (40%, 25%) va sau (5%, 2%), 27/10 -> 28/10, chi card 1.
# Moi run train xong la cham Video-MME ngay o dung muc cua no.
#  1  H10ce  H3 o 40%, KL + CE-gate (--w_ceg 0.2): CE chi tren token caption ma ban day du doan dung top-1
#            -> muc tieu vuot dense (65,15). KL thuan tu dat tran o muc dense.               [train ~6,5 h + cham ~1,3 h]
#     Neu H10ce >= ELA 40% (64,59, ban train tot nhat cu o 40%) thi coi CE-gate la "on" va dung cho cac run sau.
#  2  H9[ce] H3 o 25%                                                                          [~6 h + ~1,1 h]
#  3  H8[ce] H3 o 5%                                                                           [~5 h + ~1 h]
#  4  H6ce / H6s1  H3 o 2% (CE on: H6ce, khong: H6 seed 1)  -- chi khi con thoi gian            [~5 h + ~1 h]
# Giai doan 2 o 40% va 25% dung cua so ±1 (±3 tran GPU 40 GB); 5% va 2% giu ±3 nhu H3.
# Moi buoc tu tiep tuc (train --resume, cham bo qua cau da co); buoc da xong thi bo qua.
# Chay: bash outputs/hds3/start.sh 1   |   Ket qua: outputs/hds3/KET_QUA.txt
CARD=${1:?can tham so card: 1}
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1   # thu muc KTV
PY=${PY:-/home/hoang/miniconda3/envs/qwen25vl/bin/python}
GPU=$((CARD - 1))   # card 1 -> GPU 0, card 2 -> GPU 1 (cung may)
export DECORD_EOF_RETRY_MAX=131072 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPU
O=outputs/hds3
STATUS=$O/STATUS_card$CARD.txt
MPATH=Qwen/Qwen2.5-VL-7B-Instruct
VROOT=Datasets/LLaVA-Video-178K-probe
mkdir -p $O/train $O/videomme $O/logs
log() { echo "$(date -u '+%Y-%m-%d %H:%M:%S') [$(hostname)] $*" >> $STATUS; }
report() { HDS_DIR=$O $PY -m tools.hds_report >> $O/logs/report.log 2>&1; }
nlines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }

DEF="--pseudo_steps 0 --pseudo_prob_after 0 --w_ce 0 --w_kd 1.0 --num_steps 4000 --model_path $MPATH --video_root $VROOT"
M3DBDP="--bias3d 1 --deform 1 --window 3 --self_mask 1 --gate 1"
CFG_H3="--hds 1 --slider 1 --stage2 1 $M3DBDP --s2_start 2000 --s1_lr_after 0.1 --w_cov 0"
CEG="--w_ceg 0.2"

do_train() {  # do_train <ten> <tham so ...>
  local name=$1; shift
  while pgrep -f "rap.train_rap_elastic --out $O/train/$name.pt" > /dev/null; do sleep 60; done
  for attempt in 1 2 3 4; do
    grep -q "\[train_rap\] done" $O/logs/$name.train.log 2>/dev/null && return 0
    wait_gpu
    log "START train $name (lan $attempt)"
    timeout 20h $PY -m rap.train_rap_elastic --out $O/train/$name.pt --resume $O/train/$name.pt $DEF "$@" >> $O/logs/$name.train.log 2>&1
    sleep 30
  done
  grep -q "\[train_rap\] done" $O/logs/$name.train.log || { log "FAIL train $name"; return 1; }
}
do_eval() {  # do_eval <ten> <muc 0.40> <ma muc 040>: cham Video-MME
  local name=$1 r=$2 rc=$3
  local gt=playground/gt_qa_files/VideoMME/val_qa.json vd=Datasets/Video-MME/data/data f=$O/videomme/${name}_r$rc.json
  local n; n=$($PY -c "import json;print(len(json.load(open('$gt'))))")
  for attempt in 1 2 3 4; do
    [ "$(nlines $f)" -ge "$n" ] && break
    wait_gpu
    log "START eval videomme/$name (lan $attempt)"
    timeout 14h $PY -m rap.eval_multi --gt_file $gt --video_dir $vd --model_path $MPATH --output_dir $O/videomme \
      --variant ${name}_r$rc=ckpt:$O/train/$name.pt@$r >> $O/logs/eval_videomme_$name.log 2>&1
    sleep 30
  done
  log "DONE eval videomme/$name ($(score $f))"; report
}
wait_gpu() {
  local free
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU 2>/dev/null | head -1)
    [ -n "$free" ] && [ "$free" -ge 32000 ] && return 0
    log "WAIT GPU: chi con ${free:-?} MiB trong, cho 5 phut"; sleep 300
  done
}
score() {  # do chinh xac Video-MME (%) cua 1 file du doan, -1 neu chua du 2.700 cau
  $PY -c "
import sys; from tools.hds_report import load, acc
d = load('$1', True); print(f'{acc(d):.2f}' if len(d) >= 2700 else -1)" 2>/dev/null || echo -1
}
run() {  # run <ten> <muc> <ma muc> <tham so ...>
  local name=$1 r=$2 rc=$3; shift 3
  do_train $name --retention $r $CFG_H3 "$@" && do_eval $name $r $rc
}

log "QUEUE START card $CARD (pid $$, $($PY -c 'import torch;print(torch.cuda.get_device_name(0))' 2>/dev/null))"
if [ "$CARD" = 1 ]; then
  run H10ce 0.40 040 $CEG --window 1   # ±3 tran bo nho o 40% (OOM o buoc 2001) -> ±1 nhu RAP R40E
fi
S=$(score $O/videomme/H10ce_r040.json)
if $PY -c "import sys; sys.exit(0 if float('$S') >= 64.59 else 1)"; then
  SUF=ce; EXTRA=$CEG; log "CE-gate ON (H10ce = $S >= ELA 40% 64,59): cac run sau dung --w_ceg 0.2"
else
  SUF=; EXTRA=; log "CE-gate OFF (H10ce = $S < 64,59): cac run sau chi KL"
fi
# 28/10: chia 2 card cung may -- card 1: H9 (da xong) + H8 o 5%; card 2: H6 o 2%
if [ "$CARD" = 1 ]; then
  run H9$SUF 0.25 025 $EXTRA --window 1
  run H8$SUF 0.05 005 $EXTRA
else
  if [ -n "$SUF" ]; then run H6ce 0.02 002 $CEG; else run H6s1 0.02 002 --seed 1; fi
fi
log "QUEUE DONE card $CARD"
report
