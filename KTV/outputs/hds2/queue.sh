#!/bin/bash
# Dot kiem chung HDS-MAP 3D, 26/10 -> 27/10, 2 card A100 (moi may 1 card, dung chung Lustre). Muc 10%, ~24 gio.
# Moi run train xong la cham Video-MME ngay (co ket qua som, neu may bi ngat giua chung van giu duoc ket qua da xong).
# Card 1 -- kiem chung ket qua tot nhat (H4):
#  1  R10S    RAP R10E + video dai ghep (doi chung cua H4: loi ich do HDS hay do du lieu?)   [train ~9,5 h + cham ~1,1 h]
#  2  H4s1    H4 seed 1 (63,33 co lap lai duoc khong?)                                     [train ~9,5 h + cham ~1,1 h]
#  3  cham    LVBench + MLVU (R10S, H4s1)                                                    [~2,5 h]
# Card 2 -- do dao dong seed + ablation giai doan 2 (H3, 10%), moi run ~5,2 h train + ~1,1 h cham:
#  1  H3s1    H3 seed 1
#  2  H3noDF  H3 --deform 0     (hop nhin tu dich co giup khong)
#  3  H3noSM  H3 --self_mask 0
#  4  H3noGate H3 --gate 0      (chi kip neu khong bi ngat)
# Moi buoc tu tiep tuc (train --resume, cham bo qua cau da co); buoc da xong thi bo qua.
# Chay: bash outputs/hds2/start.sh <1|2>   |   Ket qua: outputs/hds2/KET_QUA.txt
CARD=${1:?can tham so card: 1 hoac 2}
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1   # thu muc KTV
PY=${PY:-/home/hoang/miniconda3/envs/qwen25vl/bin/python}
export DECORD_EOF_RETRY_MAX=131072 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
O=outputs/hds2
STATUS=$O/STATUS_card$CARD.txt
MPATH=Qwen/Qwen2.5-VL-7B-Instruct
VROOT=Datasets/LLaVA-Video-178K-probe
mkdir -p $O/train $O/videomme $O/lvbench $O/mlvu $O/logs
log() { echo "$(date -u '+%Y-%m-%d %H:%M:%S') [$(hostname)] $*" >> $STATUS; }
report() { HDS_DIR=$O $PY -m tools.hds_report >> $O/logs/report.log 2>&1; }
nlines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }

DEF="--pseudo_steps 0 --pseudo_prob_after 0 --w_ce 0 --w_kd 1.0 --num_steps 4000 --model_path $MPATH --video_root $VROOT"
M3DBDP="--bias3d 1 --deform 1 --window 3 --self_mask 1 --gate 1"
CFG_H3="--hds 1 --slider 1 --stage2 1 $M3DBDP --s2_start 2000 --s1_lr_after 0.1 --w_cov 0 --retention 0.10"
CFG_R10E="--retention 0.10 --w_cov 0.1 --w_dpp 0.05 --window 1"          # = args trong outputs/checkpoints/R10E.pt
STITCH="--stitch 3 --stitch_prob 0.5 --max_caption_tokens 576"           # = H4

do_train() {  # do_train <ten> <tham so ...>
  local name=$1; shift
  # neu queue bi khoi dong lai trong khi run nay van dang train (tien trinh mo coi) thi cho no xong, khong chay chong
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
do_eval() {  # do_eval <bo: videomme|lvbench|mlvu> <ten log> <--variant ...>
  local ds=$1 tag=$2; shift 2
  local gt vd n
  case $ds in
    videomme) gt=playground/gt_qa_files/VideoMME/val_qa.json; vd=Datasets/Video-MME/data/data ;;
    lvbench)  gt=playground/gt_qa_files/LVBench/val_qa.json;  vd=Datasets/LVBench/videos ;;
    mlvu)     gt=playground/gt_qa_files/MLVU_Test/val_qa_mc.json; vd=Datasets/MLVU_Test/videos ;;
  esac
  n=$($PY -c "import json;print(len(json.load(open('$gt'))))")
  for attempt in 1 2 3 4; do
    local missing=0
    for v in "$@"; do [ "$v" = "--variant" ] && continue; f=$O/$ds/${v%%=*}.json; [ "$(nlines $f)" -ge "$n" ] || missing=1; done
    [ $missing = 0 ] && break
    wait_gpu
    log "START eval $ds/$tag (lan $attempt)"
    timeout 14h $PY -m rap.eval_multi --gt_file $gt --video_dir $vd --model_path $MPATH --output_dir $O/$ds "$@" >> $O/logs/eval_${ds}_$tag.log 2>&1
    sleep 30
  done
  log "DONE eval $ds/$tag"; report
}
wait_gpu() {  # nguoi khac dang dung GPU (con < 32 GB trong) -> cho, khong dot het so lan thu
  local free
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 2>/dev/null | head -1)
    [ -n "$free" ] && [ "$free" -ge 32000 ] && return 0
    log "WAIT GPU: chi con ${free:-?} MiB trong, cho 5 phut"; sleep 300
  done
}
ok() { grep -q "\[train_rap\] done" $O/logs/$1.train.log 2>/dev/null; }
ck() { ok $1 && echo "--variant $1_r010=ckpt:$O/train/$1.pt@0.10"; }   # chi them bien the da train xong

log "QUEUE START card $CARD (pid $$, $($PY -c 'import torch;print(torch.cuda.get_device_name(0))' 2>/dev/null))"
if [ "$CARD" = 1 ]; then
  do_train R10S $CFG_R10E $STITCH && do_eval videomme R10S $(ck R10S)
  do_train H4s1 $CFG_H3 $STITCH --seed 1 && do_eval videomme H4s1 $(ck H4s1)
  V="$(ck R10S) $(ck H4s1)"
  [ -n "${V// }" ] && do_eval lvbench C1 $V
  [ -n "${V// }" ] && do_eval mlvu C1 $V
elif [ "$CARD" = 2 ]; then
  do_train H3s1 $CFG_H3 --seed 1 && do_eval videomme H3s1 $(ck H3s1)
  do_train H3noDF $CFG_H3 --deform 0 && do_eval videomme H3noDF $(ck H3noDF)
  do_train H3noSM $CFG_H3 --self_mask 0 && do_eval videomme H3noSM $(ck H3noSM)
  do_train H3noGate $CFG_H3 --gate 0 && do_eval videomme H3noGate $(ck H3noGate)
fi
log "QUEUE DONE card $CARD"
report
