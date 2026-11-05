#!/bin/bash
# Giu card chay lien tuc den khi queue xong.  bash outputs/hds5/keepalive.sh <1|2>
#  - queue chet ma chua "QUEUE DONE card N" -> chay lai (queue tu --resume / bo qua buoc da xong)
#  - 90 phut khong co tien trien trong khi job cua card nay dang chay -> dung job do (nhan job theo CUDA_VISIBLE_DEVICES)
#  - moi 30 phut viet lai outputs/hds5/KET_QUA.txt
CARD=${1:?can tham so card}
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1   # thu muc KTV
O=outputs/hds5
PY=/home/hoang/miniconda3/envs/qwen25vl/bin/python
GPU=$((CARD - 1))
ST=$O/STATUS_card$CARD.txt
log() { echo "$(date -u '+%Y-%m-%d %H:%M:%S') [$(hostname)] KEEPALIVE $*" >> $ST; }
start_queue() { log "chay lai queue"; setsid nohup bash $O/queue.sh $CARD >> $O/logs/queue_card$CARD.out 2>&1 < /dev/null & }
my_job() {  # job train/cham cua dot hds4 dang chay tren GPU cua card nay
  local p
  for p in $(pgrep -f "rap\.(train_rap_elastic|eval_multi) "); do
    tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$GPU" || continue
    readlink /proc/$p/fd/1 2>/dev/null | grep -q "$O/logs/" && { echo $p; return; }
  done
}
log "START (pid $$)"
i=0
while true; do
  sleep 300; i=$((i + 1))
  [ $((i % 6)) = 0 ] && HDS_DIR=$O $PY -m tools.hds_report >> $O/logs/report.log 2>&1
  if ! pgrep -f "^bash $O/queue.sh $CARD" > /dev/null; then
    if grep -q "QUEUE DONE card $CARD" $ST; then log "queue da xong het, dung keepalive"; exit 0; fi
    log "queue khong con chay"; start_queue; continue
  fi
  job=$(my_job)
  [ -z "$job" ] && continue
  out=$(readlink /proc/$job/fd/1 2>/dev/null)
  # tien do = file moi nhat trong: log cua job, metrics khi train, file du doan khi cham
  newest=$(ls -t $out $O/train/*.metrics.jsonl $O/*/*.json outputs/hds/lvbench/*.json outputs/mlvu_dev/pred/*.json 2>/dev/null | head -1)
  age=$(( $(date +%s) - $(stat -c %Y "$newest" 2>/dev/null || echo 0) ))
  if [ $age -gt 5400 ]; then
    log "khong tien trien ${age}s (file moi nhat: $newest) -> dung pid $job, queue tu chay lai"
    kill $job
  fi
done
