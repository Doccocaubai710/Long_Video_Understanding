#!/bin/bash
# Giu card chay lien tuc den khi queue xong, tach khoi phien Claude.  bash outputs/hds2/keepalive.sh <1|2>
#  - queue chet ma chua "QUEUE DONE card N" -> chay lai (queue tu --resume / bo qua buoc da xong)
#  - 90 phut khong co file tien do nao cua may nay thay doi trong khi co tien trinh train/cham -> dung tien trinh do
#  - moi 30 phut viet lai outputs/hds2/KET_QUA.txt
CARD=${1:?can tham so card}
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1   # thu muc KTV
O=outputs/hds2
PY=/home/hoang/miniconda3/envs/qwen25vl/bin/python
ST=$O/STATUS_card$CARD.txt
log() { echo "$(date -u '+%Y-%m-%d %H:%M:%S') [$(hostname)] KEEPALIVE $*" >> $ST; }
start_queue() { log "chay lai queue"; setsid nohup bash $O/queue.sh $CARD >> $O/logs/queue_card$CARD.out 2>&1 < /dev/null & }
log "START (pid $$)"
i=0
while true; do
  sleep 300; i=$((i + 1))
  [ $((i % 6)) = 0 ] && HDS_DIR=$O $PY -m tools.hds_report >> $O/logs/report.log 2>&1
  if ! pgrep -f "^bash $O/queue.sh $CARD" > /dev/null; then
    if grep -q "QUEUE DONE card $CARD" $ST; then log "queue da xong het, dung keepalive"; exit 0; fi
    log "queue khong con chay"; start_queue; continue
  fi
  job=$(pgrep -f "rap\.(train_rap_elastic|eval_multi) .*$O" | head -1)
  [ -z "$job" ] && continue
  # file tien do cua job dang chay: log cua chinh tien trinh do (lay tu /proc)
  newest=$(ls -t $(readlink /proc/$job/fd/1 2>/dev/null) $O/train/*.metrics.jsonl $ST 2>/dev/null | head -1)
  age=$(( $(date +%s) - $(stat -c %Y "$newest") ))
  if [ $age -gt 5400 ]; then
    log "khong tien trien ${age}s (file moi nhat: $newest) -> dung pid $job, queue tu chay lai"
    kill $job
  fi
done
