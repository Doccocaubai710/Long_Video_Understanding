#!/bin/bash
# Giu GPU chay lien tuc den khi queue xong, tach khoi phien Claude.
#  - queue chet ma chua "QUEUE DONE" -> chay lai (queue tu --resume / bo qua buoc da xong)
#  - 90 phut khong co file tien do nao thay doi trong khi co tien trinh train/cham -> dung tien trinh do, queue tu chay lai
#  - moi 30 phut viet lai outputs/hds/KET_QUA.txt
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1   # thu muc KTV
O=outputs/hds
PY=/root/local_env/qwen25vl/bin/python
log() { echo "$(date -u '+%Y-%m-%d %H:%M:%S') KEEPALIVE $*" >> $O/STATUS.txt; }
start_queue() { log "chay lai queue"; setsid nohup bash $O/queue.sh >> $O/logs/queue.out 2>&1 < /dev/null & }
log "START (pid $$)"
i=0
while true; do
  sleep 300; i=$((i + 1))
  [ $((i % 6)) = 0 ] && $PY -m tools.hds_report >> $O/logs/report.log 2>&1
  if ! pgrep -f "^bash $O/queue.sh" > /dev/null; then
    if grep -q "QUEUE DONE" $O/STATUS.txt; then log "queue da xong het, dung keepalive"; exit 0; fi
    log "queue khong con chay"; start_queue; continue
  fi
  job=$(pgrep -f "rap\.(train_rap_elastic|eval_multi) .*$O" | head -1)
  newest=$(ls -t $O/logs/*.log $O/train/*.metrics.jsonl $O/*/*.json $O/STATUS.txt 2>/dev/null | head -1)
  age=$(( $(date +%s) - $(stat -c %Y "$newest") ))
  if [ -n "$job" ] && [ $age -gt 5400 ]; then
    log "khong tien trien ${age}s (file moi nhat: $newest) -> dung pid $job, queue tu chay lai"
    kill $job
  fi
done
