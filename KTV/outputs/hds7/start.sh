#!/bin/bash
# Chay (lai) ca 2 queue + keepalive cua dot hds7; cai nao dang chay thi bo qua. Dung sau khi pod khoi dong lai:
#   bash /home/hoang/Long_Video_Understanding/KTV/outputs/hds7/start.sh
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1
O=outputs/hds7; mkdir -p $O/logs
for g in 0 1; do
  q=gpu$g.sh; grep -q "QUEUE DONE gpu$g" $O/STATUS_gpu$g.txt 2>/dev/null && q=gpu${g}_extra.sh
  pgrep -f "^bash $O/gpu$g(_extra)?.sh" > /dev/null || setsid nohup bash $O/$q >> $O/logs/queue_${q%.sh}.out 2>&1 < /dev/null &
  pgrep -f "^bash $O/keepalive.sh $g" > /dev/null || setsid nohup bash $O/keepalive.sh $g > /dev/null 2>&1 < /dev/null &
done
sleep 2; pgrep -af "$O/(gpu|keepalive)"
