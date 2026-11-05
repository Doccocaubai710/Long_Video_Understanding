#!/bin/bash
# Chay (lai) queue + keepalive cho mot card; neu dang chay thi bo qua.
#   bash /home/hoang/Long_Video_Understanding/KTV/outputs/hds2/start.sh 1   (may nay)
#   bash /home/hoang/Long_Video_Understanding/KTV/outputs/hds2/start.sh 2   (may kia)
CARD=${1:?can tham so card: 1 hoac 2}
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1   # thu muc KTV
O=outputs/hds2
mkdir -p $O/logs
pgrep -f "^bash $O/queue.sh $CARD" > /dev/null || setsid nohup bash $O/queue.sh $CARD >> $O/logs/queue_card$CARD.out 2>&1 < /dev/null &
sleep 1
pgrep -f "^bash $O/keepalive.sh $CARD" > /dev/null || setsid nohup bash $O/keepalive.sh $CARD > /dev/null 2>&1 < /dev/null &
sleep 1; pgrep -af "$O/(queue|keepalive).sh"
