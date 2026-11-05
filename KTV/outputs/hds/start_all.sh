#!/bin/bash
# Chay (lai) queue HDS + keepalive; neu dang chay thi bo qua.
#   bash /home/hoang/Long_Video_Understanding/KTV/outputs/hds/start_all.sh
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1   # thu muc KTV
O=outputs/hds
mkdir -p $O/logs
pgrep -f "^bash $O/queue.sh" > /dev/null || setsid nohup bash $O/queue.sh >> $O/logs/queue.out 2>&1 < /dev/null &
sleep 1
pgrep -f "^bash $O/keepalive.sh" > /dev/null || setsid nohup bash $O/keepalive.sh > /dev/null 2>&1 < /dev/null &
sleep 1; pgrep -af "$O/(queue|keepalive).sh"
