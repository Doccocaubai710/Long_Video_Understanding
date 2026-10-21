#!/bin/bash
# Chay (lai) hang doi rap_elastic; neu dang chay thi bo qua.
#   bash /home/hoang/Long_Video_Understanding/KTV/outputs/rap_elastic/start_all.sh
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1   # thu muc KTV
pgrep -f "outputs/rap_elastic/queue6.sh" > /dev/null || { setsid nohup bash outputs/rap_elastic/queue6.sh > outputs/rap_elastic/queue6.out 2>&1 & echo "da chay hang doi rap_elastic"; }
sleep 1; pgrep -af "outputs/rap_elastic/queue6.sh"
