#!/bin/bash
# Chay (lai) hang doi rap_elastic sau khi container tao lai; neu dang chay thi bo qua.
#   bash /home/hoang/Long_Video_Understanding/KTV/outputs/rap_elastic/start_all.sh
# O local /root/local_* la o cache, mat khi reboot -> bootstrap.sh chep lai env/model/data roi moi chay queue6 + keepalive.
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1   # thu muc KTV
O=outputs/rap_elastic
if pgrep -f "^bash $O/(queue6|bootstrap).sh" > /dev/null; then echo "dang chay roi"
else setsid nohup bash $O/bootstrap.sh > /dev/null 2>&1 < /dev/null & echo "da chay bootstrap rap_elastic"; fi
sleep 1; pgrep -af "$O/(queue6|bootstrap|keepalive|copy_local|copy_env).sh"
