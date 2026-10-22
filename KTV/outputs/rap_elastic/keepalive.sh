#!/bin/bash
# 21/10: giu GPU chay lien tuc qua dem (chay tach rieng, khong phu thuoc phien Claude).
#  - queue6 chet ma chua "QUEUE6 DONE" -> chay lai (queue tu --resume / bo qua run da xong)
#  - 120 phut khong co file tien do nao thay doi trong khi dang co tien trinh train/cham -> dung tien trinh do,
#    queue tu chay lai lan sau tu checkpoint gan nhat
#  - env python da chep xong sang o local (copy_env.sh) ma queue con chay python tren Lustre -> chay lai queue bang python local
#  - khi train da chay on dinh: chep tiep Video-MME sang o local (4 luong) de cham khong phu thuoc Lustre
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1   # thu muc KTV
O=outputs/rap_elastic
LPY=/root/local_env/qwen25vl/bin/python
RPY=/home/hoang/miniconda3/envs/qwen25vl/bin/python
UPY=${PY:-}   # server khac: PY=$(which python) -> luon dung python nay, khong chep du lieu sang o local
JOBPAT="^(${UPY:+$UPY|}$RPY|$LPY) (train_rap|infer_rap)"
log() { echo "$(date -u '+%Y-%m-%d %H:%M:%S') KEEPALIVE $*" >> $O/STATUS.txt; }
pick_py() { [ -n "$UPY" ] && { echo $UPY; return; }; [ -f /root/local_env/qwen25vl/.complete ] && echo $LPY || echo $RPY; }
start_queue() { local py; py=$(pick_py); log "chay queue6 voi PY=$py"; echo $py > $O/.queue_py
  PY=$py setsid nohup bash $O/queue6.sh >> $O/queue6.out 2>&1 < /dev/null & }
log "START (pid $$)"
while true; do
  sleep 300
  qpid=$(pgrep -f "^bash outputs/rap_elastic/queue6.sh" | head -1)
  if [ -z "$qpid" ]; then
    if tail -n 3 $O/STATUS.txt | grep -q "QUEUE6 DONE"; then log "queue6 da xong het, dung keepalive"; exit 0; fi
    log "queue6 khong con chay"; start_queue; continue
  fi
  job=$(pgrep -f "$JOBPAT" | head -1)
  if [ "$(pick_py)" = "$LPY" ] && [ "$(cat $O/.queue_py 2>/dev/null)" != "$LPY" ]; then
    log "env local da san sang -> dung queue6 (pid $qpid) va tien trinh $job (python Lustre), chay lai bang python local"
    kill $qpid; sleep 2; [ -n "$job" ] && kill $(pgrep -f "^timeout .*$RPY") $job 2>/dev/null; sleep 30
    start_queue; continue
  fi
  newest=$(ls -t $O/*.metrics.jsonl $O/*.log $O/pred_*.json $O/STATUS.txt 2>/dev/null | head -1)
  age=$(( $(date +%s) - $(stat -c %Y "$newest") ))
  if [ -n "$job" ] && [ $age -gt 7200 ]; then
    log "khong tien trien ${age}s (file moi nhat: $newest) -> dung pid $job, queue tu chay lai"
    kill $job
  fi
  if [ -z "$UPY" ] && [ ! -f /root/local_data/KTV/Datasets/Video-MME/data/data/.complete ] && [ $age -lt 600 ] \
     && ! pgrep -f "^bash outputs/rap_elastic/copy_(local|env).sh" > /dev/null \
     && [ $(( $(date +%s) - $(stat -c %Y $O/M3D-BD.pt.metrics.jsonl) )) -lt 600 ]; then   # train dang ghi metrics
    log "chep tiep Video-MME sang o local (4 luong)"
    NPAR=4 setsid nohup bash $O/copy_local.sh > /dev/null 2>&1 < /dev/null &
  fi
done
