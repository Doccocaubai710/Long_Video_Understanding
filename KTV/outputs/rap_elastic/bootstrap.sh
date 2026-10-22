#!/bin/bash
# 22/10: Lustre van nghen (sep: tam chep data sang /root/local_data de train; o nay la o cache, MAT khi container reboot).
# Sau moi lan container tao lai: chep lai env + model + data train sang o local, xong moi chay queue6 + keepalive.
#   1) copy_env.sh   -> /root/local_env/qwen25vl, /root/local_data/KTV/Qwen/Qwen2.5-VL-7B-Instruct
#   2) copy_local.sh -> data train (LLaVA-Video-178K-probe), roi chep tiep Video-MME, MSRVTT trong luc train
#   3) env + model + data train xong (hoac doi qua 4 gio) -> chay queue6 (tu --resume tu checkpoint) + keepalive
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1   # thu muc KTV
O=outputs/rap_elastic
L=/root/local_data/KTV
LPY=/root/local_env/qwen25vl/bin/python
RPY=/home/hoang/miniconda3/envs/qwen25vl/bin/python
log() { echo "$(date -u '+%Y-%m-%d %H:%M:%S') BOOTSTRAP $*" >> $O/STATUS.txt; }
env_ok() { [ -f /root/local_env/qwen25vl/.complete ] && [ -f $L/Qwen/Qwen2.5-VL-7B-Instruct/.complete ]; }
data_ok() {
  [ -f $L/Datasets/LLaVA-Video-178K-probe/academic_source/.complete ] || return 1
  local j
  for j in Datasets/LLaVA-Video-178K-probe/*.json; do
    [ "$(stat -c %s $L/$j 2>/dev/null)" = "$(stat -c %s $j)" ] || return 1
  done
}

log "START (pid $$)"
if [ -n "$PY" ]; then   # server khac: PY=$(which python) bash start_all.sh -> khong chep env/data cua server cu
  log "PY=$PY -> bo qua chep env/data, chay queue6 + keepalive"
  pgrep -f "^bash $O/queue6.sh" > /dev/null || { echo $PY > $O/.queue_py; PY=$PY setsid nohup bash $O/queue6.sh >> $O/queue6.out 2>&1 < /dev/null & }
  pgrep -f "^bash $O/keepalive.sh" > /dev/null || PY=$PY setsid nohup bash $O/keepalive.sh > /dev/null 2>&1 < /dev/null &
  log "END"; exit 0
fi
mkdir -p $L
env_ok || { log "chep env + model"; setsid nohup bash $O/copy_env.sh > /dev/null 2>&1 < /dev/null & }
pgrep -f "^bash $O/copy_local.sh" > /dev/null || { log "chep data (16 luong)"; NPAR=16 setsid nohup bash $O/copy_local.sh > /dev/null 2>&1 < /dev/null & }

t0=$(date +%s)
until env_ok && data_ok; do
  if [ $(( $(date +%s) - t0 )) -gt 14400 ]; then log "doi qua 4 gio chua chep xong (env_ok=$(env_ok && echo 1 || echo 0), data_ok=$(data_ok && echo 1 || echo 0)) -> van chay queue"; break; fi
  sleep 60
done
log "env + data train san sang sau $(( ($(date +%s) - t0) / 60 )) phut"

py=$RPY; [ -f /root/local_env/qwen25vl/.complete ] && py=$LPY
if ! pgrep -f "^bash $O/queue6.sh" > /dev/null; then
  log "chay queue6 voi PY=$py"; echo $py > $O/.queue_py
  PY=$py setsid nohup bash $O/queue6.sh >> $O/queue6.out 2>&1 < /dev/null &
fi
pgrep -f "^bash $O/keepalive.sh" > /dev/null || setsid nohup bash $O/keepalive.sh > /dev/null 2>&1 < /dev/null &
log "END"
