#!/bin/bash
# Giu card chay lien tuc den khi queue xong.  bash outputs/hds7/keepalive.sh <0|1>
#  - queue chet ma chua "QUEUE DONE gpuN" -> chay lai; da xong -> chay gpuN_extra.sh (lap vo han, card khong ranh) (train tu --resume, eval tu bo qua cau da cham)
#  - job cua card nay khong ghi log them trong 90 phut -> dung job do (queue tu chay lai lan sau)
#  - moi 30 phut viet lai outputs/hds7/KET_QUA.txt
GPU=${1:?can tham so GPU 0 hoac 1}
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1
O=outputs/hds7; PY=/home/hoang/miniconda3/envs/qwen25vl/bin/python; ST=$O/STATUS_gpu$GPU.txt
log() { echo "$(date -u '+%Y-%m-%d %H:%M:%S') KEEPALIVE $*" >> $ST; }
my_job() {
  local p
  for p in $(pgrep -f "rap\.(train_rap_elastic|eval_multi) "); do
    tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$GPU" || continue
    readlink /proc/$p/fd/1 2>/dev/null | grep -qE "outputs/hds[67]/logs/" && { echo $p; return; }
  done
}
log "START (pid $$)"
i=0
while true; do
  sleep 300; i=$((i + 1))
  [ $((i % 6)) = 0 ] && $PY $O/report.py >> $O/logs/report.log 2>&1
  if ! pgrep -f "^bash $O/gpu$GPU(_extra)?.sh" > /dev/null; then
    q=gpu$GPU.sh; grep -q "QUEUE DONE gpu$GPU" $ST && q=gpu${GPU}_extra.sh   # queue chinh xong -> chay queue noi tiep (vo han)
    log "$q khong con chay -> chay"; setsid nohup bash $O/$q >> $O/logs/queue_${q%.sh}.out 2>&1 < /dev/null &
    continue
  fi
  job=$(my_job); [ -z "$job" ] && continue
  out=$(readlink /proc/$job/fd/1 2>/dev/null)
  age=$(( $(date +%s) - $(stat -c %Y "$out" 2>/dev/null || echo 0) ))
  run=$(ps -o etimes= -p $job 2>/dev/null | tr -d " "); run=${run:-0}  # job moi chay chua kip ghi log -> khong tinh
  if [ $age -gt 5400 ] && [ $run -gt 5400 ]; then log "job $job khong ghi log ${age}s ($out) -> dung, queue tu chay lai"; kill $job; fi
done
