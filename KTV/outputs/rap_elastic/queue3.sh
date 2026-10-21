#!/bin/bash
# Hang doi 20/10: cham du 2.700 cau cho chia neo theo canh (AS10R, AS10U).
cd /home/hoang/Long_Video_Understanding/KTV || exit 1
PY=/home/hoang/miniconda3/envs/qwen25vl/bin/python
O=outputs/rap_elastic
B=outputs/kitoke_baseline
P=pred_videomme_f128_
STATUS=$O/STATUS.txt
SUB=playground/gt_qa_files/VideoMME/val_qa_subset300.json
log() { echo "$(date -u '+%Y-%m-%d %H:%M:%S') $*" >> $STATUS; }
nlines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
report() { $PY $O/tools.py report >> $O/tools.log 2>&1; }
backup_chat() { mkdir -p /home/hoang/claude_backup/-home-hoang && cp -f /root/.claude/projects/-home-hoang/*.jsonl /home/hoang/claude_backup/-home-hoang/ 2>/dev/null; }
delta() { $PY $O/tools.py delta "$1" "$2" 2>/dev/null || echo nan; }
ge() { [[ "$1" =~ ^-?[0-9]+(\.[0-9]+)?$ ]] && awk -v a="$1" -v b="$2" 'BEGIN{exit !(a+0>=b+0)}'; }   # ge <so> <nguong>; nan -> sai

STEPS=0; DEFAULT_STEPS="--pseudo_steps 0 --pseudo_prob_after 0 --w_ce 0 --w_kd 1.0"
do_train() {  # do_train <ten> <tham so train_rap_elastic.py ...>
  local name=$1; shift
  for attempt in 1 2 3 4 5; do
    grep -q "\[train_rap\] done" $O/$name.train.log 2>/dev/null && return 0
    log "START train $name (lan $attempt)"
    timeout 22h $PY train_rap_elastic.py --out $O/$name.pt --resume $O/$name.pt $DEFAULT_STEPS "$@" >> $O/$name.train.log 2>&1
    sleep 30
  done
  grep -q "\[train_rap\] done" $O/$name.train.log
}
do_eval() {  # do_eval <sub|full> <ten> <policy|none> <muc giu> <hau to> [tham so infer them ...]
  local kind=$1 name=$2 policy=$3 ret=$4 tag=$5; shift 5
  local want=2700 extra="" out=${P}${name}_r$tag
  if [ "$kind" = sub ]; then want=300; extra="--gt_file $SUB"; out=sub300_$name; fi
  for attempt in 1 2 3 4 5; do
    [ "$(nlines $O/$out.json)" -ge $want ] && break
    log "START eval$kind $name r=$ret (lan $attempt, da co $(nlines $O/$out.json)/$want)"
    timeout 12h $PY infer_rap_videomme.py --policy $policy --retention $ret --output_dir $O --output_name $out $extra "$@" >> $O/$out.log 2>&1
    sleep 30
  done
  $PY eval/compute_accuracy_by_bucket.py $O/$out.json --csv playground/gt_qa_files/VideoMME/val_qa.csv > $O/${out}_accuracy_by_bucket.txt 2>&1
  log "DONE eval$kind $name ($(nlines $O/$out.json)/$want): $(grep overall $O/${out}_accuracy_by_bucket.txt)"
  report; backup_chat
  [ "$(nlines $O/$out.json)" -ge $want ]
}
trained() { log "OK   train $1"; report; backup_chat; }

log "QUEUE3 START (pid $$) -- cham du 2700 cau AS10R, AS10U"
do_eval full AS10R outputs/rap/R10E.pt 0.10 010 --alloc scene --alloc_floor 0.5
do_eval full AS10U none 0.10 010 --alloc scene --alloc_floor 0.5
log "QUEUE3 DONE"
