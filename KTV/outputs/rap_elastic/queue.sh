#!/bin/bash
# Hang doi thu nghiem 16/10 -> 20/10/2026 (Qwen2.5-VL-7B, 1 A100 40GB), chay tuan tu, tu quyet dinh theo ket qua.
# Bat dau / chay lai sau khi container bi tao lai (moi buoc tu tiep tuc, buoc xong thi bo qua):
#   bash /home/hoang/Long_Video_Understanding/KTV/outputs/rap_elastic/start_all.sh
# Ket qua de doc: outputs/rap_elastic/KET_QUA.txt   Tien do: outputs/rap_elastic/STATUS.txt
#
# Thu tu (cot moc nguong la tren cac cau chung):
#   Q1a  sub300  Uniform 10% + chia lai neo theo canh (alloc_model.py, khong train)
#   Q1b  sub300  R10E + chia lai neo theo canh
#   Q1   full    cai nao hon moc >= 0,7 diem tren 300 cau thi cham du 2.700 cau
#   Q2   train   ELA: RAP-Elastic, 1 checkpoint, moi buoc boc r trong {2,5,10,15,25,40}%, 6.000 buoc
#   Q3   full    ELA o 25% va 10% (so voi runE va R10E)
#   Q4   train+full  R02W3: RAP 2%, cua so +-3 unit (so voi R02E, cua so +-1)
#   Q5b  sub300  Dense 256 frame (32.768 token): thay co manh hon dense 128 khong
#   Q5c  train+full  T256 (neu Q5b hon dense128 >= 1,0): tro nhan 256 frame giu 12,5% (~4.000 token), thay dense 256
#   Q5a  train+full  R05W3 (neu Q4 hon R02E >= 0,5): cua so +-3 o 5%
#   Q6   train+full  R10S (neu Q1 full hon moc >= 0,5): train RAP 10% voi neo chia theo canh
#   F1   train+full  ELA_s1: Elastic seed 1 (kiem tra do on dinh) o 25% va 10%
#   F2   full    DENSE256 du 2.700 cau (neu Q5c da chay)
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

log "QUEUE START (pid $$)"

# ---- Q1: chia lai neo theo canh, khong train ----
do_eval sub AS10U none 0.10 010 --alloc scene --alloc_floor 0.5
do_eval sub AS10R outputs/rap/R10E.pt 0.10 010 --alloc scene --alloc_floor 0.5
dU=$(delta $O/sub300_AS10U.json $B/${P}uniform_r010.json); dR=$(delta $O/sub300_AS10R.json outputs/rap/${P}rap_R10E_r010.json)
log "Q1 sub300: AS10U - Uniform10 = $dU ; AS10R - R10E = $dR (nguong cham du: 0.7)"
ge "$dU" 0.7 && do_eval full AS10U none 0.10 010 --alloc scene --alloc_floor 0.5
ge "$dR" 0.7 && do_eval full AS10R outputs/rap/R10E.pt 0.10 010 --alloc scene --alloc_floor 0.5

# ---- Q2 + Q3: RAP-Elastic ----
do_train ELA --num_steps 6000 --retention 0.25 --retention_set 0.02,0.05,0.10,0.15,0.25,0.40 \
  --val_retentions 0.25,0.10 --val_every 500 && trained ELA
[ -f $O/ELA.pt ] && grep -q "\[train_rap\] done" $O/ELA.train.log && {
  do_eval full ELA $O/ELA.pt 0.25 025
  do_eval full ELA $O/ELA.pt 0.10 010
}

# ---- Q4: cua so rong o muc nen sau ----
do_train R02W3 --num_steps 4000 --retention 0.02 --window 3 && trained R02W3 && do_eval full R02W3 $O/R02W3.pt 0.02 002

# ---- Q5b/Q5c: thay manh hon dense 128 ----
do_eval sub DENSE256 none 1.0 100 --max_frames 256 --max_pixels_units 32768
dD=$(delta $O/sub300_DENSE256.json $B/${P}dense.json)
log "Q5b sub300: DENSE256 - Dense128 = $dD (nguong train T256: 1.0)"
if ge "$dD" 1.0; then
  do_train T256 --num_steps 4000 --retention 0.125 --max_frames 256 --max_pixels_units 32768 && trained T256 && \
    do_eval full T256 $O/T256.pt 0.125 0125 --max_frames 256 --max_pixels_units 32768
fi

# ---- Q5a: cua so +-3 o 5% (chi khi Q4 thang) ----
dW=$(delta $O/${P}R02W3_r002.json outputs/rap_next/${P}rap_R02E_r002.json)
log "Q4: R02W3 - R02E = $dW (nguong chay R05W3: 0.5)"
if ge "$dW" 0.5; then
  do_train R05W3 --num_steps 4000 --retention 0.05 --window 3 && trained R05W3 && do_eval full R05W3 $O/R05W3.pt 0.05 005
fi

# ---- Q6: train RAP voi neo chia theo canh (chi khi Q1 full thang) ----
fU=$(delta $O/${P}AS10U_r010.json $B/${P}uniform_r010.json); fR=$(delta $O/${P}AS10R_r010.json outputs/rap/${P}rap_R10E_r010.json)
log "Q1 full: AS10U - Uniform10 = $fU ; AS10R - R10E = $fR (nguong train R10S: 0.5)"
if ge "$fU" 0.5 || ge "$fR" 0.5; then
  do_train R10S --num_steps 4000 --retention 0.10 --alloc scene --alloc_floor 0.5 && trained R10S && \
    do_eval full R10S $O/R10S.pt 0.10 010 --alloc scene --alloc_floor 0.5
fi

# ---- F1: Elastic seed 1 ----
do_train ELA_s1 --seed 1 --num_steps 6000 --retention 0.25 --retention_set 0.02,0.05,0.10,0.15,0.25,0.40 \
  --val_retentions 0.25,0.10 --val_every 500 && trained ELA_s1 && {
  do_eval full ELA_s1 $O/ELA_s1.pt 0.25 025
  do_eval full ELA_s1 $O/ELA_s1.pt 0.10 010
}

# ---- F2: dense 256 du 2.700 cau (moc cho T256) ----
[ -f $O/${P}T256_r0125.json ] && do_eval full DENSE256 none 1.0 100 --max_frames 256 --max_pixels_units 32768

report
log "QUEUE DONE"
backup_chat
