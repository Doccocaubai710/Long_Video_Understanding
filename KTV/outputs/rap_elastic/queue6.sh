#!/bin/bash
# 20/10 toi -> qua dem: cac dong gop chinh trong 2 file cua sep (MAP-3D, HTP-LVU), cham du 2.700 cau.
#   1  M3D-B     RAP 25% + cross-attn phat khoang cach 3D (MAP-3D 3.3), KV +-3, gamma 1        so voi Run E, Uniform25
#   2  SU25      Uniform 25% + token tom tat moi canh (HTP-LVU residual bypass), khong train     so voi Uniform25
#   3  M3D-BD    M3D-B + Deformable 3D Tube Adapter (MAP-3D 3.2)                                so voi M3D-B, Run E
#   4a V10E_s1   kiem chung V-JEPA (HTP-LVU S2): V-JEPA pretrain seed 1 + KL 4.000 buoc, 10%    so voi V10E, R10E
#   4b R10E7k    doi chung cho 4a: KL 7.000 buoc, khong V-JEPA, 10%                            so voi R10E, V10E
#   -- het 4 y tren thi chay tiep (khong de GPU trong):
#   5  RSS25     RAP 25% + neo chi nhin cung canh (HTP-LVU S1) + token tom tat; neu SU25 bi ngat/khong hon Uniform -> RSW25 (chi cung canh)
#   6  M3DF      MAP-3D day du: 3.3 + 3.2 + Elastic 4.1 (1 checkpoint), cham 25% va 10%
#   7  M3D-B seed 1, 2 (do on dinh), roi lap lai Uniform-free seeds den khi bi dung
# Ngat som: trong luc cham, cu 5 phut so cap tung cau voi moc; sau >= 900 cau ma thap hon moc voi z <= -1,96 -> ngat,
#   ghi vao STOPS.txt va KET_QUA.txt.
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1   # thu muc KTV
PY=${PY:-/home/hoang/miniconda3/envs/qwen25vl/bin/python}   # may khac: PY=$(which python) bash ...
O=outputs/rap_elastic
P=pred_videomme_f128_
STATUS=$O/STATUS.txt
log() { echo "$(date -u '+%Y-%m-%d %H:%M:%S') $*" >> $STATUS; }
nlines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
report() { $PY $O/tools.py report >> $O/tools.log 2>&1; }
backup_chat() { [ -d /home/hoang/claude_backup ] || return 0; mkdir -p /home/hoang/claude_backup/-home-hoang && cp -f /root/.claude/projects/-home-hoang/*.jsonl /home/hoang/claude_backup/-home-hoang/ 2>/dev/null; }
delta() { $PY $O/tools.py delta "$1" "$2" 2>/dev/null || echo nan; }
gt0() { [[ "$1" =~ ^-?[0-9]+(\.[0-9]+)?$ ]] && awk -v a="$1" 'BEGIN{exit !(a+0>0)}'; }

DEFAULT_STEPS="--pseudo_steps 0 --pseudo_prob_after 0 --w_ce 0 --w_kd 1.0"
do_train() {  # do_train <ten> <tham so train_rap_elastic.py ...>
  local name=$1; shift
  for attempt in 1 2 3 4 5; do
    grep -q "\[train_rap\] done" $O/$name.train.log 2>/dev/null && return 0
    log "START train $name (lan $attempt)"
    timeout 22h $PY train_rap_elastic.py --out $O/$name.pt --resume $O/$name.pt $DEFAULT_STEPS "$@" >> $O/$name.train.log 2>&1
    sleep 30
  done
  grep -q "\[train_rap\] done" $O/$name.train.log || { log "FAIL train $name"; return 1; }
}
do_vjepa() {  # do_vjepa <ten> <tham so train_rap_vjepa.py ...>
  local name=$1; shift
  for attempt in 1 2 3 4 5; do
    grep -q "\[train_rap_vjepa\] done" $O/$name.train.log 2>/dev/null && return 0
    log "START train $name (lan $attempt)"
    timeout 20h $PY train_rap_vjepa.py --out $O/$name.pt --resume $O/$name.pt "$@" >> $O/$name.train.log 2>&1
    sleep 30
  done
  grep -q "\[train_rap_vjepa\] done" $O/$name.train.log || { log "FAIL train $name"; return 1; }
}
do_eval() {  # do_eval <ten> <policy|none> <muc giu> <hau to> <moc ngat som, vd Uniform25,RunE25> [tham so infer them ...]
  local name=$1 policy=$2 ret=$3 tag=$4 guard=$5; shift 5
  local want=2700 out=${P}${name}_r$tag
  for attempt in 1 2 3 4 5; do
    [ -f $O/$out.STOPPED ] && break
    [ "$(nlines $O/$out.json)" -ge $want ] && break
    log "START evalfull $name r=$ret (lan $attempt, da co $(nlines $O/$out.json)/$want)"
    timeout 12h $PY infer_rap_videomme.py --policy $policy --retention $ret --output_dir $O --output_name $out "$@" >> $O/$out.log 2>&1 &
    local ip=$!
    while kill -0 $ip 2>/dev/null; do
      sleep 300
      local g; g=$($PY $O/tools.py guard $O/$out.json $guard 2>/dev/null)
      if [[ "$g" == STOP* ]]; then
        echo "$(date -u '+%Y-%m-%d %H:%M') UTC  $name (r=$ret): $g" >> $O/STOPS.txt
        log "STOP evalfull $name: $g"
        touch $O/$out.STOPPED
        kill $ip 2>/dev/null; sleep 5; pkill -f "output_name $out( |$)" 2>/dev/null
        break
      fi
    done
    wait $ip 2>/dev/null
    sleep 30
  done
  $PY eval/compute_accuracy_by_bucket.py $O/$out.json --csv playground/gt_qa_files/VideoMME/val_qa.csv > $O/${out}_accuracy_by_bucket.txt 2>&1
  local st="DONE"; [ -f $O/$out.STOPPED ] && st="DONE (NGAT SOM)"
  log "$st evalfull $name ($(nlines $O/$out.json)/$want): $(grep overall $O/${out}_accuracy_by_bucket.txt)"
  report; backup_chat
  [ ! -f $O/$out.STOPPED ] && [ "$(nlines $O/$out.json)" -ge $want ]
}
trained() { log "OK   train $1"; report; backup_chat; }

# doi queue5 (C0) xong neu con chay
while pgrep -f "queue5.sh" >/dev/null; do sleep 30; done
log "QUEUE6 START (pid $$) -- M3D-B, SU25, M3D-BD, V10E_s1, R10E7k, roi RSS25/RSW25, M3DF, seeds"

# 1. MAP-3D 3.3
do_train M3D-B --retention 0.25 --num_steps 4000 --window 3 --bias3d 1 --gamma 1.0 && trained M3D-B
[ -f $O/M3D-B.pt ] && grep -q "\[train_rap\] done" $O/M3D-B.train.log && do_eval M3D-B $O/M3D-B.pt 0.25 025 Uniform25,RunE25

# 2. HTP-LVU residual bypass, khong train
do_eval SU25 none 0.25 025 Uniform25 --summary 1

# 3. MAP-3D 3.2
do_train M3D-BD --retention 0.25 --num_steps 4000 --window 3 --bias3d 1 --gamma 1.0 --deform 1 && trained M3D-BD
grep -q "\[train_rap\] done" $O/M3D-BD.train.log && do_eval M3D-BD $O/M3D-BD.pt 0.25 025 Uniform25,RunE25

# 4a. kiem chung V-JEPA: seed 1 cua V10E (3.000 buoc V-JEPA + 4.000 buoc KL)
do_vjepa V10_s1seed1 --retention 0.10 --num_steps 3000 --seed 1 && \
  do_train V10E_s1 --init_from $O/V10_s1seed1.pt --retention 0.10 --num_steps 4000 --seed 1 && trained V10E_s1
grep -q "\[train_rap\] done" $O/V10E_s1.train.log && do_eval V10E_s1 $O/V10E_s1.pt 0.10 010 Uniform10,R10E

# 4b. doi chung: train lau bang (7.000 buoc), khong V-JEPA
do_train R10E7k --retention 0.10 --num_steps 7000 && trained R10E7k
grep -q "\[train_rap\] done" $O/R10E7k.train.log && do_eval R10E7k $O/R10E7k.pt 0.10 010 Uniform10,R10E
log "QUEUE6 -- xong 4 y chinh, chay tiep y bo sung"

# 5. HTP-LVU S1 lam attention mask (+ token tom tat neu SU25 co ich)
if [ ! -f $O/${P}SU25_r025.STOPPED ] && gt0 "$(delta $O/${P}SU25_r025.json outputs/kitoke_baseline/${P}uniform_r025.json)"; then
  do_train RSS25 --retention 0.25 --num_steps 4000 --scene_window 1 --summary 1 && trained RSS25
  grep -q "\[train_rap\] done" $O/RSS25.train.log && do_eval RSS25 $O/RSS25.pt 0.25 025 Uniform25,RunE25 --scene_window 1 --summary 1
else
  do_train RSW25 --retention 0.25 --num_steps 4000 --scene_window 1 && trained RSW25
  grep -q "\[train_rap\] done" $O/RSW25.train.log && do_eval RSW25 $O/RSW25.pt 0.25 025 Uniform25,RunE25 --scene_window 1
fi

# 6. MAP-3D day du (3.3 + 3.2 + Elastic 4.1), cung cong thuc ELA (6.000 buoc)
do_train M3DF --retention 0.25 --retention_set 0.02,0.05,0.10,0.15,0.25,0.40 --val_retentions 0.25,0.10 \
  --num_steps 6000 --window 3 --bias3d 1 --gamma 1.0 --deform 1 && trained M3DF
if grep -q "\[train_rap\] done" $O/M3DF.train.log; then
  do_eval M3DF $O/M3DF.pt 0.25 025 Uniform25,RunE25
  do_eval M3DF $O/M3DF.pt 0.10 010 Uniform10,R10E
fi

# 7. giu GPU ban: them seed cho M3D-B
for s in 1 2 3 4 5 6 7 8; do
  do_train M3D-B_s$s --retention 0.25 --num_steps 4000 --window 3 --bias3d 1 --gamma 1.0 --seed $s && trained M3D-B_s$s
  grep -q "\[train_rap\] done" $O/M3D-B_s$s.train.log && do_eval M3D-B_s$s $O/M3D-B_s$s.pt 0.25 025 Uniform25,RunE25
done
log "QUEUE6 DONE"
