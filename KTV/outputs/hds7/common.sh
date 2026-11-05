#!/bin/bash
# Ham dung chung cho dot hds7 (plan 3 ngay 30/10 -> 2/11). source tu gpu0.sh / gpu1.sh, can bien GPU truoc khi source.
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." || exit 1   # thu muc KTV
PY=/home/hoang/miniconda3/envs/qwen25vl/bin/python
export DECORD_EOF_RETRY_MAX=131072 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPU
O=outputs/hds7; MP=Qwen/Qwen2.5-VL-7B-Instruct; ST=$O/STATUS_gpu$GPU.txt
mkdir -p $O/train $O/videomme $O/lvbench $O/logs
log() { echo "$(date -u '+%Y-%m-%d %H:%M:%S') $*" >> $ST; }
nlines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
report() { $PY $O/report.py >> $O/logs/report.log 2>&1; }

DEF="--pseudo_steps 0 --pseudo_prob_after 0 --w_ce 0 --w_kd 1.0 --num_steps 4000 --model_path $MP --video_root Datasets/LLaVA-Video-178K-probe --val_extra_root Datasets/LLaVA-Video-178K-x2 --n_val_extra 48 --val_extra_every 500"
CFG_H3="--hds 1 --slider 1 --stage2 1 --bias3d 1 --deform 1 --window 3 --self_mask 1 --gate 1 --s2_start 2000 --s1_lr_after 0.1 --w_cov 0"
VRIPT="--vript_index Datasets/Vript/vript_index.jsonl --vript_prob 0.5 --vript_scenes 8 --max_caption_tokens 768"

# --- vi tri checkpoint / log / muc giu cua tung ten ---
ckpt() { case $1 in H3) echo outputs/hds/train/H3.pt;; H3s1|H3noGate) echo outputs/hds2/train/$1.pt;; H3ce) echo outputs/hds4/train/H3ce.pt;;
                    H3cv) echo outputs/hds6/train/H3cv.pt;; *) echo $O/train/$1.pt;; esac; }
tlog() { case $1 in H3cv) echo outputs/hds6/logs/H3cv.train.log;; *) echo $O/logs/$1.train.log;; esac; }
ret()  { case $1 in *x25) echo 0.25;; *x40) echo 0.40;; X25|K25*|SM25*) echo 0.25;; X5|K5|K5s*|BX5*|SG5*|SM5*|SMg5*) echo 0.05;; X40|X40s1|K40*) echo 0.40;; X2|K2s*) echo 0.02;; *) echo 0.10;; esac; }
rc()   { case $1 in *x25) echo 025;; *x40) echo 040;; X25|K25*|SM25*) echo 025;; X5|K5|K5s*|BX5*|SG5*|SM5*|SMg5*) echo 005;; X40|X40s1|K40*) echo 040;; X2|K2s*) echo 002;; *) echo 010;; esac; }
is_done()   { case $1 in H3|H3s1|H3ce) return 0;; esac; grep -q "\[train_rap\] done" $(tlog $1) 2>/dev/null; }
is_failed() { grep -qhE "(FAIL|SKIP) train $1\b" $O/STATUS_gpu*.txt 2>/dev/null; }
wait_train() {  # cho den khi train xong hoac FAIL; tra ve 1 neu FAIL
  until is_done $1 || is_failed $1; do sleep 120; done
  is_done $1
}

# train <ten> <tham so them...>  (resume duoc; neu dang co tien trinh train cung --out thi cho no chay xong)
train() {
  local name=$1; shift
  local out; out=$(ckpt $name); local L; L=$(tlog $name)
  while pgrep -f -- "--out $out " > /dev/null; do sleep 120; done
  for a in 1 2 3; do
    is_done $name && break
    if grep -qi "out of memory" $L 2>/dev/null; then log "OOM $name -> bo qua"; break; fi
    log "START train $name (lan $a): r=$(ret $name) $*"
    timeout 20h $PY -m rap.train_rap_elastic --out $out --resume $out $DEF --retention $(ret $name) $CFG_H3 "$@" >> $L 2>&1
    sleep 30
  done
  if is_done $name; then log "DONE train $name"; report; return 0; fi
  log "FAIL train $name"; return 1
}

# evalb <mlvu|videomme|lvbench> <ten...> : cham gop nhieu checkpoint trong 1 lan eval_multi (bo qua ten FAIL / da du)
evalb() {
  local b=$1; shift; local gt vd od
  case $b in
    mlvu)     gt=playground/gt_qa_files/MLVU_Dev/val_qa_mc.json; vd=Datasets/MLVU_Dev/MLVU/video; od=outputs/mlvu_dev/pred;;
    videomme) gt=playground/gt_qa_files/VideoMME/val_qa.json; vd=Datasets/Video-MME/data/data; od=$O/videomme;;
    lvbench)  gt=playground/gt_qa_files/LVBench/val_qa.json; vd=Datasets/LVBench/videos; od=$O/lvbench;;
  esac
  local n; n=$($PY -c "import json;print(len(json.load(open('$gt'))))")
  for a in 1 2 3; do
    local args="" names=""
    for x in "$@"; do
      wait_train $x || continue
      [ "$(nlines $od/${x}_r$(rc $x).json)" -ge "$n" ] && continue
      args="$args --variant ${x}_r$(rc $x)=ckpt:$(ckpt $x)@$(ret $x)"; names="$names $x"
    done
    [ -z "$args" ] && break
    log "START eval $b:$names (lan $a)"
    timeout 16h $PY -m rap.eval_multi --gt_file $gt --video_dir $vd --model_path $MP --output_dir $od --max_frames 128 $args >> $O/logs/eval_${b}_gpu$GPU.log 2>&1
    sleep 20
  done
  log "END eval $b: $(for x in "$@"; do echo -n "$x=$(nlines $od/${x}_r$(rc $x).json) "; done)/$n"
  report
}

# C*: chon tu dong sau giai doan 1 (outputs/hds7/choose_cstar.py), luu o CSTAR.txt: dong 1 = ten, dong 2 = tham so train
wait_cstar() {
  until [ -s $O/CSTAR.txt ]; do $PY $O/choose_cstar.py >> $O/logs/cstar.log 2>&1 || sleep 300; done
  CNAME=$(sed -n 1p $O/CSTAR.txt); CARGS=$(sed -n 2p $O/CSTAR.txt)
}
