#!/bin/bash
# Dot HDS-MAP 3D, 23/10 -> 26/10, 1 A100 (card 0). Moi idea 1 run, khong seed. Muc chinh 10%.
#  1  P0   Pha 0 (khong train): chon 1 token / hop 3D theo luat center|var|norm|medoid, o 10% va 2%   [Video-MME]
#  2  H1   Chi giai doan 1: bo truot CB-STE co hoc (100% token goc)                                [train + Video-MME]
#  3  H2   Chi giai doan 2 = M3D-BD+: neo Uniform + M3D-BD + che token neo + cong (1-r)              [train + Video-MME]
#  4  LONG Moc tren LVBench + MLVU-Test: dense, Uniform 10%, HoliTom 10%, R10E, H1, H2              [cham]
#  5  H3   HDS-MAP 3D day du (giai doan 1, mo giai doan 2 tu buoc 2000, lr giai doan 1 x0.1)        [train + Video-MME]
#  6  H4   Ban tot nhat H1-H3 + video dai ghep (3 clip 1-3 phut, 50% so buoc)                       [train + Video-MME]
#  7  H5   Ban tot nhat H1-H3, Elastic 2/5/10/25%                                                  [train + Video-MME 4 muc]
#  8  LONG them H3, H4, H5 o 10%                                                                    [cham]
#  9  H6   Ban tot nhat H1-H3 train rieng o 2% + LONG o 2%                                         [train + cham]
# Moi buoc tu tiep tuc (train --resume, cham bo qua cau da co); buoc da xong thi bo qua.
# Chay: bash outputs/hds/start_all.sh   |   Ket qua: outputs/hds/KET_QUA.txt
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1   # thu muc KTV
PY=${PY:-/root/local_env/qwen25vl/bin/python}
export DECORD_EOF_RETRY_MAX=131072 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
O=outputs/hds
STATUS=$O/STATUS.txt
LOCAL=/root/local_data/KTV
MPATH=Qwen/Qwen2.5-VL-7B-Instruct; [ -f $LOCAL/$MPATH/.complete ] && MPATH=$LOCAL/$MPATH
VROOT=Datasets/LLaVA-Video-178K-probe; [ -d $LOCAL/$VROOT ] && VROOT=$LOCAL/$VROOT
mkdir -p $O/train $O/videomme $O/lvbench $O/mlvu $O/logs
log() { echo "$(date -u '+%Y-%m-%d %H:%M:%S') $*" >> $STATUS; }
report() { $PY -m tools.hds_report >> $O/logs/report.log 2>&1; }
nlines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }

DEF="--pseudo_steps 0 --pseudo_prob_after 0 --w_ce 0 --w_kd 1.0 --num_steps 4000 --model_path $MPATH --video_root $VROOT"
M3DBDP="--bias3d 1 --deform 1 --window 3 --self_mask 1 --gate 1"
CFG_H1="--hds 1 --slider 1 --stage2 0 --w_cov 0"
CFG_H2="--hds 1 --slider 0 --stage2 1 $M3DBDP"
CFG_H3="--hds 1 --slider 1 --stage2 1 $M3DBDP --s2_start 2000 --s1_lr_after 0.1 --w_cov 0"

do_train() {  # do_train <ten> <tham so ...>
  local name=$1; shift
  for attempt in 1 2 3 4; do
    grep -q "\[train_rap\] done" $O/logs/$name.train.log 2>/dev/null && return 0
    log "START train $name (lan $attempt)"
    timeout 20h $PY -m rap.train_rap_elastic --out $O/train/$name.pt --resume $O/train/$name.pt $DEF "$@" >> $O/logs/$name.train.log 2>&1
    sleep 30
  done
  grep -q "\[train_rap\] done" $O/logs/$name.train.log || { log "FAIL train $name"; return 1; }
}
do_eval() {  # do_eval <bo: videomme|lvbench|mlvu> <ten log> <--variant ...>
  local ds=$1 tag=$2; shift 2
  local gt vd n
  case $ds in
    videomme) gt=playground/gt_qa_files/VideoMME/val_qa.json; vd=Datasets/Video-MME/data/data ;;
    lvbench)  gt=playground/gt_qa_files/LVBench/val_qa.json;  vd=Datasets/LVBench/videos ;;
    mlvu)     gt=playground/gt_qa_files/MLVU_Test/val_qa_mc.json; vd=Datasets/MLVU_Test/videos ;;
  esac
  [ -f $gt ] || { log "SKIP eval $ds/$tag: chua co $gt"; return 1; }
  n=$($PY -c "import json;print(len(json.load(open('$gt'))))")
  for attempt in 1 2 3 4; do
    local missing=0
    for v in "$@"; do [ "$v" = "--variant" ] && continue; f=$O/$ds/${v%%=*}.json; [ "$(nlines $f)" -ge "$n" ] || missing=1; done
    [ $missing = 0 ] && break
    log "START eval $ds/$tag (lan $attempt)"
    timeout 14h $PY -m rap.eval_multi --gt_file $gt --video_dir $vd --model_path $MPATH --output_dir $O/$ds "$@" >> $O/logs/eval_${ds}_$tag.log 2>&1
    sleep 30
  done
  log "DONE eval $ds/$tag"; report
}
ck() { echo "--variant $1_r$2=ckpt:$O/train/$1.pt@$3"; }   # ck <ten run> <hau to muc> <muc>
long_eval() {  # long_eval <tag> <muc hau to> <muc> <run ...>
  local tag=$1 rs=$2 r=$3; shift 3
  [ -f $O/.long_ready ] || wait_long
  local V="--variant dense=dense --variant uniform_r$rs=uniform@$r --variant holitom_r$rs=holitom@$r"
  [ $rs = 010 ] && V="$V --variant R10E_r010=ckpt:outputs/checkpoints/R10E.pt@0.10"
  [ $rs = 002 ] && V="$V --variant R02E_r002=ckpt:outputs/checkpoints/R02E.pt@0.02"
  for h in "$@"; do [ -f $O/train/$h.pt ] && grep -q "\[train_rap\] done" $O/logs/$h.train.log && V="$V $(ck $h $rs $r)"; done
  do_eval lvbench $tag $V
  do_eval mlvu $tag $V
}
wait_long() {  # cho giai nen xong roi tao file cau hoi (toi da 3 gio)
  for i in $(seq 1 36); do grep -q "EXTRACT DONE" Datasets/extract_long.log 2>/dev/null && break; sleep 300; done
  $PY -m tools.prepare_long_gt >> $STATUS 2>&1 && touch $O/.long_ready
}
best() {  # ten run tot nhat (Video-MME 10%) trong H1-H3
  $PY - <<'EOF'
import json, os
from eval.compute_accuracy import load_jsonl, prediction_to_index
best = ("H3", -1)
for h in ["H1", "H2", "H3"]:
    p = f"outputs/hds/videomme/{h}_r010.json"
    if os.path.exists(p):
        r = load_jsonl(p)
        a = sum(prediction_to_index(x["pred"], x["candidates"]) == int(x["answer_number"]) for x in r) / max(1, len(r))
        if len(r) >= 2700 and a > best[1]:
            best = (h, a)
print(best[0])
EOF
}

log "QUEUE START (pid $$, model=$MPATH)"
report
# 1. Pha 0
do_eval videomme P0 --variant P0_center_r010=rule:center@0.10 --variant P0_var_r010=rule:var@0.10 \
  --variant P0_norm_r010=rule:norm@0.10 --variant P0_medoid_r010=rule:medoid@0.10 \
  --variant P0_center_r002=rule:center@0.02 --variant P0_var_r002=rule:var@0.02 \
  --variant P0_norm_r002=rule:norm@0.02 --variant P0_medoid_r002=rule:medoid@0.02
# 2-3. H1, H2
do_train H1 --retention 0.10 $CFG_H1 && do_eval videomme H1 $(ck H1 010 0.10)
do_train H2 --retention 0.10 $CFG_H2 && do_eval videomme H2 $(ck H2 010 0.10)
# 4. bo dai: moc + H1, H2
long_eval A 010 0.10 H1 H2
# 5. H3
do_train H3 --retention 0.10 $CFG_H3 && do_eval videomme H3 $(ck H3 010 0.10)
# 6-7. ban tot nhat
B=$(best); eval "CFG_B=\$CFG_$B"; log "BEST H1-H3 = $B ($CFG_B)"
echo "$B" > $O/best.txt
do_train H4 --retention 0.10 $CFG_B --stitch 3 --stitch_prob 0.5 --max_caption_tokens 576 && do_eval videomme H4 $(ck H4 010 0.10)
do_train H5 --retention 0.10 $CFG_B --retention_set 0.02,0.05,0.10,0.25 --val_retentions 0.10 && \
  do_eval videomme H5 $(ck H5 010 0.10) $(ck H5 002 0.02) $(ck H5 005 0.05) $(ck H5 025 0.25)
# 8. bo dai: them H3, H4, H5
long_eval B 010 0.10 H1 H2 H3 H4 H5
# 9. 2%
do_train H6 --retention 0.02 $CFG_B && do_eval videomme H6 $(ck H6 002 0.02)
long_eval C 002 0.02 H6 H5
log "QUEUE DONE"
report
