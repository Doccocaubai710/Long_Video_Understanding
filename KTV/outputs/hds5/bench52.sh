#!/bin/bash
# Cham not muc 5%/2% (30/10). Card 1 (GPU0): MLVU dev - 192 cau anomaly_reco con thieu cua FastVID (moi muc) / Uniform / HoliTom 5,2%
#   (loi torch.cat o FastVID 2%, da sua baselines/fastvid_port.py), roi RAP R05E/R02E + ELA 5/2% tren MLVU dev.
# Card 2 (GPU1): LVBench RAP R05E + ELA 5/2%.   bash outputs/hds5/bench52.sh <1|2>
CARD=${1:?card}
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1
PY=/home/hoang/miniconda3/envs/qwen25vl/bin/python
export DECORD_EOF_RETRY_MAX=131072 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$((CARD - 1))
O=outputs/hds5; ST=$O/STATUS_bench52_card$CARD.txt; MP=Qwen/Qwen2.5-VL-7B-Instruct; CK=outputs/checkpoints
log() { echo "$(date -u '+%Y-%m-%d %H:%M:%S') $*" >> $ST; }
ev() {  # ev <gt> <video_dir> <out_dir> <tag> <variant...>
  local gt=$1 vd=$2 od=$3 tag=$4; shift 4; local a=""; for v in "$@"; do a="$a --variant $v"; done
  for i in 1 2 3; do
    log "START $tag (lan $i)"
    timeout 16h $PY -m rap.eval_multi --gt_file $gt --video_dir $vd --model_path $MP --output_dir $od --max_frames 128 $a >> $O/logs/bench52_$tag.log 2>&1 && break
    sleep 30
  done
  log "END $tag: $(for v in "$@"; do echo -n "${v%%=*}=$(wc -l < $od/${v%%=*}.json) "; done)"
}
MLVU="playground/gt_qa_files/MLVU_Dev/val_qa_mc.json Datasets/MLVU_Dev/MLVU/video outputs/mlvu_dev/pred"
LVB="playground/gt_qa_files/LVBench/val_qa.json Datasets/LVBench/videos outputs/hds/lvbench"
if [ $CARD = 1 ]; then
  ev $MLVU mlvu_fix fastvid_r040=fastvid@0.40 fastvid_r025=fastvid@0.25 fastvid_r010=fastvid@0.10 fastvid_r005=fastvid@0.05 fastvid_r002=fastvid@0.02 \
     uniform_r005=uniform@0.05 uniform_r002=uniform@0.02 holitom_r005=holitom@0.05 holitom_r002=holitom@0.02
  ev $MLVU mlvu_rap R05E_r005=ckpt:$CK/R05E.pt@0.05 R02E_r002=ckpt:$CK/R02E.pt@0.02 ELA_r005=ckpt:$CK/ELA.pt@0.05 ELA_r002=ckpt:$CK/ELA.pt@0.02
else
  ev $LVB lvb_rap R05E_r005=ckpt:$CK/R05E.pt@0.05 ELA_r005=ckpt:$CK/ELA.pt@0.05 ELA_r002=ckpt:$CK/ELA.pt@0.02
fi
$PY -m tools.mlvu_dev_report >> $O/logs/report.log 2>&1
log "DONE card $CARD"
