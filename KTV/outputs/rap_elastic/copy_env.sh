#!/bin/bash
# 21/10: Lustre nghen -> import torch tu env tren Lustre mat > 1 gio. Chep env qwen25vl + model Qwen sang o local.
#   env  -> /root/local_env/qwen25vl            (xong: file .complete)
#   Qwen -> /root/local_data/KTV/Qwen/Qwen2.5-VL-7B-Instruct (xong: file .complete; local_data.py tu dung)
# Chep song song tung goi/thu muc con de khong bi ket sau mot file nam tren OST cham.
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1   # thu muc KTV
SRC=/home/hoang/miniconda3/envs/qwen25vl
DST=/root/local_env/qwen25vl
LOG=outputs/rap_elastic/copy_env.log
log() { echo "$(date -u '+%Y-%m-%d %H:%M:%S') $*" >> $LOG; }
log "START env (pid $$)"
mkdir -p $DST/lib/python3.10
items() {
  for x in $SRC/*; do [ "$(basename $x)" != lib ] && echo "$x|$DST/"; done
  for x in $SRC/lib/*; do [ "$(basename $x)" != python3.10 ] && echo "$x|$DST/lib/"; done
  for x in $SRC/lib/python3.10/*; do [ "$(basename $x)" != site-packages ] && echo "$x|$DST/lib/python3.10/"; done
  mkdir -p $DST/lib/python3.10/site-packages
  for x in $SRC/lib/python3.10/site-packages/*; do echo "$x|$DST/lib/python3.10/site-packages/"; done
}
items > /tmp/copy_env_items.txt
log "env: $(wc -l < /tmp/copy_env_items.txt) muc"
xargs -a /tmp/copy_env_items.txt -d '\n' -P 48 -I{} bash -c 's=${1%%|*}; d=${1##*|}; cp -a "$s" "$d" || echo "LOI $s"' _ {} >> $LOG 2>&1
if [ -x $DST/bin/python ] && $DST/bin/python -c "import torch, transformers, decord, cv2, qwen_vl_utils" >> $LOG 2>&1; then
  touch $DST/.complete; log "DONE env (import torch/transformers/decord/cv2/qwen_vl_utils ok)"
else
  log "LOI env: khong import duoc"
fi
M=Qwen/Qwen2.5-VL-7B-Instruct; MD=/root/local_data/KTV/$M
log "START model $M"
mkdir -p $MD
ls -A $M | xargs -d '\n' -P 8 -I{} cp -a "$M/{}" "$MD/" >> $LOG 2>&1
n1=$(ls -A $M | wc -l); n2=$(ls -A $MD | wc -l)
[ "$n1" = "$n2" ] && touch $MD/.complete && log "DONE model ($n2 file)" || log "LOI model ($n2/$n1)"
log "END"
