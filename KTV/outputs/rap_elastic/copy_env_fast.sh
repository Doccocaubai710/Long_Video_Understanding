#!/bin/bash
# 22/10: copy_env.sh dung "cp -a" ca thu muc -> torch (14.754 file nho) chi ~60 file/phut tren Lustre nghen (~3 gio).
# Ban nay chep tung file song song (64 luong), bo qua file da chep du kich thuoc; chep model Qwen song song cung luc.
#   bash outputs/rap_elastic/copy_env_fast.sh   (chay lai duoc, chi chep phan con thieu)
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1   # thu muc KTV
SRC=/home/hoang/miniconda3/envs/qwen25vl
DST=/root/local_env/qwen25vl
LOG=outputs/rap_elastic/copy_env.log
NPAR=${NPAR:-64}
log() { echo "$(date -u '+%Y-%m-%d %H:%M:%S') [fast] $*" >> $LOG; }
copy_f() {  # copy_f <file tuyet doi trong SRC>
  local d=$DST/${1#$SRC/}
  [ -L "$1" ] && { [ -L "$d" ] || { mkdir -p "$(dirname "$d")" && cp -a "$1" "$d"; }; return; }
  [ -f "$d" ] && [ "$(stat -c %s "$d")" = "$(stat -c %s "$1")" ] && return 0
  mkdir -p "$(dirname "$d")" && cp -a "$1" "$d.part" && mv "$d.part" "$d"
}
export -f copy_f; export SRC DST

log "START (pid $$, $NPAR luong)"
M=Qwen/Qwen2.5-VL-7B-Instruct; MD=/root/local_data/KTV/$M
( mkdir -p $MD; ls -A $M | xargs -d '\n' -P 8 -I{} cp -a "$M/{}" "$MD/" >> $LOG 2>&1
  n1=$(ls -A $M | wc -l); n2=$(ls -A $MD | wc -l)
  [ "$n1" = "$n2" ] && touch $MD/.complete && log "DONE model ($n2 file)" || log "LOI model ($n2/$n1)" ) &
mp=$!

L=/tmp/copy_env_fast.txt
find $SRC \( -type f -o -type l \) > $L
log "env: $(wc -l < $L) file"
xargs -a $L -d '\n' -P $NPAR -I{} bash -c 'copy_f "$1" || echo "LOI $1"' _ {} >> $LOG 2>&1
if $DST/bin/python -c "import torch, transformers, decord, cv2, qwen_vl_utils" >> $LOG 2>&1; then
  touch $DST/.complete; log "DONE env (import ok)"
else
  log "LOI env: khong import duoc"
fi
wait $mp
log "END"
