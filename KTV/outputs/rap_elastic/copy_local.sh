#!/bin/bash
# 21/10: Lustre (OST0001/2/4) nghen -> chep du lieu video sang o local cua container de train/cham khong phu thuoc Lustre.
# Ban sao o /root/local_data/KTV/<duong dan tuong doi>. O local MAT khi container bi tao lai; ban goc van tren Lustre.
# Moi thu muc chep xong + du so file -> ghi <dich>/.complete. Code chi dung ban local khi co .complete (khong thi doc Lustre).
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1   # thu muc KTV
DST=${KTV_LOCAL_DATA:-/root/local_data/KTV}
LOG=outputs/rap_elastic/copy_local.log
NPAR=${NPAR:-16}
log() { echo "$(date -u '+%Y-%m-%d %H:%M:%S') $*" >> $LOG; }
copy_one() {  # copy_one <file tuong doi>
  local f=$1 d=$DST/$1
  [ -f "$d" ] && [ "$(stat -c %s "$d")" = "$(stat -c %s "$f")" ] && return 0
  mkdir -p "$(dirname "$d")" && cp "$f" "$d.part" && mv "$d.part" "$d"
}
export -f copy_one; export DST
copy_dir() {  # copy_dir <thu muc tuong doi> [find -not -path ...]
  local src=$1; shift
  [ -f $DST/$src/.complete ] && { log "SKIP $src (da xong)"; return 0; }
  log "LIST $src"
  find "$src" -type f "$@" > $DST/.list.tmp 2>> $LOG
  local n; n=$(wc -l < $DST/.list.tmp)
  log "COPY $src: $n file, $(xargs -a $DST/.list.tmp -d '\n' stat -c %s | awk '{s+=$1} END {printf "%.1f GB", s/2^30}')"
  xargs -a $DST/.list.tmp -d '\n' -P $NPAR -I{} bash -c 'copy_one "$1" || echo "LOI $1"' _ {} >> $LOG 2>&1
  local ok; ok=$(cd $DST && xargs -a $DST/.list.tmp -d '\n' -I{} sh -c '[ -f "$1" ] && echo y' _ {} | wc -l)
  if [ "$ok" = "$n" ]; then touch $DST/$src/.complete; log "DONE $src ($ok/$n)"; else log "THIEU $src ($ok/$n)"; fi
}
mkdir -p $DST
log "START (pid $$, $NPAR luong)"
copy_dir Datasets/LLaVA-Video-178K-probe/academic_source -not -path "*/ego4d/*"
for j in Datasets/LLaVA-Video-178K-probe/*.json; do copy_one "$j"; done
copy_dir Datasets/Video-MME/data/data
copy_dir Datasets/MSRVTT/video
log "END"
