#!/bin/bash
# Dot hds7, GPU1 -- 5/11 (theo y kien anh: ablation o 25% / 40%): nhom D = cach chia hop.
#   O 40% hop mac dinh chi 2 token va BO TRANG 20% so hop; o 25% hop 4 token. Thu:
#   BK2 / BK4 = hop to ~2x / 4x, moi hop chon 2 / 4 token khac nhau (40%: bo 2% hop thay vi 20%)
#   BM1 = hop mac dinh, cho chon vuot bien 1 token moi phia (giai trung token).  x25 / x40 = muc giu.
GPU=1; source "$(dirname "$(readlink -f "$0")")/common.sh"
log "ABLATION D (25% / 40%) START gpu1 (pid $$)"
train BK2x40 --w_ceg 0 --box_k 2; train BK2x25 --w_ceg 0 --box_k 2
evalb mlvu BK2x40 BK2x25
train BK4x40 --w_ceg 0 --box_k 4; train BK4x25 --w_ceg 0 --box_k 4
evalb mlvu BK4x40 BK4x25
train BM1x40 --w_ceg 0 --box_margin 1; train BM1x25 --w_ceg 0 --box_margin 1
evalb mlvu BM1x40 BM1x25
ALL="BK2x40 BK2x25 BK4x40 BK4x25 BM1x40 BM1x25"
evalb videomme $ALL; evalb lvbench $ALL
log "ABLATION D DONE gpu1"
# --- tiep (card khong ranh): seed bang chinh chi KL ---
k=3
while true; do
  train K25s$k --w_ceg 0 --seed $k; train K40s$k --w_ceg 0 --seed $k
  evalb mlvu K25s$k K40s$k; evalb videomme K25s$k K40s$k; evalb lvbench K25s$k K40s$k
  k=$((k + 1))
done
