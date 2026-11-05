#!/bin/bash
# Dot hds7, GPU0 -- 5/11 (theo y kien anh: 10% da on -> ablation o 25% / 40%, tim cach nang 2 muc nay):
# nhom C = do luoi phat lech tam, 1 lambda chung (thoi gian + khong gian), CO DINH. Moc 0,5 da co (= K25 / K40, TB 3 seed).
#   Chay moi o ca 25% (x25) va 40% (x40): lambda 0 (PN0), 2 (PNf2), 1 (PNf1) -> pick_lambda.py chon moc MLVU dev tot nhat
#   -> do min 1-2 moc lan can -> VideoMME + LVBench cho moi moc. Truoc het cham not VideoMME / LVBench seed 2 (K25s2, K40s2) = moc tham chieu.
GPU=0; source "$(dirname "$(readlink -f "$0")")/common.sh"
log "ABLATION C (25% / 40%) START gpu0 (pid $$)"
evalb videomme K25s2 K40s2; evalb lvbench K25s2 K40s2
train PN0x25 --w_ceg 0 --lam_mode fixed --lambda_init 0;  train PN0x40 --w_ceg 0 --lam_mode fixed --lambda_init 0
evalb mlvu PN0x25 PN0x40
train PNf2x25 --w_ceg 0 --lam_mode fixed --lambda_init 2; train PNf2x40 --w_ceg 0 --lam_mode fixed --lambda_init 2
evalb mlvu PNf2x25 PNf2x40
train PNf1x25 --w_ceg 0 --lam_mode fixed --lambda_init 1; train PNf1x40 --w_ceg 0 --lam_mode fixed --lambda_init 1
evalb mlvu PNf1x25 PNf1x40
ALL="PN0x25 PN0x40 PNf2x25 PNf2x40 PNf1x25 PNf1x40"
for LV in x25 x40; do
  EXTRA=$($PY $O/pick_lambda.py $LV 2>> $O/logs/pick_lambda.log)
  log "pick_lambda $LV: $(cat $O/PEN_pick_$LV.txt 2>/dev/null)"
  while read -r n v; do
    [ -n "$n" ] || continue
    train $n --w_ceg 0 --lam_mode fixed --lambda_init $v; ALL="$ALL $n"
    evalb mlvu $n
  done <<< "$EXTRA"
done
evalb videomme $ALL; evalb lvbench $ALL
log "ABLATION C DONE gpu0"
# --- tiep (card khong ranh): seed bang chinh chi KL (K5s3 dang do, tu resume) ---
k=2
while true; do
  train K5s$k --w_ceg 0 --seed $k; train K2s$k --w_ceg 0 --seed $k
  evalb mlvu K5s$k K2s$k; evalb videomme K5s$k K2s$k; evalb lvbench K5s$k K2s$k
  k=$((k + 1))
done
