#!/bin/bash
# Card 1 (GPU0), sau khi queue.sh (H3cv) xong:
#  1  H6cv    = H6ce (2%, CE 0.2, +-3) + 50% Vript
#  2  H9ng5   = H9ce5 (25%, CE 0.5, +-1) nhung bo cong (1-r)
#  3  H9ce10  (dot hds5, w_ceg 1.0, dang tam dung) chay tiep tu checkpoint
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1
VRIPT="--vript_index Datasets/Vript/vript_index.jsonl --vript_prob 0.5 --vript_scenes 8 --max_caption_tokens 768"
until grep -q "QUEUE DONE\|FAIL" outputs/hds6/STATUS.txt 2>/dev/null; do sleep 300; done
echo "$(date -u '+%Y-%m-%d %H:%M:%S') QUEUE START gpu0 (sau H3cv)" >> outputs/hds6/STATUS_gpu0.txt
bash outputs/hds6/run.sh 0 outputs/hds6 H6cv 0.02 002 --w_ceg 0.2 $VRIPT
bash outputs/hds6/run.sh 0 outputs/hds6 H9ng5 0.25 025 --w_ceg 0.5 --window 1 --gate 0
bash outputs/hds6/run.sh 0 outputs/hds5 H9ce10 0.25 025 --w_ceg 1.0 --window 1
echo "$(date -u '+%Y-%m-%d %H:%M:%S') QUEUE DONE gpu0" >> outputs/hds6/STATUS_gpu0.txt
