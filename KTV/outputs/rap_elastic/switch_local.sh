#!/bin/bash
# 21/10: khi chep xong video train sang o local -> doi checkpoint ke tiep cua M3D-BD duoc luu -> dung tien trinh train;
# queue6 (do_train) tu chay lai voi --resume, luc nay doc video tu o local (local_data.py).
O=/home/hoang/Long_Video_Understanding/KTV/outputs/rap_elastic
M=/root/local_data/KTV/Datasets/LLaVA-Video-178K-probe/academic_source/.complete
until [ -f $M ]; do sleep 60; done
pid=$(pgrep -f "python train_rap_elastic.py --out outputs/rap_elastic/M3D-BD.pt")
[ -z "$pid" ] && { echo "$(date -u '+%Y-%m-%d %H:%M:%S') SWITCH: khong thay tien trinh train M3D-BD" >> $O/STATUS.txt; exit 0; }
t0=$(stat -c %Y $O/M3D-BD.pt)
until [ "$(stat -c %Y $O/M3D-BD.pt)" != "$t0" ]; do sleep 30; done
sleep 120   # cho ghi checkpoint xong han
echo "$(date -u '+%Y-%m-%d %H:%M:%S') SWITCH M3D-BD sang doc video tu o local (dung pid $pid sau checkpoint, queue tu --resume)" >> $O/STATUS.txt
kill $pid
