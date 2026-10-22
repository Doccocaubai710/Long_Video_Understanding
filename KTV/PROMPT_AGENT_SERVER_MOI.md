# Prompt cho agent (Claude Code) trên server mới

Copy nguyên khối bên dưới (từ dòng `---` đầu tiên đến hết file) và dán cho agent ở server mới.
Trước khi dán, sửa 2 chỗ trong ngoặc `<...>`: đường dẫn repo, và cách lấy checkpoint từ server cũ.

---

Bạn đang ở server mới, tiếp nhận một hàng đợi thí nghiệm ML (train + chấm) chuyển sang từ server cũ.
Repo đã clone sẵn ở `<DUONG_DAN>/Long_Video_Understanding`. Mọi lệnh bên dưới chạy từ thư mục `KTV/`.

## Bối cảnh

- Đề tài: RAP, một bộ chọn token thị giác học được (một policy nhỏ được train, Qwen2.5-VL-7B giữ đóng băng).
  Policy giữ lại 25% hoặc 10% token video trước khi đưa vào Qwen2.5-VL-7B.
  Chấm trên Video-MME (2.700 câu trắc nghiệm, 128 frame). Mốc so sánh: Uniform, HoliTom, RunE25, R10E.
- Toàn bộ hàng đợi nằm trong `outputs/rap_elastic/queue6.sh`. Đọc phần chú thích đầu file để biết thứ tự các run và ý tưởng.
- Tiến độ: `outputs/rap_elastic/STATUS.txt`. Kết quả: `outputs/rap_elastic/KET_QUA.txt`. Run bị ngắt sớm: `STOPS.txt`.
  Queue tự bỏ qua run đã xong (dựa vào `*.train.log` có dòng `[train_rap] done` và file `pred_*.json` đủ 2.700 dòng).
  Vì vậy **không xóa** file log/metrics/pred nào trong `outputs/rap_elastic/`.
- Đã xong ở server cũ: M3D-B (63,85), SU25 (63,59), C0, ELA/ELA_s1... M3D-BD đã train xong 4000/4000
  và đang chấm dở. Queue sẽ chấm tiếp phần còn lại.
- `KTV/README.md` là README gốc của paper KTV (pipeline cũ), **không dùng** cho việc này.

## Việc cần làm

**Hướng dẫn chi tiết (link tải, cấu trúc thư mục, dung lượng): `KTV/SETUP_SERVER_MOI.txt`. Đọc toàn bộ file đó trước.**
Tóm tắt các bước:

1. **Kiểm tra máy:** GPU (`nvidia-smi`). Mỗi run cần 1 GPU khoảng 32 GB VRAM (server cũ dùng A100 40 GB) và CUDA 12.8.
   Kiểm tra ổ đĩa còn trống: cần khoảng 150 GB cho data + model. Ghi lại thư mục nào là ổ local, thư mục nào là ổ mạng.
2. **Môi trường:** `conda env create -f qwen25vl_environment.yml` (xóa dòng `prefix:` cuối file nếu lỗi), rồi `conda activate qwen25vl`.
   Kiểm tra: `python -c "import torch, transformers, decord, cv2, qwen_vl_utils; print(torch.__version__, transformers.__version__, torch.cuda.is_available())"`.
   Phải ra `2.11.0+cu128 5.14.1 True`.
3. **Model** (mục 3 của SETUP): `Qwen/Qwen2.5-VL-7B-Instruct` vào `KTV/Qwen/Qwen2.5-VL-7B-Instruct`,
   và `facebook/vjepa2-vitl-fpc64-256` vào `KTV/facebook/vjepa2-vitl-fpc64-256` (dùng cho bước 4a của queue).
4. **Dữ liệu** (mục 4 của SETUP), đặt trong `KTV/Datasets/`:
   - LLaVA-Video-178K: **chỉ đúng 2.601 video** liệt kê trong `playground/llava_video_probe_files.txt`, cùng 6 file json.
     Code train dùng mọi video tìm thấy, nên tải thừa sẽ làm tập train khác server cũ. Tải tar rồi chỉ giải nén các file có trong danh sách.
   - Video-MME: 900 video `.mp4` vào `Datasets/Video-MME/data/data/`.
   - COCO: 200 ảnh trong `playground/coco_val_pseudo_files.txt` vào `Datasets/COCO/train2017/`. **Bắt buộc**, thiếu là train lỗi ngay.
   - MSRVTT: `Datasets/MSRVTT/video/*.mp4`, chỉ cho run V-JEPA.
   - Sau khi tải, đếm lại: 2.601 video train, 900 video Video-MME, 200 ảnh COCO. Đếm sai thì báo tôi, không tự chạy tiếp.
5. **Checkpoint:** `outputs/rap_elastic/M3D-BD.pt` (587 MB) không có trên GitHub.
   `<CACH LAY: vi du "toi da scp san vao outputs/rap_elastic/M3D-BD.pt" hoac "hoi toi">`.
   Nếu chưa có file này thì dừng lại và hỏi tôi. Không được để queue train lại M3D-BD từ đầu.
6. **Chạy thử nhanh trước khi chạy cả queue:**
   `python infer_rap_videomme.py --help` không lỗi, và `python outputs/rap_elastic/tools.py report` chạy được.
7. **Chạy queue:**
   `PY=$(which python) bash outputs/rap_elastic/start_all.sh`
   **Bắt buộc có `PY=`**. Thiếu `PY` thì script chạy nhánh của server cũ (chép env từ `/home/hoang/miniconda3`), sẽ sai.
   Lệnh này chạy `queue6.sh` + `keepalive.sh` ở chế độ nền (setsid nohup), nên đóng phiên vẫn không bị dừng.
   Nếu `Datasets/` nằm trên ổ mạng chậm, có thể chép sang ổ local trước:
   `KTV_LOCAL_DATA=/o/local/KTV bash outputs/rap_elastic/copy_local.sh`, rồi chạy queue với cùng biến đó.
   Code tự dùng bản local khi có file `.complete`.
8. **Theo dõi:** sau khoảng 10 phút xem `STATUS.txt` và log của run đang chạy
   (`outputs/rap_elastic/<ten>.train.log` hoặc `pred_videomme_f128_<ten>_r*.log`) để chắc chắn đang chạy, không lỗi.
   Báo lại cho tôi: run nào đang chạy, tốc độ (giây/bước khi train, câu/phút khi chấm), ước tính thời gian xong.

## Quy tắc

- Không sửa công thức hoặc tham số của các run trong `queue6.sh`: kết quả phải so sánh được với server cũ.
  Chỉ sửa đường dẫn hoặc môi trường nếu cần, và nói rõ đã sửa gì.
- Không commit checkpoint `.pt`, data hay model. Không force-push. Commit/push chỉ khi tôi bảo.
- Không xóa file trong `outputs/`.
- Gặp lỗi thì đọc log, sửa nguyên nhân thật rồi chạy lại. Queue có `--resume` nên dừng giữa chừng không mất tiến độ.
  Nếu phải đổi điều gì ảnh hưởng tới kết quả thí nghiệm thì hỏi tôi trước.
