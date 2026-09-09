# Giải thích ý tưởng: DySeg + Phân bổ ngân sách không đều + STPrune

Tài liệu này giải thích **trực giác đằng sau từng thành phần** của ý tưởng (idea.md), dành cho việc đọc lại hiểu nhanh — không đi sâu công thức toán, chỉ tập trung "vì sao làm vậy" và "nó hoạt động thế nào" bằng ví dụ cụ thể. Kết quả thực nghiệm cuối cùng và kết luận nằm ở cuối tài liệu.

## 0. Bài toán tổng quát, bằng một câu

Video dài → mã hóa thành rất nhiều token thị giác (vài chục nghìn) → không đưa hết vào LLM được → phải **nén**. Câu hỏi: nén thế nào để giữ được nhiều thông tin nhất với ngân sách token cho trước?

Ý tưởng chia việc nén thành **3 tầng xử lý tuần tự**, mỗi tầng trả lời một câu hỏi con:

```
Video → [Tầng 1: DySeg]        → chia video thành các "đoạn"
      → [Tầng 2: Phân bổ]      → mỗi đoạn được BAO NHIÊU token?
      → [Tầng 3: STPrune]      → trong đoạn đó, giữ ĐÚNG token NÀO?
```

Ba tầng này độc lập với nhau về mặt thiết kế — có thể thay từng tầng riêng lẻ mà không cần đổi hai tầng còn lại (đây cũng chính là lý do các thí nghiệm ablation — `dtm_ratio=0`, `mode=uniform` — làm được dễ dàng: chỉ thay Tầng 3 hoặc bỏ luôn Tầng 1+2).

---

## 1. Tầng 1 — DySeg: chia video thành các "đoạn"

**Vì sao cần chia đoạn trước?** Một video không đồng nhất: có lúc camera đứng yên quay một người nói chuyện (rất ít thay đổi giữa các khung hình), có lúc chuyển cảnh đột ngột hoặc có hành động dồn dập (thay đổi nhiều). Nếu coi cả video là MỘT khối để xử lý, ta sẽ đối xử bất công — đoạn tĩnh và đoạn động nhận cùng một "công thức" nén, dù nhu cầu thông tin của chúng khác hẳn nhau.

**Cách làm — đo "độ khác biệt" giữa các khung liên tiếp:**

Sau khi video được mã hóa (mỗi "khung" ở đây thực ra là một *unit* đã gộp 2 frame gốc theo cơ chế nén thời gian riêng của Qwen2.5-VL — không phải frame gốc), ta tính độ tương đồng cosine giữa unit `i` và unit `i+1`:

```
trans[i] = cos(unit_i, unit_i+1)
```

- `trans[i]` **cao** (gần 1) → hai unit rất giống nhau → khả năng cao đang ở giữa một cảnh liên tục, KHÔNG nên cắt đoạn ở đây.
- `trans[i]` **thấp** → hai unit khác nhau nhiều → khả năng cao có chuyển cảnh/chuyển động lớn → NÊN đặt ranh giới đoạn tại đây.

**Hai tập ranh giới, hợp lại:**
- `S1`: lấy `c-1` điểm có `trans` thấp nhất trong TOÀN VIDEO — đảm bảo dù video "mượt" tuyệt đối (không có cảnh cắt rõ ràng nào), vẫn có tối thiểu `c` đoạn để phân bổ có ý nghĩa (không rơi về "cả video là 1 đoạn duy nhất").
- `S2`: lấy MỌI điểm có `trans < τ` (ngưỡng cố định, mặc định 0.9) — bắt các điểm cắt cảnh thật sự rõ ràng, bất kể có bao nhiêu điểm như vậy.

**Ví dụ cụ thể:** video 2 phút, phút đầu quay cảnh phỏng vấn tĩnh (một người ngồi nói), phút sau chuyển sang cảnh hành động (đi lại, nhiều góc quay). DySeg sẽ: (a) đặt một ranh giới rõ ràng đúng tại điểm chuyển cảnh (rơi vào `S2`, vì `trans` giảm mạnh ở đó), và (b) rải thêm vài ranh giới nhỏ trong đoạn phỏng vấn tĩnh (từ `S1`) dù đoạn đó không có gì đặc biệt — chỉ để đảm bảo có đủ số đoạn tối thiểu.

---

## 2. Tầng 2 — Phân bổ ngân sách token cho từng đoạn (đây là ý tưởng CHÍNH của bài)

Sau khi có các đoạn, cần quyết định: đoạn nào được cấp NHIỀU token hơn, đoạn nào ÍT hơn.

### 2a. Đo "độ giàu thông tin" của một đoạn — `v_s`

Hai chỉ số độc lập, đo hai khía cạnh khác nhau của "sự phong phú":

**(1) RECON — cần bao nhiêu khung "đại diện" để dựng lại cả đoạn?**

Coi mỗi đoạn như một tập vector (mỗi unit là 1 vector đặc trưng). Thuật toán *greedy pivot*: lặp lại việc chọn vector "khác biệt nhất so với phần đã dựng được" cho đến khi tái tạo được ~90% năng lượng (variance) của cả đoạn. Đếm số vòng lặp cần dùng, chia cho tổng số unit trong đoạn → tỉ lệ `k/P_s`.

- Đoạn **đơn điệu** (mọi unit gần giống nhau — ví dụ camera đứng yên): chỉ cần 1 unit "đại diện" là dựng lại được gần hết → tỉ lệ THẤP.
- Đoạn **đa dạng** (nhiều nội dung/góc nhìn khác nhau): cần nhiều unit đại diện mới dựng lại được → tỉ lệ CAO.

**(2) Delta-energy — trung bình mức thay đổi giữa các unit liên tiếp trong đoạn.**

```
delta[t] = unit[t] - unit[t-1]
```

Lấy độ lớn (norm) của `delta`, trung bình trong cả đoạn. Đoạn tĩnh → delta nhỏ. Đoạn có chuyển động/thay đổi liên tục → delta lớn.

**Kết hợp:**
```
v_s = λ · RECON_chuẩn_hóa + (1-λ) · delta_chuẩn_hóa      (λ = 0.5 mặc định)
```
Chuẩn hóa min-max **trong cùng một video** (không so giữa các video khác nhau, vì thang đo tuyệt đối không có ý nghĩa so sánh chéo).

### 2b. Từ `v_s` ra số token thật — cơ chế "nhiệt độ" (softmax) + "cổng entropy"

Ý tưởng cơ bản: giống cơ chế attention — đoạn có `v_s` cao thì "trọng số" softmax cao → được cấp nhiều token hơn.

```
B_s = B_min + (Ngân_sách_còn_lại) × softmax(v_s / T)
```

- `T` (nhiệt độ) càng **thấp** → phân bổ càng "sắc nét", chênh lệch giữa đoạn giàu/nghèo thông tin càng lớn (gần như dồn hết ngân sách cho đoạn tốt nhất).
- `T` càng **cao** → phân bổ càng gần với **chia đều** (uniform) — đây là điểm hay: `T → ∞` thì công thức này suy biến chính xác thành phân bổ đều, nghĩa là "chia đều" là một trường hợp riêng của công thức này, không phải một baseline hoàn toàn khác biệt.

**"Cổng entropy" — tự động chọn `T` phù hợp cho TỪNG VIDEO:**

Không phải video nào cũng nên phân bổ "sắc nét" như nhau. Nếu tất cả các đoạn trong video này đều có `v_s` na ná nhau (không có đoạn nào nổi bật hẳn) → không có lý do gì để dồn ngân sách vào một chỗ → nên phân bổ GẦN ĐỀU (tăng `T`). Nếu có vài đoạn nổi bật rõ rệt → nên tập trung ngân sách (giảm `T`).

Đo "độ nổi bật" bằng entropy của phân bố `v_s` (entropy cao = các đoạn đều nhau = không có gì nổi bật; entropy thấp = có đoạn nổi bật hẳn), rồi dùng nó để điều chỉnh `T` tự động — video nào "phẳng" thì tự làm phẳng luôn phân bổ, video nào "có điểm nhấn" thì tự làm sắc phân bổ.

`b_min`, `b_max`: sàn và trần cho mỗi đoạn — đảm bảo không đoạn nào bị bỏ hoàn toàn (0 token) hoặc nuốt hết ngân sách của cả video.

---

## 3. Tầng 3 — STPrune: trong một đoạn, giữ ĐÚNG token nào?

Đến đây, mỗi đoạn đã biết mình được bao nhiêu token (`B_s`). Nhưng một đoạn có `P_s × N` token tổng (P_s unit, N token/unit theo không gian) — cần chọn ra đúng `B_s` token để giữ, bỏ phần còn lại. STPrune chia `B_s` thành hai phần theo tỉ lệ `dtm_ratio` (viết tắt `d`):

```
B_s^DTM      = d · B_s        (token "đại diện")
B_s^salience = (1-d) · B_s    (token "nổi bật")
```

### 3a. DTM (Density-based Token Merging) — tìm token "đại diện"

Trong mỗi unit (N token, ví dụ N ~ vài trăm patch không gian), tính hai đại lượng cho từng token:

- **`rho` (mật độ cục bộ):** token này có bao nhiêu "hàng xóm" gần giống nó? Nhiều hàng xóm giống → `rho` cao → token này *đại diện* cho một vùng lớn (ví dụ: một mảng tường trơn, lặp lại nhiều patch giống nhau).
- **`delta` (khoảng cách tới cụm khác):** token gần nhất có `rho` CAO HƠN nó nằm cách bao xa? Nếu xa → token này là "đỉnh" của một cụm riêng biệt, không bị "che khuất" bởi một đại diện khác gần đó.

Token có **cả `rho` cao lẫn `delta` cao** (`score = rho × delta`) → vừa đại diện cho một vùng lớn, vừa không trùng lặp với đại diện khác → được chọn làm **anchor**.

Sau khi chọn anchor, các token còn lại trong unit được "gán" về anchor gần nhất (theo cosine similarity). Giá trị cuối cùng của anchor không giữ nguyên 100% — nó được **trộn**:

```
giá_trị_anchor_cuối = α × giá_trị_gốc_của_anchor + (1-α) × trung_bình_các_token_được_gán_về_nó
```

Ý tưởng: anchor không chỉ đại diện cho chính nó, mà "hấp thụ" một phần thông tin trung bình của cả cụm nó đại diện — giảm mất mát so với việc chỉ giữ nguyên 1 token và vứt bỏ hoàn toàn phần còn lại của cụm.

### 3b. Salience (độ nổi bật theo thời gian) — thay thế [CLS]-attention

Các phương pháp gốc (FastVID) dùng attention của token `[CLS]` (một token đặc biệt tổng hợp thông tin toàn ảnh trong CLIP) để đo "token nào quan trọng". **Qwen2.5-VL không có token `[CLS]`**, nên cần thước đo khác, hoạt động ở mọi backbone: đo độ thay đổi của TỪNG token so với đúng vị trí tương ứng ở unit trước đó:

```
sal(t, i) = ‖unit[t, token i] − unit[t-1, token i]‖
```

Token ở một vị trí không gian cố định (ví dụ góc trên-trái của khung hình) mà giá trị thay đổi nhiều so với unit trước → có gì đó MỚI vừa xảy ra ở đó (vật thể di chuyển vào, ánh sáng đổi...) → giữ lại. Token không đổi → tĩnh, thông tin đã "có ở chỗ khác" (unit trước), có thể bỏ.

Lấy top `(1-d)·B_s` token có `sal` cao nhất trong CẢ ĐOẠN (không giới hạn theo từng unit như DTM), ưu tiên những token chưa bị DTM chọn trước đó.

### 3c. Thử thay `sal` bằng tín hiệu khác — có 4 lựa chọn (`salience_mode`)

Công thức `sal(t,i)` ở trên (`delta`) chỉ nhìn "1 bước lùi" (so với unit ngay trước), là tín hiệu chuyển động thô. Code hỗ trợ đổi công thức chấm điểm này mà không đụng gì tới phần còn lại của pipeline (DTM, DySeg, gather/scatter giữ nguyên):

| `salience_mode` | Công thức | Ý tưởng |
|---|---|---|
| `delta` (mặc định) | `‖X[t]-X[t-1]‖` | Chuyển động thô, chỉ nhìn 1 bước lùi |
| `norm` | `‖X[t]‖` | Độ lớn nội dung thô, không so sánh thời gian |
| `grid` | (không chấm điểm) | Chọn đều trong các vị trí còn lại — đối chứng "không cần tín hiệu gì" |
| `vjepa` | sai số dự đoán che của V-JEPA 2 | Ngữ cảnh TOÀN VIDEO, đo "độ khó đoán/bất ngờ" về ngữ nghĩa |

**Pipeline của `vjepa`** (tách hoàn toàn khỏi lúc suy luận, chạy 1 lần/video, cache lại — không đọc câu hỏi, đúng tinh thần query-agnostic):

```
                    ┌─ precompute 1 LẦN/video, cache lại ─────────────────────┐
                    │                                                          │
   Video gốc  →  lấy đều 64 frame  →  V-JEPA 2 (che 1 phần, đoán lại)  →  bản đồ
                                        "độ khó đoán" mỗi vùng             lỗi dự đoán
                                                                          [32×16×16]
                    └──────────────────────────────────────────────────────────┘
                                                    │
                                                    ▼  (nội suy vị trí, quy đổi sang lưới token Qwen)
   Video gốc  →  Qwen2.5-VL vision encoder  →  token lưới [T×H×W]  ←──────┘
                                                    │
                                                    ▼
                                     DySeg → STPrune (40% DTM giữ nguyên,
                                              60% chọn theo sal = lỗi dự đoán V-JEPA)
```

1. Model dùng: `facebook/vjepa2-vitl-fpc64-256`, chia video thành lưới patch 32×16×16 = 8192 patch (thời gian × cao × rộng).
2. Vì V-JEPA vốn huấn luyện kiểu che ngẫu nhiên 75-90% rồi đoán lại, nhưng ta cần MỌI patch đều có điểm số: chia 8192 patch thành 4 nhóm ngẫu nhiên không trùng nhau, chạy model 4 lần — mỗi lần 1 nhóm là "phần bị che cần đoán", 3 nhóm còn lại (75%) làm ngữ cảnh.
3. Với mỗi patch bị che: `sal = ‖vector model đoán − vector thật (do chính encoder tạo ra)‖` — càng lớn càng "bất ngờ", càng khó đoán từ ngữ cảnh xung quanh.
4. Lưới lỗi dự đoán V-JEPA [32×16×16] khác kích thước lưới token thật của Qwen (Qwen lấy mẫu khung động theo độ dài video, còn V-JEPA luôn cố định 64 khung/256×256) — cần nội suy: mỗi vị trí token Qwen mượn điểm của ô V-JEPA gần nhất theo tỷ lệ vị trí. Đây là một phép quy đổi gần đúng, không khớp 1-1 hoàn hảo — với video dài, lấy mẫu cố định 64 khung của V-JEPA khá thô so với chính lưới token động của Qwen, là một nghi vấn khả dĩ (chưa kiểm chứng) cho kết quả ở §5.

Xem `precompute_vjepa_salience.py` (bước 1-3) và `map_vjepa_to_qwen_grid()` trong `run_inference_qwen25vl_dyseg.py` (bước 4).

---

## 4. Bước bắt buộc phía sau: bảo toàn vị trí (position preservation)

Đây không phải một "tầng" thuật toán, mà là một **ràng buộc kỹ thuật bắt buộc** khi cài đặt: Qwen2.5-VL dùng mRoPE — mã hóa vị trí không-thời-gian TUYỆT ĐỐI của mỗi token thị giác (token này ở khung thứ mấy, hàng/cột nào trong ảnh). Khi bỏ bớt token, nếu không giữ đúng chỉ số vị trí GỐC của các token còn lại (mà đánh số lại từ 0), model sẽ hiểu sai hoàn toàn cấu trúc không-thời-gian của video — **không báo lỗi, không crash, chỉ âm thầm trả lời sai nhiều hơn**. Cách làm an toàn nhất (đã dùng trong code): để model tự tính vị trí cho TOÀN BỘ chuỗi chưa bị cắt (một phép tính đã biết chắc đúng), rồi chỉ GOM lại đúng các hàng tương ứng với token được giữ — không bao giờ tự tính lại công thức mRoPE bằng tay.

---

## 5. Kết quả thực nghiệm cuối cùng và kết luận

Toàn bộ so sánh dưới đây chạy trên VideoMME (2700 câu), Qwen2.5-VL-7B-Instruct, cùng harness đã kiểm chứng khớp literature (Native dense 1x ≈ 63.33% ≈ SCORE Vanilla 63.8%). Hai regime token khác nhau:

- **1x** (`MAX_PIXELS = 16384×28×28`): pool dense chỉ ~14-16K token/video — gần bằng `tokens_num=16384`, nên DySeg gần như không có margin nén thật (chủ yếu chỉ phân bổ lại một ngân sách đã nhỏ sẵn).
- **3x** (`MAX_PIXELS = 16384×3×28×28`): pool dense ~42-48K token/video, nén thật xuống 16384 (~36% retention) — phép đo có ý nghĩa hơn.

| Cấu hình | Regime | Overall | Short | Medium | Long |
|---|---|---|---|---|---|
| Native (dense 100%, không nén) | 1x (~14-16K) | 63.33% | 73.33 | 63.22 | 53.44 |
| DySeg + STPrune, `dtm_ratio=0.4` | 1x (→16384, ~95-100% retention) | 61.78% | 71.56 | 61.00 | 52.78 |
| Native (dense 100%, không nén) | 3x (~42-48K) | 64.26% | 73.67 | 65.00 | 54.11 |
| **Uniform (lấy đều, không DySeg/STPrune)** | 3x (→16384, ~36%) | **64.41%** | 73.00 | 66.00 | 54.22 |
| DySeg + STPrune, `dtm_ratio=0.4` (đầy đủ, `salience_mode=delta`) | 3x (→16384, ~36%) | 61.96% | 70.00 | 61.67 | 54.22 |
| DySeg (Tầng 1+2) + salience thuần, `dtm_ratio=0` (bỏ DTM) | 3x (→16384, ~36%) | 62.15% | 69.22 | 62.11 | 55.11 |
| DySeg + STPrune, `temperature0→∞` (tắt Tầng 2, phân bổ đều) | 3x (→16384, ~36%) | 62.41% | 69.89 | 63.11 | 54.22 |
| DySeg + STPrune, `salience_mode=grid` (không chấm điểm gì) | 3x (→16384, ~36%) | 62.41% | 70.11 | 62.67 | 54.44 |
| DySeg + STPrune, `salience_mode=vjepa` (V-JEPA, xem §3c) | 3x (→16384, ~36%) | 61.74% | 69.11 | 61.44 | 54.67 |
| **Uniform** (lấy đều) | 3x (→3240, **~7.2%**) | **60.30%** | 68.11 | 61.44 | 51.33 |
| DySeg + STPrune, `dtm_ratio=0.4` (đầy đủ) | 3x (→3240, **~7.2%**) | 58.70% | 65.44 | 58.56 | 52.11 |

*(Uniform và `dtm_ratio=0` chỉ chạy ở regime 3x — ở 1x không có margin nén thật nên phép so sánh không nhiều ý nghĩa; xem §5.2. Hàng ~7.2% dùng `tokens_num=3240`, khớp đúng số token tuyệt đối FastVID paper báo cáo — xem §5.1.)*

**DySeg vẫn thua Uniform ở mức nén sâu ~7.2%** (−1.60 điểm, z≈−1.20, không có ý nghĩa thống kê nhưng cùng chiều mọi lần đo khác). Mẫu hình bucket lặp lại lần thứ 3, rất nhất quán: Short −2.67, Medium −2.88, **Long +0.78** (DySeg vẫn nhỉnh hơn Uniform ở bucket Long) — giống hệt xu hướng ở 36% retention. Cả Uniform lẫn DySeg đều tụt mạnh so với 16384 token (Uniform −4.11, DySeg −3.26), xác nhận 3240 token mới là vùng "nén có giá", khác 16384 (vùng "nén miễn phí").

### 5.1 So với chính paper FastVID trên Qwen2.5-VL — Vanilla của họ cao hơn hẳn

Bảng của FastVID (Table 4, Qwen2.5-VL, cùng cấu hình "max 16384 video token, max 768 frame" — gần như đúng thiết lập 1x của mình):

| | # Token | Overall | % giữ so với Vanilla của chính nó |
|---|---|---|---|
| Vanilla (FastVID paper) | 13447.1 (100%) | 67.3% | 100% |
| FastVID (paper) | 3240.7 (24.1%) | 62.8% | 93.3% |
| PruneVID (paper) | 3294.5 (24.5%) | 59.3% | 88.1% |

Điểm đáng chú ý: **13447.1 token của FastVID paper gần khớp đúng dải 13,680-16,272 mà mình tự đo được ở Native 1x** — tức cùng một cấu hình "max_pixels=16384" cơ bản. Nhưng Vanilla của họ (67.3%) cao hơn Vanilla của mình (63.33%) tới **~4 điểm**, dù đã sửa cả 4 chỗ harness (prompt, system prompt, generation config, frame/resolution sampling). Đây là khoảng cách CHƯA giải thích được, tách biệt hoàn toàn khỏi câu hỏi "DySeg có thắng Uniform không".

**Vì vậy so sánh công bằng hơn giữa các paper là tính theo % giữ được so với CHÍNH Vanilla của từng bên**, không so số tuyệt đối:

| | Retention | % giữ so với Vanilla của chính nó |
|---|---|---|
| FastVID (paper) | 24.1% | 93.3% |
| PruneVID (paper) | 24.5% | 88.1% |
| Uniform (của mình, 3x, 36%) | 36% | **100.2%** |
| DySeg `dtm_ratio=0.4` (của mình, 3x, 36%) | 36% | 96.4% |
| DySeg `dtm_ratio=0` (của mình, 3x, 36%) | 36% | 96.7% |
| Uniform (của mình, 3x, ~7.2% — khớp absolute token FastVID) | ~7.2% | 93.8% |
| DySeg `dtm_ratio=0.4` (của mình, 3x, ~7.2%) | ~7.2% | 91.4% |

Theo cách tính này, **ở 36% DySeg giữ % tốt hơn FastVID publish** (96.4-96.7% > 93.3%), nhưng **ở ~7.2% (khớp đúng absolute token count của FastVID) thì lại giữ KÉM HƠN** (91.4% < 93.3%) — khoảng cách với FastVID nới rộng khi nén càng sâu, khớp hướng với giả thuyết DTM/salience (chỉ hoạt động đúng thiết kế trong FastVID gốc — xem §3.2-3.3 và §9.3 của tài liệu chính) ngày càng quan trọng khi ngân sách càng eo hẹp. Điều này không đổi kết luận nội bộ (§5.2: DySeg vẫn thua Uniform ở cùng harness, cùng ngân sách, ở cả hai mức retention), nhưng cho thấy một phần khoảng cách tuyệt đối với literature đến từ gap Vanilla-vs-Vanilla chưa giải thích được (~4 điểm, không đổi theo mức nén), phần còn lại đến từ chính chất lượng nén khi siết sâu.

### 5.2 Kết luận

1. **Nén gần như không mất gì so với dense**, bất kể cách nén: Uniform (lấy đều, không cần thuật toán gì) đạt 64.41%, ngang bằng thậm chí nhỉnh hơn Native dense 3x (64.26%, không có ý nghĩa thống kê). Pool dense ~42-48K token có độ dư thừa rất lớn — nén xuống còn 36% gần như "miễn phí".

2. **Nhưng phân bổ không đều theo đoạn (ý tưởng chính) KHÔNG thắng nổi phân bổ đều đơn giản** — thậm chí thua nhẹ: DySeg đầy đủ thua Uniform 2.45 điểm (z≈-1.87, p≈0.062 — sát ngưỡng ý nghĩa thống kê). Bỏ hẳn DTM merge (`dtm_ratio=0`, chỉ còn salience thô) **không cứu được** — vẫn thua Uniform ~2.3 điểm với ý nghĩa tương tự. Điều này loại trừ giả thuyết "lỗi do DTM merge làm hỏng giá trị token" — vấn đề nằm ở chính công thức đo `v_s` và/hoặc cách phân bổ ngân sách (Tầng 1+2), không phải ở Tầng 3.

3. **Điểm sáng duy nhất, nhất quán qua cả 2 lần thử ở 3x:** bucket **Long** luôn ngang bằng hoặc nhỉnh hơn cả Uniform lẫn Native (54.11-54.22% ở dense/uniform, lên tới 55.11% khi bỏ DTM) — trong khi Short/Medium luôn bị mất điểm rõ rệt. Đây đúng hướng giả thuyết ban đầu của ý tưởng (video dài có nhiều dư thừa hơn nên phân bổ không đều nên có lợi thế ở đó), nhưng hiện tại chỉ đủ mạnh để "không thua", chưa đủ mạnh để "thắng" — và bị pha loãng bởi khoản lỗ ở Short/Medium khi tính overall.

4. **Cả 4 ablation đơn-yếu-tố (bỏ DTM, tắt Tầng 2, `salience_mode=grid`, `salience_mode=vjepa`) đều hội tụ về cùng một dải hẹp 61.7-62.4%** — cách Uniform (64.41%) một khoảng gần như không đổi bất kể sửa yếu tố nào. Đáng chú ý nhất: đổi tín hiệu salience từ chuyển động thô (`delta`) sang tín hiệu ngữ nghĩa mạnh hơn hẳn dựa trên V-JEPA 2 (§3c) **không cải thiện gì** (61.74% so với 61.96%, và còn kém hơn cả `grid` — tức "không chấm điểm gì cả"). Kết luận: **chất lượng tín hiệu chấm điểm salience không phải là nút thắt** của khoảng cách với Uniform — vấn đề nằm ở đâu đó cấu trúc hơn (rất có thể là chính cơ chế DTM hoặc DySeg/gather), nhưng vì không ablation đơn lẻ nào đóng được khoảng cách, bài test quyết định (tắt ĐỒNG THỜI DTM + Tầng 2 + dùng `salience_mode=grid`) vẫn còn treo, chưa chạy.

**Ở cấu hình và siêu tham số hiện tại, ý tưởng phân bổ ngân sách không đều theo độ khó tái cấu trúc + salience thời gian không cho thấy lợi thế so với phân bổ đều đơn giản trên Qwen2.5-VL-7B/VideoMME** (phép so sánh nội bộ, cùng harness, cùng ngân sách, cùng pool dense — đây là phép đo đáng tin nhất). Hướng đáng cân nhắc nếu muốn tiếp tục: (a) tập trung riêng cho video dài (nơi tín hiệu tích cực nhất quán xuất hiện) thay vì áp dụng đồng loạt cho mọi bucket, (b) xem lại chính công thức `v_s` (RECON + delta-energy) có thực sự đo đúng "độ đáng giữ lại" hay không, (c) truy tiếp gap Vanilla-vs-FastVID-paper (~4 điểm, §5.1) vì nó ảnh hưởng đến mọi số tuyệt đối đang báo cáo.
