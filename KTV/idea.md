# PHÂN BỔ NGÂN SÁCH TOKEN KHÔNG ĐỒNG NHẤT, ĐỘC LẬP VỚI CÂU HỎI

**Query-Agnostic Non-Uniform Budget Allocation for Efficient Video LLMs**

*Tài liệu thiết kế phương pháp — v1.0*

---

## MỤC LỤC

- [Phần 0 — Vì sao `Rel(x|Q)` xuất hiện trong một phương pháp query-agnostic](#phần-0)
- [Phần 1 — Đặt bài toán](#phần-1)
- [Phần 2 — Tầng 1: Phân đoạn thời gian động](#phần-2)
- [Phần 3 — Tầng 2: Phân bổ ngân sách không đồng nhất (đóng góp chính)](#phần-3)
- [Phần 4 — Tầng 3: Chọn token trong đoạn](#phần-4)
- [Phần 5 — Thuật toán đầy đủ](#phần-5)
- [Phần 6 — Định vị và novelty](#phần-6)
- [Phần 7 — Quy trình thực nghiệm](#phần-7)
- [Phần 8 — Sổ trích dẫn](#phần-8)
- [Phần 9 — Abstract](#phần-9)
- [Phụ lục A — Bối cảnh: các thử nghiệm đã thực hiện](#phụ-lục-a)
- [Phụ lục B — Survey nhánh token compression](#phụ-lục-b)
- [Phụ lục C — Về đề xuất V-JEPA / RL của leader](#phụ-lục-c)

---

<a name="phần-0"></a>
## PHẦN 0 — VÌ SAO `Rel(x_i | Q)` XUẤT HIỆN TRONG MỘT PHƯƠNG PHÁP QUERY-AGNOSTIC

Đây là chỗ dễ hiểu nhầm nhất của toàn bộ thiết kế, nên đặt lên đầu.

Câu trả lời ngắn: **`Rel` được định nghĩa để chứng minh rằng ta KHÔNG cần nó.** Nó là đối tượng bị loại bỏ, không phải đối tượng được sử dụng.

### 0.1 Phân biệt hai loại xuất hiện của Q

| | Nơi xuất hiện | Có phá vỡ query-agnostic không? |
|---|---|---|
| **(A)** Trong **phân tích lý thuyết** — để định nghĩa khái niệm và chứng minh mệnh đề | ❌ Không |
| **(B)** Trong **thuật toán chạy lúc inference** — thuật toán đọc Q để ra quyết định | ✅ Có |

`Rel(x_i | Q)` chỉ xuất hiện ở (A). Không có dòng nào trong pseudo-code từ bước 1 đến bước 15 đọc `Q`. Thuật toán chỉ nhận `Q` ở bước cuối cùng, khi đưa vào LLM — nhưng lúc đó việc nén đã hoàn tất từ lâu và đã được cache.

**So sánh:** khi chứng minh một thuật toán sắp xếp là đúng, bạn phải định nghĩa "hoán vị sai" và "hoán vị đúng". Việc định nghĩa "hoán vị sai" không có nghĩa thuật toán tạo ra hoán vị sai — nó có nghĩa bạn cần khái niệm đó để chứng minh thuật toán tránh được nó.

### 0.2 Cấu trúc lập luận đầy đủ

Định nghĩa hai đại lượng ở mức token:

```
R(x_i | X)     = mức độ x_i có thể được dự đoán từ X \ {x_i}     (ĐỘ DƯ THỪA)
Rel(x_i | Q)   = mức độ x_i đóng góp vào việc trả lời Q          (ĐỘ LIÊN QUAN)
```

Lý tưởng nhất, ta muốn giữ token có `Rel` cao. Nhưng `Rel` cần `Q`, mà `Q` thay đổi theo từng câu hỏi → **không cache được**.

> **Mệnh đề.** `R(x_i | X) ⊥ Q` — độ dư thừa độc lập với câu hỏi.

**Chứng minh trực giác.** `R` chỉ phụ thuộc vào phân bố chung của các token trong chính video đó. Một patch tường tĩnh có thể dự đoán được từ các patch lân cận **bất kể ai đang hỏi gì**. Loại bỏ một token dư thừa hoàn toàn là thao tác bảo toàn thông tin **dưới mọi câu hỏi**, vì token đó khôi phục được từ phần còn lại.

> **Hệ quả.** Nếu một bộ nén **chỉ** loại bỏ token dư thừa (không cố gắng ước lượng `Rel`), thì:
> 1. Nó không cần `Q` → tính một lần, cache, dùng cho mọi câu hỏi của video.
> 2. Đầu ra hợp lệ cho mọi `Q`, không chỉ `Q` đã thấy.

**Đây chính là chỗ `Rel` phát huy tác dụng trong lập luận:** phải định nghĩa `Rel` mới nói được câu "chúng tôi tối ưu `R` chứ không phải `Rel`, và đó là lý do phương pháp cache được." Không có `Rel` trong phần setup thì mệnh đề trên không phát biểu được.

### 0.3 Bỏ `Rel` thì mất gì?

Đặt bài toán chọn tối ưu ở hai chế độ:

```
S*(Q) = argmax_{|S|=K}  A(X_S, Q)              (query-aware: mỗi câu hỏi một tập)
S*    = argmax_{|S|=K}  E_Q [ A(X_S, Q) ]      (query-agnostic: một tập chung)
```

Chi phí của việc bỏ `Q`:

```
Gap = E_Q[ A(X_{S*(Q)}) ] − E_Q[ A(X_{S*}) ]  ≥ 0
```

Gap luôn không âm. Câu hỏi là **lớn bao nhiêu** — và câu trả lời phụ thuộc hoàn toàn vào việc bạn đang **chọn frame** hay **phân bổ token**.

### 0.4 ⭐ Bất đối xứng cốt lõi — luận điểm chính của bài báo

**Chọn frame là quyết định recall CỨNG và KHÔNG THỂ ĐẢO NGƯỢC.**
Bỏ frame `t` → đóng góp của nó bằng 0 vĩnh viễn. Nếu bằng chứng nằm ở đó, không có gì cứu được. Và bạn **không thể biết** frame nào chứa bằng chứng nếu không biết câu hỏi. Nên `Gap` lớn.

**Phân bổ token là điều chỉnh MỀM về độ chi tiết.**
Với sàn `B_min > 0`, **mọi đoạn đều hiện diện**. Bạn chỉ quyết định đoạn nào được xem ở độ nét cao, đoạn nào chỉ thumbnail. Bằng chứng bị giảm chất lượng chứ không bị xóa. Nên `Gap` nhỏ.

**Ba dữ kiện độc lập cùng khớp:**

| Bằng chứng | Kết quả |
|---|---|
| QCA ablation: bỏ hạng tử content-deviation (query-agnostic) | VideoMME **tăng** 70.1 → 70.4 |
| KTV (chọn frame query-agnostic) trên LLaVA-1.6 | 42.48 < uniform-24 **43.30** |
| SCORE (nén token query-agnostic) trên Qwen2.5-VL | **65.3 > Vanilla 63.8** |

Cùng một triết lý query-agnostic: **thất bại ở tầng frame, SOTA ở tầng token.**

> **Câu văn cho bài báo:**
> *Độ dư thừa độc lập với câu hỏi; độ liên quan thì không. Chọn frame buộc phải phá hủy thông tin nên cần câu hỏi. Phân bổ token chỉ điều chỉnh độ chi tiết nên không cần.*

### 0.5 Bằng chứng từ literature: nhánh thành công đều query-agnostic

| Method | Thấy câu hỏi? | Qwen2.5-VL / VideoMME |
|---|---|---|
| FastVID | ✗ | 63.3 @ 25% |
| HoliTom | ✗ | 61.6 @ 40% |
| VidCom² | ✗ | 64.6 @ 25% |
| **SCORE** | ✗ (input chỉ là `[X; ΔX]`) | **65.3 @ 25%** |
| PruneVid | ✓ (query attention ở layer 10) | yếu nhất nhóm |

Policy network của SCORE **không bao giờ nhìn thấy text**. Nó nhận token embedding và residual liên frame, xuất xác suất giữ/bỏ. Và nó vượt cả model không nén. Phương pháp query-aware duy nhất trong nhóm lại là phương pháp kém nhất.

**Kết luận:** query-agnostic không phải hạn chế. Nó là paradigm chủ đạo của nhánh này. Ý tưởng ban đầu đúng — chỉ là trước đây áp dụng ở sai tầng (frame thay vì token).

---

<a name="phần-1"></a>
## PHẦN 1 — ĐẶT BÀI TOÁN

### 1.1 Setup

Video được sample thành `F` frame, mỗi frame mã hóa thành `N` token bởi vision encoder đóng băng:

```
X ∈ R^{F × N × D},    X[t] ∈ R^{N × D}
```

Chi phí prefill của LLM bị chi phối bởi self-attention:

```
O( (FN + |Q| + |Y|)² ) ≈ O( (FN)² )
```

vì `FN >> |Q|, |Y|`. Trên Qwen2.5-VL-7B, 128 frame: `FN = 42,566` token thị giác so với 42,657 token tổng — **text chỉ chiếm 0.2%**. (SCORE Table 3)

### 1.2 Mục tiêu ràng buộc

SCORE §3, Eq. (1):

```
min_{X_comp ⊆ X}  |X_comp|
        s.t.       A(X_comp) ≥ A(X) − δ
```

Giảm token tối đa với ràng buộc accuracy không tụt quá `δ`. Không giải trực tiếp được: không gian tập con là `2^{FN}` và `A(·)` không khả vi theo lựa chọn rời rạc.

### 1.3 ⭐ Phân rã hai tầng — bước đi cấu trúc quan trọng nhất

```
     {B_s}_{s=1..S},  Σ_s B_s ≤ B              T_s = Select(X_s, B_s)
   └────────────────────────────────┘      └──────────────────────────┘
   TẦNG 1: phân bổ ngân sách                TẦNG 2: chọn trong đoạn
   (ĐÓNG GÓP CỦA TA)                        (mượn FastVID)
```

**Luận điểm:** literature đã tối ưu Tầng 2 đến cạn kiệt và để Tầng 1 ở nghiệm tầm thường `B_s = ρ·|X_s|` (tỉ lệ thuận với độ dài đoạn). Chúng tôi chỉ ra Tầng 1 mới là chỗ còn dư địa.

---

<a name="phần-2"></a>
## PHẦN 2 — TẦNG 1: PHÂN ĐOẠN THỜI GIAN ĐỘNG

### 2.1 Tại sao phải phân đoạn

FastVID §3.2 đưa hai lý do:

**Toàn vẹn thời gian.** Xáo trộn thứ tự frame hoặc bỏ frame → hiểu sai nội dung. Phải prune *trong* đoạn liên tục, không bao giờ xuyên đoạn.

**Dư thừa trong đoạn.** Merge chỉ an toàn giữa các token rất giống nhau. Đoạn phải tối đa hóa độ tương đồng nội bộ.

**Ablation — FastVID Table 6** (LLaVA-OV, r=10%):

| Phân đoạn | MVBench | LongVideoBench | MLVU | VideoMME | Avg Acc |
|---|---|---|---|---|---|
| Fixed-interval | 55.1 | 53.6 | 61.7 | 55.4 | 95.3% |
| Cluster-based | 53.2 | 53.0 | 61.7 | 54.8 | 93.9% |
| **DySeg** | **55.9** | **56.3** | **62.7** | **57.3** | **98.0%** |

> ⚠️ **Clustering TỆ HƠN cả chia đều** (93.9 < 95.3). Lý do FastVID đưa ra: clustering bỏ qua thứ tự thời gian, gộp các frame ở những thời điểm khác nhau vào cùng một đoạn.
>
> Điều này vừa **xác nhận gợi ý #2 trong báo cáo** (chia clip ngắn 10–30s), vừa **cảnh báo đừng dùng clustering**.

### 2.2 Thuật toán DySeg

Đặc trưng frame toàn cục `f_i` (mean-pool token, hoặc DINOv2 CLS). Độ tương đồng transition:

```
t_i = cos(f_i, f_{i+1}),    i = 1, ..., F−1
T   = {t_1, ..., t_{F−1}}
```

Tập biên:

```
S₁ = argmin_{c−1} T            (c−1 transition thấp nhất)
S₂ = { i : t_i < τ }           (mọi transition dưới ngưỡng)
S  = S₁ ∪ S₂
```

- `S₁`: đảm bảo tối thiểu `c` đoạn ngay cả với video mượt, không có cắt cảnh.
- `S₂`: bắt cắt cảnh thật.

**Tự thích nghi:** video đơn giản → `S₂ = ∅`, `S₁` chia mịn các thay đổi tinh tế. Video phức tạp → `S₁ ⊂ S₂`, `S₂` chi phối.

**Siêu tham số:** `c = 8`, `τ = 0.9`. Ablation Fig. 6 của FastVID: ổn định với `1 ≤ c ≤ 16`, tụt ở `c = 32` (một frame/đoạn phá hủy quan hệ thời gian).

**Chi phí:** 0.5 ms (FastVID Table 3), so với 5.2 ms cho clustering của PruneVid.

### 2.3 Điều chỉnh cho setting hiện tại

Ở `F = 128` trên video 60 phút, một frame ≈ 28 giây. Mục tiêu 10–30s tương ứng 1–2 frame/đoạn — quá mịn. Hai lựa chọn:

- **(a)** Sample FPS cao hơn chỉ cho khâu phân đoạn (rẻ: chỉ forward DINOv2), rồi phân bổ frame cho đoạn ở Tầng 2.
- **(b)** Giữ `F = 128`, "đoạn" nghĩa là "khoảng liên tục cùng shot", thường 4–16 frame. Đặt `c` sao cho `F/c ≈ 8–16`.

> **Khuyến nghị (b)** cho bản đầu — trùng khớp setting FastVID nên bước reproduce kiểm chứng được trực tiếp.

### 2.4 Cache

`{S, f}` độc lập với câu hỏi → tính một lần/video. VideoMME có ~3 câu hỏi/video → **khấu hao 3×**.

---

<a name="phần-3"></a>
## PHẦN 3 — TẦNG 2: PHÂN BỔ NGÂN SÁCH KHÔNG ĐỒNG NHẤT (ĐÓNG GÓP CHÍNH)

### 3.1 Khoảng trống, phát biểu chính xác

FastVID §3.3 viết: *"Với một đoạn gồm P frame, chúng tôi giữ lại rPN token, trong đó N là số token mỗi frame và r là tỉ lệ giữ lại."*

`r` là **hằng số toàn cục**:

```
B_s^FastVID = r · P_s · N
```

> Một đoạn 16 frame quay người ngồi nói và một đoạn 16 frame hành động dày đặc nhận **ngân sách y hệt nhau**.

- **HoliTom, VisionZip, DyCoke**: đều vậy.
- **VidCom²**: điều chỉnh theo từng frame nhưng **không** phân bổ xuyên đoạn thời gian.
- **SCORE**: có phân bổ không đồng nhất nhưng chỉ *ngầm* qua global top-K học được — không phát biểu thành nguyên lý, không ablate.

> **Chưa ai đặt câu hỏi: bản thân tỉ lệ giữ lại có nên không đồng nhất theo thời gian không, và mức nào?**

### 3.2 Đo mật độ thông tin của đoạn

Cần `v_s` = mật độ thông tin của đoạn `s`, query-agnostic. Ba ước lượng.

#### 3.2.1 Ước lượng A — Residual tái cấu trúc (RECON, dùng lại đúng chỗ)

Với `F_s = [f_{s,1}, ..., f_{s,P_s}] ∈ R^{D × P_s}`, chạy greedy QR-pivot. Ở bước `k`:

```
R_k = F_s − F_{s,P_k} · F_{s,P_k}^† · F_s = (I − Π_{P_k}) F_s
```

trong đó `Π_{P_k}` là phép chiếu trực giao lên `span(F_{s,P_k})`.

Đường cong tái cấu trúc:

```
RC_s(k) = 1 − ‖R_k‖²_F / ‖F_s‖²_F

v_s^RC  = min{ k : RC_s(k) ≥ γ } / P_s
```

Diễn giải: đoạn tĩnh đạt `γ = 0.9` với 1 pivot (`v_s` nhỏ); đoạn động cần nhiều pivot (`v_s` lớn).

> ### ⭐ Tại sao lần này RECON hoạt động còn trước đây thì không
>
> RECON thất bại ở chọn frame vì `argmax_k ‖r_k‖` bị chi phối bởi **thay đổi ngoại hình toàn cục** — cắt cảnh, ánh sáng, camera pan — chứ không phải bằng chứng QA (thường nhỏ về không gian, thấp về năng lượng).
>
> Nhưng ở đây **ta không dùng residual để CHỌN gì cả**. Ta dùng *tốc độ suy giảm tổng hợp trên một khoảng* làm một con số vô hướng tóm tắt lượng nội dung độc lập trong khoảng đó. **Chế độ hỏng (chọn nhầm frame outlier) biến mất về mặt cấu trúc vì không có phép argmax nào được thực hiện.**
>
> *Đây là câu trả lời của bài báo cho câu hỏi "tại sao hướng reconstruction trước đây thất bại?"*

#### 3.2.2 Ước lượng B — Effective rank

```
erank(F_s) = exp( − Σ_i p_i log p_i ),    p_i = σ_i / Σ_j σ_j

v_s^er = erank(F_s) / P_s
```

với `σ_i` là các trị kỳ dị của `F_s`. (Roberts, Varshavsky & Wolf 2007 — entropy của phổ chuẩn hóa, một nới lỏng liên tục của hạng ma trận.)

> **Cứu lại phát hiện effective rank.** Trước đây đo erank ≈ 141 trên toàn video và hiểu là số sự kiện. Không phải — chuyển động camera liên tục cũng đẩy erank lên cao trong `R^768`.
>
> **Nhưng trong một đoạn ngắn cùng shot, erank là thước đo hợp lệ của độ đa dạng thị giác**, vì chuyển động camera bị chặn trong đoạn. Chia cho `P_s` và **so sánh tương đối giữa các đoạn cùng video** loại bỏ hoàn toàn vấn đề thang tuyệt đối.

#### 3.2.3 Ước lượng C — Năng lượng residual thời gian (surprise signal của SCORE)

SCORE §4.2, Eq. (2)–(3):

```
ΔX[t] = X[t] − X[t−1] ∈ R^{N × D},   X[0] = 0
H[t]  = [ X[t] ; ΔX[t] ] ∈ R^{N × 2D}
```

Tổng hợp về mức đoạn:

```
v_s^Δ = (1 / (P_s · N)) · Σ_{t ∈ s} Σ_{i=1..N} ‖ΔX[t,i]‖₂
```

**Lý do SCORE đưa ra (§4.2):** embedding các frame liền kề tương quan rất cao do liên tục thời gian; chỉ dùng `X[t]` khiến policy gán điểm gần như y hệt qua các frame và không tìm được vùng thay đổi. `ΔX` là **bộ lọc thông cao theo thời gian**.

**SCORE Table 5** (VideoMME, Qwen2.5-VL):

| | R | Short | Medium | Long | Overall |
|---|---|---|---|---|---|
| Vanilla | 100% | 74.4 | 64.2 | 52.8 | 63.8 |
| không residual | 40% | 74.4 | 65.3 | 54.3 | 64.7 |
| **có residual** | 40% | 75.0 | 65.0 | 55.1 | **65.0** |
| không residual | 25% | 74.8 | 64.9 | 54.8 | 64.5 |
| **có residual** | 25% | 74.8 | 65.4 | 55.7 | **65.3** |
| không residual | 10% | 71.4 | 62.0 | 52.6 | 62.0 |
| **có residual** | 10% | 73.6 | 63.1 | 53.9 | **63.6** |

> Chênh lệch **nới rộng khi ngân sách siết**: +0.3 @40%, +0.8 @25%, **+1.6 @10%**.
>
> Đây là kết quả chuyển giao được nhất trong survey — tín hiệu residual quan trọng nhất **đúng ở vùng ta muốn hoạt động**.

#### 3.2.4 Kết hợp khuyến nghị

```
v_s = λ · ṽ_s^RC + (1−λ) · ṽ_s^Δ       (ṽ = min-max chuẩn hóa TRONG video)
```

- `v^RC` bắt **đa dạng không gian/ngoại hình** (bao nhiêu "vẻ ngoài" khác nhau)
- `v^Δ` bắt **động lực thời gian** (bao nhiêu chuyển động)

Bổ trợ nhau: một đoạn có thể đa dạng nhưng tĩnh (pan chậm qua bức tranh tường) hoặc đồng nhất nhưng động (người khoa tay trước tường trơn).

> ⚠️ **Chuẩn hóa TRONG video, không bao giờ toàn cục.** Độ lớn residual tuyệt đối biến thiên rất mạnh giữa các video (độ phân giải, codec, nội dung), và ta chỉ cần so các đoạn của **cùng một video** với nhau.

### 3.3 Phân bổ với sàn phủ

```
┌───────────────────────────────────────────────────────────────┐
│                                    exp(ṽ_s / T)               │
│  B_s = B_min + (B − S·B_min) · ─────────────────────          │
│                                  Σ_{s'} exp(ṽ_{s'} / T)       │
└───────────────────────────────────────────────────────────────┘

ràng buộc:  Σ_s B_s = B   và   B_min ≤ B_s ≤ B_max
```

**Ba thành phần:**

| Thành phần | Vai trò |
|---|---|
| `B_min` | **Sàn phủ.** Mọi đoạn được đảm bảo mức tối thiểu. Đảm bảo `∀s: B_s > 0` — không đoạn nào bị xóa hẳn. |
| `T` | **Nhiệt độ.** `T → ∞` **khôi phục đúng phân bổ đều** (chính là baseline FastVID); `T → 0` dồn toàn bộ ngân sách tùy ý vào đoạn có `v` cao nhất. |
| `B_max` | Trần, chống một đoạn nuốt hết. |

> `T → ∞` khiến **phân bổ đều trở thành trường hợp đặc biệt** của phương pháp này — một dòng ablation cực sạch.

#### ⭐ Tại sao sàn phủ là bắt buộc — dữ liệu của chính chúng ta

Từ Bảng 2 (LLaVA-1.6-Vicuna-7B):

| Điều kiện | Overall | Short | Medium | Long |
|---|---|---|---|---|
| Uniform 12 frame | **43.15** | 52.78 | 41.56 | 35.11 |
| Oracle, K=30 → top-12 theo đáp án đúng | 42.81 | 52.78 | 40.44 | 35.22 |

> Một bộ chọn **có quyền truy cập đáp án đúng**, chọn 12 từ pool 30, lại **thua** `np.linspace`.
>
> Giải thích duy nhất mạch lạc: **dồn cục** — xếp hạng theo relevance tập trung các lựa chọn vào một cửa sổ thời gian hẹp, và mất độ phủ tốn hơn cái được từ relevance.
>
> **Vì ngay cả oracle cũng không thắng nổi uniform khi không có ràng buộc phủ, không bộ chọn khả thi nào làm được.** Đây là dữ liệu gốc của chúng ta và nó thuộc về hình minh họa động lực của bài báo.

**Phát biểu hình thức:** gọi `L_cov` là mất mát do bỏ sót bằng chứng hoàn toàn, `G_det` là cái được từ độ chi tiết thêm. Kết quả trên cho thấy `L_cov > G_det` khi phân bổ không ràng buộc. **Sàn chặn `L_cov` theo cấu trúc.**

**Giá trị gợi ý** (`B` = 15% dense):
- `B_min ≈ 18–36` token/frame-tương-đương
- `B_max ≈ 4×` trung bình
- `T ∈ [0.5, 2.0]`

### 3.4 Cổng entropy ở mức video

Nhiệt độ tối ưu không cố định giữa các video. Định nghĩa **độ tập trung thông tin**:

```
H_v = − Σ_{s=1..S} p_s log p_s,    p_s = exp(ṽ_s/T₀) / Σ_{s'} exp(ṽ_{s'}/T₀)

Ĥ_v = H_v / log S  ∈ [0, 1]
```

| `Ĥ_v` | Ý nghĩa | Hành động |
|---|---|---|
| → 1 | Thông tin trải đều (phim tài liệu liên tục, một cảnh dài) | Không có gì để khai thác → **làm phẳng về uniform** |
| → 0 | Thông tin tập trung ở vài đoạn (phần lớn tĩnh, thỉnh thoảng có hoạt động) | **Làm sắc phân bổ** |

```
T(Ĥ_v) = T₀ · exp( κ (Ĥ_v − H₀) )
```

Đây là **thích nghi bậc hai**: không chỉ "phân bổ không đồng nhất" mà "thích nghi *mức độ* không đồng nhất theo từng video". Query-agnostic (chỉ phụ thuộc `{v_s}`), chi phí một phép entropy, và cache cùng video.

> **Chưa ai có cái này.** FastVID: `r` cố định. HoliTom: cố định. VidCom²: theo frame, không theo cấu trúc video. SCORE: ngầm qua top-K học được, không bao giờ phơi bày hay ablate.

Nó còn sinh ra **dự đoán kiểm chứng được**: mức tăng phải tương quan với `1 − Ĥ_v` → biểu đồ scatter gain theo độ tập trung, một kết quả ở **mức cơ chế** chứ không chỉ một con số.

---

<a name="phần-4"></a>
## PHẦN 4 — TẦNG 3: CHỌN TOKEN TRONG ĐOẠN

> Mượn nguyên STPrune của FastVID. **Đừng phát minh lại** — nó đã được ablate kỹ và ta không được gì khi làm khác.

### 4.1 Hai loại token bổ trợ

```
B_s^DTM = d · B_s        (đại diện — representative)
B_s^ATS = (1−d) · B_s    (nổi bật — salient)
```

**FastVID Table 4** (LLaVA-OV, r=10%):

| d | MVBench | LongVideoBench | MLVU | VideoMME | Avg Acc |
|---|---|---|---|---|---|
| 0.0 (chỉ ATS) | 55.6 | 54.5 | 62.0 | 56.2 | 96.3% |
| **0.4** | **55.9** | **56.3** | **62.7** | **57.3** | **98.0%** |
| 1.0 (chỉ DTM) | 54.1 | 50.1 | 61.5 | 55.9 | 93.5% |

Kết hợp hơn ATS-đơn **1.8 điểm** và DTM-đơn **4.5 điểm**.

### 4.2 DTM — Density-based Token Merging

Density peaks clustering (Rodriguez & Laio, *Science* 2014). Với mỗi token `v_i` trong anchor frame:

```
ρ_i = exp( −(1/k) Σ_{v_j ∈ kNN(v_i)} d(v_i, v_j)² )         (mật độ cục bộ)

       ┌ min_{j: ρ_j > ρ_i} d(v_i, v_j)    nếu ∃j: ρ_j > ρ_i
δ_i = ─┤                                                     (độ tách biệt)
       └ max_j d(v_i, v_j)                 ngược lại
```

**Điểm = ρ_i × δ_i**: bao quanh bởi nhiều token tương tự (**đại diện**) VÀ xa các token mật độ cao khác (**phân biệt**). Token điểm cao nhất trở thành anchor.

**Gán:** mỗi token còn lại trong đoạn về anchor gần nhất theo cosine.

**Tổng hợp (FastVID Eq. 5):**

```
a* = α · a + ((1−α)/n) · Σ_{i=1..n} b_i,    α = 0.6
```

**Anchor frame:** chỉ lấy anchor từ frame cách nhau `p = 4`, **không** lấy từ toàn đoạn — vì chọn anchor xuyên toàn đoạn sẽ phá vỡ quan hệ không gian (anchor của các phần khác nhau của cùng vật thể rơi vào các frame khác nhau).

### 4.3 ⚠️ Ràng buộc positional — YÊU CẦU CỨNG

FastVID Fig. 4 & Table 5. Cluster-based merging nối các centroid, **vứt bỏ index vị trí gốc**. RoPE mã hóa vị trí tương đối; không có index đúng thì không biểu diễn được cấu trúc không-thời gian.

**FastVID Table 5:**

| Merging | MVBench | LongVideoBench | MLVU | VideoMME | Avg Acc |
|---|---|---|---|---|---|
| *LLaVA-OneVision* | | | | | |
| Uniform (VisionZip) | 55.0 | 54.7 | 62.0 | 56.4 | 96.2% |
| Cluster-based | 55.6 | 55.2 | 62.4 | 57.3 | 97.2% |
| **DTM** | **55.9** | **56.3** | **62.7** | **57.3** | **98.0%** |
| *LLaVA-Video* | | | | | |
| Uniform | 55.4 | 54.6 | 62.9 | 59.8 | 91.5% |
| Cluster-based | 55.3 | 55.1 | 62.9 | 60.1 | 91.8% |
| **DTM** | **58.5** | **56.5** | **64.9** | **60.7** | **94.7%** |

DTM hơn cluster-based **2.9 điểm** trên LLaVA-Video. SCORE §4.5 độc lập đi đến cùng yêu cầu: *giữ nguyên index vị trí gốc của token được chọn, bảo toàn thứ tự không-thời gian mà không thay đổi positional encoding.*

> **Hai paper, độc lập, cùng xác định bảo toàn vị trí là then chốt. Coi đây là ràng buộc cứng.**
> Trên Qwen2.5-VL còn quan trọng hơn vì mRoPE mã hóa **thời gian tuyệt đối**.

### 4.4 Thay thế ATS trên Qwen2.5-VL — bắt buộc, và là một delta chính đáng

ATS dùng `[CLS]` attention. FastVID §3.3 ghi nhận Video LLM dùng SigLIP lấy layer áp chót, bỏ qua head sinh `[CLS]` — nên họ **tích hợp lại một SigLIP head pretrain** (15.2M tham số).

**Qwen2.5-VL không có `[CLS]` token.** Vision encoder của nó dùng window attention với native dynamic resolution. Nên **phải** thay ATS.

Thay thế query-agnostic tự nhiên chính là residual thời gian, đã được SCORE kiểm chứng:

```
sal(t, i) = ‖ΔX[t,i]‖₂ = ‖X[t,i] − X[t−1,i]‖₂
```

Chọn top `(1−d)·B_s` token theo `sal`. **Diễn giải:** token có nội dung thay đổi so với frame trước mang thông tin mới; token tĩnh thì không.

> Đây là đóng góp thật, không phải giải pháp chữa cháy — ta chứng minh một tín hiệu salience **thuần thời gian, độc lập kiến trúc**, có thể thay attention proxy phụ thuộc kiến trúc. Nó tổng quát cho mọi backbone.
>
> Đưa vào ablation: **ATS ([CLS], chỉ LLaVA) vs. residual salience (phổ quát)**.

---

<a name="phần-5"></a>
## PHẦN 5 — THUẬT TOÁN ĐẦY ĐỦ

```
INPUT: video V, ngân sách token B
       c=8, τ=0.9, d=0.4, p=4, α=0.6, λ=0.5, T₀=1.0, κ, B_min, B_max

═══ GIAI ĐOẠN 0: MÃ HÓA (một lần/video, cache) ═══
 1  frames ← sample(V, F=128)
 2  X ← VisionEncoder(frames)              # [F,N,D], đóng băng
 3  f ← mean_pool(X, dim=token)            # [F,D]
 4  ΔX ← X[1:] − X[:−1];  ΔX[0] ← X[0]     # SCORE Eq.(2)

═══ TẦNG 1: PHÂN ĐOẠN (cache) ═══                          # FastVID §3.2
 5  t_i ← cos(f_i, f_{i+1})
 6  S₁ ← argmin_{c−1}(T);  S₂ ← {i: t_i<τ};  S ← S₁ ∪ S₂
 7  segments ← split(F, boundaries=S)

═══ TẦNG 2: PHÂN BỔ NGÂN SÁCH (cache) ═══                  # ĐÓNG GÓP CỦA TA
 8  for mỗi đoạn s:
 9      v_s^RC ← greedy_QR_pivot_curve(F_s, γ) / P_s
10      v_s^Δ  ← mean(‖ΔX[t,i]‖) với t ∈ s
11  ṽ ← minmax_normalize({ λ·v^RC + (1−λ)·v^Δ })            # TRONG video
12  Ĥ ← entropy(softmax(ṽ/T₀)) / log(S)
13  T ← T₀ · exp(κ(Ĥ − H₀))                                 # cổng entropy
14  B_s ← B_min + (B − S·B_min) · softmax(ṽ/T)_s
15  B_s ← clip(B_s, B_min, B_max); chuẩn hóa lại Σ B_s = B

═══ TẦNG 3: PRUNE TRONG ĐOẠN ═══                           # FastVID §3.3
16  for mỗi đoạn s:
17      anchor_frames ← mỗi frame thứ p trong s
18      ρ_i ← exp(−(1/k) Σ_{kNN} d(v_i,v_j)²)
19      δ_i ← min_{ρ_j>ρ_i} d(v_i,v_j)
20      anchors ← top-(d·B_s / ⌈P_s/p⌉) theo ρ_i·δ_i
21      gán non-anchor → anchor gần nhất (cosine)
22      a* ← α·a + ((1−α)/n)·Σ b_i
23      salient ← top-((1−d)·B_s) theo ‖ΔX[t,i]‖₂            # thay ATS
24      keep(s) ← anchors* ∪ salient
25      ⚠️ GIỮ NGUYÊN INDEX VỊ TRÍ GỐC

═══ OUTPUT ═══
26  X_comp ← ghép các đoạn, đúng thứ tự thời gian
27  answer ← FrozenLLM(X_comp, Q)          ← Q CHỈ XUẤT HIỆN Ở ĐÂY
```

| | |
|---|---|
| **Cache được** | Dòng 1–15 — toàn bộ **trước khi thấy `Q`** |
| **Mỗi câu hỏi** | **KHÔNG GÌ CẢ.** Tập token nén phục vụ mọi câu hỏi của video. |

---

<a name="phần-6"></a>
## PHẦN 6 — ĐỊNH VỊ VÀ NOVELTY

### 6.1 ⭐ Khoảng trống, bằng chính lời của SOTA

FastVID §C.1 (Limitations) thừa nhận: trên model lấy nhiều frame (LLaVA-Video 64, Qwen2-VL 768), accuracy tụt rõ so với model gốc, vì nhiều frame → ít token/frame → mỗi token quan trọng hơn → prune mạnh có nguy cơ bỏ thông tin then chốt. Và họ kết luận rằng **pruning query-agnostic của họ không làm nổi bật được thông tin liên quan câu hỏi vốn phân bố thưa thớt trong chuỗi frame dài**, đề xuất hướng tương lai là keyframe selection dẫn dắt bằng text.

> **Đọc kỹ — đây là cửa của chúng ta, nhưng KHÔNG theo hướng họ đề xuất.**
>
> Họ chẩn đoán đúng vấn đề (prune đồng nhất trên nhiều frame phá hủy thông tin thưa) nhưng đề xuất cách sửa query-aware.
>
> **Luận điểm của chúng ta: vấn đề giải được KHÔNG CẦN câu hỏi.** Lỗi nằm ở chỗ *ngân sách* đồng nhất, không phải ở chỗ *pruning* query-agnostic. Nếu đoạn nhiều thông tin nhận nhiều token hơn, thông tin thưa sẽ sống sót mà không cần nhìn câu hỏi.

Đây là một bất đồng trực tiếp, có cơ sở, với future work của một paper NeurIPS'25. Reviewer thích điều này.

### 6.2 Bảng đóng góp

| | Trạng thái |
|---|---|
| **C1.** Ngân sách token không đồng nhất xuyên đoạn, theo độ khó tái cấu trúc | **Mới.** Mọi công trình trước dùng `r` hằng số. |
| **C2.** Sàn phủ `B_min`, có động lực từ hiện tượng dồn cục đã đo được | **Mới**, hậu thuẫn bởi thí nghiệm oracle của chính chúng ta. |
| **C3.** Cổng entropy mức video điều chỉnh độ sắc của phân bổ | **Mới.** Không ai thích nghi độ quyết liệt theo cấu trúc video. |
| **C4.** Salience dựa residual thay `[CLS]` attention | **Mới**, độc lập backbone, cho phép chạy trên Qwen2.5-VL. |
| **C5.** Bất đối xứng dư thừa/liên quan → phân bổ token cache được, chọn frame thì không | **Framing mới**, hậu thuẫn bởi bằng chứng liên nhánh. |

### 6.3 Related Work — viết theo cách này

> **DyCoke** chia thành cửa sổ 4 frame cố định và giữ token của frame đầu — content-agnostic và bị chặn sàn ở 25% retention. **PruneVid** cluster token và dùng query attention bên trong LLM, nhưng clustering vứt bỏ thông tin vị trí và tốn latency cao. **FastVID** đóng góp phân đoạn động và density-based pruning trong đoạn, nhưng áp tỉ lệ giữ lại *hằng số* cho mọi đoạn. **HoliTom** phân đoạn theo dư thừa toàn cục và merge spatio-temporal, cũng ở tỉ lệ đồng nhất. **VidCom²** điều chỉnh cường độ nén theo tính độc đáo của từng frame nhưng không phân bổ xuyên đoạn thời gian. **SCORE** học chính sách giữ token bằng RL; global top-K của nó tạo ra phân bổ không đồng nhất một cách ngầm định, nhưng nguyên lý này không được phát biểu cũng không được ablate, và nó cần ~67K mẫu huấn luyện với re-annotation bằng teacher model.
>
> Chúng tôi trực giao với tất cả những công trình trên **ở tầng phân bổ**: chúng tôi giữ nguyên phân đoạn và chọn token trong đoạn của FastVID, và chỉ ra rằng làm cho bản thân ngân sách không đồng nhất — theo độ khó tái cấu trúc, với sàn phủ và độ sắc thích nghi theo video — mang lại thêm cải thiện, hoàn toàn training-free.

**Về QCA.** Cite nổi bật. QCA (arXiv 2607.00983) chung tác giả với KTV (Song, Peng, Zhou, Ji — Đại học Hạ Môn), nên họ nhiều khả năng sẽ review bài này. Định vị rõ: *QCA phân bổ **frame** cho đoạn dùng query matching; chúng tôi phân bổ **token** cho đoạn dùng độ khó tái cấu trúc, không cần query và cache theo video.*

### 6.4 Bảng kết quả mục tiêu

Qwen2.5-VL-7B, VideoMME w/o subtitles, 128 frame:

| Method | R | Query? | Train? | Overall |
|---|---|---|---|---|
| Vanilla | 100% | — | — | 63.8 |
| HoliTom | 40% | ✗ | ✗ | 61.6 |
| FastVID | 40% | ✗ | ✗ | 63.7 |
| VidCom² | 25% | ✗ | ✗ | 64.6 |
| SCORE | 25% | ✗ | **✓** | 65.3 |
| **Ours** | **15%** | ✗ | ✗ | **≥64.6** |

**Luận điểm bán:** *đạt ngang phương pháp training-free tốt nhất ở mức retention thấp hơn đáng kể, không cần huấn luyện.*

### 6.5 Bảng chi phí — luận điểm "con gà quả trứng"

| | Chi phí/video | Chi phí/câu hỏi |
|---|---|---|
| AKS / QCA | mã hóa frame | **BLIP-2 ITM trên mọi frame** |
| SCORE | mã hóa + policy | policy forward |
| **Ours** | mã hóa + phân đoạn + phân bổ | **KHÔNG** |

Với ~3 câu hỏi/video trên VideoMME → **khấu hao 3×** toàn bộ giai đoạn chọn lọc.

> Đây là phiên bản mạnh hơn của phê phán "con gà quả trứng": nó áp dụng cho cả **relevance scorer** (BLIP-2), không chỉ VLM.

---

<a name="phần-7"></a>
## PHẦN 7 — QUY TRÌNH THỰC NGHIỆM

### 7.1 Thống kê — đưa vào bài báo

VideoMME: `n = 2700`, 4 lựa chọn, chance 25%. Ở `p ≈ 0.60`:

```
SE = sqrt( p(1−p)/n ) = sqrt( 0.60 × 0.40 / 2700 ) ≈ 0.94 pp
```

Theo bucket (`n = 900`): `SE ≈ 1.63 pp`.

> **Hệ quả:** chênh lệch dưới ~1 pp **không đo được**. Toàn bộ Bảng 1 (39.96–43.70) và Bảng 2 (41.56–43.59) nằm trong một dải nhiễu — nói thẳng điều này khi giải thích vì sao bỏ nhánh LLaVA-1.6.

### 7.2 Thí nghiệm cổng — làm TRƯỚC mọi thứ

#### G1 — Sửa harness ⚠️ QUAN TRỌNG NHẤT

Trần dense hiện tại 59.41; số công bố Qwen2.5-VL-7B là 65.3, Vanilla của SCORE là 63.8. Nguyên nhân theo thứ tự khả năng:

| | Nguyên nhân |
|---|---|
| **(a)** | `min_pixels = max_pixels` ép độ phân giải đồng nhất, ghi đè native dynamic resolution. Đây là **giảm độ phân giải**, không phải **prune token** — nó làm mờ đều mọi thứ, phá hủy chữ, mặt người, vật nhỏ. Cả literature prune token nhưng **giữ vùng còn lại ở độ nét gốc**. |
| **(b)** | Truyền frame dạng list ảnh thay vì video có metadata FPS → mất căn chỉnh thời gian tuyệt đối của mRoPE. Qwen2.5-VL được train với dynamic FPS sampling và mRoPE căn chỉnh thời gian tuyệt đối. |
| **(c)** | Prompt template lệch. |
| **(d)** | Tỉ lệ câu trả lời không parse được — **in ra**; trên 1% là đang mất điểm do parsing chứ không phải do vision. |

> **CỔNG: uniform-128 native ≈ 63.8. Không qua thì không đi tiếp.**

#### G2 — Reproduce FastVID

Code công khai, LMMs-Eval, ở `r ∈ {25%, 15%, 10%}`.

> **CỔNG: lệch < 0.5 so với số công bố.** Không reproduce được baseline thì không con số nào công bố được.

#### G3 — Oracle query-agnostic (chưa từng chạy)

Với mỗi video, tìm tập token tối đa hóa accuracy **trung bình trên MỌI câu hỏi của video đó** — **một tập chung**.

Đây là **trần thật của paradigm query-agnostic**. Xấp xỉ bằng greedy trên tập ứng viên nhỏ, subsample 300 video.

| Kết quả | Kết luận |
|---|---|
| Gần dense (≈59–64) | Query-agnostic có đủ dư địa → luận điểm đứng vững |
| Thấp hơn nhiều | Một tập chung không phục vụ được mọi câu hỏi → biết trần trước khi xây |

### 7.3 Bảng ablation — mỗi dòng là một flag

| Variant | Phân đoạn | Ngân sách | Sàn | Cổng | Salience | VideoMME |
|---|---|---|---|---|---|---|
| Uniform sampling | — | — | — | — | — | |
| FastVID (repro) | DySeg | `r` đều | — | — | ATS | |
| + không đồng nhất (`v^Δ`) | DySeg | `v^Δ` | — | — | ATS | |
| + không đồng nhất (`v^RC`) | DySeg | `v^RC` | — | — | ATS | |
| + kết hợp `v` | DySeg | λ-mix | — | — | ATS | |
| + sàn phủ | DySeg | λ-mix | ✓ | — | ATS | |
| + cổng entropy | DySeg | λ-mix | ✓ | ✓ | ATS | |
| **Full** | DySeg | λ-mix | ✓ | ✓ | Δ-sal | |
| *(tham chiếu)* + query relevance | DySeg | +rel | ✓ | ✓ | Δ-sal | |

> **Về dòng cuối — NÊN CHẠY.** Nếu nó chỉ thêm +0.3–0.6, đó là **bằng chứng ỦNG HỘ** thiết kế: cho thấy kiến trúc cache-một-lần gần như không mất gì, và chặn trước câu hỏi hiển nhiên của reviewer. Đừng bỏ thí nghiệm này vì trung thành với framing query-agnostic.

**Phân tích thêm:**
- `T → ∞` khôi phục đúng uniform → chứng minh uniform là trường hợp đặc biệt của phương pháp.
- Scatter gain/video theo `1 − Ĥ_v` → kiểm chứng cơ chế cổng entropy.
- Đường cong accuracy–retention ở `R ∈ {40, 25, 15, 10, 5}%` → format chuẩn của nhánh này.

### 7.4 Benchmark

VideoMME (w/o sub), MLVU, LongVideoBench, MVBench — **đúng bộ FastVID và SCORE dùng**, để số liệu so sánh được. Báo cáo trung bình làm headline.

---

<a name="phần-8"></a>
## PHẦN 8 — SỔ TRÍCH DẪN

### Mượn — phải cite rõ

| Thành phần | Nguồn | Nội dung cụ thể |
|---|---|---|
| Phân đoạn rồi prune | FastVID §3.1–3.2 | Xáo/bỏ frame phá hủy hiểu biết |
| Công thức DySeg | FastVID Eq.(1)–(2) | `c=8`, `τ=0.9` |
| Ablation phân đoạn | FastVID Table 6 | DySeg 98.0% > fixed 95.3% > cluster 93.9% |
| DTM / density peaks | FastVID §3.3; Rodriguez & Laio, *Science* 2014 | `ρ_i × δ_i` |
| Bảo toàn vị trí | FastVID Table 5; SCORE §4.5 | DTM +2.9 so với cluster trên LLaVA-Video |
| Cân bằng DTM/ATS | FastVID Table 4 | `d=0.4` tối ưu; +1.8/+4.5 so với từng cái riêng |
| Surprise signal `ΔX` | SCORE §4.2, Eq.(2)–(3) | Lọc thông cao thời gian; phá đối xứng thời gian |
| Ablation residual | SCORE Table 5 | +1.6 @10% retention |
| Context rot | SCORE §1; Hong et al. 2025 | 65.3@25% > 63.8@100% |
| Mục tiêu ràng buộc | SCORE Eq.(1) | `min |X_comp|` s.t. `A ≥ A(X) − δ` |
| Khoảng trống được nêu | FastVID §C.1 | Prune đồng nhất hỏng với thông tin thưa |
| Effective rank | Roberts, Varshavsky & Wolf 2007 | Entropy phổ |
| Số baseline | SCORE Tables 1–3; FastVID Tables 1–3, 7 | — |

### Của chúng ta

- **C1** Ngân sách không đồng nhất xuyên đoạn theo độ khó tái cấu trúc
- **C2** Sàn phủ, có động lực từ hiện tượng dồn cục đo được
- **C3** Cổng entropy mức video
- **C4** Salience dựa residual thay `[CLS]` attention
- **C5** Lập luận bất đối xứng dư thừa/liên quan
- Phân tích oracle-headroom liên backbone (phần động lực)

---

<a name="phần-9"></a>
## PHẦN 9 — ABSTRACT

> Video LLM bị nghẽn bởi số lượng token thị giác, và các bộ nén training-free hiện tại giảm nó bằng cách áp một tỉ lệ giữ lại **đồng nhất** theo thời gian, dồn toàn bộ nỗ lực thiết kế vào việc chọn token nào trong hạn mức cố định đó. Chúng tôi lập luận rằng bản thân ngân sách nên không đồng nhất. Dựa trên nhận xét rằng độ dư thừa — khác với độ liên quan — độc lập với câu hỏi, chúng tôi đề xuất một bộ phân bổ hoàn toàn query-agnostic, tính một lần cho mỗi video và tái sử dụng cho mọi câu hỏi. Chúng tôi đo mật độ thông tin của từng đoạn thời gian bằng tốc độ suy giảm của residual tái cấu trúc greedy kết hợp năng lượng residual token liên frame, phân bổ ngân sách qua softmax có nhiệt độ trên các mật độ này với một sàn phủ cho mỗi đoạn, và điều chỉnh nhiệt độ theo entropy của phân bố mật độ để video có thông tin tập trung nhận phân bổ sắc hơn. Trên Qwen2.5-VL-7B, phương pháp của chúng tôi đạt ngang bộ nén training-free tốt nhất ở mức retention thấp hơn đáng kể, không cần huấn luyện và không cần truy cập câu hỏi. Chúng tôi còn chỉ ra rằng sàn phủ là cần thiết chứ không phải thận trọng: trong một nghiên cứu có kiểm soát, một bộ chọn oracle có quyền truy cập đáp án đúng lại đạt điểm **thấp hơn** lấy mẫu đều khi không bị ràng buộc, vì xếp hạng theo relevance gây dồn cục về thời gian.

---

<a name="phụ-lục-a"></a>
## PHỤ LỤC A — BỐI CẢNH: CÁC THỬ NGHIỆM ĐÃ THỰC HIỆN

### A.1 Baseline trước Phase 0 (LLaVA-1.6-Vicuna-7B)

| Method | VideoMME acc. |
|---|---|
| KTV (reproduction) | 42.48% |
| KTV (paper, khác setup) | 44.00% |
| Temporal-Chain (best) | 43.48% |
| Temporal-Chain (default) | 43.70% |
| Temporal-Chain + rerank | 42.59% |
| Temporal-Chain + F2C | 39.96% |

### A.2 Phase 0 trên LLaVA-1.6-Vicuna-7B

| Condition | Overall | Short | Medium | Long |
|---|---|---|---|---|
| Oracle, K=12 (rank theo GT answer) | 43.59% | 53.22 | 41.78 | 35.78 |
| Oracle, K=30→top-12 | 42.81% | 52.78 | 40.44 | 35.22 |
| Uniform 8 frames (234 tok/frame) | 41.56% | 50.78 | 40.00 | 33.89 |
| Uniform 12 frames (156 tok/frame) | 43.15% | 52.78 | 41.56 | 35.11 |
| Uniform 16 frames (117 tok/frame) | 41.89% | 51.78 | 39.22 | 34.67 |
| Uniform 24 frames (78 tok/frame) | 43.30% | 53.44 | 41.33 | 35.11 |
| Uniform 32 frames (58 tok/frame) | 42.48% | 51.56 | 42.00 | 33.89 |

**Kết luận:** oracle (43.59) hơn best uniform (43.30) chỉ **+0.29 pp**, trong khi SE = 0.95 pp. **Trên backbone này KHÔNG CÓ dư địa để đo bất kỳ selector nào.** Mọi phương pháp đều ra cùng một con số vì không có chỗ nào để thắng.

### A.3 Phase 0 trên Qwen2.5-VL-7B-Instruct

| Condition | Overall | Short | Medium | Long |
|---|---|---|---|---|
| Uniform 8 frames | 52.33% | 62.89 | 50.33 | 43.78 |
| Uniform 12 frames | 54.11% | 63.78 | 50.78 | 47.78 |
| Uniform 16 frames | 54.67% | 65.22 | 52.44 | 46.33 |
| Uniform 24 frames | 54.63% | 65.33 | 51.89 | 46.67 |
| Uniform 32 frames | 55.52% | 66.33 | 52.22 | 48.00 |
| Oracle, K=12 | 56.19% | 67.11 | 52.11 | 49.33 |

**Kết luận:** dư địa oracle = **+2.08 pp** (56.19 vs 54.11), lần này vượt ngưỡng nhiễu. Đổi backbone mang lại **+11 điểm** — lớn hơn ~25× toàn bộ không gian thuật toán đã thử.

### A.4 Token/frame budget scaling trên Qwen2.5-VL-7B

| num_frames | tokens | tok/frame | Overall | Short | Medium | Long |
|---|---|---|---|---|---|---|
| 32 | 1872 | 58 | 55.52% | 66.33 | 52.22 | 48.00 |
| 48 | 7200 | 150 | 58.07% | 69.33 | 55.56 | 49.33 |
| 64 | 9600 | 150 | 58.74% | 69.89 | 57.89 | 48.44 |
| 96 | 14400 | 150 | 59.41% | 69.44 | 58.67 | 50.11 |
| 128 | 16384 | 128 | 59.19% | 69.67 | 57.78 | 50.11 |

**Ba kết luận:**

1. **Từ 48 → 128 frame chỉ được +1.12 pp** (≈ 1 SE). Đường cong **bão hòa từ 48 frame trở đi**. Đây chính là lý do tồn tại của việc nén: trong 96 frame uniform có ~50–80 frame dư thừa.

2. **Phần lớn gain đến từ độ phân giải, không phải số frame.** 32@58 → 48@150 = +2.55 nhưng đổi hai biến cùng lúc; trục frame ở độ phân giải thấp chỉ đáng ~+3.2 tổng cộng.

3. **Bucket Long gần như không phản ứng với ngân sách.** Ở ~150 tok/frame, 12 → 96 frame: Short +5.66, Medium **+7.89**, Long chỉ **+2.33**. Cơ hội tập trung ở bucket **Medium**, ngược với giả định của mọi paper trong lĩnh vực.

### A.5 Khoảng cách với SOTA — chẩn đoán

Quy đổi về retention ratio (Qwen native ở 128 frame ≈ 42,566 token ≈ 333 tok/frame):

| Cấu hình hiện tại | Retention | Overall | SOTA cùng retention |
|---|---|---|---|
| 96 frame @ 150 | ~34% | 59.41 | VidCom² 64.6 / SCORE 65.3 (@25%) |
| 128 frame @ 128 | ~38% | 59.19 | FastVID 63.7 / SCORE 65.0 (@40%) |

> **Chẩn đoán:** thấp hơn ~4–5 điểm ở **cùng mức retention**. Nguyên nhân gần như chắc chắn: giảm token bằng cách **hạ độ phân giải đều toàn ảnh** (`min_pixels`) thay vì **bỏ token dư thừa giữ nét vùng còn lại**. Hạ độ phân giải phá hủy chữ, mặt người, vật nhỏ — đúng những thứ VideoMME hỏi.
>
> Đây là lý do SCORE ở 10% token vẫn được 63.6 còn ta ở 34% token chỉ được 59.41. **Sửa được trong vài ngày, giá trị cao nhất hiện tại.**

### A.6 Kết quả Gate G1 — KHÔNG đạt (2026-08-12 → 08-13)

Test trực tiếp giả thuyết (a)+(b) ở mục A.5: dựng script riêng (`run_inference_qwen25vl_native.py`), truyền frame dạng `{"type":"video","video":frames,"fps":sample_fps}` qua `qwen_vl_utils.process_vision_info` thay vì N `{"type":"image"}`, bỏ hẳn ép `min_pixels=max_pixels`.

**Xác minh bằng chẩn đoán trực tiếp (không suy diễn từ pixel math), 1 câu mẫu:**
```
input_keys có 'video_grid_thw' + 'second_per_grid_ts'  → đúng nhánh video, không phải nhánh ảnh
video_grid_thw = [64, 40, 72]   (T=64 khớp 128 frame / temporal_patch_size=2)
visual_tokens sau spatial merge (÷4) = 46,080  →  360 tok/frame
```
**360 tok/frame gần khớp 333 tok/frame của SCORE Vanilla** (chênh <8%) — xác nhận đã sửa đúng cả hai giả thuyết (a) độ phân giải bị ép và (b) thiếu metadata fps cho mRoPE.

**Nhưng full 2700 câu lại KHÔNG cải thiện:**

| Cấu hình | tok/frame | Overall | Short | Medium | Long |
|---|---|---|---|---|---|
| SCORE Vanilla (mục tiêu gate) | ~333 | **63.8** | 74.4 | 64.2 | 52.8 |
| Cũ — `f128_t16384` (ép min=max_pixels) | 128 | 59.19 | 69.67 | 57.78 | 50.11 |
| **Mới — native video, không ép pixel** | **~360** | **57.85** | 68.78 | 56.11 | 48.67 |

Mới vs Cũ: *z*≈1.0 (p≈0.32) — không có ý nghĩa thống kê, và về số tuyệt đối **thấp hơn**, dù dùng token nhiều hơn ~2.8×.

> **Kết luận: Gate G1 KHÔNG đạt** (cách 63.8% mục tiêu ~6 điểm). Giả thuyết (a)+(b) — dù xác nhận đã sửa đúng bằng chẩn đoán trực tiếp — **không giải thích được phần lớn khoảng cách với SCORE Vanilla**. Theo đúng nguyên tắc gate của tài liệu ("Không qua thì không đi tiếp"), **chưa nên sang Bước 1 (reproduce FastVID)**.
>
> **Nghi phạm còn lại, chưa kiểm tra:** (c) prompt template khác lmms-eval, (e) cách sample frame (linspace đều của `dataset.load_video` vs fps-cap sampling của lmms-eval/Qwen). (d) đã loại (unparsed 2/2700 = 0.07%, không đáng kể).

### A.7 Kết quả triển khai đầy đủ Tầng 1+2+3 (`run_inference_qwen25vl_dyseg.py`, 2026-08-13 → 08-14)

**Quyết định lệch khỏi lộ trình gốc:** thay vì đi tuần tự Bước 0(gate)→1(FastVID)→2(Tầng1+3)→3(Tầng2), triển khai **thẳng cả 3 tầng cùng lúc** (DySeg + phân bổ non-uniform + cổng entropy + STPrune) trên nền prompt `lmms-eval` đã sửa ở A.6, không chờ gate G1 đạt, không reproduce FastVID trước. Đánh đổi có ý thức: nhanh có con số để báo cáo, nhưng mất khả năng quy kết chính xác phần cải thiện đến từ tầng nào (không có ablation từng tầng theo đúng khuyến nghị §7.3).

**Xác minh bắt buộc trước khi tin số liệu** — `verify_mode=true` (giữ nguyên 100% token, chỉ test cơ chế dựng lại `inputs_embeds`/`position_ids` cho tập con không liên tục): **30/30 khớp tuyệt đối** với `run_inference_qwen25vl_native.py` cùng điều kiện (`temperature=0`) → xác nhận "ràng buộc cứng" bảo toàn vị trí (§4.3) được tuân thủ đúng.

**Kết quả full 2700 câu**, `tokens_num=16384` (mục tiêu, thực tế đạt ~76-95%/đoạn do dedup giữa DTM và salience — xem "Hạn chế" bên dưới):

| Điều kiện | Overall | Short | Medium | Long |
|---|---|---|---|---|
| Cũ — `f128_t16384` (uniform, ép độ phân giải) | 59.19% | 69.67 | 57.78 | 50.11 |
| Native dense, không cắt (A.6, prompt cũ) | 57.85% | 68.78 | 56.11 | 48.67 |
| **DySeg + QTA + STPrune (đầy đủ)** | **60.33%** | 68.89 | 59.78 | 52.33 |

**Kiểm định thống kê (so với `f128_t16384`, mốc yêu cầu):**
```
z ≈ 0.85 (p≈0.39) — vượt số tuyệt đối (+1.14 pts) nhưng CHƯA đạt ý nghĩa thống kê ở n=2700
```
Theo bucket (cùng so với `f128_t16384`): Short *z*≈−0.36 (nhích xuống, không đáng kể), Medium *z*≈0.86 (nhích lên), Long *z*≈0.94 (nhích lên nhiều nhất). Không bucket nào riêng lẻ có ý nghĩa thống kê, nhưng **hướng nhất quán cả 3 bucket** (Medium/Long tăng, Short đứng yên/giảm nhẹ) — khớp đúng dự đoán của chính triết lý phân bổ non-uniform: đoạn/video giàu thông tin (Medium, Long — effective rank cao hơn theo phát hiện mục 4) hưởng lợi nhiều hơn từ việc được cấp thêm ngân sách có chọn lọc, video đơn giản (Short) không cần thêm mà cũng không mất.

**Hạn chế cần ghi nhận trung thực:**
- **Gate G1 chưa từng đạt** (57.85% ở A.6, chưa re-verify dense/no-prune với prompt mới) — nên con số 60.33% này **không có mốc "dense chuẩn" đáng tin để tính % retention so sánh với SCORE/FastVID** (§6.4 kỳ vọng ≥64.6 ở 15% retention — 60.33% ở đây thấp hơn, và retention thật cũng không rõ ràng vì actual_total hụt mục tiêu).
- **Chưa ablate từng thành phần** (DySeg riêng, DTM riêng, salience riêng, cổng entropy riêng) — không biết phần nào trong 60.33% đến từ đâu.
- **actual_total hụt mục tiêu** (12507–15621 / 16384 quan sát được, ~76-95%) do vòng lặp salience bỏ qua token trùng với DTM mà không tìm ứng viên thay thế — cần sửa nếu muốn ngân sách chính xác.
- So với native-dense (A.6, nhưng khác prompt nên không sạch 100%): *z*≈1.85 (p≈0.064) — gợi ý nhiều hơn nhưng vẫn chưa chắc chắn.

**Đối chiếu với luận điểm cốt lõi của tài liệu (mục 0.4):** kết quả **phù hợp hướng** với luận điểm "phân bổ token (mềm) nên hoạt động tốt hơn chọn frame (cứng)" — đây là lần đầu tiên trong toàn bộ quá trình Phase-0 một cơ chế non-uniform vượt qua uniform ở cùng ngân sách (dù chưa đạt ý nghĩa thống kê chặt), khác hẳn mọi thử nghiệm chọn-frame trước đó (KTV oracle, CAS-QTA) đều hòa hoặc thua. Cần thêm dữ liệu (ablation, gate G1, benchmark khác) trước khi khẳng định chắc chắn.

---

<a name="phụ-lục-b"></a>
## PHỤ LỤC B — SURVEY NHÁNH TOKEN COMPRESSION

### B.1 Hai nhánh riêng biệt — đừng lẫn số

| | Nhánh A — Frame selection | Nhánh B — Token compression |
|---|---|---|
| Làm gì | N frame → K frame | K frame → nén token trong mỗi frame |
| Đại diện | AKS, BOLT, Q-Frame, MDP3, F2C, QCA, KTV | DyCoke, PruneVid, FastVID, HoliTom, VidCom², SCORE |
| Đo bằng | accuracy ở cùng số frame | accuracy ở cùng **retention ratio** |
| Baseline | uniform sampling | Vanilla (100% token) |

**Số của hai nhánh không so trực tiếp được.** Bài này nằm ở giữa (KTV = *Keyframes and key Tokens*).

### B.2 Bảng so sánh phương pháp

| Method | Phân đoạn video | Chọn token trong đoạn | Query-aware? | Cần train? |
|---|---|---|---|---|
| **DyCoke** (CVPR'25) | Cố định 4 frame/đoạn | Giữ toàn bộ frame đầu, prune còn lại | ✗ | ✗ |
| **PruneVid** (ACL'25) | Clustering | Static/dynamic + cluster + query attention layer 10 | Một phần | ✗ |
| **FastVID** (NeurIPS'25) | **DySeg** — động, theo cosine liền frame | **DTM** (density peak) + **ATS** (CLS attention) | ✗ | ✗ |
| **HoliTom** (NeurIPS'25) | Global redundancy-aware | Spatio-temporal merging, >90% giảm | ✗ | ✗ |
| **VidCom²** (EMNLP'25) | — | Điều chỉnh cường độ nén theo "frame uniqueness" | ✗ | ✗ |
| **SCORE** (2026) | — | MLP học xác suất giữ/bỏ từng token | Gián tiếp | ✓ RL |

### B.3 Điểm chung đã hội tụ

1. **Phân đoạn trước, prune trong đoạn.** FastVID ablation: Fixed 95.3% → Cluster 93.9% → DySeg 98.0%.
2. **Giữ nguyên positional information.** Cluster merging vứt vị trí → RoPE hỏng. DTM +2.9 trên LLaVA-Video.
3. **Hai loại token bổ trợ.** Chỉ ATS 96.3%, chỉ DTM 93.5%, kết hợp `d=0.4` → 98.0%.
4. **Nén token có thể TĂNG accuracy.** SCORE 65.3 @25% > Vanilla 63.8 (context rot).

### B.4 Số trên Qwen2.5-VL-7B, VideoMME w/o sub, 128 frame

| Method | Retention | Overall | Short | Medium | Long |
|---|---|---|---|---|---|
| **Vanilla** (42,566 token) | 100% | **63.8** | 74.4 | 64.2 | 52.8 |
| SCORE | 40% | **65.0** | 75.0 | 65.0 | 55.1 |
| DyCoke | 40% | 64.9 | 74.3 | 65.3 | 55.0 |
| VidCom² | 40% | 64.7 | 74.8 | 64.7 | 54.6 |
| FastVID | 40% | 63.7 | 73.9 | 62.8 | 54.4 |
| HoliTom | 40% | 61.6 | 73.1 | 59.9 | 51.8 |
| **SCORE** | **25%** | **65.3** | 74.8 | 65.4 | 55.7 |
| VidCom² | 25% | 64.6 | 73.3 | 65.2 | 55.3 |
| FastVID | 25% | 63.3 | 74.3 | 61.7 | 53.9 |
| SCORE | 10% | 63.6 | 73.6 | 63.1 | 53.9 |
| FastVID | 10% | 62.0 | 72.9 | 61.7 | 51.4 |

### B.5 Số trên LLaVA-Video-7B (tham chiếu)

| Nhánh B — token compression, 64 frame | Retention | Overall |
|---|---|---|
| Vanilla | 100% | 60.7 |
| SCORE / FastVID | 40% | 60.3 / 60.1 |
| SCORE / FastVID | 25% | 59.7 / 59.3 |
| SCORE / FastVID | 10% | 57.6 / 56.4 |

| Nhánh A — frame selection, 64 frame | Overall |
|---|---|
| Uniform | 64.4 |
| AKS | 65.3 |
| Q-Frame | 64.7 |
| QCA | 66.1 |

> ⚠️ **Cái bẫy:** cùng backbone LLaVA-Video-7B, cùng VideoMME, nhưng "Vanilla/uniform" là 60.7 ở paper này và 64.4 ở paper kia — **lệch 3.7 điểm** chỉ vì cấu hình eval khác nhau. **Không thể copy số baseline từ paper khác**; phải tự reproduce trong harness của mình.

### B.6 Sơ đồ khoảng trống

```
Nhánh A (AKS, QCA, KTV):  chọn frame query-aware  →  tiêu token ĐỀU
Nhánh B (FastVID, SCORE): input uniform sampling  →  nén token GIỎI
                          ↑                            ↑
                    thiếu query-awareness         thiếu frame selection

           CHƯA AI NỐI HAI ĐẦU — và ta không nối bằng query,
           mà bằng NGÂN SÁCH KHÔNG ĐỒNG NHẤT (query-agnostic)
```

---

<a name="phụ-lục-c"></a>
## PHỤ LỤC C — VỀ ĐỀ XUẤT V-JEPA / RL CỦA LEADER

### C.1 Đối chiếu kiến trúc

| | Leader đề xuất | TSPO (nhánh A) | SCORE (nhánh B) |
|---|---|---|---|
| Backbone đóng băng | DINOv2 / V-JEPA | CLIP-Large | vision encoder của VLM |
| Cái được huấn luyện | MLP / transformer mỏng | agent 0.4B | MLP nhẹ |
| Lấy mẫu | Top-K / Gumbel-Softmax | — | Bernoulli → top-K |
| **Action space** | **frame** | **frame** | **token** |
| **Reward** | **V-JEPA reconstruction loss** | **đúng/sai câu trả lời** | **CE câu trả lời + độ thưa** |
| Curriculum | không đề cập | — | pseudo → real |

**Bốn dòng đầu trùng gần như hoàn toàn.** Leader đã suy ra đúng kiến trúc mà hai paper đã hội tụ tới — trực giác tốt. Ba dòng cuối mới là khác biệt, và cả ba đều nghiêng về phía SCORE.

### C.2 Ba khác biệt then chốt

**(1) Token vs frame.** Frame selection có rủi ro recall tuyệt đối (bỏ nhầm frame chứa bằng chứng thì không cứu được). Token compression thì mọi frame đều hiện diện, chỉ khác độ chi tiết. Đây là lý do SCORE ở 10% token vẫn giữ 99.5% accuracy.

**(2) Reward — chỗ cần sửa.** Reconstruction loss là **query-agnostic** theo nghĩa xấu: nó đo độ đầy đủ vật lý, không đo tính đủ ngữ nghĩa. RECON chính là bản training-free của objective này và nó phẳng. Cả TSPO lẫn SCORE đều dùng tín hiệu từ **câu trả lời**. Ngoài ra reconstruction reward dễ bị hack (agent học "luôn chọn frame đầu/giữa/cuối") và cần load thêm V-JEPA (~300M) trong vòng lặp — làm hệ thống **nặng hơn**, phản lại chính động lực ban đầu.

**(3) Curriculum — chỗ thiếu.** Với 42,566 quyết định nhị phân mỗi video, policy khởi tạo ngẫu nhiên không học được gì. SCORE giải bằng pseudo-video warm-up (ghép ảnh tĩnh, `ΔX` gần như nhị phân) + split-advantage. Ablation: chỉ pseudo 55.7 → pseudo+real 57.0 @R=10%.

### C.3 Cái nên giữ từ leader

| | Giữ? |
|---|---|
| ✅ **Phê phán "con gà quả trứng"** | Giữ và **mở rộng** — nó đúng cả với BLIP-2 relevance scorer của AKS/QCA, không chỉ VLM. Pipeline này zero cost mỗi câu hỏi → xem Bảng 6.5. |
| ✅ **V-JEPA làm ENCODER (không phải critic)** | Một dòng ablation, 2 ngày. Features pretrain trên **video** encode dynamics tốt hơn DINOv2 (image-pretrained) cho việc tính `v_s^Δ`. |
| ⏸ **Policy head + RL** | Phase 2, có điều kiện. |
| ❌ **V-JEPA reconstruction reward** | Đã đo là phẳng qua RECON. |

**Ablation V-JEPA-as-encoder:**

| Feature space cho `v_s` | VideoMME |
|---|---|
| DINOv2 (image-pretrained) | ? |
| SigLIP / encoder của VLM | ? |
| **V-JEPA-2 encoder** (video-pretrained) | ? |

Nếu +0.5–1.0 → phát hiện sạch: *features pretrain trên video là ước lượng mật độ thông tin tốt hơn features pretrain trên ảnh.* Và V-JEPA có mặt trong bài báo, đúng vai trò nó giúp được.

### C.4 Điều kiện tiên quyết cho RL

RL mua được **0.7 điểm** so với training-free tốt nhất (SCORE 65.3 vs VidCom² 64.6), đổi lấy ~67K mẫu train + re-annotate bằng model 235B + K=16 rollout/video.

**Quy tắc quyết định** (sau khi chạy G3):

```
gap = oracle_query_agnostic − training_free_best

gap ≥ 3 pp  →  RL đáng làm, sang Phase 2
gap < 2 pp  →  RL không có gì để đóng, dừng
```

Nếu vẫn làm RL, action space nên **hai chiều**:

```
Policy → mỗi đoạn s: (giữ/bỏ, mức token ∈ {288,144,72,36,18})
Ràng buộc: Σ_s token_s ≤ B
Reward:    τ·CE_base − CE − λ·(vi phạm ngân sách)
```

Đây là mở rộng tự nhiên của SCORE (chỉ nhị phân) và TSPO (chỉ frame), và mượn nguyên `[X; ΔX]`, split-advantage, pseudo-video warm-up.

### C.5 Cách trình bày với leader

> "Kiến trúc anh đề xuất đúng — hai paper (TSPO, SCORE) đã xác nhận, cùng shape: frozen backbone + policy head nhỏ + Gumbel/Top-K + GRPO. Ba điều chỉnh em đề xuất dựa trên số liệu:
>
> 1. **Reward**: cả hai paper thành công đều dùng tín hiệu từ câu trả lời (CE / accuracy), không paper nào dùng reconstruction. Phase 0 của em cũng cho thấy tín hiệu reconstruction phẳng còn tín hiệu answer thì không.
> 2. **V-JEPA**: em muốn dùng nó làm *encoder* để tính mật độ thông tin — features pretrain trên video encode dynamics tốt hơn DINOv2 — chứ không làm critic. Một dòng ablation, hai ngày.
> 3. **Thời điểm**: RL mua được 0.7 điểm so với training-free tốt nhất, đổi lấy 67K mẫu train + re-annotate bằng model 235B. Em đề xuất xây bản training-free trước, đo khoảng cách tới oracle, rồi mới quyết định.
>
> Còn phê phán 'con gà quả trứng' của anh thì em giữ nguyên và mở rộng — nó đúng cả với BLIP-2 relevance scorer của AKS/QCA, không chỉ VLM. Pipeline của em zero cost mỗi câu hỏi, và đó là một claim bán được."

---

## LỘ TRÌNH

| Bước | Việc | Thời gian | Điều kiện qua | Trạng thái |
|---|---|---|---|---|
| **0** | Bỏ `min_pixels`, prune token sau encoder | 1 tuần | uniform-128 ≈ 63.8 | 🔶 Sửa video-native input xong (đúng, xem A.6) nhưng dừng ở 57.85% — **chưa qua gate, và bị bỏ qua có chủ đích** để đi tiếp (xem A.7). Chưa re-verify với prompt mới. |
| **1** | Reproduce FastVID (có code) trong harness | 1 tuần | lệch < 0.5 so với công bố | ⬜ Bỏ qua có chủ đích (chưa làm) |
| **2** | Tầng 1+3 (DySeg + STPrune, chưa có phân bổ) | 3 ngày | ≈ FastVID → xác nhận harness | 🔶 Đã code trong `run_inference_qwen25vl_dyseg.py` nhưng **gộp chung với Bước 3+4**, không tách riêng để xác nhận harness như khuyến nghị |
| **3** | Thêm Tầng 2 (phân bổ + sàn phủ) | 1 tuần | +1~2 so với bước 2 | ✅ Đã chạy full 2700 câu, **60.33%** (xem A.7) — nhưng gộp chung Bước 2+4, không cô lập được đóng góp riêng của Tầng 2 |
| **4** | Thêm cổng entropy | 3 ngày | gain tương quan với `1−Ĥ_v` | 🔶 Đã code (chạy chung với Bước 2+3), **chưa vẽ scatter gain vs `1-Ĥ_v` để kiểm chứng cơ chế** |
| **5** | Oracle query-agnostic (G3) | 2 ngày | quyết định có làm Phase 2 không | ⬜ Chưa bắt đầu |
| **6** | 4 benchmark × 3 retention | 1 tuần | — | ⬜ Chưa bắt đầu (mới có VideoMME, 1 mức retention) |

> **Bước 0 và 1 là bắt buộc và không thể bỏ qua.** Nếu không reproduce được FastVID, mọi con số sau đều không được reviewer tin. Đây là hai tuần không hào nhoáng nhưng bỏ qua chúng là cách phổ biến nhất khiến dự án dạng này bị desk-reject vì baseline không đáng tin.

---

## GHI CHÚ THIẾT KẾ

Thiết kế này cố ý **bảo thủ ở các thành phần đi mượn** và **quyết liệt ở đúng một chỗ** — tầng phân bổ ngân sách.

Đó là hồ sơ rủi ro đúng với thời điểm hiện tại: những phần dễ hỏng nhất (phân đoạn, chọn token trong đoạn, prune) đều lấy từ công trình đã công bố, đã ablate, có mã nguồn mở. Rủi ro tập trung vào đúng một tầng — tầng mang novelty.
