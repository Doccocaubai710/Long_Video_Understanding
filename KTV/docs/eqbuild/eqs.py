# Noi dung cong thuc (paper-style) cho HDS_MAP3D_trinh_bay.html. Moi block: list cac phan tu
#   ("p", text co $...$ inline) | ("eq", so, tex) | ("ul", [text,...])
NOTATION = [
 ("p", r"<b>Ký hiệu.</b> Gọi $\mathcal{H}\in\mathbb{R}^{T\times H_p\times W_p\times D}$ là lưới token sau ViT ❄ và merger 2×2 của Qwen2.5-VL, với $T=64$ unit thời gian, $H_p=12$, $W_p=21$, $D=3{.}584$. Tổng số token là $M=T\,H_p\,W_p=16{.}128$. Ở mức giữ $r$, ngân sách token bằng đúng Uniform:"),
 ("eq", 1, r"K=\lfloor r\,M\rfloor ."),
 ("p", r"Kích thước hộp $(s_t,s_y,s_x)$ được chọn nhỏ nhất sao cho số hộp không ít hơn $K$. Khi hoà nhau, ưu tiên hộp cân đối:"),
 ("eq", 2, r"(s_t,s_y,s_x)=\operatorname*{arg\,min}_{K_tK_yK_x\,\ge\,K}\Big(K_tK_yK_x,\;\max(s_t,s_y,s_x),\;|s_y-s_x|\Big),\qquad K_t=\Big\lceil \tfrac{T}{s_t}\Big\rceil,\;K_y=\Big\lceil \tfrac{H_p}{s_y}\Big\rceil,\;K_x=\Big\lceil \tfrac{W_p}{s_x}\Big\rceil ."),
 ("p", r"Lưới được chia thành các hộp không giao nhau:"),
 ("eq", 3, r"\mathcal{N}_{u,v,w}=\Big\{\,h_{t,y,x}\;\Big|\;t\in[u s_t,(u{+}1)s_t-1],\;y\in[v s_y,(v{+}1)s_y-1],\;x\in[w s_x,(w{+}1)s_x-1]\Big\}."),
 ("p", r"Số hộp thừa $K_tK_yK_x-K$ được bỏ đều (linspace trên thứ tự hộp). Ta đánh chỉ số phẳng các hộp là $k\in\{1,\dots,K\}$, các ứng viên trong hộp $\mathcal{N}_k$ là $h_{k,j}$ với $j\in\{1,\dots,S\}$, $S=s_ts_ys_x$, và toạ độ cục bộ $(\delta t_j,\delta y_j,\delta x_j)$. Tâm hộp là $c=(c_t,c_y,c_x)=\big(\lfloor s_t'/2\rfloor,\lfloor s_y'/2\rfloor,\lfloor s_x'/2\rfloor\big)$, trong đó $s'$ là kích thước thực của hộp (hộp ở biên lưới có thể nhỏ hơn)."),
]
STAGE1 = [
 ("p", r"<b>Bản đồ ngữ cảnh 3D</b> (tính một lần cho cả lưới, không theo hộp):"),
 ("eq", 4, r"\hat{\mathcal{H}}=\mathrm{LN}(\mathcal{H}),\qquad \mathcal{G}=\mathrm{GELU}\Big(\mathrm{DWConv3D}_{3\times3\times3}\big(\hat{\mathcal{H}}\,\mathbf{W}_{\downarrow}\big)\Big)\in\mathbb{R}^{T\times H_p\times W_p\times D'},\quad D'=\tfrac{D}{4}=896 ."),
 ("p", r"<b>Điểm của ứng viên</b> $j$ trong hộp $k$. Điểm gồm hai phần: phần nội dung đọc từ $\mathcal{G}$ tại đúng vị trí của ứng viên và độ khác so với tâm hộp, trừ đi khoản phạt lệch tâm có hệ số học được:"),
 ("eq", 5, r"\begin{aligned} l_{k,j}\;=\;&\underbrace{\mathbf{w}_{\text{zero}}^{\top}\,\mathrm{GELU}\Big(\mathbf{W}_1\big[\,\mathbf{g}_{k,j}\;\|\;(\hat h_{k,j}-\hat h_{k,c})\,\big]\Big)}_{\text{điểm nội dung } s_{k,j}}\\ &-\underbrace{\operatorname{softplus}(\lambda_t)\,\frac{(\delta t_j-c_t)^2}{s_t^{2}}\;-\;\operatorname{softplus}(\lambda_s)\,\frac{(\delta y_j-c_y)^2+(\delta x_j-c_x)^2}{s_y^{2}+s_x^{2}}}_{\text{phạt lệch tâm (center bias)}} \end{aligned}"),
 ("p", r"<b>Chọn cứng, học mềm (straight-through):</b>"),
 ("eq", 6, r"\pi_{k,j}=\frac{\exp\!\big(l_{k,j}/\tau\big)}{\sum_{j'=1}^{S}\exp\!\big(l_{k,j'}/\tau\big)},\qquad j_k^{*}=\arg\max_{j}\;l_{k,j},"),
 ("eq", 7, r"u_k^{*}=\sum_{j=1}^{S}\Big(\mathbb{1}[\,j=j_k^{*}\,]+\pi_{k,j}-\operatorname{sg}(\pi_{k,j})\Big)\,h_{k,j}\;\;\xrightarrow{\;\text{chiều xuôi}\;}\;\;h_{k,j_k^{*}},"),
 ("eq", 8, r"\tau(n)=\tau_1+\tfrac{1}{2}(\tau_0-\tau_1)\Big(1+\cos\frac{\pi n}{N}\Big),\qquad \tau_0=1,\;\tau_1=0{,}1 ."),
 ("ul", [
   r"<b>Bảo toàn Uniform tại bước 0:</b> $\mathbf{w}_{\text{zero}}$ khởi tạo bằng $\mathbf{0}$, nên $s_{k,j}=0$ và $l_{k,j}$ đạt cực đại duy nhất (bằng 0) tại tâm hộp. Bộ chọn bắt đầu đúng từ tâm, tức ≈ Uniform.",
   r"<b>Token gốc nguyên vẹn:</b> chiều xuôi của (7) là one-hot, nên $u_k^{*}$ đúng bằng một token gốc. Gradient đi qua $\pi_{k,j}$; $\operatorname{sg}(\cdot)$ là stop-gradient; $\tau$ giảm theo cosine qua $N=4{.}000$ bước.",
   r"<b>Vị trí thật:</b> token $u_k^{*}$ mang vị trí $p_k^{*}=(t_k^{*},y_k^{*},x_k^{*})$ trong lưới gốc. Các token được sắp theo $p_k^{*}$ trước khi vào LLM, nên mRoPE của Qwen giữ đúng.",
 ]),
]
STAGE2 = [
 ("p", r"<b>Deform adapter</b> (bản đồ ngữ cảnh riêng $\mathcal{G}'$, đọc tại vị trí neo $p_k^{*}$):"),
 ("eq", 9, r"\mathbf{g}'_k=\Big[\mathrm{GELU}\big(\mathrm{DWConv3D}_{3\times3\times3}(\mathrm{LN}(\mathcal{H})\,\mathbf{W}'_{\downarrow})\big)\Big]_{p_k^{*}},\qquad \big[\tilde o^{t},\tilde o^{y},\tilde o^{x},a^{t},a^{s}\big]^{\top}=\mathbf{W}_{\text{geo}}\,\mathbf{g}'_k,\qquad \Delta\mathbf{q}_k=\mathbf{W}_{dq}\,\mathbf{g}'_k,"),
 ("eq", 10, r"o^{t}_k=w\tanh\tilde o^{t},\quad o^{y}_k=3\tanh\tilde o^{y},\quad o^{x}_k=3\tanh\tilde o^{x},\qquad \sigma^{t}_k=\Big(\frac{M}{K}\Big)^{1/3}e^{\operatorname{clip}(a^{t},-2,2)},\quad \sigma^{s}_k=\Big(\frac{M}{K}\Big)^{1/3}e^{\operatorname{clip}(a^{s},-2,2)} ."),
 ("p", r"<b>Query, key/value.</b> Query lấy từ neo (đã chặn gradient). Key/value là mọi token trong cửa sổ $\pm w$ unit thời gian ($w=3$), gồm cả các token bị bỏ:"),
 ("eq", 11, r"\mathbf{q}_k=\mathbf{W}_q\,\mathrm{LN}\big(\operatorname{sg}(u_k^{*})\big)+\mathbf{E}_{\text{row}}(y_k^{*})+\mathbf{E}_{\text{col}}(x_k^{*})+\mathbf{E}_{\Delta t}(0)+\Delta\mathbf{q}_k,"),
 ("eq", 12, r"\mathcal{W}_k=\big\{(t',y',x'):\,|t'-t_k^{*}|\le w\big\},\qquad \mathbf{m}_{t',y',x'}=\mathbf{W}_{kv}\,\mathrm{LN}(h_{t',y',x'})+\mathbf{E}_{\text{row}}(y')+\mathbf{E}_{\text{col}}(x')+\mathbf{E}_{\Delta t}(t'-t_k^{*}) ."),
 ("p", r"<b>Bias khoảng cách 3D có deform và self-mask</b>, với $\Delta t=t'-t_k^{*}$, $\Delta y=y'-y_k^{*}$, $\Delta x=x'-x_k^{*}$:"),
 ("eq", 13, r"b_{k,(t',y',x')}=\begin{cases}-\infty, & (t',y',x')=p_k^{*}\quad\text{(self-mask)}\\[6pt] -\gamma\left[\left(\dfrac{\Delta t-o^{t}_k}{\sigma^{t}_k}\right)^{2}+\dfrac{(\Delta y-o^{y}_k)^2+(\Delta x-o^{x}_k)^2}{(\sigma^{s}_k)^{2}}\right], & \text{ngược lại.}\end{cases}"),
 ("p", r"<b>Hai lớp attention</b> ($\ell=1,2$; $d=1{.}024$, 8 head). $\mathbf{Q}_t$ là tập query của các neo cùng unit $t$:"),
 ("eq", 14, r"\begin{aligned} \mathbf{Q}_t&\leftarrow\mathbf{Q}_t+\mathrm{MHSA}\big(\mathrm{LN}(\mathbf{Q}_t)\big),\\ \mathbf{q}_k&\leftarrow\mathbf{q}_k+\operatorname{softmax}\!\Big(\frac{\mathrm{LN}(\mathbf{q}_k)\mathbf{W}^{Q}\big(\mathbf{M}_k\mathbf{W}^{K}\big)^{\top}}{\sqrt{d_h}}+\mathbf{b}_k\Big)\mathbf{M}_k\mathbf{W}^{V},\qquad \mathbf{M}_k=\{\mathbf{m}_{t',y',x'}\}_{(t',y',x')\in\mathcal{W}_k},\\ \mathbf{q}_k&\leftarrow\mathbf{q}_k+\mathrm{FFN}\big(\mathrm{LN}(\mathbf{q}_k)\big). \end{aligned}"),
 ("p", r"<b>Phần chỉnh và cổng theo mức nén:</b>"),
 ("eq", 15, r"\Delta_k=\mathbf{W}_{\text{out}}\,\mathrm{LN}(\mathbf{q}_k),\qquad z_k=u_k^{*}+(1-r)\,\alpha\,\Delta_k,\qquad \mathbf{W}_{\text{out}}\overset{\text{init}}{=}\mathbf{0},\;\;\alpha\overset{\text{init}}{=}1 ."),
 ("ul", [
   r"<b>Bảo toàn token gốc tại khởi tạo:</b> $\mathbf{W}_{\text{out}}=\mathbf{0}$ nên $z_k=u_k^{*}$. Các lớp $\mathbf{W}_{\text{geo}}$ và $\mathbf{W}_{dq}$ cũng khởi tạo bằng $\mathbf{0}$, nên ban đầu vùng nhìn đặt đúng tại neo ($o=0$, $\sigma=(M/K)^{1/3}$).",
   r"<b>Đầu ra</b> $\mathcal{Z}=\{z_k\}_{k=1}^{K}$ có đúng $K$ token, đặt tại các vị trí $p_k^{*}$ của GĐ1. Ví dụ $(M/K)^{1/3}=2{,}15$ ở $r=10\%$ và $3{,}68$ ở $r=2\%$.",
   r"Các bản 25% và 40% dùng $w=1$ vì bộ nhớ, các mức còn lại dùng $w=3$. $\gamma=1$.",
 ]),
]
LOSS = [
 ("p", r"Gọi $p_{\phi}$ là Qwen2.5-VL-7B đóng băng, $c=(c_1,\dots,c_{|\mathcal{C}|})$ là caption. <b>Thầy</b> xem toàn bộ $M$ token $\mathcal{H}$, <b>trò</b> xem $K$ token $\mathcal{Z}$ đặt ở vị trí thật. Hai bên cùng một mô hình, chỉ khác đầu vào hình ảnh:"),
 ("eq", 16, r"\mathcal{L}_{\text{KL}}=\frac{1}{|\mathcal{C}|}\sum_{i=1}^{|\mathcal{C}|}\mathrm{KL}\Big(p_{\phi}\big(\cdot\mid\mathcal{H},c_{<i}\big)\;\Big\|\;p_{\phi}\big(\cdot\mid\mathcal{Z},c_{<i}\big)\Big),"),
 ("eq", 17, r"\mathcal{C}^{+}=\Big\{\,i:\ \arg\max_{v}\,p_{\phi}(v\mid\mathcal{H},c_{<i})=c_i\Big\},\qquad \mathcal{L}_{\text{CEg}}=-\frac{1}{|\mathcal{C}^{+}|}\sum_{i\in\mathcal{C}^{+}}\log p_{\phi}\big(c_i\mid\mathcal{Z},c_{<i}\big),"),
 ("eq", 18, r"\mathcal{L}_{\text{DPP}}=\frac{1}{Q_{\max}}\cdot\frac{1}{T}\sum_{t=1}^{T}\max\Big(0,\;\log\det\big(\mathbf{G}(\mathcal{U}^{*}_t)+\epsilon\mathbf{I}\big)-\log\det\big(\mathbf{G}(\mathcal{Z}_t)+\epsilon\mathbf{I}\big)\Big),"),
 ("eq", 19, r"\mathcal{L}=\mathcal{L}_{\text{KL}}+\lambda_{\text{DPP}}\,\mathcal{L}_{\text{DPP}}+\lambda_{\text{CEg}}\,\mathcal{L}_{\text{CEg}},\qquad \lambda_{\text{DPP}}=0{,}05,\;\;\lambda_{\text{CEg}}\in\{0;\;0{,}2\}."),
 ("ul", [
   r"(17) <b>CE-gate:</b> chỉ học từ những token caption mà thầy đoán đúng top-1, để tránh bắt trò học theo các chỗ caption “khó” mà chính thầy cũng sai.",
   r"(18) <b>DPP dạng hinge:</b> $\mathbf{G}(\cdot)$ là ma trận Gram của các token đã chuẩn hoá L2 trong cùng unit $t$, $\mathcal{U}^{*}_t$ là các neo, $\mathcal{Z}_t$ là đầu ra. Loss chỉ phạt khi đầu ra kém đa dạng hơn chính các neo, $\epsilon=10^{-3}$.",
   r"Chỉ bộ nén (GĐ1 7,3M + GĐ2 48,9M tham số) được cập nhật. $p_{\phi}$ và ViT đóng băng. Mỗi bước tốn 1 lượt forward của thầy (không cần gradient) và 1 lượt của trò.",
 ]),
]
