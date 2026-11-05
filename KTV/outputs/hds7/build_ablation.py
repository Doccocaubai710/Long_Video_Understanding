"""Tao bang ablation de doc (bb_ablation). python build_ablation.py <state_in.json> <state_out.json>"""
import json, sys
S = json.load(open(sys.argv[1], encoding='utf-8'))
old = [b for b in S['benchmarks'] if b['id'] == 'bb_ablation']
link = old[0].get('paperLink', '') if old else ''
S['benchmarks'] = [b for b in S['benchmarks'] if b['id'] != 'bb_ablation']

COLS = [("c_exp","Thí nghiệm","text",280),("c_s1","GĐ1: chọn token","text",170),("c_s2","GĐ2: tinh chỉnh","text",150),
        ("c_ld","Loss · Data train","text",190),("c_ret","Mức giữ","text",64),("c_seed","Số seed","text",56),
        ("c_mlvu","MLVU dev","number",70),("c_vm","VideoMME","number",76),("c_lvb","LVBench","number",70),("c_avg","TB VM+LVB","number",80),
        ("c_d","Chênh vs ★ (MLVU / VM)","text",150),("c_st","Trạng thái","text",104),
        ("c_note","Nhận xét","text",300),("c_code","Mã run","text",110)]
NUMC = ("c_mlvu", "c_vm", "c_lvb", "c_avg")
# cot so de kieu text + num: code cu khong in dam toan bang; code moi in dam theo tung muc
cols = [dict({"id":i,"label":l,"type":"text","width":w}, **({"num": True} if i in NUMC else {})) for i,l,t,w in COLS]
cats, rows = [], []
DONE, RUN, TODO, CODE, QUEUE = "✅ Đã có", "⏳ Đang chạy", "📝 Cần chạy", "🛠 Cần code + chạy", "🕒 Trong hàng đợi"
N = [0]

# cau hinh mac dinh, viet tat de dien vao cot
S1 = "Chọn có học trong hộp 3D"; S1N = "Không (chọn đều)"
S2 = "Có (đầy đủ)"; S2N = "Không"
KL = "Chỉ KL · 2.569 video"; CSTAR = "KL + CE-gate 0.2 · 2.569 video"

def cat(name):
    cid = "acat_%d" % len(cats); cats.append({"id": cid, "name": name}); return cid
def f(v): return "%.2f" % v if isinstance(v, float) else v
def row(c, exp, s1, s2, ld, ret, seed, st, mlvu="", vm="", lvb="", note="", code="", ref=None, nomax=False, only=False):
    avg = "%.2f" % ((vm + lvb) / 2) if vm != "" and lvb != "" else ""
    d = ""
    if ref:
        parts = []
        parts.append("%+.2f" % (mlvu - ref[0]) if mlvu != "" else "–")
        parts.append("%+.2f" % (vm - ref[1]) if vm != "" else "–")
        d = " / ".join(parts) if (mlvu != "" or vm != "") else ""
    N[0] += 1
    cells = {"c_exp":exp,"c_s1":s1,"c_s2":s2,"c_ld":ld,"c_ret":ret,"c_seed":seed,"c_mlvu":mlvu,"c_vm":vm,"c_lvb":lvb,
             "c_avg":avg,"c_d":d,"c_st":st,"c_note":note,"c_code":code}
    r = {"id":"ab_%03d" % N[0],"categoryId":c,"cells":{k:{"v":f(v)} for k,v in cells.items()}}
    if ref: r["ref"] = list(ref)
    if nomax: r["noMax"] = True
    if only: r["only"] = ["c_mlvu"]   # chi cham MLVU dev
    rows.append(r)
def info(c, text):   # dong giai thich, gop ca hang
    N[0] += 1
    rows.append({"id":"ab_%03d" % N[0],"categoryId":c,"cells":{"c_exp":{"v":text,"colSpan":len(COLS)}}})

RKL = (64.43, 63.42)        # mac dinh chi KL, TB 3 seed
RCS = (65.46, 63.52)        # KL + CE-gate 0.2 (seed 0) -- tham chieu cho cac run cu dung CE-gate

c = cat("📘 CÁCH ĐỌC BẢNG")
info(c, "Phương pháp (HDS-MAP 3D) nén token hình ảnh của video trước khi đưa vào LLM, Qwen2.5-VL-7B giữ đóng băng. 128 frame → 16.128 token; giữ 10% ≈ 1.613 token.")
info(c, "GĐ1 — chọn token gốc: chia lưới token thành các hộp 3D (thời gian × cao × rộng), mỗi hộp giữ đúng 1 token gốc. Mạng nhỏ (7,3M tham số) chấm điểm từng token trong hộp, trừ một khoản phạt lệch tâm hộp, rồi chọn token điểm cao nhất (argmax xuôi, gradient qua softmax = straight-through).")
info(c, "GĐ2 — tinh chỉnh token: Transformer 2 lớp (49M tham số). Mỗi token đã chọn làm query, cross-attention vào các token gốc xung quanh (cửa sổ ±3 đơn vị thời gian, có deform + self-mask: không nhìn chính nó), rồi cộng một phần dư nhân với cổng (1 − mức giữ) vào token đã chọn.")
info(c, "Loss mặc định: chỉ KL — ép phân phối từ tiếp theo khi LLM xem token đã nén giống khi xem đủ 100% token (trên caption video). Data mặc định: 2.569 video LLaVA-Video-178K, 4000 bước, 1 GPU A100.")
info(c, "CẤU HÌNH MẶC ĐỊNH = GĐ1 có học + GĐ2 đầy đủ + chỉ KL + 2.569 video. Mỗi dòng bên dưới chỉ đổi đúng 1 yếu tố so với mặc định (hoặc so với dòng tham chiếu ghi trong nhóm).")
info(c, "Benchmark: MLVU dev (2.174 câu) — dùng DUY NHẤT để chọn cấu hình; VideoMME (2.700 câu) và LVBench (1.549 câu, video ~1 giờ) chỉ để báo cáo. Đơn vị: % trả lời đúng. Ô trống = chưa chấm bộ đó.")
info(c, "Nhiễu giữa các seed ở 10% khoảng ±1 điểm → chênh dưới 1 điểm coi như HÒA. Cột 'Chênh' = dòng đó trừ dòng tham chiếu của nhóm (đánh dấu ★), ghi MLVU dev / VideoMME.")
info(c, "Ô IN ĐẬM + TÔ VÀNG = kết quả cao nhất TRONG NHÓM đó, theo từng cột benchmark (dòng 'Không nén' là giới hạn trên, không tính).")

c = cat("A. MỖI GIAI ĐOẠN ĐÓNG GÓP BAO NHIÊU? (giữ 10%)")
row(c,"Không nén — LLM xem đủ 100% token (giới hạn trên, không tính khi so)","–","–","–","100%","–",DONE,68.08,65.15,44.48,code="dense",nomax=True)
row(c,"Chọn đều theo không gian–thời gian (Uniform), không train","Không (chọn đều)",S2N,"không train","10%","–",DONE,64.26,59.59,39.90,ref=RKL,code="uniform")
row(c,"Chọn ngẫu nhiên, không train","Ngẫu nhiên",S2N,"không train","10%","3",TODO,note="Chỉ cần chấm, không train (~3 GPU-h)")
row(c,"HoliTom — phương pháp không train tốt nhất trong các baseline","(HoliTom)","(HoliTom)","không train","10%","–",DONE,63.20,61.33,39.57,ref=RKL,code="holitom")
row(c,"Chỉ GĐ1 (chọn có học), bỏ GĐ2",S1,S2N,KL,"10%","1",DONE,"",61.59,39.64,ref=RKL,note="Chọn có học hơn chọn đều +2.0 VideoMME",code="H1")
row(c,"Chỉ GĐ2 (tinh chỉnh token chọn đều), bỏ GĐ1",S1N,S2,KL+" (+coverage 0.1)","10%","1",DONE,"",62.22,40.03,ref=RKL,note="GĐ2 một mình hơn chọn đều +2.6 VideoMME",code="H2")
row(c,"★ Cả 2 giai đoạn (MẶC ĐỊNH), trung bình 3 seed",S1,S2,KL,"10%","3",DONE,64.43,63.42,40.37,ref=RKL,note="Cả 2 GĐ > từng GĐ riêng: +1.8 VideoMME so với chỉ GĐ1, +1.2 so với chỉ GĐ2. Từng seed: VideoMME 62.52 / 63.70 / 64.04",code="H3, H3s1, H3s2")
row(c,"Chỉ GĐ1 — thêm 2 seed",S1,S2N,KL,"10%","+2",TODO,note="ƯU TIÊN: 2 dòng chỉ-GĐ1 / chỉ-GĐ2 mới có 1 seed, chưa chấm MLVU dev")
row(c,"Chỉ GĐ2 — thêm 2 seed",S1N,S2,KL,"10%","+2",TODO)

c = cat("B. GĐ1: CHỌN TOKEN TRONG HỘP BẰNG LUẬT CỐ ĐỊNH HAY BẰNG MẠNG CÓ HỌC? (cùng cách chia hộp, không GĐ2, giữ 10%)")
row(c,"Lấy token ở tâm hộp","Luật: tâm hộp",S2N,"không train","10%","–",DONE,"",57.30,"",note="Giữ 2%: VideoMME 52.26",code="P0_center")
row(c,"Lấy token đại diện nhất (medoid: giống các token khác trong hộp nhất)","Luật: medoid",S2N,"không train","10%","–",DONE,"",58.56,"",note="Giữ 2%: 50.30",code="P0_medoid")
row(c,"Lấy token có chuẩn (norm) lớn nhất","Luật: chuẩn lớn nhất",S2N,"không train","10%","–",DONE,"",56.11,"",note="Giữ 2%: 49.33",code="P0_norm")
row(c,"Lấy token thay đổi nhiều nhất so với lân cận (không gian + thời gian)","Luật: biến đổi lớn nhất",S2N,"không train","10%","–",DONE,"",57.15,"",note="Giữ 2%: 50.89",code="P0_var")
row(c,"★ Chọn bằng mạng có học (= Chỉ GĐ1 ở nhóm A)",S1,S2N,KL,"10%","1",DONE,"",61.59,39.64,note="Có học hơn luật tốt nhất (medoid) +3.0 VideoMME",code="H1")

R25 = (66.50, 64.56); R40 = (66.84, 64.59)   # chi KL, TB 3 seed
for lv, R, rows_b, base in (("25%", R25,
        [("Uniform (không train)", 65.96, 63.07, 41.70), ("HoliTom", 66.10, 63.85, 42.35), ("VidCom²", "", 63.96, ""), ("FastVID", "", 63.00, ""), ("DyCoke", "", 60.52, "")],
        ("K25, K25s1, K25s2", 66.50, 64.56, 41.92)),
    ("40%", R40,
        [("Uniform (không train)", 67.34, 64.07, 41.90), ("HoliTom", 66.74, 65.26, 44.42), ("VidCom²", "", 65.44, ""), ("FastVID", "", 65.11, ""), ("DyCoke", "", 64.37, "")],
        ("K40, K40s1, K40s2", 66.84, 64.59, 43.21))):
    c = cat("M%s. MỐC CẦN VƯỢT Ở MỨC GIỮ %s (các phương pháp khác, cùng Qwen2.5-VL-7B · 128 frame)" % (lv[:2], lv))
    row(c,"Không nén — 100% token (giới hạn trên, không tính khi so)","–","–","–","100%","–",DONE,68.08,65.15,44.48,code="dense",nomax=True)
    for nm, a, b, cc in rows_b:
        row(c, nm + " (không train)" if "Uniform" not in nm else nm, "(%s)" % nm.split(" ")[0], "", "không train", lv, "–", DONE, a, b, cc, ref=R)
    row(c,"★ Của mình hiện tại: chỉ KL, cấu hình mặc định",S1,S2,KL,lv,"3",DONE,base[1],base[2],base[3],ref=R,code=base[0],
        note="TB 3 seed")

for lv, R, box in (("25", R25, "1×2×2 (4 token)"), ("40", R40, "2×1×1 (2 token)")):
    c = cat("C%s. GĐ1: PHẠT LỆCH TÂM — DÒ LƯỚI λ Ở MỨC GIỮ %s%% (1 λ chung cho thời gian + không gian, cố định, không train; chỉ KL)" % (lv, lv))
    row(c,"★ λ = 0,5 (= mặc định), TB seed","λ = 0,5",S2,KL,lv+"%","3",DONE,R[0],R[1],(41.92 if lv=="25" else 43.21),ref=R,code="K%s…" % lv,
        note="Mặc định để λ thời gian / không gian train được nhưng sau train vẫn ≈ 0,5 như lúc khởi tạo (mọi mức giữ 2–40%) → thực chất là λ cố định 0,5, dùng làm mốc. Hộp ở mức này: " + box)
    for v, cd in ((0, "PN0"), (1.0, "PNf1"), (2.0, "PNf2")):
        row(c,"λ = %g%s" % (v, " (không phạt)" if v == 0 else ""),"λ = %g" % v,S2,KL,lv+"%","1",QUEUE,ref=R,code=cd+"x"+lv)
    info(c,"Dò mịn: khi có MLVU dev của 4 mốc trên, tự chọn mốc tốt nhất và thêm 1–2 mốc quanh nó (tốt nhất 0 → thêm 0,25 · 0,5 → 0,25 & 0,75 · 1 → 0,75 & 1,5 · 2 → 1,5 & 3); các dòng đó tự hiện bên dưới. VideoMME / LVBench chấm sau cùng.")

c = cat("D25. GĐ1: CÁCH CHIA HỘP Ở MỨC GIỮ 25% (chỉ KL) — mặc định hộp 1×2×2 = 4 token, chọn 1")
row(c,"★ Mặc định: hộp 1×2×2, chọn 1 token / hộp",S1,S2,KL,"25%","3",DONE,R25[0],R25[1],41.92,ref=R25,code="K25…")
row(c,"Hộp to gấp 2 (2×2×2 = 8 token), chọn 2 token / hộp","Hộp 2×2×2, chọn 2",S2,KL,"25%","1",QUEUE,ref=R25,code="BK2x25",note="Cùng số token, mỗi lựa chọn có nhiều ứng viên hơn")
row(c,"Hộp to gấp 4 (4×2×2 = 16 token), chọn 4 token / hộp","Hộp 4×2×2, chọn 4",S2,KL,"25%","1",QUEUE,ref=R25,code="BK4x25")
row(c,"Hộp mặc định, cho chọn vượt biên 1 token mỗi phía (hộp chồng lấn; trùng token → hộp điểm cao giữ)","Hộp 1×2×2, biên +1",S2,KL,"25%","1",QUEUE,ref=R25,code="BM1x25")
c = cat("D40. GĐ1: CÁCH CHIA HỘP Ở MỨC GIỮ 40% (chỉ KL) — mặc định hộp 2×1×1 = 2 token, chọn 1, và phải BỎ TRẮNG 20% số hộp")
row(c,"★ Mặc định: hộp 2×1×1, chọn 1 token / hộp (bỏ trắng 20% số hộp)",S1,S2,KL,"40%","3",DONE,R40[0],R40[1],43.21,ref=R40,code="K40…",
    note="Lưới chỉ chia được 8.064 hộp cho 6.451 token → bỏ đều 1.613 hộp: có vùng không giữ token nào. Có thể là lý do thua baseline ở 40%")
row(c,"Hộp 5×1×1 (5 token), chọn 2 token / hộp","Hộp 5×1×1, chọn 2",S2,KL,"40%","1",QUEUE,ref=R40,code="BK2x40",note="Chỉ bỏ 2% số hộp")
row(c,"Hộp 5×1×2 (10 token), chọn 4 token / hộp","Hộp 5×1×2, chọn 4",S2,KL,"40%","1",QUEUE,ref=R40,code="BK4x40",note="Chỉ bỏ 2% số hộp")
row(c,"Hộp mặc định, cho chọn vượt biên 1 token mỗi phía","Hộp 2×1×1, biên +1",S2,KL,"40%","1",QUEUE,ref=R40,code="BM1x40",note="Vẫn bỏ trắng 20% số hộp")

c = cat("E. GĐ1: CÁCH TRAIN BƯỚC CHỌN (rời rạc) — straight-through hay softmax giảm nhiệt độ?")
row(c,"★ Mặc định: xuôi argmax, gradient qua softmax (straight-through), nhiệt độ 1 → 0,1","Straight-through",S2,KL,"10%","3",DONE,64.43,63.42,40.37,ref=RKL,code="H3…")
row(c,"Xuôi softmax (trung bình có trọng số), nhiệt độ giảm 1 → 0,05; 15% bước cuối + lúc test dùng argmax","Softmax giảm nhiệt",S2,KL,"10%","1",DONE,64.72,63.63,39.90,ref=RKL,note="Hòa",code="SM10")
row(c,"Softmax cả lúc test (gộp token có trọng số, nhiệt độ 0,3)","Softmax cả khi test",S2,KL,"10%","1",DONE,65.27,63.22,40.80,ref=RKL,note="Hòa (nhỉnh MLVU dev, chưa đủ 1 điểm)",code="SMg10")

c = cat("F. GĐ2: TỪNG THÀNH PHẦN CÓ CẦN KHÔNG? (bỏ từng thành phần, giữ 10%)")
row(c,"★ GĐ2 đầy đủ (MẶC ĐỊNH, seed 0)",S1,S2,KL,"10%","1",DONE,64.31,62.52,39.44,ref=(64.31,62.52),code="H3")
row(c,"Bỏ deform (vùng nhìn cố định, không tự dịch / co giãn)",S1,"Bỏ deform",KL,"10%","1",DONE,"",63.22,"",ref=(64.31,62.52),code="H3noDF")
row(c,"Bỏ self-mask (token được nhìn chính nó)",S1,"Bỏ self-mask",KL,"10%","1",DONE,"",62.63,"",ref=(64.31,62.52),code="H3noSM")
row(c,"Bỏ cổng (1 − mức giữ) nhân vào phần dư",S1,"Bỏ cổng",KL,"10%","1",DONE,65.36,63.30,39.12,ref=(64.31,62.52),code="H3noGate")
row(c,"Chấm bổ sung MLVU dev + LVBench cho 2 dòng bỏ deform / bỏ self-mask","","","","10%","",TODO,note="Chỉ chấm (~6 GPU-h); hiện mọi dòng nhóm này 1 seed, chênh < 1 điểm = hòa")

c = cat("G1. GĐ2: MỞ RỘNG VÙNG ATTENTION CÓ TỐT HƠN KHÔNG? (chỉ KL; cửa sổ thời gian ± đơn vị, bán kính không gian)")
row(c,"★ Mặc định: cửa sổ ±3 đơn vị thời gian",S1,"Cửa sổ ±3",KL,"10%","3",DONE,64.43,63.42,40.37,ref=RKL,code="H3…")
row(c,"Cửa sổ ±1",S1,"Cửa sổ ±1",KL,"10%","1",TODO)
row(c,"Cửa sổ ±5",S1,"Cửa sổ ±5",KL,"10%","1",TODO)
row(c,"Cửa sổ ±8 (gần cả đoạn)",S1,"Cửa sổ ±8",KL,"10%","1",TODO,note="Có thể tràn bộ nhớ 40GB")
row(c,"Vùng nhìn không gian rộng gấp đôi",S1,"Bán kính ×2",KL,"10%","1",TODO)
row(c,"Bỏ giới hạn không gian (nhìn đều cả cửa sổ)",S1,"Không giới hạn KG",KL,"10%","1",TODO)
c = cat("G2. (ĐÃ CHẠY, với loss KL + CE-gate 0,2) CỬA SỔ ATTENTION ±1 / ±3 / ±5")
row(c,"★ Cửa sổ ±3 (loss KL + CE-gate 0,2)",S1,"Cửa sổ ±3",CSTAR,"10%","1",DONE,65.46,63.52,40.15,ref=RCS,code="H3ce")
row(c,"Cửa sổ ±1",S1,"Cửa sổ ±1",CSTAR,"10%","1",DONE,63.85,63.81,40.03,ref=RCS,note="Hẹp hơn: MLVU dev −1,6",code="W1")
row(c,"Cửa sổ ±5",S1,"Cửa sổ ±5",CSTAR,"10%","1",DONE,65.18,63.41,40.93,ref=RCS,note="Rộng hơn: hòa",code="W2")

c = cat("H. HÀM LOSS: XUẤT PHÁT TỪ KL, THÊM DPP / CE CÓ LỢI KHÔNG? (giữ 10%)")
row(c,"★ Mặc định: chỉ KL (kèm DPP hinge 0,05 gần như không hoạt động)",S1,S2,"Chỉ KL","10%","3",DONE,64.43,63.42,40.37,ref=RKL,code="H3…",note="Mặc định có sẵn DPP dạng hinge trọng số 0,05, nhưng nó = 0 ở ~95–100% số bước (ràng buộc luôn tự thỏa) → tăng trọng số sẽ không đổi gì. Muốn biết DPP vô ích do trọng số hay do bản chất cần thêm DPP dạng đầy đủ (2 dòng bên dưới)")
row(c,"Bỏ hẳn DPP",S1,S2,"KL, DPP = 0","10%","1",TODO,note="Kỳ vọng ≈ mặc định")
row(c,"DPP hinge trọng số 0,5 (×10)",S1,S2,"KL + DPP hinge 0,5","10%","1",TODO,note="Kèm thống kê tỉ lệ bước DPP > 0")
row(c,"DPP đầy đủ (log-det, không hinge) trọng số 0,05",S1,S2,"KL + DPP đầy đủ 0,05","10%","1",CODE)
row(c,"DPP đầy đủ trọng số 0,5",S1,S2,"KL + DPP đầy đủ 0,5","10%","1",CODE)
row(c,"Thêm CE caption trọng số 0,2",S1,S2,"KL + CE 0,2","10%","1",DONE,65.04,62.89,39.38,ref=RKL,code="L3")
row(c,"Thêm CE caption trọng số 1,0",S1,S2,"KL + CE 1,0","10%","1",TODO,note="Cùng dòng 0,2 và 'chỉ CE' cho xu hướng theo trọng số CE")
row(c,"Chỉ CE caption, bỏ KL",S1,S2,"Chỉ CE","10%","1",DONE,66.70,60.96,40.54,ref=RKL,note="MLVU dev cao nhưng VideoMME −2,5 → CE học kiểu viết caption, không chuyển sang hỏi-đáp",code="CEo")
row(c,"Thêm CE-gate 0,2 (CE chỉ trên từ mà bản không nén đoán đúng), TB 2 seed",S1,S2,"KL + CE-gate 0,2","10%","2",DONE,65.25,63.30,39.41,ref=RKL,note="MLVU dev +0,8 nhưng VideoMME / LVBench không tăng",code="H3ce, L4")
row(c,"Thêm CE-gate 0,5",S1,S2,"KL + CE-gate 0,5","10%","1",DONE,65.32,63.41,39.90,ref=RKL,code="L1")
row(c,"Thêm CE-gate 1,0",S1,S2,"KL + CE-gate 1,0","10%","1",DONE,65.46,62.85,40.03,ref=RKL,code="L2")
row(c,"KL trên mẫu hỏi-đáp (50% mẫu train là câu hỏi trắc nghiệm)",S1,S2,"KL + 50% QA","10%","1",DONE,63.39,63.11,39.77,ref=RKL,note="Kém hơn",code="Q1")

c = cat("I1. DỮ LIỆU TRAIN: CẦN BAO NHIÊU VIDEO? (chỉ KL)")
row(c,"★ Chỉ KL, 2.569 video (MẶC ĐỊNH)",S1,S2,"Chỉ KL · 2.569 video","10%","3",DONE,64.43,63.42,40.37,ref=RKL,code="H3…")
row(c,"Chỉ KL, 20 video",S1,S2,"Chỉ KL · 20 video","10%","1",DONE,64.72,61.85,39.83,ref=RKL,note="MLVU dev hòa nhưng VideoMME −1,6",code="D20")
row(c,"Chỉ KL, 80 video",S1,S2,"Chỉ KL · 80 video","10%","1",DONE,64.21,62.44,39.70,ref=RKL,code="D80")
c = cat("I2. DỮ LIỆU TRAIN: LƯỢNG & LOẠI VIDEO (đã chạy với loss KL + CE-gate 0,2)")
row(c,"★ 2.569 video (1×)",S1,S2,"KL + CE-gate · 2.569","10%","1",DONE,65.46,63.52,40.15,ref=RCS,code="H3ce")
row(c,"1/4 data — 640 video",S1,S2,"KL + CE-gate · 640","10%","1",DONE,65.55,62.56,40.74,ref=RCS,code="Dq")
row(c,"1/2 data — 1.280 video",S1,S2,"KL + CE-gate · 1.280","10%","1",DONE,65.41,63.93,39.83,ref=RCS,code="Dh")
row(c,"1,7× data — 4.342 video",S1,S2,"KL + CE-gate · 4.342","10%","1",DONE,64.12,63.67,40.41,ref=RCS,code="D2x")
row(c,"1,7× data, train gấp đôi (8.000 bước)",S1,S2,"KL + CE-gate · 4.342 · 8k bước","10%","1",DONE,65.27,62.78,39.77,ref=RCS,code="D2x8k")
row(c,"Thêm 50% video dài (Vript)",S1,S2,"KL + CE-gate · + Vript","10%","1",DONE,64.44,62.85,38.93,ref=RCS,note="Không giúp",code="H3cv")
c = cat("I3. DỮ LIỆU TRAIN: THÊM VIDEO DÀI GHÉP TỪ NHIỀU CLIP (chỉ KL, so với mặc định seed 0)")
row(c,"★ Mặc định, seed 0 (clip đơn)",S1,S2,"Chỉ KL · clip đơn","10%","1",DONE,64.31,62.52,39.44,ref=(64.31,62.52),code="H3")
row(c,"Ghép 3 clip thành video dài",S1,S2,"Chỉ KL · ghép 3 clip","10%","1",DONE,"",63.33,39.51,ref=(64.31,62.52),note="VideoMME +0,8: hòa",code="H4")
row(c,"Ghép 6 clip thành video dài",S1,S2,"Chỉ KL · ghép 6 clip","10%","1",DONE,"",62.93,39.32,ref=(64.31,62.52),note="Hòa",code="H12")

c = cat("J. PHÂN BỔ TOKEN THÍCH ỨNG (mỗi hộp không nhất thiết giữ đúng 1 token)")
row(c,"★ Mặc định: mọi hộp giữ đúng 1 token, TB 3 seed",S1,S2,KL,"10%","3",DONE,64.43,63.42,40.37,ref=RKL,code="H3…")
row(c,"Phân bổ token thích ứng: học chọn hộp nào được giữ (thay vì mọi hộp giữ 1)","Chọn hộp có học",S2,KL,"10%","1",DONE,62.74,62.96,40.67,ref=RKL,note="Kém hơn",code="BX10")
row(c,"Phân bổ token thích ứng: ngân sách token theo từng đoạn thời gian","Ngân sách theo đoạn",S2,KL,"10%","1",DONE,62.60,62.00,38.93,ref=RKL,note="Kém hơn",code="SG10")
c = cat("K. MỘT MODEL CHO MỌI MỨC GIỮ (so với model riêng từng mức, seed 0)")
row(c,"★ Model riêng cho mức 10% (mặc định, seed 0)",S1,S2,"Chỉ KL","10%","1",DONE,64.31,62.52,39.44,ref=(64.31,62.52),code="H3")
row(c,"Một model train lẫn 2/5/10/25%, dùng ở 10%",S1,S2,"Chỉ KL · nhiều mức","10%","1",DONE,"",62.85,39.70,ref=(64.31,62.52),note="Hòa. Ở 2%: VM 58.15 (model riêng 58.74) · 5%: 61.63 · 25%: 64.04",code="H5")

# to vang o cao nhat moi nhom (de doc ngay ca tren code cu); chi khi cot co >= 2 gia tri trong nhom
for cdef in cats:
    rs = [r for r in rows if r["categoryId"] == cdef["id"] and not r.get("noMax") and "c_mlvu" in r["cells"]]
    for col in NUMC:
        vals = [(float(r["cells"][col]["v"]), r) for r in rs if r["cells"][col]["v"] not in ("", None)]
        if len(vals) < 2: continue
        best = max(v for v, _ in vals)
        for v, r in vals:
            if v == best: r["cells"][col]["marked"] = True
S['benchmarks'].append({"id":"bb_ablation","kind":"ablation","name":"Ablation study (HDS-MAP 3D)","columns":cols,"rows":rows,"categories":cats,
    "paperLink":link,"desc":""})
json.dump(S, open(sys.argv[2],"w",encoding='utf-8'), ensure_ascii=False, indent=1)
print(len(rows), "rows,", len(cats), "groups")
