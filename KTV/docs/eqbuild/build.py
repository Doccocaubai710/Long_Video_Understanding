import re, sys, base64, json, html as H
sys.path.insert(0, sys.argv[1])
from eqs import NOTATION, STAGE1, STAGE2, LOSS
from playwright.sync_api import sync_playwright
KD = sys.argv[2]; PAGE = sys.argv[3]
items = []
for blk in (NOTATION, STAGE1, STAGE2, LOSS):
    for it in blk:
        if it[0] == "eq": items.append((it[2], True))
        elif it[0] == "p": items += [(m, False) for m in re.findall(r"\$(.+?)\$", it[1])]
        else:
            for t in it[1]: items += [(m, False) for m in re.findall(r"\$(.+?)\$", t)]
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(); pg.set_content("<html><body></body></html>")
    pg.add_script_tag(path=f"{KD}/katex.min.js")
    out = pg.evaluate("""(items)=>items.map(([t,d])=>{try{return katex.renderToString(t,{displayMode:d,throwOnError:true,strict:'ignore',output:'html'})}catch(e){return 'ERR:'+e.message}})""", items)
    b.close()
R = dict(zip([(t, d) for t, d in items], out))
errs = [k for k, v in R.items() if v.startswith("ERR:")]
if errs:
    for k in errs: print("KATEX ERROR", R[k], k[0][:80]); 
    sys.exit(1)
def inl(t): return re.sub(r"\$(.+?)\$", lambda m: R[(m.group(1), False)], t)
def block(name, title, blk):
    h = [f'<!--EQ:{name}-->', '<div class="eqblock">', f'<div class="eqtitle">{title}</div>']
    for it in blk:
        if it[0] == "p": h.append(f'<p>{inl(it[1])}</p>')
        elif it[0] == "eq":
            h.append(f'<div class="eqrow" data-tex="{H.escape(it[2])}"><div class="eqbody">{R[(it[2], True)]}</div><div class="eqno">({it[1]})</div></div>')
        else: h.append('<ul>' + ''.join(f'<li>{inl(t)}</li>' for t in it[1]) + '</ul>')
    h += ['</div>', f'<!--/EQ:{name}-->']
    return "\n".join(h)
s = open(PAGE).read()
s = re.sub(r"<!--EQ:(\w+)-->.*?<!--/EQ:\1-->\n?", "", s, flags=re.S)       # idempotent
s = re.sub(r"<!--KATEXCSS-->.*?<!--/KATEXCSS-->\n?", "", s, flags=re.S)
B = {"NOT": block("NOT", "Công thức · Chia hộp 3D", NOTATION), "S1": block("S1", "Công thức · GĐ1 (CB-STE)", STAGE1),
     "S2": block("S2", "Công thức · GĐ2 (M3D-BD+)", STAGE2), "LOSS": block("LOSS", "Công thức · Hàm mất mát", LOSS)}
# vi tri chen
def ins_before(s, key, txt):
    assert s.count(key) == 1, key; return s.replace(key, txt + "\n" + key)
s = re.sub(r'  <details><summary>Công thức GĐ1 \(đầy đủ\)</summary>.*?</details>\n', '', s, flags=re.S)
s = re.sub(r'  <details><summary>Công thức GĐ2 \(đầy đủ\)</summary>.*?</details>\n', '', s, flags=re.S)
s = ins_before(s, '  <h3>2.3 Hai giai đoạn', B["NOT"])
s = ins_before(s, '  <h3>2.5 GĐ2: M3D-BD+', B["S1"])
s = ins_before(s, '  <p class="small muted">Lịch train: GĐ2 chỉ bật từ bước 2.000', B["S2"])
s = ins_before(s, '  <div class="sched">', B["LOSS"])
# CSS katex (font nhung base64, chi woff2)
css = open(f"{KD}/katex.min.css").read()
def font(m):
    f = m.group(1); data = base64.b64encode(open(f"{KD}/fonts/{f}.woff2", "rb").read()).decode()
    return f'src:url(data:font/woff2;base64,{data}) format("woff2")'
css = re.sub(r'src:url\(fonts/([\w-]+)\.woff2\) format\("woff2"\)(?:,url\([^)]*\) format\("[^"]*"\))*', font, css)
assert "fonts/" not in css, "font chua nhung het"
eqcss = """.eqblock{background:var(--surface);border:1px solid var(--line);border-left:3px solid var(--accent);border-radius:10px;padding:14px 18px;margin:14px 0}
.eqtitle{font-size:11.5px;font-weight:650;text-transform:uppercase;letter-spacing:.06em;color:var(--accent);margin-bottom:4px}
.eqblock p{margin:8px 0;font-size:14.5px}.eqblock ul{font-size:13.5px;margin:6px 0 0}
.eqrow{display:flex;align-items:center;gap:12px;margin:6px 0}
.eqbody{flex:1;min-width:0;overflow-x:auto;overflow-y:hidden;padding:2px 0}
.eqbody .katex-display{margin:.4em 0}
.eqno{flex:none;color:var(--ink-3);font-size:13.5px;font-variant-numeric:tabular-nums}
.katex{font-size:1.08em}"""
s = s.replace("</head>", f"<!--KATEXCSS--><style>{css}\n{eqcss}</style><!--/KATEXCSS-->\n</head>", 1)
open(PAGE, "w").write(s)
print("ok", len(items), "bieu thuc,", len(s)//1024, "KB")
