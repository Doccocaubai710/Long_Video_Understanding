# Dựng công thức cho HDS_MAP3D_trinh_bay.html

Công thức viết bằng LaTeX trong `eqs.py`, được KaTeX render sẵn thành HTML (nhúng CSS + font base64, mở offline được).
Sửa `eqs.py` rồi chạy lại (idempotent, thay các khối `<!--EQ:...-->` cũ):

    python build.py docs/eqbuild docs/eqbuild/katex docs/HDS_MAP3D_trinh_bay.html   # từ thư mục KTV

Cần Python có `playwright` + chromium (`pip install playwright && playwright install chromium`).
