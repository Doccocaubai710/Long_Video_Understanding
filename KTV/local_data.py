"""Ban chep du lieu o o local cua container (outputs/rap_elastic/copy_local.sh), de khong phu thuoc Lustre khi Lustre nghen.

local_mirror(p, marker) tra ve ban local cua thu muc p neu da chep xong (co file .complete o p/marker), khong thi tra ve p.
O local mat khi container bi tao lai -> khi do tu dong doc lai tu Lustre.
"""
import os

KTV = os.path.dirname(os.path.abspath(__file__))
LOCAL = os.environ.get("KTV_LOCAL_DATA", "/root/local_data/KTV")


def local_mirror(p, marker=""):
    if not p:
        return p
    rel = os.path.relpath(os.path.abspath(p), KTV)
    if rel.startswith(".."):
        return p
    q = os.path.join(LOCAL, rel)
    if os.path.exists(os.path.join(q, marker, ".complete")):
        print(f"[local_data] dung ban local {q} thay cho {p}", flush=True)
        return q
    return p
