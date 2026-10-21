"""
alloc_model.py -- chia lai so neo giua cac unit theo muc bien doi canh (khong can train).

Y tuong lay tu buoc S1 cua HTP-LVU (phan doan hau projector kieu KiToke), nhung
KHONG dung de cat bo token: chi dung de quyet dinh MOI unit duoc bao nhieu neo.
  - floor (vd 0.5) ngan sach chia deu cho moi unit (giong luoi Uniform),
  - phan con lai chia theo do bien doi cua unit so voi unit truoc
    Delta_t = trung binh_i min_{j trong lan can 3x3} (1 - cos(h_t,i, h_{t-1,j})).
Trong moi unit, neo van rai deu (linspace) tren luoi hang/cot.
floor = 1.0 ~ Uniform (doi chung).
"""

import torch
import torch.nn.functional as F

from rap_model import uniform_indices


@torch.no_grad()
def unit_change(feat, H, W):
    """feat: [T, N, D] -> [T] do bien doi cua moi unit so voi unit truoc (unit 0 lay bang unit 1)."""
    T, N, D = feat.shape
    if T < 2:
        return feat.new_ones(T)
    X = F.normalize(feat.float(), dim=-1).view(T, H, W, D)
    cur, prev = X[1:], X[:-1]
    prev_p = F.pad(prev.permute(0, 3, 1, 2), (1, 1, 1, 1)).permute(0, 2, 3, 1)  # [T-1, H+2, W+2, D], vien = 0
    best = None
    for dy in range(3):
        for dx in range(3):
            sim = (cur * prev_p[:, dy:dy + H, dx:dx + W]).sum(-1)              # vien = 0 -> sim 0, khong thang
            best = sim if best is None else torch.maximum(best, sim)
    d = (1.0 - best).mean(dim=(1, 2)).clamp(min=0)                               # [T-1]
    return torch.cat([d[:1], d])


def _round_counts(share, K, cap):
    """Lam tron share (tong = K) thanh so nguyen, moi unit trong [1, cap], tong = K."""
    c = share.floor().long().clamp(1, cap)
    while int(c.sum()) != K:
        diff = K - int(c.sum())
        rem = share - c.float()
        if diff > 0:
            rem = rem.masked_fill(c >= cap, -1e9)
            c[torch.topk(rem, min(diff, int((c < cap).sum()))).indices] += 1
        else:
            rem = rem.masked_fill(c <= 1, 1e9)
            c[torch.topk(-rem, min(-diff, int((c > 1).sum()))).indices] -= 1
    return c


@torch.no_grad()
def scene_alloc_indices(feat, H, W, retention, floor=0.5, max_mult=4.0):
    """Tra ve (kept_idx da sap xep trong T*N, counts [T]). Tong so neo bang dung Uniform.
    max_mult: moi unit toi da max_mult lan phan chia deu (tranh don het neo vao 1 cho cat canh)."""
    T, N, _ = feat.shape
    dev = feat.device
    K = int(uniform_indices(T * N, retention, dev).numel())
    d = unit_change(feat, H, W)
    w = d / d.sum() if float(d.sum()) > 0 else torch.full_like(d, 1.0 / T)
    share = floor * K / T + (1.0 - floor) * K * w
    cap = min(N, max(int(-(-max_mult * K // T)), -(-K // T)))
    counts = _round_counts(share, K, cap)
    parts = [t * N + torch.linspace(0, N - 1, int(k), device=dev).round().long() for t, k in enumerate(counts.tolist())]
    return torch.unique(torch.cat(parts), sorted=True), counts
