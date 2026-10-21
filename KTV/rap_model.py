"""
rap_model.py -- Perceiver co neo (Residual Anchored Perceiver, RAP), muc giu co dinh 25%.

Khac ban Perceiver cu (train_perceiver_stage1/2.py) o 3 diem, moi diem nham
vao mot loi da do duoc o outputs/perceiver_analysis (09/10):

  1. Query KHONG con la 64 vector hoc co dinh dung chung cho moi dau vao.
     Moi token dau ra duoc "neo" vao dung 1 token goc ma Uniform se giu
     (cung cong thuc linspace voi run_inference_qwen25vl_kitoke.select_uniform),
     query = chinh token goc do. 64 token cu co cosine 0.90 voi nhau vi chung
     xuat phat tu cung mot bo query; o day cac query khac nhau ngay tu dau.
  2. Dau ra = token neo + delta, lop chieu cuoi khoi tao bang 0. Luc chua
     train, dau ra TRUNG KHOP Uniform 25% (moc 63.07). Train chi co the di
     len tu moc do, khong phai hoc lai tu dau cach "noi chuyen" voi LLM.
  3. Moi token dau ra giu NGUYEN vi tri (thoi gian, hang, cot) cua token neo,
     nen mRoPE cua Qwen van dung. Ban cu gan vi tri 1D tang dan.

Delta duoc tinh bang cross-attention tu token neo sang moi token goc trong
cua so thoi gian [t-window, t+window] (mac dinh window=1: 3 unit = 756 token),
tuc la token neo "gom" thong tin cua nhung token bi bo quanh no.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_GRID = 64  # so hang/cot toi da cua luoi token trong 1 unit


def uniform_indices(n_total, retention, device):
    """Giong het select_uniform() cua run_inference_qwen25vl_kitoke.py."""
    K = max(1, int(retention * n_total))
    idx = torch.linspace(0, n_total - 1, K, device=device).round().long()
    return torch.unique(idx, sorted=True)


class AnchoredResampler(nn.Module):
    def __init__(self, dim, d_model=1024, num_layers=2, num_heads=8, window=1, ffn_mult=4,
                 bias3d=False, gamma=1.0, deform=False, deform_ratio=0.25):
        super().__init__()
        self.dim, self.d_model, self.num_layers, self.num_heads, self.window = dim, d_model, num_layers, num_heads, window
        # MAP-3D 3.3: cross-attn cong them phat khoang cach 3D, ban kinh theo S = 1/r (St = Ss = S^(1/3) moi truc)
        self.bias3d, self.gamma = bias3d, gamma
        # MAP-3D 3.2: Deformable 3D Tube Adapter -- DWConv3D tren luoi token goc (bottleneck), du doan cho moi neo
        # do dich (dt, dy, dx), do co gian hop (st, ss) va dQ; lop ra khoi tao 0 -> buoc 0 trung ban khong adapter
        self.deform, self.deform_ratio = deform, deform_ratio
        if deform:
            mid = int(dim * deform_ratio)
            self.ad_norm = nn.LayerNorm(dim)
            self.ad_down = nn.Linear(dim, mid)
            self.ad_conv = nn.Conv3d(mid, mid, 3, padding=1, groups=mid)
            self.ad_geo = nn.Linear(mid, 5)
            self.ad_dq = nn.Linear(mid, d_model)
            for m in (self.ad_geo, self.ad_dq):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
        self.ln_in = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, d_model)
        self.kv_proj = nn.Linear(dim, d_model)
        self.row_emb = nn.Embedding(MAX_GRID, d_model)
        self.col_emb = nn.Embedding(MAX_GRID, d_model)
        self.dt_emb = nn.Embedding(2 * window + 1, d_model)
        for emb in (self.row_emb, self.col_emb, self.dt_emb):
            nn.init.normal_(emb.weight, std=0.02)
        self.self_attns = nn.ModuleList([nn.MultiheadAttention(d_model, num_heads, batch_first=True) for _ in range(num_layers)])
        self.cross_attns = nn.ModuleList([nn.MultiheadAttention(d_model, num_heads, batch_first=True) for _ in range(num_layers)])
        self.ln_s = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.ln_c = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.ln_f = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.ffns = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model * ffn_mult), nn.GELU(), nn.Linear(d_model * ffn_mult, d_model))
            for _ in range(num_layers)
        ])
        self.ln_out = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, dim)
        nn.init.zeros_(self.out.weight)  # delta = 0 luc khoi tao -> dau ra == Uniform
        nn.init.zeros_(self.out.bias)

    def config(self):
        return {"dim": self.dim, "d_model": self.d_model, "num_layers": self.num_layers,
                "num_heads": self.num_heads, "window": self.window, "bias3d": self.bias3d, "gamma": self.gamma,
                "deform": self.deform, "deform_ratio": self.deform_ratio}

    def forward(self, X, kept_idx, H, W, return_attn=False, scene=None):
        """X: [T, N, D] token goc (float32). kept_idx: chi so phang da sap xep
        trong T*N. scene: [T] chi so canh (scene_model.scene_ids) -> neo chi nhin
        token cung canh; None = nhu cu. Tra ve (T2 [K, D], layout) voi layout de tinh loss phu."""
        T, N, D = X.shape
        dev = X.device
        flat = X.reshape(T * N, D)
        anchors = flat[kept_idx]
        t_of = kept_idx // N
        n_of = kept_idx % N
        r_of = (n_of // W).clamp(max=MAX_GRID - 1)
        c_of = (n_of % W).clamp(max=MAX_GRID - 1)
        # vi tri cua moi query trong unit cua no -> xep thanh [T, Qmax]
        counts = torch.bincount(t_of, minlength=T)
        first = torch.cumsum(counts, 0) - counts
        pos = torch.arange(kept_idx.numel(), device=dev) - first[t_of]
        Qmax = int(counts.max().item())
        q_valid = torch.zeros(T, Qmax, dtype=torch.bool, device=dev)
        q_valid[t_of, pos] = True

        w = self.window
        rows = (torch.arange(N, device=dev) // W).clamp(max=MAX_GRID - 1)
        cols = (torch.arange(N, device=dev) % W).clamp(max=MAX_GRID - 1)
        kv = self.kv_proj(self.ln_in(flat)).view(T, N, -1) + (self.row_emb(rows) + self.col_emb(cols)).unsqueeze(0)
        kv_parts, mask_parts = [], []
        for k, dt in enumerate(range(-w, w + 1)):
            src = torch.arange(T, device=dev) + dt
            ok = (src >= 0) & (src < T)
            if scene is not None:
                ok = ok & (scene[src.clamp(0, T - 1)] == scene)
            part = kv[src.clamp(0, T - 1)] + self.dt_emb.weight[k]
            kv_parts.append(part)
            mask_parts.append((~ok).unsqueeze(1).expand(T, N))
        kvw = torch.cat(kv_parts, dim=1)                # [T, (2w+1)N, d]
        kpm = torch.cat(mask_parts, dim=1)              # True = bo qua (unit ngoai video)

        q_flat = self.q_proj(self.ln_in(anchors)) + self.row_emb(r_of) + self.col_emb(c_of) + self.dt_emb.weight[w]
        attn_mask = None
        if self.bias3d or self.deform:
            off = q_flat.new_zeros(kept_idx.numel(), 3)          # (dt, dy, dx) cua tam hop
            lsc = q_flat.new_zeros(kept_idx.numel(), 2)          # log co gian (thoi gian, khong gian)
            if self.deform:
                g = self.ad_down(self.ad_norm(flat)).view(T, H, W, -1).permute(3, 0, 1, 2).unsqueeze(0)
                g = F.gelu(self.ad_conv(g))[0].permute(1, 2, 3, 0).reshape(T * N, -1)[kept_idx]
                geo = self.ad_geo(g)
                off = torch.stack([w * torch.tanh(geo[:, 0]), 3.0 * torch.tanh(geo[:, 1]), 3.0 * torch.tanh(geo[:, 2])], 1)
                lsc = geo[:, 3:5].clamp(-2.0, 2.0)
                q_flat = q_flat + self.ad_dq(g)
            S1 = (float(T * N) / kept_idx.numel()) ** (1.0 / 3.0)   # S = 1/r chia deu cho 3 truc
            st, ss = S1 * torch.exp(lsc[:, 0]), S1 * torch.exp(lsc[:, 1])
            kdt = torch.arange(-w, w + 1, device=dev).float().repeat_interleave(N)       # [L]
            krow = (torch.arange(N, device=dev) // W).float().repeat(2 * w + 1)
            kcol = (torch.arange(N, device=dev) % W).float().repeat(2 * w + 1)
            qr, qc = (n_of // W).float(), (n_of % W).float()
            bt = ((kdt.unsqueeze(0) - off[:, :1]) / st.unsqueeze(1)) ** 2
            bs = ((krow.unsqueeze(0) - qr.unsqueeze(1) - off[:, 1:2]) ** 2
                  + (kcol.unsqueeze(0) - qc.unsqueeze(1) - off[:, 2:3]) ** 2) / ss.unsqueeze(1) ** 2
            b = -self.gamma * (bt + bs)                                                 # [K, L]
            bias = b.new_zeros(T, Qmax, b.shape[1])
            bias[t_of, pos] = b
            bias = bias.masked_fill(kpm.unsqueeze(1), float("-inf"))
            attn_mask = bias.repeat_interleave(self.num_heads, dim=0)                    # [T*heads, Qmax, L]
        q = q_flat.new_zeros(T, Qmax, q_flat.shape[-1])
        q[t_of, pos] = q_flat
        q_pad = ~q_valid
        attn_maps = []
        for sa, ca, ln_s, ln_c, ln_f, ffn in zip(self.self_attns, self.cross_attns, self.ln_s, self.ln_c, self.ln_f, self.ffns):
            h = ln_s(q)
            q = q + sa(h, h, h, key_padding_mask=q_pad, need_weights=False)[0]
            h = ln_c(q)
            if attn_mask is None:
                co, aw = ca(h, kvw, kvw, key_padding_mask=kpm, need_weights=return_attn, average_attn_weights=True)
            else:
                co, aw = ca(h, kvw, kvw, attn_mask=attn_mask, need_weights=return_attn, average_attn_weights=True)
            if return_attn:
                attn_maps.append(aw.detach())
            q = q + co
            q = q + ffn(ln_f(q))
        delta = self.out(self.ln_out(q))[t_of, pos]
        T2 = anchors + delta
        layout = {"t_of": t_of, "pos": pos, "Qmax": Qmax, "q_valid": q_valid, "anchors": anchors, "attn": attn_maps}
        return T2, layout


def _pad_units(tok, layout, T):
    out = tok.new_zeros(T, layout["Qmax"], tok.shape[-1])
    out[layout["t_of"], layout["pos"]] = tok
    return out


def coverage_loss(X, T2, layout):
    """Moi token goc cua unit phai co it nhat 1 token dau ra (cua cung unit)
    gan no. Day la chieu nguoc lai cua Anchor Loss cu (cu: moi dau ra gan 1
    token goc -> ca 64 dau ra don ve 3-4 token goc)."""
    T = X.shape[0]
    P = F.normalize(_pad_units(T2, layout, T), dim=-1)
    sim = F.normalize(X, dim=-1) @ P.transpose(1, 2)                 # [T, N, Qmax]
    sim = sim.masked_fill(~layout["q_valid"].unsqueeze(1), -1.0)
    return 1.0 - sim.max(dim=-1).values.mean()


def _logdet_units(tok, layout, T, delta=1e-3):
    P = F.normalize(_pad_units(tok, layout, T), dim=-1)
    G = P @ P.transpose(1, 2)
    valid = layout["q_valid"]
    eye = torch.eye(G.shape[-1], device=G.device).unsqueeze(0)
    pair = (valid.unsqueeze(1) & valid.unsqueeze(2)).float()
    G = G * pair + eye * (1.0 - valid.float()).unsqueeze(-1)          # hang dem -> don vi, khong dong gop
    L = torch.linalg.cholesky(G.double() + delta * eye.double())
    return 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1).float()   # [T]


def dpp_hinge_loss(T2, layout, T):
    """So hang da dang DPP (log det ma tran Gram, nhu R_DPP trong dac ta), dang
    hinge: chi phat khi dau ra KEM da dang hon chinh cac token neo cua no.
    Khong ep truc giao tuyet doi (token goc tu nhien da giong nhau ~0.3)."""
    with torch.no_grad():
        ref = _logdet_units(layout["anchors"], layout, T)
    cur = _logdet_units(T2, layout, T)
    return F.relu(ref - cur).mean() / layout["Qmax"]


@torch.no_grad()
def diversity_stats(T2, layout, T):
    """Chan doan giong outputs/perceiver_analysis: erank + cosine trung binh, tinh theo unit."""
    def one(tok):
        P = F.normalize(_pad_units(tok, layout, T), dim=-1)[: min(T, 16)]
        v = layout["q_valid"][: min(T, 16)]
        eranks, coss = [], []
        for p, m in zip(P, v):
            p = p[m]
            s = torch.linalg.svdvals(p.double())
            pr = s / s.sum()
            eranks.append(float(torch.exp(-(pr * torch.log(pr + 1e-12)).sum())))
            g = p @ p.t()
            n = p.shape[0]
            coss.append(float((g.sum() - n) / max(1, n * (n - 1))))
        return sum(eranks) / len(eranks), sum(coss) / len(coss)
    e2, c2 = one(T2)
    e1, c1 = one(layout["anchors"])
    return {"erank_out": e2, "erank_anchor": e1, "cos_out": c2, "cos_anchor": c1}


def save_resampler(path, resampler, step, extra=None):
    torch.save({"state_dict": resampler.state_dict(), "config": resampler.config(), "step": step, **(extra or {})}, path)


def load_resampler(path, device):
    ckpt = torch.load(path, map_location=device)
    m = AnchoredResampler(**ckpt["config"]).to(device).float()
    m.load_state_dict(ckpt["state_dict"])
    return m, ckpt
