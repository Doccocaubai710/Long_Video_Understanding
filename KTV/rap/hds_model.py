"""
hds_model.py -- HDS-MAP 3D (docs/De Xuat Nghien Cuu_ HDS-MAP 3D ...pdf) tren nen RAP.

Giai doan 1 (CuboidSlider): chia video [T, H, W] thanh cac hop 3D (st, sy, sx) khong chong nhau,
  moi hop chon DUNG 1 token goc. Diem = MLP([DWConv3D cuc bo || LN(h) - LN(tam)]) - phat lech tam,
  lop cuoi khoi tao 0 -> buoc 0 chon tam hop. Chon cung bang argmax, gradient qua straight-through (CB-STE).
  Token chon giu nguyen vi tri goc nen mRoPE tu dung vi tri moi.
  So token = K = int(r * T * N) nhu Uniform: hop duoc chon kich thuoc nho nhat cho so hop >= K,
  thua thi bo deu (linspace tren thu tu hop).
Giai doan 2: AnchoredResampler (RAP, + bias3d/deform = M3D-BD) voi self_mask (che chinh token neo),
  tam phat khoang cach dat tai vi tri moi (tu dong vi dung kept_idx moi) va cong (1 - r).

Luat khong train (Pha 0): center | var (bien doi 3D) | norm (chuan lon nhat) | medoid (tong cosine trong hop).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rap.rap_model import AnchoredResampler, uniform_indices


def choose_box(T, H, W, K, shape="auto"):
    """(st, sy, sx) nho nhat sao cho so hop >= K; hoa thi uu tien hop can doi.
    shape (ablation): auto = mac dinh | t = chi theo thoi gian (sy = sx = 1) | s = chi theo khong gian (st = 1)
                      | b = can doi nhat (canh dai nhat nho nhat, roi moi den so hop)."""
    best = None
    for st in range(1, min(T, 16) + 1):
        for sy in range(1, min(H, 8) + 1):
            for sx in range(1, min(W, 8) + 1):
                if (shape == "t" and (sy, sx) != (1, 1)) or (shape == "s" and st != 1):
                    continue
                kb = math.ceil(T / st) * math.ceil(H / sy) * math.ceil(W / sx)
                if kb < K or (shape == "b" and kb > 1.3 * K):   # b: chi xet hop khong bo qua > 30% so hop
                    continue
                key = (max(st, sy, sx), kb, abs(sy - sx)) if shape == "b" else (kb, max(st, sy, sx), abs(sy - sx))
                if best is None or key < best[0]:
                    best = (key, (st, sy, sx))
    return best[1] if best else (1, 1, 1)


def cuboid_layout(T, H, W, retention, device, shape="auto", margin=0):
    """Tra ve box_idx [K, S] (chi so phang trong T*N, -1 = ngoai video), center [K] (vi tri tam trong hop),
    loc [S, 3] (dt, dy, dx cuc bo), (st, sy, sx)."""
    return cuboid_layout_k(T, H, W, max(1, int(retention * T * H * W)), device, shape, margin)


def cuboid_layout_k(T, H, W, K, device, shape="auto", margin=0):
    """Nhu cuboid_layout nhung cho truoc so hop K (dung cho phan bo thich ung).
    margin > 0 (ablation): ung vien mo rong them margin o moi phia moi truc (hop chong lan, tam van la tam hop goc)."""
    N = H * W
    st, sy, sx = choose_box(T, H, W, K, shape)
    Kt, Ky, Kx = math.ceil(T / st), math.ceil(H / sy), math.ceil(W / sx)
    m = margin
    lt, ly, lx = torch.meshgrid(torch.arange(-m, st + m, device=device), torch.arange(-m, sy + m, device=device),
                                torch.arange(-m, sx + m, device=device), indexing="ij")
    loc = torch.stack([lt.reshape(-1), ly.reshape(-1), lx.reshape(-1)], 1)            # [S, 3]
    bt, by, bx = torch.meshgrid(torch.arange(Kt, device=device), torch.arange(Ky, device=device),
                                torch.arange(Kx, device=device), indexing="ij")
    base = torch.stack([bt.reshape(-1) * st, by.reshape(-1) * sy, bx.reshape(-1) * sx], 1)  # [Kb, 3]
    pos = base.unsqueeze(1) + loc.unsqueeze(0)                                           # [Kb, S, 3]
    valid = (pos >= 0).all(-1) & (pos[..., 0] < T) & (pos[..., 1] < H) & (pos[..., 2] < W)
    box_idx = torch.where(valid, pos[..., 0] * N + pos[..., 1] * W + pos[..., 2], torch.full_like(valid, -1, dtype=torch.long))
    ext = torch.stack([(T - base[:, 0]).clamp(max=st), (H - base[:, 1]).clamp(max=sy), (W - base[:, 2]).clamp(max=sx)], 1)
    c = ext // 2 + m                                                                      # tam theo kich thuoc thuc
    Ly, Lx = sy + 2 * m, sx + 2 * m
    center = c[:, 0] * (Ly * Lx) + c[:, 1] * Lx + c[:, 2]
    if box_idx.shape[0] > K:   # bo deu so hop thua
        keep = torch.unique(torch.linspace(0, box_idx.shape[0] - 1, K, device=device).round().long())
        box_idx, center = box_idx[keep], center[keep]
    return box_idx, center, loc, (st, sy, sx)


def rule_scores(X, H, W, rule):
    """Diem tung token [T*N] cho luat khong train."""
    T, N, D = X.shape
    if rule == "norm":
        return X.norm(dim=-1).reshape(-1)
    if rule == "var":
        G = X.view(T, H, W, D)
        dt = torch.zeros(T, H, W, device=X.device)
        if T > 1:
            d = (G[1:] - G[:-1]).pow(2).sum(-1)
            dt[1:] += d
            dt[:-1] += d
            dt[1:-1] /= 2
        ds = torch.zeros(T, H, W, device=X.device)
        ds[:, :, :-1] += (G[:, :, 1:] - G[:, :, :-1]).pow(2).sum(-1)
        ds[:, :-1, :] += (G[:, 1:] - G[:, :-1]).pow(2).sum(-1)
        return (dt + ds).reshape(-1)
    raise ValueError(rule)


@torch.no_grad()
def rule_select(X, H, W, retention, rule):
    """Pha 0: chon 1 token / hop bang luat. Tra ve kept_idx da sap xep, stats."""
    T, N, D = X.shape
    box_idx, center, loc, box = cuboid_layout(T, H, W, retention, X.device)
    valid = box_idx >= 0
    if rule == "center":
        j = center
    elif rule == "medoid":
        flat = F.normalize(X.reshape(T * N, D), dim=-1)
        Hb = flat[box_idx.clamp(min=0)] * valid.unsqueeze(-1)                           # [K, S, D]
        s = (Hb @ Hb.transpose(1, 2)).sum(-1)
        j = s.masked_fill(~valid, -1e9).argmax(-1)
    else:
        s = rule_scores(X, H, W, rule)[box_idx.clamp(min=0)]
        j = s.masked_fill(~valid, -1e9).argmax(-1)
    kept = box_idx.gather(1, j.unsqueeze(1)).squeeze(1)
    kept, _ = kept.sort()
    return kept, slide_stats(j, center, loc, box)


def slide_stats(j, center, loc, box):
    lj, lc = loc[j].float(), loc[center].float()
    return {"hop_rate": float((j != center).float().mean()), "move_t": float((lj[:, 0] - lc[:, 0]).abs().mean()),
            "move_s": float(((lj[:, 1:] - lc[:, 1:]) ** 2).sum(-1).sqrt().mean()), "box": "x".join(map(str, box))}


def _inv_softplus(v):
    return math.log(math.expm1(v))


class CuboidSlider(nn.Module):
    """Giai doan 1: CB-STE trong hop 3D."""

    def __init__(self, dim, mid_ratio=0.25, lambda_init=0.5, lam_mode="learn", pen_scale=1.0, box_shape="auto", box_margin=0,
                 box_k=1):
        super().__init__()
        # ablation phat lech tam: learn = 2 he so hoc | shared = 1 he so hoc chung | fixed = hang so lambda_init (0 tham so)
        # pen_scale: nhan phat luc test (khong train lai); box_shape / box_margin: cach chia hop (xem choose_box / cuboid_layout_k)
        self.lam_mode, self.lambda_init, self.pen_scale = lam_mode, lambda_init, pen_scale
        self.box_shape, self.box_margin = box_shape, box_margin
        # box_k > 1 (ablation): moi hop chon k token khac nhau (hop to gap ~k lan, so hop = ceil(K / k))
        self.box_k = box_k
        mid = int(dim * mid_ratio)
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, mid, bias=False)
        self.conv = nn.Conv3d(mid, mid, 3, padding=1, groups=mid)
        self.score = nn.Sequential(nn.Linear(mid + dim, mid), nn.GELU(), nn.Linear(mid, 1))
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)
        if lam_mode in ("learn", "shared"):
            self.lam_t = nn.Parameter(torch.tensor(_inv_softplus(lambda_init)))
        if lam_mode == "learn":
            self.lam_s = nn.Parameter(torch.tensor(_inv_softplus(lambda_init)))

    def _lams(self):
        if self.lam_mode == "fixed":
            lt = ls = self.lambda_init
        elif self.lam_mode == "shared":
            lt = ls = F.softplus(self.lam_t)
        else:
            lt, ls = F.softplus(self.lam_t), F.softplus(self.lam_s)
        return lt * self.pen_scale, ls * self.pen_scale

    def _features(self, X, H, W):
        T, N, D = X.shape
        flat = X.reshape(T * N, D)
        hn = self.norm(flat)
        g = self.down(hn).view(T, H, W, -1).permute(3, 0, 1, 2).unsqueeze(0)
        g = F.gelu(self.conv(g))[0].permute(1, 2, 3, 0).reshape(T * N, -1)
        return flat, hn, g

    def _choose(self, flat, hn, g, box_idx, center, loc, box, tau, soft=False, k=1):
        """CB-STE trong tung hop: tra ve kept [K] (chua sap xep), anchors [K, D], j [K], probs [K, S].
        soft=True: xuoi = trung binh co trong so softmax(logits / tau) (vi tri van lay theo argmax)."""
        valid = box_idx >= 0
        bi = box_idx.clamp(min=0)
        hc = hn[box_idx.gather(1, center.unsqueeze(1)).squeeze(1)]                       # [K, D]
        # = self.score(cat[g, hn - hc]) nhung tach lop dau: chieu tung token 1 lan roi tru phan tam (khong tao [K, S, mid + D])
        lin, mid = self.score[0], g.shape[-1]
        a = g @ lin.weight[:, :mid].t() + hn @ lin.weight[:, mid:].t() + lin.bias          # [T*N, mid]
        h1 = a[bi] - (hc @ lin.weight[:, mid:].t()).unsqueeze(1)                          # [K, S, mid]
        raw = self.score[2](self.score[1](h1)).squeeze(-1)                                # [K, S]
        st, sy, sx = box
        lc = loc[center].float()                                                          # [K, 3]
        lf = loc.float().unsqueeze(0)                                                     # [1, S, 3]
        pt = ((lf[..., 0] - lc[:, :1]) / max(1, st)) ** 2
        ps = ((lf[..., 1] - lc[:, 1:2]) ** 2 + (lf[..., 2] - lc[:, 2:3]) ** 2) / max(1, sy * sy + sx * sx)
        lam_t, lam_s = self._lams()
        logits = raw - lam_t * pt - lam_s * ps
        logits = logits.masked_fill(~valid, -1e9)
        if self.box_margin > 0:
            logits = self._dedup(logits, box_idx)
        probs = F.softmax(logits / tau, dim=-1)
        j = logits.argmax(-1)
        hard = F.one_hot(j, probs.shape[-1]).to(probs.dtype)
        w = probs if soft else hard + probs - probs.detach()                               # xuoi: 1-hot (hoac soft), nguoc: softmax
        anchors = (w.unsqueeze(-1) * flat[bi]).sum(1)                                      # [K, D] = token goc
        kept = box_idx.gather(1, j.unsqueeze(1)).squeeze(1)
        if k > 1:   # chon tiep k - 1 token: bo token da chon, CB-STE lai tren phan con lai cua hop
            ks, an, lg, taken = [kept], [anchors], logits, hard.bool()
            for _ in range(k - 1):
                lg = lg.masked_fill(taken, -1e9)
                ok = lg.max(-1).values > -1e8                                              # hop con ung vien hop le
                p2 = F.softmax(lg / tau, dim=-1)
                j2 = lg.argmax(-1)
                h2 = F.one_hot(j2, p2.shape[-1]).to(p2.dtype)
                w2 = p2 if soft else h2 + p2 - p2.detach()
                ks.append(box_idx.gather(1, j2.unsqueeze(1)).squeeze(1)[ok])
                an.append((w2.unsqueeze(-1) * flat[bi]).sum(1)[ok])
                taken = taken | h2.bool()
            kept, anchors = torch.cat(ks), torch.cat(an)
        return kept, anchors, j, probs

    @torch.no_grad()
    def _dedup_mask(self, logits, box_idx, rounds=30):
        """Hop chong lan: neu nhieu hop cung chon 1 token thi hop co diem cao nhat giu, cac hop khac
        bo ung vien do va chon token tot ke tiep (lap toi da `rounds` lan). Tra ve mask [K, S] cac ung vien bi bo."""
        lg = logits.detach().clone()
        drop = torch.zeros_like(lg, dtype=torch.bool)
        K = lg.shape[0]
        ar = torch.arange(K, device=lg.device)
        for _ in range(rounds):
            j = lg.argmax(-1)
            tok = box_idx[ar, j].clamp(min=0)
            val = lg[ar, j]
            best = torch.full((int(box_idx.max()) + 1,), -float("inf"), device=lg.device).scatter_reduce(0, tok, val, "amax")
            win = val >= best[tok]
            # hoa diem: giu hop co chi so nho nhat
            first = torch.full_like(best, K, dtype=torch.long).scatter_reduce(0, tok, torch.where(win, ar, K), "amin")
            lose = first[tok] != ar
            if not lose.any():
                break
            drop[ar[lose], j[lose]] = True
            lg[ar[lose], j[lose]] = -1e9
        return drop

    def _dedup(self, logits, box_idx):
        return logits.masked_fill(self._dedup_mask(logits, box_idx), -1e9)

    def forward(self, X, H, W, retention, tau=1.0, soft=False):
        T, N, D = X.shape
        K = max(1, int(retention * T * H * W))
        box_idx, center, loc, box = cuboid_layout_k(T, H, W, math.ceil(K / self.box_k), X.device, self.box_shape, self.box_margin)
        flat, hn, g = self._features(X, H, W)
        kept, anchors, j, probs = self._choose(flat, hn, g, box_idx, center, loc, box, tau, soft, self.box_k)
        if kept.numel() > K:   # chon k / hop co the du < k token -> cat bot cuoi (cac lan chon sau cung)
            kept, anchors = kept[:K], anchors[:K]
        order = kept.argsort()
        st_ = slide_stats(j, center, loc, box)
        st_["entropy"] = float(-(probs.detach() * (probs.detach() + 1e-12).log()).sum(-1).mean())
        lt, ls = self._lams()
        st_["lam_t"], st_["lam_s"] = float(torch.as_tensor(lt).detach()), float(torch.as_tensor(ls).detach())
        if self.box_margin > 0:   # hop chong lan: ti le token bi chon trung
            st_["dup"] = float(1 - kept.unique().numel() / kept.numel())
        return kept[order], anchors[order], st_, None


def _pool_feats(hn, g, idx, valid):
    """Dac trung gop cho moi nhom token: [mean(hn), mean(g), std(g)]. idx [M, S] (clamp >= 0), valid [M, S]."""
    v = valid.unsqueeze(-1).float()
    n = v.sum(1).clamp(min=1)
    mh = (hn[idx] * v).sum(1) / n
    gg = g[idx]
    mg = (gg * v).sum(1) / n
    sg = (((gg - mg.unsqueeze(1)) ** 2 * v).sum(1) / n).clamp(min=1e-6).sqrt()
    return torch.cat([mh, mg, sg], -1)


def _gumbel(x):
    return -torch.log(-torch.log(torch.rand_like(x).clamp(1e-6, 1 - 1e-6)))


def _unit_cv(kept, T, N):
    c = torch.bincount(kept // N, minlength=T).float()
    return float(c.std() / c.mean().clamp(min=1e-6))


class BoxSelectSlider(CuboidSlider):
    """Cach 1 -- chon hop co hoc: luoi min gap doi (2K hop), diem moi hop tu dac trung gop, giu K hop diem cao nhat.
    Buoc 0: diem hoc = 0 (lop cuoi khoi tao 0), prior +1 cho K hop rai deu -> giong luoi goc.
    Train: them nhieu Gumbel (noise) de kham pha; gradient vao diem hop qua he so nhan straight-through
    m = exp(s - s.detach()) (xuoi = 1) nhan vao token dau ra cua hop duoc chon."""

    def __init__(self, dim, mid_ratio=0.25, lambda_init=0.5, noise=0.3, fine=2.0):
        super().__init__(dim, mid_ratio, lambda_init)
        mid = int(dim * mid_ratio)
        self.noise, self.fine = noise, fine
        self.box_score = nn.Sequential(nn.Linear(dim + 2 * mid, mid), nn.GELU(), nn.Linear(mid, 1))
        nn.init.zeros_(self.box_score[-1].weight)
        nn.init.zeros_(self.box_score[-1].bias)

    def forward(self, X, H, W, retention, tau=1.0):
        T, N, D = X.shape
        K = max(1, int(retention * T * N))
        K2 = min(T * N, int(round(self.fine * K)))
        box_idx, center, loc, box = cuboid_layout_k(T, H, W, K2, X.device)
        flat, hn, g = self._features(X, H, W)
        valid = box_idx >= 0
        s = self.box_score(_pool_feats(hn, g, box_idx.clamp(min=0), valid)).squeeze(-1)   # [K2]
        prior = torch.zeros_like(s)
        prior[torch.linspace(0, s.numel() - 1, K, device=s.device).round().long()] = 1.0
        key = s.detach() + prior
        if self.training and self.noise > 0:
            key = key + self.noise * _gumbel(key)
        sel = torch.topk(key, K).indices
        kept, anchors, j, probs = self._choose(flat, hn, g, box_idx[sel], center[sel], loc, box, tau)
        mult = torch.exp(s[sel] - s[sel].detach())
        order = kept.argsort()
        st_ = slide_stats(j, center[sel], loc, box)
        st_["entropy"] = float(-(probs * (probs + 1e-12).log()).sum(-1).mean())
        st_["box_swap"] = float(1.0 - prior[sel].mean())          # ti le hop chon khac luoi rai deu
        st_["unit_cv"] = _unit_cv(kept, T, N)                      # do lech so token giua cac unit thoi gian
        return kept[order], anchors[order], st_, mult[order]


def _round_to(share, K, cap):
    """Lam tron share (tong = K) thanh so nguyen trong [1, cap] (cap: tensor), tong = K."""
    c = torch.minimum(share.floor().long().clamp(min=1), cap)
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


class SegBudgetSlider(CuboidSlider):
    """Cach 2 -- ngan sach theo doan thoi gian: chia T thanh doan dai bang chieu thoi gian cua hop goc (st),
    dau nho du doan ti le token moi doan: share = floor/S + (1 - floor) softmax(s). Moi doan dung luoi hop rieng
    (cuboid_layout_k voi so hop = ngan sach doan), ben trong van CB-STE. Buoc 0: s = 0 -> chia deu ~ luoi goc.
    Gradient vao s qua he so nhan straight-through m = share / share.detach() tren token cua doan."""

    def __init__(self, dim, mid_ratio=0.25, lambda_init=0.5, noise=0.3, floor=0.5, max_mult=4.0):
        super().__init__(dim, mid_ratio, lambda_init)
        mid = int(dim * mid_ratio)
        self.noise, self.floor, self.max_mult = noise, floor, max_mult
        self.seg_score = nn.Sequential(nn.Linear(dim + 2 * mid, mid), nn.GELU(), nn.Linear(mid, 1))
        nn.init.zeros_(self.seg_score[-1].weight)
        nn.init.zeros_(self.seg_score[-1].bias)

    def forward(self, X, H, W, retention, tau=1.0):
        T, N, D = X.shape
        dev = X.device
        K = max(1, int(retention * T * N))
        st0 = choose_box(T, H, W, K)[0]
        bounds = [(t0, min(T, t0 + st0)) for t0 in range(0, T, st0)]
        S = len(bounds)
        flat, hn, g = self._features(X, H, W)
        seg_tok = torch.full((S, st0 * N), -1, dtype=torch.long, device=dev)
        for i, (a, b) in enumerate(bounds):
            seg_tok[i, : (b - a) * N] = torch.arange(a * N, b * N, device=dev)
        s = self.seg_score(_pool_feats(hn, g, seg_tok.clamp(min=0), seg_tok >= 0)).squeeze(-1)   # [S]
        logit = s + (self.noise * _gumbel(s) if (self.training and self.noise > 0) else 0.0)
        lens = torch.tensor([b - a for a, b in bounds], device=dev, dtype=torch.float)
        base = lens / lens.sum()                                    # phan chia deu theo do dai doan
        share = self.floor * base + (1.0 - self.floor) * F.softmax(logit, -1)
        cap = torch.minimum((lens * N).long(), (self.max_mult * K * base).ceil().long())
        counts = _round_to((share * K).detach(), K, cap)
        kept_l, anc_l, mult_l, j_l, c_l, ent = [], [], [], [], [], []
        m_seg = share / share.detach()
        for i, (a, b) in enumerate(bounds):
            bidx, cen, loc, box = cuboid_layout_k(b - a, H, W, int(counts[i]), dev)
            bidx = torch.where(bidx >= 0, bidx + a * N, bidx)
            kept, anchors, j, probs = self._choose(flat, hn, g, bidx, cen, loc, box, tau)
            kept_l.append(kept); anc_l.append(anchors); mult_l.append(m_seg[i].expand(kept.numel()))
            ent.append(float(-(probs * (probs + 1e-12).log()).sum(-1).mean()))
            j_l.append(float((j != cen).float().mean()))
        kept, anchors, mult = torch.cat(kept_l), torch.cat(anc_l), torch.cat(mult_l)
        order = kept.argsort()
        cf = counts.float() / lens
        st_ = {"hop_rate": sum(j_l) / S, "entropy": sum(ent) / S, "seg_cv": float(cf.std() / cf.mean()),
               "seg_max": float(cf.max() / cf.mean()), "unit_cv": _unit_cv(kept, T, N)}
        return kept[order], anchors[order], st_, mult[order]


def unit_layout(kept_idx, anchors, T, N):
    """Layout nhu AnchoredResampler (cho coverage/dpp/diversity) khi khong co giai doan 2."""
    t_of = kept_idx // N
    counts = torch.bincount(t_of, minlength=T)
    first = torch.cumsum(counts, 0) - counts
    pos = torch.arange(kept_idx.numel(), device=kept_idx.device) - first[t_of]
    Qmax = int(counts.max().item())
    q_valid = torch.zeros(T, Qmax, dtype=torch.bool, device=kept_idx.device)
    q_valid[t_of, pos] = True
    return {"t_of": t_of, "pos": pos, "Qmax": Qmax, "q_valid": q_valid, "anchors": anchors, "attn": []}


class HDSPolicy(nn.Module):
    """slider (giai doan 1, co the tat -> neo Uniform linspace) + stage2 (AnchoredResampler, co the tat)."""

    def __init__(self, dim, slider=True, mid_ratio=0.25, lambda_init=0.5, stage2=None, alloc="", alloc_noise=0.3,
                 alloc_floor=0.5, soft="", soft_tau=0.3, lam_mode="learn", pen_scale=1.0, box_shape="auto", box_margin=0,
                 box_k=1):
        super().__init__()
        # soft: "" = CB-STE (xuoi argmax); "anneal" = xuoi soft khi train, giam tau, cuoi train + test argmax;
        #       "merge" = xuoi soft ca khi test (gop token co trong so, tau = soft_tau)
        self.soft, self.soft_tau = soft, soft_tau
        self.dim, self.use_slider, self.mid_ratio, self.lambda_init = dim, slider, mid_ratio, lambda_init
        self.alloc, self.alloc_noise, self.alloc_floor = alloc, alloc_noise, alloc_floor
        self.lam_mode, self.pen_scale, self.box_shape, self.box_margin, self.box_k = lam_mode, pen_scale, box_shape, box_margin, box_k
        if not slider:
            self.slider = None
        elif alloc == "box":
            self.slider = BoxSelectSlider(dim, mid_ratio, lambda_init, noise=alloc_noise)
        elif alloc == "seg":
            self.slider = SegBudgetSlider(dim, mid_ratio, lambda_init, noise=alloc_noise, floor=alloc_floor)
        else:
            self.slider = CuboidSlider(dim, mid_ratio, lambda_init, lam_mode, pen_scale, box_shape, box_margin, box_k)
        self.stage2 = AnchoredResampler(**stage2) if stage2 else None

    def config(self):
        return {"hds": True, "dim": self.dim, "slider": self.use_slider, "mid_ratio": self.mid_ratio,
                "lambda_init": self.lambda_init, "stage2": self.stage2.config() if self.stage2 is not None else None,
                "alloc": self.alloc, "alloc_noise": self.alloc_noise, "alloc_floor": self.alloc_floor,
                "soft": self.soft, "soft_tau": self.soft_tau, "lam_mode": self.lam_mode, "pen_scale": self.pen_scale,
                "box_shape": self.box_shape, "box_margin": self.box_margin, "box_k": self.box_k}

    def forward(self, X, H, W, retention, tau=None, use_stage2=True, soft=None):
        """X [T, N, D] float32. Tra ve (T2 [K, D], layout co them kept_idx, stats).
        soft/tau = None (eval): theo config (merge -> soft voi soft_tau, con lai argmax)."""
        T, N, D = X.shape
        if soft is None:
            soft = self.soft == "merge"
        if tau is None:
            tau = self.soft_tau if soft else 1.0
        if self.slider is not None:
            extra = {"soft": True} if soft else {}
            kept, anchors, stats, mult = self.slider(X, H, W, retention, tau, **extra)
        else:
            kept, mult = uniform_indices(T * N, retention, X.device), None
            anchors, stats = X.reshape(T * N, D)[kept], {"hop_rate": 0.0}
        if self.stage2 is not None and use_stage2:
            T2, layout = self.stage2(X, kept, H, W, anchors=anchors, retention=retention)
            stats["raer"] = float((T2 - anchors).norm(dim=-1).mean() / anchors.norm(dim=-1).mean())
        else:
            T2, layout = anchors, unit_layout(kept, anchors, T, N)
        if mult is not None:   # phan bo thich ung: he so straight-through (xuoi = 1) day gradient vao diem phan bo
            T2 = T2 * mult.unsqueeze(-1)
        layout["kept_idx"] = kept
        return T2, layout, stats


def load_policy(path, device):
    """Nap checkpoint RAP cu (AnchoredResampler) hoac HDSPolicy."""
    ckpt = torch.load(path, map_location=device)
    cfg = dict(ckpt["config"])
    if cfg.pop("hds", False):
        m = HDSPolicy(**cfg)
    else:
        m = AnchoredResampler(**cfg)
    m = m.to(device).float()
    m.load_state_dict(ckpt["state_dict"])
    return m, ckpt
