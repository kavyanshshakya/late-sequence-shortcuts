"""Custom Mamba (selective SSM) model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseModel


class SSMBlock(nn.Module):
    def __init__(self, dm, ds=16, dc=4, ex=2, dropout=0.1):
        super().__init__()
        self.ds = ds
        di = dm * ex
        self.ip = nn.Linear(dm, di * 2, bias=False)
        self.cv = nn.Conv1d(di, di, dc, padding=dc - 1, groups=di)
        self.xp = nn.Linear(di, ds * 2 + 1, bias=False)
        self.A = nn.Parameter(torch.randn(di, ds))
        self.D = nn.Parameter(torch.ones(di))
        self.op = nn.Linear(di, dm, bias=False)
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        B, L, _ = x.shape
        xz = self.ip(x)
        xp, z = xz.chunk(2, dim=-1)
        xc = F.silu(self.cv(xp.transpose(1, 2))[:, :, :L].transpose(1, 2))
        proj = self.xp(xc)
        Bm = proj[..., :self.ds]
        C = proj[..., self.ds:2 * self.ds]
        dt = F.softplus(proj[..., -1])
        A = -torch.exp(self.A.float())
        h = torch.zeros(B, xc.size(-1), self.ds, device=x.device)
        outs = []
        for t in range(L):
            dt_t = dt[:, t].unsqueeze(-1).unsqueeze(-1)
            h = h * torch.exp(A * dt_t) + xc[:, t].unsqueeze(-1) * Bm[:, t].unsqueeze(1) * dt_t
            outs.append((h * C[:, t].unsqueeze(1)).sum(-1) + self.D * xc[:, t])
        return self.do(self.op(torch.stack(outs, 1) * F.silu(z)))


class MambaModel(BaseModel):
    def __init__(self, vocab_size, d_model=64, n_layers=2, n_classes=2,
                 dropout=0.1, max_seq_len=320, pad_id=0, n_heads=4, d_ff=256):
        super().__init__(vocab_size, d_model, n_layers, n_classes, dropout,
                         max_seq_len, pad_id)
        self.bs = nn.ModuleList([
            SSMBlock(d_model, dropout=dropout) for _ in range(n_layers)
        ])
        self.ns = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.fn = nn.LayerNorm(d_model)

    def encode(self, ids, mask):
        x = self.emb_do(self.emb(ids))
        inters = {"l0": x.detach()}
        for i, (b, n) in enumerate(zip(self.bs, self.ns)):
            x = x + b(n(x))
            inters[f"l{i+1}"] = x.detach()
        x = self.fn(x)
        m = mask.unsqueeze(-1).float()
        return (x * m).sum(1) / m.sum(1).clamp(min=1), inters
