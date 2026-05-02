"""Mamba with positional encoding, and official mamba-ssm wrappers."""

import math
import torch
import torch.nn as nn
from .base import BaseModel
from .mamba_custom import SSMBlock


class MambaPosEncModel(BaseModel):
    def __init__(self, vocab_size, d_model=64, n_layers=2, n_classes=2,
                 dropout=0.1, max_seq_len=320, pad_id=0, n_heads=4, d_ff=256):
        super().__init__(vocab_size, d_model, n_layers, n_classes, dropout,
                         max_seq_len, pad_id)
        pe = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
        self.pe_do = nn.Dropout(dropout)
        self.bs = nn.ModuleList([
            SSMBlock(d_model, dropout=dropout) for _ in range(n_layers)
        ])
        self.ns = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.fn = nn.LayerNorm(d_model)

    def encode(self, ids, mask):
        x = self.pe_do(self.emb_do(self.emb(ids)) + self.pe[:, :ids.size(1)])
        inters = {"l0": x.detach()}
        for i, (b, n) in enumerate(zip(self.bs, self.ns)):
            x = x + b(n(x))
            inters[f"l{i+1}"] = x.detach()
        x = self.fn(x)
        m = mask.unsqueeze(-1).float()
        return (x * m).sum(1) / m.sum(1).clamp(min=1), inters


class OfficialMamba1Model(BaseModel):
    def __init__(self, vocab_size, d_model=64, n_layers=2, n_classes=2,
                 dropout=0.1, max_seq_len=320, pad_id=0, **kwargs):
        super().__init__(vocab_size, d_model, n_layers, n_classes, dropout,
                         max_seq_len, pad_id)
        from mamba_ssm import Mamba
        self.blocks = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.fn = nn.LayerNorm(d_model)

    def encode(self, ids, mask):
        x = self.emb_do(self.emb(ids))
        inters = {"l0": x.detach()}
        for i, (b, n) in enumerate(zip(self.blocks, self.norms)):
            x = x + b(n(x))
            inters[f"l{i+1}"] = x.detach()
        x = self.fn(x)
        m = mask.unsqueeze(-1).float()
        return (x * m).sum(1) / m.sum(1).clamp(min=1), inters


class OfficialMamba2Model(BaseModel):
    def __init__(self, vocab_size, d_model=64, n_layers=2, n_classes=2,
                 dropout=0.1, max_seq_len=320, pad_id=0, **kwargs):
        super().__init__(vocab_size, d_model, n_layers, n_classes, dropout,
                         max_seq_len, pad_id)
        from mamba_ssm import Mamba2
        headdim = 64
        assert d_model % headdim == 0
        self.blocks = nn.ModuleList([
            Mamba2(d_model=d_model, d_state=64, d_conv=4, expand=2, headdim=headdim)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.fn = nn.LayerNorm(d_model)

    def encode(self, ids, mask):
        x = self.emb_do(self.emb(ids))
        inters = {"l0": x.detach()}
        for i, (b, n) in enumerate(zip(self.blocks, self.norms)):
            x = x + b(n(x))
            inters[f"l{i+1}"] = x.detach()
        x = self.fn(x)
        m = mask.unsqueeze(-1).float()
        return (x * m).sum(1) / m.sum(1).clamp(min=1), inters
