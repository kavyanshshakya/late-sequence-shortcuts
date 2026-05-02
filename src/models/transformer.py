"""Transformer encoder model."""

import math
import torch
import torch.nn as nn
from .base import BaseModel


class TransformerModel(BaseModel):
    def __init__(self, vocab_size, d_model=64, n_layers=2, n_heads=4,
                 d_ff=256, n_classes=2, dropout=0.1, max_seq_len=320, pad_id=0):
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
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model, n_heads, d_ff, dropout, "gelu",
                batch_first=True, norm_first=True,
            )
            for _ in range(n_layers)
        ])
        self.fn = nn.LayerNorm(d_model)

    def encode(self, ids, mask):
        x = self.pe_do(self.emb_do(self.emb(ids)) + self.pe[:, :ids.size(1)])
        pm = (mask == 0)
        inters = {"l0": x.detach()}
        for i, layer in enumerate(self.layers):
            x = layer(x, src_key_padding_mask=pm)
            inters[f"l{i+1}"] = x.detach()
        x = self.fn(x)
        m = mask.unsqueeze(-1).float()
        return (x * m).sum(1) / m.sum(1).clamp(min=1), inters
