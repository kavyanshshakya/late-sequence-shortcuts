"""Bidirectional LSTM encoder model."""

import torch.nn as nn
from .base import BaseModel


class LSTMModel(BaseModel):
    def __init__(self, vocab_size, d_model=64, n_layers=2, n_classes=2,
                 dropout=0.1, max_seq_len=320, pad_id=0, n_heads=4, d_ff=256):
        super().__init__(vocab_size, d_model, n_layers, n_classes, dropout,
                         max_seq_len, pad_id)
        hs = d_model // 2
        self.ls = nn.ModuleList([
            nn.LSTM(d_model, hs, 1, batch_first=True, bidirectional=True)
            for _ in range(n_layers)
        ])
        self.ns = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.ds = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])

    def encode(self, ids, mask):
        x = self.emb_do(self.emb(ids))
        lengths = mask.sum(1).cpu()
        inters = {"l0": x.detach()}
        for i, (l, n, d) in enumerate(zip(self.ls, self.ns, self.ds)):
            pk = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False)
            o, _ = l(pk)
            xo, _ = nn.utils.rnn.pad_packed_sequence(
                o, batch_first=True, total_length=ids.size(1))
            x = n(x + d(xo)) if xo.size(-1) == x.size(-1) else n(d(xo))
            inters[f"l{i+1}"] = x.detach()
        m = mask.unsqueeze(-1).float()
        return (x * m).sum(1) / m.sum(1).clamp(min=1), inters
