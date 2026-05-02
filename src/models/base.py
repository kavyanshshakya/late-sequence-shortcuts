"""Base model class shared by all architectures."""

import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, vs, dm, nl, nc=2, do=0.1, sl=320, pi=0):
        super().__init__()
        self.emb = nn.Embedding(vs, dm, padding_idx=pi)
        self.emb_do = nn.Dropout(do)
        self.clf = nn.Sequential(
            nn.Linear(dm, dm), nn.GELU(), nn.Dropout(do), nn.Linear(dm, nc),
        )

    def forward(self, ids, mask, return_intermediates=False):
        p, inters = self.encode(ids, mask)
        out = {"logits": self.clf(p)}
        if return_intermediates:
            out["intermediates"] = inters
        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
