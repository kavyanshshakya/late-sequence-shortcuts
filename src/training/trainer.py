"""Training loop with early stopping."""

import time
import torch
import torch.nn as nn
from .scheduler import CosineWarmupScheduler


class Trainer:
    def __init__(self, model, tr, vl, tc, ts, tm=None, epochs=150,
                 lr=5e-4, device="auto", eval_every=5, patience=30):
        if device == "auto":
            if torch.cuda.is_available():
                self.dev = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.dev = torch.device("mps")
            else:
                self.dev = torch.device("cpu")
        else:
            self.dev = torch.device(device)

        self.model = model.to(self.dev)
        self.tr = tr
        self.vl = vl
        self.tc = tc
        self.ts = ts
        self.tm = tm
        self.epochs = epochs
        self.ee = eval_every
        self.pat = patience
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        self.sch = CosineWarmupScheduler(self.opt, 200, epochs * len(tr))
        self.crit = nn.CrossEntropyLoss()
        self.history = []
        self.bv = 0
        self.w = 0

    @torch.no_grad()
    def _ev(self, dl):
        if dl is None:
            return 0.0
        self.model.eval()
        c = t = 0
        for b in dl:
            ids = b["input_ids"].to(self.dev)
            lab = b["labels"].to(self.dev)
            mask = b["attention_mask"].to(self.dev)
            p = self.model(ids, mask)["logits"].argmax(-1)
            c += (p == lab).sum().item()
            t += lab.size(0)
        return c / max(t, 1)

    def train(self):
        for ep in range(1, self.epochs + 1):
            self.model.train()
            tl = n = 0
            for b in self.tr:
                ids = b["input_ids"].to(self.dev)
                lab = b["labels"].to(self.dev)
                mask = b["attention_mask"].to(self.dev)
                loss = self.crit(self.model(ids, mask)["logits"], lab)
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.sch.step()
                tl += loss.item()
                n += 1
            al = tl / max(n, 1)

            if ep % self.ee == 0 or ep == 1 or ep == self.epochs:
                v = self._ev(self.vl)
                tc = self._ev(self.tc)
                ts = self._ev(self.ts)
                tm = self._ev(self.tm)
                gap = ts - tc
                self.history.append({
                    "epoch": ep, "loss": al, "val": v,
                    "test_clean": tc, "test_shortcut": ts,
                    "test_memo": tm, "gap": gap,
                })
                if v > self.bv:
                    self.bv = v
                    self.w = 0
                else:
                    self.w += self.ee
                    if self.w >= self.pat:
                        break

        return self.history[-1] if self.history else {}
