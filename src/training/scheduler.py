"""Cosine schedule with warmup."""

import math


class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.opt = optimizer
        self.wu = warmup_steps
        self.total = total_steps
        self.mn = min_lr
        self.base_lrs = [p["lr"] for p in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.wu:
            scale = self.step_count / max(self.wu, 1)
        else:
            progress = (self.step_count - self.wu) / max(self.total - self.wu, 1)
            scale = 0.5 * (1 + math.cos(math.pi * progress))
        for pg, base in zip(self.opt.param_groups, self.base_lrs):
            pg["lr"] = max(base * scale, self.mn)
