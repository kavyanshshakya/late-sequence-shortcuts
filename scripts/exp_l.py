#!/usr/bin/env python3
"""
Experiment L: Pretrained Mamba backbone cliff test (self-contained)
===================================================================
Tests whether the position cliff from Experiment H transfers to a pretrained
sequential model. The paper's main experiments train from scratch; a natural
question is whether the vulnerability is architectural or emerges from
task-specific training dynamics.

Approach:
  1. Load pretrained state-spaces/mamba-130m-hf and freeze the backbone.
  2. Train a two-layer linear classifier head on 500 clean ToM examples.
  3. Evaluate the frozen-backbone + trained-head model against shortcuts at
     nine positions {0, 25, 50, 75, 80, 85, 90, 95, 100}%.

Finding: the cliff does NOT transfer. Across positions 25–100% the shortcut
gap is roughly flat at ≈−3%, with no sharp transition at 95–100%. Position 0%
is noisy (std 15.7) due to sequence-start edge effects and is reported but
not interpreted. This scopes the cliff to task-specific training dynamics
rather than architectural priors alone.

Seeds: [42, 137, 256, 789, 1024]. Five training runs (one per seed; each run
trains the classifier head and evaluates at all nine positions).
Expected runtime: ≈1–2 hours on a T4 GPU (frozen backbone keeps cost low).

Output: results/exp_l.json (matches results/provided/exp_l.json schema; will
not overwrite the canonical results).

Requirements: torch, transformers.
"""

import os, sys, json, random, math, time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# =====================================================================
# CONFIG
# =====================================================================
SEEDS_FULL = [42, 137, 256, 789, 1024]
SEEDS_SMOKE = [42]
POSITIONS = [0.0, 0.25, 0.50, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]

PROTOCOL = {
    "n_train": 500, "n_val": 200, "n_test": 300,
    "max_seq_len": 320, "batch_size": 8,  # Smaller batch for pretrained model
    "max_epochs": 30,  # Fewer epochs — head only
    "lr": 5e-4, "weight_decay": 0.01,
    "dropout": 0.1, "grad_clip": 1.0,
    "patience": 10, "eval_every": 3,
}

# Pretrained model identifier
MODEL_ID = "state-spaces/mamba-130m-hf"

# =====================================================================
# TOKENIZER (from paper)
# =====================================================================
PERS = [f"n{i}" for i in range(20)]
OBJS = [f"o{i}" for i in range(15)]
LOCS = [f"l{i}" for i in range(12)]


@dataclass
class Example:
    premises: str
    question: str
    answer: str
    label: int
    shortcut_available: bool = False
    shortcut_cue: Optional[str] = None
    reasoning_type: Optional[str] = None
    n_distractors: Optional[int] = None


# =====================================================================
# GENERATOR (identical to paper)
# =====================================================================
class HardToMGenerator:
    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self._train = PERS[:16]

    def _dist(self, persons, obj):
        temps = [
            f"{self.rng.choice(persons)} enters the room",
            f"{self.rng.choice(persons)} talks to {self.rng.choice(persons)}",
            f"{self.rng.choice(persons)} leaves the room",
            f"{self.rng.choice(persons)} sees {self.rng.choice(OBJS)}",
            f"{self.rng.choice(persons)} puts the {self.rng.choice(OBJS)} in {self.rng.choice(LOCS)}",
        ]
        return self.rng.choice(temps)

    def _build_parts(self, pool):
        pl = self.rng.sample(list(pool), min(3, len(pool)))
        p1 = pl[0]; p2 = pl[1] if len(pl) > 1 else pl[0]
        obj = self.rng.choice(OBJS)
        ls = self.rng.sample(LOCS, min(4, len(LOCS)))
        la, lb, lc = ls[0], ls[1], ls[2]
        ld = ls[3] if len(ls) > 3 else ls[0]
        sc_type = self.rng.randint(0, 4)
        if sc_type == 0:
            parts = [f"{p1} puts the {obj} in {la}", f"the {obj} moves to {lb}",
                f"{p1} sees the {obj} in {lb}", f"{p1} leaves the room",
                f"the {obj} moves to {lc}", f"{p1} returns to the room"]
            question = f"where does {p1} think the {obj} is ?"
            answer = lb; label = 0; rt = "2move_fb"
        elif sc_type == 1:
            parts = [f"{p1} puts the {obj} in {la}", f"the {obj} moves to {lb}",
                f"{p1} sees the {obj} in {lb}", f"{p1} leaves the room",
                f"the {obj} moves to {lc}", f"the {obj} moves to {ld}",
                f"{p1} returns to the room"]
            question = f"where does {p1} think the {obj} is ?"
            answer = lb; label = 0; rt = "3move_fb"
        elif sc_type == 2:
            parts = [f"{p2} puts the {obj} in {la}",
                f"{p1} sees {p2} put the {obj} in {la}", f"{p2} leaves the room",
                f"the {obj} moves to {lb}", f"{p1} sees the {obj} in {lb}",
                f"{p2} returns to the room"]
            question = f"where does {p1} think {p2} think the {obj} is ?"
            answer = la; label = 0; rt = "2nd_order"
        elif sc_type == 3:
            parts = [f"{p1} puts the {obj} in {la}", f"the {obj} moves to {lb}",
                f"{p1} sees the {obj} in {lb}", f"the {obj} moves to {lc}",
                f"{p1} sees the {obj} in {lc}"]
            question = f"where does {p1} think the {obj} is ?"
            answer = lc; label = 1; rt = "true_2move"
        else:
            parts = [f"{p1} puts the {obj} in {la}", f"{p1} leaves the room",
                f"{p2} moves the {obj} to {lb}", f"{p1} returns to the room"]
            question = f"where does {p1} think the {obj} is ?"
            answer = la; label = 0; rt = "classic_fb"
        return parts, question, answer, label, rt, pl, obj

    def generate_clean(self, n, dist_range=(4, 8)):
        out = []
        for _ in range(n):
            parts, question, answer, label, rt, pl, obj = self._build_parts(self._train)
            nd = self.rng.randint(dist_range[0], dist_range[1])
            for _ in range(nd):
                pos = self.rng.randint(1, max(1, len(parts) - 1))
                parts.insert(pos, self._dist(pl, obj))
            prem = " . ".join(parts) + " ."
            out.append(Example(premises=prem, question=question, answer=answer,
                label=label, shortcut_available=False, reasoning_type=rt,
                n_distractors=nd))
        self.rng.shuffle(out)
        return out

    def generate_shortcut_at_fraction(self, n, correlation=0.9,
                                       position_frac=1.0, dist_range=(4, 8)):
        assert 0.0 <= position_frac <= 1.0
        out = []
        for _ in range(n):
            parts, question, answer, label, rt, pl, obj = self._build_parts(self._train)
            nd = self.rng.randint(dist_range[0], dist_range[1])
            for _ in range(nd):
                pos = self.rng.randint(1, max(1, len(parts) - 1))
                parts.insert(pos, self._dist(pl, obj))
            if self.rng.random() < correlation:
                s = f"the {answer} is near the room"
                insert_idx = int(round(position_frac * len(parts)))
                insert_idx = max(0, min(insert_idx, len(parts)))
                parts.insert(insert_idx, s)
                cue = f"hint_frac{int(position_frac*100)}"
            else:
                cue = f"no_hint_frac{int(position_frac*100)}"
            prem = " . ".join(parts) + " ."
            out.append(Example(premises=prem, question=question, answer=answer,
                label=label, shortcut_available=True, shortcut_cue=cue,
                reasoning_type=rt, n_distractors=nd))
        self.rng.shuffle(out)
        return out


# =====================================================================
# DATASET using HuggingFace tokenizer
# =====================================================================
class ReasoningDataset(Dataset):
    def __init__(self, examples, tokenizer, max_seq_len=320):
        self.ex = examples
        self.tok = tokenizer
        self.msl = max_seq_len

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        e = self.ex[i]
        text = e.premises + " " + e.question + " [answer:" + e.answer + "]"
        enc = self.tok(text, max_length=self.msl, padding="max_length",
                       truncation=True, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(e.label, dtype=torch.long),
        }


# =====================================================================
# PRETRAINED MAMBA + CLASSIFIER HEAD
# =====================================================================
class PretrainedMambaClassifier(nn.Module):
    """Frozen pretrained Mamba + trainable linear classifier on mean-pooled hidden states."""

    def __init__(self, model_id=MODEL_ID, n_classes=2, freeze_backbone=True):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        self.backbone = AutoModel.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, n_classes),
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # mean-pool hidden states over sequence (respecting attention mask)
        hidden = out.last_hidden_state  # (B, L, D)
        m = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * m).sum(1) / m.sum(1).clamp(min=1)
        logits = self.classifier(pooled)
        return {"logits": logits}

    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =====================================================================
# TRAIN + EVAL
# =====================================================================
class Trainer:
    def __init__(self, model, tr_dl, vl_dl, epochs=30, lr=5e-4, patience=10, eval_every=3):
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.dev)
        self.tr_dl = tr_dl
        self.vl_dl = vl_dl
        # Only optimize the classifier head (backbone is frozen)
        self.opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr, weight_decay=PROTOCOL["weight_decay"]
        )
        self.epochs = epochs
        self.eval_every = eval_every
        self.patience = patience

    def train(self):
        best_val = -1
        best_state = None
        patience_ctr = 0
        for ep in range(1, self.epochs + 1):
            self.model.train()
            for batch in self.tr_dl:
                ids = batch["input_ids"].to(self.dev)
                mask = batch["attention_mask"].to(self.dev)
                lab = batch["labels"].to(self.dev)
                out = self.model(ids, mask)
                loss = F.cross_entropy(out["logits"], lab)
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    PROTOCOL["grad_clip"]
                )
                self.opt.step()
            if ep % self.eval_every == 0:
                val_acc = evaluate(self.model, self.vl_dl, self.dev) * 100
                if val_acc > best_val:
                    best_val = val_acc
                    best_state = {k: v.detach().clone()
                                  for k, v in self.model.classifier.state_dict().items()}
                    patience_ctr = 0
                else:
                    patience_ctr += self.eval_every
                    if patience_ctr >= self.patience:
                        break
        if best_state is not None:
            self.model.classifier.load_state_dict(best_state)
        return {"best_val": best_val}


@torch.no_grad()
def evaluate(model, dl, dev):
    model.eval()
    correct = 0; total = 0
    for batch in dl:
        ids = batch["input_ids"].to(dev)
        mask = batch["attention_mask"].to(dev)
        lab = batch["labels"].to(dev)
        pred = model(ids, mask)["logits"].argmax(-1)
        correct += (pred == lab).sum().item()
        total += lab.size(0)
    return correct / max(1, total)


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# =====================================================================
# RUN PIPELINE
# =====================================================================
def run_one_config(seed, n_epochs=None):
    """Train classifier head on clean data, then evaluate against shortcuts at each position."""
    sl, bs = PROTOCOL["max_seq_len"], PROTOCOL["batch_size"]
    epochs = n_epochs if n_epochs is not None else PROTOCOL["max_epochs"]

    set_seed(seed)
    model = PretrainedMambaClassifier(freeze_backbone=True)
    tokenizer = model.tokenizer

    gen = HardToMGenerator(seed=seed)
    train_ex = gen.generate_clean(PROTOCOL["n_train"])
    val_ex = gen.generate_clean(PROTOCOL["n_val"])
    test_clean = gen.generate_clean(PROTOCOL["n_test"])

    tr_dl = DataLoader(ReasoningDataset(train_ex, tokenizer, sl), batch_size=bs, shuffle=True)
    vl_dl = DataLoader(ReasoningDataset(val_ex, tokenizer, sl), batch_size=bs)
    tc_dl = DataLoader(ReasoningDataset(test_clean, tokenizer, sl), batch_size=bs)

    # Train classifier head
    trainer = Trainer(model, tr_dl, vl_dl, epochs=epochs, lr=PROTOCOL["lr"],
                      patience=PROTOCOL["patience"], eval_every=PROTOCOL["eval_every"])
    trainer.train()

    clean_acc = round(evaluate(model, tc_dl, trainer.dev) * 100, 2)

    # Evaluate at all 9 positions
    position_results = {}
    for frac in POSITIONS:
        set_seed(seed + 5000)
        gen_t = HardToMGenerator(seed=seed + 5000)
        short_ex = gen_t.generate_shortcut_at_fraction(
            PROTOCOL["n_test"], correlation=0.9, position_frac=frac
        )
        short_dl = DataLoader(ReasoningDataset(short_ex, tokenizer, sl), batch_size=bs)
        short_acc = round(evaluate(model, short_dl, trainer.dev) * 100, 2)
        gap = round(short_acc - clean_acc, 2)
        position_results[f"frac_{int(frac*100)}"] = {"short_acc": short_acc, "gap": gap}

    return {
        "model": MODEL_ID,
        "seed": seed,
        "trainable_params": model.count_trainable_parameters(),
        "clean_acc": clean_acc,
        "positions": position_results,
    }


def smoke_test():
    print("=" * 70)
    print("  SMOKE TEST: 1 seed, 3 epochs, classifier head only")
    print("=" * 70)
    try:
        import transformers
        print(f"  transformers version: {transformers.__version__}")
    except ImportError:
        print("  ERROR: transformers not installed. Run: pip install transformers")
        return False

    t0 = time.time()
    result = run_one_config(42, n_epochs=3)
    dt = time.time() - t0
    print(f"\nSmoke completed in {dt:.1f}s")
    print(f"Clean: {result['clean_acc']:.2f}")
    for k, v in result["positions"].items():
        print(f"  {k}: gap={v['gap']:+.2f}")
    assert 0 <= result["clean_acc"] <= 100
    print("\n[OK] Smoke test passed.\n")
    return True


def full_run():
    print("=" * 70)
    print(f"  PRETRAINED MAMBA CLIFF: 5 seeds x 9 positions")
    print(f"  Model: {MODEL_ID}")
    print(f"  Seeds: {SEEDS_FULL}")
    print(f"  Training: classifier head only (backbone frozen)")
    print(f"  Expected: ~1-2 hours on T4")
    print("=" * 70)

    results = []
    t_start = time.time()
    for seed in SEEDS_FULL:
        print(f"\n--- SEED {seed} | elapsed {(time.time()-t_start)/60:.1f} min ---")
        t0 = time.time()
        r = run_one_config(seed)
        dt = time.time() - t0
        gaps_str = "  ".join(f"{k}:{v['gap']:+.1f}" for k, v in r["positions"].items())
        print(f"  clean={r['clean_acc']:.2f}  {gaps_str}  ({dt:.0f}s)")
        results.append(r)

    total = time.time() - t_start

    # Summary table
    print(f"\n{'='*70}")
    print(f"  RESULTS: Pretrained Mamba cliff ({total/60:.1f} min total)")
    print(f"{'='*70}")
    print(f"\n{'Position':<12} {'Mean gap':<15} {'Std':<10} {'n':<5}")
    print("-" * 45)
    for frac in POSITIONS:
        key = f"frac_{int(frac*100)}"
        gaps = [r["positions"][key]["gap"] for r in results]
        mean = np.mean(gaps)
        std = np.std(gaps, ddof=1)
        print(f"  {int(frac*100):>3}%       {mean:+5.2f}          {std:5.2f}     {len(gaps)}")

    print(f"\nCompare to paper's trained-from-scratch Mamba at 100%: -10.52 ± 4.06")
    print(f"If the pretrained cliff at 100% is similarly negative, hypothesis confirmed.")

    out = {
        "experiment": "pretrained_mamba_cliff",
        "model_id": MODEL_ID,
        "seeds": SEEDS_FULL,
        "positions_tested": [int(p*100) for p in POSITIONS],
        "approach": "Frozen backbone + trainable classifier head on clean ToM data, then evaluate against shortcuts at 9 positions.",
        "runs": results,
    }
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "exp_l.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if smoke_test():
        full_run()
