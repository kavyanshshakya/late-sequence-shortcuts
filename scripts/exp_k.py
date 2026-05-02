#!/usr/bin/env python3
"""
Experiment K: Unidirectional LSTM Control (self-contained)
==========================================================
Tests whether the end-position vulnerability in LSTM depends on bidirectionality.
Same architecture parameters and training protocol as the bidirectional LSTM in
Experiments A/B/H, but with bidirectional=False.

Reference (bidirectional LSTM at medium scale, Exp H, 5 seeds):
  end  −11.00 ± 9.34   (from 9-position sweep)
  begin   0.00 ± 0.00

Result (unidirectional LSTM, 10 seeds):
  end  −3.77 ± 4.99
  begin −0.03 ± 0.33
  Welch's t = 1.62, p = 0.16 (not significantly different from bidirectional)

10 seeds: [42, 137, 256, 789, 1024, 2048, 4096, 8192, 16384, 32768]. First five
match Exp H for direct comparability; next five are fresh powers-of-2 seeds.
Per seed: one training run plus end-position and begin-position evaluations.
Expected runtime: ~10 min on T4.

Output: exp_k.json (matches the schema of results/provided/exp_k.json).
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
SEEDS_FULL = [42, 137, 256, 789, 1024, 2048, 4096, 8192, 16384, 32768]
SEEDS_SMOKE = [42]
POSITIONS = ["end", "begin"]

PROTOCOL = {
    "n_train": 500, "n_val": 200, "n_test": 300,
    "max_seq_len": 320, "batch_size": 16,
    "max_epochs": 150, "lr": 5e-4, "weight_decay": 0.01,
    "dropout": 0.15, "grad_clip": 1.0,
    "warmup_steps": 200, "patience": 30, "eval_every": 5,
}

SCALE_CFG = {"d_model": 96, "n_layers": 4, "n_heads": 4, "d_ff": 384}

# =====================================================================
# TOKENIZER + DATA (identical to paper codebase)
# =====================================================================
PERS = [f"n{i}" for i in range(20)]
OBJS = [f"o{i}" for i in range(15)]
LOCS = [f"l{i}" for i in range(12)]

class SimpleTokenizer:
    SPECIAL = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<sep>": 3,
               "<cot>": 4, "<ans>": 5, "<unk>": 6}
    def __init__(self):
        self.t2i = dict(self.SPECIAL)
        self.i2t = {v: k for k, v in self.t2i.items()}
        self._n = len(self.SPECIAL)
    def add(self, tokens):
        for t in (tokens if isinstance(tokens, list) else [tokens]):
            if t not in self.t2i:
                self.t2i[t] = self._n
                self.i2t[self._n] = t
                self._n += 1
    def encode(self, text):
        ids = [self.t2i["<bos>"]]
        for tok in text.split():
            ids.append(self.t2i.get(tok, self.t2i["<unk>"]))
        ids.append(self.t2i["<eos>"])
        return ids
    @property
    def vocab_size(self): return self._n
    @property
    def pad_id(self): return self.t2i["<pad>"]

def make_tokenizer():
    tok = SimpleTokenizer()
    tok.add(PERS); tok.add(OBJS); tok.add(LOCS)
    vocab_words = [
        "enters","leaves","returns","talks","to","the","room","puts","in","sees",
        "moves","near","is","are","where","does","think","on","at","and","a","an",
        "of","with","from","by","for","as","it","its","this","that","has","have",
        "had","was","were","be","been","being","do","did","done","will","would",
        "can","could","should","may","might","must","shall","which","what","who",
        "when","why","how","?",".",",","!",";",":","'","\"","(",")","-","_","/",
        "object","person","location","thinks","thought","knows","know","known","put",
    ]
    tok.add(vocab_words)
    return tok

@dataclass
class Example:
    premises: str
    question: str
    answer: str
    label: int
    cot: Optional[str] = None
    shortcut_available: bool = False
    shortcut_cue: Optional[str] = None
    reasoning_type: Optional[str] = None
    n_distractors: Optional[int] = None

class ReasoningDataset(Dataset):
    def __init__(self, examples, tokenizer, max_seq_len=320):
        self.ex = examples; self.tok = tokenizer; self.msl = max_seq_len
    def __len__(self): return len(self.ex)
    def __getitem__(self, i):
        e = self.ex[i]
        text = e.premises + " " + e.question
        ids = self.tok.encode(text)[:self.msl]
        pad = self.msl - len(ids)
        mask = [1] * len(ids) + [0] * pad
        ids = ids + [self.tok.pad_id] * pad
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(e.label, dtype=torch.long),
        }

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

    def generate_shortcut_at_position(self, n, correlation=0.9,
                                       position="end", dist_range=(4, 8)):
        assert position in ("begin", "end")
        out = []
        for _ in range(n):
            parts, question, answer, label, rt, pl, obj = self._build_parts(self._train)
            nd = self.rng.randint(dist_range[0], dist_range[1])
            for _ in range(nd):
                pos = self.rng.randint(1, max(1, len(parts) - 1))
                parts.insert(pos, self._dist(pl, obj))
            if self.rng.random() < correlation:
                s = f"the {answer} is near the room"
                if position == "begin":
                    parts.insert(0, s)
                else:
                    parts.append(s)
                cue = f"hint_{position}"
            else:
                cue = f"no_hint_{position}"
            prem = " . ".join(parts) + " ."
            out.append(Example(premises=prem, question=question, answer=answer,
                label=label, shortcut_available=True, shortcut_cue=cue,
                reasoning_type=rt, n_distractors=nd))
        self.rng.shuffle(out)
        return out

# =====================================================================
# UNIDIRECTIONAL LSTM MODEL (key difference: bidirectional=False)
# =====================================================================
class UnidirectionalLSTMModel(nn.Module):
    def __init__(self, vocab_size, d_model=96, n_layers=4, n_classes=2,
                 dropout=0.1, max_seq_len=320, pad_id=0, **kwargs):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.emb_do = nn.Dropout(dropout)
        # KEY CHANGE: hidden size = d_model (not d_model//2),
        # single direction means we need full d_model to match param count roughly
        self.ls = nn.ModuleList([
            nn.LSTM(d_model, d_model, 1, batch_first=True, bidirectional=False)
            for _ in range(n_layers)
        ])
        self.ns = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.ds = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])
        self.clf = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(d_model, n_classes))

    def forward(self, ids, mask):
        x = self.emb_do(self.emb(ids))
        lengths = mask.sum(1).cpu()
        for l, n, d in zip(self.ls, self.ns, self.ds):
            pk = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            o, _ = l(pk)
            xo, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True, total_length=ids.size(1))
            x = n(x + d(xo)) if xo.size(-1) == x.size(-1) else n(d(xo))
        m = mask.unsqueeze(-1).float()
        p = (x * m).sum(1) / m.sum(1).clamp(min=1)
        return {"logits": self.clf(p)}

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# =====================================================================
# SCHEDULER + TRAINER
# =====================================================================
class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.opt = optimizer; self.wu = warmup_steps; self.total = total_steps
        self.mn = min_lr; self.base_lrs = [p["lr"] for p in optimizer.param_groups]
        self.step_count = 0
    def step(self):
        self.step_count += 1
        if self.step_count < self.wu:
            lr_scale = self.step_count / max(1, self.wu)
        else:
            progress = (self.step_count - self.wu) / max(1, self.total - self.wu)
            lr_scale = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
        for i, pg in enumerate(self.opt.param_groups):
            pg["lr"] = max(self.mn, self.base_lrs[i] * lr_scale)

class Trainer:
    def __init__(self, model, tr_dl, vl_dl, tc_dl, ts_dl, epochs=150,
                 lr=5e-4, eval_every=5, patience=30):
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.dev)
        self.tr_dl, self.vl_dl, self.tc_dl, self.ts_dl = tr_dl, vl_dl, tc_dl, ts_dl
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr,
            weight_decay=PROTOCOL["weight_decay"])
        total_steps = len(tr_dl) * epochs
        self.sched = CosineWarmupScheduler(self.opt, PROTOCOL["warmup_steps"], total_steps)
        self.epochs = epochs; self.eval_every = eval_every; self.patience = patience
    def train(self):
        best_val = -1; best_state = None; patience_ctr = 0
        for ep in range(1, self.epochs + 1):
            self.model.train()
            for batch in self.tr_dl:
                ids = batch["input_ids"].to(self.dev)
                mask = batch["attention_mask"].to(self.dev)
                lab = batch["labels"].to(self.dev)
                out = self.model(ids, mask)
                loss = F.cross_entropy(out["logits"], lab)
                self.opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), PROTOCOL["grad_clip"])
                self.opt.step(); self.sched.step()
            if ep % self.eval_every == 0:
                val_acc = evaluate(self.model, self.vl_dl, self.dev) * 100
                if val_acc > best_val:
                    best_val = val_acc
                    best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                    patience_ctr = 0
                else:
                    patience_ctr += self.eval_every
                    if patience_ctr >= self.patience:
                        break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return {"best_val": best_val}

@torch.no_grad()
def evaluate(model, dl, dev):
    model.eval(); correct = 0; total = 0
    for batch in dl:
        ids = batch["input_ids"].to(dev); mask = batch["attention_mask"].to(dev)
        lab = batch["labels"].to(dev)
        pred = model(ids, mask)["logits"].argmax(-1)
        correct += (pred == lab).sum().item(); total += lab.size(0)
    return correct / max(1, total)

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

# =====================================================================
# RUN CONFIG
# =====================================================================
def run_one(seed, tok, n_epochs=None):
    sl, bs = PROTOCOL["max_seq_len"], PROTOCOL["batch_size"]
    epochs = n_epochs if n_epochs is not None else PROTOCOL["max_epochs"]

    set_seed(seed)
    gen = HardToMGenerator(seed=seed)
    train_ex = gen.generate_clean(PROTOCOL["n_train"])
    val_ex = gen.generate_clean(PROTOCOL["n_val"])
    test_clean = gen.generate_clean(PROTOCOL["n_test"])
    dummy = gen.generate_shortcut_at_position(PROTOCOL["n_test"],
        correlation=0.9, position="end")

    tr_dl = DataLoader(ReasoningDataset(train_ex, tok, sl), batch_size=bs, shuffle=True)
    vl_dl = DataLoader(ReasoningDataset(val_ex, tok, sl), batch_size=bs)
    tc_dl = DataLoader(ReasoningDataset(test_clean, tok, sl), batch_size=bs)
    ts_dl = DataLoader(ReasoningDataset(dummy, tok, sl), batch_size=bs)

    set_seed(seed)
    model = UnidirectionalLSTMModel(vocab_size=tok.vocab_size,
        dropout=PROTOCOL["dropout"], max_seq_len=sl, **SCALE_CFG)
    params = model.count_parameters()

    trainer = Trainer(model, tr_dl, vl_dl, tc_dl, ts_dl,
        epochs=epochs, lr=PROTOCOL["lr"],
        eval_every=PROTOCOL["eval_every"], patience=PROTOCOL["patience"])
    trainer.train()

    clean_acc = round(evaluate(model, tc_dl, trainer.dev) * 100, 2)

    gaps = {}
    for position in POSITIONS:
        set_seed(seed + 3000)
        gen_t = HardToMGenerator(seed=seed + 3000)
        short_ex = gen_t.generate_shortcut_at_position(PROTOCOL["n_test"],
            correlation=0.9, position=position)
        short_dl = DataLoader(ReasoningDataset(short_ex, tok, sl), batch_size=bs)
        short_acc = round(evaluate(model, short_dl, trainer.dev) * 100, 2)
        gap = round(short_acc - clean_acc, 2)
        gaps[f"gap_{position}"] = gap
        gaps[f"short_acc_{position}"] = short_acc

    return {
        "arch": "unidirectional_lstm", "seed": seed, "scale": "medium", "params": params,
        "clean": clean_acc,
        "gap_end": gaps["gap_end"], "gap_begin": gaps["gap_begin"],
        "short_acc_end": gaps["short_acc_end"],
        "short_acc_begin": gaps["short_acc_begin"],
    }

def smoke_test():
    print("=" * 70)
    print("  SMOKE TEST: 1 seed, 5 epochs")
    print("=" * 70)
    tok = make_tokenizer()
    t0 = time.time()
    result = run_one(42, tok, n_epochs=5)
    dt = time.time() - t0
    print(f"\nSmoke completed in {dt:.1f}s")
    print(f"Result: {result}")
    assert 0 <= result["clean"] <= 100
    print("\n[OK] Smoke test passed.\n")

def full_run():
    print("=" * 70)
    print(f"  UNIDIRECTIONAL LSTM: 10 seeds x 2 positions")
    print(f"  Seeds: {SEEDS_FULL}")
    print(f"  Expected time: ~10 min on T4")
    print("=" * 70)

    tok = make_tokenizer()
    results = []
    t_start = time.time()

    for seed in SEEDS_FULL:
        print(f"\n--- SEED {seed} | elapsed {(time.time()-t_start)/60:.1f} min ---")
        t0 = time.time()
        r = run_one(seed, tok)
        dt = time.time() - t0
        print(f"  clean={r['clean']:.2f}  gap_end={r['gap_end']:+.2f}  "
              f"gap_begin={r['gap_begin']:+.2f}  ({dt:.0f}s)")
        results.append(r)

    total = time.time() - t_start

    print(f"\n{'='*70}")
    print(f"  RESULTS ({total/60:.1f} min total)")
    print(f"{'='*70}")
    print(f"{'Seed':<8} {'Clean':<8} {'Gap-End':<10} {'Gap-Begin':<10}")
    print("-" * 40)
    for r in results:
        print(f"{r['seed']:<8} {r['clean']:<8.2f} {r['gap_end']:<+10.2f} {r['gap_begin']:<+10.2f}")

    ends = [r["gap_end"] for r in results]
    begs = [r["gap_begin"] for r in results]
    print(f"\nUnidirectional LSTM 10-seed mean:")
    print(f"  end   = {np.mean(ends):+.2f} +/- {np.std(ends, ddof=1):.2f}")
    print(f"  begin = {np.mean(begs):+.2f} +/- {np.std(begs, ddof=1):.2f}")
    print(f"\nCompare to paper's BIDIRECTIONAL LSTM (Exp H 5-seed):")
    print(f"  end   = -11.00 +/- 9.34")
    print(f"  begin = +0.00 +/- 0.00")
    print(f"\nIf unidirectional end-gap is MORE negative, hypothesis confirmed.")

    out = {
        "description": "Experiment K: Unidirectional LSTM control (10 seeds, medium scale)",
        "experiment": "unidirectional_lstm_comparison",
        "scale": "medium",
        "seeds": SEEDS_FULL,
        "runs": results,
        "bidirectional_baseline_comparison": {
            "source": "Exp H, 5 seeds, medium scale",
            "end": {"mean": -11.00, "std": 9.34},
            "begin": {"mean": 0.00, "std": 0.00},
        },
    }
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "exp_k.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    smoke_test()
    full_run()
