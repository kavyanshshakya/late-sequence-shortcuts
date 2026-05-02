#!/usr/bin/env python3
"""
Experiment P: When during training does the cliff emerge?
==========================================================
SELF-CONTAINED single-cell script. No imports from other files.
Uses the paper's own custom Mamba architecture (same as Exp H).

Question: Exp H shows the cliff in fully-trained Mamba (final-token gap
~ -10.5% at 100%). Exp M shows the representation cliff at the same position.
But WHEN during training does the cliff emerge? Does it:
  (a) Appear gradually as the model learns the task?
  (b) Emerge sharply at a specific epoch?
  (c) Get worse as training continues (cliff deepens over time)?
  (d) Appear early then stabilise?

Design: Train a custom Mamba on clean ToM data. At a set of checkpoint epochs,
pause training and evaluate shortcut gap at position 100% (and 0%/50% as
controls). Also measure clean accuracy to correlate with task-solving progress.

This connects to the critical-window observation from Figs 9 and 10 (bimodal
divergence and magic-seed dynamics emerge within the first 5-10 epochs) but
adds the cliff itself as a function of training progress, which no existing
experiment directly measures.

Seeds: [42, 137, 256].
Checkpoint epochs: [1, 3, 5, 7, 10, 15, 20, 30, 50].
Expected runtime: 30-50 minutes on T4 for all three seeds.

Usage on Kaggle or Colab (standalone, paste entire file into one cell):
  # no pip installs needed beyond torch (already present)
  <paste this file, then run>

If session disconnects, re-run - exp_p_checkpoint.json lets it resume.

Output: results/exp_p.json (matches results/provided/exp_p.json schema; will
not overwrite the canonical results).
"""

import os, json, random, math, time
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
SEEDS = [42, 137, 256]
CHECKPOINT_EPOCHS = [1, 3, 5, 7, 10, 15, 20, 30, 50]
EVAL_POSITIONS = [0.0, 0.50, 1.00]

PROTOCOL = {
    "n_train": 500, "n_val": 200, "n_test": 300,
    "max_seq_len": 320, "batch_size": 16,
    "max_epochs": 50,
    "lr": 5e-4,
    "weight_decay": 0.01,
    "dropout": 0.15,
    "grad_clip": 1.0,
    "warmup_steps": 200,
    "d_model": 96,                         # medium scale (matches Exp B/H)
    "n_layers": 4,
}

_RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(_RESULTS_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(_RESULTS_DIR, "exp_p_checkpoint.json")
OUTPUT_PATH = os.path.join(_RESULTS_DIR, "exp_p.json")

PERS = [f"n{i}" for i in range(20)]
OBJS = [f"o{i}" for i in range(15)]
LOCS = [f"l{i}" for i in range(12)]


# =====================================================================
# TOKENIZER
# =====================================================================
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
        return [1] + [self.t2i.get(w, 6) for w in text.strip().split()] + [2]

    @property
    def vocab_size(self):
        return self._n

    @property
    def pad_id(self):
        return 0


def make_tokenizer():
    tok = SimpleTokenizer()
    tok.add([
        "all", "are", "is", "a", "not", "yes", "no", ".", "?",
        "causes", "does", "cause", "puts", "the", "in", "leaves",
        "room", "moves", "to", "returns", "where", "think", "sees",
        "after", "and", "but", "some", "with", "also", "because",
        "since", "therefore", "so", "often", "together", "seen",
        "observed", "before", "saw", "left", "last", "knows",
        "enters", "talks", "happens", "when", "occurs", "near",
        "tells", "frequently",
    ])
    tok.add(PERS + OBJS + LOCS)
    return tok


# =====================================================================
# EXAMPLE + GENERATOR
# =====================================================================
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
# DATASET
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
        text = e.premises + " " + e.question
        ids = self.tok.encode(text)[:self.msl]
        pad_len = self.msl - len(ids)
        mask = [1] * len(ids) + [0] * pad_len
        ids = ids + [self.tok.pad_id] * pad_len
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(e.label, dtype=torch.long),
        }


# =====================================================================
# CUSTOM MAMBA (identical to paper's Exp H)
# =====================================================================
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


class MambaModel(nn.Module):
    def __init__(self, vocab_size, d_model=96, n_layers=4, n_classes=2,
                 dropout=0.15, max_seq_len=320, pad_id=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.emb_do = nn.Dropout(dropout)
        self.bs = nn.ModuleList([SSMBlock(d_model, dropout=dropout) for _ in range(n_layers)])
        self.ns = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.fn = nn.LayerNorm(d_model)
        self.clf = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, n_classes),
        )

    def forward(self, ids, mask):
        x = self.emb_do(self.emb(ids))
        for b, n in zip(self.bs, self.ns):
            x = x + b(n(x))
        x = self.fn(x)
        m = mask.unsqueeze(-1).float()
        pooled = (x * m).sum(1) / m.sum(1).clamp(min=1)
        return {"logits": self.clf(pooled)}

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =====================================================================
# TRAIN + EVAL
# =====================================================================
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


def evaluate_at_positions(model, tokenizer, dev, seed_offset, n_test, bs, sl):
    """Evaluate clean accuracy + shortcut gap at EVAL_POSITIONS."""
    model.eval()
    results = {}

    # Clean accuracy (shared across positions)
    set_seed(seed_offset + 9000)
    clean_gen = HardToMGenerator(seed=seed_offset + 9000)
    clean_ex = clean_gen.generate_clean(n_test)
    clean_dl = DataLoader(ReasoningDataset(clean_ex, tokenizer, sl), batch_size=bs)
    clean_acc = round(evaluate(model, clean_dl, dev) * 100, 2)

    for frac in EVAL_POSITIONS:
        set_seed(seed_offset + 9000)
        short_gen = HardToMGenerator(seed=seed_offset + 9000)
        short_ex = short_gen.generate_shortcut_at_fraction(
            n_test, correlation=0.9, position_frac=frac
        )
        short_dl = DataLoader(ReasoningDataset(short_ex, tokenizer, sl), batch_size=bs)
        short_acc = round(evaluate(model, short_dl, dev) * 100, 2)
        gap = round(short_acc - clean_acc, 2)
        results[f"frac_{int(frac*100)}"] = {
            "clean_acc": clean_acc, "short_acc": short_acc, "gap": gap,
        }
    return results, clean_acc


# =====================================================================
# ONE SEED WITH CHECKPOINTS
# =====================================================================
def run_one_seed(seed):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sl, bs = PROTOCOL["max_seq_len"], PROTOCOL["batch_size"]

    set_seed(seed)
    tokenizer = make_tokenizer()

    gen = HardToMGenerator(seed=seed)
    train_ex = gen.generate_clean(PROTOCOL["n_train"])
    val_ex = gen.generate_clean(PROTOCOL["n_val"])

    tr_dl = DataLoader(ReasoningDataset(train_ex, tokenizer, sl), batch_size=bs, shuffle=True)
    vl_dl = DataLoader(ReasoningDataset(val_ex, tokenizer, sl), batch_size=bs)

    model = MambaModel(
        vocab_size=tokenizer.vocab_size,
        d_model=PROTOCOL["d_model"],
        n_layers=PROTOCOL["n_layers"],
        dropout=PROTOCOL["dropout"],
        max_seq_len=sl,
        pad_id=tokenizer.pad_id,
    ).to(dev)

    print(f"  model params: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=PROTOCOL["lr"],
        weight_decay=PROTOCOL["weight_decay"],
    )
    total_steps = PROTOCOL["max_epochs"] * len(tr_dl)
    warmup_steps = PROTOCOL["warmup_steps"]
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    checkpoint_epochs_set = set(CHECKPOINT_EPOCHS)
    trajectory = []

    for ep in range(1, PROTOCOL["max_epochs"] + 1):
        model.train()
        train_loss = 0.0; n_batches = 0
        for batch in tr_dl:
            ids = batch["input_ids"].to(dev)
            mask = batch["attention_mask"].to(dev)
            lab = batch["labels"].to(dev)
            out = model(ids, mask)
            loss = F.cross_entropy(out["logits"], lab)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), PROTOCOL["grad_clip"])
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            n_batches += 1

        if ep in checkpoint_epochs_set:
            # Pause training, evaluate at all positions
            results, clean_acc = evaluate_at_positions(
                model, tokenizer, dev, seed, PROTOCOL["n_test"], bs, sl
            )
            val_acc = round(evaluate(model, vl_dl, dev) * 100, 2)
            entry = {
                "epoch": ep,
                "train_loss": round(train_loss / n_batches, 4),
                "val_acc": val_acc,
                "clean_acc": clean_acc,
                "positions": results,
            }
            trajectory.append(entry)
            gap_100 = results["frac_100"]["gap"]
            print(f"    epoch {ep:3d}  loss={train_loss/n_batches:.4f}  "
                  f"val={val_acc:.2f}%  clean={clean_acc:.2f}%  "
                  f"gap@100%={gap_100:+.2f}%")

    return {
        "arch": "mamba",
        "seed": seed,
        "params": model.count_parameters(),
        "checkpoint_epochs": CHECKPOINT_EPOCHS,
        "trajectory": trajectory,
    }


# =====================================================================
# MAIN
# =====================================================================
def main():
    print("=" * 70)
    print("  EXPERIMENT P: Cliff emergence during training")
    print("=" * 70)
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Checkpoint epochs: {CHECKPOINT_EPOCHS}")
    print(f"  Eval positions: {[int(p*100) for p in EVAL_POSITIONS]}%")
    print(f"  Reference: Exp H from-scratch @ 100% (5 seeds, full train): -10.52 ± 4.06")
    print()

    results = {
        "description": (
            "Experiment P: Measures when during training the cliff emerges. "
            "Custom Mamba (medium scale) trained on clean ToM data for 50 epochs, "
            "with evaluation at 9 checkpoint epochs. At each checkpoint, measures "
            "clean accuracy and shortcut gap at positions 0%, 50%, 100%. "
            "Answers whether the cliff emerges gradually or sharply, and its "
            "relationship to the critical window (Figs 9, 10)."
        ),
        "seeds": SEEDS,
        "checkpoint_epochs": CHECKPOINT_EPOCHS,
        "eval_positions": EVAL_POSITIONS,
        "reference_exp_h_frac_100_final": {"mean": -10.52, "std": 4.06},
        "protocol": PROTOCOL,
        "runs": [],
    }

    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH) as f:
                checkpoint = json.load(f)
            results["runs"] = checkpoint.get("runs", [])
            print(f"[resume] loaded {len(results['runs'])} runs from {CHECKPOINT_PATH}")
        except Exception as e:
            print(f"[resume] could not load: {e}")

    completed = {r["seed"] for r in results["runs"]}

    for seed in SEEDS:
        if seed in completed:
            print(f"[skip] seed {seed} already done")
            continue

        print(f"\n{'=' * 60}\n  SEED {seed}\n{'=' * 60}")
        t0 = time.time()
        try:
            run = run_one_seed(seed)
            results["runs"].append(run)

            with open(CHECKPOINT_PATH, "w") as f:
                json.dump(results, f, indent=2)

            elapsed = (time.time() - t0) / 60
            print(f"[done] seed {seed} in {elapsed:.1f} min")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"[error] seed {seed}: {e}")
            import traceback; traceback.print_exc()

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # ---- Summary: gap at 100% as a function of epoch, averaged over seeds ----
    print("\n" + "=" * 70)
    print("  CLIFF EMERGENCE TRAJECTORY (gap at 100% position)")
    print("=" * 70)
    print(f"  {'epoch':>6}  {'clean_acc':>10}  {'gap@0%':>10}  {'gap@50%':>10}  {'gap@100%':>11}")
    for ep in CHECKPOINT_EPOCHS:
        gaps_100 = []; gaps_50 = []; gaps_0 = []; cleans = []
        for run in results["runs"]:
            for entry in run["trajectory"]:
                if entry["epoch"] == ep:
                    gaps_100.append(entry["positions"]["frac_100"]["gap"])
                    gaps_50.append(entry["positions"]["frac_50"]["gap"])
                    gaps_0.append(entry["positions"]["frac_0"]["gap"])
                    cleans.append(entry["clean_acc"])
        if gaps_100:
            c_m = np.mean(cleans)
            g0_m = np.mean(gaps_0); g0_s = np.std(gaps_0, ddof=1) if len(gaps_0) > 1 else 0
            g50_m = np.mean(gaps_50); g50_s = np.std(gaps_50, ddof=1) if len(gaps_50) > 1 else 0
            g100_m = np.mean(gaps_100); g100_s = np.std(gaps_100, ddof=1) if len(gaps_100) > 1 else 0
            print(f"  {ep:>6}  {c_m:>9.2f}%  {g0_m:>+6.2f}±{g0_s:.2f}  "
                  f"{g50_m:>+6.2f}±{g50_s:.2f}  {g100_m:>+7.2f}±{g100_s:.2f}")

    print(f"\n  Output: {OUTPUT_PATH}")
    print()
    print("  Interpretation guide:")
    print("    If gap@100% stays near 0 while clean_acc rises, then cliff emerges")
    print("    AFTER the task is solved. If gap@100% tracks clean_acc closely, the")
    print("    cliff is a side-effect of task learning. If gap@100% appears at a")
    print("    specific epoch range (e.g. 5-10), this identifies a cliff-emergence")
    print("    window that can be related to the training dynamics in Figs 9, 10.")


if __name__ == "__main__":
    main()
