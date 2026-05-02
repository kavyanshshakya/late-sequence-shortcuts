#!/usr/bin/env python3
"""
Experiment J: Shortcut Type Generalisation (self-contained)
===========================================================
Tests whether the attention-vs-sequential split is specific to one shortcut form
or generalises across shortcut types. Trains 3 models on clean data, then
evaluates each against 3 distinct shortcut types, all placed at end of sequence:

  Type 1 (semantic):  "the {answer_location} is near the room" (original)
  Type 2 (lexical):   answer token repeated 3x at end ("{loc} {loc} {loc}")
  Type 3 (arbitrary): fixed label-correlated nonsense tokens at end
                      label=0 → "k5 v2 e7 ."   label=1 → "k9 v4 e1 ."

If all three shortcut forms disrupt LSTM and Mamba but leave Transformer immune,
the attention-vs-sequential separation is shortcut-type-independent rather than
an artefact of the original semantic cue.

Total: 3 training runs + 9 evaluations (3 archs × 3 shortcut types), ≈20–30 min on T4.
Self-contained: depends only on torch, numpy, and Python stdlib.
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
# TOKENIZER
# =====================================================================
PERS = [f"n{i}" for i in range(20)]
OBJS = [f"o{i}" for i in range(15)]
LOCS = [f"l{i}" for i in range(12)]
CATS = [f"k{i}" for i in range(80)]
ENTS = [f"e{i}" for i in range(60)]
EVTS = [f"v{i}" for i in range(50)]

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
    def vocab_size(self): return self._n
    @property
    def pad_id(self): return 0

def make_tokenizer():
    tok = SimpleTokenizer()
    tok.add(["all","are","is","a","not","yes","no",".","?",
        "causes","does","cause","puts","the","in","leaves",
        "room","moves","to","returns","where","think","sees",
        "after","and","but","some","with","also","because",
        "since","therefore","so","often","together","seen",
        "observed","before","saw","left","last","knows",
        "enters","talks","happens","when","occurs","near",
        "tells","frequently"])
    tok.add(CATS + ENTS + EVTS + PERS + OBJS + LOCS)
    return tok

# =====================================================================
# DATASET
# =====================================================================
@dataclass
class Example:
    premises: str; question: str; answer: str; label: int
    cot: Optional[str]; shortcut_available: bool; reasoning_type: str
    n_hops: int; n_distractors: int; shortcut_cue: Optional[str] = None

class ReasoningDataset(Dataset):
    def __init__(self, examples, tokenizer, max_seq_len=320):
        self.examples = examples
        ids_l, labels, masks = [], [], []
        for ex in examples:
            text = f"{ex.premises} <sep> {ex.question} <ans> {ex.answer}"
            ids = tokenizer.encode(text)
            if len(ids) > max_seq_len:
                ids = ids[:max_seq_len - 1] + [2]
            m = [1] * len(ids)
            pad = max_seq_len - len(ids)
            ids += [0] * pad; m += [0] * pad
            ids_l.append(ids); labels.append(ex.label); masks.append(m)
        self.input_ids = torch.tensor(ids_l, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.masks = torch.tensor(masks, dtype=torch.long)
    def __len__(self): return len(self.examples)
    def __getitem__(self, i):
        return {"input_ids": self.input_ids[i], "labels": self.labels[i],
                "attention_mask": self.masks[i]}

# =====================================================================
# GENERATOR WITH MULTIPLE SHORTCUT TYPES
# =====================================================================
class MultiShortcutGenerator:
    """Generates ToM narratives with configurable shortcut types."""

    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self.tokenizer = make_tokenizer()
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

    def _make_story(self):
        """Generate one ToM story, return (parts, question, answer, label, rt, nh)."""
        pl = self.rng.sample(list(self._train), min(3, len(self._train)))
        p1, p2 = pl[0], pl[1] if len(pl) > 1 else pl[0]
        obj = self.rng.choice(OBJS)
        ls = self.rng.sample(LOCS, min(4, len(LOCS)))
        la, lb, lc = ls[0], ls[1], ls[2]
        ld = ls[3] if len(ls) > 3 else ls[0]
        nd = self.rng.randint(4, 8)
        sc_type = self.rng.randint(0, 4)

        if sc_type == 0:
            parts = [f"{p1} puts the {obj} in {la}", f"the {obj} moves to {lb}",
                f"{p1} sees the {obj} in {lb}", f"{p1} leaves the room",
                f"the {obj} moves to {lc}", f"{p1} returns to the room"]
            question = f"where does {p1} think the {obj} is ?"
            answer = lb; label = 0; rt = "2move_fb"; nh = 3
        elif sc_type == 1:
            parts = [f"{p1} puts the {obj} in {la}", f"the {obj} moves to {lb}",
                f"{p1} sees the {obj} in {lb}", f"{p1} leaves the room",
                f"the {obj} moves to {lc}", f"the {obj} moves to {ld}",
                f"{p1} returns to the room"]
            question = f"where does {p1} think the {obj} is ?"
            answer = lb; label = 0; rt = "3move_fb"; nh = 4
        elif sc_type == 2:
            parts = [f"{p2} puts the {obj} in {la}",
                f"{p1} sees {p2} put the {obj} in {la}", f"{p2} leaves the room",
                f"the {obj} moves to {lb}", f"{p1} sees the {obj} in {lb}",
                f"{p2} returns to the room"]
            question = f"where does {p1} think {p2} think the {obj} is ?"
            answer = la; label = 0; rt = "2nd_order"; nh = 4
        elif sc_type == 3:
            parts = [f"{p1} puts the {obj} in {la}", f"the {obj} moves to {lb}",
                f"{p1} sees the {obj} in {lb}", f"the {obj} moves to {lc}",
                f"{p1} sees the {obj} in {lc}"]
            question = f"where does {p1} think the {obj} is ?"
            answer = lc; label = 1; rt = "true_2move"; nh = 2
        else:
            parts = [f"{p1} puts the {obj} in {la}", f"{p1} leaves the room",
                f"{p2} moves the {obj} to {lb}", f"{p1} returns to the room"]
            question = f"where does {p1} think the {obj} is ?"
            answer = la; label = 0; rt = "classic_fb"; nh = 2

        for _ in range(nd):
            pos = self.rng.randint(1, max(1, len(parts) - 1))
            parts.insert(pos, self._dist(pl, obj))

        return parts, question, answer, label, rt, nh, nd

    def generate_clean(self, n):
        examples = []
        for _ in range(n):
            parts, question, answer, label, rt, nh, nd = self._make_story()
            prem = " . ".join(parts) + " ."
            examples.append(Example(premises=prem, question=question, answer=answer,
                label=label, cot=None, shortcut_available=False,
                reasoning_type=rt, n_hops=nh, n_distractors=nd))
        self.rng.shuffle(examples)
        return examples

    def generate_shortcut(self, n, shortcut_type="semantic", correlation=0.9):
        """
        Generate test set with specified shortcut type appended at end.

        shortcut_type:
          "semantic"  - "the {answer_loc} is near the room" (original)
          "lexical"   - answer token repeated 3x: "{loc} {loc} {loc}"
          "arbitrary" - fixed nonsense phrase correlated with label
                        label=0 → "k5 v2 e7" | label=1 → "k9 v4 e1"
        """
        examples = []
        for _ in range(n):
            parts, question, answer, label, rt, nh, nd = self._make_story()

            if self.rng.random() < correlation:
                if shortcut_type == "semantic":
                    cue = f"the {answer} is near the room"
                elif shortcut_type == "lexical":
                    cue = f"{answer} {answer} {answer}"
                elif shortcut_type == "arbitrary":
                    # Fixed tokens correlated with label, not with answer
                    if label == 0:
                        cue = "k5 v2 e7"
                    else:
                        cue = "k9 v4 e1"
                else:
                    raise ValueError(f"Unknown shortcut type: {shortcut_type}")
                parts.append(cue)
                cue_name = f"hint_{shortcut_type}"
            else:
                cue_name = f"no_hint_{shortcut_type}"

            prem = " . ".join(parts) + " ."
            examples.append(Example(premises=prem, question=question, answer=answer,
                label=label, cot=None, shortcut_available=True,
                reasoning_type=rt, n_hops=nh, n_distractors=nd, shortcut_cue=cue_name))
        self.rng.shuffle(examples)
        return examples


# =====================================================================
# MODELS (identical to previous scripts)
# =====================================================================
class BaseModel(nn.Module):
    def __init__(self, vs, dm, nl, nc=2, do=0.1, sl=320, pi=0):
        super().__init__()
        self.emb = nn.Embedding(vs, dm, padding_idx=pi)
        self.emb_do = nn.Dropout(do)
        self.clf = nn.Sequential(nn.Linear(dm, dm), nn.GELU(), nn.Dropout(do), nn.Linear(dm, nc))
    def forward(self, ids, mask, return_intermediates=False):
        p, inters = self.encode(ids, mask)
        out = {"logits": self.clf(p)}
        if return_intermediates: out["intermediates"] = inters
        return out
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class TransformerModel(BaseModel):
    def __init__(self, vocab_size, d_model=64, n_layers=2, n_heads=4,
                 d_ff=256, n_classes=2, dropout=0.1, max_seq_len=320, pad_id=0):
        super().__init__(vocab_size, d_model, n_layers, n_classes, dropout, max_seq_len, pad_id)
        pe = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
        self.pe_do = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, "gelu",
                batch_first=True, norm_first=True) for _ in range(n_layers)])
        self.fn = nn.LayerNorm(d_model)
    def encode(self, ids, mask):
        x = self.pe_do(self.emb_do(self.emb(ids)) + self.pe[:, :ids.size(1)])
        pm = (mask == 0); inters = {"l0": x.detach()}
        for i, layer in enumerate(self.layers):
            x = layer(x, src_key_padding_mask=pm); inters[f"l{i+1}"] = x.detach()
        x = self.fn(x); m = mask.unsqueeze(-1).float()
        return (x * m).sum(1) / m.sum(1).clamp(min=1), inters

class LSTMModel(BaseModel):
    def __init__(self, vocab_size, d_model=64, n_layers=2, n_classes=2,
                 dropout=0.1, max_seq_len=320, pad_id=0, n_heads=4, d_ff=256):
        super().__init__(vocab_size, d_model, n_layers, n_classes, dropout, max_seq_len, pad_id)
        hs = d_model // 2
        self.ls = nn.ModuleList([nn.LSTM(d_model, hs, 1, batch_first=True, bidirectional=True)
            for _ in range(n_layers)])
        self.ns = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.ds = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])
    def encode(self, ids, mask):
        x = self.emb_do(self.emb(ids)); lengths = mask.sum(1).cpu()
        inters = {"l0": x.detach()}
        for i, (l, n, d) in enumerate(zip(self.ls, self.ns, self.ds)):
            pk = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            o, _ = l(pk)
            xo, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True, total_length=ids.size(1))
            x = n(x + d(xo)) if xo.size(-1) == x.size(-1) else n(d(xo))
            inters[f"l{i+1}"] = x.detach()
        m = mask.unsqueeze(-1).float()
        return (x * m).sum(1) / m.sum(1).clamp(min=1), inters

class SSMBlock(nn.Module):
    def __init__(self, dm, ds=16, dc=4, ex=2, dropout=0.1):
        super().__init__()
        self.ds = ds; di = dm * ex
        self.ip = nn.Linear(dm, di * 2, bias=False)
        self.cv = nn.Conv1d(di, di, dc, padding=dc - 1, groups=di)
        self.xp = nn.Linear(di, ds * 2 + 1, bias=False)
        self.A = nn.Parameter(torch.randn(di, ds))
        self.D = nn.Parameter(torch.ones(di))
        self.op = nn.Linear(di, dm, bias=False)
        self.do = nn.Dropout(dropout)
    def forward(self, x):
        B, L, _ = x.shape
        xz = self.ip(x); xp, z = xz.chunk(2, dim=-1)
        xc = F.silu(self.cv(xp.transpose(1, 2))[:, :, :L].transpose(1, 2))
        proj = self.xp(xc); Bm = proj[..., :self.ds]; C = proj[..., self.ds:2*self.ds]
        dt = F.softplus(proj[..., -1]); A = -torch.exp(self.A.float())
        h = torch.zeros(B, xc.size(-1), self.ds, device=x.device); outs = []
        for t in range(L):
            dt_t = dt[:, t].unsqueeze(-1).unsqueeze(-1)
            h = h * torch.exp(A * dt_t) + xc[:, t].unsqueeze(-1) * Bm[:, t].unsqueeze(1) * dt_t
            outs.append((h * C[:, t].unsqueeze(1)).sum(-1) + self.D * xc[:, t])
        return self.do(self.op(torch.stack(outs, 1) * F.silu(z)))

class MambaModel(BaseModel):
    def __init__(self, vocab_size, d_model=64, n_layers=2, n_classes=2,
                 dropout=0.1, max_seq_len=320, pad_id=0, n_heads=4, d_ff=256):
        super().__init__(vocab_size, d_model, n_layers, n_classes, dropout, max_seq_len, pad_id)
        self.bs = nn.ModuleList([SSMBlock(d_model, dropout=dropout) for _ in range(n_layers)])
        self.ns = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.fn = nn.LayerNorm(d_model)
    def encode(self, ids, mask):
        x = self.emb_do(self.emb(ids)); inters = {"l0": x.detach()}
        for i, (b, n) in enumerate(zip(self.bs, self.ns)):
            x = x + b(n(x)); inters[f"l{i+1}"] = x.detach()
        x = self.fn(x); m = mask.unsqueeze(-1).float()
        return (x * m).sum(1) / m.sum(1).clamp(min=1), inters

MODEL_REG = {"transformer": TransformerModel, "lstm": LSTMModel, "mamba": MambaModel}
SCALE_CONFIGS = {"medium": {"d_model": 96, "n_layers": 4, "n_heads": 4, "d_ff": 384}}

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
        if self.step_count <= self.wu: scale = self.step_count / max(self.wu, 1)
        else:
            progress = (self.step_count - self.wu) / max(self.total - self.wu, 1)
            scale = 0.5 * (1 + math.cos(math.pi * progress))
        for pg, base in zip(self.opt.param_groups, self.base_lrs):
            pg["lr"] = max(base * scale, self.mn)

class Trainer:
    def __init__(self, model, tr, vl, tc, ts, epochs=150,
                 lr=5e-4, device="auto", eval_every=5, patience=30):
        if device == "auto":
            self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else: self.dev = torch.device(device)
        self.model = model.to(self.dev)
        self.tr, self.vl, self.tc, self.ts = tr, vl, tc, ts
        self.epochs, self.ee, self.pat = epochs, eval_every, patience
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        self.sch = CosineWarmupScheduler(self.opt, 200, epochs * len(tr))
        self.crit = nn.CrossEntropyLoss(); self.bv = 0; self.w = 0

    @torch.no_grad()
    def _ev(self, dl):
        if dl is None: return 0.0
        self.model.eval(); c = t = 0
        for b in dl:
            ids = b["input_ids"].to(self.dev); lab = b["labels"].to(self.dev)
            mask = b["attention_mask"].to(self.dev)
            c += (self.model(ids, mask)["logits"].argmax(-1) == lab).sum().item()
            t += lab.size(0)
        return c / max(t, 1)

    def train(self):
        history = []
        for ep in range(1, self.epochs + 1):
            self.model.train(); tl = n = 0
            for b in self.tr:
                ids = b["input_ids"].to(self.dev); lab = b["labels"].to(self.dev)
                mask = b["attention_mask"].to(self.dev)
                loss = self.crit(self.model(ids, mask)["logits"], lab)
                self.opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step(); self.sch.step(); tl += loss.item(); n += 1
            if ep % self.ee == 0 or ep == 1 or ep == self.epochs:
                v = self._ev(self.vl); tc = self._ev(self.tc); ts = self._ev(self.ts)
                history.append({"epoch": ep, "val": v, "clean": tc, "short": ts, "gap": ts - tc})
                if v > self.bv: self.bv = v; self.w = 0
                else:
                    self.w += self.ee
                    if self.w >= self.pat: break
        return history[-1] if history else {}

# =====================================================================
# MAIN
# =====================================================================
SEEDS = [42, 137, 256]
SHORTCUT_TYPES = ["semantic", "lexical", "arbitrary"]
ARCHS = ["transformer", "lstm", "mamba"]
PROTOCOL = dict(n_train=500, n_test=300, n_val=200, batch_size=16, lr=5e-4,
    max_epochs=150, patience=30, eval_every=5, max_seq_len=320, dropout=0.15)

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

@torch.no_grad()
def evaluate(model, dl, dev):
    model.eval(); c = t = 0
    for b in dl:
        ids = b["input_ids"].to(dev); lab = b["labels"].to(dev)
        mask = b["attention_mask"].to(dev)
        c += (model(ids, mask)["logits"].argmax(-1) == lab).sum().item()
        t += lab.size(0)
    return c / max(t, 1)

def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = make_tokenizer()
    sl, bs = PROTOCOL["max_seq_len"], PROTOCOL["batch_size"]
    cfg = SCALE_CONFIGS["medium"]

    print(f"Device: {dev}")
    print(f"Experiment J: Shortcut Type Generalization")
    print(f"Archs: {ARCHS}")
    print(f"Shortcut types: {SHORTCUT_TYPES}")
    print(f"Seeds: {SEEDS}, Scale: medium")
    print("=" * 70)

    results = []
    t_start = time.time()

    for SEED in SEEDS:
        for arch in ARCHS:
            print(f"\n{'='*70}")
            print(f"  TRAINING: {arch} (medium, seed={SEED}) on CLEAN data")
            print(f"{'='*70}")

            # Train on clean data
            set_seed(SEED)
            gen = MultiShortcutGenerator(seed=SEED)
            train_ex = gen.generate_clean(PROTOCOL["n_train"])
            val_ex = gen.generate_clean(PROTOCOL["n_val"])
            test_clean = gen.generate_clean(PROTOCOL["n_test"])

            # Dummy shortcut for trainer
            dummy = gen.generate_shortcut(PROTOCOL["n_test"], shortcut_type="semantic")

            tr_dl = DataLoader(ReasoningDataset(train_ex, tok, sl), batch_size=bs, shuffle=True)
            vl_dl = DataLoader(ReasoningDataset(val_ex, tok, sl), batch_size=bs)
            tc_dl = DataLoader(ReasoningDataset(test_clean, tok, sl), batch_size=bs)
            ts_dl = DataLoader(ReasoningDataset(dummy, tok, sl), batch_size=bs)

            set_seed(SEED)
            model = MODEL_REG[arch](vocab_size=tok.vocab_size, dropout=PROTOCOL["dropout"],
                max_seq_len=sl, **cfg)
            params = model.count_parameters()
            print(f"  Params: {params:,}")

            trainer = Trainer(model, tr_dl, vl_dl, tc_dl, ts_dl,
                epochs=PROTOCOL["max_epochs"], lr=PROTOCOL["lr"],
                eval_every=PROTOCOL["eval_every"], patience=PROTOCOL["patience"])
            t0 = time.time()
            trainer.train()
            print(f"  Trained in {time.time()-t0:.0f}s")

            clean_acc = round(evaluate(model, tc_dl, trainer.dev) * 100, 2)
            print(f"  Clean accuracy: {clean_acc:.1f}%")

            # Evaluate against each shortcut type
            for sc_type in SHORTCUT_TYPES:
                set_seed(SEED + 2000)
                gen_test = MultiShortcutGenerator(seed=SEED + 2000)
                short_ex = gen_test.generate_shortcut(PROTOCOL["n_test"],
                    shortcut_type=sc_type, correlation=0.9)
                short_dl = DataLoader(ReasoningDataset(short_ex, tok, sl), batch_size=bs)

                short_acc = round(evaluate(model, short_dl, trainer.dev) * 100, 2)
                gap = round(short_acc - clean_acc, 2)

                print(f"    {sc_type:12s}: short_acc={short_acc:.1f}%  gap={gap:+.1f}%")

                results.append({
                    "arch": arch, "seed": SEED, "scale": "medium", "params": params,
                    "shortcut_type": sc_type, "clean_acc": clean_acc,
                    "short_acc": short_acc, "gap": gap,
                })

    total = time.time() - t_start

    # Summary
    print(f"\n{'='*70}")
    print(f"  RESULTS: Shortcut Type Generalization ({total/60:.1f} min)")
    print(f"{'='*70}")
    print(f"{'Arch':<15} {'Seed':<6} {'Type':<12} {'Clean':<8} {'Short':<8} {'Gap':<8}")
    print("-" * 57)
    for r in results:
        print(f"{r['arch']:<15} {r['seed']:<6} {r['shortcut_type']:<12} {r['clean_acc']:<8.1f} {r['short_acc']:<8.1f} {r['gap']:+.1f}%")

    # Save
    out = {"experiment": "J_shortcut_types", "seeds": SEEDS, "scale": "medium",
           "shortcut_types": SHORTCUT_TYPES, "runs": results}
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "exp_j.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
