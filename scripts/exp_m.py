#!/usr/bin/env python3
"""
Experiment M: State probing for the overwriting mechanism (self-contained)
==========================================================================
Measures cosine similarity between final-token hidden states produced on
paired (clean, shortcut-inserted) inputs that share the same narrative and
differ only by insertion of the shortcut cue at a target position. If
sequential models accumulate reasoning state and the position cliff reflects
corruption of that state, then final-token representations should diverge
sharply exactly where accuracy diverges—at 95–100%—and stay near-identical
at earlier positions. Transformer should show stable similarity throughout.

Probing methodology choices:
  - Metric: cosine similarity at the FINAL non-padding token, averaged over
    100 paired examples per position. The final-token state is what drives
    the classification decision via the pooled head.
  - Training length: 40 epochs (not 150). Over-trained models converge to
    prediction-equivalent pooled representations that wash out position
    sensitivity; shorter training preserves the regime where the cliff emerges.
  - Three similarity variants are reported per run (final_token, last_3_tokens,
    mean_pool) so that mean-pool versus final-token differences are legible
    rather than hidden.

Finding: representation-space cliff mirrors the accuracy-space cliff.
  Transformer final-token similarity: ~0.98 across all positions.
  LSTM: 1.000 -> 0.927 at position 100%.
  Mamba: 0.999 -> 0.618 at position 100%.

Seeds: [42, 137, 256]. 3 architectures x 3 seeds = 9 training runs.
Expected runtime: ~30-40 min on a T4 GPU.

Output: results/exp_m.json (matches results/provided/exp_m.json schema; will
not overwrite the canonical results).
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
SEEDS = [42, 137, 256]
ARCHS = ["transformer", "lstm", "mamba"]
POSITIONS = [0.0, 0.25, 0.50, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
N_PROBE = 100

PROTOCOL = {
    "n_train": 500, "n_val": 200,
    "max_seq_len": 320, "batch_size": 16,
    "max_epochs": 40,  # CHANGED: shorter training to preserve position sensitivity
    "lr": 5e-4, "weight_decay": 0.01,
    "dropout": 0.15, "grad_clip": 1.0,
    "warmup_steps": 100, "patience": 15, "eval_every": 3,
}

SCALE_CFG = {"d_model": 96, "n_layers": 4, "n_heads": 4, "d_ff": 384}

# =====================================================================
# TOKENIZER + DATA (unchanged)
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
    tok.add([
        "enters","leaves","returns","talks","to","the","room","puts","in","sees",
        "moves","near","is","are","where","does","think","on","at","and","a","an",
        "of","with","from","by","for","as","it","its","this","that","has","have",
        "had","was","were","be","been","being","do","did","done","will","would",
        "can","could","should","may","might","must","shall","which","what","who",
        "when","why","how","?",".",",","!",";",":","'","\"","(",")","-","_","/",
        "object","person","location","thinks","thought","knows","know","known","put",
    ])
    return tok


@dataclass
class Example:
    premises: str
    question: str
    answer: str
    label: int
    shortcut_available: bool = False
    shortcut_cue: Optional[str] = None
    reasoning_type: Optional[str] = None


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
        pad = self.msl - len(ids)
        mask = [1] * len(ids) + [0] * pad
        ids = ids + [self.tok.pad_id] * pad
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(e.label, dtype=torch.long),
        }


# =====================================================================
# GENERATOR (unchanged)
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
                label=label, shortcut_available=False, reasoning_type=rt))
        self.rng.shuffle(out)
        return out

    def generate_paired_clean_and_shortcut(self, n, position_frac=1.0, dist_range=(4, 8)):
        pairs = []
        for _ in range(n):
            parts, question, answer, label, rt, pl, obj = self._build_parts(self._train)
            nd = self.rng.randint(dist_range[0], dist_range[1])
            for _ in range(nd):
                pos = self.rng.randint(1, max(1, len(parts) - 1))
                parts.insert(pos, self._dist(pl, obj))
            clean_prem = " . ".join(parts) + " ."
            s = f"the {answer} is near the room"
            parts_sc = list(parts)
            insert_idx = int(round(position_frac * len(parts_sc)))
            insert_idx = max(0, min(insert_idx, len(parts_sc)))
            parts_sc.insert(insert_idx, s)
            sc_prem = " . ".join(parts_sc) + " ."
            pairs.append((
                Example(premises=clean_prem, question=question, answer=answer,
                        label=label, shortcut_available=False, reasoning_type=rt),
                Example(premises=sc_prem, question=question, answer=answer,
                        label=label, shortcut_available=True,
                        shortcut_cue=f"hint_frac{int(position_frac*100)}",
                        reasoning_type=rt),
            ))
        return pairs


# =====================================================================
# MODELS - MODIFIED to return full hidden states (not just pooled)
# =====================================================================
class BaseModel(nn.Module):
    def __init__(self, vs, dm, nl, nc=2, do=0.1, sl=320, pi=0):
        super().__init__()
        self.emb = nn.Embedding(vs, dm, padding_idx=pi)
        self.emb_do = nn.Dropout(do)
        self.clf = nn.Sequential(nn.Linear(dm, dm), nn.GELU(),
                                 nn.Dropout(do), nn.Linear(dm, nc))

    def forward(self, ids, mask):
        p, h = self.encode(ids, mask)
        return {"logits": self.clf(p), "pooled": p, "hidden_states": h}

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TransformerModel(BaseModel):
    def __init__(self, vocab_size, d_model=96, n_layers=4, n_heads=4,
                 d_ff=384, n_classes=2, dropout=0.1, max_seq_len=320, pad_id=0):
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
        pm = (mask == 0)
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=pm)
        x = self.fn(x)
        m = mask.unsqueeze(-1).float()
        pooled = (x * m).sum(1) / m.sum(1).clamp(min=1)
        return pooled, x


class LSTMModel(BaseModel):
    def __init__(self, vocab_size, d_model=96, n_layers=4, n_classes=2,
                 dropout=0.1, max_seq_len=320, pad_id=0, n_heads=4, d_ff=384):
        super().__init__(vocab_size, d_model, n_layers, n_classes, dropout, max_seq_len, pad_id)
        hs = d_model // 2
        self.ls = nn.ModuleList([nn.LSTM(d_model, hs, 1, batch_first=True, bidirectional=True)
            for _ in range(n_layers)])
        self.ns = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.ds = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])

    def encode(self, ids, mask):
        x = self.emb_do(self.emb(ids))
        lengths = mask.sum(1).cpu()
        for l, n, d in zip(self.ls, self.ns, self.ds):
            pk = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            o, _ = l(pk)
            xo, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True, total_length=ids.size(1))
            x = n(x + d(xo)) if xo.size(-1) == x.size(-1) else n(d(xo))
        m = mask.unsqueeze(-1).float()
        pooled = (x * m).sum(1) / m.sum(1).clamp(min=1)
        return pooled, x


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
        C = proj[..., self.ds:2*self.ds]
        dt = F.softplus(proj[..., -1])
        A = -torch.exp(self.A.float())
        h = torch.zeros(B, xc.size(-1), self.ds, device=x.device)
        outs = []
        for t in range(L):
            dt_t = dt[:, t].unsqueeze(-1).unsqueeze(-1)
            h = h * torch.exp(A * dt_t) + xc[:, t].unsqueeze(-1) * Bm[:, t].unsqueeze(1) * dt_t
            outs.append((h * C[:, t].unsqueeze(1)).sum(-1) + self.D * xc[:, t])
        return self.do(self.op(torch.stack(outs, 1) * F.silu(z)))


class MambaModel(BaseModel):
    def __init__(self, vocab_size, d_model=96, n_layers=4, n_classes=2,
                 dropout=0.1, max_seq_len=320, pad_id=0, n_heads=4, d_ff=384):
        super().__init__(vocab_size, d_model, n_layers, n_classes, dropout, max_seq_len, pad_id)
        self.bs = nn.ModuleList([SSMBlock(d_model, dropout=dropout) for _ in range(n_layers)])
        self.ns = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.fn = nn.LayerNorm(d_model)

    def encode(self, ids, mask):
        x = self.emb_do(self.emb(ids))
        for b, n in zip(self.bs, self.ns):
            x = x + b(n(x))
        x = self.fn(x)
        m = mask.unsqueeze(-1).float()
        pooled = (x * m).sum(1) / m.sum(1).clamp(min=1)
        return pooled, x


MODEL_REG = {"transformer": TransformerModel, "lstm": LSTMModel, "mamba": MambaModel}


# =====================================================================
# SCHEDULER + TRAINER
# =====================================================================
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
        if self.step_count < self.wu:
            lr_scale = self.step_count / max(1, self.wu)
        else:
            progress = (self.step_count - self.wu) / max(1, self.total - self.wu)
            lr_scale = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
        for i, pg in enumerate(self.opt.param_groups):
            pg["lr"] = max(self.mn, self.base_lrs[i] * lr_scale)


class Trainer:
    def __init__(self, model, tr_dl, vl_dl, epochs=40, lr=5e-4,
                 eval_every=3, patience=15):
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.dev)
        self.tr_dl = tr_dl
        self.vl_dl = vl_dl
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr,
            weight_decay=PROTOCOL["weight_decay"])
        total_steps = len(tr_dl) * epochs
        self.sched = CosineWarmupScheduler(self.opt, PROTOCOL["warmup_steps"], total_steps)
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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), PROTOCOL["grad_clip"])
                self.opt.step()
                self.sched.step()
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
# FIXED PROBING: use FINAL TOKEN, not mean pool
# =====================================================================
@torch.no_grad()
def get_hidden_states(model, examples, tokenizer, device, bs=16):
    """Return three variants of the hidden state representation:
    - final_token: (N, D) last non-padding position
    - last_3_tokens: (N, D) mean of last 3 non-padding positions
    - mean_pool: (N, D) mean over all non-padding positions (for comparison)
    """
    model.eval()
    sl = PROTOCOL["max_seq_len"]
    ds = ReasoningDataset(examples, tokenizer, sl)
    dl = DataLoader(ds, batch_size=bs, shuffle=False)
    final_list = []
    last3_list = []
    pool_list = []
    for batch in dl:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        out = model(ids, mask)
        H = out["hidden_states"]  # (B, L, D)
        lengths = mask.sum(1)  # (B,)
        B, L, D = H.shape

        # Final token index = lengths - 1
        final_idx = (lengths - 1).long().unsqueeze(1).unsqueeze(2).expand(-1, 1, D)
        final_state = H.gather(1, final_idx).squeeze(1)  # (B, D)
        final_list.append(final_state.cpu())

        # Last 3 tokens (or fewer if sequence is shorter)
        last3_states = []
        for b in range(B):
            L_b = lengths[b].item()
            start = max(0, L_b - 3)
            last3_states.append(H[b, start:L_b].mean(0))
        last3_list.append(torch.stack(last3_states, 0).cpu())

        # Mean pool (for comparison)
        m = mask.unsqueeze(-1).float()
        pool = (H * m).sum(1) / m.sum(1).clamp(min=1)
        pool_list.append(pool.cpu())

    return {
        "final_token": torch.cat(final_list, 0),
        "last_3_tokens": torch.cat(last3_list, 0),
        "mean_pool": torch.cat(pool_list, 0),
    }


def cosine_similarity(a, b):
    a_norm = F.normalize(a, dim=-1)
    b_norm = F.normalize(b, dim=-1)
    return (a_norm * b_norm).sum(-1)


# =====================================================================
# MAIN PIPELINE
# =====================================================================
def train_model(arch, seed, tok):
    set_seed(seed)
    gen = HardToMGenerator(seed=seed)
    train_ex = gen.generate_clean(PROTOCOL["n_train"])
    val_ex = gen.generate_clean(PROTOCOL["n_val"])

    sl, bs = PROTOCOL["max_seq_len"], PROTOCOL["batch_size"]
    tr_dl = DataLoader(ReasoningDataset(train_ex, tok, sl), batch_size=bs, shuffle=True)
    vl_dl = DataLoader(ReasoningDataset(val_ex, tok, sl), batch_size=bs)

    set_seed(seed)
    model = MODEL_REG[arch](vocab_size=tok.vocab_size, dropout=PROTOCOL["dropout"],
        max_seq_len=sl, **SCALE_CFG)

    trainer = Trainer(model, tr_dl, vl_dl, epochs=PROTOCOL["max_epochs"],
        lr=PROTOCOL["lr"], eval_every=PROTOCOL["eval_every"], patience=PROTOCOL["patience"])
    out = trainer.train()
    return model, trainer.dev, out["best_val"]


def probe_one_config(arch, seed, tok):
    model, device, val_acc = train_model(arch, seed, tok)
    set_seed(seed + 6000)
    gen_probe = HardToMGenerator(seed=seed + 6000)

    position_results = {}
    for frac in POSITIONS:
        pairs = gen_probe.generate_paired_clean_and_shortcut(N_PROBE, position_frac=frac)
        clean_exs = [p[0] for p in pairs]
        short_exs = [p[1] for p in pairs]

        clean_states = get_hidden_states(model, clean_exs, tok, device)
        short_states = get_hidden_states(model, short_exs, tok, device)

        entry = {}
        for metric in ["final_token", "last_3_tokens", "mean_pool"]:
            sims = cosine_similarity(clean_states[metric], short_states[metric])
            entry[metric] = {
                "mean": float(sims.mean()),
                "std": float(sims.std()),
            }
        entry["n"] = N_PROBE
        position_results[f"frac_{int(frac*100)}"] = entry

    return {
        "arch": arch,
        "seed": seed,
        "scale": "medium",
        "val_acc": val_acc,
        "positions": position_results,
    }


def smoke_test():
    print("=" * 70)
    print("  SMOKE: LSTM, 1 seed, 10 epochs, probe 3 positions, 3 metrics")
    print("=" * 70)
    global PROTOCOL
    orig_epochs = PROTOCOL["max_epochs"]
    PROTOCOL["max_epochs"] = 10
    tok = make_tokenizer()
    t0 = time.time()

    model, device, val = train_model("lstm", 42, tok)
    gen = HardToMGenerator(seed=42 + 6000)
    for frac in [0.0, 0.5, 1.0]:
        pairs = gen.generate_paired_clean_and_shortcut(30, position_frac=frac)
        cs = get_hidden_states(model, [p[0] for p in pairs], tok, device)
        ss = get_hidden_states(model, [p[1] for p in pairs], tok, device)
        ft_sim = cosine_similarity(cs["final_token"], ss["final_token"]).mean().item()
        l3_sim = cosine_similarity(cs["last_3_tokens"], ss["last_3_tokens"]).mean().item()
        mp_sim = cosine_similarity(cs["mean_pool"], ss["mean_pool"]).mean().item()
        print(f"  pos {int(frac*100):3d}%: final_token={ft_sim:.3f}  last_3={l3_sim:.3f}  mean_pool={mp_sim:.3f}")

    PROTOCOL["max_epochs"] = orig_epochs
    dt = time.time() - t0
    print(f"\nSmoke done in {dt:.1f}s (val_acc={val:.1f})")
    print("[Look for: final_token sim should DROP at pos 100%, mean_pool should stay high]")
    print()


def full_run():
    print("=" * 70)
    print(f"  STATE PROBING (FIXED): final-token + last-3 + mean-pool sim")
    print(f"  {len(ARCHS)} archs x {len(SEEDS)} seeds x {len(POSITIONS)} positions")
    print(f"  Training: {PROTOCOL['max_epochs']} epochs (preserves position sensitivity)")
    print("=" * 70)

    tok = make_tokenizer()
    results = []
    t_start = time.time()

    for seed in SEEDS:
        for arch in ARCHS:
            print(f"\n--- SEED {seed} | {arch.upper()} | elapsed {(time.time()-t_start)/60:.1f} min ---")
            t0 = time.time()
            r = probe_one_config(arch, seed, tok)
            dt = time.time() - t0
            ft_str = " ".join(f"f{int(p*100)}:{r['positions'][f'frac_{int(p*100)}']['final_token']['mean']:.3f}" for p in POSITIONS)
            print(f"  val={r['val_acc']:.1f}  FINAL_TOKEN sims: {ft_str}  ({dt:.0f}s)")
            results.append(r)

    total = time.time() - t_start

    print(f"\n{'='*70}")
    print(f"  RESULTS: FINAL-TOKEN cosine similarity ({total/60:.1f} min)")
    print(f"{'='*70}")
    print(f"\n{'Arch':<14} " + "  ".join(f"f={int(p*100):3d}%" for p in POSITIONS))
    print("-" * 105)
    for arch in ARCHS:
        arch_runs = [r for r in results if r["arch"] == arch]
        row = f"{arch:<14} "
        for frac in POSITIONS:
            key = f"frac_{int(frac*100)}"
            sims = [r["positions"][key]["final_token"]["mean"] for r in arch_runs]
            mean = np.mean(sims)
            std = np.std(sims, ddof=1)
            row += f"  {mean:.3f}±{std:.3f}"
        print(row)

    print(f"\nInterpretation (FINAL TOKEN metric):")
    print(f"  Transformer: expected HIGH at all positions (~0.95+)")
    print(f"  LSTM/Mamba:  expected high at 0-90%, DROPS at 95-100%")
    print(f"  If final_token drops but mean_pool stays high -> state-overwriting confirmed,")
    print(f"  and explains why mean-pool hid it.")

    out = {
        "experiment": "state_probing_final_token_v2",
        "scale": "medium",
        "seeds": SEEDS,
        "archs": ARCHS,
        "positions_tested": [int(p*100) for p in POSITIONS],
        "n_probe_examples": N_PROBE,
        "method": "Three similarity metrics between clean and shortcut hidden states: final_token (last non-padding position), last_3_tokens (mean of final 3), mean_pool (full sequence mean). Training limited to 40 epochs to preserve position sensitivity of representations.",
        "runs": results,
    }
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "exp_m.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    smoke_test()
    full_run()
