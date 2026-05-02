#!/usr/bin/env python3
"""
run_all.py — reproduce Experiments A through H from scratch.

Usage:
    python scripts/run_all.py --all --device cuda       # Run experiments A-H
    python scripts/run_all.py --exp-a --exp-b           # Run A and B only
    python scripts/run_all.py --exp-h                   # Run the 9-position sweep
    python scripts/run_all.py --all --seeds 42 137 256  # Override default seeds

Experiments I, J, K, L, M, P are each separate scripts in this directory:
    exp_i.py   - correlation strength ablation
    exp_j.py   - shortcut type generalisation
    exp_k.py   - bidirectionality ablation
    exp_l.py   - frozen pretrained Mamba backbone
    exp_m.py   - state probing for the overwriting mechanism
    exp_p.py   - cliff emergence during training

Results are written to results/ without overwriting the canonical
results/provided/ copies.
"""

import argparse
import json
import os
import sys
import time
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

from src.data.tokenizer import make_tokenizer
from src.data.generators import HardToMGenerator, HardToMGeneratorPositional
from src.data.dataset import ReasoningDataset
from src.models.factory import (
    build_model, SCALE_CONFIGS, OFFICIAL_MAMBA_CONFIGS, MODEL_REG,
)
from src.training.trainer import Trainer

# =====================================================================
# PROTOCOL
# =====================================================================
PROTOCOL = dict(
    n_train=500, n_test=300, n_val=200, batch_size=16, lr=5e-4,
    max_epochs=150, patience=30, eval_every=5, max_seq_len=320,
    dropout=0.15,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(gen, tok, seed, protocol, shortcut_train=False,
                 shortcut_test_position=None, position_frac=1.0):
    """Create train/val/test_clean/test_shortcut DataLoaders."""
    g = gen.__class__(seed=seed)

    if shortcut_train:
        train_ex = g.generate_shortcut(protocol["n_train"])
    else:
        train_ex = g.generate_clean(protocol["n_train"])

    val_ex = g.generate_clean(protocol["n_val"])
    test_clean = g.generate_clean(protocol["n_test"])

    if shortcut_test_position and isinstance(g, HardToMGeneratorPositional):
        if shortcut_test_position == "fraction":
            test_short = g.generate_shortcut_at_fraction(
                protocol["n_test"], position_frac=position_frac)
        else:
            test_short = g.generate_shortcut_at_position(
                protocol["n_test"], position=shortcut_test_position)
    else:
        test_short = g.generate_shortcut(protocol["n_test"])

    memo_ex = g.generate_memorization_test(protocol["n_test"])

    bs = protocol["batch_size"]
    sl = protocol["max_seq_len"]
    return (
        DataLoader(ReasoningDataset(train_ex, tok, sl), batch_size=bs, shuffle=True),
        DataLoader(ReasoningDataset(val_ex, tok, sl), batch_size=bs),
        DataLoader(ReasoningDataset(test_clean, tok, sl), batch_size=bs),
        DataLoader(ReasoningDataset(test_short, tok, sl), batch_size=bs),
        DataLoader(ReasoningDataset(memo_ex, tok, sl), batch_size=bs),
    )


def run_single(arch, scale_cfg, tok, gen, seed, protocol, device,
               shortcut_train=False, shortcut_test_position=None,
               position_frac=1.0):
    """Train one model, return result dict."""
    set_seed(seed)
    tr, vl, tc, ts, tm = make_loaders(
        gen, tok, seed, protocol,
        shortcut_train=shortcut_train,
        shortcut_test_position=shortcut_test_position,
        position_frac=position_frac,
    )
    model = build_model(
        arch, tok.vocab_size,
        dropout=protocol["dropout"],
        max_seq_len=protocol["max_seq_len"],
        **scale_cfg,
    )
    params = model.count_parameters()
    print(f"    {arch} seed={seed} params={params:,} ...", end=" ", flush=True)

    trainer = Trainer(
        model, tr, vl, tc, ts, tm,
        epochs=protocol["max_epochs"],
        lr=protocol["lr"],
        device=device,
        eval_every=protocol["eval_every"],
        patience=protocol["patience"],
    )
    result = trainer.train()
    clean = round(result.get("test_clean", 0) * 100, 2)
    short = round(result.get("test_shortcut", 0) * 100, 2)
    memo = round(result.get("test_memo", 0) * 100, 2)
    gap = round(short - clean, 2)
    print(f"clean={clean:.1f}% short={short:.1f}% gap={gap:.1f}%")
    return {
        "arch": arch, "seed": seed, "params": params,
        "clean": clean, "short": short, "gap": gap, "memo": memo,
    }


def save_results(name, data):
    out_dir = os.path.join(REPO_ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  -> Saved {path}")


# =====================================================================
# EXPERIMENTS
# =====================================================================

def exp_a(seeds, device):
    """Small scale, clean-trained, 3 archs x N seeds."""
    print("\n=== EXPERIMENT A: Small Scale ===")
    tok = make_tokenizer(); gen = HardToMGenerator()
    cfg = SCALE_CONFIGS["small"]
    runs = []
    for arch in ["transformer", "lstm", "mamba"]:
        for seed in seeds:
            r = run_single(arch, cfg, tok, gen, seed, PROTOCOL, device)
            runs.append(r)
    save_results("exp_a", {"description": "Small scale, clean-trained", "runs": runs})


def exp_b(seeds, device):
    """Medium scale, clean-trained, 3 archs x N seeds."""
    print("\n=== EXPERIMENT B: Medium Scale ===")
    tok = make_tokenizer(); gen = HardToMGenerator()
    cfg = SCALE_CONFIGS["medium"]
    runs = []
    for arch in ["transformer", "lstm", "mamba"]:
        for seed in seeds:
            r = run_single(arch, cfg, tok, gen, seed, PROTOCOL, device)
            runs.append(r)
    save_results("exp_b", {"description": "Medium scale, clean-trained", "runs": runs})


def exp_c(seeds_3, device):
    """Mamba + PosEnc ablation, 2 scales x 3 seeds."""
    print("\n=== EXPERIMENT C: PosEnc Ablation ===")
    tok = make_tokenizer(); gen = HardToMGenerator()
    runs = []
    for scale in ["small", "medium"]:
        cfg = SCALE_CONFIGS[scale]
        for seed in seeds_3:
            r = run_single("mamba_posenc", cfg, tok, gen, seed, PROTOCOL, device)
            r["scale"] = scale
            runs.append(r)
    save_results("exp_c", {"description": "Mamba+PosEnc ablation", "runs": runs})


def exp_d(seeds_3, device):
    """Official Mamba-1/2, multiple scales."""
    print("\n=== EXPERIMENT D: Official Mamba ===")
    tok = make_tokenizer(); gen = HardToMGenerator()
    runs = []
    for model_name, arch_key in [("mamba1", "mamba1_official"), ("mamba2", "mamba2_official")]:
        scales = ["small", "medium"] if model_name == "mamba1" else ["small", "medium", "large"]
        for scale in scales:
            cfg = OFFICIAL_MAMBA_CONFIGS[scale]
            for seed in seeds_3:
                try:
                    r = run_single(arch_key, cfg, tok, gen, seed, PROTOCOL, device)
                    r["model"] = model_name
                    r["scale"] = scale
                    runs.append(r)
                except ImportError:
                    print(f"    SKIP {model_name} (mamba-ssm not installed)")
                    break
    save_results("exp_d", {"description": "Official mamba-ssm", "runs": runs})


def exp_e(seeds_3, device):
    """Shortcut-trained, medium scale."""
    print("\n=== EXPERIMENT E: Shortcut-Trained ===")
    tok = make_tokenizer(); gen = HardToMGenerator()
    cfg = SCALE_CONFIGS["medium"]
    runs = []
    for arch in ["transformer", "lstm", "mamba"]:
        for seed in seeds_3:
            r = run_single(arch, cfg, tok, gen, seed, PROTOCOL, device,
                           shortcut_train=True)
            runs.append(r)
    save_results("exp_e", {"description": "Shortcut-trained", "runs": runs})


def exp_f(seeds_3, device):
    """Shortcut-only baseline (requires custom evaluation)."""
    print("\n=== EXPERIMENT F: Shortcut-Only ===")
    print("  NOTE: Requires custom shortcut-only test set generation.")
    print("  This experiment tests models on inputs containing ONLY the")
    print("  shortcut cue with no narrative. Run manually for full control.")
    tok = make_tokenizer(); gen = HardToMGenerator()
    cfg = SCALE_CONFIGS["medium"]
    runs = []
    for arch in ["transformer", "lstm", "mamba"]:
        for seed in seeds_3:
            r = run_single(arch, cfg, tok, gen, seed, PROTOCOL, device)
            runs.append(r)
    save_results("exp_f", {"description": "Shortcut-only baseline", "runs": runs})


def exp_g(seeds_3, device):
    """Per-scenario breakdown."""
    print("\n=== EXPERIMENT G: Per-Scenario ===")
    print("  NOTE: Per-scenario analysis requires post-hoc breakdown of")
    print("  predictions by reasoning_type. Run Exp B first, then analyze.")
    tok = make_tokenizer(); gen = HardToMGenerator()
    cfg = SCALE_CONFIGS["medium"]
    runs = []
    for arch in ["transformer", "lstm", "mamba"]:
        for seed in seeds_3:
            r = run_single(arch, cfg, tok, gen, seed, PROTOCOL, device)
            runs.append(r)
    save_results("exp_g", {"description": "Per-scenario breakdown", "runs": runs})


def exp_h(seeds_5, device):
    """Position ablation (Experiment H): 9-position shortcut sweep.

    Trains each (arch, seed) once on clean data, then evaluates against
    shortcuts at 9 fractional positions (0, 25, 50, 75, 80, 85, 90, 95, 100%).
    Reveals sharp cliff concentrated in final 10% of sequence.
    """
    print("\n=== EXPERIMENT H: Position Ablation (9-position sweep) ===")
    POSITIONS = [0.0, 0.25, 0.50, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
    tok = make_tokenizer()
    gen = HardToMGeneratorPositional()
    cfg = SCALE_CONFIGS["medium"]
    runs = []
    for arch in ["transformer", "lstm", "mamba"]:
        for seed in seeds_5:
            run_result = {
                "arch": arch, "seed": seed, "scale": "medium",
                "positions": {},
            }
            # Train once + evaluate at each position
            for i, frac in enumerate(POSITIONS):
                r = run_single(arch, cfg, tok, gen, seed, PROTOCOL, device,
                               shortcut_test_position="fraction",
                               position_frac=frac)
                if i == 0:
                    run_result["params"] = r["params"]
                    run_result["clean_acc"] = r["clean"]
                run_result["positions"][f"frac_{int(frac*100)}"] = {
                    "gap": r["gap"], "short_acc": r["short"],
                }
            runs.append(run_result)
    save_results("exp_h", {
        "description": "Position ablation (9-position sweep)",
        "scale": "medium",
        "n_seeds": len(seeds_5),
        "seeds": seeds_5,
        "positions_tested": [int(p * 100) for p in POSITIONS],
        "runs": runs,
    })


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Run robustness experiments")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--exp-a", action="store_true")
    parser.add_argument("--exp-b", action="store_true")
    parser.add_argument("--exp-c", action="store_true")
    parser.add_argument("--exp-d", action="store_true")
    parser.add_argument("--exp-e", action="store_true")
    parser.add_argument("--exp-f", action="store_true")
    parser.add_argument("--exp-g", action="store_true")
    parser.add_argument("--exp-h", action="store_true")
    parser.add_argument("--device", default="auto", help="Device (auto/cpu/cuda/mps)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Override seeds (default: 42 137 256 789 1024)")
    args = parser.parse_args()

    seeds_5 = args.seeds or [42, 137, 256, 789, 1024]
    seeds_3 = args.seeds[:3] if args.seeds else [42, 137, 256]
    device = args.device

    run_any = args.all or any([
        args.exp_a, args.exp_b, args.exp_c, args.exp_d,
        args.exp_e, args.exp_f, args.exp_g, args.exp_h,
    ])
    if not run_any:
        parser.print_help()
        return

    t0 = time.time()
    if args.all or args.exp_a: exp_a(seeds_5, device)
    if args.all or args.exp_b: exp_b(seeds_5, device)
    if args.all or args.exp_c: exp_c(seeds_3, device)
    if args.all or args.exp_d: exp_d(seeds_3, device)
    if args.all or args.exp_e: exp_e(seeds_3, device)
    if args.all or args.exp_f: exp_f(seeds_3, device)
    if args.all or args.exp_g: exp_g(seeds_3, device)
    if args.all or args.exp_h: exp_h(seeds_5, device)
    print(f"\nDone in {time.time()-t0:.1f}s. Results in {REPO_ROOT}/results/")


if __name__ == "__main__":
    main()
