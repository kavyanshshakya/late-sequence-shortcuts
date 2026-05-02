# Late-Sequence Shortcuts: Code, Data, and Reproduction

Code and data accompanying our paper on shortcut vulnerability in sequence
architectures. The paper characterises four findings on synthetic Theory of
Mind reasoning: a sharp position cliff in the accuracy space (§4.2), a
mirroring representation-space cliff revealed by hidden-state probing
(Appendix on Experiment M), the cliff's non-transfer to a frozen pretrained
Mamba backbone (Appendix on Experiment L) which scopes the phenomenon to
task-specific training dynamics, and the emergence of the cliff during a
specific task-learning window in training (Appendix on Experiment P).

This repository contains 14 experiments, 174 total training runs, and
pre-computed JSON results for every run. Figures are regenerated locally
from the JSON files; they are not stored in the repo.

## Repository layout

```
late-sequence-shortcuts/
├── src/                              Reusable package: data, models, training
│   ├── data/                         Tokenizer, ToM story generators, dataset
│   ├── models/                       Transformer, bidirectional LSTM, custom
│   │                                 Mamba, Mamba+PosEnc, official mamba-ssm
│   └── training/                     Trainer with cosine warmup, early stop
├── scripts/
│   ├── run_all.py                    Main entry point: Experiments A–H
│   ├── exp_i.py                      Correlation strength ablation
│   ├── exp_j.py                      Shortcut type generalisation
│   ├── exp_k.py                      Unidirectional LSTM control
│   ├── exp_l.py                      Frozen pretrained Mamba backbone
│   ├── exp_m.py                      Hidden-state probing
│   ├── exp_p.py                      Cliff emergence during training
│   └── generate_figures.py           Build figures from JSON results
├── results/
│   └── provided/                     Pre-computed canonical results (read-only)
│       └── exp_a.json … exp_p.json   14 files, 174 runs total
├── requirements.txt
├── LICENSE
└── README.md
```

## Two design notes worth knowing

**`run_all.py` uses the `src/` package; the per-experiment scripts are
self-contained.** This is intentional. The package gives a clean interface
for the 8 main runs (A–H) that share data generators and training loops. The
per-experiment scripts (I, J, K, L, M, P) each inline their own data
generator and model so they can be dropped into a single Colab/Kaggle cell
and run end-to-end without setting up the package. Both styles produce the
same JSON schema.

**Reproduction writes to `results/`, not `results/provided/`.** The
canonical results that back every table and figure in the paper live under
`results/provided/` and are never overwritten. New runs go to `results/`
(created automatically). The two directories are deliberately separated so
that small reruns or single-seed checks do not silently replace the
published numbers.

## Why letters skip from M to P

The 14 experiments are labelled A–H, I–M, then P. There is no Experiment N
or O. Earlier development included two attempts under those letters that
were abandoned before the paper was finalised; the letter P was retained for
the cliff-emergence experiment to keep its identifier stable across
internal versions. Reviewers and readers can ignore the gap — all 14
experiments listed below are present and accounted for.

## Quick start

### Rebuild figures from the provided results (no GPU needed)

```bash
pip install -r requirements.txt
python scripts/generate_figures.py
```

Reads `results/provided/` and writes publication-ready PDFs to a `figures/`
directory at the repo root.

Note: figures 12, 13, 14 (representation cliff, pretrained backbone, cliff
emergence) are produced by the corresponding experiment scripts directly:
`exp_m.py`, `exp_l.py`, and `exp_p.py`. `generate_figures.py` covers
figures 1–11.

### Reproduce experiments (GPU required)

```bash
# Experiments A–H via the unified entry point (~4–6 hours total on a T4)
python scripts/run_all.py --all --device cuda

# Or run individual experiments
python scripts/run_all.py --exp-a --exp-b
python scripts/run_all.py --exp-h --device cuda
python scripts/run_all.py --exp-a --seeds 42 137 256

# Self-contained scripts for I, J, K, L, M, P (no shared imports needed)
python scripts/exp_i.py    # ~20–30 min on T4
python scripts/exp_j.py    # ~20–30 min on T4
python scripts/exp_k.py    # ~10 min on T4
python scripts/exp_l.py    # ~1–2 hours on T4 (requires transformers)
python scripts/exp_m.py    # ~30–40 min on T4
python scripts/exp_p.py    # ~30–50 min on T4
```

All fresh outputs land in `results/exp_X.json` matching the schema of the
corresponding `results/provided/exp_X.json`. The provided files stay
untouched.

### Official Mamba implementations (Experiment D only)

Experiment D compares our custom Mamba against the official `mamba-ssm`
package at three scales:

```bash
pip install mamba-ssm[causal-conv1d] --no-build-isolation
python scripts/run_all.py --exp-d --device cuda
```

Experiment L additionally requires `transformers` for the pretrained
`state-spaces/mamba-130m-hf` checkpoint.

## Experiments

The order below follows the paper's narrative: main results first (A–H),
then robustness ablations (I, J, K), then mechanistic and scope experiments
(L, M), then training dynamics (P).

| Exp | Finding                                                       | Architectures              | Seeds | Runs |
|-----|---------------------------------------------------------------|----------------------------|-------|------|
| A   | Robustness split at small scale                               | Transformer / LSTM / Mamba | 5     | 15   |
| B   | Robustness split at medium scale                              | Transformer / LSTM / Mamba | 5     | 15   |
| C   | Positional encoding partially mitigates Mamba vulnerability   | Mamba+PosEnc               | 3     | 6    |
| D   | Vulnerability persists over 65× parameter scaling             | Official Mamba-1 / Mamba-2 | 3     | 15   |
| E   | Training exposure to the shortcut eliminates the gap          | Transformer / LSTM / Mamba | 3     | 9    |
| F   | Shortcut in isolation (no reasoning context) is trivial       | Transformer / LSTM / Mamba | 3     | 9    |
| G   | Vulnerability concentrates in one ToM scenario type           | Transformer / LSTM / Mamba | 3     | 9    |
| H   | **Position cliff at 95–100% for sequential architectures**    | Transformer / LSTM / Mamba | 5     | 15   |
| I   | Separation emerges as soon as correlation is above chance     | Transformer / LSTM / Mamba | 3     | 27   |
| J   | Cliff holds across three distinct shortcut forms              | Transformer / LSTM / Mamba | 3     | 27   |
| K   | Bidirectionality is not required for the end-position cliff   | Unidirectional LSTM        | 10    | 10   |
| L   | Cliff does not transfer to frozen pretrained Mamba backbone   | Pretrained Mamba-130M      | 5     | 5    |
| M   | **Representation-space cliff mirrors the accuracy cliff**     | Transformer / LSTM / Mamba | 3     | 9    |
| P   | **Cliff emerges during the task-learning window (ep 5–10)**   | Custom Mamba               | 3     | 3    |
|     |                                                               |                            |       | **174** |

## Training protocol

- 500 train / 200 val / 300 test examples per run
- AdamW (lr = 5×10⁻⁴, weight_decay = 0.01) with cosine warmup (200 steps)
- Max 150 epochs, early stopping patience 30, evaluate every 5 epochs
- Gradient clipping at 1.0, dropout 0.15, batch size 16

Experiment L uses a shorter protocol (30 epochs, head only) since the
backbone is frozen. Experiment M uses 40 epochs, chosen to preserve the
position-sensitive representations that full training washes out.
Experiment P trains for 50 epochs with evaluation at 9 checkpoint epochs.

## Figures

`generate_figures.py` produces figures 1–11 from `results/provided/`.
Figures 12, 13, 14 are produced by `exp_m.py`, `exp_l.py`, and `exp_p.py`
respectively (each generates its own figure as part of its run).

| #  | File                          | Content                                                  | Source script              |
|----|-------------------------------|----------------------------------------------------------|----------------------------|
| 1  | fig1_position_ablation.pdf    | End vs begin shortcut position (hero figure)             | generate_figures.py        |
| 2  | fig2_main_hierarchy.pdf       | Attention-vs-sequential split (small + medium scale)     | generate_figures.py        |
| 3  | fig3_per_scenario.pdf         | Per-scenario gap and true-belief accuracy across seeds   | generate_figures.py        |
| 4  | fig4_scale_invariance.pdf     | Scale invariance across Mamba implementations (Exp D)    | generate_figures.py        |
| 5  | fig5_training_condition.pdf   | Clean-trained vs shortcut-trained (Exp E)                | generate_figures.py        |
| 6  | fig6_posenc_ablation.pdf      | Positional encoding ablation (Exp C)                     | generate_figures.py        |
| 7  | fig7_per_seed.pdf             | Per-seed breakdown showing LSTM bimodality               | generate_figures.py        |
| 8  | fig8_shortcut_only.pdf        | Shortcut-only baseline (Exp F)                           | generate_figures.py        |
| 9  | fig9_lstm_bimodality.pdf      | LSTM training dynamics: robust vs fragile seeds          | generate_figures.py        |
| 10 | fig10_magic_seed.pdf          | Mamba-1 training dynamics: the "magic seed"              | generate_figures.py        |
| 11 | fig11_correlation_ablation.pdf| Correlation strength ablation (Exp I)                    | generate_figures.py        |
| 12 | fig_repr_cliff.pdf            | Representation-space cliff (Exp M)                       | exp_m.py                   |
| 13 | fig_pretrained.pdf            | Pretrained backbone evaluation (Exp L)                   | exp_l.py                   |
| 14 | fig_cliff_emergence.pdf       | Cliff emergence during task-learning window (Exp P)      | exp_p.py                   |

The repository does not ship rendered figure files. Run
`python scripts/generate_figures.py` to build them from the canonical results.

## Paper

> **Late-Sequence Shortcuts Selectively Disrupt Sequential Architectures: A
> Controlled Study of Distributional Robustness in Learned Reasoning**
> Kavyansh Shakya. 2026.
> arXiv:XXXX.XXXXX (preprint, to be updated after announcement)

If you use this code or the released results in your own work, please cite:

```bibtex
@article{shakya2026latesequence,
  title   = {Late-Sequence Shortcuts Selectively Disrupt Sequential Architectures:
             A Controlled Study of Distributional Robustness in Learned Reasoning},
  author  = {Shakya, Kavyansh},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## Author

Kavyansh Shakya, Indian Institute of Science Education and Research, Bhopal.
Contact: work.kavyanshshakya@gmail.com

## License

The code in this repository is released under the MIT License (see `LICENSE`).
The pre-computed result JSON files in `results/provided/` are released under
CC BY 4.0 to enable reuse and citation in derivative work.
