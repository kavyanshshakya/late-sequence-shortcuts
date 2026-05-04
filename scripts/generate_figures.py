#!/usr/bin/env python3
"""Generate all 14 figures from JSON result files.

Data priority: for each experiment, tries results/<exp>.json first (your
fresh runs), then falls back to results/provided/<exp>.json (canonical).
Only generates figures whose JSON data is available; skips the rest with
a warning.

Writes matplotlib-rendered PDFs to figures/ at the repo root.
"""

import os, sys, json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import matplotlib.font_manager as fm
from collections import defaultdict

# =====================================================================
# PATHS
# =====================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
RESULT_DIR_FRESH    = os.path.join(REPO_ROOT, "results")
RESULT_DIR_PROVIDED = os.path.join(REPO_ROOT, "results", "provided")
OUTPUT_DIR = os.path.join(REPO_ROOT, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load(name):
    """Load experiment JSON with per-file fallback.

    For each experiment, tries results/<name>.json first (fresh runs),
    then falls back to results/provided/<name>.json (canonical results
    that ship with the repo). Returns the parsed JSON dict, or None if
    neither location has the file (in which case the corresponding
    figure will be skipped with a warning).

    This means a user who re-runs only a subset of experiments
    (e.g. exp_i.py) will see those figures rebuilt from fresh data
    while the remaining figures continue to use canonical results.
    """
    fresh    = os.path.join(RESULT_DIR_FRESH,    f"{name}.json")
    provided = os.path.join(RESULT_DIR_PROVIDED, f"{name}.json")
    for path in (fresh, provided):
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return None

# =====================================================================
# STYLE SYSTEM  (identical to hardcoded version)
# =====================================================================
STYLE = {
    "trans":    "#EE7733", "trans_dk": "#BB5500",
    "lstm":     "#0077BB", "lstm_dk":  "#005588", "lstm_lt": "#66AADD",
    "mamba":    "#CC3311", "mamba_dk": "#992200",
    "posenc":   "#009988", "posenc_dk":"#007766",
    "m1":       "#228833", "m1_dk":    "#115522",
    "m2":       "#CC3311", "m2_dk":    "#992200",
    "chance":   "#BBBBBB", "grid":     "#EBEBEB",
    "spine":    "#BBBBBB", "text":     "#1A1A1A",
    "subtext":  "#555555",
    "scale_lt": "#D3D3D3", "scale_md": "#808080", "scale_dk": "#404040",
}

_available = {f.name for f in fm.fontManager.ttflist}
_serif_candidates = ["Times New Roman", "cmr10"]
FONT_SERIF = next((f for f in _serif_candidates if f in _available), "serif")
_mathtext_fontset = "stix" if "Times New Roman" in _available else "cm"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "figure.dpi": 150,
    "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.12,
    "font.family": "serif",
    "font.serif": ["Times New Roman","cmr10","Computer Modern Roman",
                    "DejaVu Serif","Palatino Linotype","Georgia","serif"],
    "font.size": 9, "text.color": STYLE["text"],
    "mathtext.fontset": _mathtext_fontset,
    "axes.formatter.use_mathtext": True,
    "axes.labelsize": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelcolor": STYLE["text"], "axes.titlecolor": STYLE["text"],
    "axes.titlepad": 7, "axes.linewidth": 0.7, "axes.edgecolor": STYLE["spine"],
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": True, "axes.spines.bottom": True,
    "xtick.color": STYLE["subtext"], "ytick.color": STYLE["subtext"],
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": STYLE["grid"], "grid.linestyle": "-",
    "grid.linewidth": 0.55, "grid.alpha": 1.0,
    "lines.linewidth": 1.8, "lines.markersize": 5.5,
    "legend.fontsize": 8, "legend.frameon": True, "legend.framealpha": 0.92,
    "legend.edgecolor": STYLE["spine"], "legend.facecolor": "white",
    "legend.labelcolor": STYLE["text"], "legend.borderpad": 0.4,
    "legend.labelspacing": 0.3,
})

def _style_ax(ax, grid_axis="y"):
    ax.spines["bottom"].set_visible(True); ax.spines["bottom"].set_color(STYLE["spine"])
    ax.spines["left"].set_visible(True);   ax.spines["left"].set_color(STYLE["spine"])
    ax.grid(True,  axis=grid_axis, color=STYLE["grid"], linewidth=0.55, linestyle="-")
    ax.grid(False, axis="x" if grid_axis == "y" else "y")
    ax.set_axisbelow(True)

def _panel_label(ax, label, x=0.03, y=0.97):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=11, fontweight="bold",
            va="top", ha="left", color=STYLE["text"], fontfamily="serif")

def _annot(ax, text, xy, xytext, color, fontsize=8.0, rad=0.20):
    ax.annotate(
        text, xy=xy, xytext=xytext, fontsize=fontsize,
        fontstyle="italic", color=color, fontfamily="serif",
        arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9,
                       connectionstyle=f"arc3,rad={rad}", mutation_scale=8),
        bbox=dict(boxstyle="round,pad=0.3,rounding_size=0.1",
                 fc="white", ec=color, alpha=0.85, linewidth=0.7),
        zorder=10)

def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, facecolor="white", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"  OK {name}")


# =====================================================================
# DATA EXTRACTION HELPERS
# =====================================================================
def _gaps(data, arch, key="gap"):
    return [r[key] for r in data["runs"] if r.get("arch") == arch]

def _gaps_by_seed(data, arch, seeds, key="gap"):
    lookup = {r["seed"]: r[key] for r in data["runs"] if r.get("arch") == arch}
    return [lookup[s] for s in seeds]

# =====================================================================
# LOAD ALL DATA
# =====================================================================
ea = load("exp_a"); eb = load("exp_b"); ec = load("exp_c")
ed = load("exp_d"); ee = load("exp_e"); ef = load("exp_f")
eg = load("exp_g"); eh = load("exp_h")

# Print availability summary at startup
def _origin(name):
    """Where did this experiment's data come from?"""
    fresh = os.path.join(RESULT_DIR_FRESH, f"{name}.json")
    provided = os.path.join(RESULT_DIR_PROVIDED, f"{name}.json")
    if os.path.exists(fresh):    return "fresh"
    if os.path.exists(provided): return "provided"
    return None

_core = {"a": ea, "b": eb, "c": ec, "d": ed,
         "e": ee, "f": ef, "g": eg, "h": eh}
_avail   = [n for n, d in _core.items() if d is not None]
_missing = [n for n, d in _core.items() if d is None]
_origins = {n: _origin(f"exp_{n}") for n in _core}
_n_fresh    = sum(1 for v in _origins.values() if v == "fresh")
_n_provided = sum(1 for v in _origins.values() if v == "provided")

print("Data source: results/ (fresh) takes priority, falls back to results/provided/")
print(f"  fresh runs:    {_n_fresh}/8 core experiments")
print(f"  canonical:     {_n_provided}/8 core experiments")
if _missing:
    print(f"  missing:       {_missing} -> dependent figures will skip")
print()

seeds_5 = [42, 137, 256, 789, 1024]
seeds_3 = [42, 137, 256]

# Conditionally extract data - only if the source experiment is available.
# Each block sets module-level variables; figure functions check for None
# and skip if their dependency is missing.

# Exp A
if ea is not None:
    sm_trans = _gaps(ea, "transformer"); sm_lstm = _gaps(ea, "lstm"); sm_mamba = _gaps(ea, "mamba")
else:
    sm_trans = sm_lstm = sm_mamba = None

# Exp B
if eb is not None:
    md_trans = _gaps(eb, "transformer"); md_lstm = _gaps(eb, "lstm"); md_mamba = _gaps(eb, "mamba")
else:
    md_trans = md_lstm = md_mamba = None

# Exp C - depends on A, B, C
if ec is not None and ea is not None and eb is not None:
    pe_sm = [r["gap"] for r in ec["runs"] if r["scale"]=="small"]
    pe_md = [r["gap"] for r in ec["runs"] if r["scale"]=="medium"]
    reg_sm_3 = [_gaps_by_seed(ea, "mamba", seeds_3, "gap")[i] for i in range(3)]
    reg_md_3 = [_gaps_by_seed(eb, "mamba", seeds_3, "gap")[i] for i in range(3)]
else:
    pe_sm = pe_md = reg_sm_3 = reg_md_3 = None

# Exp D
if ed is not None:
    def _d_gaps(model, scale):
        return [r["gap"] for r in ed["runs"] if r["model"]==model and r["scale"]==scale]
    m1_sm = _d_gaps("mamba1","small"); m1_md = _d_gaps("mamba1","medium")
    m2_sm = _d_gaps("mamba2","small"); m2_md = _d_gaps("mamba2","medium"); m2_lg = _d_gaps("mamba2","large")
else:
    m1_sm = m1_md = m2_sm = m2_md = m2_lg = None

# Exp E
if ee is not None:
    sc_trans = _gaps(ee, "transformer"); sc_lstm = _gaps(ee, "lstm"); sc_mamba = _gaps(ee, "mamba")
else:
    sc_trans = sc_lstm = sc_mamba = None

# Exp F
if ef is not None:
    f_trans = [r["short_only"] for r in ef["runs"] if r["arch"]=="transformer"]
    f_lstm  = [r["short_only"] for r in ef["runs"] if r["arch"]=="lstm"]
    f_mamba = [r["short_only"] for r in ef["runs"] if r["arch"]=="mamba"]
else:
    f_trans = f_lstm = f_mamba = None

# Exp G
if eg is not None:
    g_true2move_trans = [r["short_acc"] for r in eg["true_2move_results"] if r["arch"]=="transformer"]
    g_true2move_lstm  = [r["short_acc"] for r in eg["true_2move_results"] if r["arch"]=="lstm"]
    g_true2move_mamba = [r["short_acc"] for r in eg["true_2move_results"] if r["arch"]=="mamba"]
else:
    g_true2move_trans = g_true2move_lstm = g_true2move_mamba = None

# Exp H: 9-position sweep. New JSON uses positions dict.
# Positions tested: 0, 25, 50, 75, 80, 85, 90, 95, 100 (% of sequence)
if eh is not None:
    H_POSITIONS = eh["positions_tested"]

    def _h_gaps(arch):
        """Return dict {position: [gaps across seeds]} for given arch."""
        out = {p: [] for p in H_POSITIONS}
        for r in eh["runs"]:
            if r["arch"] != arch:
                continue
            for p in H_POSITIONS:
                out[p].append(r["positions"][f"frac_{p}"]["gap"])
        return out

    h_trans_pos = _h_gaps("transformer")
    h_lstm_pos  = _h_gaps("lstm")
    h_mamba_pos = _h_gaps("mamba")

    # Legacy variables used by fig03/fig07 for endpoints (kept for compatibility)
    h_trans_end = h_trans_pos[100]
    h_trans_beg = h_trans_pos[0]
    h_lstm_end  = h_lstm_pos[100]
    h_lstm_beg  = h_lstm_pos[0]
    h_mamba_end = h_mamba_pos[100]
    h_mamba_beg = h_mamba_pos[0]
else:
    H_POSITIONS = None
    h_trans_pos = h_lstm_pos = h_mamba_pos = None
    h_trans_end = h_trans_beg = None
    h_lstm_end  = h_lstm_beg  = None
    h_mamba_end = h_mamba_beg = None

# Training dynamics (not in JSON; hardcoded supplementary data)
lstm_bimodal_epochs = [1, 5, 10, 15, 20, 25, 30, 35, 40]
lstm_s256_gap_over_time = [-6.00, -5.00, -0.67, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
lstm_s789_gap_over_time = [0.67, -0.67, -18.00, -18.67, -18.67, -18.67, -18.67, -18.33, -18.33]

magic_s256_epochs = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45]
magic_s256_gaps   = [-6.00, -6.00, 0.33, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
magic_s42_epochs  = [1, 5, 10, 15, 20, 25, 30, 35, 40]
magic_s42_gaps    = [-2.67, -2.67, -11.33, -12.00, -14.33, -14.33, -14.67, -14.33, -13.67]


# =====================================================================
# FIGURE 1 (HERO): Shortcut Position Ablation (Exp H)
# =====================================================================
def fig01():
    """Cliff curve (Exp H, 9-position): hero figure showing disruption
    concentrates in final 10% of sequence."""
    if eh is None:
        print("  SKIP (no exp_h.json found)")
        return False
    fig, ax = plt.subplots(figsize=(7.0, 3.5), constrained_layout=True)

    # Compute means and stds per position for each arch
    t_m = np.array([np.mean(h_trans_pos[p]) for p in H_POSITIONS])
    t_s = np.array([np.std(h_trans_pos[p], ddof=1) for p in H_POSITIONS])
    l_m = np.array([np.mean(h_lstm_pos[p]) for p in H_POSITIONS])
    l_s = np.array([np.std(h_lstm_pos[p], ddof=1) for p in H_POSITIONS])
    m_m = np.array([np.mean(h_mamba_pos[p]) for p in H_POSITIONS])
    m_s = np.array([np.std(h_mamba_pos[p], ddof=1) for p in H_POSITIONS])

    # Cliff vulnerability zone
    ax.fill_between([89.5, 101.5], -24, 4, alpha=0.08, color=STYLE["mamba_dk"],
                    linewidth=0, zorder=1)
    ax.axhline(0, color=STYLE["spine"], lw=0.7, ls=(0, (3, 2)), zorder=2)

    # Error bands for sequential archs
    for means, stds, color in [(l_m, l_s, STYLE["lstm"]),
                                (m_m, m_s, STYLE["mamba"])]:
        ax.fill_between(H_POSITIONS, means - stds, means + stds,
                        color=color, alpha=0.12, linewidth=0, zorder=3)

    # Lines with white-fill markers (matches fig04 design language)
    ax.plot(H_POSITIONS, t_m, "-o", color=STYLE["trans"], lw=2.0,
            markersize=7, markerfacecolor="white", markeredgecolor=STYLE["trans"],
            markeredgewidth=1.6, label="Transformer", zorder=5)
    ax.plot(H_POSITIONS, l_m, "-s", color=STYLE["lstm"], lw=2.0,
            markersize=7, markerfacecolor="white", markeredgecolor=STYLE["lstm"],
            markeredgewidth=1.6, label="LSTM", zorder=5)
    ax.plot(H_POSITIONS, m_m, "-^", color=STYLE["mamba"], lw=2.0,
            markersize=7, markerfacecolor="white", markeredgecolor=STYLE["mamba"],
            markeredgewidth=1.6, label="Mamba", zorder=5)

    ax.text(95.5, 2.8, "cliff zone", fontsize=7.5, color=STYLE["mamba_dk"],
            fontstyle="italic", va="top", ha="center", fontweight="bold", zorder=4)

    _annot(ax, "cliff: sequential archs drop\nto $-10.5\\%$ gap at final token",
           xy=(99.5, -10.5), xytext=(57, -18),
           color=STYLE["mamba_dk"], fontsize=8, rad=0.22)

    ax.set_xlabel("Shortcut position (% of sequence)", fontsize=10)
    ax.set_ylabel("Shortcut gap (%)", fontsize=10)
    ax.set_xlim(-2, 103)
    ax.set_ylim(-24, 4)
    ax.set_xticks(H_POSITIONS)
    ax.set_xticklabels([str(p) for p in H_POSITIONS])
    _style_ax(ax, grid_axis="y")

    ax.legend(frameon=True, fontsize=8, loc="lower left",
              bbox_to_anchor=(0.01, 0.02),
              edgecolor="#CCCCCC", handlelength=1.8)
    ax.set_title("Shortcut disruption concentrates in the final $10\\%$ of the sequence",
                 fontsize=10.5, fontweight="bold", pad=6)
    save(fig, "fig1_position_ablation.pdf")


# =====================================================================
# FIGURE 2: Main Robustness Hierarchy (Exp A + B)
# =====================================================================
def fig02():
    if ea is None or eb is None:
        print("  SKIP (need exp_a + exp_b)")
        return False
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True, constrained_layout=True)
    for ax, data, label in [
        (axes[0], [sm_trans, sm_lstm, sm_mamba], "A"),
        (axes[1], [md_trans, md_lstm, md_mamba], "B"),
    ]:
        means = [np.mean(d) for d in data]
        stds = [np.std(d, ddof=1) if np.std(d, ddof=1) > 0 else 0 for d in data]
        colors = [STYLE["trans"], STYLE["lstm"], STYLE["mamba"]]
        dk = [STYLE["trans_dk"], STYLE["lstm_dk"], STYLE["mamba_dk"]]
        hatches = ["", "///", "\\\\"]
        for i in range(3):
            ax.bar(i, means[i], 0.55, yerr=stds[i], capsize=3.5,
                   color=colors[i], alpha=0.85, edgecolor=dk[i],
                   linewidth=0.5, hatch=hatches[i], zorder=3,
                   error_kw=dict(elinewidth=0.9, ecolor=dk[i]))
            if i > 0 and abs(means[i]) > 0.5:
                ax.text(i, means[i] - stds[i] - 2.0, f"{means[i]:.1f}%",
                        ha="center", fontsize=8, color=dk[i], fontweight="bold")
        ax.set_xticks(np.arange(3))
        ax.set_xticklabels(["Transformer", "LSTM", "Mamba"], fontsize=8.5)
        ax.axhline(0, color=STYLE["subtext"], lw=0.8, ls="--", zorder=2)
        ax.set_ylim(-24, 5)
        _style_ax(ax, grid_axis="y")
        _panel_label(ax, label)
    axes[0].set_ylabel("Shortcut gap (%)", fontsize=10)
    axes[0].set_title("Small scale (45-73K params)", fontsize=10, fontweight="bold")
    axes[1].set_title("Medium scale (260-485K params)", fontsize=10, fontweight="bold")
    save(fig, "fig2_main_hierarchy.pdf")


# =====================================================================
# FIGURE 3: Per-Scenario Breakdown (Exp G)
# =====================================================================
def fig03():
    if eg is None or eh is None:
        print("  SKIP (need exp_g + exp_h)")
        return False
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.4), constrained_layout=True,
                                   gridspec_kw={"width_ratios": [3, 2]})
    types_labels = ["2-move\nFB", "3-move\nFB", "2nd\norder", "Classic\nFB", "True\nbelief"]
    g_trans_s42 = [0.0, 0.0, 0.0, 0.0, 0.0]
    g_lstm_s42  = [0.0, 0.0, 0.0, 0.0, -7.94]
    g_mamba_s42 = [0.0, 0.0, 0.0, 0.0, -84.13]

    x = np.arange(5); w = 0.25
    ax1.bar(x - w, g_trans_s42, w, color=STYLE["trans"], alpha=0.85,
            edgecolor=STYLE["trans_dk"], lw=0.5, zorder=3, label="Transformer")
    ax1.bar(x,     g_lstm_s42,  w, color=STYLE["lstm"], alpha=0.85,
            edgecolor=STYLE["lstm_dk"], lw=0.5, hatch="///", zorder=3, label="LSTM")
    ax1.bar(x + w, g_mamba_s42, w, color=STYLE["mamba"], alpha=0.85,
            edgecolor=STYLE["mamba_dk"], lw=0.5, hatch="\\\\", zorder=3, label="Mamba")

    ax1.axhline(-50, color=STYLE["mamba"], lw=1.2, ls="--", alpha=0.7, zorder=2)
    ax1.text(0.5, -48, "catastrophic threshold", fontsize=7.5, color=STYLE["mamba"],
             fontstyle="italic", fontweight="bold")
    ax1.set_xticks(x); ax1.set_xticklabels(types_labels, fontsize=8)
    ax1.set_ylabel("Shortcut gap (%)", fontsize=10)
    ax1.set_ylim(-95, 8)
    ax1.axhline(0, color=STYLE["subtext"], lw=0.8, ls="--", zorder=2)
    ax1.legend(frameon=True, loc="lower left", ncol=3, fontsize=7.5)
    _annot(ax1, "Mamba: $-$84.1%\non true belief",
           xy=(4 + w, -84.13), xytext=(2.5, -65),
           color=STYLE["mamba_dk"], fontsize=8, rad=0.2)
    _style_ax(ax1, grid_axis="y")
    _panel_label(ax1, "A")
    ax1.set_title("Per-scenario gap (seed 42)", fontsize=10, fontweight="bold")

    seeds_x = np.arange(3); seed_labels = ["42", "137", "256"]; w2 = 0.25
    ax2.bar(seeds_x - w2, g_true2move_trans, w2, color=STYLE["trans"], alpha=0.85,
            edgecolor=STYLE["trans_dk"], lw=0.5, zorder=3, label="Transformer")
    ax2.bar(seeds_x,      g_true2move_lstm,  w2, color=STYLE["lstm"], alpha=0.85,
            edgecolor=STYLE["lstm_dk"], lw=0.5, hatch="///", zorder=3, label="LSTM")
    ax2.bar(seeds_x + w2, g_true2move_mamba, w2, color=STYLE["mamba"], alpha=0.85,
            edgecolor=STYLE["mamba_dk"], lw=0.5, hatch="\\\\", zorder=3, label="Mamba")
    for i, v in enumerate(g_true2move_mamba):
        ax2.scatter([i + w2], [v], s=30, color=STYLE["mamba_dk"],
                   edgecolors=STYLE["mamba_dk"], linewidths=0.5, zorder=5, marker="^")
    mamba_mean = np.mean(g_true2move_mamba)
    mamba_std = np.std(g_true2move_mamba, ddof=1)
    stats_text = f"Mamba true-belief:\nMean: {mamba_mean:.1f}% +/- {mamba_std:.1f}%"
    ax2.text(0.95, 0.98, stats_text, transform=ax2.transAxes,
             fontsize=7, va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=STYLE["mamba_dk"],
                      alpha=0.9, linewidth=0.7), fontfamily="serif", fontstyle="italic")
    ax2.set_xticks(seeds_x)
    ax2.set_xticklabels([f"s{s}" for s in seed_labels], fontsize=8.5)
    ax2.set_xlabel("Seed", fontsize=9)
    ax2.set_ylabel("True-belief accuracy (%)", fontsize=9)
    ax2.set_ylim(0, 115)
    ax2.axhline(50, color=STYLE["chance"], lw=1.0, ls="--", zorder=2)
    _style_ax(ax2, grid_axis="y")
    _panel_label(ax2, "B")
    ax2.set_title("True-belief across seeds", fontsize=10, fontweight="bold")
    save(fig, "fig3_per_scenario.pdf")


# =====================================================================
# FIGURE 4: Scale Invariance (Exp D)
# =====================================================================
def fig04():
    if ea is None or eb is None or ed is None:
        print("  SKIP (need exp_a + exp_b + exp_d)")
        return False
    fig, ax = plt.subplots(figsize=(7.0, 3.2), constrained_layout=True)
    scales_x = np.array([0, 1, 2])

    custom_means = [np.mean(sm_mamba), np.mean(md_mamba)]
    ax.plot([0, 1], custom_means, "-o", color=STYLE["mamba"], lw=2.0,
            markersize=7, markerfacecolor="white", markeredgecolor=STYLE["mamba"],
            markeredgewidth=1.6, label="Custom Mamba", zorder=4)

    m1_means = [np.mean(m1_sm), np.mean(m1_md)]
    ax.plot([0, 1], m1_means, "-s", color=STYLE["m1"], lw=2.0,
            markersize=7, markerfacecolor="white", markeredgecolor=STYLE["m1"],
            markeredgewidth=1.6, label="Official Mamba-1", zorder=4)

    m2_means = [np.mean(m2_sm), np.mean(m2_md), np.mean(m2_lg)]
    ax.plot([0, 1, 2], m2_means, "-^", color=STYLE["posenc"], lw=2.0,
            markersize=7, markerfacecolor="white", markeredgecolor=STYLE["posenc"],
            markeredgewidth=1.6, label="Official Mamba-2", zorder=4)

    ax.axhline(0, color=STYLE["trans"], lw=1.8, ls="-", label="Transformer (0%)",
               zorder=3, alpha=0.8)
    ax.fill_between([-0.3, 2.5], -10, -16, alpha=0.06, color=STYLE["mamba_dk"],
                    linewidth=0, zorder=1)
    ax.text(1.0, -11, "vulnerability zone", fontsize=6.5, color=STYLE["mamba_dk"],
            fontstyle="italic", va="top", ha="center")
    ax.set_xticks(scales_x)
    ax.set_xticklabels(["Small", "Medium", "Large"], fontsize=9)
    ax.set_xlabel("Scale", fontsize=10); ax.set_ylabel("Shortcut gap (%)", fontsize=10)
    ax.set_xlim(-0.3, 2.5); ax.set_ylim(-22, 3)
    ax.legend(frameon=True, fontsize=7.5, loc="upper right", bbox_to_anchor=(0.99, 0.82))
    _annot(ax, "65x more parameters,\nidentical vulnerability",
           xy=(2, m2_means[2]), xytext=(1.2, -6),
           color=STYLE["mamba_dk"], fontsize=8, rad=0.3)
    _style_ax(ax, grid_axis="y")
    ax.set_title("Vulnerability persists across scale and implementation",
                 fontsize=11, fontweight="bold")
    save(fig, "fig4_scale_invariance.pdf")


# =====================================================================
# FIGURE 5: Training Condition (Exp E)
# =====================================================================
def fig05():
    if ee is None:
        print("  SKIP (no exp_e.json found)")
        return False
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True, constrained_layout=True)
    colors = [STYLE["trans"], STYLE["lstm"], STYLE["mamba"]]
    dk = [STYLE["trans_dk"], STYLE["lstm_dk"], STYLE["mamba_dk"]]
    hatches = ["", "///", "\\\\"]
    labels_x = ["Transformer", "LSTM", "Mamba"]

    cl_means = [np.mean(md_trans), np.mean(md_lstm), np.mean(md_mamba)]
    cl_stds  = [0, np.std(md_lstm, ddof=1), np.std(md_mamba, ddof=1)]
    for i in range(3):
        axes[0].bar(i, cl_means[i], 0.55, yerr=cl_stds[i], capsize=3.5,
                    color=colors[i], alpha=0.85, edgecolor=dk[i], linewidth=0.5,
                    hatch=hatches[i], zorder=3,
                    error_kw=dict(elinewidth=0.9, ecolor=dk[i]))
        if i > 0 and abs(cl_means[i]) > 0.5:
            axes[0].text(i, cl_means[i] - cl_stds[i] - 2.0, f"{cl_means[i]:.1f}%", ha="center",
                        fontsize=8.5, fontweight="bold", color=dk[i])

    sc_means = [np.mean(sc_trans), np.mean(sc_lstm), np.mean(sc_mamba)]
    sc_stds  = [0, 0, np.std(sc_mamba, ddof=1) if np.std(sc_mamba, ddof=1) > 0 else 0]
    for i in range(3):
        axes[1].bar(i, sc_means[i], 0.55, yerr=sc_stds[i], capsize=3.5,
                    color=colors[i], alpha=0.85, edgecolor=dk[i], linewidth=0.5,
                    hatch=hatches[i], zorder=3,
                    error_kw=dict(elinewidth=0.9, ecolor=dk[i]))
    axes[1].text(1, -4, "All near 0%\n(9/9 runs)", ha="center", fontsize=9,
                color=STYLE["subtext"], fontstyle="italic", fontweight="bold")

    for ax, label, title in [
        (axes[0], "A", "Clean-trained (unseen shortcut)"),
        (axes[1], "B", "Shortcut-trained (seen in training)"),
    ]:
        ax.set_xticks(np.arange(3)); ax.set_xticklabels(labels_x, fontsize=8.5)
        ax.axhline(0, color=STYLE["subtext"], lw=0.8, ls="--", zorder=2)
        ax.set_ylim(-22, 4); _style_ax(ax, grid_axis="y")
        _panel_label(ax, label); ax.set_title(title, fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Shortcut gap (%)", fontsize=10)
    save(fig, "fig5_training_condition.pdf")


# =====================================================================
# FIGURE 6: PosEnc Ablation (Exp C)
# =====================================================================
def fig06():
    if ec is None or ea is None or eb is None:
        print("  SKIP (need exp_c + exp_a + exp_b)")
        return False
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True, constrained_layout=True)
    for ax, reg, pe, label, title in [
        (axes[0], reg_sm_3, pe_sm, "A", "Small scale"),
        (axes[1], reg_md_3, pe_md, "B", "Medium scale"),
    ]:
        means = [np.mean(reg), np.mean(pe)]
        stds  = [np.std(reg, ddof=1), np.std(pe, ddof=1)]
        colors_l = [STYLE["mamba"], STYLE["posenc"]]
        dk_c = [STYLE["mamba_dk"], STYLE["posenc_dk"]]
        hatch_list = ["\\\\", "///"]
        bar_labels = ["Mamba", "Mamba\n+PosEnc"]
        for i in range(2):
            ax.bar(i, means[i], 0.55, yerr=stds[i], capsize=3.5,
                   color=colors_l[i], alpha=0.85, edgecolor=dk_c[i],
                   linewidth=0.5, hatch=hatch_list[i], zorder=3,
                   error_kw=dict(elinewidth=0.9, ecolor=dk_c[i]))
            ax.text(i, means[i] - stds[i] - 2.0, f"{means[i]:.1f}%", ha="center",
                    fontsize=8.5, fontweight="bold", color=dk_c[i])
        imp = means[1] - means[0]
        ax.annotate("", xy=(1, means[1] - 0.5), xytext=(0, means[0] - 0.5),
                    arrowprops=dict(arrowstyle="<->", color=STYLE["subtext"], lw=1.0))
        mid_y = (means[0] + means[1]) / 2
        ax.text(0.5, mid_y + 1.8, f"+{abs(imp):.1f}pp", ha="center",
                fontsize=9, color=STYLE["subtext"], fontstyle="italic", fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels(bar_labels, fontsize=8.5)
        ax.axhline(0, color=STYLE["subtext"], lw=0.8, ls="--", zorder=2)
        ax.set_ylim(-22, 4); _style_ax(ax, grid_axis="y")
        _panel_label(ax, label); ax.set_title(title, fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Shortcut gap (%)", fontsize=10)
    save(fig, "fig6_posenc_ablation.pdf")


# =====================================================================
# FIGURE 7: Per-Seed Breakdown (Exp B)
# =====================================================================
def fig07():
    if eb is None:
        print("  SKIP (no exp_b.json found)")
        return False
    # Match paper: solid lines, filled markers, legend below the plot,
    # annotation pointing to the robust LSTM seed at 256.
    fig, ax = plt.subplots(figsize=(7.0, 3.6), constrained_layout=True)

    # Compute ranges for legend labels
    lstm_min, lstm_max = min(md_lstm), max(md_lstm)
    mamba_min, mamba_max = min(md_mamba), max(md_mamba)

    # Canonical marker convention: Transformer="o", LSTM="s", Mamba="^"
    ax.plot(seeds_5, md_trans, "-", color=STYLE["trans"], lw=2.2,
            marker="o", markersize=10, markerfacecolor=STYLE["trans"],
            markeredgecolor=STYLE["trans"], markeredgewidth=1.0,
            label="Transformer (0% gap)", zorder=5)
    ax.plot(seeds_5, md_lstm, "-", color=STYLE["lstm"], lw=2.2,
            marker="s", markersize=10, markerfacecolor=STYLE["lstm"],
            markeredgecolor=STYLE["lstm"], markeredgewidth=1.0,
            label=f"LSTM (range: {lstm_max:.1f} to {lstm_min:.1f}%)", zorder=4)
    ax.plot(seeds_5, md_mamba, "-", color=STYLE["mamba"], lw=2.2,
            marker="^", markersize=10, markerfacecolor=STYLE["mamba"],
            markeredgecolor=STYLE["mamba"], markeredgewidth=1.0,
            label=f"Mamba (range: {mamba_max:.1f} to {mamba_min:.1f}%)", zorder=4)

    ax.axhline(0, color=STYLE["subtext"], lw=0.8, ls=":", zorder=2)
    ax.set_xlabel("Random seed", fontsize=10)
    ax.set_ylabel("Shortcut gap (%)", fontsize=10)
    ax.set_xticks(seeds_5)
    ax.set_xticklabels(["42", "137", "256", "789", "1024"], fontsize=9)
    ax.set_ylim(-22, 4)
    ax.set_xlim(-30, 1100)
    # Annotation removed per user feedback - legend ranges already convey
    # "LSTM has a robust seed (range starts at 0.0)".

    # Legend below the plot
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=3, frameon=False, fontsize=8.5, handlelength=2.5)

    ax.set_title("Per-seed shortcut gap (Experiment B, medium scale)",
                 fontsize=10.5, fontweight="bold")
    _style_ax(ax, grid_axis="y")
    save(fig, "fig7_per_seed.pdf")


# =====================================================================
# FIGURE 8: Shortcut-Only Baseline (Exp F)
# =====================================================================
def fig08():
    if ef is None:
        print("  SKIP (no exp_f.json found)")
        return False
    fig, ax = plt.subplots(figsize=(3.5, 3.2), constrained_layout=True)
    means = [np.mean(f_trans), np.mean(f_lstm), np.mean(f_mamba)]
    stds  = [np.std(f_trans, ddof=1), np.std(f_lstm, ddof=1), np.std(f_mamba, ddof=1)]
    colors = [STYLE["trans"], STYLE["lstm"], STYLE["mamba"]]
    dk = [STYLE["trans_dk"], STYLE["lstm_dk"], STYLE["mamba_dk"]]
    hatches = ["", "///", "\\\\"]
    labels_x = ["Transformer", "LSTM", "Mamba"]
    for i in range(3):
        ax.bar(i, means[i], 0.55, yerr=stds[i], capsize=3.5,
               color=colors[i], alpha=0.85, edgecolor=dk[i],
               linewidth=0.5, hatch=hatches[i], zorder=3,
               error_kw=dict(elinewidth=0.9, ecolor=dk[i]))
        ax.text(i, means[i] + stds[i] + 1.8, f"{means[i]:.1f}%", ha="center",
                fontsize=9, fontweight="bold", color=dk[i])
    ax.axhline(50, color=STYLE["chance"], lw=1.8, ls="--", label="Chance (50%)", zorder=2)
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(labels_x, fontsize=9)
    ax.set_ylabel("Accuracy (%)", fontsize=10); ax.set_ylim(0, 65)
    ax.legend(frameon=True, fontsize=8.5, loc="upper right")
    _style_ax(ax, grid_axis="y")
    ax.set_title("Shortcut-only input (3 seeds)", fontsize=10, fontweight="bold")
    save(fig, "fig8_shortcut_only.pdf")


# =====================================================================
# FIGURE 9: LSTM Bimodality Training Dynamics
# =====================================================================
def fig09():
    # Match paper: solid lines, end-of-line value labels, "Seed N (label)" legend.
    # No annotation box (per user request - was overlapping/cluttering).
    fig, ax = plt.subplots(figsize=(7.0, 3.6), constrained_layout=True)

    # Critical window shaded zone
    ax.fill_between([5, 10], -22, 4, alpha=0.10, color=STYLE["mamba_dk"],
                    linewidth=0, zorder=1)
    ax.text(7.5, 2.8, "critical window", fontsize=8.5, color=STYLE["mamba_dk"],
            fontstyle="italic", fontweight="bold", ha="center")

    # Two-tone "robust vs fragile" palette using Wong colorblind-safe colors,
    # complementing the main paper figs: deep blue = robust, vermillion = fragile.
    ROBUST_COLOR  = "#0072B2"   # deep blue
    FRAGILE_COLOR = "#D55E00"   # vermillion
    ax.plot(lstm_bimodal_epochs, lstm_s256_gap_over_time, "-",
            color=ROBUST_COLOR, lw=2.4, marker="o", markersize=8,
            markerfacecolor="white", markeredgecolor=ROBUST_COLOR, markeredgewidth=1.8,
            label="Seed 256 (robust: 0.0%)", zorder=5)
    ax.plot(lstm_bimodal_epochs, lstm_s789_gap_over_time, "-",
            color=FRAGILE_COLOR, lw=2.4, marker="s", markersize=8,
            markerfacecolor="white", markeredgecolor=FRAGILE_COLOR, markeredgewidth=1.8,
            label="Seed 789 (fragile: -18.3%)", zorder=4)

    # End-of-line value labels
    ax.text(41, lstm_s256_gap_over_time[-1], f"{lstm_s256_gap_over_time[-1]:.1f}%",
            ha="left", va="center", fontsize=9, color=ROBUST_COLOR, fontweight="bold")
    ax.text(41, lstm_s789_gap_over_time[-1], f"{lstm_s789_gap_over_time[-1]:.1f}%",
            ha="left", va="center", fontsize=9, color=FRAGILE_COLOR, fontweight="bold")

    ax.axhline(0, color=STYLE["subtext"], lw=0.8, ls=":", zorder=2)

    ax.set_xlabel("Training epoch", fontsize=10)
    ax.set_ylabel("Shortcut gap (%)", fontsize=10)
    ax.set_ylim(-22, 4)
    ax.set_xticks(lstm_bimodal_epochs)
    ax.set_xlim(0, 44)
    ax.legend(loc="center right", fontsize=9, frameon=True,
              bbox_to_anchor=(0.95, 0.5))
    ax.set_title("LSTM training dynamics: bimodal divergence",
                 fontsize=10.5, fontweight="bold")
    _style_ax(ax, grid_axis="y")
    save(fig, "fig9_lstm_bimodality.pdf")


# =====================================================================
# FIGURE 10: Magic Seed Training Dynamics
# =====================================================================
def fig10():
    # Match paper: solid lines, "Only 1 of 18 Mamba runs..." annotation positioned
    # in lower part of plot to avoid overlapping the line trajectories.
    fig, ax = plt.subplots(figsize=(7.0, 3.6), constrained_layout=True)

    # Critical window
    ax.fill_between([5, 10], -18, 4, alpha=0.10, color=STYLE["mamba_dk"],
                    linewidth=0, zorder=1)
    ax.text(7.5, 2.8, "critical window", fontsize=8.5, color=STYLE["mamba_dk"],
            fontstyle="italic", fontweight="bold", ha="center")

    # Two-tone "robust vs fragile" palette (same as fig9 for consistency)
    ROBUST_COLOR  = "#0072B2"   # deep blue
    FRAGILE_COLOR = "#D55E00"   # vermillion
    ax.plot(magic_s256_epochs, magic_s256_gaps, "-",
            color=ROBUST_COLOR, lw=2.4, marker="o", markersize=8,
            markerfacecolor="white", markeredgecolor=ROBUST_COLOR, markeredgewidth=1.8,
            label="Seed 256 (magic, robust: 0.0%)", zorder=5)
    ax.plot(magic_s42_epochs, magic_s42_gaps, "-",
            color=FRAGILE_COLOR, lw=2.4, marker="s", markersize=8,
            markerfacecolor="white", markeredgecolor=FRAGILE_COLOR, markeredgewidth=1.8,
            label="Seed 42 (fragile, -13.7%)", zorder=4)

    # End labels
    ax.text(46, magic_s256_gaps[-1], f"{magic_s256_gaps[-1]:.1f}%",
            ha="left", va="center", fontsize=9, color=ROBUST_COLOR, fontweight="bold")
    ax.text(41, magic_s42_gaps[-1], f"{magic_s42_gaps[-1]:.1f}%",
            ha="left", va="center", fontsize=9, color=FRAGILE_COLOR, fontweight="bold")

    ax.axhline(0, color=STYLE["subtext"], lw=0.8, ls=":", zorder=2)

    # Annotation box positioned in MIDDLE-UPPER region (y=-5), arrow pointing UP
    # to the robust point at (10, 0.33). User feedback: move slightly up the y-axis.
    ax.annotate("Only 1 of 18 Mamba runs\nfinds a robust minimum",
                xy=(10, 0.33), xytext=(20, -5),
                fontsize=8.5, color=ROBUST_COLOR, fontstyle="italic",
                ha="center",
                arrowprops=dict(arrowstyle="->", color=ROBUST_COLOR,
                                lw=1.0, connectionstyle="arc3,rad=-0.25"),
                bbox=dict(boxstyle="round,pad=0.4", fc="white",
                          ec=ROBUST_COLOR, alpha=0.95, linewidth=0.8),
                zorder=10)

    ax.set_xlabel("Training epoch", fontsize=10)
    ax.set_ylabel("Shortcut gap (%)", fontsize=10)
    ax.set_ylim(-18, 4)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40, 45])
    ax.set_xlim(0, 49)
    ax.legend(loc="center right", fontsize=9, frameon=True,
              bbox_to_anchor=(0.95, 0.5))
    ax.set_title("Mamba-1 training dynamics: 1 of 18 seeds finds robustness",
                 fontsize=10.5, fontweight="bold")
    _style_ax(ax, grid_axis="y")
    save(fig, "fig10_magic_seed.pdf")


# =====================================================================
# FIGURE 11: Correlation Strength Ablation (Experiment I)
# =====================================================================
def fig11():
    data = load("exp_i")
    runs = data["runs"]
    correlations = data["correlations"]  # [0.5, 0.7, 0.9]
    archs = ["transformer", "lstm", "mamba"]

    from collections import defaultdict
    agg = defaultdict(list)
    for r in runs:
        agg[(r["arch"], r["correlation"])].append(r["gap"])

    means = {a: np.array([np.mean(agg[(a, c)]) for c in correlations]) for a in archs}
    stds  = {a: np.array([np.std(agg[(a, c)], ddof=1) if len(agg[(a, c)]) > 1 else 0
                          for c in correlations]) for a in archs}

    fig, ax = plt.subplots(figsize=(7.0, 3.8), constrained_layout=True)

    x_pct = [int(c * 100) for c in correlations]

    # Chance region: very subtle grey rectangle from 49 to 60 (covers x=50 area)
    ax.fill_betweenx([-18, 1], 49, 60, alpha=0.10,
                     color=STYLE["chance"], linewidth=0, zorder=1)
    # "chance region" label centered horizontally and vertically inside the band.
    # Band spans x=[49, 60] and y=[-18, 1] - centre is x=54.5, y=-8.5.
    ax.text(54.5, -8.5, "chance\nregion", ha="center", va="center",
            fontsize=8.5, color=STYLE["subtext"], fontstyle="italic")

    # Canonical markers: Transformer="o", LSTM="s", Mamba="^"
    arch_cfg = [
        ("transformer", STYLE["trans"], "o", "Transformer"),
        ("lstm",        STYLE["lstm"],  "s", "LSTM"),
        ("mamba",       STYLE["mamba"], "^", "Mamba"),
    ]
    # Plot in z-order so Mamba (most extreme) is on top
    for arch, color, marker, label in arch_cfg:
        m = means[arch]
        s = stds[arch]
        # Filled error band
        ax.fill_between(x_pct, m - s, m + s, alpha=0.18, color=color,
                        linewidth=0, zorder=2)
        # Line + open markers + error bars
        ax.errorbar(x_pct, m, yerr=s, fmt=f"-{marker}", color=color, lw=2.2,
                    markersize=8, markerfacecolor="white", markeredgecolor=color,
                    markeredgewidth=1.6, capsize=4, capthick=1.0,
                    elinewidth=1.0, ecolor=color, label=label, zorder=5)

    # Zero line on top
    ax.axhline(0, color=STYLE["subtext"], lw=0.8, ls=":", zorder=2)

    ax.set_xticks(x_pct)
    ax.set_xticklabels([f"{c}" for c in x_pct], fontsize=9)
    ax.set_xlim(46, 94)
    ax.set_xlabel("Shortcut correlation strength (%)", fontsize=10)
    ax.set_ylabel("Shortcut gap (%)", fontsize=10)

    # Match paper: range -17.5 to 0.5, ticks every 2.5
    ax.set_ylim(-17.5, 1.0)
    ax.set_yticks([-17.5, -15, -12.5, -10, -7.5, -5, -2.5, 0])

    ax.legend(frameon=True, fontsize=9, loc="lower left")
    ax.set_title("Separation emerges as correlation carries signal (Experiment I)",
                 fontsize=10.5, fontweight="bold")
    _style_ax(ax, grid_axis="y")
    save(fig, "fig11_correlation_ablation.pdf")


# =====================================================================
# FIGURE 12: Representation-Space Cliff (Experiment M)
# =====================================================================
def fig12():
    data = load("exp_m")
    if data is None:
        print("  SKIP (no exp_m.json found)")
        return False

    positions = data["positions_tested"]
    fig, ax = plt.subplots(figsize=(7.0, 3.5), constrained_layout=True)

    arch_cfg = [
        ("transformer", STYLE["trans"], "o", "Transformer"),
        ("lstm",        STYLE["lstm"],  "s", "LSTM"),
        ("mamba",       STYLE["mamba"], "^", "Mamba"),
    ]
    for arch, color, marker, label in arch_cfg:
        means, stds = [], []
        for pos in positions:
            sims = [r["positions"][f"frac_{pos}"]["final_token_cos_sim"]
                    for r in data["runs"] if r["arch"] == arch]
            means.append(np.mean(sims))
            stds.append(np.std(sims, ddof=1) if len(sims) > 1 else 0)
        means = np.array(means); stds = np.array(stds)
        ax.plot(positions, means, f"-{marker}", color=color, lw=2.0,
                markersize=7, markerfacecolor="white", markeredgecolor=color,
                markeredgewidth=1.6, label=label, zorder=5)
        ax.fill_between(positions, means - stds, means + stds,
                        color=color, alpha=0.10, linewidth=0, zorder=3)

    ax.fill_between([92.5, 101.5], 0.45, 1.06, alpha=0.08,
                    color=STYLE["mamba_dk"], linewidth=0, zorder=1)
    ax.text(96.5, 0.48, "cliff\nzone", ha="center", va="bottom", fontsize=8.5,
            color=STYLE["mamba_dk"], fontstyle="italic", fontweight="bold")
    ax.axhline(1.0, color=STYLE["spine"], lw=0.7, ls=":", zorder=2)

    # "Mamba: 0.999 to 0.618 (representation cliff)" annotation - matches paper
    ax.annotate("Mamba: 0.999 to 0.618\n(representation cliff)",
                xy=(100, 0.618), xytext=(80, 0.62),
                fontsize=8.5, color=STYLE["mamba_dk"], fontstyle="italic",
                ha="center",
                arrowprops=dict(arrowstyle="->", color=STYLE["mamba_dk"],
                                lw=1.0, connectionstyle="arc3,rad=-0.25"),
                bbox=dict(boxstyle="round,pad=0.4", fc="white",
                          ec=STYLE["mamba_dk"], alpha=0.95, linewidth=0.8),
                zorder=10)

    ax.set_xlabel("Shortcut position (% of sequence)", fontsize=10)
    ax.set_ylabel("Final-token cosine similarity\n(clean vs shortcut-inserted)", fontsize=10)
    ax.set_xlim(-3, 103); ax.set_ylim(0.45, 1.06)
    ax.set_xticks(positions)
    ax.legend(loc="lower left", frameon=True, fontsize=8.5)
    ax.set_title("Representation-space cliff mirrors accuracy cliff (Experiment M)",
                 fontsize=10, fontweight="bold")
    _style_ax(ax, grid_axis="y")
    save(fig, "fig_repr_cliff.pdf")
    return True


# =====================================================================
# FIGURE 13: Pretrained Backbone Control (Experiment L)
# =====================================================================
def fig13():
    data = load("exp_l")
    if data is None:
        print("  SKIP (no exp_l.json found)")
        return False

    # Exclude position 0 (pretrained model shows noisy artifacts there;
    # paper figure starts at position 25 for clarity)
    positions = [p for p in data["positions_tested"] if p > 0]
    fig, ax = plt.subplots(figsize=(7.0, 3.5), constrained_layout=True)

    # Pretrained flat curve
    means, stds = [], []
    for pos in positions:
        gaps = [r["positions"][f"frac_{pos}"]["gap"] for r in data["runs"]]
        means.append(np.mean(gaps))
        stds.append(np.std(gaps, ddof=1) if len(gaps) > 1 else 0)
    means = np.array(means); stds = np.array(stds)
    # Two-line palette complementing main paper figs (Wong colorblind-safe):
    # deep blue for pretrained (the "no-cliff" baseline), keep vermillion
    # for the from-scratch Mamba so its colour matches Mamba in figs 1-6.
    PRETRAINED_BLUE = "#0072B2"
    ax.plot(positions, means, "-D", color=PRETRAINED_BLUE, lw=2.0,
            markersize=7, markerfacecolor="white", markeredgecolor=PRETRAINED_BLUE,
            markeredgewidth=1.6, label="Pretrained Mamba-130M (frozen) + linear head",
            zorder=5)
    ax.fill_between(positions, means - stds, means + stds,
                    color=PRETRAINED_BLUE, alpha=0.13, linewidth=0, zorder=3)

    # Overlay from-scratch Mamba cliff from Exp H for comparison
    if eh is not None:
        fs_m = np.array([np.mean(h_mamba_pos[p]) for p in positions])
        fs_s = np.array([np.std(h_mamba_pos[p], ddof=1) for p in positions])
        ax.plot(positions, fs_m, "--^", color=STYLE["mamba"], lw=2.0,
                markersize=7, markerfacecolor="white", markeredgecolor=STYLE["mamba"],
                markeredgewidth=1.6, label="From-scratch custom Mamba (Exp H)",
                zorder=4)
        ax.fill_between(positions, fs_m - fs_s, fs_m + fs_s,
                        color=STYLE["mamba"], alpha=0.10, linewidth=0, zorder=2)

    ax.axhline(0, color=STYLE["spine"], lw=0.7, ls=":", zorder=2)

    # Cliff zone shaded (matches paper)
    ax.fill_between([92.5, 103], -16, 2, alpha=0.07,
                    color=STYLE["mamba_dk"], linewidth=0, zorder=1)

    # "Sharp cliff at 100%" annotation - shifted slightly right (xtext=85, was 78)
    # per user feedback, while keeping clear separation from the cliff curve.
    ax.annotate("Sharp cliff at 100%",
                xy=(100, -10.5), xytext=(85, -13.5),
                fontsize=8.5, color=STYLE["mamba_dk"], fontstyle="italic",
                ha="center",
                arrowprops=dict(arrowstyle="->", color=STYLE["mamba_dk"],
                                lw=1.0, connectionstyle="arc3,rad=0.3"),
                bbox=dict(boxstyle="round,pad=0.4", fc="white",
                          ec=STYLE["mamba_dk"], alpha=0.95, linewidth=0.8),
                zorder=10)

    ax.set_xlabel("Shortcut position (% of sequence)", fontsize=10)
    ax.set_ylabel("Shortcut gap (%)", fontsize=10)
    ax.set_xlim(20, 105); ax.set_xticks(positions)
    ax.set_ylim(-16, 2)
    ax.legend(loc="lower left", frameon=True, fontsize=9)
    ax.set_title("Cliff does not transfer to frozen pretrained backbone (Experiment L)",
                 fontsize=10.5, fontweight="bold")
    _style_ax(ax, grid_axis="y")
    save(fig, "fig_pretrained.pdf")
    return True


# =====================================================================
# FIGURE 14: Cliff Emergence During Training (Experiment P)
# =====================================================================
def fig14():
    # Rebuilt from scratch for visual clarity:
    # - Dual-axis layout (clean accuracy left in green, gap@100% right in red)
    # - Subtle fill bands, clean error bars, no axis-color tinting noise
    # - Critical window shading anchored to plot top, label inside the band
    # - Single combined legend below the plot
    data = load("exp_p")
    if data is None:
        print("  SKIP (no exp_p.json found)")
        return False

    checkpoint_epochs = data["checkpoint_epochs"]
    # Two-tone palette complementing main paper figs:
    # deep blue for clean accuracy (the "good" signal),
    # vermillion for shortcut gap (the "bad" signal that emerges).
    BLUE  = "#0072B2"
    RED   = "#D55E00"

    fig, ax_left = plt.subplots(figsize=(7.0, 3.8), constrained_layout=True)
    ax_right = ax_left.twinx()

    # Aggregate clean accuracy and gap@100% across seeds
    clean_means, clean_stds = [], []
    gap_means, gap_stds = [], []
    for ep in checkpoint_epochs:
        cleans, gaps = [], []
        for run in data["runs"]:
            for t in run["trajectory"]:
                if t["epoch"] == ep:
                    cleans.append(t["clean_acc"])
                    gaps.append(t["positions"]["frac_100"]["gap"])
        clean_means.append(np.mean(cleans) if cleans else np.nan)
        clean_stds.append(np.std(cleans, ddof=1) if len(cleans) > 1 else 0)
        gap_means.append(np.mean(gaps) if gaps else np.nan)
        gap_stds.append(np.std(gaps, ddof=1) if len(gaps) > 1 else 0)
    clean_means = np.array(clean_means); clean_stds = np.array(clean_stds)
    gap_means = np.array(gap_means);     gap_stds = np.array(gap_stds)

    # Plot bounds chosen to make both axes share visual range nicely
    LEFT_LIM  = (75, 105)
    RIGHT_LIM = (-15, 3)

    # Critical window shading - subtle, anchored to plot bounds
    ax_left.fill_between([4.7, 10.3], LEFT_LIM[0], LEFT_LIM[1],
                         alpha=0.09, color=STYLE["mamba_dk"],
                         linewidth=0, zorder=1)
    ax_left.text(7.5, LEFT_LIM[1] - 1.5, "cliff emergence window",
                 ha="center", va="top", fontsize=8.5,
                 color=STYLE["mamba_dk"], fontstyle="italic", fontweight="bold")

    # Reference: zero line on right axis (gap = 0)
    ax_right.axhline(0, color=STYLE["subtext"], lw=0.8, ls=":", zorder=2)

    # Left axis: clean accuracy (green, solid line, circle markers)
    ax_left.fill_between(checkpoint_epochs, clean_means - clean_stds,
                         clean_means + clean_stds,
                         alpha=0.15, color=BLUE, linewidth=0, zorder=3)
    line_clean = ax_left.errorbar(
        checkpoint_epochs, clean_means, yerr=clean_stds,
        fmt="-o", color=BLUE, lw=2.2, markersize=7.5,
        markerfacecolor="white", markeredgecolor=BLUE, markeredgewidth=1.6,
        capsize=3, capthick=0.9, elinewidth=0.9,
        label="Clean accuracy (left axis)", zorder=5)

    # Right axis: gap @ 100% (red, dashed line, triangle markers)
    ax_right.fill_between(checkpoint_epochs, gap_means - gap_stds,
                          gap_means + gap_stds,
                          alpha=0.15, color=RED, linewidth=0, zorder=3)
    line_gap = ax_right.errorbar(
        checkpoint_epochs, gap_means, yerr=gap_stds,
        fmt="--^", color=RED, lw=2.2, markersize=7.5,
        markerfacecolor="white", markeredgecolor=RED, markeredgewidth=1.6,
        capsize=3, capthick=0.9, elinewidth=0.9,
        label="Shortcut gap at 100% (right axis)", zorder=5)

    # Axes labels - colored to match their data
    ax_left.set_xlabel("Training epoch", fontsize=10)
    ax_left.set_ylabel("Clean accuracy (%)", fontsize=10, color=BLUE)
    ax_right.set_ylabel("Shortcut gap at 100% (%)", fontsize=10, color=RED)

    # Tick label colors
    ax_left.tick_params(axis="y", colors=BLUE, labelsize=8.5)
    ax_right.tick_params(axis="y", colors=RED, labelsize=8.5)

    # Spines: only show colored ones for left/right that match the data
    ax_left.spines["left"].set_color(BLUE)
    ax_left.spines["left"].set_linewidth(1.2)
    ax_right.spines["right"].set_visible(True)
    ax_right.spines["right"].set_color(RED)
    ax_right.spines["right"].set_linewidth(1.2)
    ax_right.spines["top"].set_visible(False)

    ax_left.set_xticks(checkpoint_epochs)
    ax_left.set_xlim(0, max(checkpoint_epochs) + 2)
    ax_left.set_ylim(*LEFT_LIM)
    ax_right.set_ylim(*RIGHT_LIM)

    # Combined legend below
    ax_left.legend(handles=[line_clean, line_gap],
                   labels=["Clean accuracy (left axis)",
                           "Shortcut gap at 100% (right axis)"],
                   loc="upper center", bbox_to_anchor=(0.5, -0.16),
                   ncol=2, frameon=False, fontsize=9)

    ax_left.set_title("Cliff emerges during task-learning window (Experiment P)",
                      fontsize=10.5, fontweight="bold")
    # Light horizontal grid only on left axis to avoid double-grid clutter
    _style_ax(ax_left, grid_axis="y")
    ax_right.grid(False)
    save(fig, "fig_cliff_emergence.pdf")
    return True


# =====================================================================
# RUN ALL
# =====================================================================
if __name__ == "__main__":
    figures = [
        ("fig1_position_ablation.pdf",      "Position cliff (Exp H)",                fig01),
        ("fig2_main_hierarchy.pdf",         "Main hierarchy (Exps A, B)",            fig02),
        ("fig3_per_scenario.pdf",           "Per-scenario breakdown (Exp G)",        fig03),
        ("fig4_scale_invariance.pdf",       "Scale invariance (Exp D)",              fig04),
        ("fig5_training_condition.pdf",     "Training condition (Exp E)",            fig05),
        ("fig6_posenc_ablation.pdf",        "Positional encoding ablation (Exp C)",  fig06),
        ("fig7_per_seed.pdf",               "Per-seed analysis (Exp B)",             fig07),
        ("fig8_shortcut_only.pdf",          "Shortcut-only baseline (Exp F)",        fig08),
        ("fig9_lstm_bimodality.pdf",        "LSTM training dynamics",                fig09),
        ("fig10_magic_seed.pdf",            "Mamba training dynamics (magic seed)",  fig10),
        ("fig11_correlation_ablation.pdf",  "Correlation strength ablation (Exp I)", fig11),
        ("fig_repr_cliff.pdf",              "Representation cliff (Exp M)",          fig12),
        ("fig_pretrained.pdf",              "Pretrained backbone (Exp L)",           fig13),
        ("fig_cliff_emergence.pdf",         "Cliff emergence (Exp P)",               fig14),
    ]

    total = len(figures)
    print(f"Generating up to {total} PDFs -> {os.path.abspath(OUTPUT_DIR)}/\n")
    np.random.seed(42)

    generated = 0
    skipped = 0
    for i, (name, desc, func) in enumerate(figures, 1):
        print(f"{i:2d}/{total}  {name:<35s} {desc}")
        try:
            result = func()
            if result is False:
                skipped += 1
            else:
                generated += 1
        except Exception as e:
            print(f"  FAIL {name}: {e}")
            skipped += 1

    print(f"\n{generated} PDFs generated, {skipped} skipped.")
    if skipped:
        print("Skipped figures had no JSON data available. Run the corresponding")
        print("experiment script first, then re-run this script.")
