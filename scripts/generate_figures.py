#!/usr/bin/env python3
"""Generate all publication figures from JSON result files.

Reads pre-computed data from results/provided/exp_{a..k}.json and writes
matplotlib-rendered PDFs to paper/. Run after results have been regenerated
or to re-style existing figures without re-running experiments.

Figures for Experiments L, M, and P are produced by their own standalone scripts:
  - ext_pretrained_mamba_cliff.py produces fig_pretrained.pdf (Experiment L)
  - ext_state_probing_v2.py produces fig_repr_cliff.pdf (Experiment M)
  - exp_p_cliff_emergence.py produces fig_cliff_emergence.pdf (Experiment P)
"""

import os, sys, json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import matplotlib.font_manager as fm

# =====================================================================
# PATHS
# =====================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
RESULT_DIR = os.path.join(REPO_ROOT, "results", "provided")
OUTPUT_DIR = os.path.join(REPO_ROOT, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load(name):
    with open(os.path.join(RESULT_DIR, f"{name}.json")) as f:
        return json.load(f)

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

seeds_5 = [42, 137, 256, 789, 1024]
seeds_3 = [42, 137, 256]

# Exp A gaps
sm_trans = _gaps(ea, "transformer"); sm_lstm = _gaps(ea, "lstm"); sm_mamba = _gaps(ea, "mamba")
# Exp B gaps
md_trans = _gaps(eb, "transformer"); md_lstm = _gaps(eb, "lstm"); md_mamba = _gaps(eb, "mamba")

# Exp C
pe_sm = [r["gap"] for r in ec["runs"] if r["scale"]=="small"]
pe_md = [r["gap"] for r in ec["runs"] if r["scale"]=="medium"]
reg_sm_3 = [_gaps_by_seed(ea, "mamba", seeds_3, "gap")[i] for i in range(3)]
reg_md_3 = [_gaps_by_seed(eb, "mamba", seeds_3, "gap")[i] for i in range(3)]

# Exp D
def _d_gaps(model, scale):
    return [r["gap"] for r in ed["runs"] if r["model"]==model and r["scale"]==scale]
m1_sm = _d_gaps("mamba1","small"); m1_md = _d_gaps("mamba1","medium")
m2_sm = _d_gaps("mamba2","small"); m2_md = _d_gaps("mamba2","medium"); m2_lg = _d_gaps("mamba2","large")

# Exp E
sc_trans = _gaps(ee, "transformer"); sc_lstm = _gaps(ee, "lstm"); sc_mamba = _gaps(ee, "mamba")

# Exp F
f_trans = [r["short_only"] for r in ef["runs"] if r["arch"]=="transformer"]
f_lstm  = [r["short_only"] for r in ef["runs"] if r["arch"]=="lstm"]
f_mamba = [r["short_only"] for r in ef["runs"] if r["arch"]=="mamba"]

# Exp G
g_true2move_trans = [r["short_acc"] for r in eg["true_2move_results"] if r["arch"]=="transformer"]
g_true2move_lstm  = [r["short_acc"] for r in eg["true_2move_results"] if r["arch"]=="lstm"]
g_true2move_mamba = [r["short_acc"] for r in eg["true_2move_results"] if r["arch"]=="mamba"]

# Exp H: 9-position sweep. New JSON uses positions dict.
# Positions tested: 0, 25, 50, 75, 80, 85, 90, 95, 100 (% of sequence)
H_POSITIONS = eh["positions_tested"]  # e.g., [0, 25, 50, 75, 80, 85, 90, 95, 100]

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
    fig, ax = plt.subplots(figsize=(3.5, 3.5), constrained_layout=True)
    lstm_min, lstm_max = min(md_lstm), max(md_lstm)
    mamba_min, mamba_max = min(md_mamba), max(md_mamba)
    ax.fill_between(seeds_5, lstm_min, lstm_max, alpha=0.12, color=STYLE["lstm"], linewidth=0, zorder=1)
    ax.fill_between(seeds_5, mamba_min, mamba_max, alpha=0.12, color=STYLE["mamba"], linewidth=0, zorder=1)
    ax.plot(seeds_5, md_trans, "-",  color=STYLE["trans"], lw=2.0,
            marker="s", markersize=6.5, markerfacecolor="white",
            markeredgecolor=STYLE["trans"], markeredgewidth=1.6, label="Transformer", zorder=5)
    ax.plot(seeds_5, md_lstm,  "--", color=STYLE["lstm"], lw=1.8,
            marker="o", markersize=6.5, markerfacecolor="white",
            markeredgecolor=STYLE["lstm"], markeredgewidth=1.6, label="LSTM", zorder=4)
    ax.plot(seeds_5, md_mamba, ":",  color=STYLE["mamba"], lw=1.8,
            marker="^", markersize=6.5, markerfacecolor="white",
            markeredgecolor=STYLE["mamba"], markeredgewidth=1.6, label="Mamba", zorder=4)
    ax.set_xlabel("Random seed", fontsize=10); ax.set_ylabel("Shortcut gap (%)", fontsize=10)
    ax.set_xticks(seeds_5); ax.set_xticklabels(["42","137","256","789","1024"], fontsize=8)
    ax.set_ylim(-22, 4)
    ax.axhline(0, color=STYLE["subtext"], lw=0.8, ls="--", zorder=2)
    ax.legend(frameon=True, fontsize=7, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.99))
    _annot(ax, "LSTM: bimodal,\nMamba: tight cluster",
           xy=(256, 0), xytext=(600, -4), color=STYLE["lstm_dk"], fontsize=7.5, rad=-0.2)
    _style_ax(ax, grid_axis="y")
    ax.set_title("Per-seed analysis (medium scale)", fontsize=10, fontweight="bold")
    save(fig, "fig7_per_seed.pdf")


# =====================================================================
# FIGURE 8: Shortcut-Only Baseline (Exp F)
# =====================================================================
def fig08():
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
    fig, ax = plt.subplots(figsize=(3.5, 3.2), constrained_layout=True)
    ax.fill_between([5, 10], -22, 4, alpha=0.08, color=STYLE["mamba_dk"], linewidth=0, zorder=1)
    ax.text(7.5, 3, "critical window", fontsize=7, color=STYLE["mamba_dk"],
            fontstyle="italic", fontweight="bold", ha="center")
    ax.plot(lstm_bimodal_epochs, lstm_s256_gap_over_time, "-",
            color="#228833", lw=2.0, marker="o", markersize=5.5,
            markerfacecolor="white", markeredgecolor="#228833", markeredgewidth=1.6,
            label="s256 (robust, 0.0%)", zorder=5)
    ax.plot(lstm_bimodal_epochs, lstm_s789_gap_over_time, "--",
            color=STYLE["lstm"], lw=1.8, marker="s", markersize=5.5,
            markerfacecolor="white", markeredgecolor=STYLE["lstm"], markeredgewidth=1.6,
            label="s789 (fragile, -18.3%)", zorder=4)
    ax.text(40, lstm_s256_gap_over_time[-1] + 1.5, f"{lstm_s256_gap_over_time[-1]:.1f}%",
            ha="left", fontsize=7.5, color="#228833", fontweight="bold")
    ax.text(40, lstm_s789_gap_over_time[-1] - 1.5, f"{lstm_s789_gap_over_time[-1]:.1f}%",
            ha="left", fontsize=7.5, color=STYLE["lstm"], fontweight="bold")
    ax.axhline(0, color=STYLE["subtext"], lw=0.8, ls=":", zorder=2)
    _annot(ax, "Fate decided in 5 epochs:\nrobust vs fragile is permanent",
           xy=(10, -18.0), xytext=(21.25, -13.0), color=STYLE["mamba_dk"], fontsize=7.5, rad=-0.2)
    ax.set_xlabel("Epoch", fontsize=10); ax.set_ylabel("Shortcut gap (%)", fontsize=10)
    ax.set_ylim(-22, 4); ax.set_xticks(lstm_bimodal_epochs)
    ax.legend(loc="center right", fontsize=7, frameon=True, bbox_to_anchor=(0.98, 0.55))
    ax.set_title("LSTM bimodality (medium scale)", fontsize=10, fontweight="bold")
    _style_ax(ax, grid_axis="y")
    save(fig, "fig9_lstm_bimodality.pdf")


# =====================================================================
# FIGURE 10: Magic Seed Training Dynamics
# =====================================================================
def fig10():
    fig, ax = plt.subplots(figsize=(3.5, 3.2), constrained_layout=True)
    ax.fill_between([5, 10], -18, 4, alpha=0.08, color=STYLE["mamba_dk"], linewidth=0, zorder=1)
    ax.text(7.5, 2.5, "critical window", fontsize=7, color=STYLE["mamba_dk"],
            fontstyle="italic", fontweight="bold", ha="center")
    ax.plot(magic_s256_epochs, magic_s256_gaps, "-",
            color="#228833", lw=2.0, marker="o", markersize=5.5,
            markerfacecolor="white", markeredgecolor="#228833", markeredgewidth=1.6,
            label="s256 (magic, 0.0%)", zorder=5)
    ax.plot(magic_s42_epochs, magic_s42_gaps, "--",
            color=STYLE["mamba"], lw=1.8, marker="^", markersize=5.5,
            markerfacecolor="white", markeredgecolor=STYLE["mamba"], markeredgewidth=1.6,
            label="s42 (fragile, -13.7%)", zorder=4)
    ax.text(45, magic_s256_gaps[-1] + 0.8, f"{magic_s256_gaps[-1]:.1f}%",
            ha="right", fontsize=7.5, color="#228833", fontweight="bold")
    ax.text(41, magic_s42_gaps[-1] - 1.2, f"{magic_s42_gaps[-1]:.1f}%",
            ha="left", fontsize=7.5, color=STYLE["mamba"], fontweight="bold")
    ax.axhline(0, color=STYLE["subtext"], lw=0.8, ls=":", zorder=2)
    _annot(ax, "1 in 18 Mamba seeds\nfinds the robust minimum",
           xy=(10, 0.33), xytext=(17, -4), color=STYLE["mamba_dk"], fontsize=7.5, rad=-0.2)
    ax.set_xlabel("Epoch", fontsize=10); ax.set_ylabel("Shortcut gap (%)", fontsize=10)
    ax.set_ylim(-18, 4); ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40, 45])
    ax.legend(loc="center right", fontsize=7, frameon=True, bbox_to_anchor=(0.98, 0.45))
    ax.set_title("Magic seed: Mamba-1 small", fontsize=10, fontweight="bold")
    _style_ax(ax, grid_axis="y")
    save(fig, "fig10_magic_seed.pdf")


# =====================================================================
# RUN ALL
# =====================================================================
if __name__ == "__main__":
    print(f"Generating 10 PDFs -> {os.path.abspath(OUTPUT_DIR)}/\n")
    np.random.seed(42)
    print(" 1/10  fig1_position_ablation.pdf   [figure*, 7.0in]   HERO"); fig01()
    print(" 2/10  fig2_main_hierarchy.pdf      [figure*, 7.0in]"); fig02()
    print(" 3/10  fig3_per_scenario.pdf        [figure*, 7.0in]"); fig03()
    print(" 4/10  fig4_scale_invariance.pdf    [figure*, 7.0in]"); fig04()
    print(" 5/10  fig5_training_condition.pdf  [figure*, 7.0in]"); fig05()
    print(" 6/10  fig6_posenc_ablation.pdf     [figure*, 7.0in]"); fig06()
    print(" 7/10  fig7_per_seed.pdf            [figure,  3.5in]"); fig07()
    print(" 8/10  fig8_shortcut_only.pdf       [figure,  3.5in]"); fig08()
    print(" 9/10  fig9_lstm_bimodality.pdf     [figure,  3.5in]  supplementary"); fig09()
    print("10/10  fig10_magic_seed.pdf         [figure,  3.5in]  supplementary"); fig10()
    print(f"\nAll 10 PDFs generated from JSON results.")
