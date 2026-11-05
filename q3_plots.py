"""Figures for Question 3."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from make_plots import BLUE, GRID, INK, MUTED, ORANGE, SURFACE

AQUA, YELLOW = "#1baf7a", "#eda100"
R = "results/q3/"
gaps = pd.read_csv(R + "attribution_gaps.csv")
by = pd.read_csv(R + "attribution_by_group.csv")
scr = pd.read_csv(R + "scramble_test.csv")
AE, EG, N0, NF = "AE + LR (before)", "AE + EG (after)", "Network, no fair loss (before)", "Network, fair loss (after)"


def fig_shap_gap():
    """Which features pull women's scores down relative to men's, among true high earners."""
    x = gaps[(gaps.scope == "high earners") & (gaps.comparison == "Women vs men")]
    order = x[x.model == AE].assign(m=lambda d: d.shap_gap.abs()).sort_values("m").feature.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True, sharex=True)
    for ax, (b, a, title) in zip(axes, [(AE, EG, "Autoencoder + classifier: EG fix"), (N0, NF, "End-to-end network: fair loss")]):
        for y, f in enumerate(order):
            rb, ra = x[(x.model == b) & (x.feature == f)].iloc[0], x[(x.model == a) & (x.feature == f)].iloc[0]
            ax.plot([rb.shap_gap, ra.shap_gap], [y, y], color=GRID, lw=3, zorder=1, solid_capstyle="round")
            ax.plot(rb.shap_gap, y, "s", color=ORANGE, ms=8, mec=SURFACE, mew=1.5, zorder=3, label="Before" if y == 0 else None)
            ax.plot(ra.shap_gap, y, "o", color=BLUE, ms=8, mec=SURFACE, mew=1.5, zorder=3, label="After" if y == 0 else None)
        ax.axvline(0, color=MUTED, lw=1)
        ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
    axes[0].set_yticks(range(len(order)), order)
    axes[0].legend(frameon=False, loc="lower left")
    fig.supxlabel("SHAP gap in log-odds, women minus men among true high earners (negative pulls women down)", fontsize=10, color=MUTED)
    fig.tight_layout(); fig.savefig(R + "fig1_shap_gap_women_men.png", dpi=180); plt.close(fig)


def fig_scramble():
    """Equal opportunity gap after scrambling a feature, without retraining."""
    x = scr[scr.comparison == "Women vs men"]
    rows = ["(nothing scrambled)", "hours-per-week", "marital-status", "relationship", "relationship + marital-status"]
    styles = [(AE, ORANGE, "s"), (EG, BLUE, "o"), (N0, YELLOW, "D"), (NF, AQUA, "^")]
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    for i, f in enumerate(rows):
        y0 = len(rows) - 1 - i
        for j, (m, c, mk) in enumerate(styles):
            v = x[(x.model == m) & (x.feature == f)].eo_gap.iloc[0]
            ax.plot(v, y0 + 0.24 - 0.16 * j, mk, color=c, ms=8, mec=SURFACE, mew=1.2, label=m if i == 0 else None)
    ax.axvline(0, color=MUTED, lw=1)
    for v in (-0.1, 0.1):
        ax.axvline(v, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.set_yticks(range(len(rows)), rows[::-1])
    ax.set_xlabel("Equal opportunity gap, women vs men (dashed lines mark the 0.10 cutoff)")
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=2, fontsize=9)
    ax.set_title("Scrambling relationship and marital status removes the gap", loc="left", fontweight="bold", fontsize=11)
    fig.tight_layout(); fig.savefig(R + "fig2_scramble_test.png", dpi=180); plt.close(fig)


def fig_techniques():
    """Share of total importance per feature: SHAP against raw gradient magnitude."""
    x = by[(by.model == N0) & (by.group == "Everyone") & (by.scope == "high earners")].copy()
    x["shap_share"] = 100 * x.shap_abs / x.shap_abs.sum()
    x["grad_share"] = 100 * x.gradient / x.gradient.sum()
    x = x.sort_values("shap_share")
    y = np.arange(len(x))
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.barh(y + 0.19, x.shap_share, 0.34, color=BLUE, label="SHAP (mean absolute value)")
    ax.barh(y - 0.19, x.grad_share, 0.34, color=ORANGE, label="Vanilla gradient (magnitude)")
    ax.set_yticks(y, x.feature)
    ax.set_xlabel("Share of total importance (%)")
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.14), ncol=2)
    ax.set_title("Raw gradients favour native-country; SHAP gives it almost nothing", loc="left", fontweight="bold", fontsize=11)
    fig.tight_layout(); fig.savefig(R + "fig3_shap_vs_gradient.png", dpi=180); plt.close(fig)


if __name__ == "__main__":
    fig_shap_gap()
    fig_scramble()
    fig_techniques()
