"""Before and after figures for Question 2."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from make_plots import BLUE, GRID, INK, MUTED, ORANGE, SURFACE
from q2_mitigate import OUT

f = pd.read_csv(OUT / "fairness.csv")
ov = pd.read_csv(OUT / "overall.csv")
CHOSEN = json.loads((OUT / "meta.json").read_text())["chosen"]
BEFORE = "Before (AE + LR)"


def fig_gaps():
    """Equal opportunity gap before and after, for groups with enough high earners to measure."""
    groups = [("sex", "Female", "Women vs men"), ("sex_race", "White Female", "White women vs White men"),
              ("sex_race", "Black Male", "Black men vs White men"), ("race", "Black", "Black vs White"),
              ("race", "Asian-Pac-Islander", "Asian-Pac vs White"), ("sex_race", "Asian-Pac-Islander Male", "Asian-Pac men vs White men")]
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    for i, (a, g, _) in enumerate(groups):
        y = len(groups) - 1 - i
        get = lambda m: f[(f.method == m) & (f.attribute == a) & (f.group == g)].iloc[0]
        b, c = get(BEFORE), get(CHOSEN)
        ax.plot([b.eo_gap, c.eo_gap], [y, y], color=GRID, lw=3, zorder=1, solid_capstyle="round")
        ax.plot([c.eo_lo, c.eo_hi], [y - 0.2, y - 0.2], color=BLUE, lw=2, solid_capstyle="round", zorder=2)
        ax.plot([b.eo_lo, b.eo_hi], [y + 0.2, y + 0.2], color=ORANGE, lw=2, solid_capstyle="round", zorder=2)
        ax.plot(b.eo_gap, y + 0.2, "s", color=ORANGE, ms=8, mec=SURFACE, mew=1.5, zorder=3, label="Before" if i == 0 else None)
        ax.plot(c.eo_gap, y - 0.2, "o", color=BLUE, ms=8, mec=SURFACE, mew=1.5, zorder=3, label="After" if i == 0 else None)
    ax.axvline(0, color=MUTED, lw=1)
    for v in (-0.1, 0.1):
        ax.axvline(v, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.set_yticks(range(len(groups)), [g[2] for g in groups[::-1]])
    ax.set_xlabel("Equal opportunity gap, with confidence interval (dashed lines mark the 0.10 cutoff)")
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("The fix closes the gap for women but not for Black men", loc="left", fontweight="bold", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "fig1_before_after_gaps.png", dpi=180); plt.close(fig)


def fig_tpr():
    """Share of real high earners recognised, showing who gains and who loses under each method."""
    methods = [(BEFORE, "Before"), (CHOSEN, "Chosen: EG equal\nopportunity (sex)"), ("Reweighing", "Reweighing"),
               ("Group thresholds (sex x race)", "Group thresholds\n(sex x race)"),
               ("EG equal opportunity (sex x race)", "EG equal opportunity\n(sex x race)"), ("EG demographic parity (sex)", "EG demographic\nparity (sex)")]
    rows = [f[(f.method == m) & (f.group == "Female") & (f.attribute == "sex")].iloc[0] for m, _ in methods]
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    a = ax.bar(x - 0.19, [100 * r.tpr_group for r in rows], 0.34, color=BLUE, label="Women")
    b = ax.bar(x + 0.19, [100 * r.tpr_reference for r in rows], 0.34, color=ORANGE, label="Men")
    ax.bar_label(a, fmt="%.0f", padding=2, color=INK, fontsize=9)
    ax.bar_label(b, fmt="%.0f", padding=2, color=INK, fontsize=9)
    ax.set_xticks(x, [m[1] for m in methods], fontsize=8.5)
    ax.tick_params(axis="x", colors=INK)
    ax.set_ylabel("High earners correctly identified (%)"); ax.set_ylim(0, 72)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", ncol=2)
    ax.set_title("Equal rates can come from lowering men's rate as well as raising women's", loc="left", fontweight="bold", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "fig2_who_gains.png", dpi=180); plt.close(fig)


if __name__ == "__main__":
    fig_gaps()
    fig_tpr()
