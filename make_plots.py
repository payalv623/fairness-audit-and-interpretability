"""Build the report figures from the results CSVs."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("results")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "text.color": INK, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.spines.top": False, "axes.spines.right": False,
})
ORDER = ["Female", "Black", "Amer-Indian-Eskimo", "Other", "Asian-Pac-Islander"]
LABEL = {"Female": "Female vs Male", "Black": "Black vs White", "Amer-Indian-Eskimo": "Amer-Indian vs White",
         "Other": "Other vs White", "Asian-Pac-Islander": "Asian-Pac vs White"}


def fig_reconstruction():
    d = pd.read_csv(OUT / "exp1_reconstruction.csv")
    d["label"] = d["group"].replace({"Amer-Indian-Eskimo": "Amer-Indian", "Asian-Pac-Islander": "Asian-Pac"})
    d = d.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    y = np.arange(len(d))
    ax.barh(y + 0.19, d["shared_error"], 0.34, color=BLUE, label="Autoencoder")
    ax.barh(y - 0.19, d["pca_error"], 0.34, color=ORANGE, label="PCA (same size code)")
    ax.set_yticks(y, [f"{a} (n={b:,})" for a, b in zip(d["label"], d["n_test"])])
    ax.set_xlabel("Mean reconstruction error on test data (lower is better)")
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Reconstruction error is highest for the Asian-Pac group", loc="left", fontweight="bold", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "fig1_reconstruction.png", dpi=180); plt.close(fig)


def fig_gaps():
    d = pd.concat([pd.read_csv(OUT / "exp1_fairness.csv"), pd.read_csv(OUT / "exp2_fairness.csv")])
    models = [("SVM (no sex/race)", BLUE, "o"), ("SVM (with sex/race)", ORANGE, "s"), ("AE latent + LR", AQUA, "^")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
    for ax, (key, title) in zip(axes, [("eo", "Equal opportunity gap (true positive rate)"),
                                       ("dp", "Demographic parity gap (predicted >$50K rate)")]):
        for i, g in enumerate(ORDER):
            y0 = len(ORDER) - 1 - i
            for j, (m, c, mk) in enumerate(models):
                r = d[(d["model"] == m) & (d["group"] == g)].iloc[0]
                yy = y0 + 0.22 - j * 0.22
                ax.plot([r[f"{key}_lo"], r[f"{key}_hi"]], [yy, yy], color=c, lw=2, solid_capstyle="round")
                ax.plot(r[f"{key}_gap"], yy, mk, color=c, ms=8, mec=SURFACE, mew=1.5, label=m if i == 0 else None)
            if key == "dp":
                b = d[d["group"] == g]["base_rate_gap"].iloc[0]
                ax.plot([b, b], [y0 - 0.38, y0 + 0.38], color=INK, lw=1.6,
                        label="Perfect model (true income gap)" if i == 0 else None)
        ax.axvline(0, color=MUTED, lw=1)
        if key == "eo":
            for v in (-0.1, 0.1):
                ax.axvline(v, color=MUTED, lw=1, ls=(0, (3, 3)))
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Group minus reference group")
        ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(ORDER)), [LABEL[g] for g in ORDER[::-1]])
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=4, loc="lower center")
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(OUT / "fig2_fairness_gaps.png", dpi=180); plt.close(fig)


if __name__ == "__main__":
    fig_reconstruction()
    fig_gaps()
