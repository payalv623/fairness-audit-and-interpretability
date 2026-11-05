"""Shared data loading and fairness metrics for the Adult Income audit."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

SPLIT_SEED = 42            # the train/val/test split is the same for every experiment
SEEDS = [0, 1, 2]          # model seeds
N_BOOT = 2000
REF_GROUP = {"sex": "Male", "race": "White"}   # largest group of each attribute
NUMERIC = ["age", "education-num", "hours-per-week"]
HEAVY_TAILED = ["capital-gain", "capital-loss"]
CATEGORICAL = ["workclass", "marital-status", "occupation", "relationship", "native-country"]
SENSITIVE = ["sex", "race"]


def load_splits():
    """Return 60/20/20 train/val/test frames, stratified on label, sex and race."""
    d = fetch_openml("adult", version=2, as_frame=True, data_home="data_cache")
    df = d.data.copy()
    df["y"] = d.target.astype(str).str.strip().str.startswith(">50K").astype(int)
    # fnlwgt is a census sampling weight; education duplicates education-num.
    df = df.drop(columns=["fnlwgt", "education"])
    for c in CATEGORICAL + SENSITIVE:
        df[c] = df[c].astype(object).where(df[c].notna(), "Missing").astype(str)
    strat = df["y"].astype(str) + df["sex"] + df["race"]
    train, rest = train_test_split(df, test_size=0.4, stratify=strat, random_state=SPLIT_SEED)
    strat_rest = rest["y"].astype(str) + rest["sex"] + rest["race"]
    val, test = train_test_split(rest, test_size=0.5, stratify=strat_rest, random_state=SPLIT_SEED)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def make_preprocessor(include_sensitive=False):
    """Scale numerics, log-scale capital columns, one-hot encode categoricals."""
    cats = CATEGORICAL + (SENSITIVE if include_sensitive else [])
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("log", Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")), ("sc", StandardScaler())]), HEAVY_TAILED),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
    ])


def classification_row(y, pred):
    """Overall task metrics."""
    tp = np.sum((pred == 1) & (y == 1))
    fp = np.sum((pred == 1) & (y == 0))
    fn = np.sum((pred == 0) & (y == 1))
    return {
        "accuracy": float(np.mean(pred == y)),
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
    }


def group_rates(y, pred, mask):
    """Rates for one group. pred may be 2-D (seeds x rows): rates average over seeds."""
    p = pred[..., mask]
    yy = y[mask]
    pos, neg = yy == 1, yy == 0
    return {
        "n": int(mask.sum()),
        "n_pos": int(pos.sum()),
        "base_rate": float(yy.mean()),
        "accuracy": float(np.mean(p == yy)),
        "selection_rate": float(np.mean(p)),
        "tpr": float(np.mean(p[..., pos])) if pos.any() else np.nan,
        "fpr": float(np.mean(p[..., neg])) if neg.any() else np.nan,
    }


def fairness_table(y, pred, groups, attribute, seed=0):
    """Signed gaps of every group against the reference group, with bootstrap CIs.

    pred has shape (n_seeds, n_rows). Each resample redraws test rows and averages
    the gap over seeds, so the CI covers test-set sampling (seed spread is reported
    separately). Signed gaps are used because a bootstrapped absolute gap can never
    straddle 0, which would make "the CI excludes 0" meaningless.
    Race has several comparisons, so its CI is Bonferroni-adjusted.
    """
    ref = REF_GROUP[attribute]
    others = [g for g in pd.unique(groups) if g != ref]
    alpha = 0.05 / len(others)
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    rows = []
    for g in sorted(others):
        gm, rm = groups == g, groups == ref
        base_g, base_r = group_rates(y, pred, gm), group_rates(y, pred, rm)
        boots = {"dp": [], "eo": [], "fpr": []}
        for ii in idx:
            yb, pb, gb = y[ii], pred[:, ii], groups[ii]
            a, b = group_rates(yb, pb, gb == g), group_rates(yb, pb, gb == ref)
            boots["dp"].append(a["selection_rate"] - b["selection_rate"])
            boots["eo"].append(a["tpr"] - b["tpr"])
            boots["fpr"].append(a["fpr"] - b["fpr"])
        row = {
            "attribute": attribute, "group": g, "reference": ref, "n_group": base_g["n"],
            "n_group_pos": base_g["n_pos"],
            "acc_group": base_g["accuracy"], "acc_reference": base_r["accuracy"],
            "base_rate_gap": base_g["base_rate"] - base_r["base_rate"],
            "dp_gap": base_g["selection_rate"] - base_r["selection_rate"],
            "eo_gap": base_g["tpr"] - base_r["tpr"],
            "fpr_gap": base_g["fpr"] - base_r["fpr"],
            "ci_level": 1 - alpha,
        }
        for k in boots:
            lo, hi = np.nanquantile(boots[k], [alpha / 2, 1 - alpha / 2])
            row[f"{k}_lo"], row[f"{k}_hi"] = float(lo), float(hi)
        # 'Unfair' on equal opportunity: at least 0.10 apart and the CI excludes 0.
        row["eo_unfair"] = bool(abs(row["eo_gap"]) > 0.10 and (row["eo_lo"] > 0 or row["eo_hi"] < 0))
        rows.append(row)
    return rows
