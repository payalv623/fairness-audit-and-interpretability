"""Experiment 2: RBF SVM. Accuracy, demographic parity and equal opportunity by group."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.svm import SVC

from common import SEEDS, SENSITIVE, classification_row, fairness_table, load_splits, make_preprocessor

OUT = Path("results")
OUT.mkdir(exist_ok=True)
N_TRAIN = 15000   # RBF SVM fitting is roughly quadratic in rows, so train on a sample
GRID = [{"C": c, "gamma": g} for c in (0.1, 1, 10) for g in ("scale", 0.01)]


def fit_one(X, y, Xv, yv, params):
    m = SVC(kernel="rbf", cache_size=1000, **params).fit(X, y)
    return float((m.predict(Xv) == yv).mean()), params, m


def run(include_sensitive, train, val, test):
    prep = make_preprocessor(include_sensitive).fit(train)
    Xtr, Xva, Xte = [prep.transform(d) for d in (train, val, test)]
    ytr, yva, yte = train["y"].to_numpy(), val["y"].to_numpy(), test["y"].to_numpy()
    preds, picked, overall = [], [], []
    for s in SEEDS:
        idx = np.random.default_rng(s).choice(len(Xtr), N_TRAIN, replace=False)
        out = Parallel(n_jobs=len(GRID))(delayed(fit_one)(Xtr[idx], ytr[idx], Xva, yva, p) for p in GRID)
        acc, params, model = max(out, key=lambda t: t[0])   # picked on validation only
        p = model.predict(Xte)
        preds.append(p); picked.append({"seed": s, "val_acc": acc, **params})
        overall.append(classification_row(yte, p))
        print(f"sensitive={include_sensitive} seed={s} {params} val={acc:.4f} test={overall[-1]}", flush=True)
    return np.stack(preds), picked, overall, yte


def main():
    train, val, test = load_splits()
    summary, all_fair = {}, []
    # 'unaware' (sex and race removed) is the main run; 'aware' tests what removal achieves.
    for name, flag in (("SVM (no sex/race)", False), ("SVM (with sex/race)", True)):
        preds, picked, overall, yte = run(flag, train, val, test)
        np.save(OUT / f"exp2_preds_{'aware' if flag else 'unaware'}.npy", preds)
        for attr in SENSITIVE:
            all_fair += [dict(r, model=name) for r in fairness_table(yte, preds, test[attr].to_numpy(), attr)]
        summary[name] = {"picked": picked, "overall_by_seed": overall}
    summary["majority_baseline_accuracy"] = float(max(yte.mean(), 1 - yte.mean()))
    pd.DataFrame(all_fair).to_csv(OUT / "exp2_fairness.csv", index=False)
    (OUT / "exp2_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))
    print(pd.DataFrame(all_fair).round(3).to_string())


if __name__ == "__main__":
    main()
