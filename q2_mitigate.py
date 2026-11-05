"""Question 2: make the autoencoder-based classifier fair, then re-audit it on the same test set."""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import DemographicParity, ExponentiatedGradient, TruePositiveRateParity
from sklearn.linear_model import LogisticRegression

from common import N_BOOT, SEEDS, classification_row, load_splits, make_preprocessor
from exp1_autoencoder import BATCH, DEVICE, LATENT, MAX_EPOCHS, PATIENCE, AE, encode, fit_ae, row_errors

warnings.filterwarnings("ignore")
OUT = Path("results/q2")
OUT.mkdir(parents=True, exist_ok=True)
MIN_POS = 30   # below this many true high earners in test, an equal opportunity gap is reported as inconclusive
# Groups used for the validation-based choice: those with enough high earners to measure.
SELECT_GROUPS = [("sex", "Female", "Male"), ("race", "Black", "White"), ("race", "Asian-Pac-Islander", "White")]


def add_groups(d):
    """Add the group columns. Very small race groups are merged for mitigation only, not for reporting."""
    d = d.copy()
    d["race4"] = d["race"].replace({"Amer-Indian-Eskimo": "AmerInd/Other", "Other": "AmerInd/Other"})
    d["sex_race4"] = d["race4"] + " " + d["sex"]
    d["sex_race"] = d["race"] + " " + d["sex"]
    return d


def fit_ae_balanced(Xtr, Xva, groups_tr, groups_va, k, seed, power):
    """Autoencoder trained with group-balanced sampling: P(row) proportional to (1 / group size) ** power."""
    torch.manual_seed(seed)
    to_t = lambda a: torch.tensor(a, dtype=torch.float32, device=DEVICE)
    Xtr_t, Xva_t = to_t(Xtr), to_t(Xva)
    counts = pd.Series(groups_tr).map(pd.Series(groups_tr).value_counts()).to_numpy()
    probs = torch.tensor((1 / counts) ** power, dtype=torch.double)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    va_codes = pd.Series(groups_va).astype("category").cat.codes.to_numpy()
    model = AE(Xtr.shape[1], k).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, state, wait = np.inf, None, 0
    for _ in range(MAX_EPOCHS):
        model.train()
        draw = torch.multinomial(probs, len(Xtr), replacement=True, generator=gen)
        for b in draw.split(BATCH):
            xb = Xtr_t[b.to(DEVICE)]
            loss = ((model(xb) - xb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            e = ((model(Xva_t) - Xva_t) ** 2).mean(1).cpu().numpy()
        v = float(np.mean([e[va_codes == c].mean() for c in np.unique(va_codes)]))  # mean of group means
        if v < best - 1e-5:
            best, wait = v, 0
            state = {n: p.detach().clone() for n, p in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(state)
    return model.eval()


def reweigh(y, s):
    """Reweighing (Kamiran and Calders): make label and group look statistically independent."""
    d = pd.DataFrame({"y": y, "s": s})
    p_y, p_s, p_sy = d.y.map(d.y.value_counts(normalize=True)), d.s.map(d.s.value_counts(normalize=True)), d.groupby(["s", "y"]).y.transform("size") / len(d)
    return (p_y * p_s / p_sy).to_numpy()


class LR64(LogisticRegression):
    """Same model; probabilities returned as float64, which fairlearn's thresholder requires."""

    def predict_proba(self, X):
        return super().predict_proba(X).astype(np.float64)


def run_methods(Z, data, seed):
    """Return {method: predict(Z, frame)} for one seed. Predictors see codes; some also see the group."""
    tr, va_a = data["train"], data["val_a"]
    Ztr, Zva = Z["train"], Z["val_a"]
    y = tr["y"].to_numpy()
    lr = lambda: LogisticRegression(max_iter=2000)
    base = LR64(max_iter=2000).fit(Ztr, y)
    out = {"Before (AE + LR)": lambda z, f: base.predict(z)}
    rw = lr().fit(Ztr, y, sample_weight=reweigh(y, tr["sex_race4"].to_numpy()))
    out["Reweighing"] = lambda z, f: rw.predict(z)
    def eg(constraint, col):
        m = ExponentiatedGradient(lr(), constraints=constraint, eps=0.01, max_iter=30)
        m.fit(Ztr, y, sensitive_features=tr[col].to_numpy())
        return lambda z, f: m.predict(z, random_state=seed)
    out["EG equal opportunity (sex)"] = eg(TruePositiveRateParity(difference_bound=0.01), "sex")
    out["EG equal opportunity (sex x race)"] = eg(TruePositiveRateParity(difference_bound=0.01), "sex_race4")
    out["EG demographic parity (sex)"] = eg(DemographicParity(difference_bound=0.01), "sex")
    # Post-processing is fitted on one half of validation; the other half is used for choosing.
    to = ThresholdOptimizer(estimator=base, constraints="true_positive_rate_parity", objective="accuracy_score",
                            prefit=True, predict_method="predict_proba")
    to.fit(Zva, va_a["y"].to_numpy(), sensitive_features=va_a["sex_race4"].to_numpy())
    out["Group thresholds (sex x race)"] = lambda z, f: to.predict(z, sensitive_features=f["sex_race4"].to_numpy(), random_state=seed)
    return out


class Boot:
    """Paired bootstrap: the same resampled test rows for every method, so reductions are comparable."""

    def __init__(self, n, seed=0):
        rng = np.random.default_rng(seed)
        self.W = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(N_BOOT)]).astype(np.float32)

    def rate(self, value, mask):
        """Weighted mean of value within mask, for every resample."""
        return (self.W @ (value * mask).astype(np.float32)) / np.maximum(self.W @ mask.astype(np.float32), 1)


def gap_samples(boot, y, p, gm, rm):
    pos, neg = y == 1, y == 0
    return {
        "dp": boot.rate(p, gm) - boot.rate(p, rm),
        "eo": boot.rate(p, gm & pos) - boot.rate(p, rm & pos),
        "fpr": boot.rate(p, gm & neg) - boot.rate(p, rm & neg),
    }


def audit(boot, y, preds_by_method, test):
    """Point estimates, CIs and paired reductions against the baseline, for each comparison."""
    rows = []
    specs = [("sex", "Male"), ("race", "White"), ("sex_race", "White Male")]
    for attr, ref in specs:
        groups = test[attr].to_numpy()
        others = sorted(g for g in np.unique(groups) if g != ref)
        alpha = 0.05 / len(others)
        for g in others:
            gm, rm = groups == g, groups == ref
            pos, neg = y == 1, y == 0
            base_s = gap_samples(boot, y, preds_by_method["Before (AE + LR)"].mean(0), gm, rm)
            for method, pred in preds_by_method.items():
                p = pred.mean(0)
                s = gap_samples(boot, y, p, gm, rm)
                r = {"method": method, "attribute": attr, "group": g, "reference": ref, "n": int(gm.sum()),
                     "n_pos": int((gm & pos).sum()), "ci_level": 1 - alpha,
                     "base_rate_gap": float(y[gm].mean() - y[rm].mean()), "accuracy_group": float((pred[:, gm] == y[gm]).mean()),
                     "accuracy_reference": float((pred[:, rm] == y[rm]).mean()),
                     "tpr_group": float(p[gm & pos].mean()), "tpr_reference": float(p[rm & pos].mean()),
                     "fpr_group": float(p[gm & neg].mean()), "fpr_reference": float(p[rm & neg].mean()),
                     "selection_group": float(p[gm].mean()), "selection_reference": float(p[rm].mean())}
                r["dp_gap"] = r["selection_group"] - r["selection_reference"]
                r["eo_gap"] = r["tpr_group"] - r["tpr_reference"]
                r["fpr_gap"] = r["fpr_group"] - r["fpr_reference"]
                for k in ("dp", "eo", "fpr"):
                    r[f"{k}_lo"], r[f"{k}_hi"] = [float(v) for v in np.nanquantile(s[k], [alpha / 2, 1 - alpha / 2])]
                    red = np.abs(base_s[k]) - np.abs(s[k])     # positive means the gap shrank
                    r[f"{k}_reduction"] = abs(r[f"{k}_gap"]) * -1 + abs(float(_before(preds_by_method, y, gm, rm, k)))
                    r[f"{k}_red_lo"], r[f"{k}_red_hi"] = [float(v) for v in np.nanquantile(red, [alpha / 2, 1 - alpha / 2])]
                r["measurable"] = r["n_pos"] >= MIN_POS
                r["eo_unfair"] = bool(abs(r["eo_gap"]) > 0.10 and (r["eo_lo"] > 0 or r["eo_hi"] < 0))
                rows.append(r)
    return rows


def _before(preds_by_method, y, gm, rm, k):
    p = preds_by_method["Before (AE + LR)"].mean(0)
    if k == "dp":
        return p[gm].mean() - p[rm].mean()
    lab = 1 if k == "eo" else 0
    return p[gm & (y == lab)].mean() - p[rm & (y == lab)].mean()


def val_scores(method_preds, y, frame):
    """Accuracy and worst measurable equal opportunity gap, for the validation-based choice."""
    p = method_preds.mean(0)
    gaps = {}
    for attr, g, ref in SELECT_GROUPS:
        a, b = (frame[attr] == g).to_numpy() & (y == 1), (frame[attr] == ref).to_numpy() & (y == 1)
        gaps[g] = float(p[a].mean() - p[b].mean())
    return {"val_accuracy": float((method_preds == y).mean()), "val_worst_eo": max(abs(v) for v in gaps.values()), **{f"val_eo_{k}": v for k, v in gaps.items()}}


def main():
    train, val, test = [add_groups(d) for d in load_splits()]
    # Half of validation fits post-processing; the other half is only used to choose a method.
    rng = np.random.default_rng(42)
    order = rng.permutation(len(val))
    va_a, va_b = val.iloc[order[: len(val) // 2]].reset_index(drop=True), val.iloc[order[len(val) // 2:]].reset_index(drop=True)
    data = {"train": train, "val_a": va_a, "val_b": va_b, "test": test}
    prep = make_preprocessor(False).fit(train)
    X = {k: prep.transform(v) for k, v in data.items()}
    Xva_all = prep.transform(val)
    y_te, y_vb = test["y"].to_numpy(), va_b["y"].to_numpy()

    # Same seeds as Question 1. Checked against its saved predictions below.
    ae_std = {s: fit_ae(X["train"], Xva_all, LATENT, s) for s in SEEDS}
    test_preds, val_preds = {}, {}
    for s, ae in ae_std.items():
        Z = {k: encode(ae, X[k]) for k in X}
        for name, fn in run_methods(Z, data, s).items():
            test_preds.setdefault(name, []).append(fn(Z["test"], test))
            val_preds.setdefault(name, []).append(fn(Z["val_b"], va_b))
        print("seed", s, "done", flush=True)

    # Group-balanced autoencoders, for the reconstruction disparity.
    rec_err = {"Before (AE)": np.mean([row_errors(m, prep.transform(test)) for m in ae_std.values()], axis=0)}
    for label, power in (("Balanced AE (sqrt weights)", 0.5), ("Balanced AE (full weights)", 1.0)):
        ms = [fit_ae_balanced(X["train"], Xva_all, train["race"].to_numpy(), val["race"].to_numpy(), LATENT, s, power) for s in SEEDS]
        rec_err[label] = np.mean([row_errors(m, X["test"]) for m in ms], axis=0)
        cl = []
        vl = []
        for s, m in zip(SEEDS, ms):
            clf = LogisticRegression(max_iter=2000).fit(encode(m, X["train"]), train["y"].to_numpy())
            cl.append(clf.predict(encode(m, X["test"])))
            vl.append(clf.predict(encode(m, X["val_b"])))
        test_preds[f"{label} + LR"] = cl
        val_preds[f"{label} + LR"] = vl
        print(label, "done", flush=True)

    test_preds = {k: np.stack(v) for k, v in test_preds.items()}
    val_preds = {k: np.stack(v) for k, v in val_preds.items()}
    q1 = np.load("results/exp1_preds.npy")
    same = float((q1 == test_preds["Before (AE + LR)"]).mean())
    print(f"baseline matches Question 1 predictions on {100 * same:.2f}% of seed-row pairs", flush=True)
    np.savez_compressed(OUT / "test_preds.npz", **{k.replace(" ", "_"): v for k, v in test_preds.items()})

    overall = []
    for m, p in test_preds.items():
        o = pd.DataFrame([classification_row(y_te, q) for q in p])
        overall.append({"method": m, **o.mean().to_dict(), "accuracy_sd": float(o["accuracy"].std()),
                        **val_scores(val_preds[m], y_vb, va_b),
                        "uses_group_at_prediction": m.startswith("Group thresholds")})
    overall = pd.DataFrame(overall)
    # Choice rule, fixed before looking at test results: among methods that do not need the person's group
    # at prediction time and are not the baseline, keep those with worst validation EO gap <= 0.10, then
    # take the highest validation accuracy. If none qualify, take the smallest worst gap.
    elig = overall[(~overall["uses_group_at_prediction"]) & (overall["method"] != "Before (AE + LR)") & (~overall["method"].str.contains("demographic"))]
    ok = elig[elig["val_worst_eo"] <= 0.10]
    chosen = ok.sort_values("val_accuracy", ascending=False).iloc[0]["method"] if len(ok) else elig.sort_values("val_worst_eo").iloc[0]["method"]
    overall["chosen"] = overall["method"] == chosen
    overall.to_csv(OUT / "overall.csv", index=False)

    fair = pd.DataFrame(audit(Boot(len(test)), y_te, test_preds, test))
    fair.to_csv(OUT / "fairness.csv", index=False)

    rec_rows = []
    for m, e in rec_err.items():
        for attr, ref in (("sex", "Male"), ("race", "White")):
            gv = test[attr].to_numpy()
            for g in sorted(np.unique(gv)):
                rec_rows.append({"method": m, "attribute": attr, "group": g, "n": int((gv == g).sum()), "error": float(e[gv == g].mean()),
                                 "ratio_to_reference": float(e[gv == g].mean() / e[gv == ref].mean())})
    rec = pd.DataFrame(rec_rows)
    rec.to_csv(OUT / "reconstruction.csv", index=False)
    (OUT / "meta.json").write_text(json.dumps({"chosen": chosen, "baseline_match_q1": same}, indent=2))

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(overall[["method", "accuracy", "precision", "recall", "f1", "val_accuracy", "val_worst_eo", "chosen"]].round(3).to_string())
    print(fair[fair.attribute != "sex_race"][["method", "group", "n_pos", "acc" + "uracy_group", "tpr_group", "tpr_reference", "eo_gap", "eo_lo", "eo_hi", "eo_reduction", "eo_red_lo", "eo_red_hi", "dp_gap", "dp_lo", "dp_hi", "base_rate_gap", "eo_unfair"]].round(3).to_string())
    print(rec.pivot_table(index="group", columns="method", values="error").round(4).to_string())


if __name__ == "__main__":
    main()
