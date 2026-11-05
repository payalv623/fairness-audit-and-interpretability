"""Question 3: do particular features drive the unfairness, and does that change after fair optimization?

Techniques: exact SHAP over the 10 original features, and vanilla gradients of the log-odds.
Models: AE + LR (Q1), EG on the same encoder (Q2), and an end-to-end network with and without a fairness loss.
"""
import copy
import json
import math
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch
from fairlearn.reductions import ExponentiatedGradient, TruePositiveRateParity
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from torch import nn

from common import SEEDS, classification_row, load_splits, make_preprocessor
from exp1_autoencoder import DEVICE, LATENT, encode, fit_ae
from q2_mitigate import Boot, add_groups, audit, val_scores

warnings.filterwarnings("ignore")
OUT = Path("results/q3")
OUT.mkdir(parents=True, exist_ok=True)
FEATURES = ["age", "education-num", "hours-per-week", "capital-gain", "capital-loss",
            "workclass", "marital-status", "occupation", "relationship", "native-country"]
LAMBDAS = [0, 2, 5, 10, 20, 50, 100]
N_BG = 50
AE_LR, EG, NN0, NNF = "AE + LR (before)", "AE + EG (after)", "Network, no fair loss (before)", "Network, fair loss (after)"
COMPARE = [("sex", "Female", "Male", "Women vs men"), ("race", "Black", "White", "Black vs White"),
           ("sex_race", "Black Male", "White Male", "Black men vs White men"),
           ("race", "Asian-Pac-Islander", "White", "Asian-Pac vs White")]


class Scorer:
    """Encoder plus one or more weighted linear heads. A single head is a plain classifier."""

    def __init__(self, enc, W, b, w):
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=DEVICE)
        self.enc, self.W, self.b, self.w = enc.to(DEVICE).eval(), t(W), t(b), t(w)

    def logit(self, x):
        """Log-odds of a high-income prediction. For a mixture, the log-odds of its mean probability."""
        s = self.enc(x) @ self.W.T + self.b
        if s.shape[-1] == 1:
            return s[..., 0]
        p = (torch.sigmoid(s) * self.w).sum(-1).clamp(1e-6, 1 - 1e-6)
        return torch.log(p / (1 - p))

    @torch.no_grad()
    def hard_prob(self, X):
        """Probability of a positive decision: 0 or 1 for one head; the heads' weighted vote for a mixture."""
        out = []
        for c in torch.as_tensor(X, dtype=torch.float32, device=DEVICE).split(8192):
            out.append((((self.enc(c) @ self.W.T + self.b) > 0).float() * self.w).sum(-1))
        return torch.cat(out).cpu().numpy()


def fit_fair_net(ae, head, Xtr, ytr, ftr, Xva, yva, fva, lam, seed, epochs=30, batch=1024, patience=4):
    """Fine-tune encoder and head on BCE + lam * (soft TPR women - soft TPR men)^2."""
    torch.manual_seed(seed)
    enc, lin = copy.deepcopy(ae.enc).to(DEVICE), nn.Linear(LATENT, 1).to(DEVICE)
    with torch.no_grad():
        lin.weight.copy_(torch.as_tensor(head.coef_, dtype=torch.float32, device=DEVICE))
        lin.bias.copy_(torch.as_tensor(head.intercept_, dtype=torch.float32, device=DEVICE))
    t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=DEVICE)
    Xt, yt, ft, Xv, yv, fv = t(Xtr), t(ytr), torch.as_tensor(ftr, device=DEVICE), t(Xva), t(yva), torch.as_tensor(fva, device=DEVICE)
    opt = torch.optim.Adam([*enc.parameters(), *lin.parameters()], lr=5e-4)
    bce = nn.BCEWithLogitsLoss()

    def objective(x, y, f):
        s = lin(enc(x)).squeeze(-1)
        p, pos = torch.sigmoid(s), y == 1
        a, b = pos & f, pos & ~f
        pen = (p[a].mean() - p[b].mean()) ** 2 if a.any() and b.any() else torch.tensor(0.0, device=DEVICE)
        return bce(s, y) + lam * pen

    gen = torch.Generator().manual_seed(seed)
    best, state, wait = np.inf, None, 0
    for _ in range(epochs):
        enc.train(); lin.train()
        for b in torch.randperm(len(Xt), generator=gen).split(batch):
            b = b.to(DEVICE)
            loss = objective(Xt[b], yt[b], ft[b])
            opt.zero_grad(); loss.backward(); opt.step()
        enc.eval(); lin.eval()
        with torch.no_grad():
            v = objective(Xv, yv, fv).item()
        if v < best - 1e-5:
            best, wait = v, 0
            state = (copy.deepcopy(enc.state_dict()), copy.deepcopy(lin.state_dict()))
        else:
            wait += 1
            if wait >= patience:
                break
    enc.load_state_dict(state[0]); lin.load_state_dict(state[1])
    return Scorer(enc, lin.weight.detach().cpu().numpy(), lin.bias.detach().cpu().numpy(), [1.0])


def exact_shap(scorer, X, bg, blocks, chunk=32):
    """Exact interventional Shapley values on log-odds, for all 2^M coalitions of the original features."""
    M = len(blocks)
    masks = torch.zeros((2 ** M, X.shape[1]), dtype=torch.bool, device=DEVICE)
    for s in range(2 ** M):
        for j in range(M):
            if s >> j & 1:
                masks[s, blocks[j]] = True
    Xt, B = torch.as_tensor(X, dtype=torch.float32, device=DEVICE), torch.as_tensor(bg, dtype=torch.float32, device=DEVICE)
    vals = []
    with torch.no_grad():
        for c in Xt.split(chunk):
            mix = torch.where(masks[None, :, None, :], c[:, None, None, :], B[None, None, :, :])
            vals.append(scorer.logit(mix).mean(2))
    v = torch.cat(vals)
    size = torch.tensor([bin(s).count("1") for s in range(2 ** M)], device=DEVICE)
    fact = [math.factorial(i) for i in range(M + 1)]
    weight = torch.tensor([fact[int(k)] * fact[M - int(k) - 1] / fact[M] if k < M else 0.0 for k in size], dtype=torch.float32, device=DEVICE)
    phi = torch.zeros((len(Xt), M), device=DEVICE)
    idx = torch.arange(2 ** M, device=DEVICE)
    for j in range(M):
        without = idx[(idx >> j) & 1 == 0]
        phi[:, j] = ((v[:, without | (1 << j)] - v[:, without]) * weight[without]).sum(1)
    return phi.cpu().numpy(), v[:, -1].cpu().numpy(), v[:, 0].cpu().numpy()


def library_check(scorer, X, bg, blocks, phi):
    """Compare with shap.KernelExplainer on a few rows. Each input column holds a row id into a pool."""
    rows = X[:5]
    pool = np.vstack([rows, bg])

    def f(ids):
        ids = np.asarray(ids).astype(int)
        out = np.zeros((len(ids), X.shape[1]), dtype=np.float32)
        for j, cols in enumerate(blocks):
            out[:, cols] = pool[ids[:, j]][:, cols]
        with torch.no_grad():
            return scorer.logit(torch.as_tensor(out, device=DEVICE)).cpu().numpy()

    r_ids = np.repeat(np.arange(5)[:, None], len(blocks), axis=1)
    b_ids = np.repeat(np.arange(5, 5 + len(bg))[:, None], len(blocks), axis=1)
    lib = shap.KernelExplainer(f, b_ids, link="identity", silent=True).shap_values(r_ids, nsamples=2 ** len(blocks), l1_reg=False, silent=True)
    return float(np.abs(np.asarray(lib) - phi[:5]).max())


def gradients(scorer, X, blocks, center):
    """Vanilla gradient of the log-odds per input. Returns per-feature magnitude and a signed first-order share."""
    grads = []
    for c in torch.as_tensor(X, dtype=torch.float32, device=DEVICE).split(4096):
        c = c.clone().requires_grad_(True)
        scorer.logit(c).sum().backward()
        grads.append(c.grad.detach().cpu().numpy())
    g = np.concatenate(grads)
    mag = np.stack([np.linalg.norm(g[:, cols], axis=1) for cols in blocks], 1)
    signed = np.stack([(g[:, cols] * (X[:, cols] - center[cols])).sum(1) for cols in blocks], 1)
    return mag, signed


def main():
    t0 = time.time()
    train, val, test = [add_groups(d) for d in load_splits()]
    rng = np.random.default_rng(42)          # the same validation halves as Question 2
    order = rng.permutation(len(val))
    va_a = val.iloc[order[: len(val) // 2]].reset_index(drop=True)
    va_b = val.iloc[order[len(val) // 2:]].reset_index(drop=True)
    prep = make_preprocessor(False).fit(train)
    Xtr, Xval, Xa, Xb, Xte = [prep.transform(d).astype(np.float32) for d in (train, val, va_a, va_b, test)]
    names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
    blocks = [[i for i, n in enumerate(names) if n == f or (f in FEATURES[5:] and n.startswith(f + "_"))] for f in FEATURES]
    assert sorted(i for b in blocks for i in b) == list(range(Xtr.shape[1]))
    ytr, yb, yte = train["y"].to_numpy(), va_b["y"].to_numpy(), test["y"].to_numpy()
    ftr, fa = (train["sex"] == "Female").to_numpy(), (va_a["sex"] == "Female").to_numpy()
    bg = Xtr[np.random.default_rng(123).choice(len(Xtr), N_BG, replace=False)]
    center = bg.mean(0)
    saved = np.load("results/q2/test_preds.npz")

    # Build the models. Lambda is chosen once for all seeds, on validation only, with Question 2's rule.
    models, candidates = {AE_LR: [], EG: []}, {l: [] for l in LAMBDAS}
    for s in SEEDS:
        ae = fit_ae(Xtr, Xval, LATENT, s)
        Ztr, Zte = encode(ae, Xtr), encode(ae, Xte)
        head = LogisticRegression(max_iter=2000).fit(Ztr, ytr)
        models[AE_LR].append(Scorer(copy.deepcopy(ae.enc), head.coef_, head.intercept_, [1.0]))
        eg = ExponentiatedGradient(LogisticRegression(max_iter=2000), constraints=TruePositiveRateParity(difference_bound=0.01), eps=0.01, max_iter=30)
        eg.fit(Ztr, ytr, sensitive_features=train["sex"].to_numpy())
        keep = [i for i in range(len(eg.weights_)) if eg.weights_.iloc[i] > 0]
        models[EG].append(Scorer(copy.deepcopy(ae.enc), np.vstack([eg.predictors_.iloc[i].coef_ for i in keep]),
                                 np.array([eg.predictors_.iloc[i].intercept_[0] for i in keep]), eg.weights_.iloc[keep].to_numpy()))
        same = (eg.predict(Zte, random_state=s) == saved["EG_equal_opportunity_(sex)"][SEEDS.index(s)]).mean()
        print(f"seed {s}: EG agrees with Question 2 on {100 * same:.2f}% of rows", flush=True)
        for l in LAMBDAS:
            candidates[l].append(fit_fair_net(ae, head, Xtr, ytr.astype(np.float32), ftr, Xa, va_a["y"].to_numpy().astype(np.float32), fa, l, s))
    print(f"models trained {time.time() - t0:.0f}s", flush=True)

    sel = []
    for l, ms in candidates.items():
        p = np.stack([(m.hard_prob(Xb) > 0.5).astype(int) for m in ms])
        te = np.stack([(m.hard_prob(Xte) > 0.5).astype(int) for m in ms])
        sel.append({"lambda": l, **val_scores(p, yb, va_b), "test_accuracy": float((te == yte).mean())})
    sel = pd.DataFrame(sel)
    pool = sel[sel["lambda"] > 0]
    ok = pool[pool["val_worst_eo"] <= 0.10]
    lam = int((ok.sort_values("val_accuracy", ascending=False) if len(ok) else pool.sort_values("val_worst_eo")).iloc[0]["lambda"])
    sel["chosen"] = sel["lambda"] == lam
    sel.to_csv(OUT / "lambda_selection.csv", index=False)
    print(sel.round(3).to_string(), "\nchosen lambda", lam, flush=True)
    models[NN0], models[NNF] = candidates[0], candidates[lam]
    MODELS = [AE_LR, EG, NN0, NNF]

    # Fairness and task metrics. EG uses the expected value of its random decision.
    hard = {m: np.stack([s.hard_prob(Xte) for s in models[m]]) for m in MODELS}
    boot = Boot(len(test))
    B = "Before (AE + LR)"
    fair = []
    for before, after in ((AE_LR, EG), (NN0, NNF)):
        for r in audit(boot, yte, {B: hard[before], after: hard[after]}, test):
            r["method"] = before if r["method"] == B else after
            fair.append(r)
    fair = pd.DataFrame(fair)
    fair.to_csv(OUT / "fairness.csv", index=False)
    overall = pd.DataFrame([{"model": m, **pd.DataFrame([classification_row(yte, (p > 0.5).astype(int)) for p in hard[m]]).mean().to_dict()} for m in MODELS])
    # EG is compared through its saved random predictions, which is what Question 2 reported.
    overall.loc[overall.model == EG, ["accuracy", "precision", "recall", "f1"]] = pd.DataFrame([classification_row(yte, p) for p in saved["EG_equal_opportunity_(sex)"]]).mean()[["accuracy", "precision", "recall", "f1"]].to_numpy()
    overall.to_csv(OUT / "overall.csv", index=False)
    print(overall.round(3).to_string(), flush=True)

    # SHAP and gradients for every test row, averaged over seeds.
    phi, mag, sgn, check = {}, {}, {}, {}
    for m in MODELS:
        ph, ma, si = [], [], []
        for s in models[m]:
            p, fx, base = exact_shap(s, Xte, bg, blocks)
            assert np.abs(p.sum(1) - (fx - base)).max() < 2e-2, "SHAP values must sum to f(x) - baseline"
            a, b = gradients(s, Xte, blocks, center)
            ph.append(p); ma.append(a); si.append(b)
        check[m] = library_check(models[m][0], Xte, bg, blocks, ph[0])
        phi[m], mag[m], sgn[m] = np.stack(ph), np.stack(ma), np.stack(si)
        print(f"{m}: attributions done {time.time() - t0:.0f}s, max difference from shap library {check[m]:.5f}", flush=True)
    np.savez_compressed(OUT / "attributions.npz", **{f"phi_{i}": phi[m].mean(0) for i, m in enumerate(MODELS)},
                        **{f"grad_{i}": mag[m].mean(0) for i, m in enumerate(MODELS)})

    # Stability of SHAP to a different background sample.
    bg2 = Xtr[np.random.default_rng(456).choice(len(Xtr), N_BG, replace=False)]
    idx = np.random.default_rng(7).choice(len(Xte), 1000, replace=False)
    stab = {m: float(np.corrcoef(phi[m][0][idx].ravel(), exact_shap(models[m][0], Xte[idx], bg2, blocks)[0].ravel())[0, 1]) for m in MODELS}

    group_rows, gap_rows, agree_rows = [], [], []
    pos = yte == 1
    groups = [("all", "Everyone", np.ones(len(test), bool))] + [(a, g, (test[a] == g).to_numpy()) for a in ("sex", "race") for g in sorted(test[a].unique())] + \
             [("sex_race", g, (test["sex_race"] == g).to_numpy()) for g in ("Black Male", "White Male", "Black Female", "White Female")]
    for m in MODELS:
        P, G, S = phi[m].mean(0), mag[m].mean(0), sgn[m].mean(0)
        for a, g, mask in groups:
            for scope, sc in (("all", mask), ("high earners", mask & pos)):
                for j, f in enumerate(FEATURES):
                    group_rows.append({"model": m, "attribute": a, "group": g, "scope": scope, "n": int(sc.sum()), "feature": f,
                                       "shap_mean": float(P[sc, j].mean()), "shap_abs": float(np.abs(P[sc, j]).mean()), "gradient": float(G[sc, j].mean())})
                if sc.sum() >= 30:
                    agree_rows.append({"model": m, "group": g, "scope": scope, "spearman_abs_shap_vs_gradient": float(spearmanr(np.abs(P[sc]).mean(0), G[sc].mean(0))[0])})
        for a, g, r, label in COMPARE:
            gm, rm = (test[a] == g).to_numpy(), (test[a] == r).to_numpy()
            for scope, flt in (("high earners", pos), ("all", np.ones(len(test), bool))):
                for j, f in enumerate(FEATURES):
                    b = boot.rate(P[:, j], gm & flt) - boot.rate(P[:, j], rm & flt)
                    per_seed = [float(phi[m][k][gm & flt, j].mean() - phi[m][k][rm & flt, j].mean()) for k in range(len(SEEDS))]
                    gap_rows.append({"model": m, "comparison": label, "scope": scope, "feature": f,
                                     "shap_gap": float(P[gm & flt, j].mean() - P[rm & flt, j].mean()), "lo": float(np.quantile(b, 0.025)), "hi": float(np.quantile(b, 0.975)),
                                     "seed_sd": float(np.std(per_seed)), "first_order_gradient_gap": float(S[gm & flt, j].mean() - S[rm & flt, j].mean()),
                                     "saliency_gap": float(G[gm & flt, j].mean() - G[rm & flt, j].mean()), "saliency_reference": float(G[rm & flt, j].mean())})
    groups_df, gaps = pd.DataFrame(group_rows), pd.DataFrame(gap_rows)
    groups_df.to_csv(OUT / "attribution_by_group.csv", index=False)
    gaps.to_csv(OUT / "attribution_gaps.csv", index=False)
    pd.DataFrame(agree_rows).to_csv(OUT / "technique_agreement.csv", index=False)

    # Faithfulness: scramble a feature for everyone and re-measure the equal opportunity gap, without retraining.
    tests = {f: blocks[j] for j, f in enumerate(FEATURES)}
    tests["relationship + marital-status"] = blocks[FEATURES.index("relationship")] + blocks[FEATURES.index("marital-status")]
    scr = []
    for m in MODELS:
        for a, g, r, label in COMPARE[:3]:
            gm, rm = (test[a] == g).to_numpy() & pos, (test[a] == r).to_numpy() & pos
            base = float(hard[m][:, gm].mean() - hard[m][:, rm].mean())
            scr.append({"model": m, "comparison": label, "feature": "(nothing scrambled)", "eo_gap": base, "change_in_abs_gap": 0.0})
            for f, cols in tests.items():
                vals = []
                for rep in range(5):
                    perm = np.random.default_rng(100 + rep).permutation(len(Xte))
                    Xs = Xte.copy()
                    Xs[:, cols] = Xte[perm][:, cols]
                    p = np.stack([s.hard_prob(Xs) for s in models[m]])
                    vals.append(p[:, gm].mean() - p[:, rm].mean())
                scr.append({"model": m, "comparison": label, "feature": f, "eo_gap": float(np.mean(vals)), "change_in_abs_gap": float(abs(np.mean(vals)) - abs(base))})
    scr = pd.DataFrame(scr)
    scr.to_csv(OUT / "scramble_test.csv", index=False)
    (OUT / "meta.json").write_text(json.dumps({"lambda": lam, "shap_library_max_difference": check, "background_stability_corr": stab, "n_background": N_BG}, indent=2))

    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 400)
    print("library check", check, "\nbackground stability", stab)
    print(fair[fair.attribute != "sex_race"][["method", "group", "n_pos", "tpr_group", "tpr_reference", "eo_gap", "eo_lo", "eo_hi", "dp_gap", "dp_reduction", "eo_reduction", "eo_red_lo", "eo_red_hi", "eo_unfair"]].round(3).to_string())
    print(fair[(fair.attribute == "sex_race") & (fair.group == "Black Male")][["method", "group", "n_pos", "tpr_group", "eo_gap", "eo_lo", "eo_hi", "eo_unfair"]].round(3).to_string())
    x = gaps[(gaps.scope == "high earners") & (gaps.comparison == "Women vs men")]
    print(x.pivot_table(index="feature", columns="model", values="shap_gap").round(3).loc[FEATURES][[*MODELS]].to_string())
    print(x.pivot_table(index="feature", columns="model", values="first_order_gradient_gap").round(3).loc[FEATURES][[*MODELS]].to_string())
    print(x.pivot_table(index="feature", columns="model", values="saliency_gap").round(3).loc[FEATURES][[*MODELS]].to_string())
    print(pd.DataFrame(agree_rows).query("scope == 'high earners' and group in ['Female','Male','Black','White']").round(2).to_string())
    print(scr[scr.comparison == "Women vs men"].pivot_table(index="feature", columns="model", values="eo_gap").round(3)[[*MODELS]].to_string())
    print(f"finished {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
