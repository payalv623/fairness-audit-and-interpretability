"""Experiment 1: deep autoencoder. Reconstruction error by group, plus DP/EO via its latent code."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from torch import nn

from common import REF_GROUP, SEEDS, SENSITIVE, classification_row, fairness_table, load_splits, make_preprocessor

OUT = Path("results")
OUT.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LATENT = 16       # fixed up front: validation error always favours a bigger code
LATENT_SENS = [8, 32]
MAX_EPOCHS, PATIENCE, BATCH = 200, 10, 256


class AE(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, k))
        self.dec = nn.Sequential(nn.Linear(k, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, d))

    def forward(self, x):
        return self.dec(self.enc(x))


def fit_ae(Xtr, Xva, k, seed):
    """Train with early stopping on validation reconstruction error."""
    torch.manual_seed(seed)
    Xtr, Xva = [torch.tensor(a, dtype=torch.float32, device=DEVICE) for a in (Xtr, Xva)]
    model = AE(Xtr.shape[1], k).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    best, best_state, wait = np.inf, None, 0
    for _ in range(MAX_EPOCHS):
        model.train()
        for b in torch.randperm(len(Xtr), generator=gen).split(BATCH):
            xb = Xtr[b.to(DEVICE)]
            loss = ((model(xb) - xb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = ((model(Xva) - Xva) ** 2).mean().item()
        if v < best - 1e-5:
            best, wait = v, 0
            best_state = {n: p.detach().clone() for n, p in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model.eval()


@torch.no_grad()
def row_errors(model, X):
    x = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    return ((model(x) - x) ** 2).mean(dim=1).cpu().numpy()


@torch.no_grad()
def encode(model, X):
    return model.enc(torch.tensor(X, dtype=torch.float32, device=DEVICE)).cpu().numpy()


def boot_ci(a, b, seed=0, n=2000, alpha=0.05):
    """CI for mean(a) - mean(b), with groups resampled independently."""
    rng = np.random.default_rng(seed)
    d = rng.choice(a, (n, len(a))).mean(1) - rng.choice(b, (n, len(b))).mean(1)
    return tuple(np.quantile(d, [alpha / 2, 1 - alpha / 2]))


def main():
    train, val, test = load_splits()
    prep = make_preprocessor(include_sensitive=False).fit(train)
    Xtr, Xva, Xte = [prep.transform(d) for d in (train, val, test)]
    print(f"rows {len(train)}/{len(val)}/{len(test)} features {Xtr.shape[1]} device {DEVICE}")

    # Shared model, one per seed. Per-row error is averaged over seeds.
    shared = {s: fit_ae(Xtr, Xva, LATENT, s) for s in SEEDS}
    err = np.mean([row_errors(m, Xte) for m in shared.values()], axis=0)
    seed_err = [float(row_errors(m, Xte).mean()) for m in shared.values()]

    # Linear baseline at the same code size.
    pca = PCA(n_components=LATENT, random_state=0).fit(Xtr)
    err_pca = ((pca.inverse_transform(pca.transform(Xte)) - Xte) ** 2).mean(axis=1)

    rows = []
    for attr in SENSITIVE:
        ref = REF_GROUP[attr]
        gv = test[attr].to_numpy()
        n_comp = max(len(np.unique(gv)) - 1, 1)
        excess = {}
        for g in sorted(np.unique(gv)):
            # Reference: an autoencoder that only ever saw this group.
            tr, va = (train[attr] == g).to_numpy(), (val[attr] == g).to_numpy()
            own = np.mean([row_errors(fit_ae(Xtr[tr], Xva[va], LATENT, s), Xte[gv == g]) for s in SEEDS], axis=0)
            excess[g] = err[gv == g] - own
            rows.append({
                "attribute": attr, "group": g, "n_train": int(tr.sum()), "n_test": int((gv == g).sum()),
                "shared_error": float(err[gv == g].mean()), "shared_median": float(np.median(err[gv == g])),
                "own_model_error": float(own.mean()), "excess_error": float(excess[g].mean()),
                "pca_error": float(err_pca[gv == g].mean()),
            })
        for r in rows:
            if r["attribute"] != attr:
                continue
            g = r["group"]
            if g == ref:
                r.update(error_gap=0.0, excess_gap=0.0)
                continue
            a = 0.05 / n_comp
            r["error_gap"] = r["shared_error"] - float(err[gv == ref].mean())
            r["error_gap_lo"], r["error_gap_hi"] = boot_ci(err[gv == g], err[gv == ref], alpha=a)
            r["excess_gap"] = r["excess_error"] - float(excess[ref].mean())
            r["excess_gap_lo"], r["excess_gap_hi"] = boot_ci(excess[g], excess[ref], alpha=a)
    pd.DataFrame(rows).to_csv(OUT / "exp1_reconstruction.csv", index=False)

    # Same gap at other code sizes, to show the finding does not hinge on k=16.
    sens = []
    for k in LATENT_SENS + [LATENT]:
        ms = list(shared.values()) if k == LATENT else [fit_ae(Xtr, Xva, k, s) for s in SEEDS]
        e = np.mean([row_errors(m, Xte) for m in ms], axis=0)
        for attr in SENSITIVE:
            gv = test[attr].to_numpy()
            for g in np.unique(gv):
                sens.append({"latent": k, "attribute": attr, "group": g, "error": float(e[gv == g].mean())})
    pd.DataFrame(sens).to_csv(OUT / "exp1_latent_sensitivity.csv", index=False)

    # Classifier on the frozen latent code, so the AE gets comparable DP/EO numbers.
    y_tr, y_te = train["y"].to_numpy(), test["y"].to_numpy()
    preds = np.stack([
        LogisticRegression(max_iter=2000).fit(encode(m, Xtr), y_tr).predict(encode(m, Xte)) for m in shared.values()
    ])
    np.save(OUT / "exp1_preds.npy", preds)
    overall = [classification_row(y_te, p) for p in preds]
    fair = []
    for attr in SENSITIVE:
        fair += fairness_table(y_te, preds, test[attr].to_numpy(), attr)
    pd.DataFrame(fair).assign(model="AE latent + LR").to_csv(OUT / "exp1_fairness.csv", index=False)
    summary = {
        "seed_mean_error": seed_err, "pca_mean_error": float(err_pca.mean()),
        "overall_by_seed": overall, "n_features": int(Xtr.shape[1]),
    }
    (OUT / "exp1_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))
    print(pd.DataFrame(rows).round(4).to_string())
    print(pd.DataFrame(fair).round(3).to_string())


if __name__ == "__main__":
    main()
