"""Why do race groups differ in reconstruction error? Feature breakdown and a size-matched reference."""
import numpy as np
import pandas as pd

from common import SEEDS, load_splits, make_preprocessor
from exp1_autoencoder import LATENT, OUT, fit_ae, row_errors, DEVICE
import torch


def main():
    train, val, test = load_splits()
    prep = make_preprocessor(False).fit(train)
    Xtr, Xva, Xte = [prep.transform(d) for d in (train, val, test)]
    names = prep.get_feature_names_out()
    def block(n):
        n = n.split("__", 1)[1]
        for c in ["workclass", "marital-status", "occupation", "relationship", "native-country"]:
            if n.startswith(c):
                return c
        return "numeric"
    blocks = np.array([block(n) for n in names])

    # 1. Share of each group's squared error coming from each feature block.
    sq = 0
    for s in SEEDS:
        m = fit_ae(Xtr, Xva, LATENT, s)
        with torch.no_grad():
            x = torch.tensor(Xte, dtype=torch.float32, device=DEVICE)
            sq = sq + ((m(x) - x) ** 2).cpu().numpy() / len(SEEDS)
    rows = []
    for attr in ["sex", "race"]:
        for g in sorted(test[attr].unique()):
            mask = (test[attr] == g).to_numpy()
            r = {"attribute": attr, "group": g, "total": sq[mask].mean(1).mean()}
            for b in np.unique(blocks):
                r[b] = sq[mask][:, blocks == b].sum(1).mean() / sq.shape[1]
            rows.append(r)
    pd.DataFrame(rows).to_csv(OUT / "exp1_error_by_feature.csv", index=False)
    print(pd.DataFrame(rows).round(5).to_string())

    # 2. Own-group models at EQUAL training size, so sample size cannot explain the difference.
    out = []
    for n in (244, 911, 2811):
        for g in ["White", "Black", "Asian-Pac-Islander", "Other"]:
            tr_idx = np.where(train["race"] == g)[0]
            if len(tr_idx) < n:
                continue
            va, te = (val["race"] == g).to_numpy(), (test["race"] == g).to_numpy()
            errs = []
            for s in SEEDS:
                idx = np.random.default_rng(s).choice(tr_idx, n, replace=False)
                errs.append(row_errors(fit_ae(Xtr[idx], Xva[va], LATENT, s), Xte[te]).mean())
            out.append({"n_train": n, "group": g, "own_error": float(np.mean(errs)), "seed_sd": float(np.std(errs))})
    pd.DataFrame(out).to_csv(OUT / "exp1_size_matched.csv", index=False)
    print(pd.DataFrame(out).round(5).to_string())


if __name__ == "__main__":
    main()
