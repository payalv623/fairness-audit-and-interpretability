"""Is the reconstruction improvement from balanced training larger than seed-to-seed variation?"""
import numpy as np
import pandas as pd

from common import SEEDS, load_splits, make_preprocessor
from exp1_autoencoder import LATENT, fit_ae, row_errors
from q2_mitigate import OUT, add_groups, fit_ae_balanced


def main():
    train, val, test = [add_groups(d) for d in load_splits()]
    prep = make_preprocessor(False).fit(train)
    Xtr, Xva, Xte = [prep.transform(d) for d in (train, val, test)]
    race, rva, rte = train["race"].to_numpy(), val["race"].to_numpy(), test["race"].to_numpy()
    rows = []
    for s in SEEDS:
        models = {"Before (AE)": fit_ae(Xtr, Xva, LATENT, s),
                  "Balanced AE (sqrt weights)": fit_ae_balanced(Xtr, Xva, race, rva, LATENT, s, 0.5),
                  "Balanced AE (full weights)": fit_ae_balanced(Xtr, Xva, race, rva, LATENT, s, 1.0)}
        for name, m in models.items():
            e = row_errors(m, Xte)
            w = e[rte == "White"].mean()
            rows.append({"method": name, "seed": s, "overall": e.mean(), "white": w,
                         "asian_pac": e[rte == "Asian-Pac-Islander"].mean(), "other": e[rte == "Other"].mean(),
                         "asian_ratio": e[rte == "Asian-Pac-Islander"].mean() / w, "other_ratio": e[rte == "Other"].mean() / w})
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "reconstruction_by_seed.csv", index=False)
    pd.set_option("display.width", 200)
    print(d.round(4).to_string())
    print(d.groupby("method").agg(["mean", "std"]).drop(columns="seed").round(4).T.to_string())


if __name__ == "__main__":
    main()
