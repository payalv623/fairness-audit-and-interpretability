"""Build Fairness_Audit_Report.pdf from the saved results."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Table, TableStyle

from common import classification_row, group_rates, load_splits, make_preprocessor

FEATURES = ["age", "education-num", "hours-per-week", "capital-gain", "capital-loss",
            "workclass", "marital-status", "occupation", "relationship", "native-country"]

R = Path("results")
INK, MUTED, RULE, HEAD = colors.HexColor("#0b0b0b"), colors.HexColor("#52514e"), colors.HexColor("#c9c8c3"), colors.HexColor("#eef2f7")
ACCENT = colors.HexColor("#1f5fae")
base = getSampleStyleSheet()["Normal"]
S = {
    "title": ParagraphStyle("title", parent=base, fontName="Helvetica-Bold", fontSize=18, leading=22, textColor=INK, spaceAfter=4),
    "sub": ParagraphStyle("sub", parent=base, fontSize=10, leading=14, textColor=MUTED, spaceAfter=10),
    "part": ParagraphStyle("part", parent=base, fontName="Helvetica-Bold", fontSize=13.5, leading=17, textColor=INK, backColor=HEAD, borderPadding=(5, 6, 5, 6), spaceBefore=16, spaceAfter=10, keepWithNext=1),
    "h1": ParagraphStyle("h1", parent=base, fontName="Helvetica-Bold", fontSize=12.5, leading=16, textColor=ACCENT, spaceBefore=12, spaceAfter=4, keepWithNext=1),
    "h2": ParagraphStyle("h2", parent=base, fontName="Helvetica-Bold", fontSize=10.5, leading=14, textColor=INK, spaceBefore=8, spaceAfter=3, keepWithNext=1),
    "p": ParagraphStyle("p", parent=base, fontSize=10, leading=14, textColor=INK, spaceAfter=6),
    "b": ParagraphStyle("b", parent=base, fontSize=10, leading=14, textColor=INK, leftIndent=12, bulletIndent=2, spaceAfter=3),
    "cap": ParagraphStyle("cap", parent=base, fontSize=8.5, leading=11, textColor=MUTED, spaceBefore=3, spaceAfter=8),
    "cell": ParagraphStyle("cell", parent=base, fontSize=8.2, leading=10.5, textColor=INK),
    "cellb": ParagraphStyle("cellb", parent=base, fontName="Helvetica-Bold", fontSize=8.2, leading=10.5, textColor=INK),
    "ref": ParagraphStyle("ref", parent=base, fontSize=8.8, leading=12, textColor=INK, leftIndent=12, firstLineIndent=-12, spaceAfter=3),
}
PAGE_W = A4[0] - 40 * mm


def num(x, d=3):
    return f"{x:.{d}f}".replace("-", "–")


def gap_ci(r, key, d=3):
    return f"{num(r[f'{key}_gap'], d)} ({num(r[f'{key}_lo'], d)} to {num(r[f'{key}_hi'], d)})"


def P(t, s="p"):
    return Paragraph(t, S[s])


def bullets(items):
    return [Paragraph(t, S["b"], bulletText="•") for t in items]


def table(rows, widths, bold=()):
    """rows[0] is the header. bold holds (row, col) cells to emphasise."""
    data = []
    for i, row in enumerate(rows):
        data.append([Paragraph(str(c), S["cellb"] if i == 0 or (i, j) in bold else S["cell"]) for j, c in enumerate(row)])
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEAD),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, MUTED),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def figure(path, caption, width=PAGE_W):
    from PIL import Image as PI
    w, h = PI.open(path).size
    return KeepTogether([Image(str(path), width=width, height=width * h / w), P(caption, "cap")])


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(20 * mm, 11 * mm, "Fairness Audit, Mitigation and Interpretability on Adult Income")
    canvas.drawRightString(A4[0] - 20 * mm, 11 * mm, f"Page {doc.page}")
    canvas.restoreState()


def main():
    train, val, test = load_splits()
    full = pd.concat([train, val, test])
    y = test["y"].to_numpy()
    preds = {"SVM (no sex/race)": np.load(R / "exp2_preds_unaware.npy"),
             "SVM (with sex/race)": np.load(R / "exp2_preds_aware.npy"),
             "AE latent + LR": np.load(R / "exp1_preds.npy")}
    MODELS = list(preds)
    fair = pd.concat([pd.read_csv(R / "exp2_fairness.csv"), pd.read_csv(R / "exp1_fairness.csv")])
    f = lambda m, g: fair[(fair["model"] == m) & (fair["group"] == g)].iloc[0]
    rec = pd.read_csv(R / "exp1_reconstruction.csv").set_index("group")
    feat = pd.read_csv(R / "exp1_error_by_feature.csv").set_index("group")
    size = pd.read_csv(R / "exp1_size_matched.csv")
    sens = pd.read_csv(R / "exp1_latent_sensitivity.csv")
    s1, s2 = json.loads((R / "exp1_summary.json").read_text()), json.loads((R / "exp2_summary.json").read_text())
    majority = s2["majority_baseline_accuracy"]
    overall = {m: np.mean([list(classification_row(y, p).values()) for p in preds[m]], axis=0) for m in MODELS}
    fem, male = (test["sex"] == "Female").to_numpy(), (test["sex"] == "Male").to_numpy()
    rates = {m: (group_rates(y, preds[m], fem), group_rates(y, preds[m], male)) for m in MODELS}
    AE, SV0, SV1 = "AE latent + LR", "SVM (no sex/race)", "SVM (with sex/race)"
    ratio = lambda g: rec.loc[g, "shared_error"] / rec.loc["White", "shared_error"]
    k = lambda latent, g: sens[(sens["latent"] == latent) & (sens["group"] == g)]["error"].iloc[0]
    sm = lambda n, g: size[(size["n_train"] == n) & (size["group"] == g)]["own_error"].iloc[0]
    share = lambda g: 100 * feat.loc[g, "native-country"] / feat.loc[g, "total"]

    # Question 2 results
    q2f, q2o = pd.read_csv(R / "q2/fairness.csv"), pd.read_csv(R / "q2/overall.csv").set_index("method")
    q2r = pd.read_csv(R / "q2/reconstruction_by_seed.csv").groupby("method")
    q2p = np.load(R / "q2/test_preds.npz")
    BF, EGM = "Before (AE + LR)", "EG equal opportunity (sex)"
    g2 = lambda m, g: q2f[(q2f["method"] == m) & (q2f["group"] == g)].iloc[0]
    assert q2o.loc[EGM, "chosen"]
    pos = y == 1
    tpr_rel = test[pos].assign(b=q2p["Before_(AE_+_LR)"].mean(0)[pos], a=q2p["EG_equal_opportunity_(sex)"].mean(0)[pos]).groupby(["sex", "relationship"]).agg(n=("b", "size"), b=("b", "mean"), a=("a", "mean"))
    # Question 3 results
    M0, M1, M2, M3 = "AE + LR (before)", "AE + EG (after)", "Network, no fair loss (before)", "Network, fair loss (after)"
    Q3M = [M0, M1, M2, M3]
    q3f, q3o = pd.read_csv(R / "q3/fairness.csv"), pd.read_csv(R / "q3/overall.csv").set_index("model")
    gaps, scr = pd.read_csv(R / "q3/attribution_gaps.csv"), pd.read_csv(R / "q3/scramble_test.csv")
    by, agree = pd.read_csv(R / "q3/attribution_by_group.csv"), pd.read_csv(R / "q3/technique_agreement.csv")
    lam = pd.read_csv(R / "q3/lambda_selection.csv")
    q3m = json.loads((R / "q3/meta.json").read_text())
    phi = np.load(R / "q3/attributions.npz")
    g3 = lambda m, g: q3f[(q3f["method"] == m) & (q3f["group"] == g)].iloc[0]
    sg = lambda m, c, ft: gaps[(gaps["model"] == m) & (gaps["comparison"] == c) & (gaps["feature"] == ft) & (gaps["scope"] == "high earners")].iloc[0]
    sc = lambda m, c, ft: scr[(scr["model"] == m) & (scr["comparison"] == c) & (scr["feature"] == ft)]["eo_gap"].iloc[0]
    WM, BW, BM = "Women vs men", "Black vs White", "Black men vs White men"
    grad = lambda m, g, ft: by[(by["model"] == m) & (by["group"] == g) & (by["scope"] == "high earners") & (by["feature"] == ft)]["gradient"].iloc[0]
    # SHAP of relationship per category, and how much of the gap is explained by who is in each category.
    jr = FEATURES.index("relationship")
    he = test[pos].copy()
    for i, m in enumerate(Q3M):
        he[m] = phi[f"phi_{i}"][pos, jr]
    rel_mean = he.groupby("relationship")[Q3M].mean()
    rel_share = pd.crosstab(he["relationship"], he["sex"], normalize="columns")
    rel_comp = {m: float(((rel_share["Female"] - rel_share["Male"]) * rel_mean[m]).sum()) for m in Q3M}
    for m in Q3M:   # the composition term must account for essentially all of the gap
        assert abs(rel_comp[m] - sg(m, WM, "relationship")["shap_gap"]) < 0.01
    wm = gaps[(gaps["comparison"] == WM) & (gaps["scope"] == "high earners")]
    r_first = [wm[wm["model"] == m][["shap_gap", "first_order_gradient_gap"]].corr().iloc[0, 1] for m in Q3M]
    r_sal = [wm[wm["model"] == m][["shap_gap", "saliency_gap"]].corr().iloc[0, 1] for m in Q3M]
    ag = agree[(agree["scope"] == "high earners") & (agree["group"].isin(["Female", "Male", "Black", "White"]))]["spearman_abs_shap_vs_gradient"]
    assert wm[wm["feature"].isin(["capital-loss", "workclass", "native-country"])]["shap_gap"].abs().max() < 0.03
    mar_share = pd.crosstab(he["marital-status"], he["sex"], normalize="columns").loc["Married-civ-spouse"]
    nw = tpr_rel.loc["Female"].drop("Wife")
    nw_b, nw_a = (nw["n"] * nw["b"]).sum() / nw["n"].sum(), (nw["n"] * nw["a"]).sum() / nw["n"].sum()
    col_names = make_preprocessor(False).fit(train).get_feature_names_out()
    col_count = {ft: sum(c.split("__", 1)[1].startswith(ft + "_") for c in col_names) for ft in ("native-country", "occupation", "workclass", "relationship")}

    st = []
    st += [P("Fairness Audit, Mitigation and Interpretability on Adult Income", "title"),
           P("Questions 1, 2 and 3<br/>Name: ____________________ &nbsp;&nbsp; Roll No.: ______________", "sub")]

    q2b, q2a, n0, n1 = g2(BF, "Female"), g2(EGM, "Female"), g3(M2, "Female"), g3(M3, "Female")
    st += [P("Summary", "h1"),
           P(f"This report audits machine learning models on the Adult Income dataset for unfairness, makes one of them fairer, and uses interpretability methods to explain the result. All results are from {len(test):,} held-out test records.")]
    st += bullets([
        f"<b>Question 1, is it unfair? Yes, for women.</b> A classifier on a deep autoencoder's compressed features correctly identifies far fewer high-earning women than men (equal opportunity gap {num(f(AE,'Female')['eo_gap'])}). An SVM's gap is smaller and borderline ({num(f(SV0,'Female')['eo_gap'])} to {num(f(SV1,'Female')['eo_gap'])}). Reconstruction error shows a different pattern: little gap by sex, but Asian-Pacific Islander error is {ratio('Asian-Pac-Islander'):.1f} times White error.",
        f"<b>Question 2, can it be fixed? On equal opportunity for women, yes.</b> Exponentiated Gradient moved the women's gap from {num(q2b['eo_gap'])} to {num(q2a['eo_gap'])}, at a cost of {num(q2o.loc[BF,'accuracy']-q2o.loc[EGM,'accuracy'])} accuracy. Women, and white women in particular, are the groups whose disparity was largest and is now reduced. The demographic parity gap is still above 0.10 ({num(q2a['dp_gap'])}), and Black men did not benefit.",
        f"<b>Question 3, why? <i>Relationship</i> and <i>marital-status</i> drive the gap, and <i>relationship</i> works as a proxy for sex.</b> SHAP and a scramble test both point to them. The <i>relationship</i> gap comes entirely from Husband and Wife being sex-specific categories. The fair models do not stop using this proxy: they score Wife above Husband. The fair optimization loss moved the network's gap from {num(n0['eo_gap'])} to {num(n1['eo_gap'])}, but the network was already under the 0.10 cutoff for women overall, so this is an improvement rather than a clear change from unfair to fair.",
    ])

    st += [P("1. Aim and scope", "h1"),
           P("The assignment has three parts: audit at least two algorithms for unfairness (Question 1), make one fair and re-evaluate it (Question 2), and use at least two interpretability techniques to investigate which features drive the unfairness and whether that changes after fair optimization (Question 3). "
             "The models are a neural network (a deep autoencoder, list item 3) and an SVM (list item 2). Sections 1 and 2 apply to all three questions.")]

    n = len(full)
    grp = lambda a: full.groupby(a)["y"].agg(["size", "mean"])
    sx, rc = grp("sex"), grp("race")
    rows = [["Attribute", "Group", "Records", "Share of data", "Earning &gt;$50K"]]
    for a, d in (("Sex", sx.sort_values("size", ascending=False)), ("Race", rc.sort_values("size", ascending=False))):
        for g, r in d.iterrows():
            rows.append([a, g, f"{int(r['size']):,}", f"{100*r['size']/n:.1f}%", f"{100*r['mean']:.1f}%"])
    st += [P("2. Dataset", "h1"),
           P(f"Adult Income (UCI, via OpenML) has {n:,} records from the 1994 US census. The task is to predict whether a person earns more than $50K ({100*full['y'].mean():.1f}% do). "
             "Sex and race are the groups audited, with Male and White as reference groups because they are the largest."),
           KeepTogether([table(rows, [24, 52, 28, 32, 34]), P("Table 1. Group sizes and income rates in the full dataset.", "cap")]),
           P(f"The data was split 60/20/20 into {len(train):,} training, {len(val):,} validation and {len(test):,} test records, stratified on label, sex and race. "
             "The same split was used throughout. <i>fnlwgt</i> (a census sampling weight) and <i>education</i> (a duplicate of <i>education-num</i>) were removed. Missing categories were kept as their own \"Missing\" value. "
             f"Numeric columns were standardised, capital gain and loss were log-transformed first, and categorical columns were one-hot encoded, giving {s1['n_features']} input features without sex and race.")]

    st += [P("Question 1: Fairness audit", "part"), P("3. Method", "h1"), P("3.1 Experiment 1: Deep autoencoder", "h2")]
    st += bullets([
        "<b>Architecture:</b> fully connected, input-64-32-16-32-64-output, with ReLU activations and a 16-dimensional code. The code size was fixed in advance, because validation error always favours a larger code; sizes 8 and 32 were checked afterwards.",
        "<b>Training:</b> Adam (learning rate 0.001), batch size 256, mean squared error, and early stopping on validation error, repeated over 3 seeds. Sex and race were excluded so the error would not simply reflect rebuilding those columns.",
        "<b>Baseline:</b> PCA with the same 16 components.",
        "<b>Classification:</b> an autoencoder makes no predictions, so a logistic regression was trained on its frozen codes to allow demographic parity and equal opportunity to be measured (\"AE latent + LR\").",
    ])
    st += [P("3.2 Experiment 2: Support vector machine", "h2")]
    st += bullets([
        "<b>Model:</b> RBF kernel, with C in {0.1, 1, 10} and gamma in {scale, 0.01} chosen by validation accuracy. C = 1 with gamma = scale was picked for two seeds and C = 10 with gamma = 0.01 for one.",
        "<b>Training data:</b> a random sample of 15,000 training records per seed (3 seeds), because RBF SVM training scales poorly with data size.",
        "<b>Two versions:</b> one without sex and race as inputs and one with them, to test what removing these columns achieves.",
    ])
    st += [P("3.3 Metrics", "h2")]
    st += [KeepTogether([table([
        ["Metric", "Definition", "Purpose"],
        ["Reconstruction error", "Mean squared difference between input and output, per record", "Task metric for the autoencoder and PCA"],
        ["Accuracy, precision, recall, F1", "Standard definitions; \"&gt;$50K\" is the positive class", "Task metrics for classifiers"],
        ["Demographic parity gap", "P(predict &gt;$50K | group) – P(predict &gt;$50K | reference)", "Compares how often each group gets the favourable prediction"],
        ["Equal opportunity gap", "TPR(group) – TPR(reference), where TPR is P(predict &gt;$50K | truly &gt;$50K)", "Compares how often real high earners are recognised"],
        ["False positive rate gap", "FPR(group) – FPR(reference)", "Supporting check on the other type of error"],
    ], [42, 78, 50]), P("Table 2. Metrics. A negative gap means the group has a lower rate than its reference group.", "cap")])]
    st += [P("<b>Verdict rule.</b> Equal opportunity is the main fairness measure, because it allows for real income differences between groups. A comparison counts as unfair if the gap is larger than 0.10 and its confidence interval excludes 0. The 0.10 cutoff is a common convention rather than a formal standard."),
           P("<b>Uncertainty.</b> Confidence intervals use 2,000 bootstrap resamples of test records, with predictions from all three seeds. They reflect test-set sampling only. The intervals are 95% for sex and Bonferroni-adjusted to 98.75% for race, which has four comparisons. Signed gaps are used because an absolute gap can never have an interval crossing 0.")]

    rows = [["Model", "Accuracy", "Precision", "Recall", "F1", "Accuracy, women", "Accuracy, men", "TPR, women", "TPR, men"]]
    for m in MODELS:
        a, b = rates[m]
        rows.append([m, *[num(v) for v in overall[m]], num(a["accuracy"]), num(b["accuracy"]), num(a["tpr"]), num(b["tpr"])])
    st += [KeepTogether([P("4. Results", "h1"), P("4.1 Classification performance", "h2"), table(rows, [38, 16, 16, 14, 13, 19, 18, 18, 18]),
           P(f"Table 3. Test performance averaged over 3 seeds. Always predicting ≤$50K scores {num(majority)} accuracy.", "cap")]),
           P(f"All three models beat the majority baseline of {num(majority)}. Women have <b>higher accuracy but lower TPR</b> than men in every model. Fewer women earn above $50K, so predicting ≤$50K is right more often for them, and that lifts accuracy while high-earning women are still missed. Accuracy alone would have hidden this problem.")]

    st += [P("4.2 Fairness gaps", "h2")]
    order = ["Female", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]
    label = {"Female": "Female vs Male", "Black": "Black vs White", "Asian-Pac-Islander": "Asian-Pac vs White", "Amer-Indian-Eskimo": "Amer-Indian vs White", "Other": "Other vs White"}
    rows, bold = [["Comparison", "High earners in group", *MODELS]], set()
    for i, g in enumerate(order, 1):
        r = [label[g], int(f(AE, g)["n_group_pos"])]
        for j, m in enumerate(MODELS):
            r.append(gap_ci(f(m, g), "eo"))
            if f(m, g)["eo_unfair"]:
                bold.add((i, j + 2))
        rows.append(r)
    st += [KeepTogether([table(rows, [32, 18, 40, 40, 40], bold), P("Table 4. Equal opportunity gaps with confidence intervals. Bold marks comparisons that meet the verdict rule.", "cap")])]
    rows = [["Comparison", "True income gap", *MODELS]]
    for g in order:
        rows.append([label[g], num(f(AE, g)["base_rate_gap"]), *[gap_ci(f(m, g), "dp") for m in MODELS]])
    st += [KeepTogether([table(rows, [32, 18, 40, 40, 40]), P("Table 5. Demographic parity gaps with confidence intervals. The true income gap is what a perfect classifier would produce on these test records.", "cap")])]
    st += [figure(R / "fig2_fairness_gaps.png", "Figure 1. Fairness gaps with confidence intervals. Dashed lines mark the ±0.10 cutoff; black bars mark the true income gap.")]
    fa, fs0, fs1 = f(AE, "Female"), f(SV0, "Female"), f(SV1, "Female")
    st += bullets([
        f"<b>Sex, equal opportunity:</b> the autoencoder-based classifier identifies {100*rates[AE][0]['tpr']:.1f}% of high-earning women against {100*rates[AE][1]['tpr']:.1f}% of men. The SVM narrows this to {100*rates[SV0][0]['tpr']:.1f}% against {100*rates[SV0][1]['tpr']:.1f}%, a borderline result whose interval still excludes 0.",
        f"<b>Sex, demographic parity:</b> the gaps are large for all models, but the SVM's ({num(fs0['dp_gap'])}) is close to the true income gap ({num(fs0['base_rate_gap'])}). The autoencoder-based classifier's interval ({num(fa['dp_lo'])} to {num(fa['dp_hi'])}) lies beyond the true gap, so it widens the difference slightly.",
        f"<b>False positives:</b> women also have a lower false positive rate ({num(fs0['fpr_gap'])} for the SVM, {num(fa['fpr_gap'])} for the autoencoder-based classifier), so these models are generally less likely to predict high income for women.",
        f"<b>Race:</b> only Black vs White under the autoencoder-based classifier meets the rule, and only just: {num(f(AE,'Black')['eo_gap'])}, with an interval of {num(f(AE,'Black')['eo_lo'])} to {num(f(AE,'Black')['eo_hi'])}. Demographic parity gaps for Black people match the true income gap. Asian-Pac intervals include 0, and the intervals for Amer-Indian-Eskimo and Other are too wide for any conclusion.",
    ])

    st += [P("4.3 Reconstruction error", "h2")]
    rows = [["Group", "Test records", "Autoencoder", "Gap to reference (CI)", "Median", "PCA"]]
    for g in ["Male", "Female", "White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]:
        r = rec.loc[g]
        gap = "reference" if g in ("Male", "White") else f"{num(r['error_gap'],4)} ({num(r['error_gap_lo'],4)} to {num(r['error_gap_hi'],4)})"
        rows.append([g, f"{int(r['n_test']):,}", num(r["shared_error"], 4), gap, num(r["shared_median"], 4), num(r["pca_error"], 4)])
    st += [KeepTogether([table(rows, [44, 20, 24, 48, 17, 17]), P("Table 6. Reconstruction error by group, averaged over 3 seeds.", "cap")]),
           figure(R / "fig1_reconstruction.png", "Figure 2. Reconstruction error by group for the autoencoder and for PCA with the same code size.", width=PAGE_W * 0.82)]
    st += bullets([
        f"<b>Sex:</b> no meaningful gap. The difference is {num(rec.loc['Female','error_gap'],4)}, the interval reaches 0, and the median points the other way (women {num(rec.loc['Female','shared_median'],4)}, men {num(rec.loc['Male','shared_median'],4)}). Overall error also varied more between seeds ({min(s1['seed_mean_error']):.4f} to {max(s1['seed_mean_error']):.4f}) than this gap.",
        f"<b>Race:</b> error for Asian-Pac is {ratio('Asian-Pac-Islander'):.1f} times White error and for Other {ratio('Other'):.1f} times, with intervals well above 0. The pattern holds at other code sizes: the Asian-Pac to White ratio is {k(8,'Asian-Pac-Islander')/k(8,'White'):.1f} at size 8 and {k(32,'Asian-Pac-Islander')/k(32,'White'):.1f} at size 32. PCA shows the same ordering.",
        f"<b>Autoencoder vs PCA:</b> the autoencoder's mean error is {np.mean(s1['seed_mean_error']):.4f} against {s1['pca_mean_error']:.4f} for PCA.",
    ])

    st += [P("4.4 Why reconstruction error differs by race", "h2"),
           P(f"<i>Native-country</i> accounts for {share('Asian-Pac-Islander'):.0f}% of Asian-Pac error and {share('Other'):.0f}% of Other error, compared with {share('White'):.0f}% for White. Most records are US-born, so rarer countries are rebuilt poorly."),
           ]
    rows = [["Training records", "White", "Black", "Asian-Pac", "Other"]]
    for nn in (244, 911, 2811):
        row = [f"{nn:,}"]
        for g in ["White", "Black", "Asian-Pac-Islander", "Other"]:
            v = size[(size["n_train"] == nn) & (size["group"] == g)]["own_error"]
            row.append(num(v.iloc[0], 4) if len(v) else "not enough data")
        rows.append(row)
    st += [KeepTogether([P("A second test asked whether these groups are harder to reconstruct regardless of the shared model, by training a separate autoencoder for each group on the same number of records:"),
                         table(rows, [40, 32, 32, 33, 33]), P("Table 7. Each group's error under an autoencoder trained only on that group, at equal training sizes.", "cap")]),
           P(f"Asian-Pac error remains higher than White at equal training size ({num(sm(911,'Asian-Pac-Islander'),4)} vs {num(sm(911,'White'),4)} at 911 records), so part of the gap comes from more varied data rather than from under-representation alone. The two causes could not be fully separated."),
           P("The planned check was to subtract each group's own-model error from its shared-model error. It failed for race: small groups' own models had too little data and did worse than the shared model, which makes the result meaningless. It is therefore not used as evidence.")]

    st += [KeepTogether([P("5. Verdict", "h1"), table([
        ["Model", "Sex", "Race"],
        ["Autoencoder (reconstruction)", "No meaningful gap", "Unequal for Asian-Pac and Other; partly due to harder data"],
        ["AE latent + LR", f"<b>Unfair to women</b> (EO {num(fa['eo_gap'])})", f"Black vs White marginally unfair (EO {num(f(AE,'Black')['eo_gap'])}); Amer-Indian and Other inconclusive"],
        ["SVM without sex/race", f"Borderline (EO {num(fs0['eo_gap'])}, just under the cutoff)", "No gap meets the rule; Amer-Indian and Other inconclusive"],
        ["SVM with sex/race", f"Marginally unfair (EO {num(fs1['eo_gap'])})", "No gap meets the rule; Amer-Indian and Other inconclusive"],
    ], [46, 52, 72]), P("Table 8. Verdict by model and attribute. EO = equal opportunity gap.", "cap")])]
    st += [P("The SVM was both more accurate and fairer on equal opportunity for women than the autoencoder-based classifier. A likely reason is that the autoencoder is trained to reconstruct inputs rather than predict income, so compression may discard information useful for recognising high-earning women. Question 3 gives partial support for this: the same network, once fine-tuned to predict income, has a much smaller gap (Section 10)."),
           P(f"The SVM's demographic parity gap should not be read as proof of fairness just because it resembles the true income gap. The {100*sx.loc['Male','mean']:.1f}% and {100*sx.loc['Female','mean']:.1f}% income rates for men and women may themselves reflect historical inequality, which a model that matches the labels will reproduce.")]

    # ---------------- Question 2 ----------------
    st += [P("Question 2: Unfair to fair", "part"), P("6. Approach", "h1"),
           P(f"The autoencoder-based classifier (AE latent + LR) was chosen because it had the clearest unfairness in Question 1. Several fixes were tried on the same frozen encoder and the same test set, and the baseline reproduced Question 1's predictions exactly.")]
    st += bullets([
        "<b>Exponentiated Gradient (EG)</b> (Agarwal et al., 2018) trains a weighted mixture of logistic regressions subject to a limit on the difference in TPR (bound 0.01) or in selection rate. It needs the sensitive attribute in training but not at prediction, and its decisions are randomised across the mixture.",
        "<b>Reweighing</b> (Kamiran and Calders, 2012) weights training records so that label and group look independent.",
        "<b>Group thresholds</b> (Hardt et al., 2016) set a separate decision threshold per group. This needs each person's group at prediction time.",
        "<b>Balanced autoencoder</b> retrains the encoder with group-balanced sampling, aimed at the reconstruction gap.",
    ])
    st += [P("<b>Choosing without using the test set.</b> Validation was split in half: one half fitted group thresholds and the other chose the method. The rule was set before looking at test results: among methods that do not need the person's group at prediction and do not target demographic parity, keep those whose worst validation equal opportunity gap (over women, Black and Asian-Pac) is at most 0.10, then take the highest validation accuracy. Only EG for equal opportunity by sex qualified.")]
    names = {BF: "Before (AE + LR)", EGM: "<b>EG equal opportunity, sex (chosen)</b>", "EG equal opportunity (sex x race)": "EG equal opportunity, sex x race", "EG demographic parity (sex)": "EG demographic parity, sex",
             "Reweighing": "Reweighing", "Group thresholds (sex x race)": "Group thresholds, sex x race", "Balanced AE (sqrt weights) + LR": "Balanced AE (sqrt weights) + LR", "Balanced AE (full weights) + LR": "Balanced AE (full weights) + LR"}
    rows = [["Method", "Worst EO gap, validation", "Accuracy", "Recall", "TPR, women", "TPR, men", "EO gap, women", "DP gap, women"]]
    for m, lab in names.items():
        r = g2(m, "Female")
        rows.append([lab, num(q2o.loc[m, "val_worst_eo"]), num(q2o.loc[m, "accuracy"]), num(q2o.loc[m, "recall"]), num(r["tpr_group"]), num(r["tpr_reference"]), num(r["eo_gap"]), num(r["dp_gap"])])
    dpm = g2("EG demographic parity (sex)", "Female")
    st += [KeepTogether([table(rows, [58, 18, 16, 14, 15, 15, 17, 17]), P("Table 9. All methods tried. Columns after the second use the test set. Averaged over 3 seeds. EO = equal opportunity, DP = demographic parity.", "cap")]),
           P(f"Table 9 shows a trade-off. EG for demographic parity nearly closes the demographic parity gap ({num(dpm['dp_gap'])}) but overshoots on equal opportunity ({num(dpm['eo_gap'])}), and accuracy falls to {num(q2o.loc['EG demographic parity (sex)','accuracy'])}. Because income rates differ by sex, the two measures generally cannot both be satisfied, so a choice was needed. Equal opportunity remained the target, as in Question 1.")]

    st += [P("7. Results", "h1"), P("7.1 Task performance", "h2"),
           P(f"Accuracy fell from {num(q2o.loc[BF,'accuracy'])} to {num(q2o.loc[EGM,'accuracy'])}, a small cost. Recall fell more, from {num(q2o.loc[BF,'recall'])} to {num(q2o.loc[EGM,'recall'])}, and precision was {num(q2o.loc[BF,'precision'])} before and {num(q2o.loc[EGM,'precision'])} after. Accuracy also varied more between seeds after the fix (standard deviation {num(q2o.loc[EGM,'accuracy_sd'])} against {num(q2o.loc[BF,'accuracy_sd'])}), which is expected for a randomised classifier.")]
    show = [("Female", "Women vs men"), ("White Female", "White women vs White men"), ("Black", "Black vs White"), ("Black Male", "Black men vs White men"), ("Asian-Pac-Islander", "Asian-Pac vs White"), ("Amer-Indian-Eskimo", "Amer-Indian vs White"), ("Other", "Other vs White")]
    rows, bold = [["Comparison", "High earners", "EO gap before (CI)", "EO gap after (CI)", "Reduction (CI)", "DP before", "DP after", "Result"]], set()
    for i, (g, lab) in enumerate(show, 1):
        b, a = g2(BF, g), g2(EGM, g)
        if a["n_pos"] < 30:
            res = "Too few to judge"
        elif a["eo_red_lo"] > 0:
            res = "<b>Reduced</b>"
        elif a["eo_unfair"]:
            res = "No detectable change; meets unfair rule"
        elif b["eo_unfair"]:
            res = "Below rule; reduction not shown"
        else:
            res = "Not unfair before or after"
        rows.append([lab, int(a["n_pos"]), gap_ci(b, "eo"), gap_ci(a, "eo"), f"{num(a['eo_reduction'])} ({num(a['eo_red_lo'])} to {num(a['eo_red_hi'])})", num(b["dp_gap"]), num(a["dp_gap"]), res])
        if b["eo_unfair"]:
            bold.add((i, 2))
        if a["eo_unfair"] and a["n_pos"] >= 30:
            bold.add((i, 3))
    st += [KeepTogether([P("7.2 Fairness before and after", "h2"), table(rows, [26, 14, 27, 27, 26, 13, 13, 24], bold), P("Table 10. Before and after EG. Bold marks gaps that meet the verdict rule, among groups with at least 30 high earners. A positive reduction means the gap shrank. Intervals are 95% for sex, 98.75% for race and 99.4% for sex x race, and come from a paired bootstrap so that before and after are compared on the same resamples.", "cap")]),
           figure(R / "q2/fig1_before_after_gaps.png", "Figure 3. Equal opportunity gaps before and after EG, with confidence intervals.", width=PAGE_W * 0.9)]
    bm0, bm1, wf0, wf1 = g2(BF, "Black Male"), g2(EGM, "Black Male"), g2(BF, "White Female"), g2(EGM, "White Female")
    st += [P("7.3 Which groups improved", "h2")]
    st += bullets([
        f"<b>Women had the largest measurable disparity and it is now reduced.</b> The gap moved from {num(q2b['eo_gap'])} to {num(q2a['eo_gap'])}, a reduction of {num(q2a['eo_reduction'])} (CI {num(q2a['eo_red_lo'])} to {num(q2a['eo_red_hi'])}). White women gained most clearly: {num(wf0['eo_gap'])} to {num(wf1['eo_gap'])} against White men.",
        f"<b>Part of this came from lowering men's rate.</b> TPR rose from {num(q2b['tpr_group'])} to {num(q2a['tpr_group'])} for women and fell from {num(q2b['tpr_reference'])} to {num(q2a['tpr_reference'])} for men (Figure 4).",
        f"<b>Demographic parity improved but is not fixed.</b> The women's gap went from {num(q2b['dp_gap'])} to {num(q2a['dp_gap'])}, a reduction of {num(q2a['dp_reduction'])} (CI {num(q2a['dp_red_lo'])} to {num(q2a['dp_red_hi'])}), but it is still above 0.10. So the model is fair for women on equal opportunity, not on demographic parity.",
        f"<b>Black people:</b> the Black vs White gap fell below the rule ({num(g2(BF,'Black')['eo_gap'])} to {num(g2(EGM,'Black')['eo_gap'])}), but that change is within noise, so it should not be counted as a real reduction.",
        f"<b>Black men did not benefit.</b> Their gap to White men did not detectably change ({num(bm0['eo_gap'])} to {num(bm1['eo_gap'])}), and their TPR fell from {num(bm0['tpr_group'])} to {num(bm1['tpr_group'])}. It now just meets the unfair rule, though it was only narrowly outside the rule before. The fix targeted sex alone. Targeting sex x race as well cost too much: accuracy fell to {num(q2o.loc['EG equal opportunity (sex x race)','accuracy'])} and recall to {num(q2o.loc['EG equal opportunity (sex x race)','recall'])}.",
        "<b>Small groups:</b> Amer-Indian-Eskimo and Other have 11 and 9 high earners in the test set, so nothing can be said about them. They should not be read as improved.",
    ])
    st += [figure(R / "q2/fig2_who_gains.png", "Figure 4. TPR for women and men under each method. Equal rates can come from lowering one group as well as raising the other.", width=PAGE_W * 0.85)]
    tr = lambda s, c: tpr_rel.loc[(s, c)]
    rows = [["Relationship (true high earners)", "Sex", "Records", "TPR before", "TPR after EG"]]
    for s, c in [("Female", "Wife"), ("Male", "Husband"), ("Female", "Not-in-family"), ("Male", "Not-in-family"), ("Female", "Unmarried"), ("Male", "Unmarried")]:
        v = tr(s, c)
        rows.append([c, "Women" if s == "Female" else "Men", int(v["n"]), num(v["b"]), num(v["a"])])
    st += [KeepTogether([P(f"<b>Equal rates by sex can hide differences inside each sex.</b> Table 11 breaks TPR down by relationship. Before the fix, women and men in the same category had similar rates; the gap was mainly wives ({num(tr('Female','Wife')['b'])}) against husbands ({num(tr('Male','Husband')['b'])}). After the fix the order is reversed, with wives at {num(tr('Female','Wife')['a'])} and husbands at {num(tr('Male','Husband')['a'])}. Women who are not wives gain less overall: {num(nw_b)} to {num(nw_a)}, against {num(tr('Female','Wife')['b'])} to {num(tr('Female','Wife')['a'])} for wives. Question 3 explains why."),
                         table(rows, [60, 25, 25, 30, 30]), P("Table 11. TPR by relationship category, before and after EG. Categories with fewer than 28 high earners are not shown.", "cap")])]
    rs = lambda m, c: q2r[c].mean()[m]
    rd = lambda m, c: q2r[c].std()[m]
    rows = [["Autoencoder", "Overall error", "White error", "Asian-Pac error", "Asian-Pac / White ratio", "Other / White ratio"]]
    for m, lab in (("Before (AE)", "Standard"), ("Balanced AE (sqrt weights)", "Balanced (square-root weights)"), ("Balanced AE (full weights)", "Balanced (full weights)")):
        rows.append([lab, num(rs(m, "overall"), 4), num(rs(m, "white"), 4), num(rs(m, "asian_pac"), 4), f"{rs(m,'asian_ratio'):.2f} ± {rd(m,'asian_ratio'):.2f}", f"{rs(m,'other_ratio'):.2f} ± {rd(m,'other_ratio'):.2f}"])
    st += [KeepTogether([P("7.4 Reconstruction error", "h2"), table(rows, [52, 22, 22, 24, 26, 24]), P("Table 12. Reconstruction error with group-balanced training. Mean and standard deviation over 3 seeds.", "cap")]),
           P(f"Balanced training narrowed the Asian-Pac to White ratio from {rs('Before (AE)','asian_ratio'):.2f} to {rs('Balanced AE (full weights)','asian_ratio'):.2f}, but this is <b>not established as a real fix</b>. The change is about the size of the variation between seeds, it comes partly from raising White error ({num(rs('Before (AE)','white'),4)} to {num(rs('Balanced AE (full weights)','white'),4)}), and overall error rose. This fits Question 1's finding that part of the gap comes from harder data, which sampling cannot fix.")]
    st += [P("8. Verdict for Question 2", "h1"),
           P(f"The model is now fair for women on equal opportunity under the rule used here, at a small accuracy cost. It is not fair in every sense: demographic parity for women is still above 0.10, Black men were not helped, wives now rank above husbands, and the reconstruction gap by race remains.")]

    # ---------------- Question 3 ----------------
    st += [P("Question 3: Interpretability of unfairness and fairness", "part"), P("9. Method", "h1"), P("9.1 Models compared", "h2")]
    st += bullets([
        "<b>AE + LR and AE + EG:</b> the before and after models from Question 2.",
        f"<b>Network with and without a fair loss:</b> EG is a constrained training method rather than a loss, so a network with an explicit fair optimization loss was added. The encoder and classifier are fine-tuned together on binary cross-entropy + λ(TPR<sub>women</sub> – TPR<sub>men</sub>)<super>2</super>, using soft TPR so that the term can be differentiated. λ = 0 gives the \"before\" network. λ was chosen once for all seeds, on validation only, using the same rule as Question 2, which gave λ = {q3m['lambda']}.",
    ])
    st += [P("All four models see the same 10 original features. None sees sex or race at prediction; sex is used only during training by the two fair models.")]
    st += [P("9.2 Interpretability techniques", "h2")]
    st += bullets([
        f"<b>SHAP</b> (Lundberg and Lee, 2017): exact Shapley values on the predicted log-odds, computed over all 2<super>10</super> combinations of the 10 features rather than sampled, for every test record and seed, using {q3m['n_background']} background records. One-hot columns count as one feature. The values match the shap library's KernelExplainer to within 0.000001 on spot checks, and a different background sample gives very similar values (correlation at least {min(q3m['background_stability_corr'].values()):.3f}).",
        "<b>Vanilla gradients</b> (Simonyan et al., 2014) of the log-odds with respect to the inputs, summarised in two ways: size (how sensitive the score is to a feature) and signed contribution, gradient × (input – mean input), which estimates how much a feature moved this record's score.",
        "<b>Scramble test:</b> attributions describe scores, while fairness depends on decisions. As a check, each feature was shuffled across all test records (5 repeats, no retraining) and the equal opportunity gap measured again.",
    ])
    st += [P("Attributions are compared as a <b>SHAP gap</b>: the mean value for high-earning women minus the mean for high-earning men, in log-odds. A negative gap means the feature pulls women down relative to men. High earners are used because equal opportunity is defined on them.")]

    st += [P("10. Did the network become fair after the fair loss?", "h1")]
    rows, bold = [["Model", "Accuracy", "Recall", "Women vs men", "White women vs White men", "Black vs White", "Black men vs White men", "DP gap, women"]], set()
    for i, m in enumerate(Q3M, 1):
        row = [m, num(q3o.loc[m, "accuracy"]), num(q3o.loc[m, "recall"])]
        for j, g in enumerate(["Female", "White Female", "Black", "Black Male"]):
            row.append(num(g3(m, g)["eo_gap"]))
            if g3(m, g)["eo_unfair"]:
                bold.add((i, j + 3))
        row.append(num(g3(m, "Female")["dp_gap"]))
        rows.append(row)
    wn0, wn1 = g3(M2, "White Female"), g3(M3, "White Female")
    st += [KeepTogether([table(rows, [50, 16, 14, 18, 20, 16, 20, 16], bold), P("Table 13. Performance and equal opportunity gaps for the four models. Bold marks gaps that meet the verdict rule. EG gaps here use the expected value of its random decision, so they differ slightly from Table 10, which uses sampled decisions (for example Black vs White, " + num(g3(M1, "Black")["eo_gap"]) + " against " + num(g2(EGM, "Black")["eo_gap"]) + ").", "cap")])]
    st += bullets([
        f"<b>The fair loss brought the gaps close to zero, for little cost.</b> The women's gap moved from {gap_ci(n0,'eo')} to {gap_ci(n1,'eo')}, and accuracy from {num(q3o.loc[M2,'accuracy'])} to {num(q3o.loc[M3,'accuracy'])}.",
        f"<b>The result is qualified.</b> For women overall, the network was already under the 0.10 cutoff before the loss, so it did not go from unfair to fair on that comparison. White women vs White men did meet the rule ({num(wn0['eo_gap'])}) and no longer does ({num(wn1['eo_gap'])}). Neither reduction is statistically established: the intervals are {num(n1['eo_red_lo'])} to {num(n1['eo_red_hi'])} and {num(wn1['eo_red_lo'])} to {num(wn1['eo_red_hi'])}.",
        f"<b>Not fair in every sense.</b> The women's demographic parity gap stayed large ({num(n0['dp_gap'])} to {num(n1['dp_gap'])}), and race gaps did not improve (Black men {num(g3(M2,'Black Male')['eo_gap'])} to {num(g3(M3,'Black Male')['eo_gap'])}).",
        f"<b>Task training mattered more than the fair loss.</b> The largest single improvement came from fine-tuning the encoder for income at all: the gap went from {num(g3(M0,'Female')['eo_gap'])} (AE + LR) to {num(n0['eo_gap'])} with λ = 0, and accuracy rose from {num(q3o.loc[M0,'accuracy'])} to {num(q3o.loc[M2,'accuracy'])}. No interval was computed for this comparison.",
    ])

    rows, bold = [["Feature", *Q3M]], set()
    feats = ["relationship", "marital-status", "hours-per-week", "education-num", "capital-gain", "occupation", "age"]
    for i, ft in enumerate(feats, 1):
        row = [ft]
        for j, m in enumerate(Q3M):
            v = sg(m, WM, ft)
            row.append(f"{num(v['shap_gap'])} (± {v['seed_sd']:.3f})")
            if abs(v["shap_gap"]) >= 0.2:
                bold.add((i, j + 1))
        rows.append(row)
    st += [KeepTogether([P("11. Which features drive the unfairness?", "h1"), P("11.1 SHAP gaps between women and men", "h2"), table(rows, [34, 34, 34, 34, 34], bold), P("Table 14. SHAP gap, women minus men, among true high earners, in log-odds, with the standard deviation over 3 seeds. Bold marks gaps of at least 0.2 in size. <i>Capital-loss</i>, <i>workclass</i> and <i>native-country</i> are under 0.03 in every model and are not shown.", "cap")]),
           figure(R / "q3/fig1_shap_gap_women_men.png", "Figure 5. SHAP gaps between women and men among high earners, before and after each fix.")]
    rm0, rm1, rm2, rm3 = [sg(m, WM, "relationship")["shap_gap"] for m in Q3M]
    st += bullets([
        f"<b>Before the fixes,</b> <i>relationship</i> is the largest factor in AE + LR ({num(rm0)}) and <i>marital-status</i> in the network ({num(sg(M2,WM,'marital-status')['shap_gap'])}). The network's <i>relationship</i> gap ({num(rm2)}) is within seed noise. These two features overlap heavily, so credit can move between them from one seed to another, as the large seed variation suggests.",
        f"<b>After the fixes,</b> the main change is in <i>relationship</i>, which flips sign: {num(rm0)} to {num(rm1)} with EG, and {num(rm2)} to {num(rm3)} with the fair loss. <i>Marital-status</i> barely moves ({num(sg(M2,WM,'marital-status')['shap_gap'])} to {num(sg(M3,WM,'marital-status')['shap_gap'])} in the network). EG also shifts <i>occupation</i> ({num(sg(M0,WM,'occupation')['shap_gap'])} to {num(sg(M1,WM,'occupation')['shap_gap'])}) and <i>hours-per-week</i>.",
        "<b>So the bias is offset, not removed.</b> After the fair loss, the negative <i>marital-status</i> gap is still there and is balanced by a positive <i>relationship</i> gap.",
    ])

    rows = [["Category", "Share of high-earning women", "Share of high-earning men", *Q3M]]
    for c in ["Husband", "Wife", "Not-in-family", "Unmarried", "Own-child"]:
        rows.append([c, f"{100*rel_share.loc[c,'Female']:.1f}%", f"{100*rel_share.loc[c,'Male']:.1f}%", *[f"{rel_mean.loc[c,m]:+.2f}".replace("-", "–") for m in Q3M]])
    st += [KeepTogether([P("11.2 <i>Relationship</i> works as a proxy for sex", "h2"), table(rows, [28, 24, 24, 23, 23, 25, 23]), P("Table 15. Mean SHAP value of <i>relationship</i> by category, among high earners. Husband and Wife are held by one sex only.", "cap")])]
    gm = lambda g, ft: grad(M0, g, ft)
    st += bullets([
        f"<b>The gap comes entirely from who holds each category.</b> {100*rel_share.loc['Husband','Male']:.0f}% of high-earning men are Husbands and {100*rel_share.loc['Wife','Female']:.0f}% of high-earning women are Wives. Weighting one shared set of category means by each sex's mix reproduces the full gap in every model ({num(rel_comp[M0])} against {num(rm0)} for AE + LR). Women and men in the same shared category receive almost the same value, so the models do not treat them differently within a category.",
        f"<b>Gradients agree.</b> The score's sensitivity to <i>relationship</i> is about equal for women and men in AE + LR (gradient size {gm('Female','relationship'):.2f} and {gm('Male','relationship'):.2f}), even though the SHAP values differ widely. The disparity comes from the values women and men hold, not from a different sensitivity.",
        f"<b>The fixes push the proxy in women's favour.</b> In AE + LR, Husband ({rel_mean.loc['Husband',M0]:+.2f}) scores above Wife ({rel_mean.loc['Wife',M0]:+.2f}), and EG reverses this (Wife {rel_mean.loc['Wife',M1]:+.2f}, Husband {rel_mean.loc['Husband',M1]:+.2f}). In the network, Wife was already ahead (Wife {rel_mean.loc['Wife',M2]:+.2f}, Husband {rel_mean.loc['Husband',M2]:+.2f}), and the fair loss widens the lead (Wife {rel_mean.loc['Wife',M3]:+.2f}, Husband {rel_mean.loc['Husband',M3]:+.2f}). Sex is not an input, so the fair models move towards parity by relying on this proxy for sex in women's favour.",
        f"<b><i>Marital-status</i> is a weaker signal of sex.</b> Its categories are not sex-specific, but high-earning women are less often married ({100*mar_share['Female']:.0f}% against {100*mar_share['Male']:.0f}% of men), and the fixes leave its gap largely in place.",
        f"<b>Women who are not wives get little from this.</b> Under the fair loss, Unmarried stays at {rel_mean.loc['Unmarried',M3]:+.2f} and Not-in-family at {rel_mean.loc['Not-in-family',M3]:+.2f}. Parity between the sexes therefore rests largely on wives. Table 11 shows a similar pattern in EG's decisions; decisions by category were not saved for the network.".replace("-0", "–0"),
    ])

    rows = [["Comparison", "Feature scrambled", *Q3M]]
    for c, fts in ((WM, ["(nothing scrambled)", "relationship", "marital-status", "relationship + marital-status"]), (BM, ["(nothing scrambled)", "education-num", "relationship + marital-status"])):
        for ft in fts:
            rows.append([c, ft, *[num(sc(m, c, ft)) for m in Q3M]])
    st += [KeepTogether([P("11.3 Scramble test", "h2"), table(rows, [38, 44, 22, 22, 22, 22]), P("Table 16. Equal opportunity gap after scrambling, without retraining. Mean over 3 seeds and 5 shuffles. No intervals were computed.", "cap")]),
           figure(R / "q3/fig2_scramble_test.png", "Figure 6. Women vs men equal opportunity gap after scrambling each feature. Only the features that move the gap most are shown.", width=PAGE_W * 0.88)]
    st += bullets([
        f"<b>The test supports SHAP.</b> Scrambling both features nearly removes the women's gap in the \"before\" models: {num(sc(M0,WM,'(nothing scrambled)'))} to {num(sc(M0,WM,'relationship + marital-status'))} in AE + LR and {num(sc(M2,WM,'(nothing scrambled)'))} to {num(sc(M2,WM,'relationship + marital-status'))} in the network. Matching Table 14, <i>relationship</i> matters most in AE + LR and <i>marital-status</i> in the network.",
        f"<b>It also supports the offset.</b> Scrambling <i>relationship</i> in the fair-loss network brings the gap back to {num(sc(M3,WM,'relationship'))}, from {num(sc(M3,WM,'(nothing scrambled)'))}. That feature is now compensating for the others.",
        f"<b>Race has an additional driver, which helps explain Black men in Question 2.</b> <i>Education-num</i> has the largest SHAP gap between Black and White men ({num(sg(M0,BM,'education-num')['shap_gap'])}), and EG leaves it unchanged ({num(sg(M1,BM,'education-num')['shap_gap'])}). Scrambling it cuts their gap from {num(sc(M0,BM,'(nothing scrambled)'))} to {num(sc(M0,BM,'education-num'))}, and scrambling <i>relationship</i> and <i>marital-status</i> has a similar effect ({num(sc(M0,BM,'relationship + marital-status'))}). A fix aimed at sex would not be expected to change the education gap. Education is a legitimate predictor but may itself reflect unequal access, and these models cannot settle whether this gap is unfair.",
    ])

    st += [P("12. Do the two techniques agree?", "h1"),
           P(f"It depends on what is compared. <b>Signed gradient contributions agree closely with SHAP</b> on the women's gaps (correlation {min(r_first):.3f} to {max(r_first):.3f} across models). This is partly expected, because the signed contribution is effectively a linear approximation of SHAP, so it is not fully independent evidence. <b>Gradient size is a different quantity</b>: its correlation with SHAP gaps ranges from {num(min(r_sal),2)} to {num(max(r_sal),2)}, and rank agreement with SHAP importance among high earners is only {ag.min():.2f} to {ag.max():.2f}. Figure 7 shows the difference: gradient size gives <i>native-country</i> the most importance, while SHAP gives it almost none. A likely reason is that gradient size is taken across all of a feature's one-hot columns, and <i>native-country</i> has by far the most ({col_count['native-country']}, against {col_count['occupation']} for <i>occupation</i> and {col_count['relationship']} for <i>relationship</i>). The scramble test sides with SHAP: scrambling <i>native-country</i> leaves the gap almost unchanged ({num(sc(M0,WM,'(nothing scrambled)'))} to {num(sc(M0,WM,'native-country'))} in AE + LR). For this question SHAP and the signed contribution are the useful measures, and gradient size alone would have pointed to the wrong feature."),
           figure(R / "q3/fig3_shap_vs_gradient.png", "Figure 7. Share of total importance per feature for the network without a fair loss, among high earners.", width=PAGE_W * 0.82)]

    st += [P("13. Verdict for Question 3", "h1")]
    st += [KeepTogether([table([
        ["Question", "Answer", "Evidence"],
        ["Do some features contribute disproportionately for particular groups?", "<b>Yes.</b> <i>Relationship</i> and <i>marital-status</i> for women; <i>education-num</i> as well as these two for Black men", "Tables 14 and 16"],
        ["Do those features make the model unfair?", "<b>Yes, for these trained models.</b> Scrambling the first two nearly removes the women's gap", "Table 16"],
        ["Why do they do this?", "<i>Relationship</i> works as a proxy for sex, because Husband and Wife are sex-specific categories. <i>Marital-status</i> is a weaker signal", "Table 15"],
        ["Did the network become fair after the fair loss?", f"<b>Partly.</b> Gaps are near zero and white women no longer meet the rule, but the network was under the cutoff for women overall, the reductions are not statistically established, and demographic parity is still above 0.10", "Table 13"],
        ["Did the model's reliance on these features end?", "<b>No.</b> The fixes push the proxy in women's favour, offsetting the bias rather than removing it", "Tables 14, 15 and 16"],
    ], [66, 80, 24]), P("Table 17. Answers to Question 3.", "cap")])]

    st += [P("14. Limitations", "h1")]
    st += bullets([
        "<b>Old data:</b> the census data dates from 1994 and may not reflect current patterns.",
        "<b>Simplified labels:</b> sex is recorded as binary and race in five broad categories, which hides differences within groups. Intersections of sex and race were analysed in Questions 2 and 3, but most have too few high earners to judge.",
        "<b>Small groups:</b> small test groups give wide intervals, so \"no gap found\" does not mean \"fair\". Comparisons with fewer than 30 high earners are treated as inconclusive.",
        "<b>Cutoff sensitivity:</b> several verdicts are close to the 0.10 cutoff, including the SVM for women, the network for women, and Black men, so a small move in the cutoff would change them.",
        "<b>Uncertainty coverage:</b> intervals reflect test-set sampling, not differences between training runs. With 3 seeds, seed variation is only roughly estimated, and it is large for some SHAP gaps.",
        "<b>Randomised decisions:</b> EG can give different decisions for the same record, which may be unacceptable in practice, and it can only be explained through its expected score.",
        "<b>Attribution is not causation:</b> SHAP and scrambling show what these trained models rely on, not what would happen if a feature were removed before training or changed in the real world. Both also create unrealistic records, such as a Husband who has never married, because <i>relationship</i> and <i>marital-status</i> are changed independently.",
        "<b>Scramble test:</b> scrambling a strong predictor weakens the model as a whole, which could shrink gaps on its own. Accuracy after scrambling was not recorded, and the test has no intervals, so it supports the SHAP findings rather than proving them.",
        "<b>Techniques not used:</b> LIME, attention and prototypes were not tried. The two techniques used are not fully independent of each other, which is why the scramble test was added.",
        "<b>Sample size:</b> the SVM used 15,000 of the available training records and may perform differently on all of them. It was not mitigated.",
        "<b>Mixed-scale error:</b> reconstruction error averages over mostly one-hot columns, so its absolute size is hard to interpret; group comparisons are more meaningful.",
        "<b>Scope:</b> neither model is a foundation model. The results apply to these models on this dataset only.",
    ])

    st += [P("15. Conclusion", "h1"),
           P("Fairness problems were found, and they depend on the model and the metric. The autoencoder-based classifier is clearly unfair to high-earning women, and reconstruction error differs mainly by race. Accuracy alone would have pointed in the wrong direction, and removing sensitive columns did little."),
           P("Exponentiated Gradient closed the equal opportunity gap for women at a small accuracy cost. The fair loss brought the network's gap close to zero as well, though that network was already under the cutoff for women overall and the reduction is not statistically established. Neither result is fairness in every sense: demographic parity remains above 0.10 and Black men did not benefit."),
           P("Interpretability explains these limits. The gap for women runs through <i>relationship</i> and <i>marital-status</i>, and <i>relationship</i> stands in for sex even though sex is not an input. The fair models do not stop using it; they shift it in women's favour, so wives score above husbands while other women gain less. Parity on a chosen measure can therefore be reached without removing the underlying dependence, which is why an audit should look inside the model as well as at its gaps.")]

    st += [P("References", "h1")]
    for t in [
        "Agarwal, A., Beygelzimer, A., Dudík, M., Langford, J. and Wallach, H. (2018). A Reductions Approach to Fair Classification. ICML.",
        "Becker, B. and Kohavi, R. (1996). Adult. UCI Machine Learning Repository.",
        "Bird, S. et al. (2020). Fairlearn: A Toolkit for Assessing and Improving Fairness in AI. Microsoft Technical Report MSR-TR-2020-32.",
        "Ding, F., Hardt, M., Miller, J. and Schmidt, L. (2021). Retiring Adult: New Datasets for Fair Machine Learning. NeurIPS.",
        "Dwork, C., Hardt, M., Pitassi, T., Reingold, O. and Zemel, R. (2012). Fairness Through Awareness. ITCS.",
        "Hardt, M., Price, E. and Srebro, N. (2016). Equality of Opportunity in Supervised Learning. NeurIPS.",
        "Kamiran, F. and Calders, T. (2012). Data Preprocessing Techniques for Classification without Discrimination. Knowledge and Information Systems.",
        "Lundberg, S. and Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.",
        "Samadi, S., Tantipongpipat, U., Morgenstern, J., Singh, M. and Vempala, S. (2018). The Price of Fair PCA: One Extra Dimension. NeurIPS.",
        "Simonyan, K., Vedaldi, A. and Zisserman, A. (2014). Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps. ICLR Workshop.",
    ]:
        st.append(P(t, "ref"))

    st += [P("Appendix: Reproducing the results", "h1"),
           P("Run these in order with the project's Python environment. Question 1: <i>exp1_autoencoder.py</i>, <i>exp1_diagnostic.py</i>, <i>exp2_svm.py</i>, <i>make_plots.py</i>. Question 2: <i>q2_mitigate.py</i>, <i>q2_recon_check.py</i>, <i>q2_plots.py</i>. Question 3: <i>q3_interpret.py</i>, <i>q3_plots.py</i>. Report: <i>build_report.py</i>. Data split seed: 42. Model seeds: 0, 1, 2. Software: PyTorch, scikit-learn, Fairlearn and SHAP. Result tables are saved in the <i>results</i> folder.")]

    doc = SimpleDocTemplate("Fairness_Audit_Report.pdf", pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm, bottomMargin=18 * mm,
                            title="Fairness Audit, Mitigation and Interpretability on Adult Income", subject="Questions 1, 2 and 3")
    doc.build(st, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    main()
