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
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from common import classification_row, group_rates, load_splits

R = Path("results")
INK, MUTED, RULE, HEAD = colors.HexColor("#0b0b0b"), colors.HexColor("#52514e"), colors.HexColor("#c9c8c3"), colors.HexColor("#eef2f7")
ACCENT = colors.HexColor("#1f5fae")
base = getSampleStyleSheet()["Normal"]
S = {
    "title": ParagraphStyle("title", parent=base, fontName="Helvetica-Bold", fontSize=18, leading=22, textColor=INK, spaceAfter=4),
    "sub": ParagraphStyle("sub", parent=base, fontSize=10, leading=14, textColor=MUTED, spaceAfter=10),
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
    canvas.drawString(20 * mm, 11 * mm, "Fairness Audit on Adult Income")
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

    st = []
    st += [P("Fairness Audit of an Autoencoder and a Support Vector Machine on Adult Income", "title"),
           P("Question 1: Fairness Audit of AI-ML and Foundation Models<br/>Name: ____________________ &nbsp;&nbsp; Roll No.: ______________", "sub")]

    st += [P("Summary", "h1"),
           P(f"Two algorithms were audited on the Adult Income dataset: a <b>deep autoencoder</b> and a <b>support vector machine (SVM)</b>. "
             f"Results on {len(test):,} held-out test records show the following:")]
    st += bullets([
        f"<b>Unfairness towards women was found.</b> A classifier built on the autoencoder's compressed features correctly identifies far fewer high-earning women than men "
        f"(equal opportunity gap {num(f(AE,'Female')['eo_gap'])}). The SVM's gap is much smaller and borderline: {num(f(SV0,'Female')['eo_gap'])} without sex and race as inputs and {num(f(SV1,'Female')['eo_gap'])} with them, either side of the 0.10 cutoff.",
        f"<b>Removing sex and race did not meaningfully change the SVM's gap.</b> The change was only {abs(f(SV0,'Female')['eo_gap']-f(SV1,'Female')['eo_gap']):.3f}, and <i>relationship</i> still reveals sex.",
        f"<b>Reconstruction error is unequal across race but not sex.</b> Error for the Asian-Pacific Islander group is {ratio('Asian-Pac-Islander'):.1f} times that of the White group, while the gap between women and men is negligible. Most of this race gap comes from <i>native-country</i>, and part of it reflects data that is harder to reconstruct.",
        f"<b>No conclusion was possible for two small groups.</b> Amer-Indian-Eskimo and Other have only {f(AE,'Amer-Indian-Eskimo')['n_group_pos']} and {f(AE,'Other')['n_group_pos']} high earners in the test set.",
    ])

    st += [P("1. Aim and scope", "h1"),
           P("The assignment asks whether at least two implemented algorithms are unfair, using task-specific metrics and fairness measures. "
             "This audit uses a neural network (a deep autoencoder, list item 3) and an SVM (list item 2). It identifies unfairness only; no bias mitigation was attempted.")]

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

    st += [P("3. Method", "h1"), P("3.1 Experiment 1: Deep autoencoder", "h2")]
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

    st += [P("4. Results", "h1"), P("4.1 Classification performance", "h2")]
    rows = [["Model", "Accuracy", "Precision", "Recall", "F1", "Accuracy, women", "Accuracy, men", "TPR, women", "TPR, men"]]
    for m in MODELS:
        a, b = rates[m]
        rows.append([m, *[num(v) for v in overall[m]], num(a["accuracy"]), num(b["accuracy"]), num(a["tpr"]), num(b["tpr"])])
    st += [KeepTogether([table(rows, [38, 16, 16, 14, 13, 19, 18, 18, 18]),
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

    st += [P("5. Verdict", "h1")]
    st += [KeepTogether([table([
        ["Model", "Sex", "Race"],
        ["Autoencoder (reconstruction)", "No meaningful gap", "Unequal for Asian-Pac and Other; partly due to harder data"],
        ["AE latent + LR", f"<b>Unfair to women</b> (EO {num(fa['eo_gap'])})", f"Black vs White marginally unfair (EO {num(f(AE,'Black')['eo_gap'])}); Amer-Indian and Other inconclusive"],
        ["SVM without sex/race", f"Borderline (EO {num(fs0['eo_gap'])}, just under the cutoff)", "No gap meets the rule; Amer-Indian and Other inconclusive"],
        ["SVM with sex/race", f"Marginally unfair (EO {num(fs1['eo_gap'])})", "No gap meets the rule; Amer-Indian and Other inconclusive"],
    ], [46, 52, 72]), P("Table 8. Verdict by model and attribute. EO = equal opportunity gap.", "cap")])]
    st += [P("The SVM was both more accurate and fairer on equal opportunity for women than the autoencoder-based classifier. A likely reason is that the autoencoder is trained to reconstruct inputs rather than predict income, so compression may discard information useful for recognising high-earning women. This explanation was not tested directly."),
           P(f"The SVM's demographic parity gap should not be read as proof of fairness just because it resembles the true income gap. The {100*sx.loc['Male','mean']:.1f}% and {100*sx.loc['Female','mean']:.1f}% income rates for men and women may themselves reflect historical inequality, which a model that matches the labels will reproduce.")]

    st += [P("6. Limitations", "h1")]
    st += bullets([
        "<b>Old data:</b> the census data dates from 1994 and may not reflect current patterns.",
        "<b>Simplified labels:</b> sex is recorded as binary and race in five broad categories, which hides differences within groups. Intersectional groups such as Black women were not analysed.",
        "<b>Small groups:</b> small test groups give wide intervals, so \"no gap found\" does not mean \"fair\".",
        "<b>Cutoff sensitivity:</b> the SVM verdict for women changes with a small move in the 0.10 cutoff.",
        "<b>Proxy variables:</b> <i>relationship</i> (Husband, Wife) reveals sex, so removing the sex column does not remove the information.",
        "<b>Sample size:</b> the SVM used 15,000 of the available training records and may perform differently on all of them. The autoencoder used all records.",
        "<b>Uncertainty coverage:</b> intervals reflect test-set sampling, not differences between training runs.",
        "<b>Mixed-scale error:</b> reconstruction error averages over mostly one-hot columns, so its absolute size is hard to interpret; group comparisons are more meaningful.",
        "<b>Scope:</b> neither model is a foundation model. The results apply to these models on this dataset only.",
    ])

    st += [P("7. Conclusion", "h1"),
           P("Fairness problems were found, and they depend on the model and the metric. The autoencoder-based classifier is clearly unfair to high-earning women, and the SVM is borderline. Reconstruction error shows a different pattern, with little difference by sex and large differences by race, driven largely by <i>native-country</i>. Removing sensitive columns did little, and accuracy alone would have pointed in the wrong direction. A fairness audit therefore needs several metrics, and results for each group, rather than one overall score.")]

    st += [P("References", "h1")]
    for t in [
        "Becker, B. and Kohavi, R. (1996). Adult. UCI Machine Learning Repository.",
        "Ding, F., Hardt, M., Miller, J. and Schmidt, L. (2021). Retiring Adult: New Datasets for Fair Machine Learning. NeurIPS.",
        "Dwork, C., Hardt, M., Pitassi, T., Reingold, O. and Zemel, R. (2012). Fairness Through Awareness. ITCS.",
        "Hardt, M., Price, E. and Srebro, N. (2016). Equality of Opportunity in Supervised Learning. NeurIPS.",
        "Samadi, S., Tantipongpipat, U., Morgenstern, J., Singh, M. and Vempala, S. (2018). The Price of Fair PCA: One Extra Dimension. NeurIPS.",
    ]:
        st.append(P(t, "ref"))

    st += [P("Appendix: Reproducing the results", "h1"),
           P("Run these in order with the project's Python environment: <i>exp1_autoencoder.py</i>, <i>exp1_diagnostic.py</i>, <i>exp2_svm.py</i>, <i>make_plots.py</i>, <i>build_report.py</i>. Data split seed: 42. Model seeds: 0, 1, 2. Software: PyTorch and scikit-learn. Result tables are saved in the <i>results</i> folder.")]

    doc = SimpleDocTemplate("Fairness_Audit_Report.pdf", pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm, bottomMargin=18 * mm,
                            title="Fairness Audit of an Autoencoder and a Support Vector Machine on Adult Income", subject="Question 1: Fairness Audit of AI-ML and Foundation Models")
    doc.build(st, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    main()
