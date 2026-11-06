"""Build the 5-page Assignment-4 report as a PDF (reportlab) and as LaTeX source, from one set of content."""
import re
from pathlib import Path

import matplotlib
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

NAME = "Assignment4_Report_5pages"
HEADER = "RAI-FM (DSE428/DSE628), Assignment-4"
AUTHOR = "Payal Verma, 2611004"
TITLE = "Fairness Audit, Mitigation and Interpretability on Adult Income"
MARGIN = 1.9 * cm
TEXT_W = A4[0] - 2 * MARGIN

# Content. "–" before a number is a minus sign. <b> marks bold.
C = [
    ("h1", "Summary"),
    ("p", "This report audits machine learning models on the Adult Income dataset for unfairness, makes one of them fairer, and uses interpretability methods to explain the result. All results are from 9,769 held-out test records, averaged over 3 seeds."),
    ("ul", [
        "<b>Question 1: unfair for women.</b> A classifier on a deep autoencoder's code finds far fewer high-earning women than men (equal opportunity gap –0.257). An SVM's gap is borderline (–0.097 to –0.108). Asian-Pacific Islander reconstruction error is 3.8 times White error.",
        "<b>Question 2: fixed on equal opportunity for women only.</b> Exponentiated Gradient moved the women's gap from –0.257 to 0.029 at a cost of 0.006 accuracy. Demographic parity is still above 0.10 (–0.115), and Black men did not benefit.",
        "<b>Question 3: relationship and marital-status drive the gap,</b> and relationship works as a proxy for sex. The fair loss moved the network's gap from –0.090 to 0.022, but that network was already under the 0.10 cutoff for women overall, so it is an improvement rather than a clear change from unfair to fair.",
    ]),
    ("h1", "1. Dataset"),
    ("p", "Adult Income (Becker and Kohavi, 1996) has 48,842 records from the 1994 US census. The task is to predict whether a person earns more than $50K (23.9% do). Sex and race are audited, with Male and White as reference groups. The data was split 60/20/20, stratified on label, sex and race. Removing fnlwgt and education and one-hot encoding categories gives 84 input features without sex and race."),
    ("t", {"caption": "Table 1. Group sizes and income rates in the full dataset.", "w": [.22, .13, .13, .17],
           "rows": [["Group", "Records", "Share", "Earning >$50K"],
                    ["Male", "32,650", "66.8%", "30.4%"], ["Female", "16,192", "33.2%", "10.9%"],
                    ["White", "41,762", "85.5%", "25.4%"], ["Black", "4,685", "9.6%", "12.1%"],
                    ["Asian-Pac-Islander", "1,519", "3.1%", "26.9%"], ["Amer-Indian-Eskimo", "470", "1.0%", "11.7%"], ["Other", "406", "0.8%", "12.3%"]]}),

    ("h1", "2. Question 1: Fairness audit"),
    ("box", "<b>What was asked.</b> Implement at least two algorithms, at least one beyond logistic regression, PCA and K-means. Identify unfairness using task metrics (reconstruction error, accuracy) and fairness measures (demographic parity, equal opportunity)."),
    ("p", "<b>Models.</b> (1) A deep autoencoder (input-64-32-16-32-64-output) trained on reconstruction error, with a logistic regression on its frozen 16-dimensional code (\"AE + LR\") so that fairness gaps can be measured, and PCA as a baseline. (2) An RBF SVM on a sample of 15,000 training records, with and without sex and race as inputs."),
    ("p", "<b>Measures.</b> The demographic parity (DP) gap compares how often groups are predicted above $50K. The equal opportunity (EO) gap compares true positive rates (TPR), meaning how often real high earners are recognised. A comparison counts as unfair if the EO gap is larger than 0.10 and its confidence interval excludes 0. Intervals use 2,000 bootstrap resamples: 95% for sex and 98.75% for race. Groups with fewer than 30 high earners are treated as inconclusive."),
    ("t", {"caption": "Table 2. Test performance and gaps for women against men. Bold marks gaps that meet the verdict rule. Always predicting ≤$50K scores 0.761 accuracy.", "w": [.22, .09, .09, .09, .09, .26, .16],
           "rows": [["Model", "Accuracy", "Accuracy, women", "TPR, women", "TPR, men", "EO gap (CI)", "DP gap"],
                    ["SVM (no sex/race)", "0.858", "0.934", "0.524", "0.621", "–0.097 (–0.154 to –0.040)", "–0.183"],
                    ["SVM (with sex/race)", "0.858", "0.934", "0.518", "0.626", "<b>–0.108</b> (–0.165 to –0.054)", "–0.186"],
                    ["AE + LR", "0.839", "0.919", "0.344", "0.601", "<b>–0.257</b> (–0.306 to –0.209)", "–0.215"]]}),
    ("fig", "results/fig2_fairness_gaps.png", 0.9, "Figure 1. Fairness gaps with confidence intervals. Dashed lines mark the ±0.10 cutoff; black bars mark the true income gap."),
    ("ul", [
        "<b>Accuracy hides the problem.</b> Women have higher accuracy but lower TPR than men in every model. Fewer women earn above $50K, so predicting ≤$50K is right more often for them, even while high-earning women are missed.",
        "<b>Race.</b> Only Black vs White under AE + LR meets the rule, and only just (–0.107, interval –0.224 to –0.003). Amer-Indian-Eskimo and Other have 11 and 9 high earners, too few to judge.",
        "<b>Reconstruction error.</b> There is no meaningful gap by sex (0.0038 against 0.0036). Asian-Pac error is 3.8 times White (0.0125 against 0.0033) and Other 2.6 times. Native-country accounts for 73% of Asian-Pac error, because rare countries are rebuilt poorly. The autoencoder's mean error is 0.0037 against 0.0096 for PCA.",
    ]),
    ("p", "<b>Verdict.</b> AE + LR is clearly unfair to women. The SVM is more accurate and borderline on the rule. Removing sex and race changed little (–0.108 to –0.097)."),

    ("h1", "3. Question 2: Unfair to fair"),
    ("box", "<b>What was asked.</b> Make one algorithm fair, re-evaluate it with the same metrics, and identify the groups with the largest earlier disparities that are now reduced."),
    ("p", "<b>Approach.</b> AE + LR was fixed because it had the clearest unfairness. Exponentiated Gradient (EG; Agarwal et al., 2018) was compared with reweighing (Kamiran and Calders, 2012), group thresholds (Hardt et al., 2016) and a group-balanced autoencoder. The method was chosen on validation data only: among methods that do not need a person's group at prediction and do not target demographic parity, keep those whose worst validation EO gap is at most 0.10, then take the highest validation accuracy. Only EG for equal opportunity by sex qualified."),
    ("t", {"caption": "Table 3. Methods tried. Columns after the second use the test set.", "w": [.34, .12, .09, .09, .09, .09, .09, .09],
           "rows": [["Method", "Worst EO gap, validation", "Accuracy", "Recall", "TPR, women", "TPR, men", "EO gap, women", "DP gap, women"],
                    ["Before (AE + LR)", "0.310", "0.839", "0.562", "0.344", "0.601", "–0.257", "–0.215"],
                    ["<b>EG equal opportunity, sex (chosen)</b>", "<b>0.013</b>", "<b>0.833</b>", "<b>0.511</b>", "<b>0.535</b>", "<b>0.506</b>", "<b>0.029</b>", "<b>–0.115</b>"],
                    ["EG equal opportunity, sex × race", "0.121", "0.788", "0.318", "0.349", "0.313", "0.037", "–0.078"],
                    ["EG demographic parity, sex", "0.138", "0.809", "0.393", "0.541", "0.367", "0.175", "–0.016"],
                    ["Reweighing", "0.126", "0.841", "0.538", "0.465", "0.551", "–0.085", "–0.158"],
                    ["Group thresholds, sex × race", "0.156", "0.835", "0.556", "0.540", "0.559", "–0.019", "–0.141"],
                    ["Balanced AE (sqrt weights) + LR", "0.248", "0.839", "0.563", "0.395", "0.593", "–0.198", "–0.202"],
                    ["Balanced AE (full weights) + LR", "0.305", "0.838", "0.554", "0.333", "0.594", "–0.260", "–0.211"]]}),
    ("t", {"caption": "Table 4. Before and after EG, with paired bootstrap intervals. Bold marks gaps that meet the verdict rule.", "w": [.19, .07, .21, .21, .19, .13],
           "rows": [["Comparison", "High earners", "EO gap before (CI)", "EO gap after (CI)", "Reduction (CI)", "Result"],
                    ["Women vs men", "354", "<b>–0.257</b> (–0.306 to –0.209)", "0.029 (–0.022 to 0.080)", "0.228 (0.131 to 0.293)", "Reduced"],
                    ["White women vs White men", "309", "<b>–0.265</b> (–0.337 to –0.196)", "0.019 (–0.060 to 0.096)", "0.246 (0.109 to 0.314)", "Reduced"],
                    ["Black vs White", "113", "<b>–0.107</b> (–0.224 to –0.003)", "–0.085 (–0.201 to 0.023)", "0.022 (–0.041 to 0.088)", "Not shown"],
                    ["Black men vs White men", "87", "–0.127 (–0.287 to 0.013)", "<b>–0.133</b> (–0.266 to –0.004)", "–0.007 (–0.087 to 0.061)", "No change"]]}),
    ("fig", "results/q2/fig1_before_after_gaps.png", 0.62, "Figure 2. Equal opportunity gaps before and after EG, with confidence intervals."),
    ("ul", [
        "<b>Groups improved.</b> Women and White women had the largest measurable disparities, and both are now reduced. Part of this came from lowering men's rate: TPR rose from 0.344 to 0.535 for women and fell from 0.601 to 0.506 for men.",
        "<b>Not fixed.</b> The women's DP gap went from –0.215 to –0.115, still above 0.10. Black men's gap did not detectably change, and their TPR fell from 0.479 to 0.379. EG decisions are randomised, so the same person can get different answers.",
        "<b>Reconstruction.</b> Balanced training narrowed the Asian-Pac to White ratio from 3.96 to 3.03 (averaged across seeds), but the change is about the size of the seed variation and partly comes from raising White error, so it is not established as a fix.",
        "<b>Trade-off.</b> EG for demographic parity nearly closes that gap (–0.016) but overshoots on equal opportunity (0.175), with accuracy at 0.809. Because income rates differ by sex, the two measures generally cannot both be satisfied.",
    ]),
    ("p", "<b>Verdict.</b> The model is now fair for women on equal opportunity under this rule, at a small accuracy cost (0.839 to 0.833). It is not fair in every sense."),

    ("h1", "4. Question 3: Interpretability of unfairness and fairness"),
    ("box", "<b>What was asked.</b> Use at least two interpretability techniques to investigate whether features contribute disproportionately to particular groups or make the network unfair, and whether it became fair after the fair optimization loss."),
    ("p", "<b>Fair loss.</b> EG is not a loss, so a network was also fine-tuned on binary cross-entropy + λ(TPR<sub>women</sub> – TPR<sub>men</sub>)², using soft TPR. λ = 0 gives the \"before\" network, and λ = 2 was chosen on validation data. No model sees sex or race at prediction."),
    ("p", "<b>Techniques.</b> <b>SHAP</b> (Lundberg and Lee, 2017): exact Shapley values on the log-odds over all 2<super>10</super> combinations of the 10 original features. <b>Vanilla gradients</b> (Simonyan et al., 2014): gradient size and signed contribution. A <b>scramble test</b> also shuffles each feature across test records without retraining and re-measures the EO gap, because attributions describe scores while fairness depends on decisions."),
    ("t", {"caption": "Table 5. Performance and EO gaps for the four models. Bold marks gaps that meet the verdict rule. EG gaps here use the expected value of its random decision.", "w": [.28, .09, .08, .1, .13, .1, .12, .1],
           "rows": [["Model", "Accuracy", "Recall", "Women vs men", "White women vs White men", "Black vs White", "Black men vs White men", "DP gap, women"],
                    ["AE + LR (before)", "0.839", "0.562", "<b>–0.257</b>", "<b>–0.265</b>", "<b>–0.107</b>", "–0.127", "–0.215"],
                    ["AE + EG (after)", "0.833", "0.511", "0.029", "0.019", "–0.092", "<b>–0.142</b>", "–0.116"],
                    ["Network, no fair loss (before)", "0.849", "0.616", "–0.090", "<b>–0.110</b>", "–0.063", "–0.101", "–0.190"],
                    ["Network, fair loss (after)", "0.847", "0.597", "0.022", "0.008", "–0.070", "–0.107", "–0.150"]]}),
    ("p", "<b>Did the network become fair?</b> Partly. The women's gap moved from –0.090 to 0.022, and White women no longer meet the rule (–0.110 to 0.008). But the network was already under the cutoff for women overall, neither reduction is statistically established (intervals –0.036 to 0.124 and –0.050 to 0.153), the DP gap stays large (–0.150), and Black men did not improve. Fine-tuning the encoder for income at all moved the gap more (–0.257 to –0.090) than the fair loss did."),
    ("fig", "results/q3/fig1_shap_gap_women_men.png", 0.76, "Figure 3. SHAP gaps between women and men among high earners, before and after each fix."),
    ("t", {"caption": "Table 6. Mean SHAP value of relationship by category, among high earners. Husband and Wife are held by one sex only.", "w": [.2, .13, .13, .12, .12, .15, .15],
           "rows": [["Category", "Share of high-earning women", "Share of high-earning men", "AE + LR (before)", "AE + EG (after)", "Network, no fair loss", "Network, fair loss"],
                    ["Husband", "0.0%", "88.6%", "+1.18", "+0.80", "+0.94", "+0.77"],
                    ["Wife", "63.6%", "0.0%", "+0.75", "+1.36", "+1.31", "+1.71"],
                    ["Unmarried", "8.5%", "1.4%", "–0.28", "+0.35", "–0.56", "–0.51"]]}),
    ("t", {"caption": "Table 7. EO gap after scrambling, without retraining. Mean over 3 seeds and 5 shuffles. No intervals were computed.", "w": [.22, .3, .12, .12, .12, .12],
           "rows": [["Comparison", "Feature scrambled", "AE + LR (before)", "AE + EG (after)", "Network, no fair loss", "Network, fair loss"],
                    ["Women vs men", "(nothing scrambled)", "–0.257", "0.029", "–0.090", "0.022"],
                    ["Women vs men", "relationship", "–0.059", "–0.015", "–0.081", "–0.081"],
                    ["Women vs men", "relationship + marital-status", "–0.010", "0.028", "–0.010", "–0.010"],
                    ["Black men vs White men", "(nothing scrambled)", "–0.127", "–0.142", "–0.101", "–0.107"],
                    ["Black men vs White men", "education-num", "–0.063", "–0.060", "–0.055", "–0.051"]]}),
    ("fig", "results/q3/fig2_scramble_test.png", 0.62, "Figure 4. Women vs men equal opportunity gap after scrambling each feature. Only the features that move the gap most are shown."),
    ("ul", [
        "<b>Which features.</b> Relationship is the largest SHAP gap in AE + LR (–0.661) and marital-status in the network (–0.246). Scrambling both nearly removes the women's gap (–0.257 to –0.010).",
        "<b>Relationship is a proxy for sex.</b> 89% of high-earning men are Husbands and 64% of high-earning women are Wives, and these category mixes reproduce the full SHAP gap (–0.663 against –0.661). Women and men in the same shared category get almost the same value.",
        "<b>The bias is offset, not removed.</b> After the fair loss, Wife scores further above Husband (+1.71 against +0.77), and scrambling relationship brings the gap back to –0.081. The marital-status gap stays (–0.262). Unmarried stays at –0.51, so parity rests largely on wives.",
        "<b>Black men have an additional driver.</b> Education-num has their largest SHAP gap (–0.224), and EG leaves it unchanged (–0.226). A fix aimed at sex would not be expected to change it.",
        "<b>Do the techniques agree?</b> Signed gradient contributions agree with SHAP (correlation 0.987 to 0.999), though they are not fully independent evidence. Gradient size does not: it ranks native-country highest, while scrambling native-country leaves the gap almost unchanged (–0.257 to –0.256).",
    ]),
    ("t", {"caption": "Table 8. Answers to Question 3.", "w": [.36, .64],
           "rows": [["Question", "Answer"],
                    ["Do some features contribute disproportionately for particular groups?", "<b>Yes.</b> Relationship and marital-status for women; education-num as well as these two for Black men."],
                    ["Do those features make the model unfair?", "<b>Yes, for these trained models.</b> Scrambling the first two nearly removes the women's gap."],
                    ["Why do they do this?", "Relationship works as a proxy for sex, because Husband and Wife are sex-specific categories. Marital-status is a weaker signal."],
                    ["Did the network become fair after the fair loss?", "<b>Partly.</b> Gaps are near zero and White women no longer meet the rule, but the network was under the cutoff for women overall, the reductions are not statistically established, and demographic parity is still above 0.10."],
                    ["Did the model's reliance on these features end?", "<b>No.</b> The fixes push the proxy in women's favour, offsetting the bias rather than removing it."]]}),

    ("h1", "5. Limitations"),
    ("p", "The data is from 1994, and sex and race are recorded in broad categories. Small groups give wide intervals, so \"no gap found\" does not mean \"fair\". Several verdicts sit close to the 0.10 cutoff. Intervals reflect test-set sampling, not differences between training runs, and 3 seeds estimate seed variation only roughly. SHAP and scrambling show what these trained models rely on, not what causes income. The scramble test has no intervals and did not record accuracy, so it supports SHAP rather than proving it. LIME, attention and prototypes were not tried, and neither model is a foundation model."),
    ("h1", "6. Conclusion"),
    ("p", "Fairness problems were found, and they depend on the model and the metric. Accuracy alone would have pointed in the wrong direction, and removing sensitive columns did little. EG closed the equal opportunity gap for women at a small accuracy cost, and the fair loss brought the network's gap close to zero, but neither is fairness in every sense. Interpretability explains why: relationship stands in for sex, and the fair models shift it in women's favour rather than stop using it. Parity on a chosen measure can be reached without removing the underlying dependence, so an audit should look inside the model as well as at its gaps."),
    ("refs", "<b>References.</b> Agarwal et al. (2018), A Reductions Approach to Fair Classification, ICML. Becker and Kohavi (1996), Adult, UCI Machine Learning Repository. Hardt, Price and Srebro (2016), Equality of Opportunity in Supervised Learning, NeurIPS. Kamiran and Calders (2012), Data Preprocessing Techniques for Classification without Discrimination, Knowledge and Information Systems. Lundberg and Lee (2017), A Unified Approach to Interpreting Model Predictions, NeurIPS. Simonyan, Vedaldi and Zisserman (2014), Deep Inside Convolutional Networks, ICLR Workshop."),
    ("refs", "<b>Code.</b> https://github.com/payalv623/fairness-audit-and-interpretability"),
]


# ---------------------------------------------------------------- PDF
def font_paths():
    lib = Path("/usr/share/fonts/truetype/liberation")
    if (lib / "LiberationSerif-Regular.ttf").exists():
        return lib / "LiberationSerif-Regular.ttf", lib / "LiberationSerif-Bold.ttf"
    d = Path(matplotlib.get_data_path()) / "fonts/ttf"
    return d / "DejaVuSerif.ttf", d / "DejaVuSerif-Bold.ttf"


def build_pdf():
    reg, bold = font_paths()
    pdfmetrics.registerFont(TTFont("Body", str(reg)))
    pdfmetrics.registerFont(TTFont("Body-Bold", str(bold)))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body", boldItalic="Body-Bold")
    base = ParagraphStyle("base", fontName="Body", fontSize=10, leading=12.3, spaceAfter=4)
    h1 = ParagraphStyle("h1", parent=base, fontName="Body-Bold", fontSize=11.5, leading=14, spaceBefore=5, spaceAfter=3)
    cell = ParagraphStyle("cell", parent=base, fontSize=7.9, leading=9.3, spaceAfter=0)
    cap = ParagraphStyle("cap", parent=base, fontSize=8.1, leading=9.6, spaceBefore=2, spaceAfter=5)
    refs = ParagraphStyle("refs", parent=base, fontSize=8.3, leading=9.9, spaceAfter=2)
    centered = ParagraphStyle("c", parent=base, alignment=TA_CENTER, spaceAfter=1)

    def page(canvas, doc):
        canvas.saveState()
        canvas.setFont("Body", 8.5)
        canvas.drawString(MARGIN, A4[1] - 1.25 * cm, HEADER)
        canvas.drawRightString(A4[0] - MARGIN, A4[1] - 1.25 * cm, AUTHOR)
        canvas.setLineWidth(0.4)
        canvas.line(MARGIN, A4[1] - 1.35 * cm, A4[0] - MARGIN, A4[1] - 1.35 * cm)
        canvas.drawCentredString(A4[0] / 2, 0.95 * cm, str(doc.page))
        canvas.restoreState()

    st = [Paragraph("<b>RAI-FM (DSE428/DSE628)</b>", centered), Paragraph("<b>Assignment-4</b>", centered),
          Paragraph(f"<font size=14><b>{TITLE}</b></font>", ParagraphStyle("t", parent=centered, leading=19, spaceBefore=3, spaceAfter=3)),
          Paragraph("Name: Payal Verma &nbsp;&nbsp;&nbsp; Roll No.: 2611004", centered), Spacer(1, 3)]
    pending_heading = None
    for it in C:
        kind = it[0]
        if kind == "h1":
            pending_heading = Paragraph(it[1], h1)
            continue
        if kind == "p":
            f = [Paragraph(esc_pdf(it[1]), base)]
        elif kind == "refs":
            f = [Paragraph(esc_pdf(it[1]), refs)]
        elif kind == "box":
            b = Table([[Paragraph(esc_pdf(it[1]), ParagraphStyle("bx", parent=base, fontSize=9, leading=10.8, spaceAfter=0))]], colWidths=[TEXT_W])
            b.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.5, colors.black), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
            f = [b, Spacer(1, 4)]
        elif kind == "ul":
            f = [ListFlowable([ListItem(Paragraph(esc_pdf(x), base), leftIndent=10) for x in it[1]], bulletType="bullet", start="•", leftIndent=10, bulletFontSize=8)]
        elif kind == "t":
            t = it[1]
            total = sum(t["w"])
            widths = [TEXT_W * w / max(total, 1) for w in t["w"]]
            rows = [[Paragraph(esc_pdf(("<b>%s</b>" % c) if i == 0 else c), cell) for c in r] for i, r in enumerate(t["rows"])]
            tb = Table(rows, colWidths=widths, hAlign="CENTER")
            tb.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black), ("LINEBELOW", (0, 0), (-1, 0), 0.4, colors.black),
                                    ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                    ("TOPPADDING", (0, 0), (-1, -1), 1.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
                                    ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]))
            f = [KeepTogether([tb, Paragraph(esc_pdf(t["caption"]), cap)])]
        elif kind == "fig":
            w, h = PILImage.open(it[1]).size
            iw = TEXT_W * it[2]
            f = [KeepTogether([Image(it[1], width=iw, height=iw * h / w), Paragraph(esc_pdf(it[3]), cap)])]
        if pending_heading is not None:
            st.append(KeepTogether([pending_heading] + f[:1]))
            st.extend(f[1:])
            pending_heading = None
        else:
            st.extend(f)
    doc = SimpleDocTemplate(f"{NAME}.pdf", pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=1.65 * cm, bottomMargin=1.45 * cm,
                            title="RAI-FM Assignment-4: " + TITLE, author="Payal Verma")
    doc.build(st, onFirstPage=page, onLaterPages=page)


def esc_pdf(s):
    return s.replace("&", "&amp;").replace(">$", "&gt;$")


# ---------------------------------------------------------------- LaTeX
def tex(s):
    url = "https://github.com/payalv623/fairness-audit-and-interpretability"
    s = s.replace(url, "@@URL@@")
    s = s.replace("\\", "\\textbackslash{}")
    for a, b in (("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"), ("_", "\\_")):
        s = s.replace(a, b)
    s = s.replace(">\\$", "$>$\\$").replace("≤\\$", "$\\leq$\\$")
    s = re.sub(r"<b>(.*?)</b>", r"\\textbf{\1}", s)
    s = re.sub(r"<sub>(.*?)</sub>", r"\\textsubscript{\1}", s)
    s = re.sub(r"<super>(.*?)</super>", r"\\textsuperscript{\1}", s)
    s = re.sub(r"\"(.*?)\"", r"``\1''", s)
    for a, b in (("–", "\\textminus{}"), ("λ", "$\\lambda$"), ("×", "$\\times$"), ("±", "$\\pm$"), ("²", "$^2$"), ("≤", "$\\leq$")):
        s = s.replace(a, b)
    return s.replace("@@URL@@", "\\url{%s}" % url)


def build_tex():
    out = [r"""\documentclass[10pt,a4paper]{article}
%% Generated by build_short_report.py. Compile with pdflatex or on Overleaf.
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{textcomp}
\usepackage[a4paper,margin=1.9cm,top=2.0cm,bottom=1.7cm]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{float}
\usepackage{enumitem}
\usepackage{microtype}
\usepackage[font=footnotesize]{caption}
\usepackage{fancyhdr}
\usepackage{url}
\setlist[itemize]{leftmargin=1.2em,itemsep=1pt,topsep=2pt}
\setlength{\parindent}{0pt}
\setlength{\parskip}{3pt}
\pagestyle{fancy}\fancyhf{}
\lhead{\small %s}\rhead{\small %s}\cfoot{\small\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\setlength{\headheight}{13pt}
\begin{document}
\thispagestyle{fancy}
\begin{center}
{\bfseries RAI-FM (DSE428/DSE628)}\\ {\bfseries Assignment-4}\\[4pt]
{\Large\bfseries %s}\\[4pt]
Name: Payal Verma \qquad Roll No.: 2611004
\end{center}
""" % (HEADER, AUTHOR, TITLE)]
    for it in C:
        kind = it[0]
        if kind == "h1":
            out.append("\\section*{%s}\n" % tex(it[1]))
        elif kind == "p":
            out.append(tex(it[1]) + "\n")
        elif kind == "refs":
            out.append("{\\small " + tex(it[1]) + "}\n")
        elif kind == "box":
            out.append("\\noindent\\fbox{\\parbox{\\dimexpr\\linewidth-2\\fboxsep-2\\fboxrule}{\\small %s}}\n" % tex(it[1]))
        elif kind == "ul":
            out.append("\\begin{itemize}\n" + "".join("  \\item %s\n" % tex(x) for x in it[1]) + "\\end{itemize}\n")
        elif kind == "t":
            t = it[1]
            total = max(sum(t["w"]), 1)
            n = len(t["w"])
            spec = "".join(">{\\raggedright\\arraybackslash}p{%.3f\\linewidth}" % (w / total * (1 - 0.012 * n)) for w in t["w"])
            rows = [" & ".join(("\\textbf{%s}" % tex(c)) if i == 0 else tex(c) for c in r) + " \\\\" for i, r in enumerate(t["rows"])]
            out.append("\\begin{table}[H]\n\\centering\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{@{}%s@{}}\n\\toprule\n%s\n\\midrule\n%s\n\\bottomrule\n\\end{tabular}\n\\caption*{%s}\n\\end{table}\n"
                       % (spec, rows[0], "\n".join(rows[1:]), tex(t["caption"])))
        elif kind == "fig":
            out.append("\\begin{figure}[H]\n\\centering\n\\includegraphics[width=%.2f\\linewidth]{%s}\n\\caption*{%s}\n\\end{figure}\n" % (it[2], it[1], tex(it[3])))
    out.append("\\end{document}\n")
    Path(f"{NAME}.tex").write_text("\n".join(out))


if __name__ == "__main__":
    build_pdf()
    build_tex()
