# RAI-FM (DSE428/DSE628), Assignment-4: Video script

Payal Verma, 2611004. Length: about 3 minutes.

Each **Show this file** line is the cue to open that file on screen.

## Intro (0:00–0:20)

**Show this file:** `Assignment4_Report_5pages.pdf`

> Hi, this is my Responsible AI assignment on fairness and interpretability. I used the Adult Income dataset from the 1994 US census: about 48,800 people, and the task is to predict who earns over 50,000 dollars. I checked fairness by sex and race.

## Question 1 (0:20–1:10)

> Question 1 asked me to audit at least two algorithms for unfairness. I used a deep autoencoder with logistic regression on its code, and an RBF SVM. I measured accuracy, reconstruction error, demographic parity and equal opportunity.

**Show this file:** `results/fig2_fairness_gaps.png`

> This file shows the results. The autoencoder model reached 0.84 accuracy, but it found only 34% of high-earning women, against 60% of high-earning men. That's an equal opportunity gap of minus 0.26. The SVM did better: 0.86 accuracy and a gap of about minus 0.10.

**Show this file:** `results/fig1_reconstruction.png`

> This file shows that reconstruction error was 3.8 times higher for Asian-Pacific-Islander people than for White people. My conclusion: the autoencoder model is clearly unfair to women, and accuracy hid it, because women actually had higher accuracy than men.

## Question 2 (1:10–1:50)

> Question 2 asked me to make one algorithm fair and re-evaluate it. I took the autoencoder model, compared several fixes on validation data, and selected Exponentiated Gradient, which pushes true positive rates for women and men together.

**Show this file:** `results/q2/fig1_before_after_gaps.png`

> This file shows the results. The women's gap went from minus 0.26 to plus 0.03, and White women's from minus 0.27 to plus 0.02. Accuracy only fell from 0.84 to 0.83. My conclusion: women and White women clearly benefited, but this isn't fairness in every sense. The demographic parity gap for women is still minus 0.12, Black men weren't helped, and the reconstruction gap remains.

## Question 3 (1:50–2:45)

> Question 3 asked which features drive the unfairness, and whether the network became fair after a fair optimization loss. I used SHAP and gradients, backed by a scramble test, and trained a network with a penalty on the gap between women and men.

**Show this file:** `results/q3/fig1_shap_gap_women_men.png`

> This file shows the SHAP results, which point to relationship and marital status. Sex isn't an input, but relationship works as a proxy for it, because Husband and Wife each belong to one sex.

**Show this file:** `results/q3/fig2_scramble_test.png`

> This file shows the scramble test. Scrambling both features shrank the women's gap from minus 0.26 to minus 0.01.

**Show this file:** `results/q3/fairness.csv`

> This file shows the fair loss results. The network's gap moved from minus 0.09 to plus 0.02, and accuracy stayed at 0.85. My conclusion: only partly fair. That network was already under my 0.10 cutoff for women, the improvement isn't statistically established, and it still relies on the proxy, now in women's favour.

## Closing (2:45–3:00)

**Show this file:** `Assignment4_Report_5pages.pdf`

> Overall, unfairness depends on the model and the metric, and accuracy alone can hide it. Fixing a number doesn't remove the underlying dependence, so an audit should look inside the model too. Thank you.
