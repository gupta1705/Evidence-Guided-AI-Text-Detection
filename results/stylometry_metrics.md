# Algorithm v1 vs v2 - test split

| algorithm | subset | n | n_human | n_ai | n_decided | coverage | decided_accuracy | decided_f1 | human_fpr | ai_recall |
|---|---|---|---|---|---|---|---|---|---|---|
| v1 baseline (4 features, distance rule) | all | 600 | 300 | 300 | 316 | 0.527 | 0.816 | 0.846 | 0.040 | 0.530 |
| v1 baseline (4 features, distance rule) | <100 | 198 | 173 | 25 | 0 | 0.000 | - | - | 0.000 | 0.000 |
| v1 baseline (4 features, distance rule) | 100-199 | 240 | 75 | 165 | 186 | 0.775 | 0.769 | 0.802 | 0.120 | 0.527 |
| v1 baseline (4 features, distance rule) | >=200 | 162 | 52 | 110 | 130 | 0.802 | 0.885 | 0.906 | 0.058 | 0.655 |
| v2 stylometric logistic regression | all | 600 | 300 | 300 | 600 | 1.000 | 0.952 | 0.952 | 0.050 | 0.953 |
| v2 stylometric logistic regression | <100 | 198 | 173 | 25 | 198 | 1.000 | 0.939 | 0.778 | 0.046 | 0.840 |
| v2 stylometric logistic regression | 100-199 | 240 | 75 | 165 | 240 | 1.000 | 0.950 | 0.963 | 0.053 | 0.952 |
| v2 stylometric logistic regression | >=200 | 162 | 52 | 110 | 162 | 1.000 | 0.969 | 0.977 | 0.058 | 0.982 |

- v2: logistic regression on 20 standardised features (4 v1 + 15 style + word log-odds, lexicon of 9005 words), fitted on the reference split.
- Threshold 0.0601 and margin 0.0000 on the logit, fitted on validation (target decided accuracy 0.95, min coverage 0.80); word rule: none (all texts scored).
- Largest standardised weights (+ = towards AI): word_logodds +3.56, repeated_trigram_rate +2.21, lower_i_rate -1.27, word_entropy +0.99, sentence_length_cv -0.87, lower_start_rate -0.56.
- v1 labels texts under 100 words 'Uncertain'; v2 scores them, so compare the band rows too.
- Accuracy and F1 (AI positive) on decided texts; coverage = share decided; human FPR over all human texts.
