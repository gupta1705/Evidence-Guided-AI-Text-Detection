# Baseline - validation split

| rule | subset | n | n_human | n_ai | n_decided | coverage | decided_accuracy | decided_f1 | human_fpr | ai_recall |
|---|---|---|---|---|---|---|---|---|---|---|
| with 100-word rule | all | 600 | 300 | 300 | 328 | 0.547 | 0.854 | 0.879 | 0.033 | 0.583 |
| with 100-word rule | >=100 | 416 | 132 | 284 | 328 | 0.788 | 0.854 | 0.879 | 0.076 | 0.616 |
| with 100-word rule | <100 | 184 | 168 | 16 | 0 | 0.000 | - | - | 0.000 | 0.000 |
| with 100-word rule | 100-199 | 262 | 80 | 182 | 200 | 0.763 | 0.820 | 0.855 | 0.087 | 0.582 |
| with 100-word rule | >=200 | 154 | 52 | 102 | 128 | 0.831 | 0.906 | 0.920 | 0.058 | 0.676 |
| without word rule | all | 600 | 300 | 300 | 442 | 0.737 | 0.846 | 0.843 | 0.093 | 0.610 |
| without word rule | >=100 | 416 | 132 | 284 | 328 | 0.788 | 0.854 | 0.879 | 0.076 | 0.616 |
| without word rule | <100 | 184 | 168 | 16 | 114 | 0.620 | 0.825 | 0.444 | 0.107 | 0.500 |
| without word rule | 100-199 | 262 | 80 | 182 | 200 | 0.763 | 0.820 | 0.855 | 0.087 | 0.582 |
| without word rule | >=200 | 154 | 52 | 102 | 128 | 0.831 | 0.906 | 0.920 | 0.058 | 0.676 |

- Threshold t = 0.0651, margin delta = 0.1916, word rule: fewer than 100 words -> Uncertain.
- Reference medians/IQRs: reference texts with >= 100 words (the texts the rule can label).
- t and delta were fitted on validation texts with >= 100 words and >= 2 valid features (n = 416); the margin's coverage target (0.60) applies to those texts (reached: 0.788). Overall coverage, which counts the always-Uncertain short texts, is the 'all' row.
- Rows '>=100' cover the texts the rule can label; '<100', '100-199', '>=200' are word-count bands.
- Accuracy and F1 (AI positive) are computed on decided texts only; coverage = share decided.
- Human FPR = human texts labelled 'Likely AI-generated' / all human texts in the row's subset.
- The score is a distance difference, not a probability; labels support manual review only.
