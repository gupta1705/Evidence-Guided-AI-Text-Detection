# Statistical baseline - test split

| rule | subset | n | n_human | n_ai | n_decided | coverage | decided_accuracy | decided_f1 | human_fpr | ai_recall |
|---|---|---|---|---|---|---|---|---|---|---|
| with 100-word rule | all | 600 | 300 | 300 | 316 | 0.527 | 0.816 | 0.846 | 0.040 | 0.530 |
| with 100-word rule | >=100 | 402 | 127 | 275 | 316 | 0.786 | 0.816 | 0.846 | 0.094 | 0.578 |
| with 100-word rule | <100 | 198 | 173 | 25 | 0 | 0.000 | - | - | 0.000 | 0.000 |
| with 100-word rule | 100-199 | 240 | 75 | 165 | 186 | 0.775 | 0.769 | 0.802 | 0.120 | 0.527 |
| with 100-word rule | >=200 | 162 | 52 | 110 | 130 | 0.802 | 0.885 | 0.906 | 0.058 | 0.655 |
| without word rule | all | 600 | 300 | 300 | 437 | 0.728 | 0.819 | 0.811 | 0.090 | 0.567 |
| without word rule | >=100 | 402 | 127 | 275 | 316 | 0.786 | 0.816 | 0.846 | 0.094 | 0.578 |
| without word rule | <100 | 198 | 173 | 25 | 121 | 0.611 | 0.826 | 0.512 | 0.087 | 0.440 |
| without word rule | 100-199 | 240 | 75 | 165 | 186 | 0.775 | 0.769 | 0.802 | 0.120 | 0.527 |
| without word rule | >=200 | 162 | 52 | 110 | 130 | 0.802 | 0.885 | 0.906 | 0.058 | 0.655 |

- Threshold t = 0.0651, margin delta = 0.1916, word rule: fewer than 100 words -> Uncertain.
- Reference medians/IQRs: reference texts with >= 100 words (the texts the rule can label).
- t and delta were fitted on validation texts with >= 100 words and >= 2 valid features (n = 416); the margin's coverage target (0.60) applies to those texts (reached: 0.788). Overall coverage, which counts the always-Uncertain short texts, is the 'all' row.
- Rows '>=100' cover the texts the rule can label; '<100', '100-199', '>=200' are word-count bands.
- Accuracy and F1 (AI positive) are computed on decided texts only; coverage = share decided.
- Human FPR = human texts labelled 'Likely AI-generated' / all human texts in the row's subset.
- The score is a distance difference, not a probability; labels support manual review only.
- n = 600 test texts (question-disjoint from reference/validation).
