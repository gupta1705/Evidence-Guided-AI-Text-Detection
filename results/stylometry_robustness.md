# Algorithm v2 - typo robustness (test split)

| variant | n | n_human | n_ai | n_decided | coverage | decided_accuracy | decided_f1 | human_fpr | ai_recall | label_unchanged |
|---|---|---|---|---|---|---|---|---|---|---|
| swap | 600.000 | 300.000 | 300.000 | 600.000 | 1.000 | 0.952 | 0.952 | 0.047 | 0.950 | 0.993 |
| drop | 600.000 | 300.000 | 300.000 | 600.000 | 1.000 | 0.952 | 0.952 | 0.047 | 0.950 | 0.997 |
| key | 600.000 | 300.000 | 300.000 | 600.000 | 1.000 | 0.950 | 0.949 | 0.040 | 0.940 | 0.988 |
| add | 600.000 | 300.000 | 300.000 | 600.000 | 1.000 | 0.957 | 0.957 | 0.040 | 0.953 | 0.995 |
| all 4 variants | - | - | - | - | - | - | - | - | - | 0.982 |

- Each variant edits 2 words (first letter kept) inside the first 300 words.
- label_unchanged = share of texts whose label is the same as on the clean text.
