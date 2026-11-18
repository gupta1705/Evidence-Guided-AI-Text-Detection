# Prompt comparison (V0-V4) - test sample

| version | adds | n | n_decided | decided_accuracy | decided_f1 | coverage | human_fpr | ai_recall | json_validity | counts_compliance | grounded_evidence_rate | strict_grounded_rate | invented_number_rate | short_text_compliance | agreement_with_baseline | label_validity | lenient_coverage | lenient_decided_accuracy | lenient_human_fpr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V0 | instruction | 20 | 0 | - | - | 0.000 [0.000, 0.161] | 0.000 [0.000, 0.278] | 0.000 [0.000, 0.278] | 0.000 [0.000, 0.161] | 0.000 [0.000, 0.161] | - | - | 0.000 [0.000, 0.168] | 1.000 [0.510, 1.000] | 0.250 [0.050, 0.450] | 0.950 [0.850, 1.000] | 0.400 [0.200, 0.600] | 0.750 [0.429, 1.000] | 0.000 [0.000, 0.278] |
| V1 | role | 20 | 0 | - | - | 0.000 [0.000, 0.161] | 0.000 [0.000, 0.278] | 0.000 [0.000, 0.278] | 0.000 [0.000, 0.161] | 0.000 [0.000, 0.161] | - | - | 0.000 [0.000, 0.161] | 1.000 [0.510, 1.000] | 0.250 [0.050, 0.450] | 1.000 [0.839, 1.000] | 0.650 [0.400, 0.850] | 0.769 [0.538, 1.000] | 0.000 [0.000, 0.278] |
| V2 | grounding | 20 | 0 | - | - | 0.000 [0.000, 0.161] | 0.000 [0.000, 0.278] | 0.000 [0.000, 0.278] | 0.000 [0.000, 0.161] | 0.000 [0.000, 0.161] | - | - | 0.000 [0.000, 0.161] | 1.000 [0.510, 1.000] | 0.250 [0.050, 0.450] | 1.000 [0.839, 1.000] | 0.350 [0.150, 0.550] | 1.000 [0.646, 1.000] | 0.000 [0.000, 0.278] |
| V3 | few_shot | 20 | 7 | 1.000 [0.646, 1.000] | 1.000 [0.342, 1.000] | 0.350 [0.150, 0.550] | 0.000 [0.000, 0.278] | 0.200 [0.000, 0.500] | 1.000 [0.839, 1.000] | 0.900 [0.750, 1.000] | 1.000 [0.839, 1.000] | 1.000 [0.839, 1.000] | 0.000 [0.000, 0.161] | 1.000 [0.510, 1.000] | 0.600 [0.399, 0.800] | 1.000 [0.839, 1.000] | 0.350 [0.150, 0.550] | 1.000 [0.646, 1.000] | 0.000 [0.000, 0.278] |
| V4 | structured_output | 20 | 7 | 1.000 [0.646, 1.000] | 1.000 [0.342, 1.000] | 0.350 [0.150, 0.550] | 0.000 [0.000, 0.278] | 0.200 [0.000, 0.500] | 1.000 [0.839, 1.000] | 1.000 [0.839, 1.000] | 1.000 [0.839, 1.000] | 1.000 [0.839, 1.000] | 0.000 [0.000, 0.161] | 1.000 [0.510, 1.000] | 0.600 [0.399, 0.800] | 1.000 [0.839, 1.000] | 0.350 [0.150, 0.550] | 1.000 [0.646, 1.000] | 0.000 [0.000, 0.278] |
| baseline | statistical rule | 20 | 15 | 0.733 [0.500, 0.938] | 0.600 [0.000, 0.909] | 0.750 [0.550, 0.950] | 0.000 [0.000, 0.278] | 0.300 [0.000, 0.625] | - | - | - | - | - | 1.000 [0.510, 1.000] | - | 1.000 [0.839, 1.000] | 0.750 [0.550, 0.950] | 0.733 [0.500, 0.938] | 0.000 [0.000, 0.278] |

- Provider: groq; requested model: openai/gpt-oss-120b; model(s) returned by the API: openai/gpt-oss-120b. Results depend on this model/version.
- Settings: temperature 0.0, max_tokens 800; V4 requests JSON mode.
- Accuracy and F1 (AI positive) are only meaningful on decided examples; coverage = share not 'Uncertain'.
- A response that fails the strict JSON schema (json_validity) counts as 'Uncertain'; the label it named is kept as parsed_label in llm_runs.jsonl.
- Human FPR = human texts labelled 'Likely AI-generated' / all human texts in the sample.
- 95% CIs: percentile bootstrap over questions (1000 resamples); an estimate of exactly 0 or 1 gets the Wilson score interval for its contributing texts instead (column <metric>_n in the CSV).
- Labels are review aids, not proof of authorship.
- counts_compliance = share of responses that are schema-valid and have 2-4 evidence items and 1-3 limitations (the counts only V4 asks for).
- grounded_evidence_rate = per response, share of evidence items that name a supplied feature and whose numbers all match some supplied number (the project definition; an item without numbers passes).
- strict_grounded_rate = share of evidence items that name a supplied feature, quote at least one number, and quote only numbers from the line of a feature they name (or the 100/300-word constants).
- invented_number_rate = share of responses with a number in evidence or limitations that matches no supplied number. short_text_compliance = share of texts under 100 words labelled 'Uncertain'.
- Strict view (decided_accuracy, coverage, human_fpr, ...): a response that breaks the JSON schema counts as 'Uncertain'. Label-only view (label_validity, lenient_*): the label the response named counts whenever it is one of the three allowed labels, even if evidence/limitations break the schema. The gap between the two views is the cost of format errors.
- Sample: 20 test texts (10 human / 10 AI; 4 under 100 words); ids in prompt_sample_ids.csv.
- The 'baseline' row applies the frozen statistical rule to the same sample.
