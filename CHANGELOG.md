# Changelog

Dated changes since Milestone 1 (project form submitted 10 Oct 2026).

## 18 Nov 2026 - Milestone 2 package
- `test-data/run_tests.py`: 14-input test table (7 normal, 4 edge, 3 invalid) run with the final prompt
  (P1 = V4, gpt-oss-120b on Groq) and algorithm v2; inputs and every actual output saved in `test-data/`.
  Result: 10 of 14 pass (the failures are kept and discussed in the Milestone 2 PDF).
- `prompts/`: every prompt version saved as text (v1 = V0 ... final = V4) with template variables.
- `docs/`: workflow, task-decomposition, input-output and architecture diagrams; screenshots of real runs;
  Milestone 2 PDF and research abstract.
- Repository published on GitHub; the small frozen models (`models/*.json`) are now tracked so the app runs
  from a fresh clone.

## 17 Nov 2026 - Algorithm v2 (`src/stylometry.py`)
- 15 more plain-Python style features, a word log-odds lexicon (cross-fitted) and a logistic regression
  replace the equal-weight distance rule of v1. Test split: labels 100 % of texts (v1: 52.7 %) at 95.2 %
  accuracy (v1: 81.6 %); 98.2 % of labels unchanged under typos.
- The app shows the v2 label, P(AI) and the five strongest terms next to the v1 baseline.

## 13 Oct 2026 - Prompt comparison V0-V4
- Five cumulative prompt versions evaluated on 20 test texts with `openai/gpt-oss-120b` (Groq, temperature 0),
  plus typo / style / prompt-injection robustness for V0 and V4 (`results/prompt_comparison.md`,
  `results/robustness_prompts.md`). The model changed from the ChatGPT API planned in Milestone 1 to the
  free Groq API (same family of open-weight GPT models, no paid key needed).

## 12 Oct 2026 - Data, features and statistical baseline (v1)
- HC3 English cleaned and normalised, question-disjoint splits (3,000 / 600 / 600).
- Four features (sentence-length variation, MATTR, repeated trigrams, word entropy) and the frozen
  distance-to-median baseline with threshold and uncertainty margin (`models/baseline.json`).
- Gradio app, offline test suite.
