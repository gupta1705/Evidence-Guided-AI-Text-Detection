---
title: Evidence-Guided AI Text Detection
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Statistics + evidence-guided LLM prompt for AI-text review
---

# Statistical AI Text Detection with Evidence-Guided Prompts

Paste an English text. **Compute statistics** shows four Python-computed features against human/AI reference
ranges and two statistical labels. **Ask the LLM** sends the text and its statistics to `openai/gpt-oss-120b`
(Groq) with our final prompt (V4) and shows the label, evidence, limitations and validity checks.
Use public or anonymised text only. The result supports a human reviewer; it never proves who wrote a text.

Team Golden Girls - SCSE4041 Prompt Engineering, Bennett University.
Code, prompts and tests: https://github.com/gupta1705/Evidence-Guided-AI-Text-Detection
