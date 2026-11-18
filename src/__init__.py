"""Statistical AI Text Detection with Evidence-Guided Prompts (SCSE4041 Prompt Engineering).

A plain-Python statistical algorithm computes reproducible evidence from an English text, a transparent
statistical baseline turns that evidence into a score and a cautious three-way label, and a pretrained
LLM interprets the same evidence through five prompt versions that change one thing at a time.

The output supports a human reviewer. It never establishes authorship, and the baseline score is not a
probability.
"""

__version__ = "1.0.0"
