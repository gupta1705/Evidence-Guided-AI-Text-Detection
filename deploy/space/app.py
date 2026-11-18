"""Hosted entry point (Render or Hugging Face): the same app as `python -m src.app`, served on 0.0.0.0.

The Groq key comes from the host's secret GROQ_API_KEY (never from a file in the repository). Render sets
PORT; Hugging Face uses 7860.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root on Render

from src.app import build_app  # noqa: E402

build_app().launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
