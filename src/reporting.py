"""Small writers shared by the CLIs: CSV + Markdown tables and JSON Lines files (all UTF-8)."""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd


def ensure_utf8_stdout() -> None:
    """Print UTF-8 on Windows consoles and pipes (default code pages mangle dashes in the prompts)."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None and (stream.encoding or "").lower() not in ("utf-8", "utf8"):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def format_cell(value: object, digits: int = 3) -> str:
    """Format one table cell: floats to ``digits`` decimals, NaN/None as ``"-"``."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if math.isnan(value):
            return "-"
        return f"{value:.{digits}f}"
    return str(value).replace("|", "\\|")


def markdown_table(df: pd.DataFrame, digits: int = 3, columns: Sequence[str] | None = None) -> str:
    """Render ``df`` as a GitHub-flavoured Markdown table (no external dependency)."""
    cols = list(columns or df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(format_cell(row[c], digits) for c in cols) + " |")
    return "\n".join(lines)


def write_table(
    df: pd.DataFrame,
    stem: Path,
    title: str,
    notes: Iterable[str] = (),
    md_df: pd.DataFrame | None = None,
    digits: int = 3,
) -> tuple[Path, Path]:
    """Write ``df`` to ``<stem>.csv`` and a titled Markdown version to ``<stem>.md``.

    Args:
        df: Table written to CSV (full precision).
        stem: Output path without extension.
        title: Markdown heading.
        notes: Bullet points printed under the table.
        md_df: Optional alternative (e.g. pre-formatted) table for the Markdown file.
        digits: Decimals for floats in the Markdown file.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    csv_path = stem.with_suffix(".csv")
    md_path = stem.with_suffix(".md")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    parts = [f"# {title}", "", markdown_table(md_df if md_df is not None else df, digits=digits)]
    notes = list(notes)
    if notes:
        parts += [""] + [f"- {n}" for n in notes]
    md_path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return csv_path, md_path


def clean_json(value: object) -> object:
    """Recursively replace NaN/inf floats with ``None`` and numpy scalars with Python values."""
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, obj: object) -> Path:
    """Write ``obj`` as pretty-printed strict JSON (NaN becomes null)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(clean_json(obj), fh, indent=2, ensure_ascii=False, default=_json_default)
    return path


def append_jsonl(path: Path, records: Iterable[dict]) -> int:
    """Append ``records`` to a JSON Lines file; returns the number of lines written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(clean_json(record), ensure_ascii=False, default=_json_default) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSON Lines file, skipping blank or truncated (unparseable) lines."""
    path = Path(path)
    if not path.is_file():
        return []
    out: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _json_default(value: object) -> object:
    """Make numpy scalars and paths JSON serialisable."""
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")
