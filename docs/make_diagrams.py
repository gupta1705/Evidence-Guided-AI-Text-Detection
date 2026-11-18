"""Draw the four Milestone 2 diagrams into docs/ (python docs/make_diagrams.py)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).parent
FILL = {"in": "#dbeafe", "py": "#dcfce7", "llm": "#fef3c7", "out": "#fce7f3", "chk": "#ede9fe"}


def draw(name, title, boxes, arrows, size=(11, 5.2), tree=False):
    """boxes: id -> (x, y, w, h, text, kind); arrows: (from, to, label). tree: always connect top/bottom."""
    fig, ax = plt.subplots(figsize=size)
    ax.set_xlim(0, 100), ax.set_ylim(0, 50), ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left")
    for x, y, w, h, text, kind in boxes.values():
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=FILL[kind], ec="#334155", lw=1.1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.6, wrap=True)
    for a, b, label in arrows:
        xa, ya, wa, ha_, *_ = boxes[a]
        xb, yb, wb, hb, *_ = boxes[b]
        start, end = (xa + wa / 2, ya + ha_ / 2), (xb + wb / 2, yb + hb / 2)
        if not tree and xa + wa < xb:  # target to the right: right edge -> left edge
            start, end = (xa + wa + 0.6, start[1]), (xb - 0.6, end[1])
        elif not tree and xb + wb < xa:  # target to the left
            start, end = (xa - 0.6, start[1]), (xb + wb + 0.6, end[1])
        elif start[1] > end[1]:
            start, end = (start[0], ya - 0.6), (end[0], yb + hb + 0.6)
        else:
            start, end = (start[0], ya + ha_ + 0.6), (end[0], yb - 0.6)
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, color="#334155", lw=1.1))
        if label:
            ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 1.2, label, ha="center", fontsize=7.5,
                    color="#475569", style="italic")
    legend = [("in", "input"), ("py", "Python (deterministic)"), ("llm", "LLM call"), ("chk", "check"),
              ("out", "output")]
    for i, (kind, text) in enumerate(legend):
        ax.add_patch(FancyBboxPatch((2 + i * 19, 1), 2.5, 2, boxstyle="round,pad=0.2", fc=FILL[kind], ec="#334155"))
        ax.text(5.5 + i * 19, 2, text, va="center", fontsize=7.5)
    fig.savefig(OUT / f"{name}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


# 1. Prompt workflow (one LLM call, final prompt P1 = V4)
draw("diagram_prompt_workflow", "Prompt workflow (final prompt P1 = V4, one LLM call per text)", {
    "t": (2, 30, 14, 10, "English text\n(public or\nanonymised)", "in"),
    "f": (21, 30, 15, 10, "Compute 4 features\n+ word/sentence\ncounts (first 300 w.)", "py"),
    "r": (21, 12, 15, 10, "Reference medians\n& IQRs (frozen,\nmodels/baseline.json)", "py"),
    "p": (41, 30, 16, 10, "Build prompt P1:\nsystem role + 3 few-shot\n+ rules + JSON schema\n+ data block", "py"),
    "l": (62, 30, 14, 10, "gpt-oss-120b\n(Groq), temp 0,\nJSON mode", "llm"),
    "c": (62, 12, 14, 10, "Parse strict JSON\n+ grounding check\n(numbers must exist)", "chk"),
    "o": (82, 12, 15, 10, "Label + evidence\n+ limitations\n+ flags", "out"),
}, [("t", "f", ""), ("f", "p", "values"), ("r", "p", "ranges"), ("p", "l", "messages"), ("l", "c", "JSON"),
    ("c", "o", "")])

# 2. Task decomposition
draw("diagram_task_decomposition", "Task decomposition", {
    "root": (35, 38, 30, 8, "Assess likely AI involvement in a text\n(support a human reviewer)", "out"),
    "a": (2, 20, 20, 10, "1. Prepare text\nnormalise, first 300 words,\ncount words", "py"),
    "b": (26, 20, 20, 10, "2. Measure evidence\n4 features vs human/AI\nreference ranges", "py"),
    "c": (50, 20, 22, 10, "3. Interpret evidence (LLM)\nlabel only from supplied\nnumbers, Uncertain if mixed", "llm"),
    "d": (76, 20, 22, 10, "4. Validate answer\nJSON schema, grounding,\ninvented numbers", "chk"),
    "e": (26, 6, 20, 9, "Algorithm v1 / v2\nindependent statistical\nlabel (no LLM)", "py"),
    "f": (76, 6, 22, 9, "5. Show to reviewer\nlabel, evidence, limits,\nmanual review advice", "out"),
}, [("root", "a", ""), ("root", "b", ""), ("root", "c", ""), ("root", "d", ""), ("b", "e", ""), ("d", "f", "")], tree=True)

# 3. Input-output flow
draw("diagram_input_output", "Input-output flow", {
    "i": (2, 26, 18, 14, "INPUT\nEnglish text pasted\nin the app\n(0-any length)", "in"),
    "s": (25, 32, 20, 9, "Offline: features table\n+ v1 label + v2 label\n+ P(AI)", "py"),
    "q": (25, 16, 20, 9, "On 'Ask the LLM':\nprompt P1 (V4) with\ntext + features", "llm"),
    "o1": (52, 32, 22, 9, "OUTPUT 1 (instant)\nstatistics + algorithm\nlabels", "out"),
    "o2": (52, 16, 22, 9, "OUTPUT 2 (after the LLM call)\n{label, evidence[2-4],\nlimitations[1-3]}", "out"),
    "x": (79, 16, 19, 9, "Flags: JSON valid,\ngrounded %, invented\nnumbers yes/no", "chk"),
    "e": (25, 4, 20, 8, "Empty input -> 'Paste a\ntext first.'; <100 words\n-> Uncertain", "chk"),
}, [("i", "s", ""), ("i", "q", ""), ("s", "o1", ""), ("q", "o2", ""), ("o2", "x", ""), ("i", "e", "")])

# 4. System architecture
draw("diagram_architecture", "System architecture (custom Gradio app, Python)", {
    "ui": (2, 33, 15, 10, "Gradio web app\nsrc/app.py", "in"),
    "tu": (22, 33, 15, 10, "text_utils.py\nnormalise,\ntokenise", "py"),
    "fe": (42, 33, 15, 10, "features.py\n4 statistics", "py"),
    "pr": (62, 33, 15, 10, "prompts.py\nbuilds P1 (V4)", "py"),
    "ll": (82, 33, 15, 10, "llm.py\nGroq client", "llm"),
    "ev": (2, 16, 15, 10, "evaluate.py\nbatch tests,\nresults/*.md", "out"),
    "m": (22, 16, 15, 10, "models/*.json\nfrozen stats,\nweights, examples", "py"),
    "bl": (42, 16, 15, 10, "baseline.py (v1)\nstylometry.py (v2)\n-> label to app", "py"),
    "pa": (62, 16, 15, 10, "parsing.py +\ngrounding.py\n-> result to app", "chk"),
    "api": (82, 16, 15, 10, "Groq API\ngpt-oss-120b", "llm"),
}, [("ui", "tu", "text"), ("tu", "fe", ""), ("fe", "pr", "values"), ("pr", "ll", "messages"), ("ll", "api", ""),
    ("api", "pa", "JSON"), ("fe", "bl", ""), ("m", "bl", "")], size=(12, 4.8))

print("wrote", *sorted(p.name for p in OUT.glob("diagram_*.png")))
