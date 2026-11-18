"""Write the Milestone 2 form answers and the reel team's brief as .docx files.

python docs/build_answers_and_brief.py [--prototype URL]  ->  ../M2_Form_Answers.docx, ../Reel_Brief.docx
"""

import argparse
import json
from pathlib import Path

import docx
from docx.shared import Pt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent  # the "Prompt Engineering" folder
REPO = "https://github.com/gupta1705/Evidence-Guided-AI-Text-Detection"
PDF = f"{REPO}/blob/main/docs/M2_PE-B25-16_StatisticalAITextDetection.pdf"


def answers(prototype):
    return [
        ("1. Team ID", "PE-B25-16"),
        ("2. Team Name as Given in Milestone 1", "Golden Girls"),
        ("3. Project Title/Name", "Statistical AI Text Detection with Evidence-Guided Prompts"),
        ("4. Team Leader Name", "Vedansh Gupta"),
        ("5. Team Leader Enrollment No.", "S24CSEU0173"),
        ("6. Select your Group (Batch)", "B25-36"),
        ("7. Enrollment numbers of all team members", "S24CSEU0173\nS24CSEU0169\nS24CSEU0176\nS24CSEU0443\nS24CSEU0114"),
        ("8. Project Type (Deliverable)", "App with code"),
        ("9. Main AI tool and model used",
         "openai/gpt-oss-120b via the Groq API (temperature 0, JSON mode), called from our Python + Gradio app"),
        ("10. Number of prompts in the final solution", "1"),
        ("11. Prototype link", prototype or "(to add: Hugging Face Space link once deployed)"),
        ("12. Git repository or folder link", REPO),
        ("13. Reel link", "(to add: Drive or unlisted YouTube link, 'Anyone with the link can view')"),
        ("14. Reel length in seconds", "(to add: exact seconds, 90-150)"),
        ("15. Supporting document type", "Research Abstract"),
        ("16. Upload supporting document", "Upload GoldenGirls_ResearchAbstract.pdf (in docs/ of the repository)"),
        ("17. Number of test inputs in the test table", "14"),
        ("18. Total number of prompt versions", "5"),
        ("19. Prototype status", "All modules working"),
        ("20. Names of the working modules",
         "1. Text preparation\n2. Statistical features\n3. Algorithm v1 baseline\n4. Algorithm v2 (stylometric)\n"
         "5. LLM assessment (prompt P1 = V4)\n6. Answer validation\n7. Gradio interface"),
        ("21. Milestone 2 PDF", PDF),
        ("22. Main task of each team member",
         "S24CSEU0173 – Vedansh Gupta – Coordination and statistical integration: features, algorithm v1 baseline, "
         "algorithm v2, repository\n"
         "S24CSEU0169 – Vartika Manish – Prompt design and versioning: prompt versions V0-V4 and the iteration log\n"
         "S24CSEU0176 – Vibhor – Labelled data and reference statistics: HC3 cleaning, splits, reference medians, "
         "few-shot examples\n"
         "S24CSEU0443 – Sahin Garg – Robustness and false positives: typo, style and injection tests, edge and "
         "invalid test cases\n"
         "S24CSEU0114 – V Hasini – Interface and structured explanations: Gradio app, JSON and grounding checks, "
         "screenshots"),
        ("23. Anything the evaluator should know?",
         "No login needed. Paste a text, click 'Compute statistics' (features, algorithm v1 and v2 labels), then "
         "'Ask the LLM' with provider groq, model openai/gpt-oss-120b and prompt version V4 (the defaults). The "
         "LLM runs on a free Groq key, so many clicks in a short time may return a rate-limit message; wait a "
         "minute and retry. The saved outputs of all 14 tests, including the raw LLM JSON, "
         "are in test-data/outputs.json."),
        ("24. Confirmation", "Tick 'I confirm' only after all five members have agreed to the Section 7 table"),
    ]


def build_answers(prototype):
    d = docx.Document()
    d.add_heading("Milestone 2 - Form answers (Prompt Engg SCSE4041)", level=1)
    d.add_paragraph("Team Golden Girls - Statistical AI Text Detection with Evidence-Guided Prompts")
    for q, a in answers(prototype):
        p = d.add_paragraph()
        r = p.add_run(q)
        r.bold = True
        d.add_paragraph(a)
    return d


def build_brief(prototype):
    outputs = json.loads((ROOT / "test-data" / "outputs.json").read_text(encoding="utf-8"))["results"]
    v0 = (ROOT / "prompts" / "v1" / "P1_V0.txt").read_text(encoding="utf-8")
    d = docx.Document()
    d.styles["Normal"].font.size = Pt(10.5)
    d.add_heading("Reel brief - Milestone 2 (90-150 seconds)", level=1)
    d.add_paragraph("Project: Statistical AI Text Detection with Evidence-Guided Prompts - Team Golden Girls "
                    "(SCSE4041 Prompt Engineering, faculty Saurabh R Srivastava)")

    d.add_heading("Links", level=2)
    for label, link in (("Repository (code, prompts, tests)", REPO),
                        ("Prototype", prototype or "to be added (Hugging Face Space); until then record the local app"),
                        ("Milestone 2 PDF", PDF),
                        ("All prompt versions as text", f"{REPO}/tree/main/prompts"),
                        ("Test inputs and saved outputs", f"{REPO}/tree/main/test-data"),
                        ("Screenshots and diagrams", f"{REPO}/tree/main/docs")):
        d.add_paragraph(f"{label}: {link}", style="List Bullet")

    d.add_heading("The project in 30 seconds", level=2)
    d.add_paragraph(
        "Teachers can't tell from fluent writing whether a student or an AI wrote a text. Our app measures four "
        "statistics in Python (how much sentence length varies, vocabulary variety, repeated three-word phrases, "
        "word-frequency entropy) and compares them with typical human and AI values. An LLM (gpt-oss-120b on Groq) "
        "then reads only those numbers through our prompt and returns a label - Likely AI-generated, Likely "
        "human-written or Uncertain - with evidence that quotes the numbers and its limitations. It never claims "
        "proof; it supports a human reviewer.")

    d.add_heading("Numbers you can say on screen (all real)", level=2)
    for line in (
        "Prompt v1-v3 (V0-V2): 0 of 20 answers were valid JSON.",
        "Adding 3 worked examples (V3): 20 of 20 valid JSON.",
        "Final prompt (V4): 7 of 20 texts decided, 7 of 7 correct, 0 human texts wrongly flagged, 100 % of evidence "
        "quotes real numbers, 0 invented numbers.",
        "Our pure-statistics algorithm v2: 95.2 % accurate on 600 test texts (v1: 81.6 % on only half of them).",
        f"Test table: {sum(o['pass'] for o in outputs)} of {len(outputs)} tests pass; it is cautious on mixed "
        "evidence and keyboard gibberish fools it (say this - honesty earns marks).",
    ):
        d.add_paragraph(line, style="List Bullet")

    d.add_heading("Script and shot list (follow this order)", level=2)
    shots = [
        ("1. Problem (~10 s)", "Title card or the app's header.",
         "\"Teachers can't tell if an essay was written by a student or by ChatGPT, and wrongly accusing a student "
         "is worse than missing AI. We built a tool that shows the evidence instead of a black-box verdict.\""),
        ("2. Prompt experiment (~35 s)",
         "Show prompts/v1/P1_V0.txt on GitHub and one V0 answer (free text, not JSON). Then show "
         "prompts/final/P1_V4.txt and the test-5 JSON answer (docs/screenshot_test5.png or a live run).",
         "\"Our first prompt just asked the model to assess the text - it answered in prose, and 0 of 20 answers "
         "were usable JSON. We added a role, rules that every claim must quote a real number, three worked examples "
         "and a strict JSON schema. The final prompt gives valid JSON 20 out of 20 times, and every piece of "
         "evidence quotes a number we computed.\""),
        ("3. Working demo (~50 s)",
         "Screen-record the app (local: python -m src.app, open http://127.0.0.1:7861, provider groq, model "
         "openai/gpt-oss-120b, version V4). Paste test 5 text (HC3 ChatGPT answer about trolls) -> Compute "
         "statistics -> Ask the LLM. Then paste test 1 (a human answer) -> Ask the LLM.",
         "\"Here's a real ChatGPT answer. Python computes the four statistics - all four sit closer to the AI "
         "range. Our algorithm says AI, and the LLM says Likely AI-generated, citing each number. Now a human "
         "answer: the numbers move to the human range, and the label follows. Every answer is checked: valid JSON, "
         "grounded evidence, no invented numbers.\""),
        ("4. Progress and next steps (~15 s)", "README or CHANGELOG on GitHub.",
         "\"Done: statistics, two algorithms, five prompt versions, a 14-case test table and the app. Next for "
         "Milestone 3: feed the algorithm's probability into the prompt so it is less cautious on mixed evidence, "
         "and a pre-check that rejects gibberish.\""),
    ]
    t = d.add_table(rows=1, cols=3)
    t.style = "Table Grid"
    for i, h in enumerate(("Part", "What to show", "Voice-over (suggested)")):
        t.rows[0].cells[i].text = h
    for row in shots:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = v

    d.add_heading("Texts to paste in the demo", level=2)
    inputs = {c["no"]: c for c in json.loads((ROOT / "test-data" / "inputs.json").read_text(encoding="utf-8"))}
    for no in (5, 1, 9):
        d.add_paragraph(f"Test {no} - {inputs[no]['description']}:").runs[0].bold = True
        d.add_paragraph(inputs[no]["text"])

    d.add_heading("Rules from the guidelines", level=2)
    for line in ("Length 90-150 seconds (note the exact seconds for the form).",
                 "Upload to Google Drive or unlisted YouTube, sharing 'Anyone with the link can view'; open it in an "
                 "incognito window before sending the link.",
                 "The demo must be a real run (mock-up screens get 0 for the demo part). Faces optional; vertical or "
                 "horizontal both fine.",
                 "Never show the .env file or any API key on screen.",
                 "Record either the online prototype or the local app (a Groq key in .env on the recording laptop)."):
        d.add_paragraph(line, style="List Bullet")
    d.add_heading("Reference: first prompt (V0) text", level=2)
    d.add_paragraph(v0.split("===== USER =====")[1].split("TEXT TO ASSESS")[0].strip())
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prototype", default="")
    a = ap.parse_args()
    build_answers(a.prototype).save(OUT / "M2_Form_Answers.docx")
    build_brief(a.prototype).save(OUT / "Reel_Brief.docx")
    print("wrote", OUT / "M2_Form_Answers.docx", "and", OUT / "Reel_Brief.docx")


if __name__ == "__main__":
    main()
