"""Screenshots of real app runs for the Milestone 2 PDF (app must be running on 127.0.0.1:7861).

python docs/shoot_app.py   ->  docs/screenshot_test<N>.png for the tests listed in SHOTS
"""

import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SHOTS = [int(x) for x in sys.argv[1:]] or [1, 5, 9]
inputs = {c["no"]: c for c in json.loads((ROOT / "test-data" / "inputs.json").read_text(encoding="utf-8"))}

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True)
    page = browser.new_page(viewport={"width": 1400, "height": 1000})
    for no in SHOTS:
        page.goto("http://127.0.0.1:7861", wait_until="networkidle")
        page.get_by_label("Text to review").fill(inputs[no]["text"])
        page.get_by_role("button", name="Ask the LLM").click()
        page.get_by_text("JSON valid (strict schema)").wait_for(timeout=120000)
        page.wait_for_timeout(800)
        out = ROOT / "docs" / f"screenshot_test{no}.png"
        page.screenshot(path=str(out), full_page=True)
        print("saved", out.name)
    browser.close()
