"""Upload the app to the Hugging Face Space (needs `hf auth login`).

python deploy/deploy_space.py [--space USER/Evidence-Guided-AI-Text-Detection]
Uploads deploy/space/* (entry point, requirements, Space card), src/ and the frozen models/*.json.
"""

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("baseline.json", "few_shot_examples.json", "stylometry.json")


def main() -> None:
    api = HfApi()
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", default=None)
    space = parser.parse_args().space or f"{api.whoami()['name']}/Evidence-Guided-AI-Text-Detection"
    api.create_repo(space, repo_type="space", space_sdk="gradio", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        shutil.copytree(ROOT / "deploy" / "space", out, dirs_exist_ok=True)
        shutil.copytree(ROOT / "src", out / "src", ignore=shutil.ignore_patterns("__pycache__"))
        (out / "models").mkdir()
        for name in MODELS:
            shutil.copy2(ROOT / "models" / name, out / "models" / name)
        api.upload_folder(folder_path=tmp, repo_id=space, repo_type="space", commit_message="Deploy app")
    print(f"https://huggingface.co/spaces/{space}")


if __name__ == "__main__":
    main()
