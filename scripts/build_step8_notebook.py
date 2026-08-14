from __future__ import annotations

import json
from pathlib import Path

from build_kaggle_notebook import code_cell, project_archive


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "kaggle_step8_notebook.ipynb"


def build() -> dict:
    archive = project_archive()
    setup = f'''from pathlib import Path
import base64
import io
import json
import shutil
import subprocess
import sys
import zipfile

PROJECT_DIR = Path("/kaggle/working/Luan-Van-GC-LSTM-GhostNet-CICDDoS2019-v1")
OUTPUT_DIR = PROJECT_DIR / "outputs" / "step8" / "practical_split80"
OUTER_TRAIN_FRACTION = 0.80
RUN_NAME = "gc-lstm-ghostnet-practical-split80-full"
PROJECT_ARCHIVE_B64 = "{archive}"

if PROJECT_DIR.exists():
    shutil.rmtree(PROJECT_DIR)
PROJECT_DIR.mkdir(parents=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PROJECT_ARCHIVE_B64))) as bundle:
    bundle.extractall(PROJECT_DIR)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(PROJECT_DIR / "requirements.txt")], check=True)

DATA_DIR = Path("/kaggle/input")
if next(DATA_DIR.rglob("dataset_summary.json"), None) is None:
    raise FileNotFoundError(
        "Attach dungnguyen28101991/cicddos2019-parquet before running Step 8."
    )
resume_candidates = sorted(DATA_DIR.rglob("last_checkpoint.pt"))
RESUME_PATH = resume_candidates[-1] if resume_candidates else None
print({{"device": "cpu", "outer_train_fraction": OUTER_TRAIN_FRACTION,
       "run_name": RUN_NAME, "resume": str(RESUME_PATH) if RESUME_PATH else None}})
'''
    run = '''command = [
    sys.executable, "train.py",
    "--data-dir", str(DATA_DIR),
    "--output-dir", str(OUTPUT_DIR),
    "--config", "configs/base.yaml",
    "--mode-config", "configs/practical_baseline.yaml",
    "--epochs", "100",
    "--batch-size", "512",
    "--learning-rate", "0.001",
    "--device", "cpu",
    "--full-dataset", "--stream-files",
    "--sequence-group-rows", "4096",
    "--stream-shuffle-buffer-sequences", "8192",
    "--stream-eval-samples-per-file", "512",
    "--train-eval-samples-per-class", "256",
    "--outer-train-fraction", str(OUTER_TRAIN_FRACTION),
    "--run-name", RUN_NAME,
    "--session-budget-minutes", "300",
]
if RESUME_PATH is not None:
    command.extend(["--resume", str(RESUME_PATH)])
subprocess.run(command, cwd=PROJECT_DIR, check=True)
'''
    verify = '''session_summary = OUTPUT_DIR / "step8_session_summary.json"
final_model = OUTPUT_DIR / "final_model_epoch_100.pt"
if final_model.exists():
    run_config = json.loads((OUTPUT_DIR / "run_config.json").read_text(encoding="utf-8"))
    assert run_config["execution_scope"] == "full_mixed_group_streaming"
    assert run_config["device"] == "cpu"
    assert run_config["counts_equal"] is True
    result = {"status": "full_run_complete", "epoch": 100, "run_config": run_config}
elif session_summary.exists():
    result = json.loads(session_summary.read_text(encoding="utf-8"))
    assert result["status"] == "controlled_session_stop"
    assert result["counts_equal_at_safe_stop"] is True
else:
    raise RuntimeError("Step 8 produced neither a safe session checkpoint nor final epoch 100")
result
'''
    notebook = {
        "cells": [
            {"cell_type": "markdown", "id": "step8-intro", "metadata": {}, "source": [
                "# Step 8 — Full mixed-group streaming, practical baseline, split 80%, CPU only\n",
                "\n",
                "Consumes every eligible train sequence exactly once per epoch and exits safely before the Kaggle session limit.\n",
            ]},
            code_cell(setup, "materialize-step8-project"),
            code_cell(run, "run-step8-full-stream"),
            code_cell(verify, "verify-step8-session"),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"<notebook:{cell['id']}>", "exec")
    return notebook


if __name__ == "__main__":
    OUTPUT.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(OUTPUT)
