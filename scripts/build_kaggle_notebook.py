from __future__ import annotations

import base64
import io
import json
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "kaggle_notebook.ipynb"


def project_archive() -> str:
    include = [
        "README.md",
        "train.py",
        "assumptions.yaml",
        "paper_alignment.md",
        "requirements.txt",
        "traceability_matrix.csv",
    ]
    include.extend(
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        for folder in ("configs", "src", "tests")
        for path in sorted((PROJECT_ROOT / folder).rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in include:
            archive.writestr(relative, (PROJECT_ROOT / relative).read_bytes())
    return base64.b64encode(payload.getvalue()).decode("ascii")


def code_cell(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def build_notebook() -> dict:
    archive = project_archive()
    setup_source = f'''from pathlib import Path
import base64
import io
import json
import os
import shutil
import subprocess
import sys
import zipfile
from kaggle_secrets import UserSecretsClient

PROJECT_DIR = Path("/kaggle/working/Luan-Van-GC-LSTM-GhostNet-CICDDoS2019-v1")
OUTPUT_DIR = PROJECT_DIR / "outputs" / "step4_smoke"
MOUNTED_DATA_CANDIDATES = [
    Path("/kaggle/input/cicddos2019-parquet"),
    Path("/kaggle/input/datasets/dungnguyen28101991/cicddos2019-parquet"),
]
DOWNLOADED_DATA_DIR = Path("/kaggle/working/cicddos2019-parquet-input")
PROJECT_ARCHIVE_B64 = "{archive}"

if PROJECT_DIR.exists():
    shutil.rmtree(PROJECT_DIR)
PROJECT_DIR.mkdir(parents=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PROJECT_ARCHIVE_B64))) as project_zip:
    project_zip.extractall(PROJECT_DIR)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-r", str(PROJECT_DIR / "requirements.txt")],
    check=True,
)

mounted_data_dir = next((path for path in MOUNTED_DATA_CANDIDATES if path.exists()), None)
if mounted_data_dir is None and next(Path("/kaggle/input").rglob("dataset_summary.json"), None):
    # Kaggle may choose a normalized mount slug that differs from the API slug.
    # The data loader recursively selects the validated manifests below this root.
    mounted_data_dir = Path("/kaggle/input")
if mounted_data_dir is not None:
    DATA_DIR = mounted_data_dir
else:
    DATA_DIR = DOWNLOADED_DATA_DIR
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    DATA_DIR.mkdir(parents=True)
    download_env = os.environ.copy()
    secret_value = UserSecretsClient().get_secret("KAGGLE_API_TOKEN")
    try:
        classic = json.loads(secret_value)
    except (TypeError, json.JSONDecodeError):
        download_env["KAGGLE_API_TOKEN"] = secret_value
    else:
        download_env["KAGGLE_USERNAME"] = classic["username"]
        download_env["KAGGLE_KEY"] = classic["key"]
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", "dungnguyen28101991/cicddos2019-parquet",
         "-p", str(DATA_DIR), "--unzip", "--quiet"],
        env=download_env,
        check=True,
    )
    for key in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        download_env.pop(key, None)
    del secret_value
print(f"Step 4 project ready; using dataset at {{DATA_DIR}}")
'''
    run_source = '''command = [
    sys.executable, "train.py",
    "--data-dir", str(DATA_DIR),
    "--output-dir", str(OUTPUT_DIR),
    "--config", "configs/base.yaml",
    "--mode-config", "configs/practical_baseline.yaml",
    "--samples-per-file", "2048",
    "--sequence-length", "16",
    "--sequence-stride", "8",
    "--epochs", "2",
    "--batch-size", "64",
    "--run-name", "kaggle-step4-smoke",
]
subprocess.run(command, cwd=PROJECT_DIR, check=True)
'''
    verify_source = '''summary_path = OUTPUT_DIR / "step4_summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
assert summary["status"] == "passed", summary
assert summary["sequence_leakage_status"] == "passed", summary
assert summary["best_epoch"] in (1, 2), summary
assert (OUTPUT_DIR / "training" / "best_model.pt").exists()
assert (OUTPUT_DIR / "training" / "test_metrics.json").exists()
summary
'''
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "step4-intro",
                "metadata": {},
                "source": [
                    "# GC-LSTM-GhostNet - Step 4 sampled training smoke test\n",
                    "\n",
                    "Train GCN → LSTM → temporal attention → GhostNet with leakage-safe contiguous windows.\n",
                ],
            },
            code_cell(setup_source, "materialize-step4-project"),
            code_cell(run_source, "run-step4-smoke"),
            code_cell(verify_source, "verify-step4-summary"),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    NOTEBOOK_PATH.write_text(
        json.dumps(build_notebook(), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
