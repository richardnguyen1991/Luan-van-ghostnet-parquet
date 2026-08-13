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
import shutil
import subprocess
import sys
import zipfile

PROJECT_DIR = Path("/kaggle/working/Luan-Van-GC-LSTM-GhostNet-CICDDoS2019-v1")
OUTPUT_DIR = PROJECT_DIR / "outputs" / "step7_sampled_end_to_end"
MOUNTED_DATA_CANDIDATES = [
    Path("/kaggle/input/cicddos2019-parquet"),
    Path("/kaggle/input/datasets/dungnguyen28101991/cicddos2019-parquet"),
]
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
    mounted_entries = sorted(str(path) for path in Path("/kaggle/input").iterdir())
    raise FileNotFoundError(
        "Dataset input is not attached. In the Kaggle editor choose Add Input -> "
        "dungnguyen28101991/cicddos2019-parquet, then Save Version / Run All. "
        f"Current /kaggle/input entries: {{mounted_entries}}"
    )
print(f"Step 7 sampled end-to-end CPU project ready; using dataset at {{DATA_DIR}}")
'''
    run_source = '''command = [
    sys.executable, "-m", "src.step6_artifact_smoke",
    "--data-dir", str(DATA_DIR),
    "--output-dir", str(OUTPUT_DIR),
    "--samples-per-file", "2048",
    "--sequence-length", "16",
    "--sequence-stride", "8",
    "--batch-size", "64",
    "--device", "cpu",
]
subprocess.run(command, cwd=PROJECT_DIR, check=True)
'''
    verify_source = '''import json

summary_path = OUTPUT_DIR / "step6_summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
assert summary["status"] == "passed", summary
assert summary["device"] == "cpu", summary
assert summary["sequence_leakage_status"] == "passed", summary
assert summary["expected_not_yet_run"] == ["ablation_comparison", "cfaco_convergence"], summary
assert len(summary["report"]["produced"]) == 11, summary
assert (OUTPUT_DIR / "report" / "report_status.json").exists()
assert (OUTPUT_DIR / "artifacts" / "benchmark.json").exists()
step7_acceptance = {{
    "status": "passed",
    "step": 7,
    "mode": "sampled_end_to_end",
    "device": "cpu",
    "source_summary": str(summary_path),
    "produced_report_groups": len(summary["report"]["produced"]),
}}
(OUTPUT_DIR / "step7_acceptance.json").write_text(
    json.dumps(step7_acceptance, indent=2, ensure_ascii=False) + "\\n",
    encoding="utf-8",
)
step7_acceptance
'''
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "step7-intro",
                "metadata": {},
                "source": [
                    "# GC-LSTM-GhostNet - Step 7 sampled end-to-end (CPU only)\n",
                    "\n",
                    "Validate the attached Parquet dataset, train a bounded two-epoch sample, create real explainability and benchmark artifacts, and verify the complete report contract.\n",
                ],
            },
            code_cell(setup_source, "materialize-step7-project"),
            code_cell(run_source, "run-step7-sampled-end-to-end"),
            code_cell(verify_source, "verify-step7-acceptance"),
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

