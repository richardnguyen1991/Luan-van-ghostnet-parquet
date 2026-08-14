import json
from pathlib import Path

import pandas as pd
import pytest

from src.data import (
    canonicalize_columns,
    canonicalize_frame,
    detect_unique_column,
    discover_dataset,
    stable_sample_ids,
    validate_dataset_manifests,
)


def test_column_whitespace_is_canonicalized_without_silent_collision():
    frame = pd.DataFrame({" Label": ["BENIGN"], " Timestamp": ["x"], "Flow ID": ["a"]})
    canonical = canonicalize_frame(frame)
    assert list(canonical.columns) == ["Label", "Timestamp", "Flow ID"]
    assert detect_unique_column(canonical.columns, ["Label", "label"], "label") == "Label"
    with pytest.raises(ValueError, match="collisions"):
        canonicalize_columns(["Label", " Label"])


def test_sample_ids_are_stable_and_source_scoped():
    frame = pd.DataFrame({
        "__source_file_id": ["01-12/a.csv", "01-12/a.csv", "03-11/a.csv"],
        "__source_row_id": [0, 1, 0],
    })
    first = stable_sample_ids(frame)
    second = stable_sample_ids(frame.copy())
    assert first.equals(second)
    assert first.nunique() == 3


def test_manifest_contract_selects_primary_structure_preserving_root(tmp_path: Path):
    root = tmp_path / "cicddos2019_parquet_preserved"
    data = root / "data" / "01-12"
    manifests = root / "manifests"
    data.mkdir(parents=True)
    manifests.mkdir(parents=True)
    for index in range(18):
        (data / f"source_{index}.parquet").touch()
    schema_hash = "abc"
    (manifests / "dataset_summary.json").write_text(json.dumps({
        "source_file_count": 18, "total_rows": 100, "schema_hash": schema_hash
    }), encoding="utf-8")
    (manifests / "validation_report.json").write_text(json.dumps({
        "status": "passed", "error_count": 0, "errors": []
    }), encoding="utf-8")
    (manifests / "column_schema.json").write_text(json.dumps({
        "original_column_count": 3,
        "output_column_count": 6,
        "schema_hash": schema_hash,
        "original_columns": [" Timestamp", " Label", "Flow ID"],
        "provenance_columns_appended": ["__capture_day", "__source_file_id", "__source_row_id"],
    }), encoding="utf-8")
    paths = discover_dataset(tmp_path, "cicddos2019_parquet_preserved", 18)
    contract = validate_dataset_manifests(paths, {
        "expected_source_files": 18,
        "expected_original_columns": 3,
        "expected_output_columns": 6,
        "expected_schema_hash": schema_hash,
        "label_candidates": ["Label", "label"],
        "timestamp_candidates": ["Timestamp", "timestamp"],
    })
    assert contract["label_column"] == "Label"
    assert contract["timestamp_column"] == "Timestamp"

