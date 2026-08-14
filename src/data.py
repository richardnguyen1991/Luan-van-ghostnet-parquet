from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - exercised by Kaggle/integration environment
    pq = None

from .config import config_hash, load_config


LABEL_CANDIDATES = ("Label", "label", "Class", "class", "target", "Target")
PROVENANCE_COLUMNS = ("__capture_day", "__source_file_id", "__source_row_id")


def canonical_column_name(name: str) -> str:
    return str(name).strip()


def canonicalize_columns(columns: Sequence[str]) -> list[str]:
    canonical = [canonical_column_name(column) for column in columns]
    counts = Counter(canonical)
    collisions = sorted(name for name, count in counts.items() if count > 1)
    if collisions:
        raise ValueError(f"Column-name collisions after whitespace normalization: {collisions}")
    return canonical


def canonicalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    canonical = canonicalize_columns(list(frame.columns))
    result = frame.copy()
    result.columns = canonical
    return result


def detect_unique_column(columns: Sequence[str], candidates: Sequence[str], purpose: str) -> str:
    canonical = set(canonicalize_columns(columns))
    found = [candidate for candidate in candidates if candidate in canonical]
    if len(found) != 1:
        raise ValueError(f"Expected exactly one {purpose} column from {list(candidates)}, found {found}")
    return found[0]


def stable_sample_ids(frame: pd.DataFrame) -> pd.Series:
    missing = [column for column in ("__source_file_id", "__source_row_id") if column not in frame]
    if missing:
        raise ValueError(f"Cannot create sample_id; missing provenance columns: {missing}")
    payload = (
        frame["__source_file_id"].astype("string").fillna("<missing>")
        + ":"
        + frame["__source_row_id"].astype("Int64").astype("string").fillna("<missing>")
    )
    return payload.map(lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())


def stable_schema_hash(columns: Sequence[str], dtypes: Sequence[str]) -> str:
    payload = json.dumps(list(zip(columns, dtypes)), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DatasetPaths:
    root: Path
    data_root: Path
    manifest_root: Path
    parquet_files: tuple[Path, ...]


def _candidate_dataset_roots(search_root: Path, hint: str) -> list[tuple[int, Path]]:
    candidates: list[tuple[int, Path]] = []
    manifest_name = "dataset_summary.json"
    for summary in search_root.rglob(manifest_name):
        if summary.parent.name != "manifests":
            continue
        root = summary.parent.parent
        data_root = root / "data"
        score = 0
        lower = str(root).lower()
        if hint.lower() in lower:
            score += 100
        if data_root.exists():
            score += 50
        if (root / "manifests" / "validation_report.json").exists():
            score += 20
        candidates.append((score, root))
    return sorted(candidates, key=lambda item: (item[0], len(item[1].parts)), reverse=True)


def discover_dataset(data_dir: str | Path, dataset_dir_hint: str, expected_files: int = 18) -> DatasetPaths:
    search_root = Path(data_dir)
    if not search_root.exists():
        raise FileNotFoundError(search_root)
    if (search_root / "manifests" / "dataset_summary.json").exists():
        root = search_root
    else:
        candidates = _candidate_dataset_roots(search_root, dataset_dir_hint)
        if not candidates:
            raise FileNotFoundError(
                f"No structure-preserving dataset with manifests/dataset_summary.json below {search_root}"
            )
        root = candidates[0][1]
    data_root = root / "data"
    manifest_root = root / "manifests"
    parquet_files = tuple(sorted(data_root.rglob("*.parquet")))
    if len(parquet_files) != int(expected_files):
        raise RuntimeError(
            f"Expected {expected_files} Parquet source files in {data_root}, found {len(parquet_files)}"
        )
    return DatasetPaths(root, data_root, manifest_root, parquet_files)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_dataset_manifests(paths: DatasetPaths, data_config: dict[str, Any]) -> dict[str, Any]:
    summary = read_json(paths.manifest_root / "dataset_summary.json")
    validation = read_json(paths.manifest_root / "validation_report.json")
    schema = read_json(paths.manifest_root / "column_schema.json")
    if validation.get("status") != "passed" or validation.get("error_count") != 0:
        raise RuntimeError(f"Dataset conversion validation did not pass: {validation}")
    checks = {
        "source_file_count": (int(summary["source_file_count"]), int(data_config["expected_source_files"])),
        "original_column_count": (
            int(schema["original_column_count"]), int(data_config["expected_original_columns"])
        ),
        "output_column_count": (
            int(schema["output_column_count"]), int(data_config["expected_output_columns"])
        ),
        "schema_hash": (str(schema["schema_hash"]), str(data_config["expected_schema_hash"])),
    }
    failures = {name: values for name, values in checks.items() if values[0] != values[1]}
    if failures:
        raise RuntimeError(f"Dataset manifest contract mismatch: {failures}")
    original_columns = canonicalize_columns(schema["original_columns"])
    for required in PROVENANCE_COLUMNS:
        if required not in schema["provenance_columns_appended"]:
            raise RuntimeError(f"Required provenance column missing from manifest: {required}")
    label = detect_unique_column(original_columns, data_config["label_candidates"], "label")
    timestamp = detect_unique_column(original_columns, data_config["timestamp_candidates"], "timestamp")
    return {
        "summary": summary,
        "validation": validation,
        "schema": schema,
        "canonical_original_columns": original_columns,
        "label_column": label,
        "timestamp_column": timestamp,
    }


def _distributed_row_group_indices(total: int, desired: int) -> list[int]:
    if total <= 0:
        return []
    count = min(total, max(1, desired))
    return sorted(set(int(round(x)) for x in np.linspace(0, total - 1, count)))


def sample_parquet_file(path: Path, samples: int, seed: int) -> pd.DataFrame:
    if pq is None:
        raise ImportError("pyarrow is required to read the Kaggle Parquet dataset")
    parquet_file = pq.ParquetFile(path)
    desired_groups = min(parquet_file.num_row_groups, max(1, math.ceil(samples / 256)))
    group_indices = _distributed_row_group_indices(parquet_file.num_row_groups, desired_groups)
    tables = [parquet_file.read_row_group(index) for index in group_indices]
    if not tables:
        return pd.DataFrame()
    frame = pd.concat([table.to_pandas() for table in tables], ignore_index=True)
    if len(frame) > samples:
        frame = frame.sample(n=samples, random_state=seed).sort_index().reset_index(drop=True)
    return canonicalize_frame(frame)


def load_sampled_dataset(
    paths: DatasetPaths,
    samples_per_file: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    expected_columns: list[str] | None = None
    for index, path in enumerate(paths.parquet_files):
        frame = sample_parquet_file(path, int(samples_per_file), seed + index)
        columns = list(frame.columns)
        if expected_columns is None:
            expected_columns = columns
        elif columns != expected_columns:
            raise RuntimeError(f"Sample schema differs: {path}")
        frame["sample_id"] = stable_sample_ids(frame)
        frames.append(frame)
        records.append({
            "file": str(path.relative_to(paths.root)).replace("\\", "/"),
            "sampled_rows": len(frame),
            "sampling": "distributed_row_groups_then_seeded_rows",
        })
    combined = pd.concat(frames, ignore_index=True)
    if combined["sample_id"].duplicated().any():
        duplicates = int(combined["sample_id"].duplicated().sum())
        raise RuntimeError(f"Stable sample_id collision/duplication detected: {duplicates}")
    manifest = {
        "mode": "sampled",
        "samples_per_file": int(samples_per_file),
        "seed": int(seed),
        "total_sampled_rows": len(combined),
        "files": records,
    }
    return combined, manifest


def sample_contiguous_parquet_file(
    path: Path,
    samples: int,
    seed: int,
    minimum_run_rows: int,
    maximum_row_groups: int = 4,
) -> pd.DataFrame:
    """Read deterministic contiguous slices without inventing temporal adjacency."""
    if pq is None:
        raise ImportError("pyarrow is required to read the Kaggle Parquet dataset")
    if int(samples) <= 0 or int(minimum_run_rows) <= 0:
        raise ValueError("samples and minimum_run_rows must be positive")
    parquet_file = pq.ParquetFile(path)
    desired_groups = min(
        parquet_file.num_row_groups,
        max(1, min(int(maximum_row_groups), int(samples) // int(minimum_run_rows))),
    )
    group_indices = _distributed_row_group_indices(parquet_file.num_row_groups, desired_groups)
    base_quota, remainder = divmod(int(samples), max(1, len(group_indices)))
    frames: list[pd.DataFrame] = []
    for position, group_index in enumerate(group_indices):
        quota = base_quota + int(position < remainder)
        if quota <= 0:
            continue
        table = parquet_file.read_row_group(group_index)
        take = min(len(table), quota)
        if take <= 0:
            continue
        maximum_start = max(0, len(table) - take)
        digest = hashlib.sha256(f"{seed}:{path.name}:{group_index}".encode("utf-8")).digest()
        raw_start = int.from_bytes(digest[:8], "big") % (maximum_start + 1)
        aligned_start = (raw_start // int(minimum_run_rows)) * int(minimum_run_rows)
        start = min(aligned_start, maximum_start)
        frames.append(canonicalize_frame(table.slice(start, take).to_pandas()))
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    required = ["__source_file_id", "__source_row_id"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise RuntimeError(f"Contiguous sample is missing provenance columns: {missing}")
    return frame.sort_values(required, kind="stable").reset_index(drop=True)


def load_contiguous_sequence_dataset(
    paths: DatasetPaths,
    samples_per_file: int,
    seed: int,
    minimum_run_rows: int,
    maximum_row_groups: int = 4,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    expected_columns: list[str] | None = None
    for index, path in enumerate(paths.parquet_files):
        frame = sample_contiguous_parquet_file(
            path,
            int(samples_per_file),
            int(seed) + index,
            int(minimum_run_rows),
            int(maximum_row_groups),
        )
        columns = list(frame.columns)
        if expected_columns is None:
            expected_columns = columns
        elif columns != expected_columns:
            raise RuntimeError(f"Contiguous sample schema differs: {path}")
        frame["sample_id"] = stable_sample_ids(frame)
        frames.append(frame)
        records.append({
            "file": str(path.relative_to(paths.root)).replace("\\", "/"),
            "sampled_rows": len(frame),
            "sampling": "distributed_row_groups_then_deterministic_contiguous_slices",
        })
    combined = pd.concat(frames, ignore_index=True)
    if combined["sample_id"].duplicated().any():
        duplicates = int(combined["sample_id"].duplicated().sum())
        raise RuntimeError(f"Stable sample_id collision/duplication detected: {duplicates}")
    manifest = {
        "mode": "contiguous_sequence_sampled",
        "samples_per_file": int(samples_per_file),
        "minimum_run_rows": int(minimum_run_rows),
        "maximum_row_groups_per_file": int(maximum_row_groups),
        "seed": int(seed),
        "total_sampled_rows": len(combined),
        "files": records,
    }
    return combined, manifest


def build_data_profile(
    manifest_contract: dict[str, Any],
    sample: pd.DataFrame,
    paths: DatasetPaths,
) -> dict[str, Any]:
    label_column = manifest_contract["label_column"]
    timestamp_column = manifest_contract["timestamp_column"]
    numeric = sample.select_dtypes(include=[np.number])
    infinity_counts = {
        column: int(np.isinf(numeric[column].to_numpy(dtype=np.float64, na_value=np.nan)).sum())
        for column in numeric.columns
    }
    infinity_counts = {key: value for key, value in infinity_counts.items() if value}
    missing_counts = {key: int(value) for key, value in sample.isna().sum().items() if value}
    parsed_timestamp = pd.to_datetime(sample[timestamp_column], errors="coerce", dayfirst=True)
    risky_columns = [
        column for column in (
            "Unnamed: 0", "Flow ID", "Source IP", "Source Port", "Destination IP",
            "Destination Port", "Protocol", timestamp_column, *PROVENANCE_COLUMNS
        ) if column in sample
    ]
    return {
        "dataset_root": str(paths.root),
        "source_file_count": len(paths.parquet_files),
        "manifest_total_rows": int(manifest_contract["summary"]["total_rows"]),
        "sample_rows": len(sample),
        "column_count": len(sample.columns),
        "columns": list(sample.columns),
        "dtypes": {column: str(dtype) for column, dtype in sample.dtypes.items()},
        "label_column": label_column,
        "label_counts_sample": {
            str(key): int(value) for key, value in sample[label_column].value_counts(dropna=False).items()
        },
        "timestamp_column": timestamp_column,
        "timestamp_parse_failures_sample": int(parsed_timestamp.isna().sum()),
        "missing_counts_sample": missing_counts,
        "infinity_counts_sample": infinity_counts,
        "leakage_or_identity_columns": risky_columns,
        "paper_claimed_feature_count": 84,
        "actual_original_column_count": int(manifest_contract["schema"]["original_column_count"]),
        "schema_difference_note": "Observed schema is authoritative; it is not forced to 84 columns.",
    }


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    temporary.replace(output)


def audit_sampled_dataset(
    data_dir: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    data_config = config["data"]
    paths = discover_dataset(
        data_dir,
        data_config["dataset_dir_hint"],
        data_config["expected_source_files"],
    )
    contract = validate_dataset_manifests(paths, data_config)
    sample, sample_manifest = load_sampled_dataset(
        paths,
        data_config["samples_per_file"],
        config["project"]["seed"],
    )
    profile = build_data_profile(contract, sample, paths)
    output_root = Path(output_dir)
    write_json(output_root / "data_profile.json", profile)
    write_json(output_root / "sample_manifest.json", sample_manifest)
    write_json(output_root / "dataset_contract.json", {
        "config_hash": config_hash(config),
        "dataset_schema_hash": contract["schema"]["schema_hash"],
        "conversion_validation": contract["validation"],
    })
    return sample, profile, sample_manifest


def audit_contiguous_sequence_dataset(
    data_dir: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    data_config = config["data"]
    step3 = config["step3"]
    paths = discover_dataset(
        data_dir,
        data_config["dataset_dir_hint"],
        data_config["expected_source_files"],
    )
    contract = validate_dataset_manifests(paths, data_config)
    sample, sample_manifest = load_contiguous_sequence_dataset(
        paths,
        data_config["samples_per_file"],
        config["project"]["seed"],
        step3["sequence_length"],
        step3["sampled_row_groups_per_file"],
    )
    profile = build_data_profile(contract, sample, paths)
    output_root = Path(output_dir)
    write_json(output_root / "data_profile.json", profile)
    write_json(output_root / "sample_manifest.json", sample_manifest)
    write_json(output_root / "dataset_contract.json", {
        "config_hash": config_hash(config),
        "dataset_schema_hash": contract["schema"]["schema_hash"],
        "conversion_validation": contract["validation"],
        "sampling_preserves_contiguous_source_rows": True,
    })
    return sample, profile, sample_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the structure-preserving CIC-DDoS2019 Parquet dataset")
    parser.add_argument("--data-dir", default="/kaggle/input")
    parser.add_argument("--output-dir", default="/kaggle/working/Luan-Van-GC-LSTM-GhostNet-CICDDoS2019-v1/outputs/audit")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode-config")
    parser.add_argument("--samples-per-file", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--target-column")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.mode_config)
    if args.samples_per_file is not None:
        config["data"]["samples_per_file"] = args.samples_per_file
    if args.seed is not None:
        config["project"]["seed"] = args.seed
    if args.target_column is not None:
        config["data"]["label_candidates"] = [args.target_column]
    _, profile, _ = audit_sampled_dataset(args.data_dir, args.output_dir, config)
    print(json.dumps(profile, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

