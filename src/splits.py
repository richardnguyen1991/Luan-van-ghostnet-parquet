from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SPLIT_NAMES = ("train", "validation", "test")


def make_contiguous_group_ids(frame: pd.DataFrame, group_rows: int) -> pd.Series:
    required = ["__source_file_id", "__source_row_id"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Cannot create contiguous groups; missing {missing}")
    if int(group_rows) <= 0:
        raise ValueError("group_rows must be positive")
    source_rows = pd.to_numeric(frame["__source_row_id"], errors="raise").astype("int64")
    if (source_rows < 0).any():
        raise ValueError("__source_row_id must be non-negative")
    blocks = source_rows // int(group_rows)
    return frame["__source_file_id"].astype("string") + "::block=" + blocks.astype("string")


def _seeded_group_order(group_ids: list[str], seed: int) -> list[str]:
    return sorted(
        group_ids,
        key=lambda group: hashlib.sha256(f"{seed}:{group}".encode("utf-8")).hexdigest(),
    )


def _take_groups_to_row_target(
    group_rows: pd.Series,
    group_ids: list[str],
    target_rows: float,
    seed: int,
) -> set[str]:
    ordered = _seeded_group_order(group_ids, seed)
    chosen: set[str] = set()
    current = 0
    for group in ordered:
        if chosen and abs(current - target_rows) <= abs(current + int(group_rows[group]) - target_rows):
            continue
        chosen.add(group)
        current += int(group_rows[group])
    if not chosen and ordered:
        chosen.add(ordered[0])
    if len(chosen) == len(ordered) and len(ordered) > 1:
        chosen.remove(ordered[-1])
    return chosen


def _distribution_score(
    frame: pd.DataFrame,
    label_column: str,
    split_column: str,
    target_fractions: dict[str, float],
) -> tuple[float, dict[str, Any]]:
    total_rows = max(1, len(frame))
    overall = frame[label_column].astype("string").value_counts(normalize=True)
    score = 0.0
    details: dict[str, Any] = {"splits": {}}
    for split in SPLIT_NAMES:
        subset = frame.loc[frame[split_column] == split]
        actual_fraction = len(subset) / total_rows
        fraction_error = abs(actual_fraction - target_fractions[split])
        score += fraction_error * 4.0
        distribution = subset[label_column].astype("string").value_counts(normalize=True)
        label_error = float((overall - distribution.reindex(overall.index, fill_value=0.0)).abs().mean())
        missing_labels = sorted(set(overall.index) - set(distribution.index))
        score += label_error + 0.25 * len(missing_labels)
        details["splits"][split] = {
            "rows": len(subset),
            "fraction": actual_fraction,
            "target_fraction": target_fractions[split],
            "fraction_error": fraction_error,
            "label_distribution_mean_absolute_error": label_error,
            "missing_labels": missing_labels,
        }
    return score, details


@dataclass(frozen=True)
class SplitResult:
    frame: pd.DataFrame
    report: dict[str, Any]


def assign_group_splits(
    frame: pd.DataFrame,
    label_column: str,
    outer_train_fraction: float,
    validation_fraction_of_outer_train: float,
    group_rows: int,
    seed: int,
    candidate_attempts: int = 128,
) -> SplitResult:
    if label_column not in frame:
        raise ValueError(f"Label column not found: {label_column}")
    outer = float(outer_train_fraction)
    validation_inner = float(validation_fraction_of_outer_train)
    if not 0.0 < outer < 1.0 or not 0.0 < validation_inner < 1.0:
        raise ValueError("Split fractions must be in (0,1)")
    result = frame.copy()
    result["group_id"] = make_contiguous_group_ids(result, group_rows)
    group_size = result.groupby("group_id", sort=False).size()
    groups = list(group_size.index.astype(str))
    if len(groups) < 3:
        raise ValueError("At least three groups are required")
    target_fractions = {
        "train": outer * (1.0 - validation_inner),
        "validation": outer * validation_inner,
        "test": 1.0 - outer,
    }
    best_frame: pd.DataFrame | None = None
    best_score = float("inf")
    best_details: dict[str, Any] | None = None
    for attempt in range(int(candidate_attempts)):
        attempt_seed = int(seed) + attempt * 104729
        test_groups = _take_groups_to_row_target(
            group_size, groups, len(result) * target_fractions["test"], attempt_seed
        )
        outer_groups = [group for group in groups if group not in test_groups]
        outer_sizes = group_size.reindex(outer_groups)
        validation_groups = _take_groups_to_row_target(
            outer_sizes,
            outer_groups,
            len(result) * target_fractions["validation"],
            attempt_seed + 1,
        )
        assigned = result.copy()
        assigned["split"] = "train"
        assigned.loc[assigned["group_id"].isin(validation_groups), "split"] = "validation"
        assigned.loc[assigned["group_id"].isin(test_groups), "split"] = "test"
        score, details = _distribution_score(assigned, label_column, "split", target_fractions)
        if score < best_score:
            best_frame, best_score, best_details = assigned, score, details
    assert best_frame is not None and best_details is not None
    validation = validate_split_integrity(best_frame)
    group_label_counts = pd.crosstab(best_frame["group_id"], best_frame[label_column].astype("string"))
    labels_with_fewer_than_three_groups = sorted(
        str(label) for label in group_label_counts.columns if int((group_label_counts[label] > 0).sum()) < 3
    )
    report = {
        "outer_train_fraction": outer,
        "validation_fraction_of_outer_train": validation_inner,
        "effective_fractions": target_fractions,
        "group_rule": f"__source_file_id + floor(__source_row_id/{int(group_rows)})",
        "group_count": int(best_frame["group_id"].nunique()),
        "candidate_attempts": int(candidate_attempts),
        "selected_score": best_score,
        "distribution": best_details,
        "integrity": validation,
        "labels_present_in_fewer_than_three_groups": labels_with_fewer_than_three_groups,
        "priority_note": "Group integrity is authoritative; label stratification is best-effort.",
    }
    return SplitResult(best_frame, report)


def validate_split_integrity(frame: pd.DataFrame) -> dict[str, Any]:
    required = ["group_id", "split", "sample_id"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Missing split-integrity columns: {missing}")
    unknown = sorted(set(frame["split"]) - set(SPLIT_NAMES))
    if unknown:
        raise ValueError(f"Unknown split names: {unknown}")
    group_sets = {
        split: set(frame.loc[frame["split"] == split, "group_id"].astype(str)) for split in SPLIT_NAMES
    }
    sample_sets = {
        split: set(frame.loc[frame["split"] == split, "sample_id"].astype(str)) for split in SPLIT_NAMES
    }
    group_intersections = {
        "train_validation": len(group_sets["train"] & group_sets["validation"]),
        "train_test": len(group_sets["train"] & group_sets["test"]),
        "validation_test": len(group_sets["validation"] & group_sets["test"]),
    }
    sample_intersections = {
        "train_validation": len(sample_sets["train"] & sample_sets["validation"]),
        "train_test": len(sample_sets["train"] & sample_sets["test"]),
        "validation_test": len(sample_sets["validation"] & sample_sets["test"]),
    }
    if any(group_intersections.values()) or any(sample_intersections.values()):
        raise RuntimeError(
            f"Leakage detected: groups={group_intersections}, samples={sample_intersections}"
        )
    return {
        "group_intersections": group_intersections,
        "sample_intersections": sample_intersections,
        "sample_id_unique": not frame["sample_id"].duplicated().any(),
        "status": "passed",
    }


def write_split_artifacts(result: SplitResult, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "leakage_report.json"
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result.report, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(report_path)
    manifest_columns = [
        column for column in (
            "sample_id", "group_id", "split", "__capture_day", "__source_file_id", "__source_row_id"
        ) if column in result.frame
    ]
    result.frame[manifest_columns].to_csv(output / "split_manifest.csv", index=False)

