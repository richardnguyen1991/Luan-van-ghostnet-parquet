from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .data import DatasetPaths, canonicalize_frame, stable_sample_ids
from .graph_sequences import AlignedSplit, SequenceGraphTensors, build_sequence_graph_tensors
from .preprocessing import LeakageSafePreprocessor


@dataclass(frozen=True)
class GroupRef:
    file_index: int
    relative_file: str
    row_group: int
    offset: int
    rows: int
    group_id: str
    split: str


def _unit_interval(key: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def assign_stream_split(group_id: str, outer_train_fraction: float, seed: int) -> str:
    outer = float(outer_train_fraction)
    if outer not in {0.7, 0.8}:
        raise ValueError("outer_train_fraction must be 0.70 or 0.80")
    value = _unit_interval(group_id, seed)
    test_fraction = 1.0 - outer
    validation_fraction = outer * 0.10
    if value < test_fraction:
        return "test"
    if value < test_fraction + validation_fraction:
        return "validation"
    return "train"


def build_group_manifest(
    paths: DatasetPaths,
    group_rows: int,
    outer_train_fraction: float,
    seed: int,
) -> list[GroupRef]:
    if int(group_rows) <= 0:
        raise ValueError("group_rows must be positive")
    groups: list[GroupRef] = []
    for file_index, path in enumerate(paths.parquet_files):
        parquet = pq.ParquetFile(path)
        relative = str(path.relative_to(paths.root)).replace("\\", "/")
        for row_group in range(parquet.num_row_groups):
            row_count = int(parquet.metadata.row_group(row_group).num_rows)
            for offset in range(0, row_count, int(group_rows)):
                rows = min(int(group_rows), row_count - offset)
                group_id = f"{relative}::rg={row_group}::offset={offset}::rows={rows}"
                groups.append(GroupRef(
                    file_index=file_index,
                    relative_file=relative,
                    row_group=row_group,
                    offset=offset,
                    rows=rows,
                    group_id=group_id,
                    split=assign_stream_split(group_id, outer_train_fraction, seed),
                ))
    if not groups or {group.split for group in groups} != {"train", "validation", "test"}:
        raise RuntimeError("Full group manifest must contain all three splits")
    return groups


def epoch_schedule(groups: list[GroupRef], epoch: int, seed: int) -> list[GroupRef]:
    train = [group for group in groups if group.split == "train"]
    random.Random(int(seed) + int(epoch)).shuffle(train)
    if len({group.group_id for group in train}) != len(train):
        raise RuntimeError("Duplicate group in global epoch schedule")
    return train


def read_group(paths: DatasetPaths, group: GroupRef) -> pd.DataFrame:
    path = paths.parquet_files[group.file_index]
    table = pq.ParquetFile(path).read_row_group(group.row_group).slice(group.offset, group.rows)
    frame = canonicalize_frame(table.to_pandas())
    frame["sample_id"] = stable_sample_ids(frame)
    frame["group_id"] = group.group_id
    frame["split"] = group.split
    return frame


def reservoir_groups(
    paths: DatasetPaths,
    groups: list[GroupRef],
    groups_per_file_and_split: int,
    seed: int,
) -> dict[str, list[pd.DataFrame]]:
    """Scan every group; retain bounded whole groups via reservoir replacement."""
    cap = int(groups_per_file_and_split)
    if cap <= 0:
        raise ValueError("groups_per_file_and_split must be positive")
    retained: dict[tuple[int, str], list[GroupRef]] = {}
    seen: dict[tuple[int, str], int] = {}
    rng = random.Random(int(seed))
    for group in groups:
        key = (group.file_index, group.split)
        bucket = retained.setdefault(key, [])
        seen[key] = seen.get(key, 0) + 1
        if len(bucket) < cap:
            bucket.append(group)
        else:
            replacement = rng.randrange(seen[key])
            if replacement < cap:
                bucket[replacement] = group
    result = {"train": [], "validation": [], "test": []}
    for key in sorted(retained):
        for group in retained[key]:
            result[group.split].append(read_group(paths, group))
    if any(not frames for frames in result.values()):
        raise RuntimeError("Reservoir scan did not retain every split")
    return result


def manifest_summary(groups: list[GroupRef]) -> dict:
    counts = {split: 0 for split in ("train", "validation", "test")}
    rows = {split: 0 for split in counts}
    files = {split: set() for split in counts}
    for group in groups:
        counts[group.split] += 1
        rows[group.split] += group.rows
        files[group.split].add(group.relative_file)
    return {
        "mode": "full_mixed_group_streaming",
        "group_counts": counts,
        "row_counts_before_preprocessing": rows,
        "source_file_counts": {key: len(value) for key, value in files.items()},
        "group_ids_unique": len({group.group_id for group in groups}) == len(groups),
    }


def fit_streaming_proxy(
    reservoirs: dict[str, list[pd.DataFrame]],
    label_column: str,
    config: dict,
    seed: int,
) -> tuple[LeakageSafePreprocessor, dict]:
    frame = pd.concat(
        [item for split in ("train", "validation", "test") for item in reservoirs[split]],
        ignore_index=True,
    )
    processor = LeakageSafePreprocessor(config, seed)
    processed = processor.fit_transform_splits(frame, label_column)
    metadata = dict(processed.metadata)
    metadata.update({
        "streaming_fit_policy": "bounded_whole_group_reservoir_scanned_across_all_files",
        "streaming_fit_scope": "train_groups_only",
        "streaming_proxy_rows": {split: sum(len(item) for item in reservoirs[split]) for split in reservoirs},
    })
    return processor, metadata


def transform_group(
    frame: pd.DataFrame,
    split: str,
    label_column: str,
    processor: LeakageSafePreprocessor,
    config: dict,
) -> SequenceGraphTensors | None:
    numeric = processor._convert_numeric(frame, processor.feature_columns)
    assert processor.imputer is not None and processor.scaler is not None
    values = processor.imputer.transform(numeric.to_numpy())
    mask = np.ones(len(frame), dtype=bool)
    if split == "train" and processor.isolation_forest is not None:
        mask = processor.isolation_forest.predict(values) == 1
    rows = frame.loc[mask].reset_index(drop=True)
    if len(rows) < int(config["step3"]["sequence_length"]):
        return None
    scaled = processor.scaler.transform(values[mask]).astype(np.float32, copy=False)
    if config["preprocessing"]["minmax"]["clip"]:
        low, high = config["preprocessing"]["minmax"]["feature_range"]
        np.clip(scaled, float(low), float(high), out=scaled)
    labels = rows[label_column].astype("string").map(processor.label_mapping)
    if labels.isna().any():
        raise RuntimeError("Streaming group contains a label absent from fitted mapping")
    aligned = AlignedSplit(rows, scaled, labels.to_numpy(dtype=np.int64))
    try:
        return build_sequence_graph_tensors(aligned, split, config)
    except RuntimeError as exc:
        if "No valid" in str(exc):
            return None
        raise


def concatenate_bundles(bundles: list[SequenceGraphTensors]) -> SequenceGraphTensors:
    if not bundles:
        raise ValueError("Cannot concatenate an empty bundle list")
    split = bundles[0].split
    if any(bundle.split != split for bundle in bundles):
        raise ValueError("All bundles must have the same split")
    node_lengths = [np.diff(bundle.node_window_ptr) for bundle in bundles]
    edge_lengths = [np.diff(bundle.edge_window_ptr) for bundle in bundles]
    node_ptr = np.concatenate(([0], np.cumsum(np.concatenate(node_lengths)))).astype(np.int64)
    edge_ptr = np.concatenate(([0], np.cumsum(np.concatenate(edge_lengths)))).astype(np.int64)
    return SequenceGraphTensors(
        split=split,
        sequence_x=np.concatenate([bundle.sequence_x for bundle in bundles]),
        sequence_y=np.concatenate([bundle.sequence_y for bundle in bundles]),
        target_y=np.concatenate([bundle.target_y for bundle in bundles]),
        sample_ids=np.concatenate([bundle.sample_ids for bundle in bundles]),
        source_row_ids=np.concatenate([bundle.source_row_ids for bundle in bundles]),
        group_ids=np.concatenate([bundle.group_ids for bundle in bundles]),
        node_hashes=np.concatenate([bundle.node_hashes for bundle in bundles]),
        node_window_ptr=node_ptr,
        edge_index=np.concatenate([bundle.edge_index for bundle in bundles], axis=1),
        edge_window_ptr=edge_ptr,
        edge_time=np.concatenate([bundle.edge_time for bundle in bundles]),
        metadata={
            "split": split,
            "sequence_count": sum(len(bundle.sequence_x) for bundle in bundles),
            "source_bundle_count": len(bundles),
            "streaming_concatenation": True,
        },
    )
