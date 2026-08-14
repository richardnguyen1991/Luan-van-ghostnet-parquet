from pathlib import Path

import pandas as pd

from src.data import DatasetPaths
from src.streaming import (
    assign_stream_split,
    build_group_manifest,
    epoch_schedule,
    manifest_summary,
    reservoir_groups,
    concatenate_bundles,
)
from src.graph_sequences import AlignedSplit, build_sequence_graph_tensors
import numpy as np


def fixture_paths(tmp_path: Path) -> DatasetPaths:
    root = tmp_path / "dataset"
    data = root / "data"
    manifests = root / "manifests"
    data.mkdir(parents=True)
    manifests.mkdir()
    paths = []
    for file_index in range(3):
        path = data / f"f{file_index}.parquet"
        rows = 96
        pd.DataFrame({
            "Label": ["A" if index % 2 else "B" for index in range(rows)],
            "Source IP": [f"10.0.{file_index}.{index % 7}" for index in range(rows)],
            "Destination IP": [f"10.1.{file_index}.{index % 5}" for index in range(rows)],
            "feature": range(rows),
            "__capture_day": "day",
            "__source_file_id": f"f{file_index}.csv",
            "__source_row_id": range(rows),
        }).to_parquet(path, row_group_size=24, index=False)
        paths.append(path)
    return DatasetPaths(root, data, manifests, tuple(paths))


def test_group_manifest_and_epoch_schedule_are_complete_and_deterministic(tmp_path: Path) -> None:
    paths = fixture_paths(tmp_path)
    groups = build_group_manifest(paths, group_rows=12, outer_train_fraction=0.8, seed=42)
    summary = manifest_summary(groups)
    assert summary["group_ids_unique"]
    assert sum(summary["row_counts_before_preprocessing"].values()) == 288
    first = epoch_schedule(groups, epoch=1, seed=42)
    second = epoch_schedule(groups, epoch=1, seed=42)
    assert [group.group_id for group in first] == [group.group_id for group in second]
    assert set(first) == {group for group in groups if group.split == "train"}
    assert [group.group_id for group in first] != [group.group_id for group in epoch_schedule(groups, 2, 42)]


def test_reservoir_scans_and_retains_whole_groups(tmp_path: Path) -> None:
    paths = fixture_paths(tmp_path)
    groups = build_group_manifest(paths, group_rows=12, outer_train_fraction=0.7, seed=8)
    retained = reservoir_groups(paths, groups, groups_per_file_and_split=1, seed=8)
    assert set(retained) == {"train", "validation", "test"}
    assert all(frames for frames in retained.values())
    for split, frames in retained.items():
        assert all(frame["split"].eq(split).all() for frame in frames)
        assert all(frame["group_id"].nunique() == 1 for frame in frames)


def test_split_assignment_supports_both_required_outer_fractions() -> None:
    for outer in (0.7, 0.8):
        values = [assign_stream_split(f"g-{index}", outer, 42) for index in range(1000)]
        assert set(values) == {"train", "validation", "test"}


def test_bundle_concatenation_preserves_window_pointers() -> None:
    config = {"step3": {
        "sequence_length": 4, "sequence_stride": 4,
        "require_consecutive_source_rows": True, "target_rule": "last_timestep",
        "source_endpoint_column": "Source IP", "destination_endpoint_column": "Destination IP",
        "endpoint_hash_namespace": "test", "missing_endpoint_policy": "drop_window",
        "graph_direction": "directed",
    }}
    rows = pd.DataFrame({
        "split": "train", "group_id": "g", "sample_id": [f"s{i}" for i in range(8)],
        "__source_file_id": "f", "__source_row_id": range(8),
        "Source IP": [f"a{i}" for i in range(8)], "Destination IP": [f"b{i}" for i in range(8)],
    })
    bundle = build_sequence_graph_tensors(
        AlignedSplit(rows, np.ones((8, 3), dtype=np.float32), np.zeros(8, dtype=np.int64)),
        "train", config,
    )
    merged = concatenate_bundles([bundle, bundle])
    assert len(merged.sequence_x) == 4
    assert len(merged.edge_window_ptr) == 5
    assert len(merged.node_window_ptr) == 5
    assert int(merged.edge_window_ptr[-1]) == merged.edge_index.shape[1]
