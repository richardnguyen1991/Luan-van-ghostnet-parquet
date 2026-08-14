from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import sample_contiguous_parquet_file
from src.graph_sequences import (
    AlignedSplit,
    align_preprocessed_split,
    build_sequence_graph_tensors,
    save_sequence_graph_tensors,
    validate_sequence_leakage,
)
from src.preprocessing import PreprocessedSplits


def step3_config(sequence_length: int = 4, stride: int = 2) -> dict:
    return {
        "step3": {
            "sequence_length": sequence_length,
            "sequence_stride": stride,
            "require_consecutive_source_rows": True,
            "target_rule": "last_timestep",
            "source_endpoint_column": "Source IP",
            "destination_endpoint_column": "Destination IP",
            "endpoint_hash_namespace": "unit-test",
            "missing_endpoint_policy": "drop_window",
            "graph_direction": "directed",
        }
    }


def aligned_split(split: str, row_ids: list[int], group_id: str | None = None) -> AlignedSplit:
    group = group_id or f"{split}.csv::block=0"
    rows = pd.DataFrame({
        "split": split,
        "group_id": group,
        "sample_id": [f"{split}:{row_id}" for row_id in row_ids],
        "__source_file_id": f"{split}.csv",
        "__source_row_id": row_ids,
        "Source IP": [f"10.0.0.{row_id % 3 + 1}" for row_id in row_ids],
        "Destination IP": [f"10.0.1.{row_id % 2 + 1}" for row_id in row_ids],
    })
    x = np.arange(len(rows) * 3, dtype=np.float32).reshape(len(rows), 3)
    y = np.asarray([row_id % 2 for row_id in row_ids], dtype=np.int64)
    return AlignedSplit(rows, x, y)


def test_builds_strict_sequences_and_window_local_graphs(tmp_path: Path):
    bundle = build_sequence_graph_tensors(
        aligned_split("train", list(range(8))), "train", step3_config()
    )
    assert bundle.sequence_x.shape == (3, 4, 3)
    assert bundle.sequence_y.shape == (3, 4)
    np.testing.assert_array_equal(bundle.target_y, bundle.sequence_y[:, -1])
    assert bundle.edge_index.shape == (2, 12)
    np.testing.assert_array_equal(np.diff(bundle.edge_window_ptr), np.full(3, 4))
    assert np.all(np.diff(bundle.source_row_ids, axis=1) == 1)
    assert all(len(node_hash) == 64 for node_hash in bundle.node_hashes)
    assert not any("10.0." in node_hash for node_hash in bundle.node_hashes)
    save_sequence_graph_tensors(bundle, tmp_path)
    assert (tmp_path / "sequence_tensors.npz").exists()
    assert (tmp_path / "graph_tensors.npz").exists()
    assert (tmp_path / "sequence_graph_manifest.json").exists()


def test_source_row_gaps_are_not_bridged():
    bundle = build_sequence_graph_tensors(
        aligned_split("train", [0, 1, 2, 4, 5, 6]),
        "train",
        step3_config(sequence_length=3, stride=1),
    )
    assert bundle.sequence_x.shape[0] == 2
    assert bundle.source_row_ids.tolist() == [[0, 1, 2], [4, 5, 6]]


def test_cross_split_sequence_and_group_leakage_is_empty():
    tensors = {
        split: build_sequence_graph_tensors(
            aligned_split(split, list(range(6))), split, step3_config(sequence_length=3, stride=3)
        )
        for split in ("train", "validation", "test")
    }
    report = validate_sequence_leakage(tensors)
    assert report["status"] == "passed"
    assert all(value == 0 for value in report["sample_id_intersections"].values())
    assert all(value == 0 for value in report["group_id_intersections"].values())


def test_train_alignment_applies_outlier_mask_without_reordering():
    rows = aligned_split("train", [0, 1, 2, 3]).rows
    processed = PreprocessedSplits(
        train_x=np.asarray([[0.0], [2.0], [3.0]]),
        validation_x=np.empty((0, 1)),
        test_x=np.empty((0, 1)),
        train_y=np.asarray([0, 0, 1]),
        validation_y=np.empty(0, dtype=np.int64),
        test_y=np.empty(0, dtype=np.int64),
        train_inlier_mask=np.asarray([True, False, True, True]),
        metadata={},
    )
    aligned = align_preprocessed_split(rows, processed, "train")
    assert aligned.rows["__source_row_id"].tolist() == [0, 2, 3]
    np.testing.assert_array_equal(aligned.x[:, 0], [0.0, 2.0, 3.0])


def test_parquet_sequence_sampler_returns_real_contiguous_runs(tmp_path: Path):
    path = tmp_path / "source.parquet"
    pd.DataFrame({
        "__source_file_id": ["source.csv"] * 1000,
        "__source_row_id": np.arange(1000),
        "Label": ["A"] * 1000,
    }).to_parquet(path, row_group_size=200, index=False)
    sample = sample_contiguous_parquet_file(
        path, samples=160, seed=42, minimum_run_rows=16, maximum_row_groups=4
    )
    assert len(sample) == 160
    breaks = int((sample["__source_row_id"].diff().fillna(1) != 1).sum())
    assert breaks <= 3

