from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import write_json
from .preprocessing import PreprocessedSplits


SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class AlignedSplit:
    rows: pd.DataFrame
    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class SequenceGraphTensors:
    split: str
    sequence_x: np.ndarray
    sequence_y: np.ndarray
    target_y: np.ndarray
    sample_ids: np.ndarray
    source_row_ids: np.ndarray
    group_ids: np.ndarray
    node_hashes: np.ndarray
    node_window_ptr: np.ndarray
    edge_index: np.ndarray
    edge_window_ptr: np.ndarray
    edge_time: np.ndarray
    metadata: dict[str, Any]


def align_preprocessed_split(
    frame: pd.DataFrame,
    processed: PreprocessedSplits,
    split: str,
) -> AlignedSplit:
    if split not in SPLIT_NAMES:
        raise ValueError(f"Unknown split: {split}")
    rows = frame.loc[frame["split"] == split].copy().reset_index(drop=True)
    x = np.asarray(getattr(processed, f"{split}_x"))
    y = np.asarray(getattr(processed, f"{split}_y"))
    if split == "train":
        mask = np.asarray(processed.train_inlier_mask, dtype=bool)
        if len(mask) != len(rows):
            raise ValueError("train_inlier_mask is not aligned with training rows")
        rows = rows.loc[mask].reset_index(drop=True)
    if len(rows) != len(x) or len(rows) != len(y):
        raise ValueError(
            f"Preprocessed arrays are not aligned for {split}: rows={len(rows)}, x={len(x)}, y={len(y)}"
        )
    return AlignedSplit(rows=rows, x=x, y=y)


def _endpoint_hash(value: str, namespace: str) -> str:
    return hashlib.sha256(f"{namespace}\0{value}".encode("utf-8")).hexdigest()


def _valid_endpoint(value: Any) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip()
    return bool(text) and text.lower() not in {"nan", "none", "<na>"}


def _contiguous_runs(source_rows: np.ndarray, require_consecutive: bool) -> list[tuple[int, int]]:
    if len(source_rows) == 0:
        return []
    if not require_consecutive:
        return [(0, len(source_rows))]
    breaks = np.flatnonzero(np.diff(source_rows) != 1) + 1
    boundaries = np.concatenate(([0], breaks, [len(source_rows)]))
    return [(int(start), int(stop)) for start, stop in zip(boundaries[:-1], boundaries[1:])]


def build_sequence_graph_tensors(
    aligned: AlignedSplit,
    split: str,
    config: dict[str, Any],
) -> SequenceGraphTensors:
    if split not in SPLIT_NAMES:
        raise ValueError(f"Unknown split: {split}")
    step3 = config["step3"]
    length = int(step3["sequence_length"])
    stride = int(step3["sequence_stride"])
    source_column = step3["source_endpoint_column"]
    destination_column = step3["destination_endpoint_column"]
    namespace = step3["endpoint_hash_namespace"]
    required = {
        "split", "group_id", "sample_id", "__source_file_id", "__source_row_id",
        source_column, destination_column,
    }
    missing = sorted(required - set(aligned.rows.columns))
    if missing:
        raise ValueError(f"Cannot create Step 3 tensors; missing columns: {missing}")
    if set(aligned.rows["split"].astype(str)) != {split}:
        raise ValueError(f"Aligned rows contain data outside split={split}")

    sequence_x: list[np.ndarray] = []
    sequence_y: list[np.ndarray] = []
    target_y: list[int] = []
    sample_ids: list[np.ndarray] = []
    source_row_ids: list[np.ndarray] = []
    group_ids: list[str] = []
    node_hashes: list[str] = []
    node_window_ptr = [0]
    edge_sources: list[int] = []
    edge_destinations: list[int] = []
    edge_window_ptr = [0]
    edge_time: list[int] = []
    dropped_missing_endpoint_windows = 0
    short_contiguous_runs = 0

    ordered = aligned.rows.assign(__array_position=np.arange(len(aligned.rows), dtype=np.int64))
    ordered = ordered.sort_values(
        ["__source_file_id", "group_id", "__source_row_id"], kind="stable"
    )
    for group_id, group in ordered.groupby("group_id", sort=False):
        if group["split"].nunique() != 1:
            raise RuntimeError(f"Group crosses split boundary: {group_id}")
        positions = group["__array_position"].to_numpy(dtype=np.int64)
        group_source_rows = pd.to_numeric(group["__source_row_id"], errors="raise").to_numpy(dtype=np.int64)
        if len(np.unique(group_source_rows)) != len(group_source_rows):
            raise RuntimeError(f"Duplicate source row inside group: {group_id}")
        for run_start, run_stop in _contiguous_runs(
            group_source_rows,
            bool(step3["require_consecutive_source_rows"]),
        ):
            run_size = run_stop - run_start
            if run_size < length:
                short_contiguous_runs += 1
                continue
            for window_start in range(run_start, run_stop - length + 1, stride):
                window_stop = window_start + length
                window_rows = group.iloc[window_start:window_stop]
                source_values = window_rows[source_column].tolist()
                destination_values = window_rows[destination_column].tolist()
                if not all(_valid_endpoint(value) for value in source_values + destination_values):
                    dropped_missing_endpoint_windows += 1
                    continue
                window_positions = positions[window_start:window_stop]
                window_x = np.asarray(aligned.x[window_positions], dtype=np.float32)
                window_y = np.asarray(aligned.y[window_positions], dtype=np.int64)
                window_source_rows = group_source_rows[window_start:window_stop]
                if bool(step3["require_consecutive_source_rows"]) and not np.all(
                    np.diff(window_source_rows) == 1
                ):
                    raise RuntimeError("A sequence window crossed a source-row gap")

                sources = [_endpoint_hash(str(value).strip(), namespace) for value in source_values]
                destinations = [_endpoint_hash(str(value).strip(), namespace) for value in destination_values]
                local_nodes = sorted(set(sources + destinations))
                local_index = {node_hash: index for index, node_hash in enumerate(local_nodes)}
                edge_sources.extend(local_index[value] for value in sources)
                edge_destinations.extend(local_index[value] for value in destinations)
                edge_time.extend(range(length))
                node_hashes.extend(local_nodes)
                node_window_ptr.append(len(node_hashes))
                edge_window_ptr.append(len(edge_sources))

                sequence_x.append(window_x)
                sequence_y.append(window_y)
                target_y.append(int(window_y[-1]))
                sample_ids.append(window_rows["sample_id"].astype(str).to_numpy(dtype=str))
                source_row_ids.append(window_source_rows.copy())
                group_ids.append(str(group_id))

    if not sequence_x:
        raise RuntimeError(
            f"No valid {split} sequences were created; length={length}, "
            f"short_runs={short_contiguous_runs}, missing_endpoint_windows={dropped_missing_endpoint_windows}"
        )
    sequence_x_array = np.stack(sequence_x).astype(np.float32, copy=False)
    sequence_y_array = np.stack(sequence_y).astype(np.int64, copy=False)
    sample_ids_array = np.stack(sample_ids).astype(str, copy=False)
    source_row_ids_array = np.stack(source_row_ids).astype(np.int64, copy=False)
    edge_index = np.vstack((edge_sources, edge_destinations)).astype(np.int64, copy=False)
    metadata = {
        "split": split,
        "sequence_count": len(sequence_x_array),
        "sequence_length": length,
        "sequence_stride": stride,
        "feature_count": int(sequence_x_array.shape[2]),
        "target_rule": "last_timestep",
        "require_consecutive_source_rows": bool(step3["require_consecutive_source_rows"]),
        "graph_direction": "directed",
        "edge_semantics": f"{source_column} -> {destination_column} per flow timestep",
        "node_identity": "sha256(namespace + NUL + endpoint); raw endpoints are not written",
        "node_count_with_window_local_repetition": len(node_hashes),
        "edge_count": edge_index.shape[1],
        "dropped_missing_endpoint_windows": dropped_missing_endpoint_windows,
        "short_contiguous_runs": short_contiguous_runs,
        "group_count_with_sequences": len(set(group_ids)),
    }
    return SequenceGraphTensors(
        split=split,
        sequence_x=sequence_x_array,
        sequence_y=sequence_y_array,
        target_y=np.asarray(target_y, dtype=np.int64),
        sample_ids=sample_ids_array,
        source_row_ids=source_row_ids_array,
        group_ids=np.asarray(group_ids, dtype=str),
        node_hashes=np.asarray(node_hashes, dtype=str),
        node_window_ptr=np.asarray(node_window_ptr, dtype=np.int64),
        edge_index=edge_index,
        edge_window_ptr=np.asarray(edge_window_ptr, dtype=np.int64),
        edge_time=np.asarray(edge_time, dtype=np.int64),
        metadata=metadata,
    )


def validate_sequence_leakage(tensors: dict[str, SequenceGraphTensors]) -> dict[str, Any]:
    if set(tensors) != set(SPLIT_NAMES):
        raise ValueError(f"Expected tensors for {SPLIT_NAMES}, found {sorted(tensors)}")
    sample_sets = {
        split: set(bundle.sample_ids.reshape(-1).tolist()) for split, bundle in tensors.items()
    }
    group_sets = {split: set(bundle.group_ids.tolist()) for split, bundle in tensors.items()}
    sample_intersections: dict[str, int] = {}
    group_intersections: dict[str, int] = {}
    failures: list[str] = []
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        key = f"{left}__{right}"
        sample_intersections[key] = len(sample_sets[left] & sample_sets[right])
        group_intersections[key] = len(group_sets[left] & group_sets[right])
        if sample_intersections[key] or group_intersections[key]:
            failures.append(key)
    for split, bundle in tensors.items():
        if np.any(np.diff(bundle.source_row_ids, axis=1) != 1):
            failures.append(f"{split}__non_contiguous_window")
        if len(bundle.edge_window_ptr) != len(bundle.sequence_x) + 1:
            failures.append(f"{split}__edge_pointer_length")
        if len(bundle.node_window_ptr) != len(bundle.sequence_x) + 1:
            failures.append(f"{split}__node_pointer_length")
        for window_index in range(len(bundle.sequence_x)):
            edge_start, edge_stop = bundle.edge_window_ptr[window_index : window_index + 2]
            node_start, node_stop = bundle.node_window_ptr[window_index : window_index + 2]
            local_node_count = int(node_stop - node_start)
            local_edges = bundle.edge_index[:, edge_start:edge_stop]
            if local_edges.size and (
                int(local_edges.min()) < 0 or int(local_edges.max()) >= local_node_count
            ):
                failures.append(f"{split}__edge_index_out_of_range__window_{window_index}")
                break
    return {
        "status": "passed" if not failures else "failed",
        "sample_id_intersections": sample_intersections,
        "group_id_intersections": group_intersections,
        "failures": failures,
    }


def save_sequence_graph_tensors(bundle: SequenceGraphTensors, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "sequence_tensors.npz",
        sequence_x=bundle.sequence_x,
        sequence_y=bundle.sequence_y,
        target_y=bundle.target_y,
        sample_ids=bundle.sample_ids,
        source_row_ids=bundle.source_row_ids,
        group_ids=bundle.group_ids,
    )
    np.savez_compressed(
        output / "graph_tensors.npz",
        node_hashes=bundle.node_hashes,
        node_window_ptr=bundle.node_window_ptr,
        edge_index=bundle.edge_index,
        edge_window_ptr=bundle.edge_window_ptr,
        edge_time=bundle.edge_time,
    )
    write_json(output / "sequence_graph_manifest.json", bundle.metadata)


def tensor_contract_hash(bundle: SequenceGraphTensors) -> str:
    contract = {
        "split": bundle.split,
        "sequence_shape": list(bundle.sequence_x.shape),
        "sequence_dtype": str(bundle.sequence_x.dtype),
        "target_dtype": str(bundle.target_y.dtype),
        "edge_shape": list(bundle.edge_index.shape),
        "metadata": bundle.metadata,
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

