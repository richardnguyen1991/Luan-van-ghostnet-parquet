from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .graph_sequences import SequenceGraphTensors
from .model import GCLSTMGhostNet, GraphBatch
from .training import GraphSequenceDataset, collate_graph_sequences


def _batch_for_index(bundle: SequenceGraphTensors, index: int) -> GraphBatch:
    return collate_graph_sequences([GraphSequenceDataset(bundle)[index]])


def generate_explainability_artifacts(
    model: GCLSTMGhostNet,
    bundle: SequenceGraphTensors,
    preprocessing_metadata: dict[str, Any],
    output_dir: str | Path,
    sample_count: int = 8,
    integrated_gradient_steps: int = 16,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.cpu().eval()
    temporal_rows: list[list[Any]] = []
    spatial_rows: list[list[Any]] = []
    importance = np.zeros(len(preprocessing_metadata["feature_order"]), dtype=np.float64)
    selected = list(range(min(int(sample_count), len(bundle.sequence_x))))
    for sample_index in selected:
        original = _batch_for_index(bundle, sample_index)
        with torch.no_grad():
            logits, temporal = model(original)
            predicted = int(logits.argmax(dim=1)[0])
            spatial = tuple(weights.detach().cpu().numpy() for weights in model.last_spatial_attention)
        for timestep, weight in enumerate(temporal[0].detach().cpu().tolist()):
            temporal_rows.append([sample_index, timestep, weight])
        node_start, node_stop = bundle.node_window_ptr[sample_index : sample_index + 2]
        hashes = bundle.node_hashes[node_start:node_stop]
        for node_index, (node_hash, weight) in enumerate(zip(hashes, spatial[0])):
            spatial_rows.append([sample_index, node_index, str(node_hash), float(weight)])

        baseline = torch.zeros_like(original.sequence_x)
        accumulated = torch.zeros_like(original.sequence_x)
        for alpha in torch.linspace(0.0, 1.0, int(integrated_gradient_steps)):
            interpolated = (baseline + alpha * (original.sequence_x - baseline)).requires_grad_(True)
            batch = GraphBatch(interpolated, original.target_y, original.edge_index, original.node_counts)
            score = model(batch)[0][0, predicted]
            gradient = torch.autograd.grad(score, interpolated)[0]
            accumulated += gradient.detach()
        integrated = (original.sequence_x - baseline) * accumulated / float(integrated_gradient_steps)
        importance += integrated.abs().mean(dim=(0, 1)).detach().cpu().numpy()

    importance /= max(1, len(selected))
    with (output / "temporal_attention.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "timestep", "weight"])
        writer.writerows(temporal_rows)
    with (output / "spatial_attention.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "node_index", "node_hash", "weight"])
        writer.writerows(spatial_rows)
    with (output / "feature_importance.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature", "integrated_gradients_mean_abs"])
        writer.writerows(zip(preprocessing_metadata["feature_order"], importance.tolist()))
    manifest = {
        "method": "integrated_gradients",
        "baseline": "all_zero_in_scaled_feature_space",
        "steps": int(integrated_gradient_steps),
        "sample_indices": selected,
        "sample_ids": [bundle.sample_ids[index].astype(str).tolist() for index in selected],
        "raw_network_identifiers_persisted": False,
    }
    (output / "explain_sample_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest
