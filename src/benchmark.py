from __future__ import annotations

import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any

import psutil
import torch

from .model import GCLSTMGhostNet, GraphBatch


def benchmark_model(
    model: GCLSTMGhostNet,
    batch: GraphBatch,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    warmups: int = 50,
    measurements: int = 500,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.cpu().eval()
    batch = batch.to("cpu")
    with torch.inference_mode():
        for _ in range(int(warmups)):
            model(batch)
        forward_ms = []
        for _ in range(int(measurements)):
            started = time.perf_counter_ns()
            model(batch)
            forward_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
    ordered = sorted(forward_ms)
    percentile = lambda q: ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]
    checkpoint = Path(checkpoint_path)
    result = {
        "device": "cpu",
        "batch_size": int(len(batch.target_y)),
        "warmups": int(warmups),
        "measurements": int(measurements),
        "forward_latency_ms_p50": percentile(0.50),
        "forward_latency_ms_p95": percentile(0.95),
        "forward_latency_ms_mean": statistics.fmean(forward_ms),
        "throughput_samples_per_second": len(batch.target_y) * 1000.0 / statistics.fmean(forward_ms),
        "model_size_mb": checkpoint.stat().st_size / (1024 ** 2),
        "peak_rss_mb": psutil.Process().memory_info().rss / (1024 ** 2),
        "t_preprocess_ms": None,
        "t_graph_sequence_ms": None,
        "t_forward_ms_p50": percentile(0.50),
        "t_total_ms_p50": percentile(0.50),
        "paper_12_5ms_comparison": "not_directly_comparable_preprocessed_single_sequence_protocol",
    }
    (output / "benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (output / "benchmark_raw.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["measurement", "forward_ms"])
        writer.writerows(enumerate(forward_ms, start=1))
    return result
