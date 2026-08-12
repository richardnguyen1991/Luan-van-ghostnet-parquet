from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import config_hash, load_config
from .data import audit_contiguous_sequence_dataset, write_json
from .graph_sequences import (
    SPLIT_NAMES,
    align_preprocessed_split,
    build_sequence_graph_tensors,
    save_sequence_graph_tensors,
    tensor_contract_hash,
    validate_sequence_leakage,
)
from .preprocessing import LeakageSafePreprocessor
from .splits import assign_group_splits, write_split_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Step 3 graph/sequence sampled smoke test")
    parser.add_argument("--data-dir", default="/kaggle/input")
    parser.add_argument(
        "--output-dir",
        default="/kaggle/working/Luan-Van-GC-LSTM-GhostNet-CICDDoS2019-v1/outputs/step3_smoke",
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--mode-config", default="configs/practical_baseline.yaml")
    parser.add_argument("--samples-per-file", type=int, default=2048)
    parser.add_argument("--sequence-group-rows", type=int)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--sequence-stride", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--target-column")
    parser.add_argument("--full-dataset", action="store_true")
    parser.add_argument("--stream-files", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.full_dataset or args.stream_files:
        raise ValueError(
            "Step 3 currently validates sampled contiguous materialization only. "
            "Full mixed-group streaming is implemented with the training loop in Step 4."
        )
    config = load_config(args.config, args.mode_config)
    config["data"]["samples_per_file"] = int(args.samples_per_file)
    if args.sequence_group_rows is not None:
        config["data"]["sequence_group_rows"] = int(args.sequence_group_rows)
    if args.sequence_length is not None:
        config["step3"]["sequence_length"] = int(args.sequence_length)
    if args.sequence_stride is not None:
        config["step3"]["sequence_stride"] = int(args.sequence_stride)
    if args.seed is not None:
        config["project"]["seed"] = int(args.seed)
    if args.target_column is not None:
        config["data"]["label_candidates"] = [args.target_column]

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sample, profile, sample_manifest = audit_contiguous_sequence_dataset(
        args.data_dir, output / "audit", config
    )
    label_column = profile["label_column"]
    run_summaries = []
    for outer in config["splits"]["outer_train_fractions"]:
        split_name = f"split_{int(round(float(outer) * 100))}"
        split_output = output / split_name
        split_result = assign_group_splits(
            sample,
            label_column=label_column,
            outer_train_fraction=float(outer),
            validation_fraction_of_outer_train=float(
                config["splits"]["validation_fraction_of_outer_train"]
            ),
            group_rows=int(config["data"]["sequence_group_rows"]),
            seed=int(config["project"]["seed"]),
            candidate_attempts=int(config["splits"]["candidate_attempts"]),
        )
        write_split_artifacts(split_result, split_output)
        preprocessor = LeakageSafePreprocessor(config, int(config["project"]["seed"]))
        processed = preprocessor.fit_transform_splits(split_result.frame, label_column)
        preprocessor.save(split_output, processed.metadata)

        bundles = {}
        for split in SPLIT_NAMES:
            aligned = align_preprocessed_split(split_result.frame, processed, split)
            bundle = build_sequence_graph_tensors(aligned, split, config)
            save_sequence_graph_tensors(bundle, split_output / split)
            bundles[split] = bundle
        leakage = validate_sequence_leakage(bundles)
        if leakage["status"] != "passed":
            raise RuntimeError(f"Step 3 leakage validation failed: {leakage}")
        write_json(split_output / "sequence_leakage_report.json", leakage)
        summary = {
            "outer_train_fraction": float(outer),
            "mode": config["project"]["mode"],
            "sequence_length": int(config["step3"]["sequence_length"]),
            "sequence_stride": int(config["step3"]["sequence_stride"]),
            "feature_count": int(processed.metadata["feature_count"]),
            "sequence_counts": {
                split: int(bundle.metadata["sequence_count"]) for split, bundle in bundles.items()
            },
            "edge_counts": {
                split: int(bundle.metadata["edge_count"]) for split, bundle in bundles.items()
            },
            "tensor_contract_hashes": {
                split: tensor_contract_hash(bundle) for split, bundle in bundles.items()
            },
            "row_split_leakage_status": split_result.report["integrity"]["status"],
            "sequence_leakage_status": leakage["status"],
        }
        write_json(split_output / "step3_summary.json", summary)
        run_summaries.append(summary)

    write_json(output / "run_config.json", {
        "config": config,
        "config_hash": config_hash(config),
        "sample_manifest": sample_manifest,
        "paper_scope_note": (
            "The paper describes GCN node relationships and temporal LSTM processing but does not "
            "publish a graph-construction rule, sequence length, or stride. Configured values are "
            "operational assumptions."
        ),
    })
    final = {
        "status": "passed",
        "dataset_rows_in_manifest": profile["manifest_total_rows"],
        "sample_rows": profile["sample_rows"],
        "runs": run_summaries,
    }
    write_json(output / "step3_summary.json", final)
    print(json.dumps(final, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

