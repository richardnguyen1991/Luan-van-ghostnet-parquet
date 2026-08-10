from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import config_hash, load_config
from .data import audit_sampled_dataset, write_json
from .preprocessing import LeakageSafePreprocessor
from .splits import assign_group_splits, write_split_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Step 2 sampled schema/leakage/preprocessing smoke test")
    parser.add_argument("--data-dir", default="/kaggle/input")
    parser.add_argument(
        "--output-dir",
        default="/kaggle/working/Luan-Van-GC-LSTM-GhostNet-CICDDoS2019-v1/outputs/step2_smoke",
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--mode-config", default="configs/practical_baseline.yaml")
    parser.add_argument("--samples-per-file", type=int, default=2048)
    parser.add_argument("--sequence-group-rows", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--target-column")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.mode_config)
    config["data"]["samples_per_file"] = int(args.samples_per_file)
    if args.sequence_group_rows is not None:
        config["data"]["sequence_group_rows"] = int(args.sequence_group_rows)
    if args.seed is not None:
        config["project"]["seed"] = int(args.seed)
    if args.target_column is not None:
        config["data"]["label_candidates"] = [args.target_column]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sample, profile, sample_manifest = audit_sampled_dataset(args.data_dir, output / "audit", config)
    label_column = profile["label_column"]
    summaries = []
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
        summary = {
            "outer_train_fraction": float(outer),
            "mode": config["project"]["mode"],
            "train_shape": list(processed.train_x.shape),
            "validation_shape": list(processed.validation_x.shape),
            "test_shape": list(processed.test_x.shape),
            "feature_count": processed.metadata["feature_count"],
            "train_rows_removed_as_outliers": (
                processed.metadata["train_rows_before_outlier_removal"]
                - processed.metadata["train_rows_after_outlier_removal"]
            ),
            "leakage_status": split_result.report["integrity"]["status"],
            "labels_missing_from_train": processed.metadata["labels_missing_from_train"],
            "labels_missing_from_validation": processed.metadata["labels_missing_from_validation"],
            "labels_missing_from_test": processed.metadata["labels_missing_from_test"],
        }
        write_json(split_output / "smoke_summary.json", summary)
        summaries.append(summary)
    write_json(output / "run_config.json", {
        "config": config,
        "config_hash": config_hash(config),
        "sample_manifest": sample_manifest,
    })
    write_json(output / "smoke_summary.json", {
        "status": "passed",
        "dataset_rows_in_manifest": profile["manifest_total_rows"],
        "sample_rows": profile["sample_rows"],
        "runs": summaries,
    })
    print(json.dumps({"status": "passed", "runs": summaries}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

