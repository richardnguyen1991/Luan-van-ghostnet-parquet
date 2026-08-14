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
    validate_sequence_leakage,
)
from .preprocessing import LeakageSafePreprocessor
from .splits import assign_group_splits, write_split_artifacts
from .training import S3ArtifactUploader, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GC-LSTM-GhostNet on CIC-DDoS2019")
    parser.add_argument("--data-dir", default="/kaggle/input")
    parser.add_argument("--output-dir", default="/kaggle/working/gc_lstm_ghostnet_step5")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--mode-config", default="configs/practical_baseline.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    parser.add_argument("--samples-per-file", type=int, default=2048)
    parser.add_argument("--sequence-group-rows", type=int)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--sequence-stride", type=int)
    parser.add_argument("--outer-train-fraction", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--target-column")
    parser.add_argument("--full-dataset", action="store_true")
    parser.add_argument("--stream-files", action="store_true")
    parser.add_argument("--upload-checkpoints-to-s3", action="store_true")
    parser.add_argument("--s3-bucket")
    parser.add_argument("--s3-prefix", default="")
    parser.add_argument("--aws-region")
    parser.add_argument("--s3-max-retries", type=int, default=3)
    parser.add_argument("--s3-upload-required", action="store_true")
    parser.add_argument("--run-name", default="step5")
    parser.add_argument("--session-id")
    parser.add_argument("--resume", nargs="?", const="auto")
    parser.add_argument("--stop-after-epoch", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.full_dataset or args.stream_files:
        raise ValueError(
            "Step 5 currently supports bounded, contiguous sampled materialization. "
            "Do not claim a full-dataset result; full mixed-group streaming remains a separate scale run."
        )
    config = load_config(args.config, args.mode_config)
    if args.device != "cpu":
        raise ValueError("Only CPU execution is supported by this project")
    config["data"]["samples_per_file"] = int(args.samples_per_file)
    if args.sequence_group_rows is not None:
        config["data"]["sequence_group_rows"] = int(args.sequence_group_rows)
    if args.sequence_length is not None:
        config["step3"]["sequence_length"] = int(args.sequence_length)
    if args.sequence_stride is not None:
        config["step3"]["sequence_stride"] = int(args.sequence_stride)
    if args.outer_train_fraction is not None:
        config["step4"]["outer_train_fraction"] = float(args.outer_train_fraction)
    if args.seed is not None:
        config["project"]["seed"] = int(args.seed)
    if args.target_column:
        config["data"]["label_candidates"] = [args.target_column]

    seed = int(config["project"]["seed"])
    batch_size = int(args.batch_size or config["future_training_contract"]["batch_size"])
    learning_rate = float(args.learning_rate or config["future_training_contract"]["learning_rate"])
    output = Path(args.output_dir)
    data_output = output / "data_contract"
    model_output = output / "training"
    output.mkdir(parents=True, exist_ok=True)

    sample, profile, sample_manifest = audit_contiguous_sequence_dataset(
        args.data_dir, data_output / "audit", config
    )
    label_column = profile["label_column"]
    split_result = assign_group_splits(
        sample,
        label_column=label_column,
        outer_train_fraction=float(config["step4"]["outer_train_fraction"]),
        validation_fraction_of_outer_train=float(
            config["splits"]["validation_fraction_of_outer_train"]
        ),
        group_rows=int(config["data"]["sequence_group_rows"]),
        seed=seed,
        candidate_attempts=int(config["splits"]["candidate_attempts"]),
    )
    write_split_artifacts(split_result, data_output)
    preprocessor = LeakageSafePreprocessor(config, seed)
    processed = preprocessor.fit_transform_splits(split_result.frame, label_column)
    preprocessor.save(data_output, processed.metadata)
    bundles = {}
    for split in SPLIT_NAMES:
        aligned = align_preprocessed_split(split_result.frame, processed, split)
        bundles[split] = build_sequence_graph_tensors(aligned, split, config)
        save_sequence_graph_tensors(bundles[split], data_output / split)
    leakage = validate_sequence_leakage(bundles)
    if leakage["status"] != "passed":
        raise RuntimeError(f"Sequence leakage validation failed: {leakage}")
    write_json(data_output / "sequence_leakage_report.json", leakage)

    s3_prefix = "/".join(
        part.strip("/") for part in (args.s3_prefix, "gc-lstm-ghostnet", args.run_name) if part
    )
    uploader = S3ArtifactUploader(
        args.upload_checkpoints_to_s3,
        args.s3_bucket,
        s3_prefix,
        args.aws_region,
        args.s3_max_retries,
        args.s3_upload_required,
    )
    run_arguments = vars(args).copy()
    run_arguments["manifest_references"] = {
        "sample_manifest": str(data_output / "audit" / "sample_manifest.json"),
        "preprocessing": str(data_output / "preprocessing.json"),
        "sequence_leakage": str(data_output / "sequence_leakage_report.json"),
    }
    resume_from = args.resume
    if resume_from == "auto":
        resume_from = str(model_output / "last_checkpoint.pt")
    if resume_from is not None and not Path(resume_from).exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_from}")
    summary = train_model(
        bundles=bundles,
        preprocessing_metadata=processed.metadata,
        config=config,
        output_dir=model_output,
        epochs=int(args.epochs),
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        run_arguments=run_arguments,
        uploader=uploader,
        resume_from=resume_from,
        stop_after_epoch=args.stop_after_epoch,
    )
    summary.update({
        "execution_scope": "bounded_contiguous_sample",
        "sample_rows": int(profile["sample_rows"]),
        "manifest_total_rows": int(profile["manifest_total_rows"]),
        "config_hash": config_hash(config),
        "sample_manifest": sample_manifest,
        "sequence_leakage_status": leakage["status"],
    })
    write_json(output / "step5_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
