from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import load_config
from .data import audit_contiguous_sequence_dataset, write_json
from .graph_sequences import SPLIT_NAMES, align_preprocessed_split, build_sequence_graph_tensors, validate_sequence_leakage
from .preprocessing import LeakageSafePreprocessor
from .splits import assign_group_splits
from .training import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 5 deterministic CPU resume acceptance test")
    parser.add_argument("--data-dir", default="/kaggle/input")
    parser.add_argument("--output-dir", default="/kaggle/working/gc_lstm_ghostnet_step5")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--mode-config", default="configs/practical_baseline.yaml")
    parser.add_argument("--samples-per-file", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--sequence-stride", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    parser.add_argument("--run-name", default="step5-resume-smoke")
    return parser.parse_args()


def _model_state_difference(left: Path, right: Path) -> dict[str, float | bool]:
    left_state = torch.load(left, map_location="cpu", weights_only=False)
    right_state = torch.load(right, map_location="cpu", weights_only=False)
    keys_match = set(left_state["model"]) == set(right_state["model"])
    maximum = 0.0
    if keys_match:
        for key in left_state["model"]:
            maximum = max(
                maximum,
                float((left_state["model"][key] - right_state["model"][key]).abs().max()),
            )
    optimizer_equal = _nested_equal(left_state["optimizer"], right_state["optimizer"])
    scheduler_equal = _nested_equal(left_state["scheduler"], right_state["scheduler"])
    return {
        "keys_match": keys_match,
        "max_absolute_parameter_difference": maximum,
        "optimizer_exact_match": optimizer_equal,
        "scheduler_exact_match": scheduler_equal,
        "exact_match": keys_match and maximum == 0.0 and optimizer_equal and scheduler_equal,
    }


def _nested_equal(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return bool(torch.equal(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(_nested_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_nested_equal(a, b) for a, b in zip(left, right))
    return left == right


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.mode_config)
    config["data"]["samples_per_file"] = args.samples_per_file
    config["step3"]["sequence_length"] = args.sequence_length
    config["step3"]["sequence_stride"] = args.sequence_stride
    config["project"]["seed"] = args.seed
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    sample, profile, manifest = audit_contiguous_sequence_dataset(
        args.data_dir, output / "data_contract" / "audit", config
    )
    split = assign_group_splits(
        sample,
        label_column=profile["label_column"],
        outer_train_fraction=float(config["step4"]["outer_train_fraction"]),
        validation_fraction_of_outer_train=float(config["splits"]["validation_fraction_of_outer_train"]),
        group_rows=int(config["data"]["sequence_group_rows"]),
        seed=args.seed,
        candidate_attempts=int(config["splits"]["candidate_attempts"]),
    )
    preprocessor = LeakageSafePreprocessor(config, args.seed)
    processed = preprocessor.fit_transform_splits(split.frame, profile["label_column"])
    bundles = {
        name: build_sequence_graph_tensors(
            align_preprocessed_split(split.frame, processed, name), name, config
        )
        for name in SPLIT_NAMES
    }
    leakage = validate_sequence_leakage(bundles)
    if leakage["status"] != "passed":
        raise RuntimeError(leakage)

    common = dict(
        bundles=bundles,
        preprocessing_metadata=processed.metadata,
        config=config,
        epochs=3,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    reference_dir = output / "uninterrupted"
    manifest_references = {"sample_manifest": manifest, "audit": str(output / "data_contract" / "audit")}
    reference_args = vars(args) | {
        "session_id": "uninterrupted-session",
        "manifest_references": manifest_references,
    }
    reference = train_model(
        output_dir=reference_dir, run_arguments=reference_args, **common
    )

    resumed_dir = output / "resumed"
    resumed_args = vars(args) | {
        "session_id": "session-1",
        "manifest_references": manifest_references,
    }
    stopped = train_model(
        output_dir=resumed_dir,
        run_arguments=resumed_args,
        stop_after_epoch=2,
        **common,
    )
    if stopped["status"] != "controlled_stop" or stopped["next_epoch"] != 3:
        raise RuntimeError(f"Controlled stop contract failed: {stopped}")
    resumed_args["session_id"] = "session-2"
    resumed = train_model(
        output_dir=resumed_dir,
        run_arguments=resumed_args,
        resume_from=resumed_dir / "last_checkpoint.pt",
        **common,
    )
    comparison = _model_state_difference(
        reference_dir / "final_model_epoch_003.pt",
        resumed_dir / "final_model_epoch_003.pt",
    )
    if not comparison["exact_match"]:
        raise RuntimeError(f"Resume state differs from uninterrupted state: {comparison}")
    resumed_checkpoint = torch.load(
        resumed_dir / "final_model_epoch_003.pt", map_location="cpu", weights_only=False
    )
    epoch_sequence = [int(row["epoch"]) for row in resumed_checkpoint["history"]]
    if epoch_sequence != [1, 2, 3]:
        raise RuntimeError(f"Resume history is not contiguous: {epoch_sequence}")
    summary = {
        "status": "passed",
        "device": "cpu",
        "interrupted_after_epoch": 2,
        "resumed_at_epoch": 3,
        "history_epochs": epoch_sequence,
        "state_comparison": comparison,
        "sequence_leakage_status": leakage["status"],
        "sample_rows": int(profile["sample_rows"]),
        "sample_manifest": manifest,
        "reference": reference,
        "resumed": resumed,
    }
    write_json(output / "step5_resume_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
