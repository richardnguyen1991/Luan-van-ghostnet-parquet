from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch
import psutil
from torch import nn
from torch.utils.data import DataLoader

from .config import config_hash, load_config
from .data import discover_dataset, validate_dataset_manifests, write_json
from .model import GCLSTMGhostNet, model_parameter_count
from .streaming import (
    audit_group_label_schema,
    build_group_manifest,
    concatenate_bundles,
    epoch_schedule,
    fit_streaming_proxy,
    manifest_summary,
    read_group,
    reservoir_groups,
    transform_group,
)
from .training import (
    GraphSequenceDataset,
    S3ArtifactUploader,
    _metric_payload,
    _restore_rng_state,
    _rng_state,
    _stable_hash,
    _write_final_artifacts,
    collate_graph_sequences,
    evaluate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 8 full mixed-group CPU streaming training")
    parser.add_argument("--data-dir", default="/kaggle/input")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--mode-config", default="configs/practical_baseline.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    parser.add_argument("--full-dataset", action="store_true", required=True)
    parser.add_argument("--stream-files", action="store_true", required=True)
    parser.add_argument("--samples-per-file", type=int, help="Forbidden in full streaming mode")
    parser.add_argument("--sequence-group-rows", type=int, default=4096)
    parser.add_argument("--stream-shuffle-buffer-sequences", type=int, default=8192)
    parser.add_argument("--stream-eval-samples-per-file", type=int, default=512)
    parser.add_argument("--train-eval-samples-per-class", type=int, default=256)
    parser.add_argument("--outer-train-fraction", type=float, choices=[0.7, 0.8], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-column")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--resume", nargs="?", const="auto")
    parser.add_argument("--session-budget-minutes", type=int, default=660)
    parser.add_argument("--upload-checkpoints-to-s3", action="store_true")
    parser.add_argument("--s3-bucket")
    parser.add_argument("--s3-prefix", default="")
    parser.add_argument("--aws-region")
    parser.add_argument("--s3-max-retries", type=int, default=3)
    parser.add_argument("--s3-upload-required", action="store_true")
    return parser.parse_args()


def _loader(bundle, batch_size: int, shuffle: bool = False, seed: int = 0) -> DataLoader:
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        GraphSequenceDataset(bundle), batch_size=batch_size, shuffle=shuffle,
        generator=generator, collate_fn=collate_graph_sequences, num_workers=0,
    )


def _find_resume(value: str | None, output: Path, uploader: S3ArtifactUploader) -> Path | None:
    if value is None:
        return None
    if value != "auto":
        return Path(value)
    local = output / "last_checkpoint.pt"
    if local.exists():
        return local
    s3_resume = output / "s3_resume_last_checkpoint.pt"
    if uploader.download("checkpoints/last_checkpoint.pt", s3_resume, required=False):
        return s3_resume
    candidates = sorted(Path("/kaggle/input").rglob("last_checkpoint.pt"))
    return candidates[-1] if candidates else None


def main() -> None:
    args = parse_args()
    if args.samples_per_file is not None:
        raise ValueError("--samples-per-file cannot cap a Step 8 full streaming train run")
    if args.mode_config != "configs/practical_baseline.yaml":
        raise ValueError("Only practical_baseline is approved; paper_faithful still requires CFACO/GAIN ablations")
    if args.epochs != 100:
        raise ValueError("Step 8 acceptance requires exactly 100 target epochs")
    config = load_config(args.config, args.mode_config)
    config["project"]["seed"] = args.seed
    config["step4"]["outer_train_fraction"] = args.outer_train_fraction
    config["data"]["sequence_group_rows"] = args.sequence_group_rows
    if args.target_column:
        config["data"]["label_candidates"] = [args.target_column]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = discover_dataset(args.data_dir, config["data"]["dataset_dir_hint"], config["data"]["expected_source_files"])
    contract = validate_dataset_manifests(paths, config["data"])
    label_column = contract["label_column"]
    groups = build_group_manifest(paths, args.sequence_group_rows, args.outer_train_fraction, args.seed)
    manifest = manifest_summary(groups)
    manifest.update({
        "outer_train_fraction": args.outer_train_fraction,
        "effective_train_fraction": args.outer_train_fraction * 0.9,
        "validation_fraction": args.outer_train_fraction * 0.1,
        "test_fraction": 1.0 - args.outer_train_fraction,
        "schedule_rule": "all train groups exactly once; Random(seed + epoch)",
        "shuffle_buffer_changes_order_only": True,
    })
    write_json(output / "sample_manifest.json", manifest)

    label_audit = audit_group_label_schema(paths, groups, label_column)
    write_json(output / "label_schema_audit.json", label_audit)

    # Full scan with bounded whole-group reservoir retention. Only retained train
    # groups are used to fit preprocessing; validation/test are transform-only.
    per_file_groups = max(1, math.ceil(args.stream_eval_samples_per_file / args.sequence_group_rows))
    reservoirs = reservoir_groups(paths, groups, per_file_groups, args.seed)
    processor, preprocessing_metadata = fit_streaming_proxy(
        reservoirs,
        label_column,
        config,
        args.seed,
        label_vocabulary=label_audit["label_vocabulary"],
    )
    preprocessing_metadata.update({
        "label_mapping_scope": label_audit["scope"],
        "label_schema_audit_hash": _stable_hash(label_audit),
        "label_counts_by_split": label_audit["label_counts_by_split"],
        "labels_missing_from_split": label_audit["labels_missing_from_split"],
    })
    processor.save(output, preprocessing_metadata)
    joblib.dump(processor, output / "preprocessor.joblib")
    eval_bundles = {}
    for split in ("train", "validation", "test"):
        built = [
            bundle for frame in reservoirs[split]
            if (bundle := transform_group(frame, split, label_column, processor, config)) is not None
        ]
        if not built:
            raise RuntimeError(f"No fixed {split} reservoir sequences")
        eval_bundles[split] = concatenate_bundles(built)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    class_names = [name for name, _ in sorted(processor.label_mapping.items(), key=lambda item: item[1])]
    model = GCLSTMGhostNet(len(processor.feature_columns), len(class_names), config)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=float(config["future_training_contract"]["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
        eta_min=float(config["future_training_contract"]["min_learning_rate"]),
    )
    history = []
    best_f1 = -math.inf
    epoch = 1
    cursor = 0
    generated = 0
    consumed = 0
    resume_online = {"loss_sum": 0.0, "correct": 0, "count": 0, "flush_index": 0}
    run_id = args.run_name
    session_id = args.session_id or uuid.uuid4().hex
    contract_hashes = {
        "config": config_hash(config),
        "schema": str(contract["schema"]["schema_hash"]),
        "preprocessing": _stable_hash(preprocessing_metadata),
        "selected_features": _stable_hash(processor.feature_columns),
        "graph": _stable_hash(config["step3"]),
        "group_manifest": _stable_hash(manifest),
    }
    prefix = "/".join(part.strip("/") for part in (args.s3_prefix, "gc-lstm-ghostnet", run_id) if part)
    uploader = S3ArtifactUploader(args.upload_checkpoints_to_s3, args.s3_bucket, prefix,
                                  args.aws_region, args.s3_max_retries, args.s3_upload_required)
    for artifact in (
        output / "sample_manifest.json", output / "label_schema_audit.json",
        output / "preprocessing.json", output / "preprocessor.joblib",
    ):
        uploader.upload(artifact, f"artifacts/{artifact.name}")

    resume = _find_resume(args.resume, output, uploader)
    if args.resume is not None and resume is None and uploader.resume_required:
        raise FileNotFoundError("S3 active run requires resume, but its last checkpoint could not be downloaded")
    if args.resume is not None and resume is None:
        print("No resumable checkpoint exists; starting the new exhaustive-label run at epoch 1")
    if resume is not None:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        checkpoint_mapping = checkpoint.get("preprocessing_metadata", {}).get("label_mapping")
        if checkpoint_mapping != preprocessing_metadata["label_mapping"]:
            raise ValueError(
                "Resume checkpoint label mapping is incompatible with the exhaustive "
                "dataset label audit; start a new run from epoch 1"
            )
        if checkpoint["contract_hashes"] != contract_hashes or checkpoint["run_id"] != run_id:
            raise ValueError("Resume checkpoint contract/run mismatch")
        model.load_state_dict(checkpoint["model"]); optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"]); _restore_rng_state(checkpoint["rng_state"])
        history = checkpoint["history"]; best_f1 = checkpoint["best_f1"]
        epoch = checkpoint["epoch"]; cursor = checkpoint["progress_cursor"]
        epoch = checkpoint.get("next_epoch", epoch)
        generated = checkpoint["generated_train_sequences"]; consumed = checkpoint["consumed_train_sequences"]
        resume_online = checkpoint.get("epoch_online", resume_online)

    session_deadline = time.monotonic() + max(1, args.session_budget_minutes - 20) * 60
    training_started = time.perf_counter()
    peak_memory_mb = 0.0
    process = psutil.Process(os.getpid())

    def update_active(status: str, completed_epoch: int) -> None:
        active = output / "active_run.json"
        write_json(active, {
            "run_id": run_id,
            "status": status,
            "completed_epoch": completed_epoch,
            "active_epoch": epoch,
            "progress_cursor": cursor,
            "generated_train_sequences": generated,
            "consumed_train_sequences": consumed,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "contract_version": "exhaustive-label-v1",
            "device": "cpu",
        })
        active_key = "/".join(part.strip("/") for part in (args.s3_prefix, "active_run.json") if part)
        uploader.upload_key(active, active_key)

    update_active("running", max(0, epoch - 1))

    def save_checkpoint(complete_epoch: bool) -> Path:
        checkpoint_epoch = epoch if complete_epoch else epoch - 1
        payload = {
            "epoch": checkpoint_epoch, "active_epoch": epoch,
            "next_epoch": epoch + 1 if complete_epoch else epoch,
            "progress_cursor": 0 if complete_epoch else cursor,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "grad_scaler": None,
            "best_f1": best_f1, "history": history, "rng_state": _rng_state(),
            "config": config, "preprocessing_metadata": preprocessing_metadata,
            "contract_hashes": contract_hashes, "run_id": run_id, "session_id": session_id,
            "generated_train_sequences": generated, "consumed_train_sequences": consumed,
            "epoch_online": ({
                "loss_sum": online_loss, "correct": online_correct,
                "count": online_count, "flush_index": flush_index,
            } if not complete_epoch else {"loss_sum": 0.0, "correct": 0, "count": 0, "flush_index": 0}),
            "manifest_references": {"sample_manifest": "sample_manifest.json", "preprocessing": "preprocessing.json"},
        }
        path = output / (f"epoch_{epoch:03d}.pt" if complete_epoch else "emergency_checkpoint.pt")
        torch.save(payload, path); shutil.copy2(path, output / "last_checkpoint.pt")
        write_json(output / "checkpoint_metadata.json", {
            key: payload[key] for key in ("epoch", "active_epoch", "next_epoch", "progress_cursor", "best_f1", "contract_hashes", "run_id", "session_id", "generated_train_sequences", "consumed_train_sequences")
        })
        uploader.upload(path, f"checkpoints/{path.name}")
        uploader.upload(output / "last_checkpoint.pt", "checkpoints/last_checkpoint.pt")
        uploader.upload(output / "checkpoint_metadata.json", "checkpoints/checkpoint_metadata.json")
        update_active("running" if complete_epoch else "paused", checkpoint_epoch)
        return path

    while epoch <= args.epochs:
        schedule = epoch_schedule(groups, epoch, args.seed)
        buffer = []
        buffered = 0
        online_loss = float(resume_online["loss_sum"]); online_correct = int(resume_online["correct"])
        online_count = int(resume_online["count"]); flush_index = int(resume_online["flush_index"])
        resume_online = {"loss_sum": 0.0, "correct": 0, "count": 0, "flush_index": 0}

        def flush() -> None:
            nonlocal buffer, buffered, online_loss, online_correct, online_count, consumed, flush_index, peak_memory_mb
            if not buffer: return
            bundle = concatenate_bundles(buffer)
            generated_now = len(bundle.sequence_x)
            loader = _loader(bundle, args.batch_size, True, args.seed + epoch * 100000 + flush_index)
            model.train()
            for batch in loader:
                optimizer.zero_grad(set_to_none=True); logits, _ = model(batch)
                loss = criterion(logits, batch.target_y); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["future_training_contract"]["gradient_clip_norm"]))
                optimizer.step()
                online_loss += float(loss.detach()) * len(batch.target_y)
                online_correct += int((logits.argmax(1) == batch.target_y).sum())
                online_count += len(batch.target_y)
                peak_memory_mb = max(peak_memory_mb, process.memory_info().rss / (1024 ** 2))
            consumed += generated_now; flush_index += 1; buffer = []; buffered = 0
            del bundle, loader; gc.collect()

        while cursor < len(schedule):
            group = schedule[cursor]
            frame = read_group(paths, group)
            bundle = transform_group(frame, "train", label_column, processor, config)
            cursor += 1
            if bundle is not None:
                generated += len(bundle.sequence_x); buffer.append(bundle); buffered += len(bundle.sequence_x)
            if buffered >= args.stream_shuffle_buffer_sequences:
                flush()
            if time.monotonic() >= session_deadline:
                flush(); save_checkpoint(False)
                write_json(output / "step8_session_summary.json", {
                    "status": "controlled_session_stop", "active_epoch": epoch,
                    "next_group_cursor": cursor, "groups_in_epoch": len(schedule),
                    "generated_train_sequences": generated, "consumed_train_sequences": consumed,
                    "counts_equal_at_safe_stop": generated == consumed, "device": "cpu",
                })
                uploader.upload(output / "step8_session_summary.json", "status/step8_session_summary.json")
                print(json.dumps(json.loads((output / "step8_session_summary.json").read_text()), indent=2)); return
        flush()
        if generated != consumed:
            raise RuntimeError(f"Full streaming dropped sequences: generated={generated}, consumed={consumed}")
        train_result = evaluate(model, _loader(eval_bundles["train"], args.batch_size), criterion, torch.device("cpu"), len(class_names))
        val_result = evaluate(model, _loader(eval_bundles["validation"], args.batch_size), criterion, torch.device("cpu"), len(class_names))
        history.append({
            "epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"],
            "train": _metric_payload(train_result), "validation": _metric_payload(val_result),
            "online_train_loss": online_loss / max(1, online_count),
            "online_train_accuracy": online_correct / max(1, online_count),
            "generated_train_sequences_cumulative": generated,
            "consumed_train_sequences_cumulative": consumed,
        })
        improved = val_result.macro_f1 > best_f1; best_f1 = max(best_f1, val_result.macro_f1)
        scheduler.step(); save_checkpoint(True)
        if improved:
            shutil.copy2(output / "last_checkpoint.pt", output / "best_model.pt")
            uploader.upload(output / "best_model.pt", "checkpoints/best_model.pt")
        epoch += 1; cursor = 0

    final = output / "final_model_epoch_100.pt"; shutil.copy2(output / "last_checkpoint.pt", final)
    final_checkpoint = torch.load(final, map_location="cpu", weights_only=False)
    model.load_state_dict(final_checkpoint["model"])
    test = evaluate(model, _loader(eval_bundles["test"], args.batch_size), criterion, torch.device("cpu"), len(class_names))
    artifacts = _write_final_artifacts(output, history, test, class_names,
                                       time.perf_counter() - training_started - uploader.transfer_seconds,
                                       peak_memory_mb)
    run_config = {
        "execution_scope": "full_mixed_group_streaming", "device": "cpu", "epochs": 100,
        "outer_train_fraction": args.outer_train_fraction, "mode": "practical_baseline",
        "generated_train_sequences": generated, "consumed_train_sequences": consumed,
        "counts_equal": generated == consumed, "model_parameters": model_parameter_count(model),
        "contract_hashes": contract_hashes, "run_id": run_id,
    }
    write_json(output / "run_config.json", run_config)
    for artifact in [*artifacts, final, output / "run_config.json"]:
        uploader.upload(artifact, f"artifacts/{artifact.name}")
    update_active("completed", args.epochs)
    print(json.dumps({"status": "passed", **run_config, "test_metrics": _metric_payload(test)}, indent=2))


if __name__ == "__main__":
    main()
