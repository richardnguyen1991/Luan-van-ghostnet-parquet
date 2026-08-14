from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import shutil
import time
import uuid
import urllib.error
import urllib.request
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data import write_json
from .graph_sequences import SequenceGraphTensors
from .model import GCLSTMGhostNet, GraphBatch, model_parameter_count


class GraphSequenceDataset(Dataset):
    def __init__(self, bundle: SequenceGraphTensors) -> None:
        self.bundle = bundle

    def __len__(self) -> int:
        return len(self.bundle.sequence_x)

    def __getitem__(self, index: int) -> dict[str, Any]:
        edge_start, edge_stop = self.bundle.edge_window_ptr[index : index + 2]
        node_start, node_stop = self.bundle.node_window_ptr[index : index + 2]
        return {
            "sequence_x": self.bundle.sequence_x[index],
            "target_y": self.bundle.target_y[index],
            "edge_index": self.bundle.edge_index[:, edge_start:edge_stop],
            "node_count": int(node_stop - node_start),
        }


def collate_graph_sequences(items: Sequence[dict[str, Any]]) -> GraphBatch:
    return GraphBatch(
        sequence_x=torch.from_numpy(np.stack([item["sequence_x"] for item in items])).float(),
        target_y=torch.as_tensor([item["target_y"] for item in items], dtype=torch.long),
        edge_index=tuple(torch.from_numpy(item["edge_index"]).long() for item in items),
        node_counts=torch.as_tensor([item["node_count"] for item in items], dtype=torch.long),
    )


@dataclass(frozen=True)
class EvaluationResult:
    loss: float
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    auc_roc: float | None
    inference_seconds: float
    inference_latency_ms_per_sample: float
    throughput_samples_per_second: float
    y_true: list[int]
    y_pred: list[int]
    probabilities: list[list[float]]


def _auc(y_true: np.ndarray, probabilities: np.ndarray, class_count: int) -> float | None:
    present = np.unique(y_true)
    try:
        if int(class_count) == 2:
            if len(present) < 2:
                return None
            return float(roc_auc_score(y_true, probabilities[:, 1]))
        if len(present) < int(class_count):
            return None
        return float(roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro"))
    except ValueError:
        return None


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_count: int,
) -> EvaluationResult:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    truths: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device)
            logits, _ = model(batch)
            loss = criterion(logits, batch.target_y)
            batch_size = len(batch.target_y)
            total_loss += float(loss.detach().cpu()) * batch_size
            total_samples += batch_size
            probability = torch.softmax(logits, dim=1)
            truths.append(batch.target_y.cpu().numpy())
            predictions.append(probability.argmax(dim=1).cpu().numpy())
            probabilities.append(probability.cpu().numpy())
    elapsed = time.perf_counter() - started
    y_true = np.concatenate(truths)
    y_pred = np.concatenate(predictions)
    probability_array = np.concatenate(probabilities)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return EvaluationResult(
        loss=total_loss / max(1, total_samples),
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_precision=float(macro_precision),
        macro_recall=float(macro_recall),
        macro_f1=float(macro_f1),
        weighted_f1=float(weighted_f1),
        auc_roc=_auc(y_true, probability_array, class_count),
        inference_seconds=elapsed,
        inference_latency_ms_per_sample=elapsed * 1000.0 / max(1, total_samples),
        throughput_samples_per_second=total_samples / max(elapsed, 1e-12),
        y_true=y_true.astype(int).tolist(),
        y_pred=y_pred.astype(int).tolist(),
        probabilities=probability_array.astype(float).tolist(),
    )


def balanced_class_weights(targets: np.ndarray, class_count: int) -> torch.Tensor:
    counts = np.bincount(np.asarray(targets, dtype=np.int64), minlength=int(class_count))
    weights = np.zeros(int(class_count), dtype=np.float32)
    present = counts > 0
    weights[present] = len(targets) / (int(present.sum()) * counts[present])
    return torch.from_numpy(weights)


class S3ArtifactUploader:
    def __init__(
        self,
        enabled: bool,
        bucket: str | None,
        prefix: str,
        region: str | None,
        max_retries: int,
        required: bool,
    ) -> None:
        self.enabled = bool(enabled)
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.region = region
        self.max_retries = int(max_retries)
        self.required = bool(required)
        self.client = None
        self.presigned: dict[str, Any] | None = None
        self.resume_required = False
        self.transfer_seconds = 0.0
        if self.enabled:
            config_path = os.environ.get("S3_PRESIGNED_CONFIG_PATH")
            if config_path:
                self.presigned = json.loads(Path(config_path).read_text(encoding="utf-8"))
                self.bucket = str(self.presigned["bucket"])
                self.resume_required = bool(self.presigned.get("resume_required", False))
            else:
                if not bucket:
                    raise ValueError("--s3-bucket is required when S3 upload is enabled")
                import boto3

                self.client = boto3.client("s3", region_name=region)

    def _key(self, relative_key: str) -> str:
        return "/".join(part for part in (self.prefix, relative_key.replace("\\", "/")) if part)

    def _presigned_transfer(self, method: str, url: str, path: Path) -> None:
        if method == "PUT":
            request = urllib.request.Request(url, data=path.read_bytes(), method="PUT")
            with urllib.request.urlopen(request, timeout=300) as response:
                response.read()
            return
        temporary = path.with_name(f".{path.name}.download-{uuid.uuid4().hex}")
        try:
            with urllib.request.urlopen(url, timeout=300) as response:
                temporary.write_bytes(response.read())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def upload_key(self, local_path: str | Path, key: str) -> bool:
        if not self.enabled:
            return False
        path = Path(local_path)
        transfer_started = time.perf_counter()
        last_error: Exception | None = None
        if self.presigned is not None:
            url = self.presigned.get("uploads", {}).get(key)
            if not url:
                message = f"No presigned upload URL was issued for S3 key {key}"
                if self.required:
                    raise RuntimeError(message)
                warnings.warn(message)
                return False
            for attempt in range(self.max_retries + 1):
                try:
                    self._presigned_transfer("PUT", str(url), path)
                    self.transfer_seconds += time.perf_counter() - transfer_started
                    return True
                except Exception as exc:
                    last_error = exc
                    if attempt < self.max_retries:
                        time.sleep(min(2 ** attempt, 8))
            message = f"Presigned S3 upload failed for {path.name}: {type(last_error).__name__}"
            self.transfer_seconds += time.perf_counter() - transfer_started
            if self.required:
                raise RuntimeError(message) from last_error
            warnings.warn(message)
            return False

        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        temporary_key = f"{key}.tmp-{uuid.uuid4().hex}"
        for attempt in range(self.max_retries + 1):
            try:
                assert self.client is not None and self.bucket is not None
                self.client.upload_file(str(path), self.bucket, temporary_key,
                                        ExtraArgs={"Metadata": {"sha256": checksum}})
                temporary = self.client.head_object(Bucket=self.bucket, Key=temporary_key)
                if int(temporary["ContentLength"]) != path.stat().st_size:
                    raise RuntimeError("temporary S3 object size mismatch")
                self.client.copy_object(Bucket=self.bucket, Key=key,
                                        CopySource={"Bucket": self.bucket, "Key": temporary_key},
                                        MetadataDirective="COPY")
                final = self.client.head_object(Bucket=self.bucket, Key=key)
                if int(final["ContentLength"]) != path.stat().st_size:
                    raise RuntimeError("final S3 object size mismatch")
                if final.get("Metadata", {}).get("sha256") != checksum:
                    raise RuntimeError("final S3 object checksum metadata mismatch")
                self.client.delete_object(Bucket=self.bucket, Key=temporary_key)
                self.transfer_seconds += time.perf_counter() - transfer_started
                return True
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 8))
        message = f"S3 upload failed for {path.name}: {type(last_error).__name__}"
        try:
            if self.client is not None and self.bucket is not None:
                self.client.delete_object(Bucket=self.bucket, Key=temporary_key)
        except Exception:
            pass
        self.transfer_seconds += time.perf_counter() - transfer_started
        if self.required:
            raise RuntimeError(message) from last_error
        warnings.warn(message)
        return False

    def upload(self, local_path: str | Path, relative_key: str) -> bool:
        return self.upload_key(local_path, self._key(relative_key))

    def download(self, relative_key: str, destination: str | Path, required: bool = False) -> bool:
        if not self.enabled:
            return False
        key = self._key(relative_key)
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                if self.presigned is not None:
                    url = self.presigned.get("downloads", {}).get(key)
                    if not url:
                        raise FileNotFoundError(f"No presigned download URL for {key}")
                    self._presigned_transfer("GET", str(url), path)
                else:
                    assert self.client is not None and self.bucket is not None
                    self.client.download_file(self.bucket, key, str(path))
                return True
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 8))
        path.unlink(missing_ok=True)
        if required:
            message = f"S3 download failed for {key}: {type(last_error).__name__}"
            raise RuntimeError(message) from last_error
        return False


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": None,
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


def _checkpoint_payload(
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    best_f1: float,
    history: list[dict[str, Any]],
    train_metrics: EvaluationResult,
    validation_metrics: EvaluationResult,
    config: dict[str, Any],
    preprocessing_metadata: dict[str, Any],
    run_arguments: dict[str, Any],
    contract_hashes: dict[str, str],
    run_id: str,
    session_id: str,
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "best_f1": float(best_f1),
        "train_loss": train_metrics.loss,
        "val_loss": validation_metrics.loss,
        "auc": validation_metrics.auc_roc,
        "validation_metrics": _metric_payload(validation_metrics),
        "history": history,
        "config": config,
        "preprocessing_metadata": preprocessing_metadata,
        "run_arguments": run_arguments,
        "model_parameters": model_parameter_count(model),
        "grad_scaler": None,
        "rng_state": _rng_state(),
        "contract_hashes": contract_hashes,
        "run_id": run_id,
        "session_id": session_id,
    }


def _metric_payload(result: EvaluationResult) -> dict[str, Any]:
    payload = asdict(result)
    payload.pop("y_true")
    payload.pop("y_pred")
    payload.pop("probabilities")
    return payload


def _plot_history(history: list[dict[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    for file_name, key, title, ylabel in (
        ("accuracy_curve.png", "accuracy", "Train and validation accuracy", "Accuracy"),
        ("loss_curve.png", "loss", "Train and validation loss", "Loss"),
        ("auc_curve.png", "auc_roc", "Validation AUC-ROC", "AUC-ROC"),
    ):
        figure, axis = plt.subplots(figsize=(7, 4.5))
        if key == "auc_roc":
            values = [row["validation"].get(key) for row in history]
            axis.plot(epochs, [np.nan if value is None else value for value in values], label="validation")
        else:
            axis.plot(epochs, ç6öÚ$z{-®éÜj×aithful still requires CFACO/GAIN ablations")
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
