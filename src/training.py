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
        if self.enabled:
            if not bucket:
                raise ValueError("--s3-bucket is required when S3 upload is enabled")
            import boto3

            self.client = boto3.client("s3", region_name=region)

    def upload(self, local_path: str | Path, relative_key: str) -> bool:
        if not self.enabled:
            return False
        path = Path(local_path)
        key = "/".join(part for part in (self.prefix, relative_key.replace("\\", "/")) if part)
        last_error: Exception | None = None
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        temporary_key = f"{key}.tmp-{uuid.uuid4().hex}"
        for attempt in range(self.max_retries + 1):
            try:
                assert self.client is not None and self.bucket is not None
                self.client.upload_file(
                    str(path), self.bucket, temporary_key,
                    ExtraArgs={"Metadata": {"sha256": checksum}},
                )
                temporary = self.client.head_object(Bucket=self.bucket, Key=temporary_key)
                if int(temporary["ContentLength"]) != path.stat().st_size:
                    raise RuntimeError("temporary S3 object size mismatch")
                self.client.copy_object(
                    Bucket=self.bucket,
                    Key=key,
                    CopySource={"Bucket": self.bucket, "Key": temporary_key},
                    MetadataDirective="COPY",
                )
                final = self.client.head_object(Bucket=self.bucket, Key=key)
                if int(final["ContentLength"]) != path.stat().st_size:
                    raise RuntimeError("final S3 object size mismatch")
                if final.get("Metadata", {}).get("sha256") != checksum:
                    raise RuntimeError("final S3 object checksum metadata mismatch")
                self.client.delete_object(Bucket=self.bucket, Key=temporary_key)
                return True
            except Exception as exc:  # boto3 exposes several optional exception packages
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 8))
        message = f"S3 upload failed for {path.name}: {type(last_error).__name__}"
        try:
            if self.client is not None and self.bucket is not None:
                self.client.delete_object(Bucket=self.bucket, Key=temporary_key)
        except Exception:
            pass
        if self.required:
            raise RuntimeError(message) from last_error
        warnings.warn(message)
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
            axis.plot(epochs, [row["train"][key] for row in history], label="train")
            axis.plot(epochs, [row["validation"][key] for row in history], label="validation")
        axis.set(title=title, xlabel="Epoch", ylabel=ylabel)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output / file_name, dpi=160)
        plt.close(figure)


def _write_final_artifacts(
    output: Path,
    history: list[dict[str, Any]],
    test_result: EvaluationResult,
    class_names: list[str],
    training_seconds: float,
    peak_memory_mb: float,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = _metric_payload(test_result)
    metrics.update({"training_time_seconds": training_seconds, "peak_memory_mb": peak_memory_mb})
    write_json(output / "history.json", history)
    write_json(output / "test_metrics.json", metrics)
    with (output / "summary_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(metrics.items())
    matrix = confusion_matrix(test_result.y_true, test_result.y_pred, labels=range(len(class_names)))
    with (output / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted", *class_names])
        for label, row in zip(class_names, matrix):
            writer.writerow([label, *row.tolist()])
    figure, axis = plt.subplots(figsize=(max(7, len(class_names) * 0.55), max(6, len(class_names) * 0.5)))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_xticks(range(len(class_names)), class_names, rotation=90)
    axis.set_yticks(range(len(class_names)), class_names)
    axis.set(xlabel="Predicted", ylabel="Actual", title="Confusion matrix")
    figure.tight_layout()
    figure.savefig(output / "confusion_matrix.png", dpi=160)
    plt.close(figure)
    _plot_history(history, output)
    return [
        output / name for name in (
            "history.json", "test_metrics.json", "summary_metrics.csv", "confusion_matrix.csv",
            "confusion_matrix.png", "accuracy_curve.png", "loss_curve.png", "auc_curve.png",
        )
    ]


def train_model(
    bundles: dict[str, SequenceGraphTensors],
    preprocessing_metadata: dict[str, Any],
    config: dict[str, Any],
    output_dir: str | Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    run_arguments: dict[str, Any],
    uploader: S3ArtifactUploader | None = None,
    resume_from: str | Path | None = None,
    stop_after_epoch: int | None = None,
) -> dict[str, Any]:
    import psutil

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True)
    requested_device = str(run_arguments.get("device", config["future_training_contract"]["device"]))
    if requested_device != "cpu":
        raise ValueError("Only CPU execution is supported by this project")
    device = torch.device("cpu")
    label_mapping = preprocessing_metadata["label_mapping"]
    class_names = [label for label, _ in sorted(label_mapping.items(), key=lambda item: item[1])]
    class_count = len(class_names)
    feature_count = int(bundles["train"].sequence_x.shape[2])
    model = GCLSTMGhostNet(feature_count, class_count, config).to(device)
    evaluation_loaders = {
        split: DataLoader(
            GraphSequenceDataset(bundle),
            batch_size=int(batch_size),
            shuffle=False,
            collate_fn=collate_graph_sequences,
            num_workers=0,
        )
        for split, bundle in bundles.items()
    }
    if config["step4"].get("class_weighting") != "none":
        raise ValueError("The approved practical baseline does not use class weighting")
    criterion = nn.CrossEntropyLoss()
    contract = config["future_training_contract"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(contract["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(epochs)),
        eta_min=float(contract["min_learning_rate"]),
    )
    history: list[dict[str, Any]] = []
    best_f1 = -math.inf
    start_epoch = 1
    run_id = str(run_arguments.get("run_name") or "step5")
    session_id = str(run_arguments.get("session_id") or uuid.uuid4().hex)
    contract_hashes = {
        "config": _stable_hash(config),
        "schema": _stable_hash({
            "feature_order": preprocessing_metadata["feature_order"],
            "label_mapping": preprocessing_metadata["label_mapping"],
        }),
        "preprocessing": _stable_hash(preprocessing_metadata),
        "selected_features": _stable_hash(preprocessing_metadata["feature_order"]),
        "graph": _stable_hash({split: bundle.metadata for split, bundle in bundles.items()}),
    }
    if resume_from is not None:
        checkpoint = torch.load(Path(resume_from), map_location=device, weights_only=False)
        if checkpoint.get("contract_hashes") != contract_hashes:
            raise ValueError("Resume checkpoint contract hashes do not match this run")
        if checkpoint.get("run_id") != run_id:
            raise ValueError("Resume checkpoint run_id does not match --run-name")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        history = list(checkpoint["history"])
        best_f1 = float(checkpoint["best_f1"])
        start_epoch = int(checkpoint["epoch"]) + 1
        _restore_rng_state(checkpoint["rng_state"])
        if start_epoch > int(epochs):
            raise ValueError("Resume checkpoint is already at or beyond the target epoch")
    uploader = uploader or S3ArtifactUploader(False, None, "", None, 0, False)
    process = psutil.Process(os.getpid())
    peak_memory_mb = process.memory_info().rss / (1024 ** 2)
    training_started = time.perf_counter()
    for epoch in range(start_epoch, int(epochs) + 1):
        train_generator = torch.Generator().manual_seed(int(seed) + int(epoch))
        train_loader = DataLoader(
            GraphSequenceDataset(bundles["train"]),
            batch_size=int(batch_size),
            shuffle=True,
            generator=train_generator,
            collate_fn=collate_graph_sequences,
            num_workers=0,
        )
        model.train()
        online_loss = 0.0
        online_correct = 0
        online_count = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(batch)
            loss = criterion(logits, batch.target_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(contract["gradient_clip_norm"]))
            optimizer.step()
            online_loss += float(loss.detach().cpu()) * len(batch.target_y)
            online_correct += int((logits.argmax(dim=1) == batch.target_y).sum().detach().cpu())
            online_count += len(batch.target_y)
            peak_memory_mb = max(peak_memory_mb, process.memory_info().rss / (1024 ** 2))
        train_result = evaluate(model, evaluation_loaders["train"], criterion, device, class_count)
        validation_result = evaluate(model, evaluation_loaders["validation"], criterion, device, class_count)
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": _metric_payload(train_result),
            "validation": _metric_payload(validation_result),
            "online_train_loss": online_loss / max(1, online_count),
            "online_train_accuracy": online_correct / max(1, online_count),
        }
        history.append(row)
        improved = validation_result.macro_f1 > best_f1
        best_f1 = max(best_f1, validation_result.macro_f1)
        scheduler.step()
        payload = _checkpoint_payload(
            epoch, model, optimizer, scheduler, best_f1, history,
            train_result, validation_result, config, preprocessing_metadata, run_arguments,
            contract_hashes, run_id, session_id,
        )
        epoch_path = output / f"epoch_{epoch:03d}.pt"
        torch.save(payload, epoch_path)
        shutil.copy2(epoch_path, output / "last_checkpoint.pt")
        if improved:
            shutil.copy2(epoch_path, output / "best_model.pt")
        checkpoint_metadata = {
            "epoch": epoch,
            "best_f1": best_f1,
            "validation": _metric_payload(validation_result),
            "model_parameters": model_parameter_count(model),
            "contract_hashes": contract_hashes,
            "run_id": run_id,
            "session_id": session_id,
            "resume_from_epoch": start_epoch - 1,
        }
        write_json(output / "checkpoint_metadata.json", checkpoint_metadata)
        uploader.upload(epoch_path, f"checkpoints/{epoch_path.name}")
        uploader.upload(output / "last_checkpoint.pt", "checkpoints/last_checkpoint.pt")
        if improved:
            uploader.upload(output / "best_model.pt", "checkpoints/best_model.pt")
        uploader.upload(output / "checkpoint_metadata.json", "checkpoints/checkpoint_metadata.json")
        if stop_after_epoch is not None and epoch >= int(stop_after_epoch) and epoch < int(epochs):
            write_json(output / "history.json", history)
            return {
                "status": "controlled_stop",
                "completed_epoch": epoch,
                "next_epoch": epoch + 1,
                "resume_checkpoint": str(output / "last_checkpoint.pt"),
                "run_id": run_id,
                "session_id": session_id,
                "contract_hashes": contract_hashes,
            }

    training_seconds = time.perf_counter() - training_started
    final_path = output / f"final_model_epoch_{int(epochs):03d}.pt"
    shutil.copy2(output / "last_checkpoint.pt", final_path)
    final_checkpoint = torch.load(final_path, map_location=device, weights_only=False)
    model.load_state_dict(final_checkpoint["model"])
    test_result = evaluate(model, evaluation_loaders["test"], criterion, device, class_count)
    artifact_paths = _write_final_artifacts(
        output, history, test_result, class_names, training_seconds, peak_memory_mb
    )
    run_config = {
        "arguments": run_arguments,
        "device": str(device),
        "model_parameters": model_parameter_count(model),
        "class_names": class_names,
        "feature_count": feature_count,
        "sequence_counts": {split: len(bundle.sequence_x) for split, bundle in bundles.items()},
        "training_time_seconds": training_seconds,
        "final_epoch": int(final_checkpoint["epoch"]),
        "best_validation_macro_f1": float(final_checkpoint["best_f1"]),
        "run_id": run_id,
        "session_id": session_id,
        "resumed_from_epoch": start_epoch - 1,
        "contract_hashes": contract_hashes,
    }
    write_json(output / "run_config.json", run_config)
    artifact_paths.extend([output / "run_config.json", output / "checkpoint_metadata.json"])
    artifact_paths.append(final_path)
    for artifact in artifact_paths:
        uploader.upload(artifact, f"artifacts/{artifact.name}")
    return {
        "status": "passed",
        "final_epoch": int(final_checkpoint["epoch"]),
        "best_validation_macro_f1": float(final_checkpoint["best_f1"]),
        "test_metrics": _metric_payload(test_result),
        "training_time_seconds": training_seconds,
        "peak_memory_mb": peak_memory_mb,
        "model_parameters": model_parameter_count(model),
        "run_id": run_id,
        "session_id": session_id,
        "resumed_from_epoch": start_epoch - 1,
        "contract_hashes": contract_hashes,
    }
