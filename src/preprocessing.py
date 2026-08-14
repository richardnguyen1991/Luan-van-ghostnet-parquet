from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from torch import nn


class _GainGenerator(nn.Module):
    def __init__(self, dimension: int, hidden: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dimension * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dimension),
            nn.Sigmoid(),
        )

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([values, mask], dim=1))


class _GainDiscriminator(nn.Module):
    def __init__(self, dimension: int, hidden: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dimension * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dimension),
            nn.Sigmoid(),
        )

    def forward(self, values: torch.Tensor, hint: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([values, hint], dim=1))


class GainImputer:
    """Train-only GAIN imputer with deterministic transform-time inference."""

    def __init__(
        self,
        epochs: int,
        batch_size: int,
        hint_rate: float,
        alpha: float,
        learning_rate: float,
        hidden_multiplier: float,
        seed: int,
        device: str = "cpu",
    ) -> None:
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.hint_rate = float(hint_rate)
        self.alpha = float(alpha)
        self.learning_rate = float(learning_rate)
        self.hidden_multiplier = float(hidden_multiplier)
        self.seed = int(seed)
        self.device = torch.device(device)
        self.generator: _GainGenerator | None = None
        self.discriminator: _GainDiscriminator | None = None
        self.minimum: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.skipped_no_missing = False
        self.history: list[dict[str, float]] = []

    def fit(self, values: np.ndarray) -> "GainImputer":
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError("GAIN input must be a 2D array")
        if np.isnan(array).all(axis=0).any():
            raise ValueError("GAIN cannot fit a feature that is entirely missing")
        self.minimum = np.nanmin(array, axis=0)
        maximum = np.nanmax(array, axis=0)
        self.scale = np.where(maximum > self.minimum, maximum - self.minimum, 1.0).astype(np.float32)
        if not np.isnan(array).any():
            self.skipped_no_missing = True
            return self
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("GAIN epochs and batch_size must be positive")
        if not 0.0 < self.hint_rate < 1.0:
            raise ValueError("GAIN hint_rate must be in (0,1)")
        normalized = (array - self.minimum) / self.scale
        mask = (~np.isnan(normalized)).astype(np.float32)
        normalized = np.nan_to_num(normalized, nan=0.0).astype(np.float32)
        dimension = normalized.shape[1]
        hidden = max(8, int(round(dimension * self.hidden_multiplier)))
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.generator = _GainGenerator(dimension, hidden).to(self.device)
        self.discriminator = _GainDiscriminator(dimension, hidden).to(self.device)
        generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.learning_rate)
        discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.learning_rate
        )
        epsilon = 1e-8
        rng = np.random.default_rng(self.seed)
        for epoch in range(self.epochs):
            permutation = rng.permutation(len(normalized))
            discriminator_total = 0.0
            generator_total = 0.0
            batches = 0
            for start in range(0, len(permutation), self.batch_size):
                indices = permutation[start : start + self.batch_size]
                x = torch.from_numpy(normalized[indices]).to(self.device)
                m = torch.from_numpy(mask[indices]).to(self.device)
                noise = torch.rand_like(x) * 0.01
                x_tilde = m * x + (1.0 - m) * noise
                hint = m * (torch.rand_like(m) < self.hint_rate).float()

                discriminator_optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    generated_detached = self.generator(x_tilde, m)
                    completed_detached = m * x + (1.0 - m) * generated_detached
                predicted_mask = self.discriminator(completed_detached, hint)
                discriminator_loss = -torch.mean(
                    m * torch.log(predicted_mask + epsilon)
                    + (1.0 - m) * torch.log(1.0 - predicted_mask + epsilon)
                )
                discriminator_loss.backward()
                discriminator_optimizer.step()

                generator_optimizer.zero_grad(set_to_none=True)
                generated = self.generator(x_tilde, m)
                completed = m * x + (1.0 - m) * generated
                predicted_mask_for_generator = self.discriminator(completed, hint)
                adversarial = -torch.sum(
                    (1.0 - m) * torch.log(predicted_mask_for_generator + epsilon)
                ) / torch.clamp(torch.sum(1.0 - m), min=1.0)
                reconstruction = torch.sum((m * x - m * generated) ** 2) / torch.clamp(
                    torch.sum(m), min=1.0
                )
                generator_loss = adversarial + self.alpha * reconstruction
                generator_loss.backward()
                generator_optimizer.step()

                discriminator_total += float(discriminator_loss.detach().cpu())
                generator_total += float(generator_loss.detach().cpu())
                batches += 1
            self.history.append({
                "epoch": float(epoch + 1),
                "discriminator_loss": discriminator_total / max(1, batches),
                "generator_loss": generator_total / max(1, batches),
            })
        self.generator.eval()
        self.discriminator.eval()
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.minimum is None or self.scale is None:
            raise RuntimeError("GAIN imputer has not been fitted")
        array = np.asarray(values, dtype=np.float32)
        normalized = (array - self.minimum) / self.scale
        mask = (~np.isnan(normalized)).astype(np.float32)
        if self.skipped_no_missing or not np.isnan(normalized).any():
            return array.astype(np.float64)
        if self.generator is None:
            raise RuntimeError("GAIN generator is unavailable")
        clean = np.nan_to_num(normalized, nan=0.0).astype(np.float32)
        completed_batches: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(clean), self.batch_size):
                x = torch.from_numpy(clean[start : start + self.batch_size]).to(self.device)
                m = torch.from_numpy(mask[start : start + self.batch_size]).to(self.device)
                generated = self.generator(x, m)
                completed = m * x + (1.0 - m) * generated
                completed_batches.append(completed.cpu().numpy())
        completed_normalized = np.concatenate(completed_batches, axis=0)
        completed = completed_normalized * self.scale + self.minimum
        observed = ~np.isnan(array)
        completed[observed] = array[observed]
        return completed.astype(np.float64)

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        return self.fit(values).transform(values)


@dataclass
class PreprocessedSplits:
    train_x: np.ndarray
    validation_x: np.ndarray
    test_x: np.ndarray
    train_y: np.ndarray
    validation_y: np.ndarray
    test_y: np.ndarray
    train_inlier_mask: np.ndarray
    metadata: dict[str, Any]


class LeakageSafePreprocessor:
    def __init__(self, config: dict[str, Any], seed: int) -> None:
        self.config = config
        self.seed = int(seed)
        self.feature_columns: list[str] = []
        self.dropped_columns: dict[str, list[str]] = {}
        self.label_mapping: dict[str, int] = {}
        self.imputer: SimpleImputer | GainImputer | None = None
        self.isolation_forest: IsolationForest | None = None
        self.scaler: MinMaxScaler | None = None
        self.clip_counts: dict[str, int] = {}
        self.fitted = False

    @staticmethod
    def _convert_numeric(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
        result = pd.DataFrame(index=frame.index)
        parse_failures: dict[str, int] = {}
        for column in columns:
            converted = pd.to_numeric(frame[column], errors="coerce")
            failures = int((frame[column].notna() & converted.isna()).sum())
            if failures:
                parse_failures[column] = failures
            result[column] = converted.astype("float64")
        if parse_failures:
            raise ValueError(f"Nonnumeric values in classifier candidate columns: {parse_failures}")
        return result.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _find_duplicate_columns(frame: pd.DataFrame) -> list[str]:
        duplicates: list[str] = []
        representatives: dict[str, list[str]] = {}
        for column in frame.columns:
            digest = hashlib.sha256(
                pd.util.hash_pandas_object(frame[column], index=False).to_numpy().tobytes()
            ).hexdigest()
            matched = False
            for representative in representatives.get(digest, []):
                if frame[column].equals(frame[representative]):
                    duplicates.append(column)
                    matched = True
                    break
            if not matched:
                representatives.setdefault(digest, []).append(column)
        return duplicates

    def _candidate_features(self, frame: pd.DataFrame, label_column: str) -> list[str]:
        excluded = {
            label_column,
            "sample_id",
            "group_id",
            "split",
            *self.config["data"]["identifier_columns"],
            *self.config["data"]["provenance_columns"],
            *self.config["data"].get("non_numeric_feature_columns", []),
        }
        return [column for column in frame.columns if column not in excluded]

    def _fit_feature_contract(self, train: pd.DataFrame, label_column: str) -> pd.DataFrame:
        candidates = self._candidate_features(train, label_column)
        numeric = self._convert_numeric(train, candidates)
        empty = [column for column in numeric if numeric[column].isna().all()]
        numeric = numeric.drop(columns=empty)
        duplicate = self._find_duplicate_columns(numeric) if self.config["preprocessing"]["drop_duplicate_columns"] else []
        numeric = numeric.drop(columns=duplicate)
        constant = [
            column for column in numeric if numeric[column].nunique(dropna=True) <= 1
        ] if self.config["preprocessing"]["drop_constant_columns"] else []
        numeric = numeric.drop(columns=constant)
        if numeric.empty:
            raise ValueError("No numeric classifier features remain")
        self.feature_columns = list(numeric.columns)
        self.dropped_columns = {
            "identity_label_provenance": sorted(set(train.columns) - set(candidates)),
            "all_missing_train": empty,
            "duplicate_train": duplicate,
            "constant_train": constant,
        }
        return numeric

    def _build_imputer(self, dimension: int) -> SimpleImputer | GainImputer:
        strategy = self.config["preprocessing"]["missing_strategy"]
        if strategy == "median":
            return SimpleImputer(strategy="median")
        gain = self.config["preprocessing"]["gain"]
        return GainImputer(
            epochs=gain["epochs"],
            batch_size=gain["batch_size"],
            hint_rate=gain["hint_rate"],
            alpha=gain["alpha"],
            learning_rate=gain["learning_rate"],
            hidden_multiplier=gain["hidden_multiplier"],
            seed=self.seed,
            device="cpu",
        )

    def fit_transform_splits(
        self,
        frame: pd.DataFrame,
        label_column: str,
        label_vocabulary: Sequence[str] | None = None,
    ) -> PreprocessedSplits:
        if "split" not in frame:
            raise ValueError("Split assignments must exist before preprocessing")
        subsets = {split: frame.loc[frame["split"] == split].copy() for split in ("train", "validation", "test")}
        if any(subset.empty for subset in subsets.values()):
            raise ValueError("train, validation, and test must all be non-empty")
        train_numeric = self._fit_feature_contract(subsets["train"], label_column)
        raw = {"train": train_numeric}
        for split in ("validation", "test"):
            raw[split] = self._convert_numeric(subsets[split], self.feature_columns)
        self.imputer = self._build_imputer(len(self.feature_columns))
        train_imputed = self.imputer.fit_transform(raw["train"].to_numpy())
        validation_imputed = self.imputer.transform(raw["validation"].to_numpy())
        test_imputed = self.imputer.transform(raw["test"].to_numpy())

        isolation = self.config["preprocessing"]["isolation_forest"]
        if isolation["enabled"]:
            self.isolation_forest = IsolationForest(
                n_estimators=int(isolation["n_estimators"]),
                contamination=float(isolation["contamination"]),
                random_state=self.seed,
                n_jobs=-1,
            )
            inlier_mask = self.isolation_forest.fit_predict(train_imputed) == 1
        else:
            inlier_mask = np.ones(len(train_imputed), dtype=bool)
        if not inlier_mask.any():
            raise RuntimeError("Isolation Forest removed every training row")
        retained_train = train_imputed[inlier_mask]
        feature_range = tuple(float(x) for x in self.config["preprocessing"]["minmax"]["feature_range"])
        self.scaler = MinMaxScaler(feature_range=feature_range)
        train_scaled = self.scaler.fit_transform(retained_train)
        validation_scaled = self.scaler.transform(validation_imputed)
        test_scaled = self.scaler.transform(test_imputed)
        if self.config["preprocessing"]["minmax"]["clip"]:
            low, high = feature_range
            arrays = {"train": train_scaled, "validation": validation_scaled, "test": test_scaled}
            for split, array in arrays.items():
                self.clip_counts[split] = int(((array < low) | (array > high)).sum())
                np.clip(array, low, high, out=array)
        else:
            self.clip_counts = {"train": 0, "validation": 0, "test": 0}

        observed_labels = sorted(
            frame[label_column].astype("string").dropna().unique().tolist()
        )
        if label_vocabulary is None:
            all_labels = observed_labels
            label_mapping_source = "fit_frame"
        else:
            all_labels = sorted({str(label) for label in label_vocabulary})
            if not all_labels:
                raise ValueError("Provided label vocabulary must not be empty")
            unexpected = sorted(set(observed_labels) - set(all_labels))
            if unexpected:
                raise ValueError(
                    f"Fit frame contains labels absent from provided vocabulary: {unexpected}"
                )
            label_mapping_source = "provided_exhaustive_vocabulary"
        self.label_mapping = {label: index for index, label in enumerate(all_labels)}
        encode = lambda series: series.astype("string").map(self.label_mapping).to_numpy(dtype=np.int64)
        train_y_all = encode(subsets["train"][label_column])
        train_y = train_y_all[inlier_mask]
        validation_y = encode(subsets["validation"][label_column])
        test_y = encode(subsets["test"][label_column])
        self.fitted = True
        missing_status = (
            "skipped_no_missing"
            if isinstance(self.imputer, GainImputer) and self.imputer.skipped_no_missing
            else self.config["preprocessing"]["missing_strategy"]
        )
        metadata = {
            "fit_scope": "train_only",
            "feature_order": self.feature_columns,
            "feature_count": len(self.feature_columns),
            "dropped_columns": self.dropped_columns,
            "missing_strategy_status": missing_status,
            "train_rows_before_outlier_removal": len(train_imputed),
            "train_rows_after_outlier_removal": len(retained_train),
            "validation_rows_unchanged": len(validation_imputed),
            "test_rows_unchanged": len(test_imputed),
            "clip_counts": self.clip_counts,
            "label_mapping": self.label_mapping,
            "label_mapping_source": label_mapping_source,
            "labels_not_observed_in_fit_proxy": sorted(
                set(all_labels) - set(observed_labels)
            ),
            "labels_missing_from_train": sorted(set(all_labels) - set(subsets["train"][label_column].astype("string"))),
            "labels_missing_from_validation": sorted(set(all_labels) - set(subsets["validation"][label_column].astype("string"))),
            "labels_missing_from_test": sorted(set(all_labels) - set(subsets["test"][label_column].astype("string"))),
        }
        return PreprocessedSplits(
            train_x=train_scaled,
            validation_x=validation_scaled,
            test_x=test_scaled,
            train_y=train_y,
            validation_y=validation_y,
            test_y=test_y,
            train_inlier_mask=inlier_mask,
            metadata=metadata,
        )

    def save(self, output_dir: str | Path, metadata: dict[str, Any]) -> None:
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before saving")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output / "preprocessor.joblib")
        payload = dict(metadata)
        payload["feature_order_hash"] = hashlib.sha256(
            json.dumps(self.feature_columns, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        temporary = output / "preprocessing.json.tmp"
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(output / "preprocessing.json")
        if isinstance(self.imputer, GainImputer) and self.imputer.generator is not None:
            torch.save({
                "generator_state_dict": self.imputer.generator.state_dict(),
                "minimum": self.imputer.minimum,
                "scale": self.imputer.scale,
                "history": self.imputer.history,
            }, output / "gain_checkpoint.pt")
