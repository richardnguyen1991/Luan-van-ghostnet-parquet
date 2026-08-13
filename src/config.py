from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(base_path: str | Path, mode_path: str | Path | None = None) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency is present on Kaggle/project install
        raise ImportError("PyYAML is required to load project configuration") from exc
    base = yaml.safe_load(Path(base_path).read_text(encoding="utf-8")) or {}
    if mode_path is not None:
        override = yaml.safe_load(Path(mode_path).read_text(encoding="utf-8")) or {}
        base = deep_merge(base, override)
    validate_config(base)
    return base


def validate_config(config: dict[str, Any]) -> None:
    required = ["project", "data", "splits", "preprocessing"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing config sections: {missing}")
    fractions = config["splits"]["outer_train_fractions"]
    if sorted(float(x) for x in fractions) != [0.7, 0.8]:
        raise ValueError("outer_train_fractions must contain exactly 0.70 and 0.80")
    validation_fraction = float(config["splits"]["validation_fraction_of_outer_train"])
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction_of_outer_train must be in (0,1)")
    group_rows = int(config["data"]["sequence_group_rows"])
    if group_rows <= 0:
        raise ValueError("sequence_group_rows must be positive")
    for field in (
        "label_candidates",
        "timestamp_candidates",
        "provenance_columns",
        "identifier_columns",
        "non_numeric_feature_columns",
    ):
        values = config["data"].get(field, [])
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise ValueError(f"data.{field} must be a list of column-name strings")
    strategy = config["preprocessing"].get("missing_strategy")
    if strategy is not None and strategy not in {"median", "gain"}:
        raise ValueError("missing_strategy must be median or gain")
    if "step3" in config:
        step3 = config["step3"]
        if int(step3["sequence_length"]) < 2:
            raise ValueError("step3.sequence_length must be at least 2")
        if int(step3["sequence_stride"]) <= 0:
            raise ValueError("step3.sequence_stride must be positive")
        if step3.get("target_rule") != "last_timestep":
            raise ValueError("Only step3.target_rule=last_timestep is currently supported")
        if step3.get("missing_endpoint_policy") != "drop_window":
            raise ValueError("Only step3.missing_endpoint_policy=drop_window is currently supported")
        if step3.get("graph_direction") != "directed":
            raise ValueError("Only step3.graph_direction=directed is currently supported")
        for field in ("source_endpoint_column", "destination_endpoint_column", "endpoint_hash_namespace"):
            if not isinstance(step3.get(field), str) or not step3[field].strip():
                raise ValueError(f"step3.{field} must be a non-empty string")
    if "step4" in config:
        step4 = config["step4"]
        for field in ("flow_embedding_dim", "graph_hidden_dim", "graph_layers", "lstm_hidden_dim",
                      "lstm_layers", "ghost_primary_channels", "ghost_ratio"):
            if int(step4[field]) <= 0:
                raise ValueError(f"step4.{field} must be positive")
        if not 0.0 <= float(step4["dropout"]) < 1.0:
            raise ValueError("step4.dropout must be in [0,1)")
        if float(step4["outer_train_fraction"]) not in {0.7, 0.8}:
            raise ValueError("step4.outer_train_fraction must be 0.70 or 0.80")
        if step4.get("checkpoint_metric") != "macro_f1":
            raise ValueError("Only step4.checkpoint_metric=macro_f1 is supported")
    if "future_training_contract" in config:
        contract = config["future_training_contract"]
        if contract.get("device") != "cpu":
            raise ValueError("This project is CPU-only; future_training_contract.device must be cpu")
        if bool(contract.get("mixed_precision_on_cuda")):
            raise ValueError("CUDA mixed precision must be disabled for the CPU-only project")
        if not bool(contract.get("deterministic_algorithms")):
            raise ValueError("Step 5 resume validation requires deterministic algorithms")


def config_hash(config: dict[str, Any]) -> str:
    serialized = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
