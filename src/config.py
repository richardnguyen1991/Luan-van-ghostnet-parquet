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


def config_hash(config: dict[str, Any]) -> str:
    serialized = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
