from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.preprocessing import GainImputer, LeakageSafePreprocessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def practical_config():
    config = load_config(
        PROJECT_ROOT / "configs" / "base.yaml",
        PROJECT_ROOT / "configs" / "practical_baseline.yaml",
    )
    config["preprocessing"]["isolation_forest"]["enabled"] = False
    return config


def test_identifier_columns_are_column_name_strings():
    config = practical_config()
    assert config["data"]["identifier_columns"][0] == "Unnamed: 0"
    assert all(isinstance(column, str) for column in config["data"]["identifier_columns"])


def preprocessing_frame() -> pd.DataFrame:
    rows = []
    for split, count, offset in (("train", 20, 0), ("validation", 5, 100), ("test", 5, 200)):
        for index in range(count):
            value = float(index + offset)
            rows.append({
                "split": split,
                "Label": "A" if index % 2 == 0 else "B",
                "f1": value,
                "f1_duplicate": value,
                "f2": np.inf if split == "train" and index == 0 else value * 2,
                "constant": 1.0,
                "Flow ID": f"flow-{split}-{index}",
                "Source IP": "10.0.0.1",
                "Destination IP": "10.0.0.2",
                "Timestamp": "01/12/2018 00:00:00",
                "Unnamed: 0": index,
                "__capture_day": "01-12",
                "__source_file_id": f"{split}.csv",
                "__source_row_id": index,
                "sample_id": f"{split}:{index}",
                "group_id": f"{split}:0",
            })
    return pd.DataFrame(rows)


def test_preprocessing_fits_only_train_and_preserves_validation_test_rows(tmp_path: Path):
    config = practical_config()
    processor = LeakageSafePreprocessor(config, seed=42)
    result = processor.fit_transform_splits(preprocessing_frame(), "Label")
    assert result.train_x.shape[0] == 20
    assert result.validation_x.shape[0] == 5
    assert result.test_x.shape[0] == 5
    assert "f1_duplicate" in result.metadata["dropped_columns"]["duplicate_train"]
    assert "constant" in result.metadata["dropped_columns"]["constant_train"]
    assert "Flow ID" not in result.metadata["feature_order"]
    assert float(processor.scaler.data_max_[result.metadata["feature_order"].index("f1")]) == 19.0
    assert result.validation_x.max() > 1.0
    processor.save(tmp_path, result.metadata)
    assert (tmp_path / "preprocessing.json").exists()
    assert (tmp_path / "preprocessor.joblib").exists()


def test_gain_skips_without_missing_values():
    values = np.arange(24, dtype=np.float32).reshape(8, 3)
    imputer = GainImputer(epochs=1, batch_size=4, hint_rate=0.9, alpha=10, learning_rate=0.001,
                          hidden_multiplier=1.0, seed=42)
    transformed = imputer.fit_transform(values)
    assert imputer.skipped_no_missing
    np.testing.assert_allclose(transformed, values)


def test_gain_imputes_missing_values_with_real_generator():
    values = np.array([
        [0.0, 1.0, np.nan],
        [1.0, np.nan, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
    ], dtype=np.float32)
    imputer = GainImputer(epochs=2, batch_size=2, hint_rate=0.9, alpha=10, learning_rate=0.001,
                          hidden_multiplier=1.0, seed=42)
    transformed = imputer.fit_transform(values)
    assert not np.isnan(transformed).any()
    np.testing.assert_allclose(transformed[~np.isnan(values)], values[~np.isnan(values)], rtol=1e-6)
