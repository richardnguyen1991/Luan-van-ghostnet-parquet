import pandas as pd

from src.splits import assign_group_splits, validate_split_integrity


def synthetic_grouped_frame() -> pd.DataFrame:
    rows = []
    for source_index in range(6):
        source = f"capture/source_{source_index}.csv"
        for row_id in range(60):
            rows.append({
                "__source_file_id": source,
                "__source_row_id": row_id,
                "sample_id": f"{source}:{row_id}",
                "Label": "A" if (row_id // 10 + source_index) % 2 == 0 else "B",
            })
    return pd.DataFrame(rows)


def test_groups_are_assigned_before_splits_without_intersection():
    frame = synthetic_grouped_frame()
    result = assign_group_splits(
        frame,
        label_column="Label",
        outer_train_fraction=0.7,
        validation_fraction_of_outer_train=0.1,
        group_rows=10,
        seed=42,
        candidate_attempts=32,
    )
    integrity = validate_split_integrity(result.frame)
    assert integrity["status"] == "passed"
    assert set(result.frame["split"]) == {"train", "validation", "test"}
    per_group = result.frame.groupby("group_id")["split"].nunique()
    assert int(per_group.max()) == 1
    assert result.report["effective_fractions"] == {
        "train": 0.63,
        "validation": 0.06999999999999999,
        "test": 0.30000000000000004,
    }


def test_both_reported_outer_train_scenarios_are_supported():
    frame = synthetic_grouped_frame()
    for fraction in (0.7, 0.8):
        result = assign_group_splits(frame, "Label", fraction, 0.1, 10, 7, 16)
        assert result.report["outer_train_fraction"] == fraction
        assert result.report["integrity"]["status"] == "passed"

