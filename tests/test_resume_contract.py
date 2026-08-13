from pathlib import Path

import torch

from src.training import _restore_rng_state, _rng_state, _stable_hash


def test_rng_round_trip_and_stable_hash(tmp_path: Path) -> None:
    torch.manual_seed(19)
    state = _rng_state()
    expected = torch.rand(5)
    _restore_rng_state(state)
    actual = torch.rand(5)
    assert torch.equal(actual, expected)
    assert _stable_hash({"b": 2, "a": 1}) == _stable_hash({"a": 1, "b": 2})
