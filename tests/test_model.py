import numpy as np
import torch

from src.config import load_config
from src.model import GCLSTMGhostNet, GraphBatch


def test_model_forward_and_backward() -> None:
    config = load_config("configs/base.yaml", "configs/practical_baseline.yaml")
    model = GCLSTMGhostNet(feature_count=6, class_count=3, config=config)
    edges = tuple(torch.tensor([[0, 1, 0, 2], [1, 2, 2, 1]]) for _ in range(2))
    batch = GraphBatch(
        sequence_x=torch.from_numpy(np.random.default_rng(3).normal(size=(2, 4, 6))).float(),
        target_y=torch.tensor([0, 2]),
        edge_index=edges,
        node_counts=torch.tensor([3, 3]),
    )
    logits, attention = model(batch)
    assert logits.shape == (2, 3)
    assert attention.shape == (2, 4)
    assert torch.allclose(attention.sum(dim=1), torch.ones(2), atol=1e-5)
    assert len(model.last_spatial_attention) == 2
    assert all(torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-5)
               for weights in model.last_spatial_attention)
    torch.nn.functional.cross_entropy(logits, batch.target_y).backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
