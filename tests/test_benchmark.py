import torch

from src.benchmark import EdgeMLP


def test_edge_model_output_shape() -> None:
    model = EdgeMLP(features=8, classes=3)
    assert model(torch.randn(2, 8)).shape == (2, 3)
