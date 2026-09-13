from __future__ import annotations

import io
import time

import torch
from torch import nn


class EdgeMLP(nn.Module):
    def __init__(self, features: int = 128, classes: int = 4) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def serialized_bytes(model: nn.Module) -> int:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell()


def average_latency(model: nn.Module, inputs: torch.Tensor, repeats: int = 50) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            model(inputs)
        start = time.perf_counter()
        for _ in range(repeats):
            model(inputs)
    return (time.perf_counter() - start) / repeats * 1000


def main() -> None:
    torch.manual_seed(7)
    baseline = EdgeMLP().eval()
    quantized = torch.ao.quantization.quantize_dynamic(
        baseline, {nn.Linear}, dtype=torch.qint8
    ).eval()
    inputs = torch.randn(1, 128)
    print('fp32 bytes:', serialized_bytes(baseline))
    print('int8 bytes:', serialized_bytes(quantized))
    print('fp32 latency ms:', round(average_latency(baseline, inputs), 3))
    print('int8 latency ms:', round(average_latency(quantized, inputs), 3))


if __name__ == '__main__':
    main()
