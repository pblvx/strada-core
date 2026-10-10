"""MLP compacto PINN termico: 5 -> 64(GELU) x3 -> 2."""

from __future__ import annotations

import torch
from torch import Tensor, nn

INPUT_MEAN = torch.tensor([180.0, 50.0, 10.0, 5.0, 9000.0])
INPUT_STD = torch.tensor([90.0, 40.0, 25.0, 2.5, 3500.0])


class ThermalPINN(nn.Module):
    """Red compacta (~12.7k params) para latencia CPU < 3 ms en ONNX.

    La cabeza lineal predice en espacio estandarizado y un affine fijo
    (buffers, exportados a ONNX) devuelve grados C. Asi el MSE esta bien
    condicionado y la perdida fisica opera en unidades fisicas.
    """

    def __init__(self, hidden: int = 64, n_layers: int = 3,
                 out_mean: tuple[float, float] = (85.0, 320.0),
                 out_std: tuple[float, float] = (15.0, 150.0)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = 5
        for _ in range(n_layers):
            layers += [nn.Linear(dim, hidden), nn.GELU()]
            dim = hidden
        layers.append(nn.Linear(hidden, 2))
        self.net = nn.Sequential(*layers)
        self.register_buffer("out_mean", torch.tensor(out_mean, dtype=torch.float32))
        self.register_buffer("out_std", torch.tensor(out_std, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        mean = INPUT_MEAN.to(dtype=x.dtype, device=x.device)
        std = INPUT_STD.to(dtype=x.dtype, device=x.device)
        z = self.net((x - mean) / std)
        om = self.out_mean.to(dtype=x.dtype, device=x.device)
        os = self.out_std.to(dtype=x.dtype, device=x.device)
        return z * os + om
