"""Perdida fisica PINN termica STRADA-CORE (SciML).

Loss = MSE_data + lambda_physics * Loss_physics

Fisica (ley de enfriamiento de Newton + calor de friccion/frenada):
    dT/dt = alpha * (brake_pct * speed) + beta * lateral_accel - gamma * (T - T_ambient)

Loss_physics penaliza el residuo entre dT/dt (via torch.autograd respecto
al tiempo) y el lado derecho fisico. Todo en tensores torch (regla #2:
torch solo aqui/entrenamiento, nunca en src/api/server.py).
"""

from __future__ import annotations

import torch
from torch import Tensor


def physics_rhs(
    t_pred: Tensor,
    brake_pct: Tensor,
    speed: Tensor,
    lateral_accel: Tensor,
    t_ambient: float | Tensor = 25.0,
    alpha: float = 1.0e-4,
    beta: float = 1.0e-2,
    gamma: float = 5.0e-2,
) -> Tensor:
    """Lado derecho del ODE termico."""
    if not isinstance(t_ambient, Tensor):
        t_ambient = torch.as_tensor(t_ambient, dtype=t_pred.dtype, device=t_pred.device)
    return (
        alpha * brake_pct * speed
        + beta * lateral_accel
        - gamma * (t_pred - t_ambient)
    )


def autograd_dtdt(t_pred: Tensor, time: Tensor) -> Tensor:
    """dT/dt via autograd; 0 si T no depende de t (red feedforward pura)."""
    if not time.requires_grad:
        return torch.zeros_like(t_pred)
    grad = torch.autograd.grad(
        outputs=t_pred,
        inputs=time,
        grad_outputs=torch.ones_like(t_pred),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if grad is None:
        return torch.zeros_like(t_pred)
    return grad


def physics_loss(
    t_pred: Tensor,
    time: Tensor,
    brake_pct: Tensor,
    speed: Tensor,
    lateral_accel: Tensor,
    t_ambient: float = 25.0,
    alpha: float = 1.0e-4,
    beta: float = 1.0e-2,
    gamma: float = 5.0e-2,
) -> Tensor:
    """MSE del residuo fisico (escalar)."""
    dtdt = autograd_dtdt(t_pred, time)
    rhs = physics_rhs(t_pred, brake_pct, speed, lateral_accel, t_ambient, alpha, beta, gamma)
    return torch.mean((dtdt - rhs) ** 2)


def combined_loss(
    t_pred: Tensor,
    t_target: Tensor,
    time: Tensor,
    brake_pct: Tensor,
    speed: Tensor,
    lateral_accel: Tensor,
    lambda_physics: float = 0.1,
    t_ambient: float = 25.0,
    alpha: float = 1.0e-4,
    beta: float = 1.0e-2,
    gamma: float = 5.0e-2,
) -> tuple[Tensor, Tensor, Tensor]:
    """Devuelve (total, mse_data, loss_physics)."""
    mse_data = torch.mean((t_pred - t_target) ** 2)
    l_phys = physics_loss(t_pred, time, brake_pct, speed, lateral_accel, t_ambient, alpha, beta, gamma)
    return mse_data + lambda_physics * l_phys, mse_data, l_phys
