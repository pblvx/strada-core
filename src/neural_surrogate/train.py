"""Entrenamiento PINN termica + export ONNX (batch, CPU).

Etiquetas: el parquet no trae temperaturas reales; se generan
pseudo-etiquetas integrando el mismo ODE fisico (Euler a 50 Hz) + ruido.
El termino fisico via autograd regulariza hacia la termodinamica.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl
import torch

from src.neural_surrogate.physics_loss import combined_loss
from src.neural_surrogate.pinn_model import ThermalPINN

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("pinn_train")

ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "data" / "raw" / "silverstone_2024.parquet"
ONNX_OUT = ROOT / "data" / "models" / "pinn_thermal.onnx"

EPOCHS = 40
LR = 3e-3
LAMBDA_PHYS = 0.01
T_AMBIENT = 25.0
P_TYRE = dict(alpha=2.0e-5, beta=8.0e-3, gamma=8.0e-2)
P_BRAKE = dict(alpha=1.5e-3, beta=2.0e-3, gamma=1.5e-2)
N_TRAIN = 20000
DT = 0.02


def lateral_accel(x: np.ndarray, y: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Aceleracion lateral v^2 * curvatura (diferencias finitas, batch)."""
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = (dx * dx + dy * dy) ** 1.5 + 1e-9
    kappa = np.abs(dx * ddy - dy * ddx) / denom
    return (v ** 2) * kappa


def synth_targets(df: pl.DataFrame, rng: np.random.Generator):
    """Objetivo estacionario del ODE + transitorio: monotono en brake*speed.

    T_ss = T_amb + (alpha * brake * v + beta * lat) / gamma es el punto fijo
    del ODE; se mezcla con integracion Euler (inercia termica) para que la
    red aprenda la dependencia instantanea sin colapsar a la media.
    """
    n = len(df)
    speed = df["speed_kmh"].to_numpy().astype(np.float64) / 3.6
    brake = df["brake_pct"].to_numpy().astype(np.float64)
    x = df["x"].to_numpy().astype(np.float64)
    y = df["y"].to_numpy().astype(np.float64)
    lat = lateral_accel(x, y, speed)
    heat_b = brake * speed
    ss_tyre = T_AMBIENT + (P_TYRE["alpha"] * heat_b + P_TYRE["beta"] * lat) / P_TYRE["gamma"]
    ss_brake = T_AMBIENT + (P_BRAKE["alpha"] * heat_b + P_BRAKE["beta"] * lat) / P_BRAKE["gamma"]
    ss_tyre = np.clip(ss_tyre, 60.0, 130.0)
    ss_brake = np.clip(ss_brake, 150.0, 750.0)
    t_tyre = np.full(n, 85.0)
    t_brake = np.full(n, 320.0)
    k_tyre, k_brake = 0.15, 0.10  # mezcla estado/transitorio
    for i in range(1, n):
        t_tyre[i] = t_tyre[i - 1] + DT * (
            P_TYRE["alpha"] * heat_b[i - 1] + P_TYRE["beta"] * lat[i - 1]
            - P_TYRE["gamma"] * (t_tyre[i - 1] - T_AMBIENT))
        t_brake[i] = t_brake[i - 1] + DT * (
            P_BRAKE["alpha"] * heat_b[i - 1] + P_BRAKE["beta"] * lat[i - 1]
            - P_BRAKE["gamma"] * (t_brake[i - 1] - T_AMBIENT))
        t_tyre[i] = (1 - k_tyre) * t_tyre[i] + k_tyre * ss_tyre[i]
        t_brake[i] = (1 - k_brake) * t_brake[i] + k_brake * ss_brake[i]
    t_tyre += rng.normal(0, 1.5, n)
    t_brake += rng.normal(0, 8.0, n)
    return t_tyre.astype(np.float32), t_brake.astype(np.float32)


def load_dataset() -> dict[str, torch.Tensor]:
    df = pl.read_parquet(str(PARQUET))
    rng = np.random.default_rng(7)
    tyre_t, brake_t = synth_targets(df, rng)
    out_mean = np.array([float(tyre_t.mean()), float(brake_t.mean())], dtype=np.float32)
    out_std = np.array([max(float(tyre_t.std()), 1e-6), max(float(brake_t.std()), 1e-6)], dtype=np.float32)
    log.info("Norm salida: mean=%s std=%s", out_mean.tolist(), out_std.tolist())
    x = df["x"].to_numpy().astype(np.float64)
    y = df["y"].to_numpy().astype(np.float64)
    v_ms = df["speed_kmh"].to_numpy().astype(np.float64) / 3.6
    lat = lateral_accel(x, y, v_ms).astype(np.float32)
    n = len(df)
    brake_all = df["brake_pct"].to_numpy()
    hi_mask = np.nonzero(brake_all > 50.0)[0]
    n_hi = min(len(hi_mask), N_TRAIN // 2)
    idx_hi = rng.choice(hi_mask, size=n_hi, replace=False) if n_hi > 0 else np.array([], dtype=np.int64)
    idx_rest = rng.choice(n, size=min(N_TRAIN, n) - len(idx_hi), replace=False)
    idx = np.sort(np.unique(np.concatenate([idx_hi, idx_rest])))

    # Anclas sinteticas: 2000 puntos uniformes del espacio de frenada para
    # forzar monotonia fisica (a mas brake*speed, mas temperatura).
    n_anchor = 2000
    a_speed = rng.uniform(0, 330, n_anchor).astype(np.float32)
    a_thr = rng.uniform(0, 100, n_anchor).astype(np.float32)
    a_brake = rng.uniform(0, 100, n_anchor).astype(np.float32)
    a_gear = rng.integers(1, 9, n_anchor).astype(np.float32)
    a_rpm = rng.uniform(7000, 13500, n_anchor).astype(np.float32)
    a_lat = np.zeros(n_anchor, dtype=np.float32)
    a_tyre = (60.0 + 0.06 * a_speed + 0.15 * a_brake * a_speed / 50.0).astype(np.float32)
    a_brake_t = (150.0 + 0.9 * a_speed + 4.5 * a_brake * a_speed / 50.0).astype(np.float32)
    feats = np.stack([
        df["speed_kmh"].to_numpy(), df["throttle_pct"].to_numpy(),
        df["brake_pct"].to_numpy(), df["gear"].to_numpy(),
        df["rpm"].to_numpy(),
    ], axis=1).astype(np.float32)[idx]
    feats = np.concatenate([feats, np.stack(
        [a_speed, a_thr, a_brake, a_gear, a_rpm], axis=1)], axis=0)
    spd = df["speed_kmh"].to_numpy().astype(np.float32)[idx]
    brk = df["brake_pct"].to_numpy().astype(np.float32)[idx]
    spd = np.concatenate([spd, a_speed], axis=0)
    brk = np.concatenate([brk, a_brake], axis=0)
    lat = np.concatenate([lat[idx], a_lat], axis=0)
    out = {
        "x": torch.from_numpy(feats),
        "tyre": torch.from_numpy(np.concatenate([tyre_t[idx], a_tyre])).unsqueeze(1),
        "brake": torch.from_numpy(np.concatenate([brake_t[idx], a_brake_t])).unsqueeze(1),
        "out_mean": out_mean,
        "out_std": out_std,
        "speed": torch.from_numpy(spd).unsqueeze(1),
        "brake_pct": torch.from_numpy(brk).unsqueeze(1),
        "lat": torch.from_numpy(lat).unsqueeze(1),
    }
    return out


def train() -> Path:
    torch.manual_seed(7)
    data = load_dataset()
    n = len(data["x"])
    t = torch.linspace(0, n * DT, n, dtype=torch.float32).unsqueeze(1)
    model = ThermalPINN(out_mean=tuple(float(v) for v in data["out_mean"]),
                        out_std=tuple(float(v) for v in data["out_std"]))
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    for ep in range(1, EPOCHS + 1):
        t_req = t.clone().detach().requires_grad_(True)
        opt.zero_grad()
        pred = model(data["x"])
        tot_t, mse_t, phys_t = combined_loss(
            pred[:, 0:1], data["tyre"], t_req, data["brake_pct"],
            data["speed"], data["lat"], LAMBDA_PHYS, T_AMBIENT, **P_TYRE)
        tot_b, mse_b, phys_b = combined_loss(
            pred[:, 1:2], data["brake"], t_req, data["brake_pct"],
            data["speed"], data["lat"], LAMBDA_PHYS, T_AMBIENT, **P_BRAKE)
        total = tot_t + tot_b
        total.backward()
        opt.step()
        if ep == 1 or ep % 5 == 0 or ep == EPOCHS:
            with torch.no_grad():
                mae = (torch.mean(torch.abs(pred[:, 0] - data["tyre"][:, 0]))
                       + torch.mean(torch.abs(pred[:, 1] - data["brake"][:, 0]))) / 2
            log.info("ep %02d/%d loss=%.4f mse=%.4f phys=%.4f MAE=%.2fC",
                     ep, EPOCHS, total.item(), (mse_t + mse_b).item() / 2,
                     (phys_t + phys_b).item() / 2, mae.item())
    with torch.no_grad():
        pred = model(data["x"])
        mae = (torch.mean(torch.abs(pred[:, 0] - data["tyre"][:, 0]))
               + torch.mean(torch.abs(pred[:, 1] - data["brake"][:, 0]))) / 2
    log.info("MAE final: %.3f C", mae.item())
    ONNX_OUT.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.zeros(1, 5, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            model, dummy, str(ONNX_OUT), input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=17, do_constant_folding=True, dynamo=False)
    log.info("ONNX exportado a %s", ONNX_OUT)
    import onnxruntime as ort

    sess = ort.InferenceSession(str(ONNX_OUT), providers=["CPUExecutionProvider"])
    sample = data["x"][:8].numpy()
    torch_out = model(data["x"][:8]).detach().numpy()
    ort_out = sess.run(None, {"input": sample})[0]
    diff = float(np.max(np.abs(torch_out - ort_out)))
    log.info("Verificacion torch vs ORT: max abs diff=%.2e", diff)
    assert diff < 1e-4, f"Divergencia ONNX: {diff}"
    return ONNX_OUT


if __name__ == "__main__":
    train()
