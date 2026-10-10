"""Tests estocasticos D3: latencia, estabilidad, sensibilidad, cotas."""

from __future__ import annotations

import time

import numpy as np

from src.quant_risk.poisson_hazards import IncidentHazardModel
from src.quant_risk.sde_grip import TyreDegradationSDE


def test_monte_carlo_latency_under_100ms():
    sde = TyreDegradationSDE()
    sde.simulate_monte_carlo(0.95, 95.0, "MEDIUM", laps_ahead=5, n_simulations=500)
    t0 = time.perf_counter()
    out = sde.simulate_monte_carlo(0.95, 95.0, "MEDIUM", laps_ahead=20, n_simulations=10000)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    assert dt_ms < 100.0, f"MC lento: {dt_ms:.1f} ms >= 100 ms"
    assert out["trajectories"].shape == (10000, 20)
    assert out["p5"].shape == out["p50"].shape == out["p95"].shape == (20,)
    assert np.all(out["trajectories"] >= 0.0) and np.all(out["trajectories"] <= 1.0)
    assert np.all(out["p5"] <= out["p50"]) and np.all(out["p50"] <= out["p95"])


def test_median_grip_monotone_decreasing():
    sde = TyreDegradationSDE()
    out = sde.simulate_monte_carlo(0.95, 95.0, "MEDIUM", laps_ahead=20,
                                   n_simulations=10000, seed=7)
    p50 = out["p50"]
    assert p50[14] < 0.95, f"p50 vuelta 15 no degrada: {p50[14]}"
    assert p50[-1] < p50[0], "mediana no monotona decreciente"


def test_thermal_sensitivity_hot_vs_optimal():
    sde = TyreDegradationSDE()
    opt = sde.simulate_monte_carlo(0.95, 95.0, "MEDIUM", laps_ahead=20,
                                   n_simulations=10000, seed=11)
    hot = sde.simulate_monte_carlo(0.95, 125.0, "MEDIUM", laps_ahead=20,
                                   n_simulations=10000, seed=11)
    assert hot["p50"][-1] + 0.05 < opt["p50"][-1], (
        f"125C no degrada mas: hot={hot['p50'][-1]:.3f} opt={opt['p50'][-1]:.3f}")
    assert hot["theta_eff"] > 2.0 * opt["theta_eff"]


def test_safety_car_probs_bounded():
    m = IncidentHazardModel()
    probs = m.safety_car_probability(1, horizon_laps=15)
    assert probs.shape == (15,)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert bool(np.all(np.diff(probs) >= 0.0)), "acumulada debe ser no-decreciente"
    assert probs[0] > 0.05, f"V1 deberia ser riesgosa: {probs[0]}"
    wet = m.safety_car_probability(10, horizon_laps=15, wet=True)
    dry = m.safety_car_probability(10, horizon_laps=15, wet=False)
    assert bool(np.all(wet >= dry)), "lluvia debe aumentar riesgo"
    assert float(np.all(wet <= 1.0))
