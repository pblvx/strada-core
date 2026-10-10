"""SDE Ornstein-Uhlenbeck de degradacion de neumatico (numpy vectorizado).

dG_t = -theta * (G_t - mu_base(tyre_age)) * dt + sigma * dW_t
"""

from __future__ import annotations

import numpy as np

COMPOUNDS: dict[str, dict[str, float]] = {
    "SOFT": {"wear": 0.028, "theta0": 0.55, "sigma": 0.022, "mu0": 0.98},
    "MEDIUM": {"wear": 0.018, "theta0": 0.45, "sigma": 0.016, "mu0": 0.98},
    "HARD": {"wear": 0.012, "theta0": 0.35, "sigma": 0.012, "mu0": 0.98},
}

T_OPT_LO = 80.0
T_OPT_HI = 105.0
K_HOT = 3.2
K_COLD = 1.2
# Penalización directa al agarre estacionario fuera de ventana (puntos de grip).
MU_HOT_SLOPE = 0.006
MU_COLD_SLOPE = 0.004


class TyreDegradationSDE:
    """Deriva OU hacia un mu(edad) decreciente; theta sensible a T (PINN)."""

    def __init__(self) -> None:
        pass

    @staticmethod
    def _norm_compound(compound: str) -> str:
        if not isinstance(compound, str) or not compound.strip():
            raise ValueError("compound debe ser str no vacio")
        c = compound.strip().upper()
        if c not in COMPOUNDS:
            raise ValueError(f"Compuesto desconocido: {compound!r} (SOFT/MEDIUM/HARD)")
        return c

    @staticmethod
    def thermal_multiplier(tyre_temp_c: float) -> float:
        t = float(tyre_temp_c)
        if t > T_OPT_HI:
            return 1.0 + K_HOT * (t - T_OPT_HI) / 20.0
        if t < T_OPT_LO:
            return 1.0 + K_COLD * (T_OPT_LO - t) / 20.0
        return 1.0

    @staticmethod
    def mu_base(tyre_age_laps: int, compound: str, tyre_temp_c: float | None = None) -> float:
        c = TyreDegradationSDE._norm_compound(compound)
        p = COMPOUNDS[c]
        mu = p["mu0"] - p["wear"] * float(max(int(tyre_age_laps), 0))
        if tyre_temp_c is not None:
            t = float(tyre_temp_c)
            if t > T_OPT_HI:
                mu -= MU_HOT_SLOPE * (t - T_OPT_HI)
            elif t < T_OPT_LO:
                mu -= MU_COLD_SLOPE * (T_OPT_LO - t)
        return float(np.clip(mu, 0.05, 1.0))

    def simulate_monte_carlo(
        self,
        current_grip: float,
        tyre_temp_c: float,
        compound: str,
        laps_ahead: int = 20,
        n_simulations: int = 10000,
        tyre_age_laps: int = 0,
        seed: int | None = None,
    ) -> dict:
        g0 = float(current_grip)
        if not 0.0 <= g0 <= 1.0:
            raise ValueError("current_grip debe estar en [0, 1]")
        H = int(laps_ahead)
        N = int(n_simulations)
        if H < 1 or N < 1:
            raise ValueError("laps_ahead y n_simulations deben ser >= 1")
        c = self._norm_compound(compound)
        p = COMPOUNDS[c]
        theta = float(p["theta0"] * self.thermal_multiplier(float(tyre_temp_c)))
        sigma = float(p["sigma"])
        wear = float(p["wear"])
        mu0 = float(p["mu0"])
        age0 = int(max(int(tyre_age_laps), 0))

        rng = np.random.default_rng(seed)
        # Mus por vuelta futura: desgaste + penalizacion termica fuera de ventana.
        ks = np.arange(1, H + 1, dtype=np.float64)
        mus = np.clip(mu0 - wear * (age0 + ks), 0.05, 1.0)
        t = float(tyre_temp_c)
        if t > T_OPT_HI:
            mus = np.clip(mus - MU_HOT_SLOPE * (t - T_OPT_HI), 0.05, 1.0)
        elif t < T_OPT_LO:
            mus = np.clip(mus - MU_COLD_SLOPE * (T_OPT_LO - t), 0.05, 1.0)

        z = rng.standard_normal((N, H)).astype(np.float64)
        traj = np.empty((N, H), dtype=np.float64)
        g = np.full((N,), g0, dtype=np.float64)
        dt = 1.0
        sq = np.sqrt(dt)
        # Bucle SOLO sobre horizontes (H<=~30); sims 100% vectorizadas.
        for k in range(H):
            g = g - theta * (g - mus[k]) * dt + sigma * sq * z[:, k]
            np.clip(g, 0.0, 1.0, out=g)
            traj[:, k] = g

        p5 = np.percentile(traj, 5, axis=0)
        p50 = np.percentile(traj, 50, axis=0)
        p95 = np.percentile(traj, 95, axis=0)
        return {
            "p5": np.asarray(p5, dtype=np.float64),
            "p50": np.asarray(p50, dtype=np.float64),
            "p95": np.asarray(p95, dtype=np.float64),
            "trajectories": traj,
            "theta_eff": theta,
            "mus": mus,
        }
