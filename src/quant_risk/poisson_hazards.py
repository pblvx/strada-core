"""Riesgo de incidentes: Poisson no homogeneo para Safety Car."""

from __future__ import annotations

import numpy as np


class IncidentHazardModel:
    """Intensidad lambda(vuelta): base + picos en V1/relanzadas + lluvia."""

    def __init__(
        self,
        base_rate: float = 0.012,
        lap1_boost: float = 0.06,
        restart_boost: float = 0.03,
        opening_decay_laps: int = 3,
        wet_multiplier: float = 1.8,
    ) -> None:
        if base_rate < 0 or lap1_boost < 0 or restart_boost < 0:
            raise ValueError("tasas deben ser >= 0")
        if opening_decay_laps < 1:
            raise ValueError("opening_decay_laps >= 1")
        if wet_multiplier < 1.0:
            raise ValueError("wet_multiplier >= 1.0")
        self.base_rate = float(base_rate)
        self.lap1_boost = float(lap1_boost)
        self.restart_boost = float(restart_boost)
        self.opening_decay_laps = int(opening_decay_laps)
        self.wet_multiplier = float(wet_multiplier)

    def intensity(
        self,
        lap: int | np.ndarray,
        wet: bool = False,
        restart_laps: tuple[int, ...] | list[int] = (),
    ) -> np.ndarray:
        laps = np.atleast_1d(np.asarray(lap, dtype=np.int64))
        lam = np.full(laps.shape, self.base_rate, dtype=np.float64)
        lam = lam + self.lap1_boost * (laps == 1)
        if len(restart_laps) > 0:
            rest = np.asarray(list(restart_laps), dtype=np.int64)
            lam = lam + self.restart_boost * np.isin(laps, rest)
        early = np.clip(self.opening_decay_laps - (laps - 1), 0, self.opening_decay_laps)
        lam = lam + (self.base_rate * 0.5) * (early / self.opening_decay_laps)
        if wet:
            lam = lam * self.wet_multiplier
        return np.clip(lam, 0.0, 1.0)

    def safety_car_probability(
        self,
        current_lap: int,
        horizon_laps: int = 15,
        wet: bool = False,
        restart_laps: tuple[int, ...] | list[int] = (),
    ) -> np.ndarray:
        cur = int(current_lap)
        h = int(horizon_laps)
        if cur < 1:
            raise ValueError("current_lap >= 1")
        if h < 1:
            raise ValueError("horizon_laps >= 1")
        future = np.arange(cur, cur + h, dtype=np.int64)
        lam = self.intensity(future, wet=wet, restart_laps=restart_laps)
        cumhaz = np.cumsum(lam)
        probs = 1.0 - np.exp(-cumhaz)
        return np.clip(probs, 0.0, 1.0)
