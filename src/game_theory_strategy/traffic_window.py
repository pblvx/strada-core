"""Tráfico en pista y ventana de parada (Silverstone, numpy vectorizado)."""

from __future__ import annotations

import numpy as np

PIT_LOSS_S = 22.0
SC_PIT_SAVING_S = 10.0
DIRTY_GAP_S = 1.5
DIRTY_LOSS_S_PER_LAP = 1.2
DIRTY_DEG_MULT = 1.25


class TrafficPredictor:
    """Penalizacion por aire sucio y ratio de aire limpio."""

    def __init__(
        self,
        pit_loss_s: float = PIT_LOSS_S,
        dirty_gap_s: float = DIRTY_GAP_S,
        dirty_loss_per_lap: float = DIRTY_LOSS_S_PER_LAP,
        dirty_deg_mult: float = DIRTY_DEG_MULT,
    ) -> None:
        if pit_loss_s <= 0:
            raise ValueError("pit_loss_s > 0")
        if dirty_gap_s <= 0:
            raise ValueError("dirty_gap_s > 0")
        self.pit_loss_s = float(pit_loss_s)
        self.dirty_gap_s = float(dirty_gap_s)
        self.dirty_loss_per_lap = float(dirty_loss_per_lap)
        self.dirty_deg_mult = float(dirty_deg_mult)

    @staticmethod
    def _check_gaps(gaps: np.ndarray) -> np.ndarray:
        g = np.asarray(gaps, dtype=np.float64)
        if g.size == 0:
            raise ValueError("gaps vacio")
        if bool(np.any(~np.isfinite(g))):
            raise ValueError("gaps con NaN/inf")
        return g

    def in_dirty_air(self, rejoin_gap_s: float) -> bool:
        gap = float(rejoin_gap_s)
        if not np.isfinite(gap):
            raise ValueError("gap no finito")
        return gap < self.dirty_gap_s

    def clean_air_ratio(self, gaps: np.ndarray) -> float:
        g = self._check_gaps(gaps)
        return float(np.mean(g >= self.dirty_gap_s))

    def traffic_risk(self, gaps: np.ndarray) -> float:
        return float(1.0 - self.clean_air_ratio(gaps))

    def projected_loss(self, n_laps: int, rejoin_gap_s: float) -> float:
        n = int(n_laps)
        if n < 0:
            raise ValueError("n_laps >= 0")
        if self.in_dirty_air(rejoin_gap_s):
            return float(self.dirty_loss_per_lap * n)
        return 0.0
