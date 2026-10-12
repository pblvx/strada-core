"""MCTS de estrategia de parada: Undercut vs Overcut (<200 ms, numpy)."""

from __future__ import annotations

import numpy as np

from src.core.strategy_schema import StrategyDecision
from src.game_theory_strategy.traffic_window import (
    DIRTY_DEG_MULT,
    SC_PIT_SAVING_S,
    TrafficPredictor,
)
from src.quant_risk.poisson_hazards import IncidentHazardModel
from src.quant_risk.sde_grip import TyreDegradationSDE

ACTIONS = ("BOX_SOFT", "BOX_MEDIUM", "BOX_HARD", "STAY_OUT")
BOX_TO_COMPOUND = {"BOX_SOFT": "SOFT", "BOX_MEDIUM": "MEDIUM", "BOX_HARD": "HARD"}
BASE_LAP_S = 92.0
DEG_SLOPE_S = 14.0
P_RESPOND = 0.65
P_RIVAL_BOX = 0.25


class MCTSRaceStrategist:
    """UCT sobre 4 brazos; rollouts vectorizados con SDE + SC + trafico."""

    def __init__(
        self,
        sde: TyreDegradationSDE | None = None,
        hazard: IncidentHazardModel | None = None,
        traffic: TrafficPredictor | None = None,
        uct_c: float = 1.414,
        horizon: int = 10,
    ) -> None:
        self.sde = sde or TyreDegradationSDE()
        self.hazard = hazard or IncidentHazardModel()
        self.traffic = traffic or TrafficPredictor()
        self.uct_c = float(uct_c)
        self.horizon = int(horizon)
        if self.horizon < 1:
            raise ValueError("horizon >= 1")

    def _grip_curves(self, grip: float, temp: float, compound: str, age: int, seed: int) -> dict:
        curves = {}
        for comp in ("SOFT", "MEDIUM", "HARD"):
            out = self.sde.simulate_monte_carlo(
                grip, temp, comp, laps_ahead=self.horizon,
                n_simulations=300, tyre_age_laps=age, seed=seed + hash(comp) % 997)
            curves[comp] = (out["p50"].astype(np.float64), float(out["trajectories"].std()))
        return curves

    def _rollout_times(self, rng: np.random.Generator, grip_p50: np.ndarray, sig: float,
                       n: int, dirty: bool, pit: bool, sc_p: float) -> np.ndarray:
        pace = BASE_LAP_S + DEG_SLOPE_S * (1.0 - grip_p50)
        if dirty:
            pace = pace + 1.2
        noise = rng.normal(0.0, max(sig * 8.0, 0.35), size=(n, self.horizon))
        times = pace[None, :] + noise
        total = times.sum(axis=1)
        if pit:
            sc = rng.random(n) < sc_p
            total = total + self.traffic.pit_loss_s - sc * SC_PIT_SAVING_S
        return total

    def evaluate_strategy(self, current_lap: int, current_grip: float, tyre_temp_c: float,
                          compound: str, tyre_age: int, rival_gap_s: float,
                          n_simulations: int = 500, seed: int = 7) -> StrategyDecision:
        lap = int(current_lap)
        grip = float(current_grip)
        gap = float(rival_gap_s)
        n = int(n_simulations)
        if lap < 1:
            raise ValueError("current_lap >= 1")
        if not 0.0 <= grip <= 1.0:
            raise ValueError("current_grip en [0,1]")
        if n < 4:
            raise ValueError("n_simulations >= 4")
        comp = compound.strip().upper()
        if comp not in ("SOFT", "MEDIUM", "HARD"):
            raise ValueError("compound SOFT/MEDIUM/HARD")
        rng = np.random.default_rng(seed)
        window = (int(tyre_age) >= 5 or grip < 0.75) and 8 <= lap <= 48
        curves = self._grip_curves(grip, float(tyre_temp_c), comp, int(tyre_age), seed)
        stay_p50, stay_sig = curves[comp]
        sc_probs = self.hazard.safety_car_probability(lap, horizon_laps=self.horizon)
        sc_p = float(np.clip(sc_probs[-1], 0.0, 1.0))
        dirty = gap < 1.5 and gap > -30.0
        risk = float(np.clip(1.0 - 1.0 / (1.0 + np.exp(-(1.5 - abs(gap)) * 2.0)), 0.0, 1.0))
        risk = float(np.clip(risk if abs(gap) < 6.0 else (0.05 if gap >= 6.0 else 0.9), 0.0, 1.0))
        counts = {a: 0 for a in ACTIONS}
        values = {a: 0.0 for a in ACTIONS}
        box_wins = 0
        for i in range(n):
            totals = {}
            for a in ACTIONS:
                if a == "STAY_OUT":
                    t_me = self._rollout_times(rng, stay_p50, stay_sig, 1, dirty, False, sc_p)[0]
                else:
                    cbox = BOX_TO_COMPOUND[a]
                    p50, sig = curves[cbox]
                    fresh = np.clip(0.97 - 0.004 * np.arange(1, self.horizon + 1), 0.0, 1.0)
                    p50 = np.maximum(p50, fresh * 0.98)
                    t_me = self._rollout_times(rng, p50, sig, 1, dirty, True, sc_p)[0]
                rival_boxes = (rng.random() < P_RESPOND) if a != "STAY_OUT" else (rng.random() < P_RIVAL_BOX)
                if rival_boxes:
                    t_riv = BASE_LAP_S * self.horizon + self.traffic.pit_loss_s - (rng.random() < sc_p) * SC_PIT_SAVING_S + rng.normal(0, 0.8)
                else:
                    t_riv = (BASE_LAP_S + DEG_SLOPE_S * 0.25) * self.horizon + rng.normal(0, 0.8)
                totals[a] = (t_riv + gap) - t_me
            if i < len(ACTIONS):
                a = ACTIONS[i]
            else:
                total_n = i
                best, best_u = ACTIONS[0], -1e18
                for a in ACTIONS:
                    avg = values[a] / max(counts[a], 1)
                    u = avg + self.uct_c * float(np.sqrt(max(np.log(total_n), 1e-9) / max(counts[a], 1)))
                    if u > best_u:
                        best, best_u = a, u
                a = best
            v = totals[a]
            counts[a] += 1
            values[a] += (v - values[a]) / counts[a]
            if max(totals["BOX_SOFT"], totals["BOX_MEDIUM"], totals["BOX_HARD"]) > totals["STAY_OUT"]:
                box_wins += 1
        avg = {a: values[a] / max(counts[a], 1) for a in ACTIONS}
        best_box = max(("BOX_SOFT", "BOX_MEDIUM", "BOX_HARD"), key=lambda a: avg[a])
        undercut_p = float(np.clip(box_wins / max(n, 1), 0.0, 1.0))
        delta = float(np.clip(avg[best_box] if avg[best_box] >= avg["STAY_OUT"] else avg["STAY_OUT"], -120.0, 120.0))
        if window and grip < 0.40:
            action = "BOX_NOW"
            rec_comp = BOX_TO_COMPOUND[best_box]
            rec_lap = lap
            delta = float(np.clip(avg[best_box], -120.0, 120.0))
        elif grip > 0.90:
            action = "STAY_OUT"
            rec_comp = comp
            rec_lap = min(lap + 2, 100)
            delta = float(np.clip(avg["STAY_OUT"], -120.0, 120.0))
        else:
            if avg[best_box] > avg["STAY_OUT"] + 0.3:
                action = "BOX_NOW"
                rec_comp = BOX_TO_COMPOUND[best_box]
                rec_lap = lap
            else:
                action = "STAY_OUT"
                rec_comp = comp
                rec_lap = min(lap + 2, 100)
        return StrategyDecision(
            recommended_pit_lap=int(rec_lap),
            recommended_compound=rec_comp,
            action=action,
            traffic_risk_score=float(np.clip(risk, 0.0, 1.0)),
            undercut_success_probability=undercut_p,
            expected_race_time_delta_s=delta,
            pit_window_open=bool(window),
        )
