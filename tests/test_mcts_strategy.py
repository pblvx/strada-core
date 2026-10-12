"""Tests MCTS D4: latencia <200 ms, contrato, tactica, compuestos."""

from __future__ import annotations

import time

from src.core.strategy_schema import StrategyDecision
from src.game_theory_strategy.mcts_strategist import MCTSRaceStrategist


def _st() -> MCTSRaceStrategist:
    return MCTSRaceStrategist()


def test_latency_under_200ms():
    st = _st()
    st.evaluate_strategy(20, 0.80, 95.0, "MEDIUM", 8, 2.5, n_simulations=50, seed=1)
    t0 = time.perf_counter()
    d = st.evaluate_strategy(20, 0.80, 95.0, "MEDIUM", 8, 2.5, n_simulations=500, seed=2)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    assert dt_ms < 200.0, f"MCTS lento: {dt_ms:.1f} ms >= 200 ms"
    assert isinstance(d, StrategyDecision)


def test_contract_valid():
    st = _st()
    d = st.evaluate_strategy(25, 0.70, 95.0, "MEDIUM", 10, 3.0, n_simulations=200, seed=3)
    assert isinstance(d, StrategyDecision)
    rt = StrategyDecision.model_validate(d.model_dump())
    assert rt == d
    assert 1 <= d.recommended_pit_lap <= 100
    assert 0.0 <= d.traffic_risk_score <= 1.0
    assert 0.0 <= d.undercut_success_probability <= 1.0
    assert -120.0 <= d.expected_race_time_delta_s <= 120.0
    assert isinstance(d.pit_window_open, bool)


def test_tactical_box_vs_stay():
    st = _st()
    worn = st.evaluate_strategy(25, 0.35, 95.0, "MEDIUM", 14, 2.0, n_simulations=300, seed=4)
    assert worn.pit_window_open is True
    assert worn.action == "BOX_NOW", f"grip critico deberia entrar: {worn}"
    fresh = st.evaluate_strategy(10, 0.95, 95.0, "SOFT", 2, 4.0, n_simulations=300, seed=5)
    assert fresh.action == "STAY_OUT", f"gomas nuevas deberian seguir: {fresh}"


def test_compound_valid():
    st = _st()
    for temp in (70.0, 95.0, 125.0):
        d = st.evaluate_strategy(22, 0.75, temp, "MEDIUM", 9, 2.0, n_simulations=200, seed=6)
        assert d.recommended_compound in ("SOFT", "MEDIUM", "HARD")
        assert d.action in ("BOX_NOW", "STAY_OUT")
