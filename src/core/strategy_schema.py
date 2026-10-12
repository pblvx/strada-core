"""Contrato de decision estrategica (Pydantic v2, regla .clinerules #1)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrategyDecision(BaseModel):
    """Salida canonica del estratega MCTS."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    recommended_pit_lap: int = Field(ge=1, le=100)
    recommended_compound: Literal["SOFT", "MEDIUM", "HARD"]
    action: Literal["BOX_NOW", "STAY_OUT"]
    traffic_risk_score: float = Field(ge=0.0, le=1.0)
    undercut_success_probability: float = Field(ge=0.0, le=1.0)
    expected_race_time_delta_s: float = Field(ge=-120.0, le=120.0)
    pit_window_open: bool
