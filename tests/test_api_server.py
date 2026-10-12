"""Tests API D5: health, decision, websocket 50 Hz."""
from __future__ import annotations
import statistics
import time
from pathlib import Path
from fastapi.testclient import TestClient
from src.api.server import app
from src.core.strategy_schema import StrategyDecision
from src.core.telemetry_schema import TelemetryPacket
def _client():
    return TestClient(app)
def test_health_ok():
    with _client() as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok", "service": "strada-core"}
def test_strategy_decision_valid():
    with _client() as client:
        r = client.get("/api/strategy/decision", params={"rival_gap_s": 2.5, "n_simulations": 100})
        assert r.status_code == 200
        d = StrategyDecision.model_validate(r.json())
        assert d.recommended_compound in ("SOFT", "MEDIUM", "HARD")
        assert d.action in ("BOX_NOW", "STAY_OUT")
def test_websocket_stream_packets():
    with _client() as client:
        with client.websocket_connect("/ws/telemetry?ws=telemetry") as ws:
            pkts = []
            ts = []
            for _ in range(3):
                t0 = time.perf_counter()
                raw = ws.receive_text()
                ts.append(time.perf_counter() - t0)
                pkts.append(TelemetryPacket.model_validate_json(raw))
    assert len(pkts) >= 3
    for p in pkts:
        assert p.pinn_tyre_temp_c is not None
        assert p.pinn_brake_temp_c is not None
        assert -50.0 <= p.pinn_tyre_temp_c <= 200.0
    assert statistics.median(ts) * 1000.0 < 5000.0
def test_no_torch_in_server():
    src = Path("src/api/server.py").read_text(encoding="utf-8")
    assert "import torch" not in src and "from torch" not in src


def test_ui_static_served():
    with _client() as client:
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers.get("content-type", "")
        body = r.text
        # Sirve el index.html del Pit-Wall HUD (no un 404)
        assert "<canvas" in body or "strada" in body.lower()
        assert "track_canvas.js" in body or "STRADA" in body


def test_circuit_endpoint():
    with _client() as client:
        r = client.get("/api/circuit")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list) and len(data) > 100
        for pt in data:
            assert "x" in pt and "y" in pt
            assert isinstance(pt["x"], (int, float))
            assert isinstance(pt["y"], (int, float))