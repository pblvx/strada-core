"""Tests PINN: existencia ONNX, latencia CPU < 3 ms, fisica de frenada."""

from __future__ import annotations

import statistics
import time
from pathlib import Path

from src.core.telemetry_schema import TelemetryPacket
from src.neural_surrogate.onnx_inference import ThermalSurrogateEngine

ROOT = Path(__file__).resolve().parents[1]
ONNX = ROOT / "data" / "models" / "pinn_thermal.onnx"


def _rest_packet(**kw) -> TelemetryPacket:
    base = dict(timestamp_ms=0.0, lap_number=1, x=500.0, y=500.0,
                speed_kmh=0.0, throttle_pct=0.0, brake_pct=0.0,
                gear=1, rpm=8000, drs_active=False,
                tyre_compound="MEDIUM", tyre_age_laps=2)
    base.update(kw)
    return TelemetryPacket(**base)


def test_onnx_model_exists():
    assert ONNX.exists(), f"Falta {ONNX}; ejecuta train.py"
    assert ONNX.stat().st_size > 10_000, "ONNX sospechosamente pequeno"


def test_inference_latency_under_3ms():
    eng = ThermalSurrogateEngine(str(ONNX))
    pkt = _rest_packet(speed_kmh=200.0, throttle_pct=80.0, gear=6, rpm=11000)
    for _ in range(100):
        eng.predict(pkt)
    ts = []
    for _ in range(1000):
        t0 = time.perf_counter()
        eng.predict(pkt)
        ts.append((time.perf_counter() - t0) * 1000.0)
    mean_ms = statistics.mean(ts)
    p95 = sorted(ts)[950]
    assert mean_ms < 3.0, f"Latencia media {mean_ms:.3f} ms >= 3.0 ms (p95={p95:.3f})"


def test_hard_braking_raises_temp():
    eng = ThermalSurrogateEngine(str(ONNX))
    rest = _rest_packet()
    braking = _rest_packet(speed_kmh=300.0, brake_pct=100.0, gear=4, rpm=12000)
    tyre_rest, brake_rest = eng.predict(rest)
    tyre_brk, brake_brk = eng.predict(braking)
    assert brake_brk > brake_rest + 5.0, f"Freno no sube: {brake_rest} -> {brake_brk}"
    assert tyre_brk >= tyre_rest, f"Neumatico baja: {tyre_rest} -> {tyre_brk}"
