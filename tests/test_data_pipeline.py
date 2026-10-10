"""Tests del pipeline de datos STRADA-CORE (regla #5: rendimiento + numerico)."""

from __future__ import annotations

import time
from pathlib import Path

import polars as pl
import pytest

from src.core.telemetry_schema import TelemetryPacket

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARQUET = PROJECT_ROOT / "data" / "raw" / "silverstone_2024.parquet"

REQUIRED_COLS = [
    "timestamp_ms", "lap_number", "x", "y", "speed_kmh", "throttle_pct",
    "brake_pct", "gear", "rpm", "drs_active", "tyre_compound", "tyre_age_laps",
]
OPTIONAL_COLS = ["pinn_tyre_temp_c", "pinn_brake_temp_c", "stochastic_grip_index"]


def test_parquet_exists_shape_and_quality():
    t0 = time.perf_counter()
    assert PARQUET.exists(), f"Falta {PARQUET}; ejecuta fastf1_extractor.main()"
    df = pl.read_parquet(str(PARQUET))
    load_s = time.perf_counter() - t0
    assert load_s < 5.0, f"Carga parquet lenta: {load_s:.2f}s"
    assert len(df) > 20_000, f"Filas insuficientes: {len(df)}"
    for c in REQUIRED_COLS:
        assert c in df.columns, f"Falta columna {c}"
    nulls = df.select(REQUIRED_COLS).null_count().row(0)
    assert all(n == 0 for n in nulls), f"Nulos en requeridas: {dict(zip(REQUIRED_COLS, nulls))}"
    assert df["x"].min() >= 0 and df["x"].max() <= 1000
    assert df["y"].min() >= 0 and df["y"].max() <= 1000
    dt = df["timestamp_ms"].diff().drop_nans().drop_nulls()
    median_dt = float(dt.median())
    assert abs(median_dt - 20.0) < 2.0, f"dt mediana {median_dt} != 20ms (50Hz)"
    assert int(df["lap_number"].min()) >= 1
    assert df["speed_kmh"].max() <= 450


def test_rows_serialize_to_telemetry_packet():
    df = pl.read_parquet(str(PARQUET))
    rows = df.to_dicts()
    t0 = time.perf_counter()
    for i, row in enumerate(rows):
        packet = TelemetryPacket(**row)
        assert packet.lap_number >= 1
        if i == 0:
            rt = TelemetryPacket.model_validate_json(packet.model_dump_json())
            assert rt == packet
    elapsed = time.perf_counter() - t0
    rate = len(rows) / max(elapsed, 1e-9)
    assert rate > 2_000, f"Validacion lenta: {rate:.0f}/s para {len(rows)} filas"
