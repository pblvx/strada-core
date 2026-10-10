"""Extractor FastF1 -> parquet STRADA-CORE (batch, no streaming).

Silverstone 2024 R (ganador: Lewis Hamilton) -> Polars ->
normalizacion X/Y [0,1000] -> 50 Hz -> data/raw/silverstone_2024.parquet.

Nota .clinerules #3: FastF1 devuelve pandas; conversion inmediata a Polars.
"""

from __future__ import annotations

import logging
from pathlib import Path

import fastf1
import numpy as np
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("fastf1_extractor")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "cache"
PARQUET_OUT = PROJECT_ROOT / "data" / "raw" / "silverstone_2024.parquet"

TARGET_HZ = 50.0

NUMERIC_COLS = ["x", "y", "speed_kmh", "throttle_pct", "brake_pct", "rpm"]
DISCRETE_COLS = ["lap_number", "gear", "tyre_age_laps"]

SCHEMA_COLUMNS = [
    "timestamp_ms", "lap_number", "x", "y", "speed_kmh", "throttle_pct",
    "brake_pct", "gear", "rpm", "drs_active", "tyre_compound", "tyre_age_laps",
]

DRS_OPEN_CODES = {10, 12, 14}


def load_session(year: int = 2024, gp: str = "British Grand Prix", kind: str = "R"):
    """Carga la sesion de carrera con cache local en data/raw/cache/."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    session = fastf1.get_session(year, gp, kind)
    session.load(telemetry=True, weather=False)
    return session


def winner_driver(session) -> str:
    """Codigo del ganador; fallback a HAM ('44')."""
    try:
        results = session.results
        if results is not None and len(results) > 0:
            abbr = str(results.iloc[0].get("Abbreviation", "HAM")).upper()
            log.info("Ganador detectado: %s", abbr)
            return abbr
    except Exception as exc:
        log.warning("Sin session.results (%s); uso HAM", exc)
    return "HAM"


def extract_hamilton_telemetry(session, driver: str = "HAM") -> pl.DataFrame:
    """Telemetria vuelta a vuelta del piloto, devuelta en Polars."""
    laps = session.laps.pick_drivers([driver])
    if len(laps) == 0:
        laps = session.laps.pick_drivers(["44"])
    if len(laps) == 0:
        raise RuntimeError(f"Sin vueltas para el piloto {driver}")

    frames: list[pl.DataFrame] = []
    for _, lap in laps.iterlaps():
        try:
            tel = lap.get_telemetry()
        except Exception as exc:
            log.warning("Vuelta sin telemetria (%s); se omite", exc)
            continue
        if tel is None or len(tel) == 0:
            continue
        lap_no = int(lap.get("LapNumber", 0))
        compound = str(lap.get("Compound", "UNKNOWN")).upper()
        tyre_life = lap.get("TyreLife")
        tyre_age = int(tyre_life) if tyre_life is not None else 0
        n = len(tel)
        t_ms = (tel["SessionTime"].dt.total_seconds() * 1000.0).to_numpy(dtype="float64")
        df = pl.DataFrame(
            {
                "timestamp_ms": t_ms.astype(np.float64),
                "lap_number": np.full(n, lap_no, dtype=np.int64),
                "x": tel["X"].to_numpy(dtype="float64"),
                "y": tel["Y"].to_numpy(dtype="float64"),
                "speed_kmh": tel["Speed"].to_numpy(dtype="float64"),
                "throttle_pct": tel["Throttle"].to_numpy(dtype="float64"),
                "brake_pct": tel["Brake"].to_numpy(dtype="float64") * 100.0,
                "gear": tel["nGear"].to_numpy(dtype="float64"),
                "rpm": tel["RPM"].to_numpy(dtype="float64"),
                "drs_raw": tel["DRS"].to_numpy(dtype="float64"),
                "tyre_compound": np.full(n, compound, dtype=object),
                "tyre_age_laps": np.full(n, tyre_age, dtype=np.int64),
            }
        )
        frames.append(df)

    if not frames:
        raise RuntimeError("FastF1 no devolvio telemetria")
    out = pl.concat(frames, how="vertical").sort("timestamp_ms")
    out = out.with_columns(
        pl.col("drs_raw").cast(pl.Int64).is_in(list(DRS_OPEN_CODES)).alias("drs_active"),
        pl.col("gear").cast(pl.Int64),
        pl.col("lap_number").cast(pl.Int64),
        pl.col("tyre_age_laps").cast(pl.Int64),
        pl.col("rpm").cast(pl.Int64),
    ).drop("drs_raw")
    return out.select(SCHEMA_COLUMNS)


def normalize_xy(df: pl.DataFrame) -> pl.DataFrame:
    """Min-Max scaling de X/Y a [0, 1000]."""
    out = df
    for col in ("x", "y"):
        lo = float(out[col].min())
        hi = float(out[col].max())
        span = hi - lo
        if span <= 0:
            out = out.with_columns(pl.lit(500.0).alias(col))
        else:
            out = out.with_columns(
                ((pl.col(col) - lo) / span * 1000.0).clip(0.0, 1000.0).alias(col)
            )
    return out

def resample_50hz(df: pl.DataFrame, hz: float = TARGET_HZ) -> pl.DataFrame:
    """Interpola a frecuencia fija: numeros con numpy, resto con ffill."""
    t_all = df["timestamp_ms"].to_numpy().astype(np.float64)
    if len(t_all) < 2:
        raise ValueError("Se necesitan 2+ muestras para interpolar")
    order = np.argsort(t_all, kind="stable")
    t_sorted = t_all[order]
    _, uniq_pos = np.unique(t_sorted, return_index=True)
    keep = np.sort(uniq_pos)
    t = t_sorted[keep]
    raw_idx = order[keep]

    def col(name: str) -> np.ndarray:
        return np.asarray(df[name].to_numpy())[raw_idx]

    step = 1000.0 / hz
    grid = np.arange(float(t[0]), float(t[-1]), step, dtype="float64")
    if len(grid) < 2:
        raise ValueError("Grid de remuestreo vacio")

    prev_idx = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 1)
    rpm_interp = np.interp(grid, t, col("rpm").astype("float64"))

    def _clip(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
        # FastF1 puede dar throttle 100-104% por ruido de sensor; se sujeta al contrato.
        return np.clip(arr, lo, hi)

    out = pl.DataFrame(
        {
            "timestamp_ms": grid,
            "lap_number": col("lap_number")[prev_idx].astype("int64"),
            "x": _clip(np.interp(grid, t, col("x").astype("float64")), 0.0, 1000.0),
            "y": _clip(np.interp(grid, t, col("y").astype("float64")), 0.0, 1000.0),
            "speed_kmh": _clip(np.interp(grid, t, col("speed_kmh").astype("float64")), 0.0, 450.0),
            "throttle_pct": _clip(np.interp(grid, t, col("throttle_pct").astype("float64")), 0.0, 100.0),
            "brake_pct": _clip(np.interp(grid, t, col("brake_pct").astype("float64")), 0.0, 100.0),
            "gear": col("gear")[prev_idx].astype("int64"),
            "rpm": np.rint(rpm_interp).astype("int64"),
            "drs_active": col("drs_active")[prev_idx].astype("bool"),
            "tyre_compound": col("tyre_compound")[prev_idx].astype("str"),
            "tyre_age_laps": col("tyre_age_laps")[prev_idx].astype("int64"),
        }
    )
    return out.select(SCHEMA_COLUMNS)


def main() -> Path:
    """Pipeline completo; guarda el parquet y devuelve su ruta."""
    session = load_session()
    driver = winner_driver(session)
    assert driver == "HAM", f"Ganador esperado HAM, obtenido {driver}"
    raw = extract_hamilton_telemetry(session, driver)
    log.info("Muestras crudas: %d", len(raw))
    normed = normalize_xy(raw)
    resampled = resample_50hz(normed)
    log.info("Muestras a 50 Hz: %d", len(resampled))
    PARQUET_OUT.parent.mkdir(parents=True, exist_ok=True)
    resampled.write_parquet(str(PARQUET_OUT))
    log.info("Parquet guardado en %s", PARQUET_OUT)
    return PARQUET_OUT


if __name__ == "__main__":
    main()