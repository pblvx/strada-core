"""Servidor realtime STRADA-CORE: replay 50 Hz + ONNX + MCTS (sin torch)."""
from __future__ import annotations
import asyncio
import functools
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
import polars as pl
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from src.core.strategy_schema import StrategyDecision
from src.core.telemetry_schema import TelemetryPacket
from src.game_theory_strategy.mcts_strategist import MCTSRaceStrategist
from src.neural_surrogate.onnx_inference import ThermalSurrogateEngine
from src.quant_risk.sde_grip import TyreDegradationSDE
log = logging.getLogger("strada-server")
ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "data" / "raw" / "silverstone_2024.parquet"
RATE_HZ = 50.0
BOOTSTRAP = os.environ.get("STRADA_BOOTSTRAP", "127.0.0.1:9092")
TOPIC = "f1.telemetry.raw"
class _State:
    engine = None
    strategist = None
    rows = []
    latest = None
    latest_grip = 0.90
    sockets = set()
    task = None
    ready = False
    circuit = None
state = _State()
def _lap_stats(df):
    """Estadisticas por vuelta (vectorizado): n, velocidad media y hueco inicio-fin.
    El hueco se mide en unidades normalizadas [0,1000] entre el primer y ultimo
    punto ordenados por timestamp (detecta parrilla y pit-lane)."""
    g = (
        df.sort("timestamp_ms")
        .group_by("lap_number")
        .agg([
            pl.len().alias("n"),
            pl.col("speed_kmh").mean().alias("mean_sp"),
            pl.col("x").first().alias("x0"),
            pl.col("y").first().alias("y0"),
            pl.col("x").last().alias("x1"),
            pl.col("y").last().alias("y1"),
        ])
    )
    return g.with_columns(
        ((pl.col("x1") - pl.col("x0")) ** 2 + (pl.col("y1") - pl.col("y0")) ** 2)
        .sqrt()
        .alias("gap")
    )

def _clean_race_laps(df):
    """Vueltas lanzadas limpias de carrera: excluye parrilla (vuelta 1), pit-lane y
    safety car por baja velocidad media. Devuelve el DataFrame de estadisticas
    filtrado (umbral 99% de la velocidad media maxima)."""
    stats = _lap_stats(df)
    max_sp = stats["mean_sp"].max()
    return stats.filter(pl.col("mean_sp") >= 0.99 * max_sp)

def _load_rows():
    """Reproduce el parquet COMPLETO en orden cronologico estricto por timestamp_ms,
    sin filtrar ni empezar en una vuelta arbitraria. Cada paquete emitido lleva el
    lap_number y tyre_age_laps reales de ese instante de la carrera."""
    return pl.read_parquet(str(PARQUET)).sort("timestamp_ms").to_dicts()

def _load_circuit():
    """Mapa del circuito: la VUELTA MAS RAPIDA limpia de carrera, submuestreada
    (~800 pts) y ordenada por timestamp. Su hueco inicio-fin es pequeno, asi que
    ctx.closePath() cierra la recta de meta sin trazar ninguna diagonal.
    """
    df = pl.read_parquet(str(PARQUET))
    clean = _clean_race_laps(df)
    best = int(clean.sort("mean_sp", descending=True)["lap_number"][0])
    d = df.filter(pl.col("lap_number") == best).sort("timestamp_ms")
    xs = d["x"].to_list()
    ys = d["y"].to_list()
    gap = ((xs[0] - xs[-1]) ** 2 + (ys[0] - ys[-1]) ** 2) ** 0.5
    step = max(1, len(xs) // 800)
    pts = [{"x": float(xs[i]), "y": float(ys[i])} for i in range(0, len(xs), step)]
    log.info("circuito: vuelta %d, %d puntos, hueco cierre %.2f (unid norm)", best, len(pts), gap)
    return pts
def _estimate_grip(packet):
    mu = TyreDegradationSDE.mu_base(packet.tyre_age_laps, packet.tyre_compound)
    deg = min(max(packet.tyre_age_laps, 0) * 0.004, 0.15)
    return float(max(0.05, min(1.0, mu - deg)))
def _ensure_ready():
    if state.ready and state.engine is not None and state.strategist is not None:
        return
    state.engine = ThermalSurrogateEngine()
    state.strategist = MCTSRaceStrategist()
    if not state.rows:
        state.rows = _load_rows()
    row0 = TelemetryPacket(**state.rows[0])
    tyre, brake = state.engine.predict(row0)
    row0.pinn_tyre_temp_c = tyre
    row0.pinn_brake_temp_c = brake
    state.latest = row0
    state.latest_grip = _estimate_grip(row0)
    state.ready = True
async def _predict(packet):
    tyre, brake = await asyncio.to_thread(state.engine.predict, packet)
    packet.pinn_tyre_temp_c = tyre
    packet.pinn_brake_temp_c = brake
    return packet
async def _tick_row(idx):
    row = state.rows[idx % len(state.rows)]
    packet = TelemetryPacket(**row)
    try:
        packet = await _predict(packet)
    except Exception as exc:
        log.warning("inferencia fallo: %s", exc)
    state.latest = packet
    state.latest_grip = _estimate_grip(packet)
    return packet
@asynccontextmanager
async def lifespan(app):
    await asyncio.to_thread(_ensure_ready)
    state.sockets = set()
    yield
    if state.task is not None:
        state.task.cancel()
        try:
            await state.task
        except asyncio.CancelledError:
            pass
    state.sockets.clear()
app = FastAPI(title="STRADA-CORE", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
@app.get("/health")
async def health():
    return {"status": "ok", "service": "strada-core"}
@app.get("/api/strategy/decision")
async def strategy_decision(rival_gap_s=2.5, n_simulations=200):
    await asyncio.to_thread(_ensure_ready)
    pkt = state.latest
    n = int(max(4, min(int(n_simulations), 200)))
    fn = functools.partial(state.strategist.evaluate_strategy, int(pkt.lap_number), float(state.latest_grip), float(pkt.pinn_tyre_temp_c if pkt.pinn_tyre_temp_c is not None else 95.0), str(pkt.tyre_compound), int(pkt.tyre_age_laps), float(rival_gap_s), n, 7)
    decision = await asyncio.to_thread(fn)
    return decision.model_dump()
@app.get("/api/circuit")
async def circuit():
    if state.circuit is None:
        state.circuit = await asyncio.to_thread(_load_circuit)
    return state.circuit

@app.websocket("/ws/telemetry")
async def ws_telemetry(ws: WebSocket):
    await ws.accept()
    await asyncio.to_thread(_ensure_ready)
    state.sockets.add(ws)
    idx = 0
    interval = 1.0 / RATE_HZ
    try:
        while True:
            t0 = asyncio.get_event_loop().time()
            packet = await _tick_row(idx)
            idx += 1
            try:
                await ws.send_text(packet.model_dump_json())
            except WebSocketDisconnect:
                break
            except Exception:
                break
            dt = asyncio.get_event_loop().time() - t0
            await asyncio.sleep(max(interval - dt, 0.0))
    finally:
        state.sockets.discard(ws)

# Montaje de la UI estatica (Pit-Wall HUD) al final para no sombrear /health, /api y /ws
UI_DIR = ROOT / "ui"
if UI_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(UI_DIR), html=True), name="ui")
