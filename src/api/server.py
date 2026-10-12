"""Servidor realtime STRADA-CORE: replay 50 Hz + ONNX + MCTS (sin torch)."""
from __future__ import annotations
import asyncio
import functools
import logging
from contextlib import asynccontextmanager
from pathlib import Path
import polars as pl
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from src.core.strategy_schema import StrategyDecision
from src.core.telemetry_schema import TelemetryPacket
from src.game_theory_strategy.mcts_strategist import MCTSRaceStrategist
from src.neural_surrogate.onnx_inference import ThermalSurrogateEngine
from src.quant_risk.sde_grip import TyreDegradationSDE
log = logging.getLogger("strada-server")
ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "data" / "raw" / "silverstone_2024.parquet"
RATE_HZ = 50.0
BOOTSTRAP = "127.0.0.1:9092"
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
state = _State()
def _load_rows():
    return pl.read_parquet(str(PARQUET)).to_dicts()
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