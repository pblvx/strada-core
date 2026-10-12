"""STRADA-CORE · Automated performance benchmark.

Reproducible, CPU-only measurement of the four latency-critical paths of the
real-time engine. Prints a Markdown table ready to paste into README.md.

Paths measured:
  a) PINN thermal inference via ONNX Runtime           (1000 iters, ms)
  b) Tyre grip Monte-Carlo SDE                          (10,000 trajectories, ms)
  c) MCTS strategy convergence                          (500 simulations, ms)
  d) Telemetry generation throughput                    (Pydantic + JSON, packets/s)

Run:  uv run python benchmark.py
"""
from __future__ import annotations

import statistics
import time

import numpy as np

from src.core.telemetry_schema import TelemetryPacket
from src.data_engine.kafka_producer import TelemetryStreamer
from src.game_theory_strategy.mcts_strategist import MCTSRaceStrategist
from src.neural_surrogate.onnx_inference import ThermalSurrogateEngine
from src.quant_risk.sde_grip import TyreDegradationSDE

REPO = __file__ and __import__("pathlib").Path(__file__).resolve().parent
PARQUET = REPO / "data" / "raw" / "silverstone_2024.parquet"

# Representative high-speed braking-zone sample (Feeds the 5 network inputs).
SAMPLE = TelemetryPacket(
    timestamp_ms=1234.0,
    lap_number=45,
    x=512.0,
    y=377.0,
    speed_kmh=300.0,
    throttle_pct=0.0,
    brake_pct=100.0,
    gear=7,
    rpm=11935,
    drs_active=False,
    tyre_compound="MEDIUM",
    tyre_age_laps=12,
)


def _pct(xs: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(xs, dtype=np.float64), q))


def bench_pinn(iters: int = 1000, warmup: int = 100) -> dict:
    """ONNX Runtime CPU inference latency (ms) over `iters` runs."""
    engine = ThermalSurrogateEngine()
    for _ in range(warmup):
        engine.predict(SAMPLE)
    lat: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        engine.predict(SAMPLE)
        lat.append((time.perf_counter() - t0) * 1000.0)
    return {
        "iters": iters,
        "mean": statistics.fmean(lat),
        "p50": _pct(lat, 50),
        "p95": _pct(lat, 95),
        "p99": _pct(lat, 99),
    }


def bench_sde(n: int = 10_000, repeats: int = 5) -> dict:
    """Tyre grip Monte-Carlo SDE (ms per simulation call)."""
    sde = TyreDegradationSDE()
    sde.simulate_monte_carlo(0.95, 95.0, "MEDIUM", 20, n, 12, seed=0)  # warmup
    ts: list[float] = []
    for r in range(repeats):
        t0 = time.perf_counter()
        sde.simulate_monte_carlo(0.95, 95.0, "MEDIUM", 20, n, 12, seed=r)
        ts.append((time.perf_counter() - t0) * 1000.0)
    return {"n": n, "mean": statistics.fmean(ts)}


def bench_mcts(sims: int = 500, repeats: int = 5) -> dict:
    """MCTS strategy convergence time (ms per evaluate_strategy call)."""
    mcts = MCTSRaceStrategist()
    mcts.evaluate_strategy(30, 0.82, 98.0, "MEDIUM", 12, 2.5, sims, 7)  # warmup
    ts: list[float] = []
    for r in range(repeats):
        t0 = time.perf_counter()
        mcts.evaluate_strategy(30, 0.82, 98.0, "MEDIUM", 12, 2.5, sims, 7)
        ts.append((time.perf_counter() - t0) * 1000.0)
    return {"sims": sims, "mean": statistics.fmean(ts)}


def bench_generator(limit: int = 20_000) -> dict:
    """Telemetry generation throughput: Pydantic validation + JSON serialization."""
    streamer = TelemetryStreamer(str(PARQUET))
    t0 = time.perf_counter()
    msgs = streamer.build_messages(limit=limit)
    dt = time.perf_counter() - t0
    return {"packets": len(msgs), "seconds": dt, "per_sec": len(msgs) / dt if dt else 0.0}


def _fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def build_markdown(pinn: dict, sde: dict, mcts: dict, gen: dict) -> str:
    lines = [
        "### Latency / Throughput (CPU, single-threaded)",
        "",
        "| Component | Metric | Value |",
        "| --- | --- | --- |",
        f"| PINN Thermal (ONNX Runtime) | mean latency ({pinn['iters']} iters) | {_fmt(pinn['mean'])} ms |",
        f"| PINN Thermal (ONNX Runtime) | p50 / p95 / p99 | {_fmt(pinn['p50'])} / {_fmt(pinn['p95'])} / {_fmt(pinn['p99'])} ms |",
        f"| Grip SDE Monte-Carlo | mean ({sde['n']:,} trajectories) | {_fmt(sde['mean'])} ms |",
        f"| MCTS Strategy | mean convergence ({mcts['sims']} sims) | {_fmt(mcts['mean'])} ms |",
        f"| Telemetry generator | throughput | {gen['per_sec']:,.0f} packets/s ({gen['packets']:,} pkts in {_fmt(gen['seconds'], 2)} s) |",
        "",
        "> Reproduce with `uv run python benchmark.py`. Sustained replay stream: 50 Hz.",
    ]
    return "\n".join(lines)


def main() -> None:
    print("STRADA-CORE benchmark · warming up…", flush=True)
    pinn = bench_pinn()
    sde = bench_sde()
    mcts = bench_mcts()
    gen = bench_generator()

    print()
    print("### Console summary")
    print(f"  PINN  mean={_fmt(pinn['mean'])} ms  p50={_fmt(pinn['p50'])}  p95={_fmt(pinn['p95'])}  p99={_fmt(pinn['p99'])}")
    print(f"  SDE   mean={_fmt(sde['mean'])} ms  ({sde['n']:,} trajectories)")
    print(f"  MCTS  mean={_fmt(mcts['mean'])} ms  ({mcts['sims']} sims)")
    print(f"  GEN   {gen['per_sec']:,.0f} packets/s  ({gen['packets']:,} pkts in {_fmt(gen['seconds'], 2)} s)")
    print()
    print(build_markdown(pinn, sde, mcts, gen))


if __name__ == "__main__":
    main()
