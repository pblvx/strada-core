"""Inferencia termica en CPU via ONNX Runtime (regla #2: sin torch aqui)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from src.core.telemetry_schema import TelemetryPacket

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "data" / "models" / "pinn_thermal.onnx"

INPUT_MEAN = np.array([180.0, 50.0, 10.0, 5.0, 9000.0], dtype=np.float32)
INPUT_STD = np.array([90.0, 40.0, 25.0, 2.5, 3500.0], dtype=np.float32)


class ThermalSurrogateEngine:
    """Carga pinn_thermal.onnx en CPU y predice (tyre_temp, brake_temp)."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Falta modelo ONNX: {self.model_path}")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(
            str(self.model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name

    def predict(self, telemetry: TelemetryPacket) -> tuple[float, float]:
        feats = np.array([[
            telemetry.speed_kmh, telemetry.throttle_pct, telemetry.brake_pct,
            float(telemetry.gear), float(telemetry.rpm),
        ]], dtype=np.float32)
        feats = (feats - INPUT_MEAN) / INPUT_STD
        out = self.sess.run(None, {self.in_name: feats})[0][0]
        tyre = float(np.clip(out[0], -50.0, 200.0))
        brake = float(np.clip(out[1], -50.0, 1500.0))
        return tyre, brake
