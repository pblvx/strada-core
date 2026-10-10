"""Productor Kafka: replay del parquet a Redpanda a tasa configurable."""

from __future__ import annotations

import logging
import time

import polars as pl
from confluent_kafka import KafkaException, Producer

from src.core.telemetry_schema import TelemetryPacket

log = logging.getLogger("kafka_producer")
DEFAULT_TOPIC = "f1.telemetry.raw"
DEFAULT_BOOTSTRAP = "127.0.0.1:9092"


class TelemetryStreamer:
    """Lee el parquet y emite JSON validados al topic a rate_per_sec."""

    def __init__(
        self,
        parquet_path: str,
        bootstrap_servers: str = DEFAULT_BOOTSTRAP,
        topic: str = DEFAULT_TOPIC,
        rate_per_sec: float = 50.0,
    ) -> None:
        if rate_per_sec <= 0:
            raise ValueError("rate_per_sec debe ser > 0")
        self.parquet_path = parquet_path
        self.topic = topic
        self.rate_per_sec = rate_per_sec
        self.producer = Producer(
            {"bootstrap.servers": bootstrap_servers, "linger.ms": 5}
        )

    def _rows(self) -> list[dict]:
        df = pl.read_parquet(self.parquet_path)
        return df.to_dicts()

    def build_messages(self, limit: int | None = None) -> list[str]:
        """Valida cada fila con TelemetryPacket y devuelve los JSON."""
        rows = self._rows()
        if limit is not None:
            rows = rows[:limit]
        out: list[str] = []
        for row in rows:
            packet = TelemetryPacket(**row)
            out.append(packet.model_dump_json())
        return out

    def stream(self, limit: int | None = None, loop: bool = False) -> int:
        """Emite al topic con sleep con correccion de deriva."""
        interval = 1.0 / self.rate_per_sec
        sent = 0
        try:
            while True:
                start = time.perf_counter()
                for msg in self.build_messages(limit=limit):
                    try:
                        self.producer.produce(self.topic, msg.encode("utf-8"))
                    except (KafkaException, BufferError) as exc:
                        log.warning("Produce omitido (%s); poll+retry", exc)
                        self.producer.poll(0.1)
                        try:
                            self.producer.produce(self.topic, msg.encode("utf-8"))
                        except (KafkaException, BufferError) as exc2:
                            log.error("Produce fallido de nuevo; se omite msg: %s", exc2)
                            continue
                    sent += 1
                    self.producer.poll(0)
                    target = start + sent * interval
                    delay = target - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                self.producer.flush(5)
                if not loop:
                    break
                sent = 0
                start = time.perf_counter()
        except KeyboardInterrupt:
            log.info("Stream interrumpido por usuario; flush final")
        finally:
            try:
                self.producer.flush(5)
            except KafkaException as exc:
                log.error("Flush final con error (broker caido?): %s", exc)
        return sent
