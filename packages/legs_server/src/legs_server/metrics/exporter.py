"""Prometheus metrics exporter for Server-side observability."""

from __future__ import annotations

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    HAS_PROM = True
except ImportError:
    HAS_PROM = False


class ServerMetrics:
    """Exports Server-side metrics to Prometheus /metrics endpoint."""

    def __init__(self, port: int = 9102) -> None:
        self._port = port
        if not HAS_PROM:
            logger.warning("prometheus_client not installed — metrics disabled")
            return

        self.obs_received = Counter(
            "legs_server_obs_received_total", "Total observations received from Dog"
        )
        self.actions_sent = Counter(
            "legs_server_actions_sent_total", "Total actions sent to Dog"
        )
        self.infer_duration = Histogram(
            "legs_server_infer_seconds",
            "Inference latency per observation",
            buckets=[0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1],
        )
        self.active_streams = Gauge(
            "legs_server_active_streams", "Number of active StreamInfer sessions"
        )
        self.model_id_info = Gauge(
            "legs_server_model_info",
            "Current model identity (label-only gauge, always 1)",
            labelnames=["model_id"],
        )
        self.ping_count = Counter(
            "legs_server_ping_total", "Total Ping (heartbeat) requests"
        )

    def start(self) -> None:
        if not HAS_PROM:
            return
        thread = threading.Thread(
            target=start_http_server,
            args=(self._port,),
            daemon=True,
            name="metrics-server",
        )
        thread.start()
        logger.info("Prometheus metrics server started on port %d", self._port)

    def set_model_id(self, model_id: str) -> None:
        if HAS_PROM:
            self.model_id_info.labels(model_id=model_id).set(1)
