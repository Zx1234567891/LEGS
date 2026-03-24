"""Prometheus metrics exporter for Dog-side observability."""

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


class DogMetrics:
    """Exports Dog-side metrics to Prometheus /metrics endpoint."""

    def __init__(self, port: int = 9101) -> None:
        self._port = port
        if not HAS_PROM:
            logger.warning("prometheus_client not installed — metrics disabled")
            return

        self.obs_sent = Counter("legs_dog_obs_sent_total", "Total observations sent")
        self.action_recv = Counter("legs_dog_action_recv_total", "Total actions received")
        self.stale_actions = Counter("legs_dog_stale_actions_total", "Stale action count")
        self.ctrl_step_duration = Histogram(
            "legs_dog_ctrl_step_seconds",
            "Control loop step duration",
            buckets=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05],
        )
        self.net_rtt_ms = Gauge("legs_dog_net_rtt_ms", "Network RTT to server (ms)")
        self.estop_active = Gauge("legs_dog_estop_active", "1 if E-Stop is latched")

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
        logger.info("Metrics server started on port %d", self._port)
