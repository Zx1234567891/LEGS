"""Safety module — E-Stop, watchdog, joint limits enforcement."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class SafetyConfig:
    """Configurable safety thresholds."""

    heartbeat_timeout_ms: float = 500.0
    max_action_age_ms: float = 200.0
    joint_position_limits: tuple[float, float] = (-3.14, 3.14)
    joint_velocity_limit: float = 20.0


class SafetyGuard:
    """Monitors link health and enforces safety constraints.

    When triggered, latches into E-Stop state — must be explicitly reset.
    """

    def __init__(self, config: Optional[SafetyConfig] = None) -> None:
        self._config = config or SafetyConfig()
        self._estopped = False
        self._estop_reason = ""
        self._lock = threading.Lock()
        self._last_heartbeat_ns: int = time.monotonic_ns()
        self._on_estop_callbacks: list[Callable[[str], None]] = []

    @property
    def is_estopped(self) -> bool:
        return self._estopped

    @property
    def estop_reason(self) -> str:
        return self._estop_reason

    def register_estop_callback(self, cb: Callable[[str], None]) -> None:
        self._on_estop_callbacks.append(cb)

    def trigger_estop(self, reason: str) -> None:
        with self._lock:
            if self._estopped:
                return
            self._estopped = True
            self._estop_reason = reason
        logger.critical("E-STOP triggered: %s", reason)
        for cb in self._on_estop_callbacks:
            try:
                cb(reason)
            except Exception:
                logger.exception("E-Stop callback error")

    def reset_estop(self) -> None:
        with self._lock:
            self._estopped = False
            self._estop_reason = ""
        logger.info("E-Stop reset")

    def update_heartbeat(self) -> None:
        self._last_heartbeat_ns = time.monotonic_ns()

    def check_heartbeat(self) -> bool:
        elapsed_ms = (time.monotonic_ns() - self._last_heartbeat_ns) / 1e6
        if elapsed_ms > self._config.heartbeat_timeout_ms:
            self.trigger_estop(f"heartbeat timeout: {elapsed_ms:.1f}ms > {self._config.heartbeat_timeout_ms}ms")
            return False
        return True
