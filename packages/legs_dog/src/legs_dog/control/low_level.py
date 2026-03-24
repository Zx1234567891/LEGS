"""Low-level control loop — hard real-time aware, network-decoupled."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol

from legs_common.protocol.canon import Action, Observation
from legs_dog.control.safety import SafetyGuard

logger = logging.getLogger(__name__)


class Actuator(Protocol):
    """Protocol for actuator backends (sim or real hardware)."""

    def apply(self, action: Action) -> None:
        """Apply an action to the robot. Must not block."""
        ...

    def estop(self, reason: str, latch: bool = True) -> None:
        """Execute emergency stop immediately."""
        ...

    def get_state(self) -> Dict[str, Any]:
        """Return current actuator state (positions, velocities, etc.)."""
        ...


@dataclass
class LoopStats:
    """Statistics for one control loop iteration."""

    step_count: int = 0
    last_step_duration_ns: int = 0
    avg_step_duration_ns: float = 0.0
    stale_action_count: int = 0
    total_duration_sum_ns: int = 0


class ActionBuffer:
    """Thread-safe single-slot buffer for the latest action from network.

    The control loop reads from this without blocking.
    The network thread writes to this without blocking the control loop.
    """

    def __init__(self) -> None:
        self._action: Optional[Action] = None
        self._lock = threading.Lock()

    def put(self, action: Action) -> None:
        with self._lock:
            self._action = action

    def get(self) -> Optional[Action]:
        with self._lock:
            return self._action


class ControlLoop:
    """Non-blocking control loop that runs at a fixed frequency.

    Design rules:
    - step() NEVER waits for network
    - Reads latest action from ActionBuffer (lock-free single-slot)
    - On timeout/stale action: executes safety degradation (hold last, stand, or estop)
    """

    def __init__(
        self,
        actuator: Actuator,
        action_buffer: ActionBuffer,
        safety: SafetyGuard,
        period_ms: int = 10,
    ) -> None:
        self._actuator = actuator
        self._action_buffer = action_buffer
        self._safety = safety
        self._period_ms = period_ms
        self._running = False
        self._stats = LoopStats()
        self._last_applied_action: Optional[Action] = None
        self._last_action_time_ns: int = 0

    @property
    def stats(self) -> LoopStats:
        return self._stats

    @property
    def is_running(self) -> bool:
        return self._running

    def step(self) -> None:
        """Execute one control step. Must complete within period_ms."""
        t_start = time.monotonic_ns()

        if self._safety.is_estopped:
            self._actuator.estop("safety guard latched")
            self._update_stats(t_start)
            return

        action = self._action_buffer.get()

        if action is not None and action is not self._last_applied_action:
            self._last_applied_action = action
            self._last_action_time_ns = time.monotonic_ns()
            self._actuator.apply(action)
        elif self._last_applied_action is not None:
            # Hold last action (degradation level 1)
            age_ms = (time.monotonic_ns() - self._last_action_time_ns) / 1e6
            if age_ms > 500:
                # Action too old — trigger estop
                self._safety.trigger_estop(f"action stale: {age_ms:.0f}ms")
                self._actuator.estop("stale action timeout")
                self._stats.stale_action_count += 1
            else:
                # Re-apply last action (hold position)
                self._actuator.apply(self._last_applied_action)
        else:
            # No action ever received — hold safe pose
            pass

        self._update_stats(t_start)

    def _update_stats(self, t_start: int) -> None:
        duration = time.monotonic_ns() - t_start
        self._stats.step_count += 1
        self._stats.last_step_duration_ns = duration
        self._stats.total_duration_sum_ns += duration
        self._stats.avg_step_duration_ns = (
            self._stats.total_duration_sum_ns / self._stats.step_count
        )

    def run_forever(self) -> None:
        """Run the control loop at fixed frequency. Call from a dedicated thread."""
        self._running = True
        period_s = self._period_ms / 1000.0
        logger.info("Control loop started at %.0f Hz", 1000.0 / self._period_ms)

        try:
            while self._running:
                t0 = time.monotonic()
                self.step()
                elapsed = time.monotonic() - t0
                sleep_time = period_s - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            self._running = False
            logger.info("Control loop stopped after %d steps", self._stats.step_count)

    def stop(self) -> None:
        self._running = False
