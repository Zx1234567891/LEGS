"""Time utilities for monotonic and wall-clock timestamps."""

import time


def mono_ns() -> int:
    """Return monotonic clock in nanoseconds (for RTT/latency calculation)."""
    return time.monotonic_ns()


def wall_ns() -> int:
    """Return wall-clock time in nanoseconds (for logging/correlation)."""
    return time.time_ns()
