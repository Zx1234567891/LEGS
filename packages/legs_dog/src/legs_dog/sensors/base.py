"""Sensor base interfaces — all sensor implementations must follow this Protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass(frozen=True)
class SensorFrame:
    """A single sensor reading with timestamp and encoding metadata."""

    t_mono_ns: int
    frame_id: str
    payload: bytes
    encoding: str  # "raw" | "lz4" | "zstd" | "msgpack"


class Sensor(Protocol):
    """Protocol for all sensor sources (real hardware or sim stubs)."""

    @property
    def name(self) -> str:
        """Unique sensor identifier (e.g. 'lidar_front', 'imu_body')."""
        ...

    def read(self) -> Optional[SensorFrame]:
        """Non-blocking read. Returns None if no data available."""
        ...

    def close(self) -> None:
        """Release hardware/sim resources."""
        ...
