"""IMU sensor interface and stub implementation."""

from __future__ import annotations

import json
import math
import random
from typing import Optional

from legs_common.time import mono_ns
from legs_dog.sensors.base import SensorFrame


class IMUSensor:
    """Stub IMU sensor — produces synthetic orientation + acceleration data."""

    def __init__(self, frame_id: str = "imu_link") -> None:
        self._frame_id = frame_id
        self._step = 0

    @property
    def name(self) -> str:
        return "imu"

    def read(self) -> Optional[SensorFrame]:
        self._step += 1
        data = {
            "orientation": {
                "roll": 0.01 * math.sin(self._step * 0.1),
                "pitch": 0.01 * math.cos(self._step * 0.1),
                "yaw": 0.0,
            },
            "angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
            "linear_acceleration": {
                "x": random.gauss(0, 0.02),
                "y": random.gauss(0, 0.02),
                "z": 9.81 + random.gauss(0, 0.05),
            },
        }
        return SensorFrame(
            t_mono_ns=mono_ns(),
            frame_id=self._frame_id,
            payload=json.dumps(data).encode(),
            encoding="json",
        )

    def close(self) -> None:
        pass
