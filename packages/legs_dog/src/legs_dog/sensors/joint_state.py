"""Joint state sensor interface and stub implementation."""

from __future__ import annotations

import json
import math
from typing import List, Optional

from legs_common.time import mono_ns
from legs_dog.sensors.base import SensorFrame

# 12-DOF quadruped: 4 legs x 3 joints (hip, thigh, calf)
DEFAULT_JOINT_NAMES: List[str] = [
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
]


class JointStateSensor:
    """Stub joint state sensor — produces synthetic joint positions and velocities."""

    def __init__(
        self,
        joint_names: Optional[List[str]] = None,
        frame_id: str = "base_link",
    ) -> None:
        self._joint_names = joint_names or DEFAULT_JOINT_NAMES
        self._frame_id = frame_id
        self._step = 0

    @property
    def name(self) -> str:
        return "joint_state"

    def read(self) -> Optional[SensorFrame]:
        self._step += 1
        positions = [0.1 * math.sin(self._step * 0.05 + i) for i in range(len(self._joint_names))]
        velocities = [0.0] * len(self._joint_names)
        data = {
            "joint_names": self._joint_names,
            "positions": positions,
            "velocities": velocities,
        }
        return SensorFrame(
            t_mono_ns=mono_ns(),
            frame_id=self._frame_id,
            payload=json.dumps(data).encode(),
            encoding="json",
        )

    def close(self) -> None:
        pass
