"""Simulation adapters — FakeSim provides a complete stub for development and testing."""

from __future__ import annotations

import json
import logging
import math
from typing import Any, Dict, List, Optional

from legs_common.protocol.canon import Action, Observation
from legs_common.time import mono_ns, wall_ns
from legs_common.ids import new_session_id, new_episode_id, SeqCounter
from legs_dog.sensors.base import Sensor, SensorFrame

logger = logging.getLogger(__name__)


class FakeActuator:
    """Stub actuator that logs actions and tracks internal state."""

    def __init__(self, num_joints: int = 12) -> None:
        self._positions = [0.0] * num_joints
        self._estopped = False
        self._estop_reason = ""
        self._last_action: Optional[Action] = None

    def apply(self, action: Action) -> None:
        if self._estopped:
            return
        self._last_action = action
        # Simulate applying action to joints
        if "joint_targets" in action.payload:
            targets = action.payload["joint_targets"]
            for i, t in enumerate(targets[:len(self._positions)]):
                self._positions[i] += (t - self._positions[i]) * 0.1

    def estop(self, reason: str, latch: bool = True) -> None:
        self._estopped = True
        self._estop_reason = reason
        logger.warning("FakeActuator E-STOP: %s", reason)

    def get_state(self) -> Dict[str, Any]:
        return {
            "positions": list(self._positions),
            "estopped": self._estopped,
            "estop_reason": self._estop_reason,
        }

    def reset(self) -> None:
        self._estopped = False
        self._estop_reason = ""


class FakeIMU:
    """Fake IMU sensor for simulation."""

    def __init__(self) -> None:
        self._step = 0

    @property
    def name(self) -> str:
        return "imu"

    def read(self) -> Optional[SensorFrame]:
        self._step += 1
        data = {
            "roll": 0.01 * math.sin(self._step * 0.1),
            "pitch": 0.01 * math.cos(self._step * 0.1),
            "yaw": 0.0,
            "acc_z": 9.81,
        }
        return SensorFrame(
            t_mono_ns=mono_ns(),
            frame_id="imu_link",
            payload=json.dumps(data).encode(),
            encoding="json",
        )

    def close(self) -> None:
        pass


class FakeJointSensor:
    """Fake joint state sensor that reads from FakeActuator state."""

    JOINT_NAMES: List[str] = [
        "FL_hip", "FL_thigh", "FL_calf",
        "FR_hip", "FR_thigh", "FR_calf",
        "RL_hip", "RL_thigh", "RL_calf",
        "RR_hip", "RR_thigh", "RR_calf",
    ]

    def __init__(self, actuator: FakeActuator) -> None:
        self._actuator = actuator

    @property
    def name(self) -> str:
        return "joint_state"

    def read(self) -> Optional[SensorFrame]:
        state = self._actuator.get_state()
        data = {
            "joint_names": self.JOINT_NAMES,
            "positions": state["positions"],
            "velocities": [0.0] * len(state["positions"]),
        }
        return SensorFrame(
            t_mono_ns=mono_ns(),
            frame_id="base_link",
            payload=json.dumps(data).encode(),
            encoding="json",
        )

    def close(self) -> None:
        pass


class FakeSim:
    """Complete simulation stub that provides sensors and actuator.

    Usage:
        sim = FakeSim()
        obs = sim.build_observation()
        sim.actuator.apply(action)
    """

    def __init__(self, source: str = "sim") -> None:
        self.actuator = FakeActuator()
        self._sensors: List[Sensor] = [
            FakeIMU(),
            FakeJointSensor(self.actuator),
        ]
        self._source = source
        self._session_id = new_session_id()
        self._episode_id = new_episode_id()
        self._seq = SeqCounter()

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def episode_id(self) -> str:
        return self._episode_id

    def new_episode(self) -> None:
        self._episode_id = new_episode_id()
        self._seq = SeqCounter()
        self.actuator.reset()

    def build_observation(self) -> Observation:
        """Read all sensors and assemble a canonical Observation."""
        sensor_data: Dict[str, Any] = {}
        for sensor in self._sensors:
            frame = sensor.read()
            if frame is not None:
                sensor_data[sensor.name] = {
                    "t_mono_ns": frame.t_mono_ns,
                    "frame_id": frame.frame_id,
                    "encoding": frame.encoding,
                    "data": json.loads(frame.payload) if frame.encoding == "json" else None,
                }

        return Observation(
            session_id=self._session_id,
            episode_id=self._episode_id,
            seq=self._seq.next(),
            t_wall_ns=wall_ns(),
            t_mono_ns=mono_ns(),
            source=self._source,
            robot_state=self.actuator.get_state(),
            sensors=sensor_data,
        )
