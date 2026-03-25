"""Real robot adapter — stub interfaces for future hardware integration.

Defines the same API surface as PyBulletSim so that ``main.py --mode=real``
can swap in real sensors and actuators without changing the navigation loop.

Hardware backends (Unitree SDK, USB camera, RPLiDAR) are stubbed here —
fill in the actual SDK calls when hardware is available.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from legs_common.ids import SeqCounter, new_episode_id, new_session_id
from legs_common.protocol.canon import Action, Observation
from legs_common.time import mono_ns, wall_ns
from legs_dog.sensors.base import SensorFrame

logger = logging.getLogger(__name__)


class RealIMUSensor:
    """Reads IMU data from Unitree Go1 via unitree_legged_sdk.

    Stub: returns zeroed orientation until SDK is integrated.
    """

    @property
    def name(self) -> str:
        return "imu"

    def read(self) -> Optional[SensorFrame]:
        data = {
            "orientation": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            "angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
            "linear_acceleration": {"x": 0.0, "y": 0.0, "z": 9.81},
        }
        return SensorFrame(
            t_mono_ns=mono_ns(),
            frame_id="imu_link",
            payload=json.dumps(data).encode(),
            encoding="json",
        )

    def close(self) -> None:
        pass


class RealJointSensor:
    """Reads joint encoders from Unitree Go1 via unitree_legged_sdk.

    Stub: returns standing angles until SDK is integrated.
    """

    JOINT_NAMES: List[str] = [
        "FL_hip", "FL_thigh", "FL_calf",
        "FR_hip", "FR_thigh", "FR_calf",
        "RL_hip", "RL_thigh", "RL_calf",
        "RR_hip", "RR_thigh", "RR_calf",
    ]

    @property
    def name(self) -> str:
        return "joint_state"

    def read(self) -> Optional[SensorFrame]:
        data = {
            "joint_names": self.JOINT_NAMES,
            "positions": [0.0, 0.65, -1.3] * 4,
            "velocities": [0.0] * 12,
        }
        return SensorFrame(
            t_mono_ns=mono_ns(),
            frame_id="base_link",
            payload=json.dumps(data).encode(),
            encoding="json",
        )

    def close(self) -> None:
        pass


class RealCameraSensor:
    """Reads RGB frames from a USB camera (e.g. RealSense, webcam).

    Stub: returns metadata-only SensorFrame until cv2 capture is integrated.
    """

    def __init__(self, device_id: int = 0, width: int = 224, height: int = 224) -> None:
        self._device_id = device_id
        self._width = width
        self._height = height

    @property
    def name(self) -> str:
        return "rgb_camera"

    def read(self) -> Optional[SensorFrame]:
        data = {
            "width": self._width,
            "height": self._height,
            "channels": 3,
            "device": self._device_id,
            "stub": True,
        }
        return SensorFrame(
            t_mono_ns=mono_ns(),
            frame_id="camera_link",
            payload=json.dumps(data).encode(),
            encoding="json",
        )

    def close(self) -> None:
        pass


class RealLiDARSensor:
    """Reads from RPLiDAR / Livox LiDAR via serial or SDK.

    Stub: returns empty scan until hardware driver is integrated.
    """

    def __init__(self, port: str = "/dev/ttyUSB0", num_rays: int = 360) -> None:
        self._port = port
        self._num_rays = num_rays

    @property
    def name(self) -> str:
        return "lidar"

    def read(self) -> Optional[SensorFrame]:
        data = {
            "distances": [10.0] * self._num_rays,
            "angles": [i * 6.283 / self._num_rays for i in range(self._num_rays)],
            "hit_mask": [False] * self._num_rays,
            "num_rays": self._num_rays,
            "max_range": 10.0,
            "stub": True,
        }
        return SensorFrame(
            t_mono_ns=mono_ns(),
            frame_id="lidar_link",
            payload=json.dumps(data).encode(),
            encoding="json",
        )

    def close(self) -> None:
        pass


class RealActuator:
    """Sends motor commands to Unitree Go1 via unitree_legged_sdk.

    Stub: logs actions without sending to hardware.
    """

    def __init__(self) -> None:
        self._estopped = False
        self._estop_reason = ""

    def apply(self, action: Action) -> None:
        if self._estopped:
            return
        logger.debug("RealActuator.apply: %s (stub, no hardware)", action.action_type)

    def estop(self, reason: str, latch: bool = True) -> None:
        self._estopped = True
        self._estop_reason = reason
        logger.critical("REAL E-STOP: %s", reason)

    def get_state(self) -> Dict[str, Any]:
        return {
            "positions": [0.0, 0.65, -1.3] * 4,
            "velocities": [0.0] * 12,
            "estopped": self._estopped,
            "estop_reason": self._estop_reason,
            "x": 0.0, "y": 0.0, "z": 0.42, "yaw": 0.0,
        }

    def reset(self) -> None:
        self._estopped = False
        self._estop_reason = ""


class RealRobotSim:
    """Drop-in replacement for PyBulletSim that reads from real hardware.

    Provides the same build_observation() / actuator / set_goal() API
    so the Navigator works unchanged.

    Stub: all sensors return placeholder data.
    """

    def __init__(self, source: str = "real") -> None:
        self.actuator = RealActuator()
        self._imu = RealIMUSensor()
        self._joints = RealJointSensor()
        self._camera = RealCameraSensor()
        self._lidar = RealLiDARSensor()
        self._sensors = [self._camera, self._lidar, self._imu, self._joints]

        self._source = source
        self._session_id = new_session_id()
        self._episode_id = new_episode_id()
        self._seq = SeqCounter()
        self._goal: Optional[tuple[float, float]] = None

        logger.info("RealRobotSim initialized (stub mode) [session=%s]", self._session_id)

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def episode_id(self) -> str:
        return self._episode_id

    @property
    def env(self) -> "RealRobotSim":
        return self

    def set_goal(self, x: float, y: float) -> None:
        self._goal = (x, y)
        logger.info("Real robot goal set: (%.2f, %.2f)", x, y)

    def distance_to_goal(self) -> float:
        if self._goal is None:
            return float("inf")
        state = self.actuator.get_state()
        dx = state["x"] - self._goal[0]
        dy = state["y"] - self._goal[1]
        return (dx * dx + dy * dy) ** 0.5

    def reached_goal(self, tolerance: float = 0.5) -> bool:
        return self.distance_to_goal() < tolerance

    def get_robot_pose(self) -> tuple:
        """Compatibility with Navigator's env.get_robot_pose() calls."""
        import numpy as np
        state = self.actuator.get_state()
        pos = np.array([state["x"], state["y"], state["z"]])
        orn = np.array([0.0, 0.0, 0.0, 1.0])
        return pos, orn, state["yaw"]

    def check_collision(self) -> bool:
        return False

    def step_simulation(self, num_steps: int = 1) -> None:
        pass

    def new_episode(self) -> None:
        self._episode_id = new_episode_id()
        self._seq = SeqCounter()
        self.actuator.reset()

    def build_observation(self) -> Observation:
        sensor_data: Dict[str, Any] = {}
        for sensor in self._sensors:
            frame = sensor.read()
            if frame is not None:
                parsed = json.loads(frame.payload) if frame.encoding == "json" else None
                sensor_data[sensor.name] = {
                    "t_mono_ns": frame.t_mono_ns,
                    "frame_id": frame.frame_id,
                    "encoding": frame.encoding,
                    "data": parsed,
                }
        if self._goal is not None:
            sensor_data["goal"] = {"data": {"x": self._goal[0], "y": self._goal[1]}}

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

    def close(self) -> None:
        for s in self._sensors:
            s.close()
