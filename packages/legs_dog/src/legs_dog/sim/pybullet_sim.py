"""PyBullet-backed simulation adapter — drop-in replacement for FakeSim.

Provides the same ``build_observation()`` / ``actuator`` interface that
``FakeSim`` exposes, but backed by a real PyBullet physics environment
with RGB camera, LiDAR, IMU, and joint sensors.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np

from legs_common.ids import SeqCounter, new_episode_id, new_session_id
from legs_common.protocol.canon import Action, Observation
from legs_common.time import mono_ns, wall_ns
from legs_dog.sensors.base import Sensor, SensorFrame
from legs_dog.sim.pybullet_env import PyBulletQuadrupedEnv

logger = logging.getLogger(__name__)


class PyBulletActuator:
    """Actuator that drives a PyBullet quadruped robot.

    Satisfies the ``Actuator`` Protocol defined in
    ``legs_dog.control.low_level``.
    """

    def __init__(self, env: PyBulletQuadrupedEnv) -> None:
        self._env = env
        self._estopped = False
        self._estop_reason = ""
        self._last_action: Optional[Action] = None

    def apply(self, action: Action) -> None:
        if self._estopped:
            return
        self._last_action = action

        payload = action.payload or {}

        # Prefer nav_delta if available (from NWM navigation)
        nav = payload.get("nav_delta")
        if nav is not None:
            dx = float(nav.get("x", 0.0))
            dy = float(nav.get("y", 0.0))
            dyaw = float(nav.get("yaw", 0.0))
            self._env.apply_nav_delta(dx, dy, dyaw)
        elif "joint_targets" in payload:
            targets = payload["joint_targets"]
            self._env.apply_joint_targets(targets)

        # Step physics (multiple sub-steps for stability)
        self._env.step_simulation(num_steps=4)

    def estop(self, reason: str, latch: bool = True) -> None:
        self._estopped = True
        self._estop_reason = reason
        self._env.estop(reason)
        logger.warning("PyBulletActuator E-STOP: %s", reason)

    def get_state(self) -> Dict[str, Any]:
        pos, orn, yaw = self._env.get_robot_pose()
        joints = self._env.get_joint_states()
        return {
            "positions": joints["positions"],
            "velocities": joints["velocities"],
            "estopped": self._estopped,
            "estop_reason": self._estop_reason,
            "x": float(pos[0]),
            "y": float(pos[1]),
            "z": float(pos[2]),
            "yaw": float(yaw),
        }

    def reset(self) -> None:
        self._estopped = False
        self._estop_reason = ""
        self._env.reset()


class PyBulletIMUSensor:
    """IMU sensor reading from PyBullet base link state."""

    def __init__(self, env: PyBulletQuadrupedEnv) -> None:
        self._env = env

    @property
    def name(self) -> str:
        return "imu"

    def read(self) -> Optional[SensorFrame]:
        data = self._env.get_imu_data()
        return SensorFrame(
            t_mono_ns=mono_ns(),
            frame_id="imu_link",
            payload=json.dumps(data).encode(),
            encoding="json",
        )

    def close(self) -> None:
        pass


class PyBulletJointSensor:
    """Joint state sensor reading from PyBullet."""

    def __init__(self, env: PyBulletQuadrupedEnv) -> None:
        self._env = env

    @property
    def name(self) -> str:
        return "joint_state"

    def read(self) -> Optional[SensorFrame]:
        data = self._env.get_joint_states()
        return SensorFrame(
            t_mono_ns=mono_ns(),
            frame_id="base_link",
            payload=json.dumps(data).encode(),
            encoding="json",
        )

    def close(self) -> None:
        pass


class PyBulletSim:
    """Complete PyBullet simulation providing FakeSim-compatible interface.

    Integrates the PyBullet physics environment with all sensor types
    (RGB camera, LiDAR, IMU, joint state) and builds rich Observations
    that include visual and geometric data for NWM + MCTS navigation.

    Usage::

        sim = PyBulletSim(gui=True, scene="indoor")
        sim.set_goal(5.0, 0.0)
        obs = sim.build_observation()
        sim.actuator.apply(action)
    """

    def __init__(
        self,
        gui: bool = True,
        scene: str = "indoor",
        source: str = "sim",
        context_size: int = 4,
    ) -> None:
        self._env = PyBulletQuadrupedEnv(gui=gui, scene=scene)
        self.actuator = PyBulletActuator(self._env)

        # Sensors
        from legs_dog.sensors.camera import PyBulletCamera
        from legs_dog.sensors.lidar import PyBulletLiDAR

        self._camera = PyBulletCamera(
            physics_client=self._env.physics_client,
            robot_id=self._env.robot_id,
            link_index=-1,
            width=224,
            height=224,
            context_size=context_size,
        )
        self._lidar = PyBulletLiDAR(
            physics_client=self._env.physics_client,
            robot_id=self._env.robot_id,
            num_rays=360,
            max_range=10.0,
        )
        self._imu = PyBulletIMUSensor(self._env)
        self._joint_sensor = PyBulletJointSensor(self._env)

        self._sensors: List[Any] = [
            self._camera,
            self._lidar,
            self._imu,
            self._joint_sensor,
        ]

        self._source = source
        self._session_id = new_session_id()
        self._episode_id = new_episode_id()
        self._seq = SeqCounter()

        logger.info(
            "PyBulletSim ready [scene=%s, session=%s]",
            scene, self._session_id,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def env(self) -> PyBulletQuadrupedEnv:
        return self._env

    @property
    def camera(self) -> Any:
        return self._camera

    @property
    def lidar(self) -> Any:
        return self._lidar

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def episode_id(self) -> str:
        return self._episode_id

    # ------------------------------------------------------------------
    # Goal management (delegates to env)
    # ------------------------------------------------------------------

    def set_goal(self, x: float, y: float) -> None:
        self._env.set_goal(x, y)

    def distance_to_goal(self) -> float:
        return self._env.distance_to_goal()

    def reached_goal(self, tolerance: float = 0.5) -> bool:
        return self._env.reached_goal(tolerance)

    # ------------------------------------------------------------------
    # Episode management
    # ------------------------------------------------------------------

    def new_episode(self) -> None:
        self._episode_id = new_episode_id()
        self._seq = SeqCounter()
        self.actuator.reset()

    # ------------------------------------------------------------------
    # Observation building
    # ------------------------------------------------------------------

    def build_observation(self) -> Observation:
        """Read all sensors and build a rich Observation.

        The observation includes:
        - Standard sensor frames (IMU, joint state)
        - RGB camera context frames (list of numpy arrays)
        - LiDAR scan data (distances, angles, hit info)
        - Robot pose (x, y, yaw) in robot_state
        - Goal position if set
        """
        sensor_data: Dict[str, Any] = {}

        # Standard sensors
        for sensor in self._sensors:
            frame = sensor.read()
            if frame is not None:
                if frame.encoding == "json":
                    parsed = json.loads(frame.payload)
                else:
                    parsed = None
                sensor_data[sensor.name] = {
                    "t_mono_ns": frame.t_mono_ns,
                    "frame_id": frame.frame_id,
                    "encoding": frame.encoding,
                    "data": parsed,
                }

        # Attach raw RGB frames for NWM encoding (in-process, not serialised)
        context_frames = self._camera.get_context_frames()
        if "rgb_camera" not in sensor_data:
            sensor_data["rgb_camera"] = {"data": {}}
        sensor_data["rgb_camera"]["data"]["frames"] = [
            f.tolist() for f in context_frames
        ]

        # Attach LiDAR scan detail
        lidar_scan = self._lidar.scan()
        if "lidar" not in sensor_data:
            sensor_data["lidar"] = {"data": {}}
        sensor_data["lidar"]["data"] = lidar_scan

        # Attach goal info
        goal_pos = self._env.goal_position
        if goal_pos is not None:
            sensor_data["goal"] = {
                "data": {"x": goal_pos[0], "y": goal_pos[1]},
            }

        # Robot state with pose
        pos, orn, yaw = self._env.get_robot_pose()
        robot_state = self.actuator.get_state()
        robot_state["x"] = float(pos[0])
        robot_state["y"] = float(pos[1])
        robot_state["z"] = float(pos[2])
        robot_state["yaw"] = float(yaw)

        return Observation(
            session_id=self._session_id,
            episode_id=self._episode_id,
            seq=self._seq.next(),
            t_wall_ns=wall_ns(),
            t_mono_ns=mono_ns(),
            source=self._source,
            robot_state=robot_state,
            sensors=sensor_data,
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        for s in self._sensors:
            s.close()
        self._env.close()
