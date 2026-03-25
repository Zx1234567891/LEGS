"""Depth camera sensor backed by PyBullet rendering.

Reuses the same camera pose computation as PyBulletCamera, but extracts
the depth buffer from ``p.getCameraImage()`` and converts it to metric
depth values (float32, in metres).
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any, Dict, Optional

import numpy as np

from legs_common.time import mono_ns
from legs_dog.sensors.base import SensorFrame

logger = logging.getLogger(__name__)

try:
    import pybullet as p
    HAS_PYBULLET = True
except ImportError:
    HAS_PYBULLET = False


class PyBulletDepthCamera:
    """Depth camera sensor that renders depth maps via PyBullet.

    The depth buffer returned by ``p.getCameraImage()`` stores non-linear
    OpenGL depth.  This sensor linearises it to metric depth (metres).
    """

    def __init__(
        self,
        physics_client: int,
        robot_id: int,
        link_index: int = -1,
        width: int = 224,
        height: int = 224,
        fov: float = 70.0,
        near: float = 0.1,
        far: float = 20.0,
        camera_height_offset: float = 0.3,
    ) -> None:
        self._cid = physics_client
        self._robot_id = robot_id
        self._link_index = link_index
        self._width = width
        self._height = height
        self._fov = fov
        self._near = near
        self._far = far
        self._height_offset = camera_height_offset

    @property
    def name(self) -> str:
        return "depth_camera"

    def read(self) -> Optional[SensorFrame]:
        if not HAS_PYBULLET:
            return None

        depth = self._capture_depth()
        payload = json.dumps({
            "width": self._width,
            "height": self._height,
            "near": self._near,
            "far": self._far,
            "min_depth": float(np.min(depth)),
            "max_depth": float(np.max(depth)),
            "mean_depth": float(np.mean(depth)),
        }).encode()

        return SensorFrame(
            t_mono_ns=mono_ns(),
            frame_id="depth_camera_link",
            payload=payload,
            encoding="json",
        )

    def get_depth_image(self) -> np.ndarray:
        """Return the latest depth image as (H, W) float32 in metres."""
        return self._capture_depth()

    def close(self) -> None:
        pass

    def _capture_depth(self) -> np.ndarray:
        """Render and linearise depth buffer."""
        if self._link_index == -1:
            cam_pos, cam_orn = p.getBasePositionAndOrientation(
                self._robot_id, physicsClientId=self._cid,
            )
        else:
            ls = p.getLinkState(self._robot_id, self._link_index, physicsClientId=self._cid)
            cam_pos, cam_orn = ls[0], ls[1]

        cam_pos = [cam_pos[0], cam_pos[1], cam_pos[2] + self._height_offset]

        rot = np.array(p.getMatrixFromQuaternion(cam_orn)).reshape(3, 3)
        forward = rot @ np.array([1.0, 0.0, 0.0])
        up = rot @ np.array([0.0, 0.0, 1.0])
        target = np.array(cam_pos) + forward * 1.0

        view = p.computeViewMatrix(
            cameraEyePosition=cam_pos,
            cameraTargetPosition=target.tolist(),
            cameraUpVector=up.tolist(),
            physicsClientId=self._cid,
        )
        proj = p.computeProjectionMatrixFOV(
            fov=self._fov,
            aspect=float(self._width) / self._height,
            nearVal=self._near,
            farVal=self._far,
            physicsClientId=self._cid,
        )

        _, _, _, depth_buf, _ = p.getCameraImage(
            width=self._width,
            height=self._height,
            viewMatrix=view,
            projectionMatrix=proj,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self._cid,
        )

        # Linearise OpenGL depth buffer to metric depth
        depth_buf = np.array(depth_buf, dtype=np.float32).reshape(self._height, self._width)
        metric = self._far * self._near / (self._far - (self._far - self._near) * depth_buf)
        return metric
