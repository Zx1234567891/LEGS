"""RGB camera sensor backed by PyBullet rendering.

Captures images from a virtual camera mounted on the robot, maintains a
context buffer of the last *context_size* frames (required by NWM CDiT),
and exposes the standard ``Sensor`` protocol.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np

from legs_common.time import mono_ns
from legs_dog.sensors.base import SensorFrame

logger = logging.getLogger(__name__)

try:
    import pybullet as p
    HAS_PYBULLET = True
except ImportError:
    HAS_PYBULLET = False


class PyBulletCamera:
    """RGB camera sensor that renders images via PyBullet.

    The camera is attached to a specific link on the robot (or the base
    if ``link_index=-1``).  Each ``read()`` call captures a new frame,
    appends it to the context buffer, and returns the current frame plus
    the context history serialised as a ``SensorFrame``.

    The images are returned as *raw uint8 RGB* in the payload; the NWM
    preprocessing transform is applied on the server side.
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
        context_size: int = 4,
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
        self._context_size = context_size
        self._height_offset = camera_height_offset

        # Ring buffer for context frames
        self._context_buffer: deque[np.ndarray] = deque(maxlen=context_size)

    @property
    def name(self) -> str:
        return "rgb_camera"

    def read(self) -> Optional[SensorFrame]:
        """Capture an RGB frame and return it with context history."""
        if not HAS_PYBULLET:
            return None

        rgb = self._capture_frame()
        self._context_buffer.append(rgb)

        # Build payload: current frame shape + flattened bytes,
        # plus metadata about the context buffer size.
        payload_dict: Dict[str, Any] = {
            "width": self._width,
            "height": self._height,
            "channels": 3,
            "context_size": self._context_size,
            "num_frames": len(self._context_buffer),
        }
        # We serialise only the metadata as JSON; the actual pixel data
        # is attached as a list of base64-encoded buffers or, for in-process
        # usage, left on the instance for direct access.
        return SensorFrame(
            t_mono_ns=mono_ns(),
            frame_id="camera_link",
            payload=json.dumps(payload_dict).encode(),
            encoding="json",
        )

    def get_current_frame(self) -> np.ndarray:
        """Return the most recent captured RGB frame (H, W, 3) uint8."""
        if not self._context_buffer:
            return np.zeros((self._height, self._width, 3), dtype=np.uint8)
        return self._context_buffer[-1]

    def get_context_frames(self) -> List[np.ndarray]:
        """Return all buffered context frames (oldest first).

        Pads with zeros if fewer than ``context_size`` frames have been
        captured so far.
        """
        frames = list(self._context_buffer)
        while len(frames) < self._context_size:
            frames.insert(0, np.zeros((self._height, self._width, 3), dtype=np.uint8))
        return frames

    def close(self) -> None:
        self._context_buffer.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _capture_frame(self) -> np.ndarray:
        """Render one RGB image from the robot-mounted camera."""
        # Get camera pose from robot link
        if self._link_index == -1:
            cam_pos, cam_orn = p.getBasePositionAndOrientation(
                self._robot_id, physicsClientId=self._cid,
            )
        else:
            link_state = p.getLinkState(
                self._robot_id, self._link_index, physicsClientId=self._cid,
            )
            cam_pos, cam_orn = link_state[0], link_state[1]

        # Offset camera upwards
        cam_pos = [cam_pos[0], cam_pos[1], cam_pos[2] + self._height_offset]

        # Compute forward / up vectors from orientation
        rot_matrix = np.array(p.getMatrixFromQuaternion(cam_orn)).reshape(3, 3)
        forward = rot_matrix @ np.array([1.0, 0.0, 0.0])
        up = rot_matrix @ np.array([0.0, 0.0, 1.0])
        target = np.array(cam_pos) + forward * 1.0

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=cam_pos,
            cameraTargetPosition=target.tolist(),
            cameraUpVector=up.tolist(),
            physicsClientId=self._cid,
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=self._fov,
            aspect=float(self._width) / self._height,
            nearVal=self._near,
            farVal=self._far,
            physicsClientId=self._cid,
        )

        _, _, rgba, _, _ = p.getCameraImage(
            width=self._width,
            height=self._height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self._cid,
        )

        rgb = np.array(rgba, dtype=np.uint8).reshape(self._height, self._width, 4)[:, :, :3]
        return rgb
