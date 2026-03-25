"""LiDAR sensor backed by PyBullet ray casting.

Simulates a 2D planar LiDAR scanner using ``pybullet.rayTestBatch()``.
Returns distance readings, hit points, and a 2D occupancy grid that can
be used by the LiDAR geometry scorer for collision evaluation.
"""

from __future__ import annotations

import json
import logging
import math
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


class PyBulletLiDAR:
    """2D planar LiDAR sensor that uses PyBullet ray casting.

    Parameters
    ----------
    physics_client : int
        PyBullet physics client id.
    robot_id : int
        PyBullet body id of the robot.
    num_rays : int
        Number of rays (angular resolution).
    max_range : float
        Maximum sensing range in metres.
    height_offset : float
        Height above robot base origin to cast rays from.
    min_angle : float
        Start angle in radians (default 0 = forward).
    max_angle : float
        End angle in radians (default 2π = full circle).
    """

    def __init__(
        self,
        physics_client: int,
        robot_id: int,
        num_rays: int = 360,
        max_range: float = 10.0,
        height_offset: float = 0.3,
        min_angle: float = 0.0,
        max_angle: float = 2.0 * math.pi,
    ) -> None:
        self._cid = physics_client
        self._robot_id = robot_id
        self._num_rays = num_rays
        self._max_range = max_range
        self._height_offset = height_offset
        self._min_angle = min_angle
        self._max_angle = max_angle

        # Pre-compute local ray directions (unit vectors on XY plane)
        self._angles = np.linspace(min_angle, max_angle, num_rays, endpoint=False)
        self._local_dirs = np.column_stack([
            np.cos(self._angles),
            np.sin(self._angles),
            np.zeros(num_rays),
        ])  # (N, 3)

    @property
    def name(self) -> str:
        return "lidar"

    def read(self) -> Optional[SensorFrame]:
        """Cast rays and return distance + hit data as a SensorFrame."""
        if not HAS_PYBULLET:
            return None

        scan = self.scan()

        payload = json.dumps({
            "distances": scan["distances"],
            "angles": scan["angles"],
            "hit_mask": scan["hit_mask"],
            "num_rays": scan["num_rays"],
            "max_range": scan["max_range"],
            "robot_x": scan["origin"][0],
            "robot_y": scan["origin"][1],
            "robot_yaw": scan["yaw"],
        }).encode()

        return SensorFrame(
            t_mono_ns=mono_ns(),
            frame_id="lidar_link",
            payload=payload,
            encoding="json",
        )

    def scan(self) -> Dict[str, Any]:
        """Perform a full LiDAR scan and return raw results.

        Returns
        -------
        dict with keys:
            distances : list[float]   — distance per ray (max_range if miss)
            angles    : list[float]   — world-frame angle per ray
            hit_points: list[list]    — 3D hit position per ray
            hit_mask  : list[bool]    — True if ray hit an obstacle
            origin    : list[float]   — scan origin [x, y, z]
            yaw       : float         — robot yaw when scan was taken
            num_rays  : int
            max_range : float
        """
        # Robot pose
        base_pos, base_orn = p.getBasePositionAndOrientation(
            self._robot_id, physicsClientId=self._cid,
        )
        yaw = p.getEulerFromQuaternion(base_orn)[2]

        origin = np.array([
            base_pos[0],
            base_pos[1],
            base_pos[2] + self._height_offset,
        ])

        # Rotate local ray directions into world frame
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        rot = np.array([
            [cos_y, -sin_y, 0],
            [sin_y,  cos_y, 0],
            [0,      0,     1],
        ])
        world_dirs = (rot @ self._local_dirs.T).T  # (N, 3)

        ray_from = np.tile(origin, (self._num_rays, 1))
        ray_to = ray_from + world_dirs * self._max_range

        results = p.rayTestBatch(
            ray_from.tolist(),
            ray_to.tolist(),
            physicsClientId=self._cid,
        )

        distances: List[float] = []
        hit_points: List[List[float]] = []
        hit_mask: List[bool] = []

        for result in results:
            obj_id, _link_idx, frac, hit_pos, _hit_normal = result
            dist = frac * self._max_range

            if obj_id == self._robot_id or obj_id == -1:
                # Self-hit or miss
                distances.append(self._max_range)
                hit_mask.append(False)
            else:
                distances.append(dist)
                hit_mask.append(True)

            hit_points.append(list(hit_pos))

        world_angles = (self._angles + yaw).tolist()

        return {
            "distances": distances,
            "angles": world_angles,
            "hit_points": hit_points,
            "hit_mask": hit_mask,
            "origin": origin.tolist(),
            "yaw": float(yaw),
            "num_rays": self._num_rays,
            "max_range": self._max_range,
        }

    def get_occupancy_grid(
        self,
        grid_size: float = 10.0,
        resolution: float = 0.1,
    ) -> np.ndarray:
        """Build a 2D occupancy grid from the latest scan.

        Parameters
        ----------
        grid_size : float
            Side length of the square grid in metres, centred on the robot.
        resolution : float
            Metres per cell.

        Returns
        -------
        np.ndarray of shape (N, N) with values in {0, 1}.
            0 = free, 1 = occupied.
        """
        scan = self.scan()
        n_cells = int(grid_size / resolution)
        grid = np.zeros((n_cells, n_cells), dtype=np.uint8)

        origin = np.array(scan["origin"][:2])

        for dist, angle, hit in zip(
            scan["distances"], scan["angles"], scan["hit_mask"]
        ):
            if not hit:
                continue
            hx = origin[0] + dist * math.cos(angle)
            hy = origin[1] + dist * math.sin(angle)

            # Convert to grid coords (origin at centre)
            gx = int((hx - origin[0] + grid_size / 2) / resolution)
            gy = int((hy - origin[1] + grid_size / 2) / resolution)

            if 0 <= gx < n_cells and 0 <= gy < n_cells:
                grid[gy, gx] = 1

        return grid

    def close(self) -> None:
        pass
