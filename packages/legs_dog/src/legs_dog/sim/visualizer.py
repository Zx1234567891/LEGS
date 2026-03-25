"""PyBullet debug visualizer — trajectory lines, LiDAR rays, HUD text, goal arrow.

All drawing uses ``p.addUserDebugLine()`` and ``p.addUserDebugText()`` so
it works in GUI mode only (calls are silently ignored in DIRECT mode).
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

try:
    import pybullet as p
    HAS_PYBULLET = True
except ImportError:
    HAS_PYBULLET = False


class Visualizer:
    """Manages all debug drawing in the PyBullet GUI."""

    def __init__(self, physics_client: int, robot_id: int) -> None:
        self._cid = physics_client
        self._robot_id = robot_id

        # Trajectory history
        self._prev_pos: Optional[List[float]] = None
        self._traj_ids: deque[int] = deque(maxlen=2000)

        # LiDAR ray line ids (reused each frame)
        self._lidar_ids: List[int] = []

        # HUD text ids
        self._hud_ids: List[int] = []

        # Goal line id
        self._goal_line_id: Optional[int] = None

        # FPS tracking
        self._last_time = time.monotonic()
        self._frame_count = 0
        self._fps = 0.0

    # ------------------------------------------------------------------
    # Trajectory
    # ------------------------------------------------------------------

    def draw_trajectory(self, pos: List[float]) -> None:
        """Draw a blue line segment from the previous position to the current one."""
        if not HAS_PYBULLET:
            return
        if self._prev_pos is not None:
            line_id = p.addUserDebugLine(
                self._prev_pos, pos,
                lineColorRGB=[0.2, 0.4, 1.0],
                lineWidth=2.0,
                lifeTime=0,
                physicsClientId=self._cid,
            )
            self._traj_ids.append(line_id)
        self._prev_pos = list(pos)

    # ------------------------------------------------------------------
    # LiDAR rays
    # ------------------------------------------------------------------

    def draw_lidar(self, scan: Dict[str, Any]) -> None:
        """Draw LiDAR rays: red = hit, green = free."""
        if not HAS_PYBULLET:
            return

        # Remove old rays
        for lid in self._lidar_ids:
            p.removeUserDebugItem(lid, physicsClientId=self._cid)
        self._lidar_ids.clear()

        origin = scan.get("origin", [0, 0, 0])
        distances = scan.get("distances", [])
        angles = scan.get("angles", [])
        hit_mask = scan.get("hit_mask", [])
        max_range = scan.get("max_range", 10.0)

        # Draw every 10th ray to reduce clutter
        step = max(1, len(distances) // 36)
        for i in range(0, len(distances), step):
            dist = distances[i]
            angle = angles[i]
            hit = hit_mask[i] if i < len(hit_mask) else False

            end = [
                origin[0] + dist * math.cos(angle),
                origin[1] + dist * math.sin(angle),
                origin[2],
            ]
            color = [1.0, 0.2, 0.2] if hit else [0.2, 0.8, 0.2]
            width = 1.5 if hit else 0.5

            lid = p.addUserDebugLine(
                origin, end,
                lineColorRGB=color,
                lineWidth=width,
                lifeTime=0.15,
                physicsClientId=self._cid,
            )
            self._lidar_ids.append(lid)

    # ------------------------------------------------------------------
    # Goal direction
    # ------------------------------------------------------------------

    def draw_goal_line(
        self, robot_pos: List[float], goal: Tuple[float, float]
    ) -> None:
        """Draw a yellow dashed line from robot to goal."""
        if not HAS_PYBULLET:
            return
        if self._goal_line_id is not None:
            p.removeUserDebugItem(self._goal_line_id, physicsClientId=self._cid)

        start = [robot_pos[0], robot_pos[1], robot_pos[2] + 0.3]
        end = [goal[0], goal[1], 0.3]
        self._goal_line_id = p.addUserDebugLine(
            start, end,
            lineColorRGB=[1.0, 0.9, 0.1],
            lineWidth=1.5,
            lifeTime=0.2,
            physicsClientId=self._cid,
        )

    # ------------------------------------------------------------------
    # HUD overlay
    # ------------------------------------------------------------------

    def draw_hud(
        self,
        robot_pos: List[float],
        goal_dist: float,
        speed: float,
        step: int,
        extra: str = "",
    ) -> None:
        """Draw text HUD above the robot."""
        if not HAS_PYBULLET:
            return

        # Update FPS
        self._frame_count += 1
        now = time.monotonic()
        dt = now - self._last_time
        if dt >= 1.0:
            self._fps = self._frame_count / dt
            self._frame_count = 0
            self._last_time = now

        # Remove old HUD
        for hid in self._hud_ids:
            p.removeUserDebugItem(hid, physicsClientId=self._cid)
        self._hud_ids.clear()

        text_pos = [robot_pos[0], robot_pos[1], robot_pos[2] + 0.8]
        lines = [
            f"Step: {step}  FPS: {self._fps:.0f}",
            f"Goal dist: {goal_dist:.2f}m  Speed: {speed:.2f}m/s",
        ]
        if extra:
            lines.append(extra)

        for i, line in enumerate(lines):
            hid = p.addUserDebugText(
                line,
                [text_pos[0], text_pos[1], text_pos[2] - i * 0.15],
                textColorRGB=[1.0, 1.0, 1.0],
                textSize=1.2,
                lifeTime=0.25,
                physicsClientId=self._cid,
            )
            self._hud_ids.append(hid)

    # ------------------------------------------------------------------
    # MCTS candidates
    # ------------------------------------------------------------------

    def draw_mcts_candidates(
        self,
        robot_pos: List[float],
        robot_yaw: float,
        candidates: List[Tuple[float, float, float]],
        scores: Optional[List[float]] = None,
    ) -> None:
        """Draw MCTS candidate trajectories as arrow lines."""
        if not HAS_PYBULLET or not candidates:
            return

        for i, (dx, dy, dyaw) in enumerate(candidates):
            cos_y = math.cos(robot_yaw)
            sin_y = math.sin(robot_yaw)
            wx = dx * cos_y - dy * sin_y
            wy = dx * sin_y + dy * cos_y

            start = [robot_pos[0], robot_pos[1], robot_pos[2] + 0.2]
            end = [robot_pos[0] + wx, robot_pos[1] + wy, robot_pos[2] + 0.2]

            # Best candidate (first) in green, others in dim grey
            if i == 0:
                color = [0.1, 1.0, 0.3]
                width = 2.0
            else:
                color = [0.5, 0.5, 0.5]
                width = 1.0

            p.addUserDebugLine(
                start, end,
                lineColorRGB=color,
                lineWidth=width,
                lifeTime=0.15,
                physicsClientId=self._cid,
            )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all debug drawings."""
        if not HAS_PYBULLET:
            return
        p.removeAllUserDebugItems(physicsClientId=self._cid)
        self._traj_ids.clear()
        self._lidar_ids.clear()
        self._hud_ids.clear()
        self._prev_pos = None
        self._goal_line_id = None
