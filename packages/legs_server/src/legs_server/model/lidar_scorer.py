"""LiDAR geometry-guided scorer for candidate navigation trajectories.

Implements the core idea from the NWM paper: use LiDAR geometric
information to evaluate candidate actions produced by the diffusion
world model.  Each candidate action (dx, dy, dyaw) is scored by
simulating where the robot would end up, then checking the LiDAR
occupancy in that region for collision risk.  The final score fuses
the diffusion energy (visual coherence) with collision energy.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ScorerConfig:
    """Tunable parameters for the LiDAR geometry scorer."""
    collision_threshold: float = 0.5      # metres — closer than this is a collision
    safety_margin: float = 0.3            # metres — robot body radius estimate
    weight_collision: float = 2.0         # weight for collision energy term
    weight_diffusion: float = 1.0         # weight for NWM diffusion energy term
    weight_goal: float = 1.5             # weight for goal-distance term
    goal_max_reward: float = 5.0          # max reward for reaching the goal
    penalty_occupied: float = 10.0        # flat penalty for hitting an occupied cell


class LiDARGeometryScorer:
    """Scores candidate navigation actions using LiDAR geometry.

    For each candidate action (dx, dy, dyaw):
    1. Predict the robot pose after executing the action.
    2. Check LiDAR distances in the predicted direction of travel.
    3. Compute a collision energy (inverse distance weighting).
    4. Fuse with the NWM diffusion energy and goal proximity.
    """

    def __init__(self, config: Optional[ScorerConfig] = None) -> None:
        self._cfg = config or ScorerConfig()

    def score_candidates(
        self,
        candidates: List[Dict[str, Any]],
        lidar_data: Dict[str, Any],
        robot_pose: Tuple[float, float, float],
        goal_position: Optional[Tuple[float, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Score all candidates and add 'total_energy' to each.

        Parameters
        ----------
        candidates : list of dicts, each must contain:
            - "action_delta": [dx, dy, dyaw]
            - "energy": float (diffusion energy from NWM)
        lidar_data : dict from LiDAR sensor scan, with keys:
            - "distances": list[float]
            - "angles": list[float] (world-frame angles)
            - "max_range": float
        robot_pose : (x, y, yaw) current robot pose.
        goal_position : optional (gx, gy) target position.

        Returns
        -------
        The same candidates list, each augmented with:
            - "collision_energy": float
            - "goal_energy": float
            - "total_energy": float (lower is better)
        """
        rx, ry, ryaw = robot_pose
        distances = np.array(lidar_data["distances"])
        angles = np.array(lidar_data["angles"])
        max_range = lidar_data["max_range"]

        for cand in candidates:
            dx, dy, dyaw = cand["action_delta"][:3]

            # Predicted pose after executing this action
            cos_y = math.cos(ryaw)
            sin_y = math.sin(ryaw)
            pred_x = rx + dx * cos_y - dy * sin_y
            pred_y = ry + dx * sin_y + dy * cos_y
            pred_yaw = ryaw + dyaw

            # --- Collision energy ---
            collision_energy = self._compute_collision_energy(
                pred_x, pred_y, rx, ry, distances, angles, max_range,
            )

            # --- Goal energy ---
            goal_energy = 0.0
            if goal_position is not None:
                gx, gy = goal_position
                # Distance from predicted pose to goal
                dist_to_goal = math.sqrt((pred_x - gx) ** 2 + (pred_y - gy) ** 2)
                # Reward for getting closer (negative energy = good)
                current_dist = math.sqrt((rx - gx) ** 2 + (ry - gy) ** 2)
                goal_energy = dist_to_goal - current_dist  # negative if approaching

            # --- Fuse energies ---
            diffusion_energy = cand.get("energy", 0.0)
            total = (
                self._cfg.weight_collision * collision_energy
                + self._cfg.weight_diffusion * diffusion_energy
                + self._cfg.weight_goal * goal_energy
            )

            cand["collision_energy"] = collision_energy
            cand["goal_energy"] = goal_energy
            cand["total_energy"] = total

        return candidates

    def score_single_pose(
        self,
        x: float,
        y: float,
        lidar_data: Dict[str, Any],
        robot_x: float,
        robot_y: float,
    ) -> float:
        """Score a single predicted position for collision risk."""
        distances = np.array(lidar_data["distances"])
        angles = np.array(lidar_data["angles"])
        max_range = lidar_data["max_range"]
        return self._compute_collision_energy(
            x, y, robot_x, robot_y, distances, angles, max_range,
        )

    def _compute_collision_energy(
        self,
        pred_x: float,
        pred_y: float,
        origin_x: float,
        origin_y: float,
        distances: np.ndarray,
        angles: np.ndarray,
        max_range: float,
    ) -> float:
        """Compute collision energy for a predicted position.

        The energy is based on how close the predicted position is to
        LiDAR-detected obstacles.  We find LiDAR rays whose direction
        is close to the direction of travel, and check if the predicted
        position is beyond any obstacle on that ray.
        """
        # Direction from current pose to predicted pose
        travel_dx = pred_x - origin_x
        travel_dy = pred_y - origin_y
        travel_dist = math.sqrt(travel_dx ** 2 + travel_dy ** 2)

        if travel_dist < 1e-6:
            return 0.0  # Not moving = no collision risk

        travel_angle = math.atan2(travel_dy, travel_dx)

        # Find LiDAR rays within a cone around the travel direction
        cone_half_angle = math.pi / 6  # ±30 degrees
        angle_diffs = np.abs(np.arctan2(
            np.sin(angles - travel_angle),
            np.cos(angles - travel_angle),
        ))
        in_cone = angle_diffs < cone_half_angle

        if not np.any(in_cone):
            return 0.0

        cone_distances = distances[in_cone]
        min_obstacle_dist = float(np.min(cone_distances))

        # Collision energy: high if predicted position goes past an obstacle
        effective_dist = min_obstacle_dist - self._cfg.safety_margin

        if effective_dist <= 0:
            # Already inside an obstacle zone
            return self._cfg.penalty_occupied

        if travel_dist >= effective_dist:
            # Would collide
            overshoot = travel_dist - effective_dist
            return self._cfg.penalty_occupied * min(overshoot / effective_dist, 1.0)

        # Safe but penalise closeness
        clearance = effective_dist - travel_dist
        if clearance < self._cfg.collision_threshold:
            return (self._cfg.collision_threshold - clearance) / self._cfg.collision_threshold
        return 0.0
