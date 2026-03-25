"""Navigation controller — orchestrates the full navigation loop.

Reads sensors (RGB + LiDAR), sends observations to the Server for
NWM + MCTS inference, receives actions, executes them in the simulator,
and tracks progress towards the goal.

Enhanced features:
- Stuck detection with recovery (back up + turn)
- Collision recovery (reverse + re-plan)
- Full trajectory recording [(x, y, yaw, t), ...]
- Multi-waypoint navigation
- Navigation statistics (steps, time, path length, collisions)
- EMA action smoothing to reduce jitter
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from legs_common.protocol.canon import Action
from legs_dog.control.low_level import ActionBuffer
from legs_dog.control.safety import SafetyGuard

logger = logging.getLogger(__name__)


@dataclass
class NavigationGoal:
    """A 2D navigation target."""
    x: float
    y: float
    yaw: Optional[float] = None
    tolerance: float = 0.5


@dataclass
class NavigationStats:
    """Cumulative statistics for a navigation episode."""
    total_steps: int = 0
    total_time_s: float = 0.0
    path_length: float = 0.0
    collision_count: int = 0
    stuck_recovery_count: int = 0
    final_distance: float = float("inf")
    reached: bool = False
    trajectory: List[Tuple[float, float, float, float]] = field(default_factory=list)

    def summary(self) -> str:
        status = "REACHED" if self.reached else "FAILED"
        return (
            f"[{status}] steps={self.total_steps} time={self.total_time_s:.1f}s "
            f"path={self.path_length:.2f}m collisions={self.collision_count} "
            f"stuck_recoveries={self.stuck_recovery_count} "
            f"final_dist={self.final_distance:.3f}m"
        )


class _ActionSmoother:
    """Exponential moving average smoother for nav_delta actions."""

    def __init__(self, alpha: float = 0.4) -> None:
        self._alpha = alpha
        self._prev_dx: float = 0.0
        self._prev_dy: float = 0.0
        self._prev_dyaw: float = 0.0

    def smooth(self, action: Action) -> Action:
        payload = action.payload or {}
        nav = payload.get("nav_delta")
        if nav is None:
            return action

        dx = float(nav.get("x", 0.0))
        dy = float(nav.get("y", 0.0))
        dyaw = float(nav.get("yaw", 0.0))

        a = self._alpha
        sdx = a * dx + (1 - a) * self._prev_dx
        sdy = a * dy + (1 - a) * self._prev_dy
        sdyaw = a * dyaw + (1 - a) * self._prev_dyaw

        self._prev_dx = sdx
        self._prev_dy = sdy
        self._prev_dyaw = sdyaw

        smoothed_payload = dict(payload)
        smoothed_payload["nav_delta"] = {"x": sdx, "y": sdy, "yaw": sdyaw}
        return Action(
            seq_ref=action.seq_ref,
            action_type=action.action_type,
            payload=smoothed_payload,
            model_id=action.model_id,
            t_infer_ns=action.t_infer_ns,
        )

    def reset(self) -> None:
        self._prev_dx = 0.0
        self._prev_dy = 0.0
        self._prev_dyaw = 0.0


class Navigator:
    """Top-level navigation controller with enhanced features.

    Runs the sense -> plan -> act loop:
    1. Build a rich Observation (RGB context + LiDAR + pose + goal).
    2. Send to Server via gRPC for NWM + MCTS inference.
    3. Receive Action into ActionBuffer.
    4. Execute via PyBulletSim actuator.
    5. Repeat until goal is reached or timeout.

    Can also run in *offline* mode where a local policy is called
    directly without gRPC.
    """

    def __init__(
        self,
        sim: object,
        action_buffer: ActionBuffer,
        safety: SafetyGuard,
        grpc_client: object = None,
        local_policy: object = None,
        obs_rate_hz: float = 10.0,
        max_steps: int = 2000,
        stuck_threshold: float = 0.02,
        stuck_window: int = 30,
        ema_alpha: float = 0.4,
    ) -> None:
        self._sim = sim
        self._action_buffer = action_buffer
        self._safety = safety
        self._grpc_client = grpc_client
        self._local_policy = local_policy
        self._obs_interval = 1.0 / obs_rate_hz
        self._max_steps = max_steps

        # Stuck detection
        self._stuck_threshold = stuck_threshold
        self._stuck_window = stuck_window

        # Action smoothing
        self._smoother = _ActionSmoother(alpha=ema_alpha)

        # Visualizer (optional, set externally)
        self.visualizer: Any = None

    def navigate_to(self, goal: NavigationGoal) -> bool:
        """Navigate to a single goal. Returns True if reached."""
        stats = self._run_to_goal(goal)
        return stats.reached

    def navigate_waypoints(self, goals: List[NavigationGoal]) -> NavigationStats:
        """Navigate through a list of waypoints sequentially.

        Returns combined stats. Stops at the first waypoint that fails.
        """
        combined = NavigationStats()
        for i, goal in enumerate(goals):
            logger.info("Waypoint %d/%d: (%.2f, %.2f)", i + 1, len(goals), goal.x, goal.y)
            stats = self._run_to_goal(goal)
            combined.total_steps += stats.total_steps
            combined.total_time_s += stats.total_time_s
            combined.path_length += stats.path_length
            combined.collision_count += stats.collision_count
            combined.stuck_recovery_count += stats.stuck_recovery_count
            combined.trajectory.extend(stats.trajectory)
            combined.final_distance = stats.final_distance
            combined.reached = stats.reached

            if not stats.reached:
                logger.warning("Failed at waypoint %d/%d", i + 1, len(goals))
                break

        logger.info("Waypoint navigation complete: %s", combined.summary())
        return combined

    def _run_to_goal(self, goal: NavigationGoal) -> NavigationStats:
        """Core navigation loop to a single goal with all enhancements."""
        self._sim.set_goal(goal.x, goal.y)
        self._smoother.reset()

        stats = NavigationStats()
        logger.info(
            "Navigation started: target=(%.2f, %.2f), tolerance=%.2f",
            goal.x, goal.y, goal.tolerance,
        )

        step = 0
        last_log_step = 0
        start_time = time.monotonic()
        prev_pos: Optional[Tuple[float, float]] = None
        position_history: List[Tuple[float, float]] = []
        recovery_cooldown = 0

        while step < self._max_steps:
            if self._safety.is_estopped:
                logger.warning("Navigation aborted: E-Stop active")
                break

            # --- Robot pose ---
            pos, _, yaw = self._sim.env.get_robot_pose()
            rx, ry = float(pos[0]), float(pos[1])
            elapsed = time.monotonic() - start_time

            # Record trajectory
            stats.trajectory.append((rx, ry, float(yaw), elapsed))

            # Accumulate path length
            if prev_pos is not None:
                dx = rx - prev_pos[0]
                dy = ry - prev_pos[1]
                stats.path_length += math.sqrt(dx * dx + dy * dy)
            prev_pos = (rx, ry)

            # --- Check goal ---
            dist = self._sim.distance_to_goal()
            if dist < goal.tolerance:
                stats.reached = True
                stats.final_distance = dist
                stats.total_steps = step
                stats.total_time_s = elapsed
                logger.info(
                    "Goal reached in %d steps (%.1fs)! dist=%.3f",
                    step, elapsed, dist,
                )
                return stats

            # --- Collision detection + recovery ---
            if hasattr(self._sim, "env") and self._sim.env.check_collision():
                stats.collision_count += 1
                if recovery_cooldown <= 0:
                    logger.info("Collision at step %d — reversing", step)
                    self._execute_recovery_backup()
                    recovery_cooldown = 15
                    stats.stuck_recovery_count += 1

            # --- Stuck detection ---
            position_history.append((rx, ry))
            if len(position_history) > self._stuck_window:
                position_history.pop(0)
            if recovery_cooldown <= 0 and self._is_stuck(position_history):
                logger.info("Stuck detected at step %d — recovering", step)
                self._execute_recovery_turn()
                position_history.clear()
                recovery_cooldown = 20
                stats.stuck_recovery_count += 1

            if recovery_cooldown > 0:
                recovery_cooldown -= 1

            # --- Build observation ---
            obs = self._sim.build_observation()

            # --- Inference ---
            if self._grpc_client is not None and self._grpc_client.is_connected:
                self._grpc_client.send_observation(obs)
                time.sleep(self._obs_interval)
            elif self._local_policy is not None:
                action = self._local_policy.infer(obs)
                self._action_buffer.put(action)
                self._safety.update_heartbeat()

            # --- Execute action (with EMA smoothing) ---
            action = self._action_buffer.get()
            if action is not None:
                action = self._smoother.smooth(action)
                self._sim.actuator.apply(action)
            else:
                self._sim.env.step_simulation(num_steps=4)

            # --- Visualization ---
            if self.visualizer is not None:
                speed = stats.path_length / max(elapsed, 0.01)
                self.visualizer.draw_trajectory(pos.tolist())
                self.visualizer.draw_goal_line(pos.tolist(), (goal.x, goal.y))
                self.visualizer.draw_hud(
                    pos.tolist(), dist, speed, step,
                    extra=f"Collisions: {stats.collision_count}",
                )

            # --- Periodic logging ---
            step += 1
            if step - last_log_step >= 50:
                logger.info(
                    "Nav step %d: pos=(%.2f, %.2f) yaw=%.1f° dist=%.2f "
                    "path=%.2fm collisions=%d elapsed=%.1fs",
                    step, rx, ry, math.degrees(yaw), dist,
                    stats.path_length, stats.collision_count, elapsed,
                )
                last_log_step = step

            if not self._grpc_client or not self._grpc_client.is_connected:
                time.sleep(self._obs_interval * 0.5)

        stats.final_distance = self._sim.distance_to_goal()
        stats.total_steps = step
        stats.total_time_s = time.monotonic() - start_time
        logger.warning("Navigation timeout: %s", stats.summary())
        return stats

    # ------------------------------------------------------------------
    # Stuck detection
    # ------------------------------------------------------------------

    def _is_stuck(self, history: List[Tuple[float, float]]) -> bool:
        """Returns True if the robot hasn't moved enough over the window."""
        if len(history) < self._stuck_window:
            return False
        x0, y0 = history[0]
        x1, y1 = history[-1]
        displacement = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        return displacement < self._stuck_threshold

    # ------------------------------------------------------------------
    # Recovery manoeuvres
    # ------------------------------------------------------------------

    def _execute_recovery_backup(self) -> None:
        """Back up for a few steps."""
        for _ in range(10):
            backup = Action(
                seq_ref=0,
                action_type="recovery",
                payload={"nav_delta": {"x": -0.3, "y": 0.0, "yaw": 0.0}},
            )
            self._sim.actuator.apply(backup)

    def _execute_recovery_turn(self) -> None:
        """Back up then turn to escape stuck position."""
        # Back up
        for _ in range(8):
            backup = Action(
                seq_ref=0,
                action_type="recovery",
                payload={"nav_delta": {"x": -0.2, "y": 0.0, "yaw": 0.0}},
            )
            self._sim.actuator.apply(backup)
        # Turn
        for _ in range(12):
            turn = Action(
                seq_ref=0,
                action_type="recovery",
                payload={"nav_delta": {"x": 0.0, "y": 0.0, "yaw": 0.5}},
            )
            self._sim.actuator.apply(turn)
