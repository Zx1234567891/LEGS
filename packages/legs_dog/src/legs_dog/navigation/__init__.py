"""Navigation controller — orchestrates the full navigation loop.

Reads sensors (RGB + LiDAR), sends observations to the Server for
NWM + MCTS inference, receives actions, executes them in the simulator,
and tracks progress towards the goal.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

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


class Navigator:
    """Top-level navigation controller.

    Runs the sense → plan → act loop:
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
    ) -> None:
        self._sim = sim
        self._action_buffer = action_buffer
        self._safety = safety
        self._grpc_client = grpc_client
        self._local_policy = local_policy
        self._obs_interval = 1.0 / obs_rate_hz
        self._max_steps = max_steps

    def navigate_to(self, goal: NavigationGoal) -> bool:
        """Navigate to the goal. Returns True if reached, False otherwise."""
        # Set goal marker in simulator
        self._sim.set_goal(goal.x, goal.y)

        logger.info(
            "Navigation started: target=(%.2f, %.2f), tolerance=%.2f",
            goal.x, goal.y, goal.tolerance,
        )

        step = 0
        last_log_step = 0
        start_time = time.monotonic()

        while step < self._max_steps:
            if self._safety.is_estopped:
                logger.warning("Navigation aborted: E-Stop active")
                return False

            # Check goal reached
            dist = self._sim.distance_to_goal()
            if dist < goal.tolerance:
                elapsed = time.monotonic() - start_time
                logger.info(
                    "Goal reached in %d steps (%.1fs)! Final distance: %.3f",
                    step, elapsed, dist,
                )
                return True

            # 1. Build observation
            obs = self._sim.build_observation()

            # 2. Send to server or run local policy
            if self._grpc_client is not None and self._grpc_client.is_connected:
                self._grpc_client.send_observation(obs)
                # Wait a bit for the action to come back
                time.sleep(self._obs_interval)
            elif self._local_policy is not None:
                action = self._local_policy.infer(obs)
                self._action_buffer.put(action)
                self._safety.update_heartbeat()

            # 3. Get latest action and execute
            action = self._action_buffer.get()
            if action is not None:
                self._sim.actuator.apply(action)
            else:
                # No action available — step physics anyway
                self._sim.env.step_simulation(num_steps=4)

            # Periodic logging
            step += 1
            if step - last_log_step >= 50:
                pos, _, yaw = self._sim.env.get_robot_pose()
                elapsed = time.monotonic() - start_time
                logger.info(
                    "Nav step %d: pos=(%.2f, %.2f) yaw=%.2f° dist=%.2f elapsed=%.1fs",
                    step, pos[0], pos[1], math.degrees(yaw), dist, elapsed,
                )
                last_log_step = step

            if not self._grpc_client or not self._grpc_client.is_connected:
                # In local/offline mode, pace the loop
                time.sleep(self._obs_interval * 0.5)

        logger.warning("Navigation timeout after %d steps", self._max_steps)
        return False
