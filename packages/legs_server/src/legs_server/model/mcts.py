"""Monte Carlo Tree Search for NWM-based navigation.

Implements test-time compute scaling from the NWM paper: instead of
greedily picking the single best candidate action, we build a search
tree that looks multiple steps ahead.  At each node the NWM diffusion
model proposes candidate actions, the LiDAR geometry scorer evaluates
collision risk, and UCB1 balances exploration vs exploitation.

Key paper insight: navigation performance scales with search budget
(more candidates per step × deeper lookahead).
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class NavigationState:
    """Snapshot of robot state at a tree node."""
    x: float
    y: float
    yaw: float
    lidar_distances: List[float] = field(default_factory=list)
    lidar_angles: List[float] = field(default_factory=list)
    lidar_max_range: float = 10.0


class MCTSNode:
    """A single node in the MCTS search tree."""

    __slots__ = (
        "state", "parent", "action", "children",
        "visits", "total_value", "depth",
    )

    def __init__(
        self,
        state: NavigationState,
        parent: Optional[MCTSNode] = None,
        action: Optional[List[float]] = None,
        depth: int = 0,
    ) -> None:
        self.state = state
        self.parent = parent
        self.action = action          # [dx, dy, dyaw] that led here
        self.children: List[MCTSNode] = []
        self.visits: int = 0
        self.total_value: float = 0.0
        self.depth: int = depth

    @property
    def value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def ucb1(self, exploration_weight: float = 1.4) -> float:
        if self.visits == 0:
            return float("inf")
        if self.parent is None or self.parent.visits == 0:
            return self.value
        return self.value + exploration_weight * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )


class NavigationMCTS:
    """MCTS planner that uses NWM as a world model for rollouts.

    Workflow per ``search()`` call:
    1. **Selection** — walk down the tree using UCB1.
    2. **Expansion** — generate candidate actions (from NWM or a
       simpler action sampler) and create child nodes.
    3. **Evaluation** — score each new node using:
       - LiDAR collision energy (from ``LiDARGeometryScorer``)
       - Goal proximity reward
    4. **Backpropagation** — propagate the evaluation value up to root.
    5. Return the first action of the most-visited root child.
    """

    def __init__(
        self,
        lidar_scorer: Any = None,
        max_depth: int = 3,
        num_candidates: int = 8,
        num_iterations: int = 50,
        exploration_weight: float = 1.4,
        action_range: float = 0.5,
    ) -> None:
        self._lidar_scorer = lidar_scorer
        self._max_depth = max_depth
        self._num_candidates = num_candidates
        self._num_iterations = num_iterations
        self._exploration_weight = exploration_weight
        self._action_range = action_range

    def search(
        self,
        current_state: NavigationState,
        goal_position: Tuple[float, float],
    ) -> List[float]:
        """Run MCTS and return the best action [dx, dy, dyaw].

        Parameters
        ----------
        current_state : NavigationState
            Current robot pose and LiDAR scan.
        goal_position : (gx, gy)
            Navigation target.

        Returns
        -------
        List[float] — best first action [dx, dy, dyaw].
        """
        root = MCTSNode(state=current_state, depth=0)

        for iteration in range(self._num_iterations):
            # 1. Selection
            node = self._select(root)

            # 2. Expansion
            if node.depth < self._max_depth and node.visits > 0:
                node = self._expand(node)

            # 3. Evaluation
            value = self._evaluate(node, goal_position)

            # 4. Backpropagation
            self._backpropagate(node, value)

        # Pick the root child with the most visits
        if not root.children:
            # Fallback: expand root once
            self._expand(root)
            for child in root.children:
                child.total_value = self._evaluate(child, goal_position)
                child.visits = 1

        if not root.children:
            # Ultimate fallback: move towards goal
            return self._fallback_action(current_state, goal_position)

        best_child = max(root.children, key=lambda c: c.visits)

        logger.debug(
            "MCTS done: %d iterations, %d root children, best visits=%d value=%.3f action=%s",
            self._num_iterations, len(root.children),
            best_child.visits, best_child.value, best_child.action,
        )

        return best_child.action if best_child.action is not None else [0.1, 0.0, 0.0]

    # ------------------------------------------------------------------
    # MCTS phases
    # ------------------------------------------------------------------

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Walk down the tree picking the child with highest UCB1."""
        while not node.is_leaf():
            node = max(node.children, key=lambda c: c.ucb1(self._exploration_weight))
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Generate candidate actions and create child nodes.

        Actions are sampled around the direction towards the goal
        with added diversity noise.
        """
        candidates = self._sample_actions(node.state)

        for action in candidates:
            child_state = self._simulate_action(node.state, action)
            child = MCTSNode(
                state=child_state,
                parent=node,
                action=action,
                depth=node.depth + 1,
            )
            node.children.append(child)

        # Return a random new child for evaluation
        if node.children:
            return random.choice(node.children)
        return node

    def _evaluate(self, node: MCTSNode, goal_position: Tuple[float, float]) -> float:
        """Compute the value of a leaf node.

        Value = goal_reward - collision_penalty
        Higher is better.
        """
        state = node.state

        # Goal proximity reward (negative distance, so closer = higher value)
        dist_to_goal = math.sqrt(
            (state.x - goal_position[0]) ** 2 +
            (state.y - goal_position[1]) ** 2,
        )
        goal_reward = -dist_to_goal

        # Reached goal bonus
        if dist_to_goal < 0.5:
            goal_reward += 10.0

        # Collision penalty from LiDAR scorer
        collision_penalty = 0.0
        if self._lidar_scorer is not None and state.lidar_distances:
            lidar_data = {
                "distances": state.lidar_distances,
                "angles": state.lidar_angles,
                "max_range": state.lidar_max_range,
            }
            collision_penalty = self._lidar_scorer.score_single_pose(
                state.x, state.y, lidar_data, state.x, state.y,
            )

        # Simple forward progress reward
        progress_reward = 0.0
        if node.parent is not None:
            parent_dist = math.sqrt(
                (node.parent.state.x - goal_position[0]) ** 2 +
                (node.parent.state.y - goal_position[1]) ** 2,
            )
            progress_reward = parent_dist - dist_to_goal  # positive if closer

        return goal_reward + progress_reward * 2.0 - collision_penalty

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """Propagate evaluation result back up to root."""
        current: Optional[MCTSNode] = node
        while current is not None:
            current.visits += 1
            current.total_value += value
            current = current.parent

    # ------------------------------------------------------------------
    # Action sampling
    # ------------------------------------------------------------------

    def _sample_actions(self, state: NavigationState) -> List[List[float]]:
        """Sample diverse candidate actions.

        Generates actions biased towards various directions with
        controllable diversity.
        """
        actions: List[List[float]] = []
        ar = self._action_range

        for _ in range(self._num_candidates):
            dx = random.uniform(0.0, ar)           # mostly forward
            dy = random.uniform(-ar * 0.5, ar * 0.5)
            dyaw = random.uniform(-0.3, 0.3)
            actions.append([dx, dy, dyaw])

        # Always include a "go straight" action
        actions[0] = [ar * 0.8, 0.0, 0.0]

        return actions

    def _simulate_action(
        self,
        state: NavigationState,
        action: List[float],
    ) -> NavigationState:
        """Predict the state after executing an action (kinematic model)."""
        dx, dy, dyaw = action
        cos_y = math.cos(state.yaw)
        sin_y = math.sin(state.yaw)

        new_x = state.x + dx * cos_y - dy * sin_y
        new_y = state.y + dx * sin_y + dy * cos_y
        new_yaw = state.yaw + dyaw

        # LiDAR data stays the same (approximation — real NWM would
        # predict future observations, but for MCTS planning this is
        # a reasonable heuristic since LiDAR geometry is mostly static)
        return NavigationState(
            x=new_x,
            y=new_y,
            yaw=new_yaw,
            lidar_distances=state.lidar_distances,
            lidar_angles=state.lidar_angles,
            lidar_max_range=state.lidar_max_range,
        )

    def _fallback_action(
        self,
        state: NavigationState,
        goal: Tuple[float, float],
    ) -> List[float]:
        """Simple action that moves directly towards the goal."""
        goal_dx = goal[0] - state.x
        goal_dy = goal[1] - state.y
        goal_dist = math.sqrt(goal_dx ** 2 + goal_dy ** 2)

        if goal_dist < 1e-6:
            return [0.0, 0.0, 0.0]

        # Convert to local frame
        cos_y = math.cos(state.yaw)
        sin_y = math.sin(state.yaw)
        local_dx = goal_dx * cos_y + goal_dy * sin_y
        local_dy = -goal_dx * sin_y + goal_dy * cos_y

        # Normalise to action range
        scale = self._action_range / max(abs(local_dx), abs(local_dy), 1e-6)
        scale = min(scale, 1.0)

        # Desired yaw change
        goal_angle = math.atan2(goal_dy, goal_dx)
        dyaw = math.atan2(math.sin(goal_angle - state.yaw),
                          math.cos(goal_angle - state.yaw))
        dyaw = max(-0.3, min(0.3, dyaw))

        return [local_dx * scale, local_dy * scale, dyaw]
