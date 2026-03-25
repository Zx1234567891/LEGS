"""LEGS Dog main entry point.

Usage:
    # PyBullet navigation with MCTS (default)
    python -m legs_dog.main --mode=sim --sim-backend=pybullet --goal=5.0,0.0

    # PyBullet offline (local stub policy, no server)
    python -m legs_dog.main --mode=sim --sim-backend=pybullet --offline --goal=3.0,2.0

    # Legacy FakeSim + gRPC server
    python -m legs_dog.main --mode=sim --sim-backend=fake --server=10.150.16.29:50051
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time

from legs_common.protocol.canon import Action
from legs_dog.control.low_level import ActionBuffer, ControlLoop
from legs_dog.control.safety import SafetyConfig, SafetyGuard
from legs_dog.net.grpc_client import GrpcInferClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("legs_dog")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LEGS Dog runtime")
    parser.add_argument(
        "--mode", choices=["sim", "real"], default="sim",
        help="Run mode: sim or real (hardware)",
    )
    parser.add_argument(
        "--sim-backend", choices=["fake", "pybullet"], default="pybullet",
        help="Simulation backend: fake (numeric stub) or pybullet (physics + GUI)",
    )
    parser.add_argument(
        "--scene", default="indoor",
        help="Navigation scene: indoor/outdoor/maze or YAML name in assets/scenes/",
    )
    parser.add_argument(
        "--goal", type=str, default="4.0,0.0",
        help="Navigation goal as x,y (e.g. '4.0,2.0')",
    )
    parser.add_argument(
        "--server", default="10.150.16.29:50051",
        help="Server gRPC address (host:port)",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="Run without server connection (local stub policy)",
    )
    parser.add_argument(
        "--use-mcts", action="store_true",
        help="Enable MCTS tree search for navigation (offline mode)",
    )
    parser.add_argument(
        "--no-gui", action="store_true",
        help="Run PyBullet in headless mode (no GUI window)",
    )
    parser.add_argument(
        "--ctrl-hz", type=int, default=100,
        help="Control loop frequency in Hz",
    )
    parser.add_argument(
        "--max-steps", type=int, default=2000,
        help="Maximum navigation steps before timeout",
    )
    parser.add_argument(
        "--metrics-port", type=int, default=9101,
        help="Prometheus metrics port",
    )
    return parser.parse_args()


def _parse_goal(goal_str: str) -> tuple[float, float]:
    """Parse 'x,y' goal string."""
    parts = goal_str.split(",")
    if len(parts) != 2:
        raise ValueError(f"Goal must be 'x,y', got: {goal_str}")
    return float(parts[0].strip()), float(parts[1].strip())


def main() -> None:
    args = parse_args()
    logger.info(
        "Starting LEGS Dog [mode=%s, backend=%s, server=%s, offline=%s]",
        args.mode, args.sim_backend, args.server, args.offline,
    )

    # --- Parse goal ---
    goal_x, goal_y = _parse_goal(args.goal)
    logger.info("Navigation goal: (%.2f, %.2f)", goal_x, goal_y)

    # --- Setup simulation / hardware ---
    if args.mode == "sim" and args.sim_backend == "pybullet":
        from legs_dog.sim.pybullet_sim import PyBulletSim

        sim = PyBulletSim(
            gui=not args.no_gui,
            scene=args.scene,
            source="sim",
        )
        sim.set_goal(goal_x, goal_y)
        actuator = sim.actuator
        logger.info(
            "PyBulletSim initialized [scene=%s, session=%s, episode=%s]",
            args.scene, sim.session_id, sim.episode_id,
        )

    elif args.mode == "sim" and args.sim_backend == "fake":
        from legs_dog.sim.adapters import FakeSim

        sim = FakeSim(source="sim")
        actuator = sim.actuator
        logger.info(
            "FakeSim initialized [session=%s, episode=%s]",
            sim.session_id, sim.episode_id,
        )

    else:
        from legs_dog.sim.real_adapter import RealRobotSim

        sim = RealRobotSim(source="real")
        actuator = sim.actuator
        logger.info("RealRobotSim initialized (stub) [session=%s]", sim.session_id)

    # --- Safety ---
    safety = SafetyGuard(SafetyConfig(heartbeat_timeout_ms=2000.0))

    # --- Action buffer (network -> control loop) ---
    action_buffer = ActionBuffer()

    # --- Local policy for offline mode ---
    local_policy = None
    if args.offline:
        if args.use_mcts and args.sim_backend == "pybullet":
            # Use MCTS with LiDAR scoring in offline mode
            from legs_server.model.lidar_scorer import LiDARGeometryScorer
            from legs_server.model.mcts import NavigationMCTS

            lidar_scorer = LiDARGeometryScorer()
            mcts = NavigationMCTS(
                lidar_scorer=lidar_scorer,
                max_depth=3,
                num_candidates=8,
                num_iterations=30,
            )
            local_policy = _MCTSLocalPolicy(mcts, goal=(goal_x, goal_y))
            logger.info("MCTS local policy enabled [depth=3, candidates=8, iterations=30]")
        else:
            from legs_server.model.nwm_infer import StubNWMPolicy
            local_policy = StubNWMPolicy(model_tag="local-stub-v0.1.0")
            logger.info("Using local StubNWMPolicy (offline mode)")

    # --- gRPC client ---
    grpc_client: GrpcInferClient | None = None
    if not args.offline:
        grpc_client = GrpcInferClient(server_addr=args.server)

        def on_action(action: Action) -> None:
            action_buffer.put(action)
            safety.update_heartbeat()

        grpc_client.connect(on_action=on_action)

    # --- Graceful shutdown ---
    shutdown_event = threading.Event()

    def handle_signal(signum: int, frame: object) -> None:
        logger.info("Received signal %d — shutting down", signum)
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # --- Navigation mode (PyBullet or Real) ---
    if args.sim_backend == "pybullet" or args.mode == "real":
        from legs_dog.navigation import Navigator, NavigationGoal

        navigator = Navigator(
            sim=sim,
            action_buffer=action_buffer,
            safety=safety,
            grpc_client=grpc_client,
            local_policy=local_policy,
            obs_rate_hz=10.0,
            max_steps=args.max_steps,
        )

        # Attach visualizer in GUI mode
        if args.sim_backend == "pybullet" and not args.no_gui:
            from legs_dog.sim.visualizer import Visualizer
            navigator.visualizer = Visualizer(
                physics_client=sim.env.physics_client,
                robot_id=sim.env.robot_id,
            )

        nav_goal = NavigationGoal(x=goal_x, y=goal_y, tolerance=0.5)

        logger.info("Starting navigation to (%.2f, %.2f) ...", goal_x, goal_y)
        try:
            success = navigator.navigate_to(nav_goal)
            if success:
                logger.info("Navigation SUCCEEDED!")
            else:
                logger.warning("Navigation FAILED (timeout or E-Stop)")
        except KeyboardInterrupt:
            logger.info("Navigation interrupted by user")
        finally:
            if grpc_client is not None:
                grpc_client.close()
            sim.close()
            logger.info("LEGS Dog stopped.")
        return

    # --- Legacy mode (FakeSim + control loop) ---
    period_ms = max(1, 1000 // args.ctrl_hz)
    ctrl = ControlLoop(
        actuator=actuator,
        action_buffer=action_buffer,
        safety=safety,
        period_ms=period_ms,
    )

    ctrl_thread = threading.Thread(target=ctrl.run_forever, daemon=True, name="ctrl-loop")
    ctrl_thread.start()

    logger.info("Entering main loop (Ctrl+C to stop)")
    obs_interval = 1.0 / min(args.ctrl_hz, 50)

    try:
        while not shutdown_event.is_set():
            obs = sim.build_observation()

            if grpc_client is not None:
                grpc_client.send_observation(obs)

            if local_policy is not None:
                action = local_policy.infer(obs)
                action_buffer.put(action)
                safety.update_heartbeat()

            if obs.seq % 100 == 0:
                logger.info(
                    "obs seq=%d | ctrl_steps=%d | avg_step=%.2fms | stale=%d",
                    obs.seq,
                    ctrl.stats.step_count,
                    ctrl.stats.avg_step_duration_ns / 1e6,
                    ctrl.stats.stale_action_count,
                )

            time.sleep(obs_interval)
    finally:
        ctrl.stop()
        if grpc_client is not None:
            grpc_client.close()
        logger.info("LEGS Dog stopped.")


class _MCTSLocalPolicy:
    """Wraps NavigationMCTS as a local NWMPolicy-compatible interface."""

    def __init__(self, mcts, goal: tuple[float, float]) -> None:
        self._mcts = mcts
        self._goal = goal
        self._model_id = "mcts-local-v0.1.0"

    def infer(self, obs) -> Action:
        from legs_server.model.mcts import NavigationState

        t_start = time.monotonic_ns()

        # Extract robot pose
        rs = obs.robot_state or {}
        rx = float(rs.get("x", 0.0))
        ry = float(rs.get("y", 0.0))
        ryaw = float(rs.get("yaw", 0.0))

        # Extract LiDAR data
        lidar = obs.sensors.get("lidar", {})
        if isinstance(lidar, dict):
            lidar_data = lidar.get("data", lidar)
        else:
            lidar_data = {}

        state = NavigationState(
            x=rx, y=ry, yaw=ryaw,
            lidar_distances=lidar_data.get("distances", []),
            lidar_angles=lidar_data.get("angles", []),
            lidar_max_range=lidar_data.get("max_range", 10.0),
        )

        # Run MCTS search
        best_action = self._mcts.search(state, self._goal)

        t_infer = time.monotonic_ns() - t_start

        return Action(
            seq_ref=obs.seq,
            action_type="nwm_navigation",
            payload={
                "nav_delta": {
                    "x": best_action[0],
                    "y": best_action[1],
                    "yaw": best_action[2],
                },
            },
            model_id=self._model_id,
            t_infer_ns=t_infer,
        )

    def model_id(self) -> str:
        return self._model_id


if __name__ == "__main__":
    main()
