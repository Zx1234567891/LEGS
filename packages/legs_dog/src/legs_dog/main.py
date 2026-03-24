"""LEGS Dog main entry point.

Usage:
    python -m legs_dog.main --mode=sim --server=10.150.16.29:50051
    python -m legs_dog.main --mode=sim --offline
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
from legs_dog.sim.adapters import FakeSim

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("legs_dog")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LEGS Dog runtime")
    parser.add_argument(
        "--mode", choices=["sim", "real"], default="sim",
        help="Run mode: sim (FakeSim stub) or real (hardware)",
    )
    parser.add_argument(
        "--server", default="10.150.16.29:50051",
        help="Server gRPC address (host:port)",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="Run without server connection (local-only mode)",
    )
    parser.add_argument(
        "--ctrl-hz", type=int, default=100,
        help="Control loop frequency in Hz",
    )
    parser.add_argument(
        "--metrics-port", type=int, default=9101,
        help="Prometheus metrics port",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Starting LEGS Dog [mode=%s, server=%s, offline=%s]", args.mode, args.server, args.offline)

    # --- Setup simulation / hardware ---
    if args.mode == "sim":
        sim = FakeSim(source="sim")
        actuator = sim.actuator
        logger.info("FakeSim initialized [session=%s, episode=%s]", sim.session_id, sim.episode_id)
    else:
        logger.error("Real mode not yet implemented — use --mode=sim")
        sys.exit(1)

    # --- Safety ---
    safety = SafetyGuard(SafetyConfig(heartbeat_timeout_ms=500.0))

    # --- Action buffer (network -> control loop) ---
    action_buffer = ActionBuffer()

    # --- Control loop ---
    period_ms = max(1, 1000 // args.ctrl_hz)
    ctrl = ControlLoop(
        actuator=actuator,
        action_buffer=action_buffer,
        safety=safety,
        period_ms=period_ms,
    )

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

    # --- Start control loop in background thread ---
    ctrl_thread = threading.Thread(target=ctrl.run_forever, daemon=True, name="ctrl-loop")
    ctrl_thread.start()

    # --- Main loop: read sensors, send observations ---
    logger.info("Entering main loop (Ctrl+C to stop)")
    obs_interval = 1.0 / min(args.ctrl_hz, 50)  # observation rate capped at 50Hz

    try:
        while not shutdown_event.is_set():
            obs = sim.build_observation()

            if grpc_client is not None:
                grpc_client.send_observation(obs)

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


if __name__ == "__main__":
    main()
