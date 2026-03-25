"""LEGS Server main entry point.

Usage:
    # Stub mode (CPU, for CI/development):
    python -m legs_server.main --grpc=0.0.0.0:50051 --metrics=0.0.0.0:9102

    # Real NWM mode (GPU, CDiT-XL):
    python -m legs_server.main --policy=real --grpc=0.0.0.0:50051 --checkpoint=nwm/cdit_xl_100000.pth.tar
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading

from legs_server.metrics.exporter import ServerMetrics
from legs_server.model.nwm_infer import StubNWMPolicy
from legs_server.service.grpc_server import serve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("legs_server")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LEGS Server runtime")
    parser.add_argument(
        "--grpc", default="0.0.0.0:50051",
        help="gRPC bind address (host:port)",
    )
    parser.add_argument(
        "--metrics", default="0.0.0.0:9102",
        help="Prometheus metrics bind address (host:port)",
    )
    parser.add_argument(
        "--policy", choices=["stub", "real"], default="stub",
        help="Policy backend: stub (CPU) or real (CDiT-XL GPU)",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to NWM CDiT-XL checkpoint (for --policy=real)",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Device for real NWM inference (cuda / cpu)",
    )
    parser.add_argument(
        "--gpu-ids", default=None,
        help="Comma-separated GPU IDs for multi-GPU inference (e.g. 0,1,2,3,4,5)",
    )
    parser.add_argument(
        "--candidates-per-gpu", type=int, default=1,
        help="Number of candidate trajectories sampled per GPU",
    )
    parser.add_argument(
        "--model-tag", default="stub-v0.1.0",
        help="Model tag for the stub NWM policy",
    )
    parser.add_argument(
        "--max-workers", type=int, default=4,
        help="Max gRPC thread pool workers",
    )
    parser.add_argument("--tls-cert", default=None, help="TLS certificate path")
    parser.add_argument("--tls-key", default=None, help="TLS private key path")
    parser.add_argument("--tls-ca", default=None, help="TLS CA certificate path (for mTLS)")
    return parser.parse_args()


def _build_policy(args: argparse.Namespace):  # type: ignore[no-untyped-def]
    """Instantiate the NWMPolicy backend based on CLI args."""
    if args.policy == "real":
        from legs_server.model.nwm_infer import RealNWMPolicy

        checkpoint = args.checkpoint
        if checkpoint is None:
            # Default path relative to repo root
            import os
            repo_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            )
            checkpoint = os.path.join(repo_root, "nwm", "cdit_xl_100000.pth.tar")

        if not _file_exists(checkpoint):
            logger.error("Checkpoint not found: %s", checkpoint)
            sys.exit(1)

        gpu_ids = None
        if args.gpu_ids:
            gpu_ids = [int(x) for x in args.gpu_ids.split(",")]

        return RealNWMPolicy(
            checkpoint_path=checkpoint,
            device=args.device,
            gpu_ids=gpu_ids,
            num_candidates_per_gpu=args.candidates_per_gpu,
        )
    else:
        return StubNWMPolicy(model_tag=args.model_tag)


def _file_exists(path: str) -> bool:
    import os
    return os.path.isfile(path)


def main() -> None:
    args = parse_args()

    # --- Metrics ---
    metrics_port = int(args.metrics.split(":")[-1]) if ":" in args.metrics else 9102
    metrics = ServerMetrics(port=metrics_port)
    metrics.start()

    # --- Model ---
    policy = _build_policy(args)
    metrics.set_model_id(policy.model_id())

    # --- gRPC server ---
    logger.info(
        "Starting LEGS Server [grpc=%s, metrics_port=%d, policy=%s, model=%s]",
        args.grpc, metrics_port, args.policy, policy.model_id(),
    )
    server, port = serve(
        policy=policy,
        bind_addr=args.grpc,
        max_workers=args.max_workers,
        tls_cert_path=args.tls_cert,
        tls_key_path=args.tls_key,
        tls_ca_path=args.tls_ca,
    )
    logger.info("gRPC server bound to port %d", port)

    # --- Graceful shutdown ---
    shutdown_event = threading.Event()

    def handle_signal(signum: int, frame: object) -> None:
        logger.info("Received signal %d — shutting down", signum)
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    logger.info("LEGS Server is running. Press Ctrl+C to stop.")

    try:
        shutdown_event.wait()
    finally:
        logger.info("Stopping gRPC server ...")
        server.stop(grace=5)
        logger.info("LEGS Server stopped.")


if __name__ == "__main__":
    main()
