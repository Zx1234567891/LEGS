"""gRPC client for Dog -> Server communication.

Runs in a background thread. Sends Observations, receives Actions into ActionBuffer.
Never blocks the control loop.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Iterator, Optional

from legs_common.protocol.canon import Action, Observation
from legs_common.serialization.codec import pack, unpack

logger = logging.getLogger(__name__)

# gRPC imports are optional at module level to allow running without grpc installed
try:
    import grpc
    HAS_GRPC = True
except ImportError:
    HAS_GRPC = False


class GrpcInferClient:
    """Bidirectional streaming gRPC client for StreamInfer.

    Thread-safe: call send_observation() from any thread.
    Actions are delivered to the provided callback.
    """

    def __init__(
        self,
        server_addr: str = "localhost:50051",
        tls_cert_path: Optional[str] = None,
        tls_key_path: Optional[str] = None,
        tls_ca_path: Optional[str] = None,
    ) -> None:
        self._server_addr = server_addr
        self._tls_cert_path = tls_cert_path
        self._tls_key_path = tls_key_path
        self._tls_ca_path = tls_ca_path
        self._channel: Optional[object] = None
        self._running = False
        self._connected = False
        self._obs_queue: list[Observation] = []
        self._obs_lock = threading.Lock()
        self._on_action_callback: Optional[object] = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    def send_observation(self, obs: Observation) -> None:
        """Queue an observation for sending. Non-blocking."""
        with self._obs_lock:
            self._obs_queue.append(obs)
            # Keep only latest 16 to prevent memory growth if server is slow
            if len(self._obs_queue) > 16:
                self._obs_queue = self._obs_queue[-8:]

    def _drain_observations(self) -> list[Observation]:
        with self._obs_lock:
            batch = self._obs_queue[:]
            self._obs_queue.clear()
            return batch

    def connect(self, on_action: object) -> None:
        """Start the gRPC streaming connection in a background thread.

        Args:
            on_action: Callable[[Action], None] — called when server sends an Action.
        """
        if not HAS_GRPC:
            logger.warning("grpc not installed — running in offline/stub mode")
            return
        self._on_action_callback = on_action
        self._running = True
        thread = threading.Thread(target=self._stream_loop, daemon=True, name="grpc-client")
        thread.start()

    def _stream_loop(self) -> None:
        """Main streaming loop — reconnects on failure."""
        while self._running:
            try:
                logger.info("Connecting to server at %s ...", self._server_addr)
                # Stub: in real implementation, create channel and call StreamInfer
                # For now, simulate connection lifecycle
                self._connected = True
                logger.info("Connected to %s (stub mode)", self._server_addr)

                while self._running:
                    batch = self._drain_observations()
                    for obs in batch:
                        logger.debug("Would send obs seq=%d to server", obs.seq)
                    time.sleep(0.01)  # 100Hz polling

            except Exception as exc:
                self._connected = False
                logger.warning("gRPC connection error: %s — reconnecting in 1s", exc)
                time.sleep(1.0)

    def close(self) -> None:
        self._running = False
        self._connected = False
