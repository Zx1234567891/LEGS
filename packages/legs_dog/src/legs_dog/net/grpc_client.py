"""gRPC client for Dog -> Server communication.

Runs in a background thread. Sends Observations via StreamInfer bidirectional
streaming, receives Actions into the provided callback. Never blocks the
control loop.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable, Iterator, Optional

from legs_common.protocol.canon import Action, Observation
from legs_common.serialization.codec import pack, unpack
from legs_common.time import mono_ns

logger = logging.getLogger(__name__)

try:
    import grpc
    from legs_server.generated import legs_pb2, legs_pb2_grpc
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
        self._channel: Optional[grpc.Channel] = None  # type: ignore[assignment]
        self._running = False
        self._connected = False
        self._obs_queue: queue.Queue[Observation] = queue.Queue(maxsize=64)
        self._on_action_callback: Optional[Callable[[Action], None]] = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    def send_observation(self, obs: Observation) -> None:
        """Queue an observation for sending. Non-blocking, drops oldest if full."""
        try:
            self._obs_queue.put_nowait(obs)
        except queue.Full:
            # Drop oldest to make room
            try:
                self._obs_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._obs_queue.put_nowait(obs)
            except queue.Full:
                pass

    def connect(self, on_action: Callable[[Action], None]) -> None:
        """Start the gRPC streaming connection in a background thread."""
        if not HAS_GRPC:
            logger.warning("grpc not installed — running in offline/stub mode")
            return
        self._on_action_callback = on_action
        self._running = True
        thread = threading.Thread(target=self._stream_loop, daemon=True, name="grpc-client")
        thread.start()

    def _create_channel(self) -> grpc.Channel:  # type: ignore[return]
        """Create gRPC channel (insecure or TLS)."""
        if self._tls_cert_path and self._tls_key_path:
            cert = open(self._tls_cert_path, "rb").read()
            key = open(self._tls_key_path, "rb").read()
            ca = open(self._tls_ca_path, "rb").read() if self._tls_ca_path else None
            creds = grpc.ssl_channel_credentials(
                root_certificates=ca,
                private_key=key,
                certificate_chain=cert,
            )
            return grpc.secure_channel(self._server_addr, creds)
        else:
            return grpc.insecure_channel(self._server_addr)

    def _obs_to_proto(self, obs: Observation) -> legs_pb2.Observation:  # type: ignore[name-defined]
        """Convert canonical Observation to protobuf."""
        import msgpack
        header = legs_pb2.Header(
            session_id=obs.session_id,
            episode_id=obs.episode_id,
            seq=obs.seq,
            t_wall_ns=obs.t_wall_ns,
            t_mono_ns=obs.t_mono_ns,
            source=obs.source,
            frame_id=obs.frame_id,
        )
        payload = msgpack.packb(
            {"robot_state": obs.robot_state, "sensors": obs.sensors},
            use_bin_type=True,
        )
        return legs_pb2.Observation(h=header, payload=payload, encoding="msgpack")

    def _proto_to_action(self, proto: legs_pb2.Action) -> Action:  # type: ignore[name-defined]
        """Convert protobuf Action to canonical Action."""
        import msgpack
        payload: dict = {}
        if proto.payload:
            payload = msgpack.unpackb(proto.payload, raw=False)
        return Action(
            seq_ref=proto.seq_ref,
            action_type=proto.action_type,
            payload=payload,
            model_id=proto.model_id,
            t_infer_ns=proto.t_infer_ns,
        )

    def _obs_generator(self) -> Iterator[legs_pb2.Observation]:  # type: ignore[name-defined]
        """Yield protobuf Observations from the queue. Blocks until data or shutdown."""
        while self._running:
            try:
                obs = self._obs_queue.get(timeout=0.05)
                yield self._obs_to_proto(obs)
            except queue.Empty:
                continue

    def _stream_loop(self) -> None:
        """Main streaming loop — connects, streams, reconnects on failure."""
        while self._running:
            try:
                logger.info("Connecting to server at %s ...", self._server_addr)
                self._channel = self._create_channel()

                # Wait for channel to be ready (with timeout)
                try:
                    grpc.channel_ready_future(self._channel).result(timeout=10)
                except grpc.FutureTimeoutError:
                    logger.warning("Server not reachable at %s — retrying in 2s", self._server_addr)
                    self._channel.close()
                    time.sleep(2.0)
                    continue

                stub = legs_pb2_grpc.LegsInferenceStub(self._channel)
                self._connected = True
                logger.info("Connected to %s — starting StreamInfer", self._server_addr)

                # Bidirectional streaming
                response_iterator = stub.StreamInfer(self._obs_generator())

                for proto_action in response_iterator:
                    if not self._running:
                        break
                    action = self._proto_to_action(proto_action)
                    if self._on_action_callback is not None:
                        self._on_action_callback(action)

                logger.info("StreamInfer ended normally")

            except grpc.RpcError as e:
                logger.warning(
                    "gRPC error: %s — reconnecting in 2s",
                    e.code().name if hasattr(e, "code") else str(e),
                )
            except Exception as exc:
                logger.warning("gRPC connection error: %s — reconnecting in 2s", exc)
            finally:
                self._connected = False
                if self._channel is not None:
                    try:
                        self._channel.close()
                    except Exception:
                        pass
                if self._running:
                    time.sleep(2.0)

    def ping(self) -> Optional[float]:
        """Send a Ping and return RTT in ms, or None on failure."""
        if not HAS_GRPC or self._channel is None:
            return None
        try:
            stub = legs_pb2_grpc.LegsInferenceStub(self._channel)
            t0 = mono_ns()
            request = legs_pb2.Heartbeat(
                node_id="dog",
                role="dog",
                t_mono_ns=t0,
                health="ok",
            )
            response = stub.Ping(request, timeout=5)
            rtt_ms = (mono_ns() - t0) / 1e6
            return rtt_ms
        except Exception as e:
            logger.debug("Ping failed: %s", e)
            return None

    def close(self) -> None:
        self._running = False
        self._connected = False
        if self._channel is not None:
            try:
                self._channel.close()
            except Exception:
                pass
