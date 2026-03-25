"""gRPC server implementing the LegsInference service.

Receives streaming Observations from Dog, runs NWMPolicy.infer(), and
streams Actions back.  Also serves Ping (latency probe) and GetModelInfo.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent import futures
from typing import Iterator, Optional

import grpc

from legs_common.protocol.canon import Action, Heartbeat, ModelInfo, Observation
from legs_common.serialization.codec import pack, unpack
from legs_common.time import mono_ns
from legs_server.generated import legs_pb2, legs_pb2_grpc
from legs_server.model.nwm_interface import NWMPolicy

logger = logging.getLogger(__name__)


class LegsInferenceServicer(legs_pb2_grpc.LegsInferenceServicer):
    """Concrete implementation of the LegsInference gRPC service."""

    def __init__(self, policy: NWMPolicy) -> None:
        self._policy = policy
        logger.info("LegsInferenceServicer ready [model_id=%s]", policy.model_id())

    def StreamInfer(
        self,
        request_iterator: Iterator[legs_pb2.Observation],
        context: grpc.ServicerContext,
    ) -> Iterator[legs_pb2.Action]:
        """Bidirectional streaming: receive Observations, yield Actions."""
        peer = context.peer() or "unknown"
        logger.info("StreamInfer session started [peer=%s]", peer)

        try:
            for proto_obs in request_iterator:
                # Decode protobuf Observation -> canonical Observation
                canon_obs = _proto_obs_to_canon(proto_obs)

                # Run inference
                canon_action = self._policy.infer(canon_obs)

                # Encode canonical Action -> protobuf Action
                proto_action = _canon_action_to_proto(canon_action)
                yield proto_action

        except grpc.RpcError:
            logger.info("StreamInfer ended (client disconnected) [peer=%s]", peer)
        except Exception:
            logger.exception("StreamInfer error [peer=%s]", peer)
            context.abort(grpc.StatusCode.INTERNAL, "inference error")

        logger.info("StreamInfer session ended [peer=%s]", peer)

    def Ping(
        self,
        request: legs_pb2.Heartbeat,
        context: grpc.ServicerContext,
    ) -> legs_pb2.Heartbeat:
        """Reply to heartbeat / latency probe."""
        reply = legs_pb2.Heartbeat(
            node_id="server",
            role="server",
            t_mono_ns=mono_ns(),
            last_seq_sent=0,
            last_seq_recv=request.last_seq_sent,
            health="ok",
        )
        return reply

    def GetModelInfo(
        self,
        request: legs_pb2.ModelInfoRequest,
        context: grpc.ServicerContext,
    ) -> legs_pb2.ModelInfoResponse:
        """Return current model version metadata."""
        mid = self._policy.model_id()
        return legs_pb2.ModelInfoResponse(
            model_id=mid,
            artifact_sha256="",
            git_commit="",
            container_image="",
            created_at="",
        )


# ---------------------------------------------------------------------------
# Proto <-> canonical conversion helpers
# ---------------------------------------------------------------------------

def _proto_obs_to_canon(proto: legs_pb2.Observation) -> Observation:
    """Convert a protobuf Observation to the canonical dataclass."""
    header = proto.h
    # Payload is msgpack-encoded sensor/robot data from Dog
    payload_data: dict = {}
    if proto.payload:
        encoding = proto.encoding or "msgpack"
        if encoding == "msgpack":
            import msgpack
            payload_data = msgpack.unpackb(proto.payload, raw=False)
        elif encoding == "json":
            import json
            payload_data = json.loads(proto.payload)

    return Observation(
        session_id=header.session_id,
        episode_id=header.episode_id,
        seq=header.seq,
        t_wall_ns=header.t_wall_ns,
        t_mono_ns=header.t_mono_ns,
        source=header.source,
        frame_id=header.frame_id,
        robot_state=payload_data.get("robot_state", {}),
        sensors=payload_data.get("sensors", {}),
    )


def _canon_action_to_proto(action: Action) -> legs_pb2.Action:
    """Convert a canonical Action dataclass to protobuf."""
    import msgpack
    payload_bytes = msgpack.packb(action.payload, use_bin_type=True)
    return legs_pb2.Action(
        seq_ref=action.seq_ref,
        action_type=action.action_type,
        payload=payload_bytes,
        model_id=action.model_id,
        t_infer_ns=action.t_infer_ns,
    )


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

def serve(
    policy: NWMPolicy,
    bind_addr: str = "0.0.0.0:50051",
    max_workers: int = 4,
    tls_cert_path: Optional[str] = None,
    tls_key_path: Optional[str] = None,
    tls_ca_path: Optional[str] = None,
) -> tuple[grpc.Server, int]:
    """Create, configure and start the gRPC server.

    Returns:
        A tuple of (server, port) where port is the actual port bound
        (useful when bind_addr uses port 0 for auto-assignment).
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    servicer = LegsInferenceServicer(policy)
    legs_pb2_grpc.add_LegsInferenceServicer_to_server(servicer, server)

    if tls_cert_path and tls_key_path:
        # mTLS / TLS mode
        cert = _read_file(tls_cert_path)
        key = _read_file(tls_key_path)
        ca: Optional[bytes] = _read_file(tls_ca_path) if tls_ca_path else None

        if ca:
            creds = grpc.ssl_server_credentials(
                [(key, cert)],
                root_certificates=ca,
                require_client_auth=True,
            )
        else:
            creds = grpc.ssl_server_credentials([(key, cert)])

        port = server.add_secure_port(bind_addr, creds)
        logger.info("gRPC server listening on %s (TLS, port=%d)", bind_addr, port)
    else:
        port = server.add_insecure_port(bind_addr)
        logger.info("gRPC server listening on %s (insecure, port=%d)", bind_addr, port)

    server.start()
    return server, port


def _read_file(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()
