"""gRPC contract tests — start a local server, simulate Dog client streaming."""

from __future__ import annotations

import os
import threading
import time
from typing import Iterator, List

import grpc
import pytest

from legs_common.protocol.canon import Observation
from legs_common.serialization.codec import pack
from legs_common.time import mono_ns, wall_ns
from legs_common.ids import new_session_id, new_episode_id

from legs_server.generated import legs_pb2, legs_pb2_grpc
from legs_server.model.nwm_infer import StubNWMPolicy
from legs_server.service.grpc_server import serve

# Ensure localhost connections bypass any HTTP proxy
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")


def _make_proto_obs(seq: int, session_id: str, episode_id: str) -> legs_pb2.Observation:
    """Build a protobuf Observation for testing."""
    import msgpack

    header = legs_pb2.Header(
        session_id=session_id,
        episode_id=episode_id,
        seq=seq,
        t_wall_ns=wall_ns(),
        t_mono_ns=mono_ns(),
        source="sim",
        frame_id="base_link",
    )
    payload = msgpack.packb(
        {"robot_state": {"positions": [0.0] * 12}, "sensors": {}},
        use_bin_type=True,
    )
    return legs_pb2.Observation(h=header, payload=payload, encoding="msgpack")


@pytest.fixture(scope="module")
def grpc_server_and_channel():
    """Start a local gRPC server with StubNWMPolicy and yield a channel."""
    policy = StubNWMPolicy(model_tag="test-model")
    server, port = serve(policy=policy, bind_addr="127.0.0.1:0", max_workers=2)

    channel = grpc.insecure_channel(f"127.0.0.1:{port}")

    # Wait for server to be ready
    try:
        grpc.channel_ready_future(channel).result(timeout=10)
    except grpc.FutureTimeoutError:
        pytest.fail("gRPC server did not become ready in time")

    yield server, channel, policy

    channel.close()
    server.stop(grace=0)


def test_stream_infer_100_frames(grpc_server_and_channel):
    """Send 100 Observations, verify Action.seq_ref monotonically matches."""
    server, channel, policy = grpc_server_and_channel
    stub = legs_pb2_grpc.LegsInferenceStub(channel)

    session_id = new_session_id()
    episode_id = new_episode_id()
    num_frames = 100

    def obs_generator() -> Iterator[legs_pb2.Observation]:
        for seq in range(1, num_frames + 1):
            yield _make_proto_obs(seq, session_id, episode_id)

    responses: List[legs_pb2.Action] = []
    for action in stub.StreamInfer(obs_generator()):
        responses.append(action)

    # Verify we got responses for all frames
    assert len(responses) == num_frames, f"Expected {num_frames} responses, got {len(responses)}"

    # Verify seq_ref is monotonically increasing and matches observation seq
    for i, action in enumerate(responses):
        expected_seq = i + 1
        assert action.seq_ref == expected_seq, (
            f"Action[{i}].seq_ref={action.seq_ref}, expected {expected_seq}"
        )

    # Verify seq_ref is monotonically increasing
    seq_refs = [a.seq_ref for a in responses]
    for i in range(1, len(seq_refs)):
        assert seq_refs[i] > seq_refs[i - 1], (
            f"seq_ref not monotonic at index {i}: {seq_refs[i-1]} -> {seq_refs[i]}"
        )


def test_ping_heartbeat(grpc_server_and_channel):
    """Verify Ping RPC returns a valid Heartbeat response."""
    server, channel, _ = grpc_server_and_channel
    stub = legs_pb2_grpc.LegsInferenceStub(channel)

    request = legs_pb2.Heartbeat(
        node_id="dog-test",
        role="dog",
        t_mono_ns=mono_ns(),
        last_seq_sent=42,
        last_seq_recv=0,
        health="ok",
    )
    response = stub.Ping(request)

    assert response.node_id == "server"
    assert response.role == "server"
    assert response.t_mono_ns > 0
    assert response.last_seq_recv == 42  # echoes back our last_seq_sent
    assert response.health == "ok"


def test_get_model_info(grpc_server_and_channel):
    """Verify GetModelInfo returns the correct model_id."""
    server, channel, policy = grpc_server_and_channel
    stub = legs_pb2_grpc.LegsInferenceStub(channel)

    response = stub.GetModelInfo(legs_pb2.ModelInfoRequest())

    assert response.model_id == policy.model_id()
    assert len(response.model_id) > 0
