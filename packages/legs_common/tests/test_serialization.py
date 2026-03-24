"""Tests for serialization roundtrip (pack/unpack) across all message types."""

from legs_common.protocol.canon import Action, Heartbeat, Observation
from legs_common.serialization.codec import from_dict, pack, to_dict, unpack


def test_observation_roundtrip():
    obs = Observation(
        session_id="ses-abc",
        episode_id="ep-def",
        seq=42,
        t_wall_ns=111,
        t_mono_ns=222,
        source="sim",
        robot_state={"positions": [0.1, 0.2, 0.3]},
        sensors={"imu": {"roll": 0.01}},
    )
    data = pack(obs)
    assert isinstance(data, bytes)
    assert len(data) > 0

    restored = unpack(Observation, data)
    assert restored.session_id == obs.session_id
    assert restored.seq == obs.seq
    assert restored.source == obs.source
    assert restored.robot_state == obs.robot_state
    assert restored.sensors == obs.sensors


def test_action_roundtrip():
    act = Action(
        seq_ref=10,
        action_type="foot_targets",
        payload={"targets": [[0.1, 0.2, -0.3]]},
        model_id="model-v1",
        t_infer_ns=5000,
    )
    restored = unpack(Action, pack(act))
    assert restored.seq_ref == act.seq_ref
    assert restored.action_type == act.action_type
    assert restored.model_id == act.model_id
    assert restored.payload == act.payload


def test_heartbeat_roundtrip():
    hb = Heartbeat(
        node_id="dog-01",
        role="dog",
        t_mono_ns=999,
        health="degraded",
        rtt_ms_p50=1.5,
        rtt_ms_p99=5.2,
    )
    restored = unpack(Heartbeat, pack(hb))
    assert restored.node_id == hb.node_id
    assert restored.health == hb.health
    assert restored.rtt_ms_p50 == hb.rtt_ms_p50


def test_to_dict_from_dict():
    obs = Observation(
        session_id="s", episode_id="e", seq=1,
        t_wall_ns=0, t_mono_ns=0, source="sim",
    )
    d = to_dict(obs)
    assert isinstance(d, dict)
    assert d["session_id"] == "s"

    # from_dict should ignore unknown keys
    d["unknown_field"] = "hello"
    restored = from_dict(Observation, d)
    assert restored.session_id == "s"
    assert not hasattr(restored, "unknown_field")
