"""Tests for canonical schema defaults and field behavior."""

from legs_common.protocol.canon import Action, EStop, Heartbeat, ModelInfo, Observation


def test_observation_defaults():
    """Observation should have sensible defaults for optional fields."""
    obs = Observation(
        session_id="ses-test",
        episode_id="ep-test",
        seq=1,
        t_wall_ns=1000,
        t_mono_ns=2000,
        source="sim",
    )
    assert obs.frame_id == "base_link"
    assert obs.robot_state == {}
    assert obs.sensors == {}
    assert obs.debug is None


def test_action_defaults():
    """Action should have sensible defaults."""
    act = Action(seq_ref=1, action_type="twist")
    assert act.payload == {}
    assert act.model_id == ""
    assert act.t_infer_ns == 0


def test_heartbeat_defaults():
    hb = Heartbeat(node_id="dog-01", role="dog")
    assert hb.health == "ok"
    assert hb.rtt_ms_p50 == 0.0
    assert hb.last_seq_sent == 0


def test_modelinfo_defaults():
    mi = ModelInfo(model_id="m-001")
    assert mi.artifact_sha256 == ""
    assert mi.git_commit == ""


def test_estop_defaults():
    es = EStop(reason="test")
    assert es.latch is True
    assert es.issued_by == ""
