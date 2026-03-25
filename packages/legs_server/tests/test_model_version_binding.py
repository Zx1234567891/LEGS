"""Tests for model version binding — every Action must carry correct model_id."""

from __future__ import annotations

from legs_common.protocol.canon import Observation
from legs_common.time import mono_ns, wall_ns
from legs_common.ids import new_session_id, new_episode_id

from legs_server.model.nwm_infer import RandomNWMPolicy, StubNWMPolicy


def _make_obs(seq: int) -> Observation:
    return Observation(
        session_id=new_session_id(),
        episode_id=new_episode_id(),
        seq=seq,
        t_wall_ns=wall_ns(),
        t_mono_ns=mono_ns(),
        source="sim",
    )


def test_stub_policy_model_id_binding():
    """Every Action from StubNWMPolicy must carry the correct model_id."""
    policy = StubNWMPolicy(model_tag="bind-test-v1")
    expected_model_id = policy.model_id()
    assert len(expected_model_id) > 0

    for seq in range(1, 51):
        obs = _make_obs(seq)
        action = policy.infer(obs)
        assert action.model_id == expected_model_id, (
            f"seq={seq}: action.model_id={action.model_id!r} != {expected_model_id!r}"
        )
        assert action.seq_ref == seq


def test_random_policy_model_id_binding():
    """Every Action from RandomNWMPolicy must carry the correct model_id."""
    policy = RandomNWMPolicy(model_tag="rand-bind-test")
    expected_model_id = policy.model_id()

    for seq in range(1, 51):
        obs = _make_obs(seq)
        action = policy.infer(obs)
        assert action.model_id == expected_model_id
        assert action.seq_ref == seq


def test_different_policies_different_ids():
    """Two policies with different tags must produce different model_ids."""
    p1 = StubNWMPolicy(model_tag="alpha-v1")
    p2 = StubNWMPolicy(model_tag="beta-v2")
    assert p1.model_id() != p2.model_id()


def test_action_has_infer_timing():
    """Every Action must have a non-negative t_infer_ns."""
    policy = StubNWMPolicy()
    obs = _make_obs(1)
    action = policy.infer(obs)
    assert action.t_infer_ns >= 0


def test_action_payload_has_joint_targets():
    """StubNWMPolicy must produce actions with joint_targets payload."""
    policy = StubNWMPolicy()
    obs = _make_obs(1)
    action = policy.infer(obs)
    assert "joint_targets" in action.payload
    assert len(action.payload["joint_targets"]) == 12
