"""Tests for ControlLoop non-blocking behavior and E-Stop safety."""

import time

from legs_common.protocol.canon import Action
from legs_dog.control.low_level import ActionBuffer, ControlLoop
from legs_dog.control.safety import SafetyConfig, SafetyGuard
from legs_dog.sim.adapters import FakeActuator


def test_control_loop_non_blocking():
    """Control loop step() must complete quickly even without any action."""
    actuator = FakeActuator()
    action_buffer = ActionBuffer()
    safety = SafetyGuard(SafetyConfig(heartbeat_timeout_ms=10000))

    ctrl = ControlLoop(actuator=actuator, action_buffer=action_buffer, safety=safety, period_ms=10)

    t0 = time.monotonic()
    for _ in range(100):
        ctrl.step()
    elapsed_ms = (time.monotonic() - t0) * 1000

    # 100 steps should complete in well under 100ms (no network blocking)
    assert elapsed_ms < 100, f"100 steps took {elapsed_ms:.1f}ms — too slow"
    assert ctrl.stats.step_count == 100


def test_control_loop_applies_action():
    """Control loop should apply action from buffer to actuator."""
    actuator = FakeActuator()
    action_buffer = ActionBuffer()
    safety = SafetyGuard(SafetyConfig(heartbeat_timeout_ms=10000))

    ctrl = ControlLoop(actuator=actuator, action_buffer=action_buffer, safety=safety)

    action = Action(
        seq_ref=1,
        action_type="joint_targets",
        payload={"joint_targets": [0.5] * 12},
        model_id="test-model",
    )
    action_buffer.put(action)
    ctrl.step()

    # Actuator should have received the action
    state = actuator.get_state()
    assert not state["estopped"]
    assert any(p != 0.0 for p in state["positions"])


def test_estop_latch():
    """Once E-Stop is triggered, control loop must not apply new actions."""
    actuator = FakeActuator()
    action_buffer = ActionBuffer()
    safety = SafetyGuard(SafetyConfig(heartbeat_timeout_ms=10000))

    ctrl = ControlLoop(actuator=actuator, action_buffer=action_buffer, safety=safety)

    # Trigger E-Stop
    safety.trigger_estop("unit test")
    assert safety.is_estopped

    # Put an action and step — it should NOT be applied
    action = Action(seq_ref=1, action_type="twist", payload={"vx": 1.0})
    action_buffer.put(action)
    ctrl.step()

    # Actuator should be in E-Stop state
    assert actuator._estopped


def test_action_buffer_latest_wins():
    """ActionBuffer should always return the most recent action."""
    buf = ActionBuffer()
    a1 = Action(seq_ref=1, action_type="twist")
    a2 = Action(seq_ref=2, action_type="twist")
    a3 = Action(seq_ref=3, action_type="twist")

    buf.put(a1)
    buf.put(a2)
    buf.put(a3)

    latest = buf.get()
    assert latest is not None
    assert latest.seq_ref == 3
