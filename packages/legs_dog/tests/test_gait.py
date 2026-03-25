"""Tests for TrotGaitController output validity."""

import math
from legs_dog.control.gait_controller import TrotGaitController, GaitConfig


def test_gait_standing_pose():
    """Standing pose should be 12 elements with known pattern."""
    gait = TrotGaitController()
    pose = gait.standing_pose()
    assert len(pose) == 12
    # Each leg: [0.0, 0.65, -1.3]
    for leg in range(4):
        assert pose[leg * 3] == 0.0
        assert pose[leg * 3 + 1] == 0.65
        assert pose[leg * 3 + 2] == -1.3


def test_gait_compute_returns_12_joints():
    """compute() must always return exactly 12 joint angles."""
    gait = TrotGaitController()
    for _ in range(100):
        targets = gait.compute(dx=0.5, dy=0.1, dyaw=0.2, dt=1 / 240)
        assert len(targets) == 12


def test_gait_joint_ranges():
    """Joint angles must stay within physically reasonable bounds."""
    gait = TrotGaitController()
    for _ in range(500):
        targets = gait.compute(dx=1.0, dy=0.5, dyaw=0.5, dt=1 / 240)
        for angle in targets:
            assert -3.14 < angle < 3.14, f"Joint angle {angle} out of range"


def test_gait_zero_command_near_standing():
    """With zero velocity command, output should be close to standing pose."""
    gait = TrotGaitController()
    targets = gait.compute(dx=0.0, dy=0.0, dyaw=0.0, dt=1 / 240)
    stand = gait.standing_pose()
    for i in range(12):
        assert abs(targets[i] - stand[i]) < 0.2, (
            f"Joint {i}: {targets[i]} too far from standing {stand[i]}"
        )


def test_gait_phase_advances():
    """Phase should monotonically advance with each compute() call."""
    gait = TrotGaitController()
    phases = []
    for _ in range(20):
        gait.compute(dx=0.3, dy=0.0, dyaw=0.0, dt=1 / 240)
        phases.append(gait.phase)
    # Phase wraps around 2π, but should change each step
    diffs = [phases[i + 1] - phases[i] for i in range(len(phases) - 1)]
    # Most diffs should be positive (some may wrap)
    positive = sum(1 for d in diffs if d > 0 or d < -5)
    assert positive >= len(diffs) - 1


def test_gait_reset():
    """reset() should zero the phase."""
    gait = TrotGaitController()
    gait.compute(dx=1.0, dy=0.0, dyaw=0.0, dt=0.1)
    assert gait.phase > 0
    gait.reset()
    assert gait.phase == 0.0
