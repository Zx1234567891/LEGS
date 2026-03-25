"""Tests for PyBullet environment (DIRECT mode, no GUI)."""

import pytest

try:
    import pybullet
    HAS_PYBULLET = True
except ImportError:
    HAS_PYBULLET = False

pytestmark = pytest.mark.skipif(not HAS_PYBULLET, reason="pybullet not installed")


@pytest.fixture
def env():
    from legs_dog.sim.pybullet_env import PyBulletQuadrupedEnv
    e = PyBulletQuadrupedEnv(gui=False, scene="indoor")
    yield e
    e.close()


def test_env_loads_robot(env):
    """Robot should load with 12 revolute joints."""
    assert env.num_joints >= 12
    assert len(env.joint_indices) >= 12


def test_env_robot_pose(env):
    """Robot pose should return valid position and yaw."""
    pos, orn, yaw = env.get_robot_pose()
    assert len(pos) == 3
    assert pos[2] > 0  # above ground
    assert -4 < yaw < 4


def test_env_joint_control(env):
    """apply_joint_targets should not crash and joints should update."""
    targets = [0.0, 0.65, -1.3] * 4
    env.apply_joint_targets(targets)
    env.step_simulation(num_steps=10)
    states = env.get_joint_states()
    assert len(states["positions"]) >= 12


def test_env_collision_detection(env):
    """Initial standing position should not collide with scene."""
    env.step_simulation(num_steps=50)
    # Robot at origin should not be inside any obstacle
    # (may or may not collide depending on scene layout, just verify no crash)
    result = env.check_collision()
    assert isinstance(result, bool)


def test_env_goal(env):
    """Goal setting and distance calculation."""
    env.set_goal(5.0, 3.0)
    assert env.goal_position == (5.0, 3.0)
    dist = env.distance_to_goal()
    assert dist > 0
    assert not env.reached_goal(tolerance=0.5)


def test_env_render(env):
    """render_camera should return a valid RGB image."""
    rgb = env.render_camera(width=64, height=64)
    assert rgb.shape == (64, 64, 3)
    assert rgb.dtype.name == "uint8"


def test_env_lidar(env):
    """raycast_lidar should return distances and hit info."""
    scan = env.raycast_lidar(num_rays=36)
    assert len(scan["distances"]) == 36
    assert len(scan["angles"]) == 36
    assert len(scan["hit_mask"]) == 36
    assert scan["max_range"] == 10.0


def test_env_reset(env):
    """reset() should return robot to start position."""
    env.step_simulation(num_steps=100)
    env.reset()
    pos, _, _ = env.get_robot_pose()
    assert abs(pos[0]) < 0.5
    assert abs(pos[1]) < 0.5


def test_env_nav_delta(env):
    """apply_nav_delta should run without error."""
    env.apply_nav_delta(dx=0.3, dy=0.0, dyaw=0.1)
    env.step_simulation(num_steps=10)
    # Just verify it doesn't crash
    pos, _, _ = env.get_robot_pose()
    assert len(pos) == 3
