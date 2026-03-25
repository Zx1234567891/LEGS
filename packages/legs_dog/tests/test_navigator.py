"""Tests for Navigator logic (DIRECT mode, no GUI)."""

import pytest

try:
    import pybullet
    HAS_PYBULLET = True
except ImportError:
    HAS_PYBULLET = False

pytestmark = pytest.mark.skipif(not HAS_PYBULLET, reason="pybullet not installed")


@pytest.fixture
def nav_setup():
    """Set up Navigator with PyBulletSim in offline mode."""
    from legs_dog.sim.pybullet_sim import PyBulletSim
    from legs_dog.navigation import Navigator, NavigationGoal
    from legs_dog.control.low_level import ActionBuffer
    from legs_dog.control.safety import SafetyConfig, SafetyGuard
    from legs_server.model.nwm_infer import StubNWMPolicy

    sim = PyBulletSim(gui=False, scene="outdoor")
    action_buffer = ActionBuffer()
    safety = SafetyGuard(SafetyConfig(heartbeat_timeout_ms=10000))
    policy = StubNWMPolicy(model_tag="test-stub")

    navigator = Navigator(
        sim=sim,
        action_buffer=action_buffer,
        safety=safety,
        local_policy=policy,
        max_steps=50,
    )

    yield sim, navigator, safety

    sim.close()


def test_navigator_timeout(nav_setup):
    """Navigator should return False on timeout (goal too far for 50 steps)."""
    sim, navigator, _ = nav_setup
    from legs_dog.navigation import NavigationGoal

    goal = NavigationGoal(x=100.0, y=100.0, tolerance=0.5)
    result = navigator.navigate_to(goal)
    assert result is False


def test_navigator_estop_aborts(nav_setup):
    """Navigator should abort if E-Stop is triggered."""
    sim, navigator, safety = nav_setup
    from legs_dog.navigation import NavigationGoal

    safety.trigger_estop("test")
    goal = NavigationGoal(x=1.0, y=0.0, tolerance=0.5)
    result = navigator.navigate_to(goal)
    assert result is False


def test_navigator_stats(nav_setup):
    """_run_to_goal should return NavigationStats with valid data."""
    sim, navigator, _ = nav_setup
    from legs_dog.navigation import NavigationGoal

    goal = NavigationGoal(x=50.0, y=0.0, tolerance=0.5)
    stats = navigator._run_to_goal(goal)
    assert stats.total_steps == 50
    assert stats.total_time_s > 0
    assert len(stats.trajectory) > 0
    assert stats.reached is False


def test_navigator_waypoints(nav_setup):
    """navigate_waypoints should handle a list of goals."""
    sim, navigator, _ = nav_setup
    from legs_dog.navigation import NavigationGoal

    goals = [
        NavigationGoal(x=100.0, y=0.0),
        NavigationGoal(x=100.0, y=100.0),
    ]
    stats = navigator.navigate_waypoints(goals)
    assert stats.total_steps > 0
    assert stats.reached is False  # too far in 50 steps
