"""Tests for sensor implementations."""

import pytest

try:
    import pybullet
    HAS_PYBULLET = True
except ImportError:
    HAS_PYBULLET = False

pytestmark = pytest.mark.skipif(not HAS_PYBULLET, reason="pybullet not installed")


@pytest.fixture(scope="module")
def sim():
    from legs_dog.sim.pybullet_sim import PyBulletSim
    s = PyBulletSim(gui=False, scene="outdoor")
    yield s
    s.close()


def test_camera_frame_shape(sim):
    """Camera should produce 224x224x3 RGB frames."""
    frame = sim.camera.get_current_frame()
    assert frame.shape == (224, 224, 3)
    assert frame.dtype.name == "uint8"


def test_camera_context_buffer(sim):
    """Context buffer should pad with zeros then fill with real frames."""
    # Read a few frames
    for _ in range(6):
        sim.camera.read()
    frames = sim.camera.get_context_frames()
    assert len(frames) == 4
    for f in frames:
        assert f.shape == (224, 224, 3)


def test_camera_sensor_frame(sim):
    """camera.read() should return a valid SensorFrame."""
    frame = sim.camera.read()
    assert frame is not None
    assert frame.frame_id == "camera_link"
    assert frame.encoding == "json"


def test_lidar_scan(sim):
    """LiDAR scan should return valid distance data."""
    scan = sim.lidar.scan()
    assert len(scan["distances"]) == 360
    assert len(scan["angles"]) == 360
    assert all(0 <= d <= 10.0 for d in scan["distances"])


def test_lidar_occupancy_grid(sim):
    """Occupancy grid should be a 2D array of the right size."""
    grid = sim.lidar.get_occupancy_grid(grid_size=10.0, resolution=0.1)
    assert grid.shape == (100, 100)
    assert grid.dtype.name == "uint8"


def test_lidar_sensor_frame(sim):
    """lidar.read() should return a valid SensorFrame."""
    frame = sim.lidar.read()
    assert frame is not None
    assert frame.frame_id == "lidar_link"


def test_depth_camera():
    """Depth camera should produce float32 depth images."""
    from legs_dog.sim.pybullet_env import PyBulletQuadrupedEnv
    from legs_dog.sensors.depth_camera import PyBulletDepthCamera

    env = PyBulletQuadrupedEnv(gui=False, scene="indoor")
    try:
        cam = PyBulletDepthCamera(
            physics_client=env.physics_client,
            robot_id=env.robot_id,
            width=64, height=64,
        )
        depth = cam.get_depth_image()
        assert depth.shape == (64, 64)
        assert depth.dtype.name == "float32"
        assert depth.min() >= 0

        frame = cam.read()
        assert frame is not None
        assert frame.frame_id == "depth_camera_link"
    finally:
        env.close()


def test_observation_contains_all_sensors(sim):
    """build_observation() should include all 4 sensor types + goal."""
    sim.set_goal(3.0, 2.0)
    obs = sim.build_observation()
    assert "rgb_camera" in obs.sensors
    assert "lidar" in obs.sensors
    assert "imu" in obs.sensors
    assert "joint_state" in obs.sensors
    assert "goal" in obs.sensors
    assert obs.robot_state["x"] is not None
    assert obs.robot_state["yaw"] is not None
