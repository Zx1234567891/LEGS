"""PyBullet quadruped simulation environment for NWM navigation.

Loads a quadruped robot (Laikago/A1-style from pybullet_data) into a
navigable scene with walls, obstacles, and goal markers.  Provides
12-DOF joint control, collision detection, and third-person rendering.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from legs_dog.control.gait_controller import TrotGaitController

logger = logging.getLogger(__name__)

try:
    import pybullet as p
    import pybullet_data

    HAS_PYBULLET = True
except ImportError:
    HAS_PYBULLET = False

# 12-DOF joint layout (Laikago / Go1 style)
JOINT_NAMES: List[str] = [
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
]

# Default standing joint positions (radians)
STAND_ANGLES: List[float] = [
    0.0, 0.65, -1.3,  # FL
    0.0, 0.65, -1.3,  # FR
    0.0, 0.65, -1.3,  # RL
    0.0, 0.65, -1.3,  # RR
]


@dataclass
class SceneConfig:
    """Configuration for a navigation scene."""
    name: str = "indoor"
    # Walls as list of (pos_x, pos_y, half_ext_x, half_ext_y, half_ext_z, yaw)
    walls: List[Tuple[float, ...]] = field(default_factory=list)
    # Box obstacles as (pos_x, pos_y, pos_z, half_x, half_y, half_z)
    obstacles: List[Tuple[float, ...]] = field(default_factory=list)
    # Ground texture color
    ground_color: Tuple[float, ...] = (0.85, 0.85, 0.80, 1.0)


def make_indoor_scene() -> SceneConfig:
    """An indoor room with corridors and a few obstacles."""
    walls = [
        # (x, y, half_x, half_y, half_z, yaw)
        (5.0, 0.0, 0.1, 5.0, 1.0, 0.0),    # right wall
        (-5.0, 0.0, 0.1, 5.0, 1.0, 0.0),   # left wall
        (0.0, 5.0, 5.0, 0.1, 1.0, 0.0),    # top wall
        (0.0, -5.0, 5.0, 0.1, 1.0, 0.0),   # bottom wall
        # Internal corridor wall
        (0.0, 1.5, 2.5, 0.1, 0.8, 0.0),
        (2.0, -1.0, 0.1, 1.5, 0.8, 0.0),
    ]
    obstacles = [
        (3.0, 2.0, 0.25, 0.25, 0.25, 0.5),
        (-2.0, -2.0, 0.25, 0.3, 0.3, 0.4),
        (1.5, -3.0, 0.25, 0.2, 0.4, 0.35),
    ]
    return SceneConfig(name="indoor", walls=walls, obstacles=obstacles)


def make_maze_scene() -> SceneConfig:
    """A maze scene for testing MCTS planning."""
    walls = [
        # Outer walls
        (4.0, 0.0, 0.1, 4.0, 0.8, 0.0),
        (-4.0, 0.0, 0.1, 4.0, 0.8, 0.0),
        (0.0, 4.0, 4.0, 0.1, 0.8, 0.0),
        (0.0, -4.0, 4.0, 0.1, 0.8, 0.0),
        # Internal maze walls
        (-2.0, -1.0, 2.0, 0.1, 0.8, 0.0),
        (0.0, 1.5, 0.1, 2.0, 0.8, 0.0),
        (2.0, 0.0, 0.1, 2.5, 0.8, 0.0),
        (-1.5, 2.5, 1.5, 0.1, 0.8, 0.0),
    ]
    return SceneConfig(name="maze", walls=walls)


def make_outdoor_scene() -> SceneConfig:
    """Open outdoor scene with sparse obstacles."""
    obstacles = [
        (3.0, 1.0, 0.3, 0.3, 0.3, 0.6),
        (-1.5, 3.5, 0.3, 0.4, 0.4, 0.5),
        (5.0, -2.0, 0.3, 0.5, 0.3, 0.7),
        (-3.0, -4.0, 0.3, 0.25, 0.25, 0.4),
    ]
    return SceneConfig(name="outdoor", obstacles=obstacles)


SCENE_REGISTRY = {
    "indoor": make_indoor_scene,
    "maze": make_maze_scene,
    "outdoor": make_outdoor_scene,
}


def load_scene_from_yaml(path: str) -> SceneConfig:
    """Load a scene definition from a YAML file."""
    import yaml
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return SceneConfig(
        name=data.get("name", "custom"),
        walls=[tuple(w) for w in data.get("walls", [])],
        obstacles=[tuple(o) for o in data.get("obstacles", [])],
        ground_color=tuple(data.get("ground_color", [0.85, 0.85, 0.80, 1.0])),
    )


def _find_yaml_scene(name: str) -> Optional[str]:
    """Search for a YAML scene file in the assets/scenes directory."""
    scenes_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets", "scenes")
    candidate = os.path.join(scenes_dir, f"{name}.yaml")
    if os.path.isfile(candidate):
        return candidate
    return None


class PyBulletQuadrupedEnv:
    """PyBullet simulation environment for a 12-DOF quadruped robot.

    Manages physics, scene construction, robot loading, joint control,
    collision detection, goal visualization, and camera rendering.
    """

    def __init__(
        self,
        gui: bool = True,
        scene: str = "indoor",
        robot: str = "go1",
        time_step: float = 1.0 / 240.0,
        robot_start_pos: Tuple[float, float, float] = (0.0, 0.0, 0.42),
        robot_start_orn_yaw: float = 0.0,
    ) -> None:
        if not HAS_PYBULLET:
            raise RuntimeError("pybullet is not installed")

        self._gui = gui
        self._time_step = time_step
        self._robot_type = robot

        # Connect to physics server
        if gui:
            self._cid = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self._cid)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=self._cid)
        else:
            self._cid = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._cid)
        p.setGravity(0, 0, -9.81, physicsClientId=self._cid)
        p.setTimeStep(time_step, physicsClientId=self._cid)

        # Load ground plane
        self._plane_id = p.loadURDF("plane.urdf", physicsClientId=self._cid)

        # Load robot
        start_orn = p.getQuaternionFromEuler([0, 0, robot_start_orn_yaw])
        urdf_path = self._resolve_urdf(robot)

        self._robot_id = p.loadURDF(
            urdf_path,
            basePosition=list(robot_start_pos),
            baseOrientation=list(start_orn),
            useFixedBase=False,
            flags=p.URDF_USE_SELF_COLLISION,
            physicsClientId=self._cid,
        )
        self._start_pos = robot_start_pos
        self._start_orn = start_orn

        # Discover controllable joints
        self._joint_indices: List[int] = []
        self._joint_names: List[str] = []
        num_joints = p.getNumJoints(self._robot_id, physicsClientId=self._cid)
        for i in range(num_joints):
            info = p.getJointInfo(self._robot_id, i, physicsClientId=self._cid)
            joint_type = info[2]
            joint_name = info[1].decode("utf-8")
            if joint_type == p.JOINT_REVOLUTE:
                self._joint_indices.append(i)
                self._joint_names.append(joint_name)

        self._num_joints = len(self._joint_indices)
        logger.info("Robot loaded: %d revolute joints: %s", self._num_joints, self._joint_names)

        # Set initial standing pose
        self._set_standing_pose()

        # Build scene
        self._scene_bodies: List[int] = []
        self._build_scene(scene)

        # Gait controller
        self._gait = TrotGaitController()

        # Goal marker
        self._goal_marker_id: Optional[int] = None
        self._goal_position: Optional[Tuple[float, float]] = None

        # Camera follow
        if gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=3.0,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=list(robot_start_pos),
                physicsClientId=self._cid,
            )

        # Step a few times to settle
        for _ in range(100):
            p.stepSimulation(physicsClientId=self._cid)

        logger.info("PyBulletQuadrupedEnv ready [robot=%s, scene=%s, joints=%d]", robot, scene, self._num_joints)

    # ------------------------------------------------------------------
    # URDF resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_urdf(robot: str) -> str:
        """Find the URDF file for the requested robot model.

        Priority:
          1. Go1 URDF in assets/urdf/go1_description/
          2. Laikago from pybullet_data (fallback)
        """
        if robot == "go1":
            # assets/ is at packages/legs_dog/assets/ (3 levels up from sim/)
            assets_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets", "urdf")
            go1_path = os.path.join(assets_dir, "go1_description", "urdf", "go1.urdf")
            if os.path.isfile(go1_path):
                logger.info("Loading Go1 URDF from %s", go1_path)
                return go1_path
            logger.warning("Go1 URDF not found at %s — falling back to Laikago", go1_path)

        # Laikago fallback
        urdf_path = os.path.join(pybullet_data.getDataPath(), "laikago", "laikago_toes.urdf")
        if not os.path.exists(urdf_path):
            urdf_path = os.path.join(pybullet_data.getDataPath(), "laikago", "laikago.urdf")
        logger.info("Loading Laikago URDF from %s", urdf_path)
        return urdf_path

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def physics_client(self) -> int:
        return self._cid

    @property
    def robot_id(self) -> int:
        return self._robot_id

    @property
    def joint_indices(self) -> List[int]:
        return self._joint_indices

    @property
    def num_joints(self) -> int:
        return self._num_joints

    @property
    def goal_position(self) -> Optional[Tuple[float, float]]:
        return self._goal_position

    # ------------------------------------------------------------------
    # Robot state queries
    # ------------------------------------------------------------------

    def get_robot_pose(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Return (position[3], orientation_quat[4], yaw)."""
        pos, orn = p.getBasePositionAndOrientation(self._robot_id, physicsClientId=self._cid)
        euler = p.getEulerFromQuaternion(orn)
        return np.array(pos), np.array(orn), euler[2]

    def get_robot_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (linear_vel[3], angular_vel[3])."""
        lin, ang = p.getBaseVelocity(self._robot_id, physicsClientId=self._cid)
        return np.array(lin), np.array(ang)

    def get_joint_states(self) -> Dict[str, Any]:
        """Return current joint positions and velocities."""
        states = p.getJointStates(self._robot_id, self._joint_indices, physicsClientId=self._cid)
        positions = [s[0] for s in states]
        velocities = [s[1] for s in states]
        return {
            "joint_names": self._joint_names[:len(positions)],
            "positions": positions,
            "velocities": velocities,
        }

    def get_imu_data(self) -> Dict[str, Any]:
        """Simulate IMU from base link state."""
        pos, orn = p.getBasePositionAndOrientation(self._robot_id, physicsClientId=self._cid)
        euler = p.getEulerFromQuaternion(orn)
        lin_vel, ang_vel = p.getBaseVelocity(self._robot_id, physicsClientId=self._cid)
        return {
            "orientation": {"roll": euler[0], "pitch": euler[1], "yaw": euler[2]},
            "angular_velocity": {"x": ang_vel[0], "y": ang_vel[1], "z": ang_vel[2]},
            "linear_acceleration": {"x": 0.0, "y": 0.0, "z": 9.81},
        }

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def apply_joint_targets(self, targets: List[float], max_force: float = 20.0) -> None:
        """Apply position control to all joints."""
        n = min(len(targets), self._num_joints)
        p.setJointMotorControlArray(
            self._robot_id,
            self._joint_indices[:n],
            p.POSITION_CONTROL,
            targetPositions=targets[:n],
            forces=[max_force] * n,
            physicsClientId=self._cid,
        )

    def apply_nav_delta(self, dx: float, dy: float, dyaw: float) -> None:
        """Convert a navigation delta (dx, dy, dyaw) into walking joint targets.

        Uses TrotGaitController for realistic diagonal trot gait with
        half-ellipse swing trajectories and Raibert-style stance retraction.
        """
        targets = self._gait.compute(dx=dx, dy=dy, dyaw=dyaw, dt=self._time_step * 4)
        self.apply_joint_targets(targets)

    def estop(self, reason: str = "") -> None:
        """Emergency stop — set all joints to standing pose with high damping."""
        self._set_standing_pose()
        logger.warning("PyBullet E-STOP: %s", reason)

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------

    def step_simulation(self, num_steps: int = 1) -> None:
        """Advance physics by num_steps sub-steps."""
        for _ in range(num_steps):
            p.stepSimulation(physicsClientId=self._cid)

        # Update camera follow in GUI mode
        if self._gui:
            pos, _, _ = self.get_robot_pose()
            p.resetDebugVisualizerCamera(
                cameraDistance=3.0,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=pos.tolist(),
                physicsClientId=self._cid,
            )

    # ------------------------------------------------------------------
    # Collision detection
    # ------------------------------------------------------------------

    def check_collision(self) -> bool:
        """Return True if robot is in contact with any scene obstacle/wall."""
        for body_id in self._scene_bodies:
            contacts = p.getContactPoints(
                bodyA=self._robot_id,
                bodyB=body_id,
                physicsClientId=self._cid,
            )
            if contacts:
                return True
        return False

    # ------------------------------------------------------------------
    # Goal management
    # ------------------------------------------------------------------

    def set_goal(self, x: float, y: float) -> None:
        """Place a visible goal marker at the given position."""
        self._goal_position = (x, y)

        # Remove old marker
        if self._goal_marker_id is not None:
            p.removeBody(self._goal_marker_id, physicsClientId=self._cid)

        # Create a green sphere as goal marker
        vis_shape = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.2,
            rgbaColor=[0.1, 0.9, 0.1, 0.7],
            physicsClientId=self._cid,
        )
        self._goal_marker_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=vis_shape,
            basePosition=[x, y, 0.3],
            physicsClientId=self._cid,
        )
        logger.info("Goal set at (%.2f, %.2f)", x, y)

    def distance_to_goal(self) -> float:
        """Return 2D distance from robot base to goal."""
        if self._goal_position is None:
            return float("inf")
        pos, _, _ = self.get_robot_pose()
        dx = pos[0] - self._goal_position[0]
        dy = pos[1] - self._goal_position[1]
        return math.sqrt(dx * dx + dy * dy)

    def reached_goal(self, tolerance: float = 0.5) -> bool:
        return self.distance_to_goal() < tolerance

    # ------------------------------------------------------------------
    # Scene building
    # ------------------------------------------------------------------

    def _build_scene(self, scene_name: str) -> None:
        """Build walls and obstacles from a scene configuration.

        Checks built-in registry first, then falls back to YAML files
        in assets/scenes/.
        """
        factory = SCENE_REGISTRY.get(scene_name)
        if factory is not None:
            config = factory()
        else:
            yaml_path = _find_yaml_scene(scene_name)
            if yaml_path is not None:
                config = load_scene_from_yaml(yaml_path)
                logger.info("Loaded YAML scene from %s", yaml_path)
            else:
                logger.warning("Unknown scene '%s', using empty scene", scene_name)
                return

        for wall in config.walls:
            x, y, hx, hy, hz, yaw = wall
            col_shape = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[hx, hy, hz],
                physicsClientId=self._cid,
            )
            vis_shape = p.createVisualShape(
                p.GEOM_BOX, halfExtents=[hx, hy, hz],
                rgbaColor=[0.6, 0.6, 0.65, 1.0],
                physicsClientId=self._cid,
            )
            orn = p.getQuaternionFromEuler([0, 0, yaw])
            body = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col_shape,
                baseVisualShapeIndex=vis_shape,
                basePosition=[x, y, hz],
                baseOrientation=orn,
                physicsClientId=self._cid,
            )
            self._scene_bodies.append(body)

        for obs in config.obstacles:
            x, y, z, hx, hy, hz = obs
            col_shape = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[hx, hy, hz],
                physicsClientId=self._cid,
            )
            vis_shape = p.createVisualShape(
                p.GEOM_BOX, halfExtents=[hx, hy, hz],
                rgbaColor=[0.8, 0.3, 0.2, 1.0],
                physicsClientId=self._cid,
            )
            body = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col_shape,
                baseVisualShapeIndex=vis_shape,
                basePosition=[x, y, z + hz],
                physicsClientId=self._cid,
            )
            self._scene_bodies.append(body)

        logger.info("Scene '%s' built: %d walls, %d obstacles",
                     config.name, len(config.walls), len(config.obstacles))

    def _set_standing_pose(self) -> None:
        """Reset joints to standing position."""
        n = min(len(STAND_ANGLES), self._num_joints)
        for i in range(n):
            p.resetJointState(
                self._robot_id,
                self._joint_indices[i],
                STAND_ANGLES[i],
                physicsClientId=self._cid,
            )
        self.apply_joint_targets(STAND_ANGLES[:n])

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset robot to starting position."""
        p.resetBasePositionAndOrientation(
            self._robot_id,
            list(self._start_pos),
            list(self._start_orn),
            physicsClientId=self._cid,
        )
        p.resetBaseVelocity(
            self._robot_id,
            linearVelocity=[0, 0, 0],
            angularVelocity=[0, 0, 0],
            physicsClientId=self._cid,
        )
        self._set_standing_pose()
        for _ in range(50):
            p.stepSimulation(physicsClientId=self._cid)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render_camera(
        self,
        width: int = 224,
        height: int = 224,
        link_index: int = -1,
        fov: float = 70.0,
        near: float = 0.1,
        far: float = 20.0,
    ) -> np.ndarray:
        """Render an RGB image from a camera mounted on the robot.

        If link_index == -1, uses the base link.
        Returns an (H, W, 3) uint8 numpy array.
        """
        if link_index == -1:
            cam_pos, cam_orn = p.getBasePositionAndOrientation(
                self._robot_id, physicsClientId=self._cid
            )
        else:
            link_state = p.getLinkState(
                self._robot_id, link_index, physicsClientId=self._cid
            )
            cam_pos, cam_orn = link_state[0], link_state[1]

        # Camera is placed slightly in front of and above the robot
        cam_pos = list(cam_pos)
        cam_pos[2] += 0.3  # raise camera

        rot_matrix = np.array(p.getMatrixFromQuaternion(cam_orn)).reshape(3, 3)
        forward = rot_matrix @ np.array([1.0, 0.0, 0.0])
        target = np.array(cam_pos) + forward * 1.0
        up = rot_matrix @ np.array([0.0, 0.0, 1.0])

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=cam_pos,
            cameraTargetPosition=target.tolist(),
            cameraUpVector=up.tolist(),
            physicsClientId=self._cid,
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=fov,
            aspect=float(width) / height,
            nearVal=near,
            farVal=far,
            physicsClientId=self._cid,
        )

        _, _, rgba, _, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL if self._gui else p.ER_TINY_RENDERER,
            physicsClientId=self._cid,
        )

        rgb = np.array(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
        return rgb

    # ------------------------------------------------------------------
    # LiDAR simulation
    # ------------------------------------------------------------------

    def raycast_lidar(
        self,
        num_rays: int = 360,
        max_range: float = 10.0,
        height_offset: float = 0.3,
    ) -> Dict[str, Any]:
        """Simulate a 2D planar LiDAR using ray casting.

        Returns dict with 'distances', 'angles', 'hit_points', 'hit_mask'.
        """
        pos, _, yaw = self.get_robot_pose()
        origin = np.array([pos[0], pos[1], pos[2] + height_offset])

        angles = np.linspace(0, 2 * math.pi, num_rays, endpoint=False)
        ray_from_list = []
        ray_to_list = []

        for angle in angles:
            world_angle = angle + yaw
            dx = math.cos(world_angle) * max_range
            dy = math.sin(world_angle) * max_range
            ray_from_list.append(origin.tolist())
            ray_to_list.append([origin[0] + dx, origin[1] + dy, origin[2]])

        results = p.rayTestBatch(
            ray_from_list, ray_to_list, physicsClientId=self._cid
        )

        distances = []
        hit_points = []
        hit_mask = []
        for i, result in enumerate(results):
            obj_id, link_idx, frac, hit_pos, hit_normal = result
            dist = frac * max_range
            # Ignore self-hits
            if obj_id == self._robot_id:
                dist = max_range
                hit_mask.append(False)
            elif obj_id == -1:
                hit_mask.append(False)
            else:
                hit_mask.append(True)
            distances.append(dist)
            hit_points.append(list(hit_pos))

        return {
            "distances": distances,
            "angles": angles.tolist(),
            "hit_points": hit_points,
            "hit_mask": hit_mask,
            "num_rays": num_rays,
            "max_range": max_range,
            "origin": origin.tolist(),
            "yaw": yaw,
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Disconnect from the physics server."""
        try:
            p.disconnect(physicsClientId=self._cid)
        except Exception:
            pass
