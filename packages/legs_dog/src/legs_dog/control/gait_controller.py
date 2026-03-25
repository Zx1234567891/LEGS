"""Trot gait controller based on Raibert heuristic.

Generates 12-DOF joint targets for a quadruped robot using a diagonal
trot pattern: FL+RR swing together, FR+RL swing together.  Swing leg
trajectories follow a half-ellipse in the sagittal plane, while stance
legs push the body forward via hip retraction.

Key parameters
--------------
- gait_freq   : step cycle frequency (Hz), default 3.5
- step_height : max foot lift during swing (m), default 0.06
- stand_height: nominal hip-to-foot length (m), default 0.35
- stride_scale: how aggressively dx/dy maps to stride length
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple


# Default standing joint angles [hip_ab, thigh, calf] × 4 legs
_STAND = [0.0, 0.65, -1.3]

# Leg order: FL, FR, RL, RR
# Diagonal trot pairs: (FL, RR) phase=0, (FR, RL) phase=π
_PHASE_OFFSETS: List[float] = [0.0, math.pi, math.pi, 0.0]

# Leg hip-abduction sign: positive = splay outward
_HIP_SIGN: List[float] = [1.0, -1.0, 1.0, -1.0]


@dataclass
class GaitConfig:
    """Tunable gait parameters."""
    gait_freq: float = 3.5       # Hz — full stride cycle frequency
    step_height: float = 0.06    # metres — peak foot lift during swing
    stand_height: float = 0.35   # metres — nominal hip-to-foot vertical
    stride_scale: float = 0.35   # maps |cmd_vel| → stride amplitude
    max_stride: float = 0.25     # clamp on stride amplitude (rad equiv)
    turn_gain: float = 0.20      # dyaw → hip abduction delta
    lateral_gain: float = 0.10   # dy → hip abduction delta


class TrotGaitController:
    """Generates joint-angle targets for a diagonal trot gait.

    Usage::

        gait = TrotGaitController()
        targets = gait.compute(dx=0.3, dy=0.0, dyaw=0.1, dt=1/240)
        env.apply_joint_targets(targets)
    """

    def __init__(self, config: GaitConfig | None = None) -> None:
        self._cfg = config or GaitConfig()
        self._phase: float = 0.0   # normalised phase [0, 2π)
        self._stand = _STAND * 4   # 12-element standing pose

    @property
    def phase(self) -> float:
        return self._phase

    def reset(self) -> None:
        self._phase = 0.0

    def compute(
        self,
        dx: float = 0.0,
        dy: float = 0.0,
        dyaw: float = 0.0,
        dt: float = 1.0 / 240.0,
    ) -> List[float]:
        """Compute 12-DOF joint targets for one timestep.

        Parameters
        ----------
        dx : float
            Forward velocity command (robot frame, m/s-ish, 0–1 typical).
        dy : float
            Lateral velocity command (positive = left).
        dyaw : float
            Yaw rate command (rad/s-ish, positive = CCW).
        dt : float
            Physics timestep in seconds.

        Returns
        -------
        List of 12 joint angles [hip, thigh, calf] × 4 legs.
        """
        cfg = self._cfg

        # Advance phase
        self._phase += 2.0 * math.pi * cfg.gait_freq * dt
        self._phase %= 2.0 * math.pi

        # Command magnitude determines stride amplitude
        cmd_mag = min(math.sqrt(dx * dx + dy * dy) + abs(dyaw) * 0.3, 1.0)
        stride = cmd_mag * cfg.stride_scale
        stride = min(stride, cfg.max_stride)

        targets: List[float] = list(self._stand)

        for leg in range(4):
            base = leg * 3
            leg_phase = (self._phase + _PHASE_OFFSETS[leg]) % (2.0 * math.pi)

            # Normalised phase in [0, 1)
            t = leg_phase / (2.0 * math.pi)

            # Swing phase: t ∈ [0, 0.5), Stance phase: t ∈ [0.5, 1.0)
            if t < 0.5:
                # --- Swing ---
                swing_t = t / 0.5  # 0→1 during swing
                # Sagittal: half-ellipse trajectory (forward arc)
                thigh_delta = stride * math.sin(math.pi * swing_t) * 0.8
                # Vertical: lift foot via calf flexion
                lift = cfg.step_height * math.sin(math.pi * swing_t)
                calf_delta = -lift * 2.5  # calf flexion lifts the foot
            else:
                # --- Stance ---
                stance_t = (t - 0.5) / 0.5  # 0→1 during stance
                # Retract hip to push body forward
                thigh_delta = -stride * (stance_t - 0.5) * 0.6
                calf_delta = 0.0

            # Forward/backward from dx
            thigh_delta += dx * 0.12

            # Hip abduction: steering + lateral
            hip_delta = (
                dyaw * cfg.turn_gain * _HIP_SIGN[leg]
                + dy * cfg.lateral_gain * _HIP_SIGN[leg]
            )

            targets[base + 0] = self._stand[base + 0] + hip_delta
            targets[base + 1] = self._stand[base + 1] + thigh_delta
            targets[base + 2] = self._stand[base + 2] + calf_delta

        return targets

    def standing_pose(self) -> List[float]:
        """Return the default standing joint angles."""
        return list(self._stand)
