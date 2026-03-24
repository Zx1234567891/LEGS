"""Canonical protocol schema — the single source of truth for all message types.

These dataclasses define the *logical* message contract between Dog and Server.
Serialization (msgpack/protobuf/json) is handled separately in legs_common.serialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Observation:
    """Sensor + state snapshot sent from Dog -> Server."""

    session_id: str
    episode_id: str
    seq: int
    t_wall_ns: int
    t_mono_ns: int
    source: str  # "sim" | "real"
    frame_id: str = "base_link"
    robot_state: Dict[str, Any] = field(default_factory=dict)
    sensors: Dict[str, Any] = field(default_factory=dict)
    debug: Optional[Dict[str, Any]] = None


@dataclass
class Action:
    """Policy output sent from Server -> Dog, bound to a specific Observation."""

    seq_ref: int  # must reference an Observation.seq
    action_type: str  # "twist" | "foot_targets" | "gait_params" | ...
    payload: Dict[str, Any] = field(default_factory=dict)
    model_id: str = ""
    t_infer_ns: int = 0


@dataclass
class Heartbeat:
    """Bidirectional health/link-quality message."""

    node_id: str
    role: str  # "dog" | "server"
    t_mono_ns: int = 0
    health: str = "ok"  # "ok" | "degraded" | "error"
    last_seq_sent: int = 0
    last_seq_recv: int = 0
    rtt_ms_p50: float = 0.0
    rtt_ms_p99: float = 0.0


@dataclass
class ModelInfo:
    """Model artifact metadata for version binding and reproducibility."""

    model_id: str
    artifact_sha256: str = ""
    git_commit: str = ""
    container_image: str = ""
    created_at: str = ""


@dataclass
class EStop:
    """Emergency stop command — must be executed locally on Dog, never wait for network."""

    reason: str
    latch: bool = True
    issued_by: str = ""
    t_mono_ns: int = 0
