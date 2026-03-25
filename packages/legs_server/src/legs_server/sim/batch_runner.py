"""Batch rollout runner — runs multiple simulation episodes and collects results.

Usage:
    python -m legs_server.sim.batch_runner --episodes 10 --steps 500 --output-dir /data/rollouts
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from legs_common.ids import new_episode_id, new_session_id, SeqCounter
from legs_common.protocol.canon import Action, Observation
from legs_common.time import mono_ns, wall_ns
from legs_server.model.nwm_infer import StubNWMPolicy
from legs_server.model.nwm_interface import NWMPolicy

logger = logging.getLogger(__name__)


@dataclass
class RolloutResult:
    """Summary of a single rollout episode."""

    episode_id: str
    steps: int
    duration_s: float
    avg_infer_ns: float
    model_id: str
    final_positions: List[float] = field(default_factory=list)


@dataclass
class BatchResult:
    """Summary of an entire batch run."""

    session_id: str
    total_episodes: int
    total_steps: int
    total_duration_s: float
    episodes: List[RolloutResult] = field(default_factory=list)


def run_single_rollout(
    policy: NWMPolicy,
    steps: int,
    session_id: str,
    source: str = "sim",
) -> RolloutResult:
    """Execute one rollout episode: generate observations, infer actions."""
    episode_id = new_episode_id()
    seq = SeqCounter()
    infer_times: List[int] = []
    positions: List[float] = [0.0] * 12

    t_start = time.monotonic()

    for _ in range(steps):
        obs = Observation(
            session_id=session_id,
            episode_id=episode_id,
            seq=seq.next(),
            t_wall_ns=wall_ns(),
            t_mono_ns=mono_ns(),
            source=source,
            robot_state={"positions": list(positions)},
            sensors={},
        )
        action = policy.infer(obs)
        infer_times.append(action.t_infer_ns)

        # Simple position integration from action
        if "joint_targets" in action.payload:
            targets = action.payload["joint_targets"]
            for i in range(min(len(positions), len(targets))):
                positions[i] += (targets[i] - positions[i]) * 0.1

    duration_s = time.monotonic() - t_start
    avg_infer = sum(infer_times) / len(infer_times) if infer_times else 0.0

    return RolloutResult(
        episode_id=episode_id,
        steps=steps,
        duration_s=duration_s,
        avg_infer_ns=avg_infer,
        model_id=policy.model_id(),
        final_positions=list(positions),
    )


def run_batch(
    policy: NWMPolicy,
    num_episodes: int,
    steps_per_episode: int,
    output_dir: Optional[str] = None,
) -> BatchResult:
    """Run a batch of rollout episodes sequentially."""
    session_id = new_session_id()
    logger.info(
        "Starting batch rollout [session=%s, episodes=%d, steps=%d]",
        session_id, num_episodes, steps_per_episode,
    )

    t_start = time.monotonic()
    episodes: List[RolloutResult] = []

    for i in range(num_episodes):
        result = run_single_rollout(
            policy=policy,
            steps=steps_per_episode,
            session_id=session_id,
        )
        episodes.append(result)
        logger.info(
            "Episode %d/%d done [ep=%s, steps=%d, dur=%.3fs, avg_infer=%dns]",
            i + 1, num_episodes, result.episode_id, result.steps,
            result.duration_s, int(result.avg_infer_ns),
        )

    total_duration = time.monotonic() - t_start
    batch = BatchResult(
        session_id=session_id,
        total_episodes=num_episodes,
        total_steps=sum(e.steps for e in episodes),
        total_duration_s=total_duration,
        episodes=episodes,
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"batch_{session_id}.json")
        with open(out_path, "w") as f:
            json.dump(asdict(batch), f, indent=2)
        logger.info("Batch results written to %s", out_path)

    logger.info(
        "Batch complete [episodes=%d, total_steps=%d, duration=%.2fs]",
        batch.total_episodes, batch.total_steps, batch.total_duration_s,
    )
    return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LEGS batch rollout runner")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--steps", type=int, default=200, help="Steps per episode")
    parser.add_argument("--output-dir", default=None, help="Directory to save results JSON")
    parser.add_argument("--model-tag", default="stub-v0.1.0", help="Model tag for the stub policy")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    args = parse_args()
    policy = StubNWMPolicy(model_tag=args.model_tag)
    result = run_batch(
        policy=policy,
        num_episodes=args.episodes,
        steps_per_episode=args.steps,
        output_dir=args.output_dir,
    )
    print(f"Batch done: {result.total_episodes} episodes, {result.total_steps} steps")


if __name__ == "__main__":
    main()
