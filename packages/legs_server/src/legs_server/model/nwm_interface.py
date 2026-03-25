"""NWM Policy Protocol — the pluggable interface for inference backends.

Any model that satisfies this Protocol can be used as a drop-in replacement
for the stub implementation.  The Server's gRPC layer calls ``infer()`` on
every incoming Observation and forwards the resulting Action back to Dog.
"""

from __future__ import annotations

from typing import Protocol

from legs_common.protocol.canon import Action, Observation


class NWMPolicy(Protocol):
    """Protocol that every inference backend must implement."""

    def infer(self, obs: Observation) -> Action:
        """Given an Observation, produce a corresponding Action.

        The returned Action.seq_ref MUST equal obs.seq so that Dog can
        correlate the response with the original request.
        The returned Action.model_id MUST equal self.model_id().
        """
        ...

    def model_id(self) -> str:
        """Return the unique identifier of the currently loaded model."""
        ...
