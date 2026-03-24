"""ID generation utilities for sessions, episodes, and sequences."""

import uuid
import threading


def new_session_id() -> str:
    return f"ses-{uuid.uuid4().hex[:12]}"


def new_episode_id() -> str:
    return f"ep-{uuid.uuid4().hex[:12]}"


class SeqCounter:
    """Thread-safe monotonically increasing sequence counter."""

    def __init__(self, start: int = 0) -> None:
        self._val = start
        self._lock = threading.Lock()

    def next(self) -> int:
        with self._lock:
            self._val += 1
            return self._val

    @property
    def current(self) -> int:
        return self._val
