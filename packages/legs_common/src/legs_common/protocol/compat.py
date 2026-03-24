"""Version compatibility helpers for protocol evolution."""

from __future__ import annotations

PROTOCOL_VERSION = "0.1.0"


def check_compat(remote_version: str) -> bool:
    """Check if remote protocol version is compatible with local.

    Rule: same major.minor is compatible; patch differences are OK.
    """
    local_parts = PROTOCOL_VERSION.split(".")
    remote_parts = remote_version.split(".")
    if len(remote_parts) < 2:
        return False
    return local_parts[0] == remote_parts[0] and local_parts[1] == remote_parts[1]
