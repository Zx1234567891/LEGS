"""Serialization utilities — msgpack as default, JSON as fallback."""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Dict, Type, TypeVar

try:
    import msgpack
    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False

T = TypeVar("T")


def to_dict(obj: Any) -> Dict[str, Any]:
    """Convert a dataclass instance to a plain dict (shallow)."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    raise TypeError(f"{type(obj)} is not a dataclass instance")


def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
    """Construct a dataclass from a dict, ignoring unknown keys."""
    field_names = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in data.items() if k in field_names}
    return cls(**filtered)


def pack(obj: Any) -> bytes:
    """Serialize a dataclass to bytes (msgpack preferred, JSON fallback)."""
    d = to_dict(obj)
    if HAS_MSGPACK:
        return msgpack.packb(d, use_bin_type=True)
    return json.dumps(d).encode("utf-8")


def unpack(cls: Type[T], data: bytes) -> T:
    """Deserialize bytes back to a dataclass instance."""
    if HAS_MSGPACK:
        d = msgpack.unpackb(data, raw=False)
    else:
        d = json.loads(data.decode("utf-8"))
    return from_dict(cls, d)
