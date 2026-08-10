"""Immutable event shapes used throughout Willow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Event:
    """One immutable observation, action, correction, or reflection."""

    seq: int
    id: str
    timestamp: str
    session_id: str
    actor: str
    kind: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    supersedes: str | None = None
    derived_from: tuple[str, ...] = ()
    prev_hash: str = ""
    hash: str = ""

    @property
    def short_id(self) -> str:
        return self.id[:16]

    @property
    def is_reflection(self) -> bool:
        return self.kind in {"meditation", "summation", "dream", "engram"}


@dataclass(frozen=True)
class EventHit:
    """An event returned by a retrieval path."""

    event: Event
    score: float
    source: str
