"""Salience: which experience survives the budget.

Retrieval decides what is *eligible*.  Salience decides what is *kept* when the
token budget runs out before the candidates do.  Without it, truncation is
decided by whichever retrieval arm happened to run first, which means a standing
rule learned six months ago loses its place to an incidental message from this
morning.

The score is a small sum of named signals rather than one opaque number, so a
ranking can always be explained:

``standing``
    Material explicitly marked as standing or foundational.  A rule that was
    learned by something going wrong should not have to win a relevance contest
    against ordinary chatter to be remembered.

``citation``
    Wikilink in-degree.  Material that later material keeps pointing at has been
    reinforced by use.  This is the closest dependency-free analogue of a
    waypoint's accumulated mass.

``reflection``
    Meditations, summations, dreams, and engrams are already distilled, so they
    carry more per token than the raw experience they were derived from.

``recency``
    A half-life rather than a cliff.  Time is provenance, not geometry: recency
    is one voice in the ranking and never a coordinate.

``query``
    Lexical overlap with the current intent.

Weights are deliberately blunt and inspectable.  They are a policy, not a
discovery, and an operator is expected to tune them.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from willow_substrate.events import Event

STANDING_WEIGHT = 8.0
CITATION_WEIGHT = 0.5
CITATION_CEILING = 4.0
REFLECTION_WEIGHT = 1.0
RECENCY_WEIGHT = 3.0
RECENCY_HALF_LIFE_DAYS = 14.0
QUERY_WEIGHT = 2.0

STANDING_KEYS = ("standing", "foundational")
STANDING_VALUES = {"standing", "foundational", "constitutional"}

_WORD = re.compile(r"[\w'-]{3,}", re.UNICODE)
_WIKILINK = re.compile(r"\[\[([^\]]+)\]\]")


@dataclass(frozen=True)
class SalienceSignal:
    """One named contribution to a salience score."""

    name: str
    value: float
    detail: str


@dataclass(frozen=True)
class SalienceScore:
    """An explainable ranking of one event."""

    event_id: str
    total: float
    signals: tuple[SalienceSignal, ...]

    def explain(self) -> str:
        parts = [
            f"{signal.name}={signal.value:+.2f} ({signal.detail})"
            for signal in self.signals
            if signal.value
        ]
        return f"{self.total:.2f} = " + (" ".join(parts) or "no active signals")


def _terms(text: str) -> set[str]:
    return {match.group(0).lower() for match in _WORD.finditer(text)}


def _normalise_link(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")


def event_names(event: Event) -> set[str]:
    """Names by which other events may cite this one."""

    names: set[str] = set()
    for key in ("name", "slug", "title"):
        value = event.metadata.get(key)
        if isinstance(value, str) and value.strip():
            names.add(_normalise_link(value))
    return {name for name in names if name}


def wikilink_in_degrees(events: Iterable[Event]) -> dict[str, int]:
    """Count how many other events cite each event by ``[[name]]``.

    Self-citation does not count.  An event that mentions its own name is not
    thereby important; it is just labelled.
    """

    materialised = list(events)
    by_name: dict[str, set[str]] = {}
    for event in materialised:
        for name in event_names(event):
            by_name.setdefault(name, set()).add(event.id)

    degrees: dict[str, int] = {event.id: 0 for event in materialised}
    for event in materialised:
        cited = {
            _normalise_link(match.group(1))
            for match in _WIKILINK.finditer(event.content)
        }
        for name in cited:
            for target_id in by_name.get(name, ()):
                if target_id != event.id:
                    degrees[target_id] = degrees.get(target_id, 0) + 1
    return degrees


def is_standing(metadata: Mapping[str, Any]) -> bool:
    """Whether metadata marks material as standing or foundational.

    Several spellings are accepted because a corpus written by a human should
    not have to match one exact key to keep its own rules.
    """

    for key in STANDING_KEYS:
        value = metadata.get(key)
        if isinstance(value, bool) and value:
            return True
        if isinstance(value, str) and value.strip().lower() in {
            "true",
            "yes",
            "1",
            *STANDING_VALUES,
        }:
            return True
    for key in ("type", "kind", "class", "tier"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip().lower() in STANDING_VALUES:
            return True
    tags = metadata.get("tags")
    if isinstance(tags, (list, tuple, set)):
        for tag in tags:
            if isinstance(tag, str) and tag.strip().lower() in STANDING_VALUES:
                return True
    return False


def _age_days(timestamp: str, now: datetime) -> float | None:
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return max(0.0, (now - parsed).total_seconds() / 86400.0)


def score_event(
    event: Event,
    *,
    query: str = "",
    now: datetime | None = None,
    in_degree: int = 0,
) -> SalienceScore:
    """Score one event, keeping every contributing signal inspectable."""

    moment = now or datetime.now(timezone.utc)
    signals: list[SalienceSignal] = []

    if is_standing(event.metadata):
        signals.append(
            SalienceSignal("standing", STANDING_WEIGHT, "marked standing")
        )

    if in_degree > 0:
        value = min(CITATION_CEILING, CITATION_WEIGHT * in_degree)
        signals.append(
            SalienceSignal("citation", value, f"cited by {in_degree}")
        )

    if event.is_reflection:
        signals.append(
            SalienceSignal("reflection", REFLECTION_WEIGHT, event.kind)
        )

    age = _age_days(event.timestamp, moment)
    if age is not None:
        decay = math.pow(0.5, age / RECENCY_HALF_LIFE_DAYS)
        signals.append(
            SalienceSignal(
                "recency",
                RECENCY_WEIGHT * decay,
                f"{age:.1f}d old",
            )
        )

    query_terms = _terms(query)
    if query_terms:
        overlap = query_terms & _terms(event.content)
        if overlap:
            fraction = len(overlap) / len(query_terms)
            signals.append(
                SalienceSignal(
                    "query",
                    QUERY_WEIGHT * fraction,
                    f"{len(overlap)}/{len(query_terms)} terms",
                )
            )

    total = sum(signal.value for signal in signals)
    return SalienceScore(
        event_id=event.id,
        total=total,
        signals=tuple(signals),
    )


def score_events(
    events: Sequence[Event],
    *,
    query: str = "",
    now: datetime | None = None,
    in_degrees: Mapping[str, int] | None = None,
) -> dict[str, SalienceScore]:
    """Score a population of events, deriving citation depth from the set."""

    degrees = (
        dict(in_degrees)
        if in_degrees is not None
        else wikilink_in_degrees(events)
    )
    moment = now or datetime.now(timezone.utc)
    return {
        event.id: score_event(
            event,
            query=query,
            now=moment,
            in_degree=degrees.get(event.id, 0),
        )
        for event in events
    }


def rank_selection(
    selection: Sequence[tuple[str, Event]],
    *,
    query: str = "",
    now: datetime | None = None,
    pinned_sources: frozenset[str] = frozenset({"hot-peer"}),
) -> tuple[list[tuple[str, Event]], dict[str, SalienceScore]]:
    """Order retrieved events so the budget keeps what matters.

    Sources in ``pinned_sources`` keep their position at the head of the list.
    Peer engrams are a live coordination signal with their own short expiry, so
    they are not asked to compete with durable material on durability.

    The remainder is sorted by salience, and ties are broken by the original
    retrieval order so the ranking stays deterministic.
    """

    scores = score_events(
        [event for _, event in selection],
        query=query,
        now=now,
    )
    pinned: list[tuple[str, Event]] = []
    rest: list[tuple[int, str, Event]] = []
    for index, (source, event) in enumerate(selection):
        if source in pinned_sources:
            pinned.append((source, event))
        else:
            rest.append((index, source, event))

    rest.sort(key=lambda item: (-scores[item[2].id].total, item[0]))
    ordered = pinned + [(source, event) for _, source, event in rest]
    return ordered, scores
