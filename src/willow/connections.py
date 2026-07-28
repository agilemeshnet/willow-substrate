"""Stereo connection-finding across words and explicit idea-shapes.

The dependency-free reference backend intentionally keeps two channels
separate:

* lexical overlap asks whether two records use related words;
* idea-shape overlap asks whether they carry the same explicit structural
  dimensions, even when their vocabulary differs.

Vista/Wave can later supply a third structural-graph score behind this
contract.  It must not be relabelled as ordinary vector similarity.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable

from willow.events import Event
from willow.store import EventStore


CONNECTION_KINDS = {
    "claim",
    "engram",
    "meditation",
    "research_result",
    "summation",
}

STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "and",
    "are",
    "been",
    "before",
    "but",
    "can",
    "could",
    "does",
    "from",
    "have",
    "into",
    "just",
    "more",
    "not",
    "only",
    "other",
    "should",
    "that",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "user",
    "what",
    "when",
    "where",
    "which",
    "will",
    "willow",
    "with",
    "would",
    "your",
}


def normalize_shapes(value: Any) -> tuple[str, ...]:
    """Normalize YAML-like shape values into ``dimension:value`` tags."""

    tags: list[str] = []
    if value is None:
        return ()
    if isinstance(value, str):
        values: Iterable[Any] = [value]
    elif isinstance(value, dict):
        values = [{key: item} for key, item in value.items()]
    elif isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = [value]

    for item in values:
        if isinstance(item, dict):
            for key, nested in item.items():
                tag = f"{key}:{nested}"
                compact = _normalize_tag(tag)
                if compact:
                    tags.append(compact)
        else:
            compact = _normalize_tag(str(item))
            if compact:
                tags.append(compact)
    return tuple(dict.fromkeys(tags))


def _normalize_tag(value: str) -> str:
    lowered = value.strip().lower()
    lowered = re.sub(r"\s+", "-", lowered)
    return re.sub(r"[^a-z0-9_:+.-]", "", lowered)


def event_shapes(event: Event) -> tuple[str, ...]:
    return normalize_shapes(
        event.metadata.get("idea_shape")
        or event.metadata.get("shapes")
        or ()
    )


def _terms(text: str) -> set[str]:
    return {
        word.lower()
        for word in re.findall(r"[\w'-]{4,}", text, flags=re.UNICODE)
        if word.lower() not in STOPWORDS
    }


def _dimensions(shapes: Iterable[str]) -> set[str]:
    return {
        tag.partition(":")[0]
        for tag in shapes
        if tag.partition(":")[0]
    }


def _cosine_set_overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / math.sqrt(len(left) * len(right))


def _shape_overlap(
    left: set[str],
    right: set[str],
) -> tuple[float, list[str], list[str]]:
    if not left or not right:
        return 0.0, [], []
    shared_tags = sorted(left & right)
    left_dimensions = _dimensions(left)
    right_dimensions = _dimensions(right)
    shared_dimensions = sorted(left_dimensions & right_dimensions)
    exact = len(shared_tags) / len(left | right)
    dimensional = (
        len(shared_dimensions) / len(left_dimensions | right_dimensions)
        if left_dimensions or right_dimensions
        else 0.0
    )
    # Exact shape identity carries most of the score. Sharing a dimension
    # with a different value is a weaker structural rhyme, not agreement.
    return 0.75 * exact + 0.25 * dimensional, shared_tags, shared_dimensions


@dataclass(frozen=True)
class ConnectionCandidate:
    """One evidence-bearing cross-record connection."""

    event: Event
    score: float
    lexical_score: float
    shape_score: float
    shared_terms: tuple[str, ...]
    shared_shapes: tuple[str, ...]
    shared_dimensions: tuple[str, ...]
    channels: tuple[str, ...]


def find_connections(
    store: EventStore,
    *,
    seed_event_id: str | None = None,
    query: str = "",
    shapes: Iterable[str] = (),
    limit: int = 10,
    kinds: Iterable[str] = CONNECTION_KINDS,
) -> list[ConnectionCandidate]:
    """Find active records related by words, shape, or both."""

    seed: Event | None = None
    if seed_event_id:
        seed = store.get(seed_event_id)
        if seed is None:
            raise KeyError(f"seed event does not exist: {seed_event_id}")

    seed_text = " ".join(
        part for part in (query.strip(), seed.content if seed else "") if part
    )
    seed_terms = _terms(seed_text)
    seed_shapes = set(normalize_shapes(shapes))
    if seed:
        seed_shapes.update(event_shapes(seed))
    if not seed_terms and not seed_shapes:
        raise ValueError("connection search needs a query, shape, or seed event")

    accepted_kinds = set(kinds)
    events = [
        event
        for event in store.events(
            limit=max(100, store.count() + 1),
            active_only=True,
            include_expired=False,
            ascending=True,
        )
        if event.kind in accepted_kinds
        and (seed is None or event.id != seed.id)
    ]

    candidates: list[ConnectionCandidate] = []
    for event in events:
        candidate_terms = _terms(event.content)
        shared_terms = sorted(seed_terms & candidate_terms)
        lexical_score = _cosine_set_overlap(seed_terms, candidate_terms)

        candidate_shapes = set(event_shapes(event))
        (
            shape_score,
            shared_shapes,
            shared_dimensions,
        ) = _shape_overlap(seed_shapes, candidate_shapes)
        if lexical_score == 0 and shape_score == 0:
            continue

        channels = []
        if lexical_score:
            channels.append("words")
        if shape_score:
            channels.append("shape")
        stereo_bonus = 0.1 if len(channels) == 2 else 0.0
        cross_form_bonus = (
            0.05
            if seed is not None and seed.kind != event.kind
            else 0.0
        )
        score = min(
            1.0,
            0.5 * lexical_score
            + 0.5 * shape_score
            + stereo_bonus
            + cross_form_bonus,
        )
        candidates.append(
            ConnectionCandidate(
                event=event,
                score=score,
                lexical_score=lexical_score,
                shape_score=shape_score,
                shared_terms=tuple(shared_terms[:12]),
                shared_shapes=tuple(shared_shapes[:12]),
                shared_dimensions=tuple(shared_dimensions[:12]),
                channels=tuple(channels),
            )
        )

    candidates.sort(
        key=lambda candidate: (candidate.score, candidate.event.seq),
        reverse=True,
    )
    return candidates[: max(0, limit)]
