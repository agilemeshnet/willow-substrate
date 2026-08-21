"""Deterministic reflection primitives and the future model-adapter boundary."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Iterable

from willow_substrate.connections import normalize_shapes
from willow_substrate.events import Event
from willow_substrate.store import EventStore


REFLECTION_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "and",
    "are",
    "been",
    "but",
    "for",
    "from",
    "have",
    "into",
    "not",
    "that",
    "the",
    "their",
    "there",
    "they",
    "this",
    "was",
    "were",
    "what",
    "when",
    "with",
    "would",
    "you",
    "your",
}


def _extractive_meditation(session_id: str, events: list[Event]) -> str:
    words: Counter[str] = Counter()
    actors: Counter[str] = Counter()
    kinds: Counter[str] = Counter()
    for event in events:
        actors[event.actor] += 1
        kinds[event.kind] += 1
        for word in re.findall(r"[\w'-]{4,}", event.content.lower()):
            if word not in REFLECTION_STOPWORDS:
                words[word] += 1

    topics = ", ".join(word for word, _ in words.most_common(8)) or "unclear"
    actor_text = ", ".join(f"{name} ({count})" for name, count in actors.items())
    kind_text = ", ".join(f"{name} ({count})" for name, count in kinds.items())
    opening = " ".join(events[0].content.split())[:220]
    closing = " ".join(events[-1].content.split())[:220]

    return (
        f"Session {session_id} contained {len(events)} active events. "
        f"Voices: {actor_text}. Event forms: {kind_text}. "
        f"Recurring terms: {topics}. "
        f"It opened with: {opening}. "
        f"It most recently moved toward: {closing}."
    )


def meditate(
    store: EventStore,
    session_id: str,
    *,
    text: str | None = None,
    actor: str = "willow",
    shapes: Iterable[str] = (),
    meditator=None,
) -> Event:
    """Append a meditation derived from active events in one session.

    Three ways to produce the meditation text, in priority order:

    1. ``text=str`` (supplied): use the caller's exact string; generator
       tag ``"supplied"``.
    2. ``meditator=Meditator`` (LLM-backed abstractive): call
       ``meditator.draft(session_id, events)`` and use the result;
       generator tag names the meditator class (e.g.,
       ``"AnthropicMeditator"``). This is the plug-in point Willow's
       reflection layer has always advertised as the future model-
       adapter boundary. See ``willow_substrate.llm``.
    3. Neither: fall back to the deterministic extractive summariser;
       generator tag ``"extractive-v1"``. Ships zero-dep, always works.

    The chain of provenance (``derived_from``) is identical across all
    three paths; only the ``content`` (and the ``generator`` tag) differs.
    Whichever path produced the text, the meditation lands as an
    immutable event with links back to its source turns.
    """

    events = store.events(
        limit=500,
        session_id=session_id,
        active_only=True,
        ascending=True,
    )
    if not events:
        raise ValueError(f"session has no active events: {session_id}")

    if text and text.strip():
        content = text.strip()
        generator = "supplied"
    elif meditator is not None:
        content = meditator.draft(session_id, events).strip()
        if not content:
            raise ValueError(
                f"meditator {type(meditator).__name__} returned empty content"
            )
        generator = type(meditator).__name__
    else:
        content = _extractive_meditation(session_id, events)
        generator = "extractive-v1"

    return store.append(
        content,
        actor=actor,
        kind="meditation",
        session_id=session_id,
        metadata={
            "generator": generator,
            "source_event_count": len(events),
            "idea_shape": list(normalize_shapes(shapes)),
        },
        derived_from=(event.id for event in events),
    )


def summarize_session(
    store: EventStore,
    session_id: str,
    *,
    actor: str = "willow",
) -> tuple[Event, bool]:
    """Append or advance the deterministic summation for one session."""

    events = [
        event
        for event in store.events(
            limit=5000,
            session_id=session_id,
            active_only=True,
            ascending=True,
        )
        if event.kind not in {"summation", "meditation", "dream", "engram"}
    ]
    if not events:
        raise ValueError(f"session has no active conversational events: {session_id}")

    actors: Counter[str] = Counter(event.actor for event in events)
    kinds: Counter[str] = Counter(event.kind for event in events)
    topics: Counter[str] = Counter()
    for event in events:
        for word in re.findall(r"[\w'-]{4,}", event.content.lower()):
            if word not in REFLECTION_STOPWORDS:
                topics[word] += 1

    corrections = sum(event.kind == "correction" for event in events)
    opening = " ".join(events[0].content.split())[:260]
    closing = " ".join(events[-1].content.split())[:260]
    content = (
        f"Session summation for {session_id}: {len(events)} active events "
        f"across {len(actors)} voice(s). "
        f"Event forms: {', '.join(f'{name} ({count})' for name, count in kinds.items())}. "
        f"Corrections preserved: {corrections}. "
        f"Recurring signals: {', '.join(word for word, _ in topics.most_common(8)) or 'unclear'}. "
        f"It opened with: {opening}. "
        f"It most recently moved toward: {closing}."
    )

    previous_rows = [
        event
        for event in store.events(
            limit=100,
            session_id=session_id,
            kind="summation",
            active_only=True,
        )
        if event.metadata.get("summary_type") == "session"
    ]
    previous = previous_rows[0] if previous_rows else None
    source_ids = tuple(event.id for event in events)
    if previous and tuple(previous.metadata.get("source_events", [])) == source_ids:
        return previous, False

    derived = list(source_ids)
    if previous:
        derived.append(previous.id)
    source_hash = hashlib.sha256(":".join(source_ids).encode("utf-8")).hexdigest()
    return store.append_idempotent(
        content,
        idempotency_key=f"session-summation:{session_id}:{source_hash}",
        actor=actor,
        kind="summation",
        session_id=session_id,
        metadata={
            "generator": "extractive-session-v1",
            "summary_type": "session",
            "source_event_count": len(events),
            "source_events": list(source_ids),
        },
        supersedes=previous.id if previous else None,
        derived_from=derived,
    )


def meditate_session(
    store: EventStore,
    session_id: str,
    *,
    actor: str = "willow",
) -> tuple[Event, bool]:
    """Append or advance an automatic, provenance-complete meditation."""

    events = [
        event
        for event in store.events(
            limit=5000,
            session_id=session_id,
            active_only=True,
            ascending=True,
        )
        if event.kind not in {"meditation", "dream", "engram"}
    ]
    if not events:
        raise ValueError(f"session has no active events: {session_id}")

    content = _extractive_meditation(session_id, events)
    previous_rows = [
        event
        for event in store.events(
            limit=100,
            session_id=session_id,
            kind="meditation",
            active_only=True,
        )
        if event.metadata.get("generator") == "extractive-session-v1"
    ]
    previous = previous_rows[0] if previous_rows else None
    source_ids = tuple(event.id for event in events)
    if previous and tuple(previous.metadata.get("source_events", [])) == source_ids:
        return previous, False

    derived = list(source_ids)
    if previous:
        derived.append(previous.id)
    source_hash = hashlib.sha256(":".join(source_ids).encode("utf-8")).hexdigest()
    return store.append_idempotent(
        content,
        idempotency_key=f"session-meditation:{session_id}:{source_hash}",
        actor=actor,
        kind="meditation",
        session_id=session_id,
        metadata={
            "generator": "extractive-session-v1",
            "source_event_count": len(events),
            "source_events": list(source_ids),
        },
        supersedes=previous.id if previous else None,
        derived_from=derived,
    )
