"""Short-lived peer engrams and durable retroactive importance."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from datetime import datetime, timedelta, timezone

from willow.events import Event
from willow.store import EventStore


ENGRAM_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "and",
    "are",
    "because",
    "been",
    "before",
    "being",
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
    "what",
    "when",
    "where",
    "which",
    "will",
    "with",
    "would",
    "your",
}


def _terms(text: str) -> set[str]:
    return {
        word.lower()
        for word in re.findall(r"[\w'-]{4,}", text, flags=re.UNICODE)
        if word.lower() not in ENGRAM_STOPWORDS
    }


def _headline(text: str, limit: int = 240) -> str:
    for line in text.splitlines():
        compact = re.sub(r"^#+\s*", "", line.strip())
        if compact:
            return compact[:limit]
    return ""


def append_turn_engram(
    store: EventStore,
    *,
    session_id: str,
    user_event: Event | None,
    assistant_event: Event | None,
    actor: str = "willow",
    ttl_seconds: int = 1800,
) -> tuple[Event | None, bool]:
    """Append one fading peer signal for a substantive completed turn."""

    sources = tuple(
        event.id
        for event in (user_event, assistant_event)
        if event is not None
    )
    if not sources:
        return None, False
    user_text = user_event.content[:600] if user_event else ""
    assistant_text = _headline(assistant_event.content) if assistant_event else ""
    tools = (
        list(assistant_event.metadata.get("tools", []))
        if assistant_event
        else []
    )
    parts = []
    if user_text:
        parts.append(f"Human: {user_text}")
    if assistant_text:
        parts.append(f"Agent: {assistant_text}")
    if tools:
        parts.append("Tools: " + ", ".join(str(tool) for tool in tools[:12]))
    if not parts:
        return None, False

    now = datetime.now(timezone.utc)
    expires = now + timedelta(seconds=max(1, ttl_seconds))
    key = "turn-engram:" + ":".join(sources)
    return store.append_idempotent(
        "\n".join(parts),
        idempotency_key=key,
        actor=actor,
        kind="engram",
        session_id=session_id,
        metadata={
            "engram_type": "peer-turn",
            "expires_at": expires.isoformat(),
            "ttl_seconds": max(1, ttl_seconds),
        },
        derived_from=sources,
    )


def peer_engrams(
    store: EventStore,
    *,
    exclude_session_id: str | None = None,
    window_seconds: int = 1800,
    limit: int = 12,
    per_session_cap: int = 3,
) -> list[Event]:
    """Return diverse, unexpired peer-turn engrams, newest first."""

    since = (
        datetime.now(timezone.utc) - timedelta(seconds=max(0, window_seconds))
    ).isoformat()
    raw = store.events(
        limit=max(limit * 8, 64),
        exclude_session_id=exclude_session_id,
        kind="engram",
        active_only=True,
        include_expired=False,
        since=since,
    )
    selected: list[Event] = []
    counts: Counter[str] = Counter()
    for event in raw:
        if event.metadata.get("engram_type") != "peer-turn":
            continue
        if counts[event.session_id] >= max(1, per_session_cap):
            continue
        counts[event.session_id] += 1
        selected.append(event)
        if len(selected) >= limit:
            break
    return selected


def crystallize_retroactive_engrams(
    store: EventStore,
    *,
    limit: int = 3,
    minimum_later_reflections: int = 1,
) -> list[Event]:
    """Append durable engrams for experience that later reflections revisit.

    Importance is represented as a new derived event. When additional later
    evidence appears, the newer engram supersedes the older interpretation;
    neither is edited or erased.
    """

    active = store.events(limit=10_000, active_only=True, ascending=True)
    reflections = [
        event
        for event in active
        if event.kind in {"meditation", "summation", "dream"}
    ]
    candidates = [
        event
        for event in active
        if event.kind not in {
            "engram",
            "meditation",
            "summation",
            "dream",
        }
    ]
    existing = [
        event
        for event in store.events(
            limit=10_000,
            kind="engram",
            active_only=False,
            include_expired=True,
        )
        if event.metadata.get("engram_type") == "retroactive"
    ]
    latest_by_origin: dict[str, Event] = {}
    for event in existing:
        origin = str(event.metadata.get("origin_event") or "")
        if origin and origin not in latest_by_origin:
            latest_by_origin[origin] = event

    scored: list[tuple[float, Event, list[Event], list[str]]] = []
    for source in candidates:
        source_terms = _terms(source.content)
        later: list[Event] = []
        shared: Counter[str] = Counter()
        for reflection in reflections:
            if reflection.seq <= source.seq:
                continue
            overlap = source_terms & _terms(reflection.content)
            explicitly_derived = source.id in reflection.derived_from
            if explicitly_derived or len(overlap) >= 2:
                later.append(reflection)
                shared.update(overlap)
        if len(later) < minimum_later_reflections:
            continue
        score = len(later) + min(2.0, len(shared) * 0.15)
        scored.append(
            (
                score,
                source,
                later,
                [term for term, _ in shared.most_common(6)],
            )
        )

    scored.sort(key=lambda item: (item[0], item[1].seq), reverse=True)
    created: list[Event] = []
    for score, source, later, shared_terms in scored[: max(0, limit)]:
        evidence_ids = tuple(event.id for event in later)
        previous = latest_by_origin.get(source.id)
        previous_evidence = (
            tuple(previous.metadata.get("later_reflections", []))
            if previous
            else ()
        )
        if previous and previous_evidence == evidence_ids:
            continue

        snippet = " ".join(source.content.split())[:280]
        signals = ", ".join(shared_terms) or "explicit provenance"
        content = (
            f"Retroactive engram: {snippet}\n"
            f"Later experience returned to it {len(later)} time(s). "
            f"Shared signals: {signals}. "
            "Its importance was discovered by subsequent use, not predicted "
            "when it was recorded."
        )
        derivation = [source.id, *evidence_ids]
        if previous:
            derivation.append(previous.id)
        key_material = ":".join([source.id, *evidence_ids])
        event, was_created = store.append_idempotent(
            content,
            idempotency_key=(
                "retroactive-engram:"
                + hashlib.sha256(key_material.encode("utf-8")).hexdigest()
            ),
            actor="willow",
            kind="engram",
            session_id="reflection",
            metadata={
                "engram_type": "retroactive",
                "origin_event": source.id,
                "later_reflections": list(evidence_ids),
                "surprise_weight": len(later),
                "score": round(score, 3),
                "shared_terms": shared_terms,
            },
            supersedes=previous.id if previous else None,
            derived_from=derivation,
        )
        if was_created:
            created.append(event)
            latest_by_origin[source.id] = event
    return created
