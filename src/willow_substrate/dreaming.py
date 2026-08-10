"""Additive associative discovery across preserved experience."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

from willow_substrate.events import Event
from willow_substrate.store import EventStore


DREAM_STOPWORDS = {
    "about",
    "after",
    "again",
    "agent",
    "also",
    "and",
    "are",
    "been",
    "before",
    "but",
    "can",
    "context",
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
    "said",
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


def _terms(event: Event) -> set[str]:
    metadata_terms = " ".join(
        str(value)
        for key, value in event.metadata.items()
        if key in {"topic", "topics", "project", "title", "name"}
    )
    return {
        word.lower()
        for word in re.findall(
            r"[\w'-]{4,}",
            f"{event.content} {metadata_terms}",
            flags=re.UNICODE,
        )
        if word.lower() not in DREAM_STOPWORDS
    }


def _snippet(event: Event, limit: int = 220) -> str:
    compact = " ".join(event.content.split())
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."


def dream(
    store: EventStore,
    *,
    query: str = "",
    limit: int = 3,
    minimum_shared_terms: int = 2,
    actor: str = "willow",
) -> list[Event]:
    """Find and append cross-session connections as provenanced dream events.

    Only active evidence is considered, so superseded or expired material
    cannot seed a dream. Repeating the same discovery is idempotent.
    """

    evidence = [
        event
        for event in store.events(
            limit=10_000,
            active_only=True,
            include_expired=False,
            ascending=True,
        )
        if event.kind not in {"dream", "engram"}
    ]
    if len(evidence) < 2 or limit <= 0:
        return []

    terms_by_id = {event.id: _terms(event) for event in evidence}
    document_frequency: Counter[str] = Counter()
    for terms in terms_by_id.values():
        document_frequency.update(terms)

    if query.strip():
        seed_ids = {
            hit.event.id
            for hit in store.search(query, limit=24, active_only=True)
        }
        seeds = [event for event in evidence if event.id in seed_ids]
    else:
        seeds = evidence[-16:]
    if not seeds:
        return []

    existing_pairs = {
        str(event.metadata.get("pair_key"))
        for event in store.events(
            limit=10_000,
            kind="dream",
            active_only=False,
            include_expired=True,
        )
        if event.metadata.get("pair_key")
    }

    scored: list[tuple[float, Event, Event, list[str], str]] = []
    total_documents = len(evidence)
    for seed in seeds:
        for other in evidence:
            if other.seq >= seed.seq or other.session_id == seed.session_id:
                continue
            pair_ids = sorted((seed.id, other.id))
            pair_key = hashlib.sha256(":".join(pair_ids).encode("utf-8")).hexdigest()
            if pair_key in existing_pairs:
                continue
            shared = terms_by_id[seed.id] & terms_by_id[other.id]
            if len(shared) < minimum_shared_terms:
                continue
            weighted = sum(
                math.log((total_documents + 1) / (document_frequency[term] + 1))
                + 1.0
                for term in shared
            )
            distance_bonus = min(1.5, math.log1p(seed.seq - other.seq) / 5)
            cross_kind_bonus = 0.25 if seed.kind != other.kind else 0.0
            score = weighted + distance_bonus + cross_kind_bonus
            ranked_terms = sorted(
                shared,
                key=lambda term: (
                    document_frequency[term],
                    term,
                ),
            )[:8]
            scored.append((score, other, seed, ranked_terms, pair_key))

    scored.sort(key=lambda item: (item[0], item[2].seq), reverse=True)
    created: list[Event] = []
    used_events: Counter[str] = Counter()
    for score, earlier, later, shared, pair_key in scored:
        # Diversity guard: one recent event cannot consume the whole dream.
        if used_events[earlier.id] >= 1 or used_events[later.id] >= 1:
            continue
        content = (
            "Dream connection across sessions:\n"
            f"- Earlier [{earlier.short_id}, {earlier.session_id}]: "
            f"{_snippet(earlier)}\n"
            f"- Later [{later.short_id}, {later.session_id}]: "
            f"{_snippet(later)}\n"
            f"- Shared signals: {', '.join(shared)}\n"
            "This is an associative proposal, not a claim of causation."
        )
        event, was_created = store.append_idempotent(
            content,
            idempotency_key=f"dream-pair:{pair_key}",
            actor=actor,
            kind="dream",
            session_id="dreams",
            metadata={
                "pair_key": pair_key,
                "association_score": round(score, 4),
                "shared_terms": shared,
                "query": query,
                "epistemic_status": "associative-proposal",
            },
            derived_from=(earlier.id, later.id),
        )
        if was_created:
            created.append(event)
            existing_pairs.add(pair_key)
            used_events[earlier.id] += 1
            used_events[later.id] += 1
        if len(created) >= limit:
            break
    return created
