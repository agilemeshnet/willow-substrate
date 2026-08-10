"""A dependency-free reference foveation policy over Willow events.

This is the local bootstrap implementation. It narrows from sessions, to
matching events, to their immediate experiential neighbourhood. The richer MRL
hypergraph implementation in agilemeshnet/foveation can implement the same
service boundary later.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from willow_substrate.events import Event, EventHit
from willow_substrate.store import EventStore


STOPWORDS = {
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
    "did",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "into",
    "its",
    "just",
    "not",
    "our",
    "should",
    "that",
    "the",
    "their",
    "then",
    "there",
    "they",
    "this",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "will",
    "with",
    "would",
    "you",
    "your",
}


def _terms(text: str) -> set[str]:
    words = {
        word.lower()
        for word in re.findall(r"[\w'-]{3,}", text, flags=re.UNICODE)
    }
    return words - STOPWORDS


def _event_terms(event: Event) -> set[str]:
    metadata_text = " ".join(
        str(value)
        for key, value in event.metadata.items()
        if key in {"topic", "topics", "project", "title", "name"}
    )
    return _terms(f"{event.content} {metadata_text} {event.kind} {event.actor}")


@dataclass(frozen=True)
class FoveationPass:
    name: str
    candidates_examined: int
    selected: tuple[str, ...]
    rationale: str


@dataclass(frozen=True)
class FoveationResult:
    query: str
    hits: tuple[EventHit, ...]
    trace: tuple[FoveationPass, ...]
    mode: str = "voluntary"

    @property
    def event_ids(self) -> tuple[str, ...]:
        return tuple(hit.event.id for hit in self.hits)

    def to_markdown(self) -> str:
        lines = [f"# Foveation: {self.query}", ""]
        for item in self.trace:
            selected = ", ".join(item.selected) if item.selected else "none"
            lines.append(
                f"- **{item.name}**: examined {item.candidates_examined}; "
                f"selected {selected}. {item.rationale}"
            )
        lines.extend(["", "## Evidence"])
        if not self.hits:
            lines.append("- No active evidence found.")
        for hit in self.hits:
            event = hit.event
            snippet = " ".join(event.content.split())
            if len(snippet) > 180:
                snippet = snippet[:177] + "..."
            lines.append(
                f"- [{event.short_id}] `{event.kind}` in `{event.session_id}` "
                f"(score {hit.score:.3f}): {snippet}"
            )
        return "\n".join(lines)


class Foveator:
    """Narrow attention from sessions to events to related experience."""

    def __init__(self, store: EventStore, max_events: int = 5000):
        self.store = store
        self.max_events = max_events

    def foveate(
        self,
        query: str,
        *,
        mode: str = "voluntary",
        session_limit: int = 5,
        event_limit: int = 12,
        neighbourhood: int = 2,
    ) -> FoveationResult:
        events = self.store.events(
            limit=self.max_events,
            active_only=True,
            ascending=True,
        )
        if not events:
            return FoveationResult(
                query=query,
                hits=(),
                trace=(
                    FoveationPass(
                        name="peripheral",
                        candidates_examined=0,
                        selected=(),
                        rationale="The shared substrate is empty.",
                    ),
                ),
                mode=mode,
            )

        query_terms = _terms(query)
        max_seq = max(event.seq for event in events)
        event_scores: dict[str, float] = {}
        sessions: dict[str, list[Event]] = defaultdict(list)

        for event in events:
            sessions[event.session_id].append(event)
            overlap = len(query_terms & _event_terms(event))
            lexical = overlap / max(len(query_terms), 1)
            phrase_bonus = (
                0.35
                if query.strip()
                and query.lower() in event.content.lower()
                else 0.0
            )
            reflection_bonus = 0.08 if event.is_reflection else 0.0
            recency = 0.08 * (event.seq / max_seq)
            event_scores[event.id] = lexical + phrase_bonus + reflection_bonus + recency

        session_scores: list[tuple[str, float]] = []
        for session_id, session_events in sessions.items():
            ranked = sorted(
                (event_scores[event.id] for event in session_events),
                reverse=True,
            )
            score = (ranked[0] if ranked else 0.0) + sum(ranked[1:4]) * 0.25
            session_scores.append((session_id, score))
        session_scores.sort(key=lambda item: item[1], reverse=True)
        selected_sessions = tuple(
            session_id for session_id, _ in session_scores[:session_limit]
        )

        pass1 = FoveationPass(
            name="peripheral",
            candidates_examined=len(sessions),
            selected=selected_sessions,
            rationale="Selected the sessions with the strongest aggregate signal.",
        )

        para_candidates = [
            event for event in events if event.session_id in selected_sessions
        ]
        para_ranked = sorted(
            para_candidates,
            key=lambda event: (event_scores[event.id], event.seq),
            reverse=True,
        )
        para_selected = para_ranked[:event_limit]
        pass2 = FoveationPass(
            name="parafoveal",
            candidates_examined=len(para_candidates),
            selected=tuple(event.short_id for event in para_selected),
            rationale="Ranked active events inside the selected sessions.",
        )

        seed_ids = {event.id for event in para_selected}
        seed_seqs = {event.seq for event in para_selected}
        foveal: dict[str, Event] = {event.id: event for event in para_selected}

        for event in events:
            linked = bool(seed_ids & set(event.derived_from))
            supersession_link = event.supersedes in seed_ids
            nearby = (
                event.session_id in selected_sessions
                and any(abs(event.seq - seq) <= neighbourhood for seq in seed_seqs)
            )
            if linked or supersession_link or nearby:
                foveal[event.id] = event

        def final_score(event: Event) -> float:
            base = event_scores[event.id]
            if event.id in seed_ids:
                base += 0.4
            if seed_ids & set(event.derived_from):
                base += 0.25
            return base

        final_events = sorted(
            foveal.values(),
            key=lambda event: (final_score(event), event.seq),
            reverse=True,
        )[:event_limit]
        hits = tuple(
            EventHit(
                event=event,
                score=final_score(event),
                source="foveation",
            )
            for event in final_events
        )

        pass3 = FoveationPass(
            name="foveal",
            candidates_examined=len(foveal),
            selected=tuple(event.short_id for event in final_events),
            rationale="Expanded around evidence, derivation links, and nearby experience.",
        )

        return FoveationResult(
            query=query,
            hits=hits,
            trace=(pass1, pass2, pass3),
            mode=mode,
        )
