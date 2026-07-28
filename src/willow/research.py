"""A deliberately small, provider-neutral research ledger.

Doozer's useful research behaviour came from a simple shape: queue a question,
run one of several interchangeable search/synthesis lanes, retain citations,
then hand the result back to Willow.  This module preserves that shape without
embedding a paid search service, a particular LLM, voice output, or a homelab
HTTP address in the core package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from willow.connections import normalize_shapes
from willow.events import Event
from willow.store import EventStore


@dataclass(frozen=True)
class Citation:
    """One source actually used by a research result."""

    title: str
    location: str
    excerpt: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "title": self.title.strip(),
            "location": self.location.strip(),
            "excerpt": self.excerpt.strip(),
        }


@dataclass(frozen=True)
class ResearchState:
    """Current projection of one immutable research commission."""

    commission: Event
    status: str
    result: Event | None = None
    failure: Event | None = None


class ResearchLedger:
    """Queue questions and append their provider-independent outcomes."""

    def __init__(self, store: EventStore):
        self.store = store

    def commission(
        self,
        query: str,
        *,
        actor: str = "user",
        session_id: str = "research",
        lane: str = "auto",
        topics: Iterable[str] = (),
        approval_required: bool = True,
    ) -> Event:
        """Append a research request without automatically spending money."""

        if not query.strip():
            raise ValueError("research query must not be empty")
        if not lane.strip():
            raise ValueError("research lane must not be empty")
        return self.store.append(
            query,
            actor=actor,
            kind="research_commission",
            session_id=session_id,
            metadata={
                "lane": lane.strip(),
                "topics": list(dict.fromkeys(topics)),
                "approval_required": bool(approval_required),
                "status": (
                    "staged_for_review" if approval_required else "queued"
                ),
            },
        )

    def complete(
        self,
        commission_id: str,
        *,
        summary: str,
        writeup: str,
        citations: Iterable[Citation | Mapping[str, str]] = (),
        shapes: Iterable[str] = (),
        provider: str,
        actor: str = "willow",
        session_id: str = "research",
    ) -> Event:
        """Append a cited result derived from a research commission."""

        commission = self._active_commission(commission_id)
        if not summary.strip() and not writeup.strip():
            raise ValueError("research result must contain a summary or writeup")
        if not provider.strip():
            raise ValueError("research provider must not be empty")

        citation_values = []
        for citation in citations:
            if isinstance(citation, Citation):
                item = citation.as_dict()
            else:
                item = {
                    "title": str(citation.get("title") or "").strip(),
                    "location": str(
                        citation.get("location")
                        or citation.get("url")
                        or ""
                    ).strip(),
                    "excerpt": str(
                        citation.get("excerpt")
                        or citation.get("snippet")
                        or ""
                    ).strip(),
                }
            if not item["location"]:
                raise ValueError("every citation requires a location or URL")
            citation_values.append(item)

        content_parts = [part.strip() for part in (summary, writeup) if part.strip()]
        return self.store.append(
            "\n\n".join(content_parts),
            actor=actor,
            kind="research_result",
            session_id=session_id,
            metadata={
                "commission_id": commission.id,
                "provider": provider.strip(),
                "summary": summary.strip(),
                "citations": citation_values,
                "citation_count": len(citation_values),
                "idea_shape": list(normalize_shapes(shapes)),
                "status": "done",
            },
            derived_from=(commission.id,),
        )

    def fail(
        self,
        commission_id: str,
        *,
        error: str,
        provider: str,
        retryable: bool = True,
        actor: str = "willow",
        session_id: str = "research",
    ) -> Event:
        """Append an honest failed attempt without losing the commission."""

        commission = self._active_commission(commission_id)
        if not error.strip():
            raise ValueError("research failure must describe the error")
        if not provider.strip():
            raise ValueError("research provider must not be empty")
        return self.store.append(
            error,
            actor=actor,
            kind="research_failure",
            session_id=session_id,
            metadata={
                "commission_id": commission.id,
                "provider": provider.strip(),
                "retryable": bool(retryable),
                "status": "error",
            },
            derived_from=(commission.id,),
        )

    def states(self) -> list[ResearchState]:
        """Project commission status from immutable result/failure events."""

        limit = max(100, self.store.count() + 1)
        commissions = self.store.events(
            limit=limit,
            kind="research_commission",
            active_only=True,
            include_expired=True,
            ascending=True,
        )
        outcomes = [
            *self.store.events(
                limit=limit,
                kind="research_result",
                active_only=False,
                include_expired=True,
                ascending=True,
            ),
            *self.store.events(
                limit=limit,
                kind="research_failure",
                active_only=False,
                include_expired=True,
                ascending=True,
            ),
        ]
        outcomes.sort(key=lambda event: event.seq)
        by_commission: dict[str, list[Event]] = {}
        for event in outcomes:
            commission_id = str(event.metadata.get("commission_id") or "")
            if commission_id:
                by_commission.setdefault(commission_id, []).append(event)

        states: list[ResearchState] = []
        for commission in commissions:
            attempts = by_commission.get(commission.id, [])
            latest = attempts[-1] if attempts else None
            result = latest if latest and latest.kind == "research_result" else None
            failure = latest if latest and latest.kind == "research_failure" else None
            if result:
                status = "done"
            elif failure:
                status = "error"
            else:
                status = str(
                    commission.metadata.get("status") or "queued"
                )
            states.append(
                ResearchState(
                    commission=commission,
                    status=status,
                    result=result,
                    failure=failure,
                )
            )
        return states

    def _active_commission(self, commission_id: str) -> Event:
        commission = next(
            (
                state.commission
                for state in self.states()
                if state.commission.id == commission_id
            ),
            None,
        )
        if commission is None:
            raise KeyError(
                f"active research commission does not exist: {commission_id}"
            )
        return commission
