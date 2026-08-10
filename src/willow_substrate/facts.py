"""Fact TTL as an append-only epistemic maintenance contract.

The original Willow system treated facts as living objects: each claim knows
when it should next be checked, and verification appends evidence rather than
silently overwriting history.  This module keeps that useful idea while
deliberately refusing a dangerous shortcut from the research prototype:
``unknown`` and verifier failures never count as confirmation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

from willow_substrate.connections import normalize_shapes
from willow_substrate.events import Event
from willow_substrate.store import EventStore


CHECK_OUTCOMES = {
    "confirmed",
    "updated",
    "contradicted",
    "unknown",
    "unreachable",
}
EVIDENCE_KINDS = {
    "primary",
    "secondary",
    "user",
    "sensor",
    "model",
    "none",
}
CONFIRMING_EVIDENCE_KINDS = EVIDENCE_KINDS - {"model", "none"}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _next_check(checked_at: datetime, ttl_days: float) -> str | None:
    if ttl_days == -1:
        return None
    return _iso(checked_at + timedelta(days=ttl_days))


@dataclass(frozen=True)
class FactState:
    """The current projection of one immutable claim and its checks."""

    claim: Event
    status: str
    next_check_at: str | None
    last_verified_at: str | None
    last_check: Event | None
    check_count: int

    def is_due(self, now: datetime | None = None) -> bool:
        if self.next_check_at is None:
            return False
        return _parse_datetime(self.next_check_at) <= (now or _utc_now())


@dataclass(frozen=True)
class FactCheckResult:
    """Events appended by a verification attempt."""

    check: Event
    replacement: Event | None = None


class FactLedger:
    """Create, project, and verify TTL-bearing factual claims."""

    def __init__(self, store: EventStore):
        self.store = store

    def add_claim(
        self,
        content: str,
        *,
        ttl_days: float = 180,
        domain: str = "research",
        source: str = "user",
        actor: str = "user",
        session_id: str = "default",
        topics: Iterable[str] = (),
        shapes: Iterable[str] = (),
        supersedes: str | None = None,
        derived_from: Iterable[str] = (),
        checked_at: datetime | None = None,
    ) -> Event:
        """Append a factual claim with its own verification cadence."""

        if ttl_days != -1 and ttl_days <= 0:
            raise ValueError("ttl_days must be positive or -1 for immutable")
        if not domain.strip():
            raise ValueError("fact domain must not be empty")
        if not source.strip():
            raise ValueError("fact source must not be empty")

        asserted_at = checked_at or _utc_now()
        sources = list(dict.fromkeys(derived_from))
        if supersedes and supersedes not in sources:
            sources.append(supersedes)
        return self.store.append(
            content,
            actor=actor,
            kind="claim",
            session_id=session_id,
            timestamp=_iso(asserted_at),
            metadata={
                "domain": domain.strip(),
                "ttl_days": ttl_days,
                "source": source.strip(),
                "topics": list(dict.fromkeys(topics)),
                "idea_shape": list(normalize_shapes(shapes)),
                "asserted_at": _iso(asserted_at),
                "last_verified_at": _iso(asserted_at),
                "next_check_at": _next_check(asserted_at, ttl_days),
                "epistemic_status": "asserted",
            },
            supersedes=supersedes,
            derived_from=sources,
        )

    def states(self, *, active_only: bool = True) -> list[FactState]:
        """Project current claim state from immutable claim/check events."""

        limit = max(100, self.store.count() + 1)
        claims = self.store.events(
            limit=limit,
            kind="claim",
            active_only=active_only,
            include_expired=True,
            ascending=True,
        )
        checks = self.store.events(
            limit=limit,
            kind="fact_check",
            active_only=False,
            include_expired=True,
            ascending=True,
        )

        checks_by_claim: dict[str, list[Event]] = {}
        for check in checks:
            claim_id = str(check.metadata.get("claim_id") or "")
            if claim_id:
                checks_by_claim.setdefault(claim_id, []).append(check)

        states: list[FactState] = []
        for claim in claims:
            claim_checks = checks_by_claim.get(claim.id, [])
            latest = claim_checks[-1] if claim_checks else None
            status = (
                str(latest.metadata.get("outcome"))
                if latest
                else str(claim.metadata.get("epistemic_status") or "asserted")
            )
            next_check_at = claim.metadata.get("next_check_at")
            last_verified_at = claim.metadata.get("last_verified_at")
            if latest:
                outcome = latest.metadata.get("outcome")
                if outcome == "confirmed":
                    next_check_at = latest.metadata.get("next_check_at")
                    last_verified_at = latest.metadata.get("verified_at")
                elif outcome in {"unknown", "unreachable"}:
                    # A failed attempt schedules another attempt but does not
                    # move the last-verified timestamp.
                    next_check_at = latest.metadata.get("retry_at")

            states.append(
                FactState(
                    claim=claim,
                    status=status,
                    next_check_at=(
                        str(next_check_at) if next_check_at is not None else None
                    ),
                    last_verified_at=(
                        str(last_verified_at)
                        if last_verified_at is not None
                        else None
                    ),
                    last_check=latest,
                    check_count=len(claim_checks),
                )
            )
        return states

    def due(
        self,
        *,
        now: datetime | None = None,
        domain: str | None = None,
        limit: int = 20,
    ) -> list[FactState]:
        """Return active claims whose own next-check time has arrived."""

        due = [
            state
            for state in self.states()
            if state.is_due(now)
            and (
                domain is None
                or str(state.claim.metadata.get("domain")) == domain
            )
        ]
        due.sort(key=lambda item: item.next_check_at or "")
        return due[: max(0, limit)]

    def record_check(
        self,
        claim_id: str,
        *,
        outcome: str,
        evidence: str = "",
        source: str = "",
        evidence_kind: str = "none",
        replacement: str | None = None,
        actor: str = "willow",
        session_id: str = "fact-ttl",
        retry_days: float = 1,
        checked_at: datetime | None = None,
    ) -> FactCheckResult:
        """Append a verification attempt and, for updates, a replacement claim.

        Confirming, updating, or contradicting a claim requires evidence that
        is not merely an LLM's parametric recollection. ``unknown`` and
        ``unreachable`` remain honest failed attempts and only set a retry
        time; they never advance ``last_verified_at``.
        """

        normalized_outcome = outcome.strip().lower()
        normalized_kind = evidence_kind.strip().lower()
        if normalized_outcome not in CHECK_OUTCOMES:
            raise ValueError(
                "outcome must be one of: " + ", ".join(sorted(CHECK_OUTCOMES))
            )
        if normalized_kind not in EVIDENCE_KINDS:
            raise ValueError(
                "evidence_kind must be one of: "
                + ", ".join(sorted(EVIDENCE_KINDS))
            )
        if retry_days <= 0:
            raise ValueError("retry_days must be positive")

        state = next(
            (item for item in self.states() if item.claim.id == claim_id),
            None,
        )
        if state is None:
            raise KeyError(f"active fact claim does not exist: {claim_id}")
        claim = state.claim

        is_assertive = normalized_outcome in {
            "confirmed",
            "updated",
            "contradicted",
        }
        if is_assertive:
            if normalized_kind not in CONFIRMING_EVIDENCE_KINDS:
                raise ValueError(
                    f"{normalized_outcome} requires primary, secondary, "
                    "user, or sensor evidence; model-only memory cannot "
                    "refresh a fact"
                )
            if not source.strip():
                raise ValueError(
                    f"{normalized_outcome} requires an identified evidence source"
                )
            if not evidence.strip():
                raise ValueError(
                    f"{normalized_outcome} requires a concise evidence note"
                )
        if normalized_outcome == "updated" and not (replacement or "").strip():
            raise ValueError("updated requires replacement claim text")

        moment = checked_at or _utc_now()
        ttl_days = float(claim.metadata.get("ttl_days", 180))
        replacement_event: Event | None = None
        if normalized_outcome == "updated":
            replacement_event = self.add_claim(
                replacement or "",
                ttl_days=ttl_days,
                domain=str(claim.metadata.get("domain") or "research"),
                source=source,
                actor=actor,
                session_id=session_id,
                topics=claim.metadata.get("topics") or (),
                shapes=claim.metadata.get("idea_shape") or (),
                supersedes=claim.id,
                derived_from=(claim.id,),
                checked_at=moment,
            )

        metadata = {
            "claim_id": claim.id,
            "outcome": normalized_outcome,
            "evidence": evidence.strip(),
            "evidence_source": source.strip(),
            "evidence_kind": normalized_kind,
            "checked_at": _iso(moment),
        }
        if normalized_outcome == "confirmed":
            metadata.update(
                {
                    "verified_at": _iso(moment),
                    "next_check_at": _next_check(moment, ttl_days),
                }
            )
        elif normalized_outcome in {"unknown", "unreachable"}:
            metadata["retry_at"] = _iso(moment + timedelta(days=retry_days))
        elif normalized_outcome == "updated" and replacement_event:
            metadata.update(
                {
                    "verified_at": _iso(moment),
                    "replacement_claim_id": replacement_event.id,
                }
            )
        elif normalized_outcome == "contradicted":
            metadata["needs_human_review"] = True

        derived_from = [claim.id]
        if replacement_event:
            derived_from.append(replacement_event.id)
        check = self.store.append(
            (
                f"{normalized_outcome.upper()}: {evidence.strip()}"
                if evidence.strip()
                else normalized_outcome.upper()
            ),
            actor=actor,
            kind="fact_check",
            session_id=session_id,
            metadata=metadata,
            # A contradiction filters the unsupported claim from ordinary
            # retrieval while retaining both events in the ledger.
            supersedes=(
                claim.id if normalized_outcome == "contradicted" else None
            ),
            derived_from=derived_from,
            timestamp=_iso(moment),
        )
        return FactCheckResult(check=check, replacement=replacement_event)
