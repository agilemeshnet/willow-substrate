"""Deterministic temporal samples for evaluating cumulative scene formation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from willow_substrate.connections import find_connections
from willow_substrate.events import Event
from willow_substrate.store import EventStore


TEMPORAL_SAMPLE_SCHEMA = "willow.temporal-sample/v1"


@dataclass(frozen=True)
class SampleLoadReport:
    sample_id: str
    created: int
    reused: int
    events: dict[str, Event]


@dataclass(frozen=True)
class EvaluationCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class SampleEvaluation:
    sample_id: str
    checks: tuple[EvaluationCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)


def read_temporal_sample(path: str | Path) -> dict[str, Any]:
    """Read and validate the inspectable JSON sample manifest."""

    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("temporal sample must be a JSON object")
    if data.get("schema") != TEMPORAL_SAMPLE_SCHEMA:
        raise ValueError(
            f"temporal sample schema must be {TEMPORAL_SAMPLE_SCHEMA!r}"
        )
    sample_id = data.get("id")
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError("temporal sample needs a non-empty id")
    rows = data.get("events")
    if not isinstance(rows, list) or not rows:
        raise ValueError("temporal sample needs a non-empty events list")

    seen: set[str] = set()
    previous_at: datetime | None = None
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"event {index} must be an object")
        key = row.get("key")
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"event {index} needs a non-empty key")
        if key in seen:
            raise ValueError(f"duplicate temporal sample key: {key}")
        seen.add(key)

        for field in ("timestamp", "session_id", "provider", "actor", "kind", "content"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"event {key} needs a non-empty {field}")

        at = _parse_timestamp(row["timestamp"], key)
        if previous_at is not None and at < previous_at:
            raise ValueError("temporal sample events must be in timestamp order")
        previous_at = at

        for reference_field in ("derived_from",):
            references = row.get(reference_field, [])
            if not isinstance(references, list) or not all(
                isinstance(item, str) for item in references
            ):
                raise ValueError(f"event {key} {reference_field} must be a list")
            missing = [item for item in references if item not in seen]
            if missing:
                raise ValueError(
                    f"event {key} references later or missing keys: "
                    + ", ".join(missing)
                )
        supersedes = row.get("supersedes")
        if supersedes is not None and supersedes not in seen:
            raise ValueError(
                f"event {key} supersedes a later or missing key: {supersedes}"
            )
        if not isinstance(row.get("metadata", {}), dict):
            raise ValueError(f"event {key} metadata must be an object")
    return data


def load_temporal_sample(
    store: EventStore,
    path: str | Path,
) -> SampleLoadReport:
    """Replay a sample into the immutable store, safely and idempotently."""

    data = read_temporal_sample(path)
    sample_id = data["id"].strip()
    events: dict[str, Event] = {}
    created_count = 0
    reused_count = 0

    for row in data["events"]:
        key = row["key"].strip()
        metadata = dict(row.get("metadata", {}))
        metadata.update(
            {
                "sample_id": sample_id,
                "sample_key": key,
                "sample_week": row.get("week"),
                "provider": row["provider"].strip(),
            }
        )
        supersedes_key = row.get("supersedes")
        derived_keys = row.get("derived_from", [])
        event, created = store.append_idempotent(
            row["content"],
            idempotency_key=f"temporal-sample:{sample_id}:{key}",
            actor=row["actor"].strip(),
            kind=row["kind"].strip(),
            session_id=row["session_id"].strip(),
            metadata=metadata,
            supersedes=(
                events[supersedes_key].id if supersedes_key is not None else None
            ),
            derived_from=[events[source_key].id for source_key in derived_keys],
            timestamp=row["timestamp"].strip(),
        )
        if not created and (
            event.metadata != metadata
            or event.timestamp != row["timestamp"].strip()
        ):
            raise ValueError(
                "temporal sample idempotency key was reused for different "
                f"event data: {sample_id}:{key}"
            )
        events[key] = event
        if created:
            created_count += 1
        else:
            reused_count += 1

    return SampleLoadReport(
        sample_id=sample_id,
        created=created_count,
        reused=reused_count,
        events=events,
    )


def evaluate_temporal_sample(
    store: EventStore,
    path: str | Path,
) -> SampleEvaluation:
    """Evaluate inspectable ground-truth expectations against a loaded sample."""

    data = read_temporal_sample(path)
    sample_id = data["id"].strip()
    expected = data.get("expectations", {})
    if not isinstance(expected, dict):
        raise ValueError("temporal sample expectations must be an object")

    sample_events = [
        event
        for event in store.events(
            limit=max(1, store.count()),
            active_only=False,
            include_expired=True,
            ascending=True,
        )
        if event.metadata.get("sample_id") == sample_id
    ]
    by_key = {
        str(event.metadata.get("sample_key")): event
        for event in sample_events
    }
    checks: list[EvaluationCheck] = []

    declared_keys = [row["key"] for row in data["events"]]
    missing = [key for key in declared_keys if key not in by_key]
    checks.append(
        EvaluationCheck(
            "all fragments loaded",
            not missing,
            "all declared fragments present"
            if not missing
            else "missing: " + ", ".join(missing),
        )
    )
    if missing:
        return SampleEvaluation(sample_id, tuple(checks))

    span_days = (
        _parse_timestamp(sample_events[-1].timestamp, "last")
        - _parse_timestamp(sample_events[0].timestamp, "first")
    ).days
    minimum_span = int(expected.get("minimum_span_days", 0))
    checks.append(
        EvaluationCheck(
            "temporal span",
            span_days >= minimum_span,
            f"{span_days} days; required at least {minimum_span}",
        )
    )

    provider_count = len(
        {
            str(event.metadata.get("provider"))
            for event in sample_events
            if event.metadata.get("provider")
        }
    )
    minimum_providers = int(expected.get("minimum_providers", 0))
    checks.append(
        EvaluationCheck(
            "provider diversity",
            provider_count >= minimum_providers,
            f"{provider_count} providers; required at least {minimum_providers}",
        )
    )

    session_count = len({event.session_id for event in sample_events})
    minimum_sessions = int(expected.get("minimum_sessions", 0))
    checks.append(
        EvaluationCheck(
            "session diversity",
            session_count >= minimum_sessions,
            f"{session_count} sessions; required at least {minimum_sessions}",
        )
    )

    active_ids = {
        event.id
        for event in store.events(
            limit=max(1, store.count()),
            active_only=True,
            include_expired=True,
        )
    }
    for key in expected.get("active_keys", []):
        checks.append(
            EvaluationCheck(
                f"active:{key}",
                by_key[key].id in active_ids,
                "active" if by_key[key].id in active_ids else "unexpectedly superseded",
            )
        )
    for key in expected.get("inactive_keys", []):
        checks.append(
            EvaluationCheck(
                f"inactive:{key}",
                by_key[key].id not in active_ids,
                "superseded"
                if by_key[key].id not in active_ids
                else "unexpectedly active",
            )
        )

    for relationship in expected.get("connections", []):
        seed_key = relationship["seed"]
        candidate_key = relationship["candidate"]
        matches = find_connections(
            store,
            seed_event_id=by_key[seed_key].id,
            limit=max(20, store.count()),
        )
        match = next(
            (
                candidate
                for candidate in matches
                if candidate.event.id == by_key[candidate_key].id
            ),
            None,
        )
        required_channels = tuple(relationship.get("channels", []))
        required_shapes = set(relationship.get("shared_shapes", []))
        passed = (
            match is not None
            and match.channels == required_channels
            and required_shapes.issubset(match.shared_shapes)
        )
        detail = (
            f"channels={match.channels}, shapes={match.shared_shapes}"
            if match is not None
            else "candidate was not returned"
        )
        checks.append(
            EvaluationCheck(
                f"connection:{seed_key}->{candidate_key}",
                passed,
                detail,
            )
        )

    reconstructed = expected.get("reconstruction", {})
    if reconstructed:
        event = by_key[reconstructed["event"]]
        lowered = event.content.lower()
        required = [str(item).lower() for item in reconstructed.get("contains", [])]
        forbidden = [
            str(item).lower()
            for item in reconstructed.get("does_not_contain", [])
        ]
        passed = all(item in lowered for item in required) and not any(
            item in lowered for item in forbidden
        )
        checks.append(
            EvaluationCheck(
                "supported reconstruction",
                passed,
                "required evidence retained; forbidden inference absent"
                if passed
                else "reconstruction content violates declared ground truth",
            )
        )

    ok, count, error = store.verify()
    checks.append(
        EvaluationCheck(
            "append-only integrity",
            ok,
            f"{count} events verified" if ok else str(error),
        )
    )
    return SampleEvaluation(sample_id, tuple(checks))


def _parse_timestamp(value: str, key: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"event {key} has an invalid ISO timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"event {key} timestamp must include a timezone")
    return parsed
