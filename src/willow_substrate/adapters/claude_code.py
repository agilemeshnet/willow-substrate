"""Claude Code transcript capture and hook lifecycle adapter.

The adapter depends only on the JSON payload and transcript path supplied to a
hook. It never invokes a model. Its output is the same Willow event contract
used by every other provider adapter.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from willow_substrate.events import Event
from willow_substrate.store import EventStore


@dataclass(frozen=True)
class CaptureReport:
    """Result of one idempotent transcript scan."""

    session_id: str
    events: tuple[Event, ...]
    created_events: tuple[Event, ...]
    created: int
    skipped: int
    last_user: Event | None
    last_assistant: Event | None
    recent_input_tokens: int

    @property
    def last_turn(self) -> tuple[Event, ...]:
        return tuple(
            event
            for event in (self.last_user, self.last_assistant)
            if event is not None
        )


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and block.get("text"):
            parts.append(str(block["text"]))
    return "\n".join(parts).strip()


def _tool_names(content: Any) -> list[str]:
    if not isinstance(content, list):
        return []
    names = [
        str(block.get("name"))
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "tool_use"
        and block.get("name")
    ]
    return list(dict.fromkeys(names))


def _source_key(
    *,
    session_id: str,
    row: dict[str, Any],
    index: int,
    role: str,
    content: str,
    occurrence: int = 1,
) -> str:
    if role == "user":
        fingerprint = hashlib.sha256(content.encode("utf-8")).hexdigest()[:20]
        return f"claude-code:{session_id}:user:{fingerprint}:{occurrence}"
    stable_id = (
        row.get("uuid")
        or row.get("id")
        or (row.get("message") or {}).get("id")
    )
    if stable_id:
        suffix = str(stable_id)
    else:
        fingerprint = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        suffix = f"{row.get('timestamp', '')}:{index}:{fingerprint}"
    return f"claude-code:{session_id}:{role}:{suffix}"


def _usage_total(row: dict[str, Any]) -> int:
    message = row.get("message") or {}
    usage = message.get("usage") or row.get("usage") or {}
    if not isinstance(usage, dict):
        return 0
    total = 0
    for key in (
        "input_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
    ):
        try:
            total += int(usage.get(key) or 0)
        except (TypeError, ValueError):
            continue
    return total


def _iter_rows(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for index, line in enumerate(handle):
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(row, dict):
                yield index, row


def capture_transcript(
    store: EventStore,
    transcript_path: str | Path,
    *,
    session_id: str,
    user_actor: str = "user",
    assistant_actor: str = "willow",
) -> CaptureReport:
    """Capture all substantive transcript messages exactly once."""

    path = Path(transcript_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"transcript does not exist: {path}")

    captured: list[Event] = []
    newly_created: list[Event] = []
    created = 0
    skipped = 0
    last_user: Event | None = None
    last_assistant: Event | None = None
    recent_input_tokens = 0
    occurrences: Counter[tuple[str, str]] = Counter()

    for index, row in _iter_rows(path):
        recent_input_tokens = max(recent_input_tokens, _usage_total(row))
        row_type = str(row.get("type") or "")
        message = row.get("message") or row
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or row.get("role") or "")
        if role not in {"user", "assistant"}:
            continue

        # Tool results are provider plumbing, not another human utterance.
        if role == "user" and (
            row.get("sourceToolAssistantUUID")
            or any(
                isinstance(block, dict) and block.get("type") == "tool_result"
                for block in (
                    message.get("content")
                    if isinstance(message.get("content"), list)
                    else []
                )
            )
        ):
            continue
        if row.get("isMeta") or row_type in {
            "file-history-snapshot",
            "progress",
            "summary",
            "system",
        }:
            continue

        content = _extract_text(message.get("content"))
        if not content:
            continue
        occurrence_key = (
            role,
            hashlib.sha256(content.encode("utf-8")).hexdigest(),
        )
        occurrences[occurrence_key] += 1
        actor = user_actor if role == "user" else assistant_actor
        tools = _tool_names(message.get("content"))
        metadata: dict[str, Any] = {
            "provider": "claude-code",
            "provider_role": role,
            "source_index": index,
        }
        if message.get("model"):
            metadata["model"] = str(message["model"])
        if tools:
            metadata["tools"] = tools

        event, was_created = store.append_idempotent(
            content,
            idempotency_key=_source_key(
                session_id=session_id,
                row=row,
                index=index,
                role=role,
                content=content,
                occurrence=occurrences[occurrence_key],
            ),
            actor=actor,
            kind="message",
            session_id=session_id,
            metadata=metadata,
            timestamp=str(row.get("timestamp") or "") or None,
        )
        captured.append(event)
        if was_created:
            created += 1
            newly_created.append(event)
        else:
            skipped += 1
        if role == "user":
            last_user = event
        else:
            last_assistant = event

    return CaptureReport(
        session_id=session_id,
        events=tuple(captured),
        created_events=tuple(newly_created),
        created=created,
        skipped=skipped,
        last_user=last_user,
        last_assistant=last_assistant,
        recent_input_tokens=recent_input_tokens,
    )


def capture_hook_prompt(
    store: EventStore,
    prompt: str,
    *,
    session_id: str,
    transcript_path: str = "",
    actor: str = "user",
) -> tuple[Event | None, bool]:
    """Capture the current prompt when the transcript has not recorded it yet."""

    prompt = prompt.strip()
    if not prompt:
        return None, False
    size = 0
    occurrence = 1
    if transcript_path:
        try:
            path = Path(transcript_path).expanduser()
            size = path.stat().st_size
            for _, row in _iter_rows(path):
                message = row.get("message") or row
                if not isinstance(message, dict):
                    continue
                if (message.get("role") or row.get("role")) != "user":
                    continue
                if _extract_text(message.get("content")) == prompt:
                    occurrence += 1
        except OSError:
            pass
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    return store.append_idempotent(
        prompt,
        idempotency_key=(
            f"claude-code:{session_id}:user:"
            f"{hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:20]}:{occurrence}"
        ),
        actor=actor,
        kind="message",
        session_id=session_id,
        metadata={
            "provider": "claude-code",
            "provider_role": "user",
            "source": "UserPromptSubmit",
            "transcript_size": size,
        },
    )
