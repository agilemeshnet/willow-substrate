"""Event-driven lifecycle hooks for temporary LLM processes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from willow_substrate.adapters.claude_code import (
    CaptureReport,
    capture_hook_prompt,
    capture_transcript,
)
from willow_substrate.context import ContextBuilder
from willow_substrate.engrams import (
    append_turn_engram,
    crystallize_retroactive_engrams,
)
from willow_substrate.reflection import meditate_session, summarize_session
from willow_substrate.store import EventStore


@dataclass(frozen=True)
class HookResult:
    """Provider-neutral result: printable context plus appended event IDs."""

    phase: str
    output: str
    event_ids: tuple[str, ...] = ()
    captured: int = 0
    skipped: int = 0


def _integer(payload: dict[str, Any], *keys: str) -> int:
    for key in keys:
        try:
            value = int(payload.get(key) or 0)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return 0


def _capture(
    store: EventStore,
    payload: dict[str, Any],
    *,
    session_id: str,
    user_actor: str,
    assistant_actor: str,
) -> CaptureReport:
    transcript_path = str(
        payload.get("transcript_path")
        or payload.get("transcript")
        or ""
    )
    if transcript_path and Path(transcript_path).expanduser().exists():
        return capture_transcript(
            store,
            transcript_path,
            session_id=session_id,
            user_actor=user_actor,
            assistant_actor=assistant_actor,
        )
    return CaptureReport(
        session_id=session_id,
        events=(),
        created_events=(),
        created=0,
        skipped=0,
        last_user=None,
        last_assistant=None,
        recent_input_tokens=0,
    )


def handle_claude_hook(
    store: EventStore,
    payload: dict[str, Any],
    *,
    phase: str,
    token_budget: int = 1600,
    context_limit: int = 0,
    user_actor: str = "user",
    assistant_actor: str = "willow",
    engram_ttl_seconds: int = 1800,
) -> HookResult:
    """Run one Claude Code lifecycle phase without invoking a model."""

    normalised = phase.strip().lower().replace("_", "-")
    aliases = {
        "turn": "prompt",
        "user-prompt-submit": "prompt",
        "post-compact": "compact",
        "session-start": "start",
        "session-end": "end",
    }
    normalised = aliases.get(normalised, normalised)
    if normalised not in {"start", "prompt", "stop", "compact", "end"}:
        raise ValueError(f"unsupported hook phase: {phase}")

    session_id = str(payload.get("session_id") or "claude-default")
    report = _capture(
        store,
        payload,
        session_id=session_id,
        user_actor=user_actor,
        assistant_actor=assistant_actor,
    )
    appended: list[str] = [event.id for event in report.created_events]

    if normalised == "start":
        packet = ContextBuilder(store).boot(
            agent=assistant_actor,
            token_budget=max(100, token_budget),
        )
        return HookResult(
            phase=normalised,
            output=packet.markdown,
            event_ids=packet.event_ids,
            captured=report.created,
            skipped=report.skipped,
        )

    if normalised == "prompt":
        prompt = str(
            payload.get("prompt")
            or payload.get("user_prompt")
            or ""
        ).strip()
        transcript_path = str(
            payload.get("transcript_path")
            or payload.get("transcript")
            or ""
        )
        prompt_event, created = capture_hook_prompt(
            store,
            prompt,
            session_id=session_id,
            transcript_path=transcript_path,
            actor=user_actor,
        )
        if prompt_event and created:
            appended.append(prompt_event.id)

        query = prompt or (
            report.last_user.content if report.last_user else "current work"
        )
        host_limit = context_limit or _integer(
            payload,
            "context_limit",
            "context_window",
            "model_context_window",
        )
        packet = ContextBuilder(store).build(
            query,
            token_budget=max(100, token_budget),
            exclude_session_id=session_id,
            mode="ambient",
            include_hot_peers=True,
            recent_input_tokens=report.recent_input_tokens,
            context_limit=host_limit,
        )
        # Quiet when no peer or durable evidence has anything to contribute.
        output = packet.markdown if packet.event_ids else ""
        return HookResult(
            phase=normalised,
            output=output,
            event_ids=tuple(appended),
            captured=report.created + int(created),
            skipped=report.skipped,
        )

    if normalised == "stop":
        engram, created = append_turn_engram(
            store,
            session_id=session_id,
            user_event=report.last_user,
            assistant_event=report.last_assistant,
            actor=assistant_actor,
            ttl_seconds=engram_ttl_seconds,
        )
        if engram and created:
            appended.append(engram.id)
        return HookResult(
            phase=normalised,
            output="",
            event_ids=tuple(appended),
            captured=report.created + int(created),
            skipped=report.skipped,
        )

    if normalised == "compact":
        summary = payload.get("summary")
        if isinstance(summary, dict):
            summary = " ".join(str(value) for value in summary.values())
        query = str(summary or "recover this session after compaction")[:1000]
        packet = ContextBuilder(store).build(
            query,
            token_budget=max(100, token_budget),
            session_id=session_id,
            mode="recovery",
            include_hot_peers=True,
        )
        return HookResult(
            phase=normalised,
            output=(
                "COMPACTION COMPLETE. Rehydrate from the preserved evidence "
                "below, state the recovered thread briefly, then continue.\n\n"
                + packet.markdown
            ),
            event_ids=packet.event_ids,
            captured=report.created,
            skipped=report.skipped,
        )

    summation, summation_created = summarize_session(
        store,
        session_id,
        actor=assistant_actor,
    )
    meditation, meditation_created = meditate_session(
        store,
        session_id,
        actor=assistant_actor,
    )
    retroactive = crystallize_retroactive_engrams(store, limit=3)
    for event, created in (
        (summation, summation_created),
        (meditation, meditation_created),
    ):
        if created:
            appended.append(event.id)
    appended.extend(event.id for event in retroactive)
    return HookResult(
        phase=normalised,
        output="",
        event_ids=tuple(appended),
        captured=(
            report.created
            + int(summation_created)
            + int(meditation_created)
            + len(retroactive)
        ),
        skipped=report.skipped,
    )
