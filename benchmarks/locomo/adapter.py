"""LoCoMo dataset loader and Willow-ingester.

The upstream LoCoMo format is a JSON file per conversation. Each contains
a list of turns (speaker, text, timestamp) and a list of questions with
their gold evidence turn ids. This adapter reads that shape, ingests each
turn into an isolated Willow EventStore preserving the metadata the
reviewer's methodology asks for (session, speaker, timestamp, LoCoMo
dialogue id), and returns a manifest for the runner.

The dataset is NOT vendored in this repo. Place a snap-research/locomo
checkout at ``benchmarks/locomo/data/`` (or any path passed to
``load_locomo_conversations``) before running.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from willow_substrate.store import EventStore


@dataclass(frozen=True)
class LocomoQuestion:
    """One question and its gold evidence."""

    question_id: str
    text: str
    gold_turn_ids: tuple[str, ...]
    category: str  # single-hop, multi-hop, temporal, knowledge-update, etc.
    gold_answer: str | None = None


@dataclass(frozen=True)
class LocomoTurn:
    """One conversational turn."""

    turn_id: str
    speaker: str
    text: str
    timestamp: str


@dataclass
class LocomoConversation:
    """One LoCoMo conversation with its questions."""

    conversation_id: str
    turns: tuple[LocomoTurn, ...]
    questions: tuple[LocomoQuestion, ...]
    turn_id_to_event_id: dict[str, str] = field(default_factory=dict)


def load_locomo_conversations(
    data_dir: str | Path,
) -> list[LocomoConversation]:
    """Read every conversation JSON under data_dir.

    Accepts either the flat snap-research layout (one file per conversation)
    or a single combined file (a top-level dict with a 'conversations' key).
    Returns conversations sorted by conversation_id for stable output.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"LoCoMo data directory not found: {data_dir}. "
            "See docs/BENCHMARK_LOCOMO.md for how to fetch the dataset."
        )

    conversations: list[LocomoConversation] = []
    if data_dir.is_file():
        conversations.extend(_read_one_json(data_dir))
    else:
        for path in sorted(data_dir.rglob("*.json")):
            conversations.extend(_read_one_json(path))

    conversations.sort(key=lambda c: c.conversation_id)
    return conversations


def _read_one_json(path: Path) -> list[LocomoConversation]:
    with path.open() as fh:
        raw = json.load(fh)
    if isinstance(raw, dict) and "conversations" in raw:
        return [_parse_conversation(item) for item in raw["conversations"]]
    if isinstance(raw, list):
        return [_parse_conversation(item) for item in raw]
    if isinstance(raw, dict):
        return [_parse_conversation(raw)]
    raise ValueError(f"Unrecognised LoCoMo shape in {path}")


def _parse_conversation(payload: dict) -> LocomoConversation:
    # Accept the two shapes upstream has used across releases: turns as a
    # flat list, or turns grouped by session. Normalise to a flat tuple.
    conversation_id = str(
        payload.get("id")
        or payload.get("conversation_id")
        or payload.get("dialog_id")
        or "unknown"
    )

    raw_turns: list[dict] = []
    if "turns" in payload:
        raw_turns = payload["turns"]
    elif "sessions" in payload:
        for session in payload["sessions"]:
            raw_turns.extend(session.get("turns", []))
    elif "dialog" in payload:
        raw_turns = payload["dialog"]

    turns: list[LocomoTurn] = []
    for index, raw in enumerate(raw_turns):
        turn_id = str(
            raw.get("id") or raw.get("turn_id") or f"turn-{index:05d}"
        )
        speaker = str(raw.get("speaker") or raw.get("actor") or "unknown")
        text = str(
            raw.get("text") or raw.get("content") or raw.get("message") or ""
        ).strip()
        timestamp = str(
            raw.get("timestamp") or raw.get("time") or raw.get("date") or ""
        )
        if not text:
            continue
        turns.append(
            LocomoTurn(
                turn_id=turn_id,
                speaker=speaker,
                text=text,
                timestamp=timestamp,
            )
        )

    raw_questions = (
        payload.get("questions")
        or payload.get("qa")
        or payload.get("qas")
        or []
    )
    questions: list[LocomoQuestion] = []
    for index, raw in enumerate(raw_questions):
        gold_ids = raw.get("evidence") or raw.get("gold_evidence") or []
        if isinstance(gold_ids, dict):
            gold_ids = list(gold_ids.get("turn_ids", []))
        gold_ids = tuple(str(x) for x in gold_ids)
        questions.append(
            LocomoQuestion(
                question_id=str(
                    raw.get("id") or raw.get("qid") or f"q-{index:04d}"
                ),
                text=str(
                    raw.get("question")
                    or raw.get("text")
                    or raw.get("query")
                    or ""
                ).strip(),
                gold_turn_ids=gold_ids,
                category=str(raw.get("category") or raw.get("type") or "unknown"),
                gold_answer=raw.get("answer") or raw.get("gold_answer"),
            )
        )

    return LocomoConversation(
        conversation_id=conversation_id,
        turns=tuple(turns),
        questions=tuple(questions),
    )


def ingest_into_store(
    store: EventStore, conversation: LocomoConversation
) -> None:
    """Ingest one conversation's turns as immutable Willow events.

    Preserves the LoCoMo turn id in metadata so scoring can map from
    Willow event ids back to gold turn ids without extra bookkeeping.
    Uses append_idempotent so re-runs against the same store are safe.
    Populates conversation.turn_id_to_event_id in place.
    """
    for turn in conversation.turns:
        event, _created = store.append_idempotent(
            turn.text,
            idempotency_key=f"locomo:{conversation.conversation_id}:{turn.turn_id}",
            actor=turn.speaker,
            kind="message",
            session_id=conversation.conversation_id,
            metadata={
                "benchmark": "locomo",
                "locomo_conversation_id": conversation.conversation_id,
                "locomo_turn_id": turn.turn_id,
            },
            timestamp=turn.timestamp or None,
        )
        conversation.turn_id_to_event_id[turn.turn_id] = event.id
