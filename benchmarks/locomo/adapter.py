"""LoCoMo dataset loader and Willow-ingester.

The upstream LoCoMo10 release ships one JSON file whose top level is a list
of 10 conversations. Each conversation looks like::

    {
      "sample_id": "conv-26",
      "conversation": {
        "speaker_a": "Caroline",
        "speaker_b": "Melanie",
        "session_1":            [ {"speaker": "...", "dia_id": "D1:1",  "text": "..."}, ... ],
        "session_1_date_time":  "1:56 pm on 8 May, 2023",
        "session_2":            [ ... ],
        "session_2_date_time":  "...",
        ...
      },
      "qa": [
        {"question": "...", "answer": "...", "evidence": "['D1:3']", "category": "2"},
        ...
      ]
    }

This adapter parses that shape (and preserves fallback paths for older
LoCoMo layouts we still occasionally hit), normalises the natural-language
per-session timestamps to timezone-aware ISO-8601 UTC (round-two reviewer
requirement, PR #23 on the store side), ingests each turn as an isolated
Willow event, and returns a manifest for the runner.

The dataset is NOT vendored in this repo. Point ``load_locomo_conversations``
at either a directory (walks ``*.json``) or a single JSON file. The
locomo10 release is one file; per-conversation shards from earlier releases
are also supported.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

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
    timestamp: str  # ISO-8601, tz-aware, after normalisation


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

    Accepts the LoCoMo10 combined file (top-level list of conversations),
    the earlier per-shard layout (one file per conversation), or a single
    combined dict with a 'conversations' key. Returns conversations sorted
    by conversation_id for stable output.
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


# ----------------------------------------------------------------------
# Parsers


def _parse_conversation(payload: dict) -> LocomoConversation:
    """Parse one conversation payload into a LocomoConversation.

    Two shapes handled:
    - LoCoMo10 combined-file shape: sample_id + conversation{session_N +
      session_N_date_time} + qa[]. This is the one currently shipped.
    - Legacy per-file shape: id/conversation_id + turns[]/sessions[]/dialog[]
      + questions[]/qa[]. Kept as a fallback so older mirrors still parse.
    """
    conversation_id = str(
        payload.get("sample_id")
        or payload.get("id")
        or payload.get("conversation_id")
        or payload.get("dialog_id")
        or "unknown"
    )

    turns = _parse_turns(payload)
    questions = _parse_questions(payload)

    return LocomoConversation(
        conversation_id=conversation_id,
        turns=turns,
        questions=questions,
    )


def _parse_turns(payload: dict) -> tuple[LocomoTurn, ...]:
    """Extract the flat, ordered turn list.

    Prefers the LoCoMo10 shape (conversation dict with session_N keys) and
    falls back to older shapes only when that key is missing. Each turn
    inherits its session's natural-language timestamp, normalised to
    timezone-aware ISO-8601 UTC before it hits the ledger.
    """
    conversation = payload.get("conversation")
    if isinstance(conversation, dict) and any(
        _is_session_turns_key(k) for k in conversation
    ):
        return _parse_locomo10_turns(conversation)

    # Legacy fallbacks. Older mirrors flattened turns to a single list,
    # or nested sessions with sub-'turns', or used a 'dialog' key.
    raw_turns: list[dict] = []
    session_ts: dict[int, str | None] = {}
    if "turns" in payload and isinstance(payload["turns"], list):
        raw_turns = payload["turns"]
    elif "sessions" in payload and isinstance(payload["sessions"], list):
        for i, session in enumerate(payload["sessions"], start=1):
            session_ts[i] = session.get("timestamp") or session.get("date")
            for raw in session.get("turns", []):
                raw = dict(raw)
                raw.setdefault("_session_index", i)
                raw_turns.append(raw)
    elif "dialog" in payload and isinstance(payload["dialog"], list):
        raw_turns = payload["dialog"]

    turns: list[LocomoTurn] = []
    for index, raw in enumerate(raw_turns):
        turn_id = str(
            raw.get("dia_id")
            or raw.get("id")
            or raw.get("turn_id")
            or f"turn-{index:05d}"
        )
        speaker = str(raw.get("speaker") or raw.get("actor") or "unknown")
        text = str(
            raw.get("text") or raw.get("content") or raw.get("message") or ""
        ).strip()
        raw_ts = (
            raw.get("timestamp")
            or raw.get("time")
            or raw.get("date")
            or session_ts.get(raw.get("_session_index", 0))
            or ""
        )
        normalised_ts = _normalise_locomo_timestamp(raw_ts) if raw_ts else ""
        if not text:
            continue
        turns.append(
            LocomoTurn(
                turn_id=turn_id,
                speaker=speaker,
                text=text,
                timestamp=normalised_ts,
            )
        )
    return tuple(turns)


def _parse_locomo10_turns(conversation: dict) -> tuple[LocomoTurn, ...]:
    """LoCoMo10 shape: walk session_N keys in numeric order.

    Timestamps come from the paired session_N_date_time entry and inherit
    to every turn in that session. Empty text turns are skipped so the
    downstream ledger is not polluted with blank rows.
    """
    session_keys = sorted(
        (k for k in conversation if _is_session_turns_key(k)),
        key=_session_index,
    )
    turns: list[LocomoTurn] = []
    for key in session_keys:
        session_turns = conversation.get(key)
        if not isinstance(session_turns, list):
            continue
        idx = _session_index(key)
        raw_ts = conversation.get(f"session_{idx}_date_time", "") or ""
        normalised_ts = (
            _normalise_locomo_timestamp(raw_ts) if raw_ts else ""
        )
        for pos, raw in enumerate(session_turns):
            if not isinstance(raw, dict):
                continue
            text = str(raw.get("text") or "").strip()
            if not text:
                continue
            turn_id = str(
                raw.get("dia_id")
                or raw.get("id")
                or f"session-{idx}-turn-{pos:05d}"
            )
            speaker = str(raw.get("speaker") or "unknown")
            turns.append(
                LocomoTurn(
                    turn_id=turn_id,
                    speaker=speaker,
                    text=text,
                    timestamp=normalised_ts,
                )
            )
    return tuple(turns)


def _parse_questions(payload: dict) -> tuple[LocomoQuestion, ...]:
    """Extract the flat question list with parsed evidence turn ids."""
    raw_questions = (
        payload.get("qa")
        or payload.get("qas")
        or payload.get("questions")
        or []
    )
    questions: list[LocomoQuestion] = []
    for index, raw in enumerate(raw_questions):
        gold_ids = _parse_evidence(
            raw.get("evidence") or raw.get("gold_evidence")
        )
        questions.append(
            LocomoQuestion(
                question_id=str(
                    raw.get("id")
                    or raw.get("qid")
                    or f"q-{index:04d}"
                ),
                text=str(
                    raw.get("question")
                    or raw.get("text")
                    or raw.get("query")
                    or ""
                ).strip(),
                gold_turn_ids=gold_ids,
                category=str(
                    raw.get("category") or raw.get("type") or "unknown"
                ),
                gold_answer=raw.get("answer") or raw.get("gold_answer"),
            )
        )
    return tuple(questions)


def _parse_evidence(value) -> tuple[str, ...]:
    """Evidence is one of: a JSON-encoded string like "['D1:3']", a plain
    list of dia_ids, or a dict with a 'turn_ids' key. Normalise to tuple.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ()
        # LoCoMo10 stores evidence as a Python-repr-like string. Try JSON
        # first (double quotes) then fall back to a naive split.
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            try:
                parsed = json.loads(stripped.replace("'", '"'))
            except json.JSONDecodeError:
                parsed = [s.strip(" '\"[]") for s in stripped.split(",")]
        if isinstance(parsed, list):
            return tuple(str(x) for x in parsed if str(x).strip())
        return (str(parsed),)
    if isinstance(value, dict):
        turn_ids = value.get("turn_ids") or []
        return tuple(str(x) for x in turn_ids)
    if isinstance(value, list):
        return tuple(str(x) for x in value)
    return (str(value),)


# ----------------------------------------------------------------------
# Timestamp normalisation


# Precompiled once. Matches "1:56 pm on 8 May, 2023" and light variants
# (case-insensitive; optional comma before year; single- or double-digit day).
_LOCOMO_TS_RE = re.compile(
    r"^\s*"
    r"(?P<hour>\d{1,2}):(?P<minute>\d{2})\s*"
    r"(?P<meridiem>am|pm)\s+on\s+"
    r"(?P<day>\d{1,2})\s+"
    r"(?P<month>[A-Za-z]+),?\s+"
    r"(?P<year>\d{4})\s*$",
    re.IGNORECASE,
)
_MONTHS = {
    m.lower(): i
    for i, m in enumerate(
        [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ],
        start=1,
    )
}


def _normalise_locomo_timestamp(value: str) -> str:
    """Convert "1:56 pm on 8 May, 2023" to "2023-05-08T13:56:00+00:00".

    LoCoMo timestamps do not carry a timezone, and there is no reliable way
    to recover one; the round-two reviewer flagged this. We treat them as
    UTC and stamp +00:00 so downstream span arithmetic works and the store's
    timezone-aware validator (PR #23) accepts them. The original string is
    preserved by the ingester in event.metadata so nothing is lost.

    If the string is already ISO-8601, pass it through (still adding UTC
    if naive). Unparseable strings raise ValueError so the caller can see
    them rather than silently drop the whole session.
    """
    if not isinstance(value, str):
        raise TypeError(f"timestamp must be str, got {type(value).__name__}")
    stripped = value.strip()
    if not stripped:
        raise ValueError("empty timestamp string")

    match = _LOCOMO_TS_RE.match(stripped)
    if match:
        hour = int(match["hour"]) % 12
        if match["meridiem"].lower() == "pm":
            hour += 12
        minute = int(match["minute"])
        day = int(match["day"])
        month = _MONTHS.get(match["month"].lower())
        year = int(match["year"])
        if month is None:
            raise ValueError(
                f"unknown month name in LoCoMo timestamp {value!r}"
            )
        dt = datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
        return dt.isoformat()

    # Try ISO-8601 directly. Add UTC if the caller was naive.
    try:
        dt = datetime.fromisoformat(stripped.replace("Z", "+00:00"))
    except ValueError as err:
        raise ValueError(
            f"cannot normalise LoCoMo timestamp {value!r} "
            f"(expected 'H:MM am/pm on D Month, YYYY' or ISO-8601)"
        ) from err
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


# ----------------------------------------------------------------------
# LoCoMo10 session-key helpers


def _is_session_turns_key(key: str) -> bool:
    """True for 'session_1', 'session_10', etc. False for 'session_1_date_time'."""
    return bool(re.fullmatch(r"session_\d+", key))


def _session_index(key: str) -> int:
    """Numeric index of a 'session_N' key; 0 for anything unparseable."""
    match = re.fullmatch(r"session_(\d+)(?:_date_time)?", key)
    return int(match.group(1)) if match else 0


# ----------------------------------------------------------------------
# Ingester


def ingest_into_store(
    store: EventStore, conversation: LocomoConversation
) -> None:
    """Ingest one conversation's turns as immutable Willow events.

    Preserves the LoCoMo turn id in metadata so scoring can map from
    Willow event ids back to gold turn ids without extra bookkeeping.
    The pre-normalisation timestamp is also preserved under
    ``metadata.locomo_timestamp_raw`` so the round trip is auditable.
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
                "locomo_timestamp_raw": turn.timestamp,
            },
            timestamp=turn.timestamp or None,
        )
        conversation.turn_id_to_event_id[turn.turn_id] = event.id
