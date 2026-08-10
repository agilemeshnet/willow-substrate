"""Concurrent, append-only local event store.

SQLite is used as the first reference substrate because it is available with
Python, supports multiple terminal processes, and can enforce immutability with
database triggers. AuraDB, Markdown, and vector indexes are adapters and
projections around this contract.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from willow_substrate.events import Event, EventHit


GENESIS_HASH = hashlib.sha256(b"willow-genesis-v1").hexdigest()
SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS willow_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    timestamp TEXT NOT NULL,
    session_id TEXT NOT NULL,
    actor TEXT NOT NULL,
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    supersedes TEXT,
    derived_from_json TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    hash TEXT NOT NULL UNIQUE,
    FOREIGN KEY (supersedes) REFERENCES events(id)
);

CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id, seq);
CREATE INDEX IF NOT EXISTS idx_events_kind ON events(kind, seq);
CREATE INDEX IF NOT EXISTS idx_events_supersedes ON events(supersedes);

CREATE VIRTUAL TABLE IF NOT EXISTS events_fts USING fts5(
    event_id UNINDEXED,
    content,
    actor,
    kind,
    session_id,
    tokenize = 'porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS events_search_after_insert
AFTER INSERT ON events
BEGIN
    INSERT INTO events_fts(event_id, content, actor, kind, session_id)
    VALUES (new.id, new.content, new.actor, new.kind, new.session_id);
END;

CREATE TRIGGER IF NOT EXISTS events_are_immutable_update
BEFORE UPDATE ON events
BEGIN
    SELECT RAISE(ABORT, 'Willow events are immutable; append a supersession');
END;

CREATE TRIGGER IF NOT EXISTS events_are_immutable_delete
BEFORE DELETE ON events
BEGIN
    SELECT RAISE(ABORT, 'Willow events are immutable; append a supersession');
END;
"""


def default_home() -> Path:
    configured = os.environ.get("WILLOW_HOME")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".willow"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validated_timestamp(value: str) -> str:
    """Reject caller-supplied timestamps that are not ISO-8601 parseable.

    The timestamp is hashed into every event's payload and the sample suite's
    pass criteria include temporal spans; those temporal claims are only as
    trustworthy as the field they rest on. Without this check, append() will
    accept and store 'banana' and hash it into the chain.

    The value that parses cleanly is returned unchanged so nothing about the
    hash of a legitimate timestamp is disturbed.
    """
    if not isinstance(value, str):
        raise TypeError(
            f"timestamp must be a string in ISO-8601 form; got {type(value).__name__}"
        )
    stripped = value.strip()
    if not stripped:
        raise ValueError("timestamp must not be empty")
    try:
        # Python 3.11+ fromisoformat accepts a wide range of ISO-8601 forms
        # including trailing 'Z' and offsets; that is exactly what we want.
        datetime.fromisoformat(stripped)
    except ValueError as exc:
        raise ValueError(
            f"timestamp is not ISO-8601 parseable: {value!r} ({exc})"
        ) from exc
    return value


class EventStore:
    """Append, verify, and retrieve immutable Willow events."""

    def __init__(self, home: str | Path | None = None):
        self.home = Path(home).expanduser() if home is not None else default_home()
        self.home.mkdir(parents=True, exist_ok=True)
        self.db_path = self.home / "willow.db"
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 15000")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA synchronous = FULL")
        return conn

    @contextmanager
    def _session(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize(self) -> None:
        # WAL selection and first-schema creation are the only operations that
        # can race before the database exists. Retry this short bootstrap so
        # several freshly opened terminals can all be first.
        for attempt in range(8):
            conn = sqlite3.connect(self.db_path, timeout=15)
            try:
                conn.execute("PRAGMA busy_timeout = 15000")
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("PRAGMA synchronous = FULL")
                conn.execute("PRAGMA foreign_keys = ON")
                conn.executescript(SCHEMA)
                conn.execute(
                    "INSERT OR IGNORE INTO willow_meta(key, value) "
                    "VALUES ('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
                # Anchored-head sentinels: the tail of the chain is only as
                # trustworthy as an independent record of where it is. Without
                # these, a bare DELETE of the highest-seq row leaves a shorter
                # but self-consistent chain that verify() cannot distinguish
                # from an honest history. Storing head_hash and event_count in
                # willow_meta forces any tail forgery to also forge these two
                # values consistently.
                conn.execute(
                    "INSERT OR IGNORE INTO willow_meta(key, value) "
                    "VALUES ('head_hash', ?)",
                    (GENESIS_HASH,),
                )
                conn.execute(
                    "INSERT OR IGNORE INTO willow_meta(key, value) "
                    "VALUES ('event_count', '0')",
                )
                conn.commit()
                return
            except sqlite3.OperationalError as exc:
                conn.rollback()
                if "locked" not in str(exc).lower() or attempt == 7:
                    raise
                time.sleep(0.025 * (attempt + 1))
            finally:
                conn.close()

    def append(
        self,
        content: str,
        *,
        actor: str = "user",
        kind: str = "message",
        session_id: str = "default",
        metadata: dict[str, Any] | None = None,
        supersedes: str | None = None,
        derived_from: Iterable[str] | None = None,
        event_id: str | None = None,
        timestamp: str | None = None,
    ) -> Event:
        """Append one event under a write lock and return it."""

        content = content.strip()
        if not content:
            raise ValueError("event content must not be empty")
        if not actor.strip() or not kind.strip() or not session_id.strip():
            raise ValueError("actor, kind, and session_id must not be empty")

        metadata_value = dict(metadata or {})
        derived_value = tuple(dict.fromkeys(derived_from or ()))
        event_id_value = event_id or f"evt-{uuid.uuid4().hex}"
        timestamp_value = _validated_timestamp(timestamp) if timestamp else _utc_now()

        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")

            if supersedes and not conn.execute(
                "SELECT 1 FROM events WHERE id = ?", (supersedes,)
            ).fetchone():
                raise KeyError(f"event to supersede does not exist: {supersedes}")

            missing_sources = [
                source_id
                for source_id in derived_value
                if not conn.execute(
                    "SELECT 1 FROM events WHERE id = ?", (source_id,)
                ).fetchone()
            ]
            if missing_sources:
                raise KeyError(
                    "derived source events do not exist: " + ", ".join(missing_sources)
                )

            row = conn.execute(
                "SELECT hash FROM events ORDER BY seq DESC LIMIT 1"
            ).fetchone()
            prev_hash = row["hash"] if row else GENESIS_HASH

            payload = {
                "id": event_id_value,
                "timestamp": timestamp_value,
                "session_id": session_id,
                "actor": actor,
                "kind": kind,
                "content": content,
                "metadata": metadata_value,
                "supersedes": supersedes,
                "derived_from": list(derived_value),
            }
            event_hash = hashlib.sha256(
                (prev_hash + _canonical_json(payload)).encode("utf-8")
            ).hexdigest()

            cursor = conn.execute(
                """
                INSERT INTO events(
                    id, timestamp, session_id, actor, kind, content,
                    metadata_json, supersedes, derived_from_json, prev_hash, hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id_value,
                    timestamp_value,
                    session_id,
                    actor,
                    kind,
                    content,
                    _canonical_json(metadata_value),
                    supersedes,
                    _canonical_json(list(derived_value)),
                    prev_hash,
                    event_hash,
                ),
            )
            seq = int(cursor.lastrowid)
            # Advance the anchored-head sentinels in the same transaction so a
            # crash between the INSERT and the UPDATE cannot leave a chain out
            # of step with its declared tail. See _initialize for the rationale.
            conn.execute(
                "UPDATE willow_meta SET value = ? WHERE key = 'head_hash'",
                (event_hash,),
            )
            conn.execute(
                "UPDATE willow_meta SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT) "
                "WHERE key = 'event_count'"
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

        return Event(
            seq=seq,
            id=event_id_value,
            timestamp=timestamp_value,
            session_id=session_id,
            actor=actor,
            kind=kind,
            content=content,
            metadata=metadata_value,
            supersedes=supersedes,
            derived_from=derived_value,
            prev_hash=prev_hash,
            hash=event_hash,
        )

    def append_idempotent(
        self,
        content: str,
        *,
        idempotency_key: str,
        actor: str = "user",
        kind: str = "message",
        session_id: str = "default",
        metadata: dict[str, Any] | None = None,
        supersedes: str | None = None,
        derived_from: Iterable[str] | None = None,
        timestamp: str | None = None,
    ) -> tuple[Event, bool]:
        """Append once for a stable external source.

        Hook invocations are deliberately repeatable: a provider may invoke a
        hook more than once, or a transcript may be rescanned from the start.
        The external source key is hashed into an event ID so retries return
        the original immutable event instead of duplicating experience.

        Returns ``(event, created)``. Reusing a key for different event content
        is rejected rather than silently merging two observations.
        """

        key = idempotency_key.strip()
        if not key:
            raise ValueError("idempotency_key must not be empty")
        event_id = "evt-" + hashlib.sha256(
            f"willow-source-v1:{key}".encode("utf-8")
        ).hexdigest()[:32]
        existing = self.get(event_id)
        if existing is not None:
            expected_sources = tuple(dict.fromkeys(derived_from or ()))
            if (
                existing.content != content.strip()
                or existing.actor != actor
                or existing.kind != kind
                or existing.session_id != session_id
                or existing.supersedes != supersedes
                or existing.derived_from != expected_sources
            ):
                raise ValueError(
                    "idempotency key was reused for different event data: "
                    f"{idempotency_key}"
                )
            return existing, False

        try:
            return (
                self.append(
                    content,
                    actor=actor,
                    kind=kind,
                    session_id=session_id,
                    metadata=metadata,
                    supersedes=supersedes,
                    derived_from=derived_from,
                    event_id=event_id,
                    timestamp=timestamp,
                ),
                True,
            )
        except sqlite3.IntegrityError:
            # A concurrent hook may have inserted the same deterministic event
            # after our first read. Confirm and return the winner.
            concurrent = self.get(event_id)
            if concurrent is None:
                raise
            return concurrent, False

    def correct(
        self,
        target_event_id: str,
        correction: str,
        *,
        actor: str = "user",
        session_id: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> Event:
        """Append a correction without altering the original event."""

        return self.append(
            correction,
            actor=actor,
            kind="correction",
            session_id=session_id,
            metadata=metadata,
            supersedes=target_event_id,
            derived_from=(target_event_id,),
        )

    def get(self, event_id: str) -> Event | None:
        with self._session() as conn:
            row = conn.execute(
                "SELECT * FROM events WHERE id = ?", (event_id,)
            ).fetchone()
        return self._row_to_event(row) if row else None

    def events(
        self,
        *,
        limit: int = 100,
        session_id: str | None = None,
        exclude_session_id: str | None = None,
        kind: str | None = None,
        active_only: bool = True,
        include_expired: bool = False,
        since: str | None = None,
        ascending: bool = False,
    ) -> list[Event]:
        where: list[str] = []
        params: list[Any] = []
        if session_id:
            where.append("e.session_id = ?")
            params.append(session_id)
        if exclude_session_id:
            where.append("e.session_id != ?")
            params.append(exclude_session_id)
        if kind:
            where.append("e.kind = ?")
            params.append(kind)
        if since:
            where.append("e.timestamp >= ?")
            params.append(since)
        if active_only:
            where.append(
                "NOT EXISTS (SELECT 1 FROM events s WHERE s.supersedes = e.id)"
            )
        if not include_expired:
            where.append(
                "("
                "json_extract(e.metadata_json, '$.expires_at') IS NULL "
                "OR json_extract(e.metadata_json, '$.expires_at') > ?"
                ")"
            )
            params.append(_utc_now())

        sql = "SELECT e.* FROM events e"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY e.seq " + ("ASC" if ascending else "DESC")
        sql += " LIMIT ?"
        params.append(max(0, limit))

        with self._session() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_event(row) for row in rows]

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        session_ids: Iterable[str] | None = None,
        exclude_session_id: str | None = None,
        active_only: bool = True,
        include_expired: bool = False,
    ) -> list[EventHit]:
        """Search active events using the bundled SQLite FTS index."""

        terms = re.findall(r"[\w'-]{2,}", query, flags=re.UNICODE)
        if not terms:
            return []
        fts_query = " OR ".join(f'"{term.replace(chr(34), "")}"' for term in terms[:20])

        where = ["events_fts MATCH ?"]
        params: list[Any] = [fts_query]

        session_values = tuple(dict.fromkeys(session_ids or ()))
        if session_values:
            placeholders = ",".join("?" for _ in session_values)
            where.append(f"e.session_id IN ({placeholders})")
            params.extend(session_values)
        if exclude_session_id:
            where.append("e.session_id != ?")
            params.append(exclude_session_id)
        if active_only:
            where.append(
                "NOT EXISTS (SELECT 1 FROM events s WHERE s.supersedes = e.id)"
            )
        if not include_expired:
            where.append(
                "("
                "json_extract(e.metadata_json, '$.expires_at') IS NULL "
                "OR json_extract(e.metadata_json, '$.expires_at') > ?"
                ")"
            )
            params.append(_utc_now())

        params.append(max(0, limit))
        sql = f"""
            SELECT e.*, bm25(events_fts) AS fts_rank
            FROM events_fts
            JOIN events e ON e.id = events_fts.event_id
            WHERE {' AND '.join(where)}
            ORDER BY fts_rank ASC, e.seq DESC
            LIMIT ?
        """

        with self._session() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [
            EventHit(
                event=self._row_to_event(row),
                score=1.0 / (index + 1),
                source="fts",
            )
            for index, row in enumerate(rows)
        ]

    def verify(self) -> tuple[bool, int, str | None]:
        """Verify the global hash chain from genesis to the latest event.

        Beyond the hash walk, two reconciliations close known audit gaps:

        - Tail: reconcile against the anchored-head sentinels stored in
          willow_meta (head_hash, event_count). Without them, a bare DELETE
          of the highest-seq row leaves a shorter but self-consistent chain
          that a hash walk alone cannot flag.
        - Retrieval: reconcile events_fts.event_id as a set against events.id.
          Without this, a silent deletion of a search index row leaves
          retrieval censored while the ledger itself still reads honest.
          Integrity and retrieval live in different tables, so both are
          checked here.
        """

        with self._session() as conn:
            rows = conn.execute("SELECT * FROM events ORDER BY seq ASC").fetchall()
            event_ids = {str(row["id"]) for row in rows}
            fts_ids = {
                str(row["event_id"])
                for row in conn.execute("SELECT event_id FROM events_fts")
            }
            meta = {
                str(row["key"]): str(row["value"])
                for row in conn.execute(
                    "SELECT key, value FROM willow_meta "
                    "WHERE key IN ('head_hash', 'event_count')"
                )
            }

        prev_hash = GENESIS_HASH
        for index, row in enumerate(rows):
            event = self._row_to_event(row)
            if event.prev_hash != prev_hash:
                return False, len(rows), f"previous hash mismatch at sequence {event.seq}"

            payload = {
                "id": event.id,
                "timestamp": event.timestamp,
                "session_id": event.session_id,
                "actor": event.actor,
                "kind": event.kind,
                "content": event.content,
                "metadata": event.metadata,
                "supersedes": event.supersedes,
                "derived_from": list(event.derived_from),
            }
            expected = hashlib.sha256(
                (prev_hash + _canonical_json(payload)).encode("utf-8")
            ).hexdigest()
            if event.hash != expected:
                return False, len(rows), f"event hash mismatch at sequence {event.seq}"
            prev_hash = event.hash

        expected_head = meta.get("head_hash", GENESIS_HASH)
        expected_count_raw = meta.get("event_count", "0")
        try:
            expected_count = int(expected_count_raw)
        except ValueError:
            return False, len(rows), (
                f"anchored event_count is not an integer: {expected_count_raw!r}"
            )

        if len(rows) != expected_count:
            return False, len(rows), (
                f"tail length mismatch: chain has {len(rows)} events "
                f"but anchored event_count is {expected_count}"
            )
        if prev_hash != expected_head:
            return False, len(rows), (
                "tail hash mismatch: chain tail does not match anchored head_hash"
            )

        missing_from_fts = event_ids - fts_ids
        if missing_from_fts:
            return False, len(rows), (
                f"search index missing {len(missing_from_fts)} event(s) present "
                f"in the ledger; example: {sorted(missing_from_fts)[0]}"
            )
        orphan_fts = fts_ids - event_ids
        if orphan_fts:
            return False, len(rows), (
                f"search index has {len(orphan_fts)} row(s) without a matching "
                f"ledger event; example: {sorted(orphan_fts)[0]}"
            )

        return True, len(rows), None

    def count(self) -> int:
        with self._session() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM events").fetchone()[0])

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> Event:
        return Event(
            seq=int(row["seq"]),
            id=str(row["id"]),
            timestamp=str(row["timestamp"]),
            session_id=str(row["session_id"]),
            actor=str(row["actor"]),
            kind=str(row["kind"]),
            content=str(row["content"]),
            metadata=json.loads(row["metadata_json"]),
            supersedes=row["supersedes"],
            derived_from=tuple(json.loads(row["derived_from_json"])),
            prev_hash=str(row["prev_hash"]),
            hash=str(row["hash"]),
        )
