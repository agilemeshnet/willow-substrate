"""Recall-frequency sidecar table.

Immutable events live in ``events.db``. Recall counts are, by design,
mutable, so they live in a separate SQLite file at ``store.home /
recall_stats.db``. Keeping them out of the hash-chained ledger is a
feature: the ledger stays tamper-evident; recall statistics stay
freely updatable per query.

Every call to ``record_recall(event_id)`` writes one row to a
``recalls`` table with the retrieval timestamp. The full recall history
per event is preserved, so downstream consumers can compute
consolidation curves exactly rather than approximating from a single
counter.

Ported from Hou, Tamoto & Miyashita (CHI EA '24, arXiv 2404.00573):
"My agent understands me better: Integrating Dynamic Human-like
Memory Recall and Consolidation in LLM-Based Agents". The paper
tracks recall history to adjust the exponential-decay rate a = 1/g_n
where g_n = g_{n-1} + S(t), S(t) = (1 - e^-t) / (1 + e^-t). This
module stores what a downstream scorer needs to compute g_n.

Additive-law honoring: no events are modified; ``recall_stats.db`` is
a distinct file that can be deleted without touching the ledger.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


SCHEMA = """
CREATE TABLE IF NOT EXISTS recalls (
    event_id TEXT NOT NULL,
    recalled_at TEXT NOT NULL,
    PRIMARY KEY (event_id, recalled_at)
);
CREATE INDEX IF NOT EXISTS idx_recalls_event ON recalls(event_id);
"""


class RecallStats:
    """Small sidecar for recall-frequency tracking.

    Not thread-safe (the LoCoMo runner is single-threaded per
    conversation; multi-threaded call sites should wrap access). Uses
    the same UTC-aware timestamp convention as the main ledger.
    """

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _ensure_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            conn.executescript(SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def record_recall(
        self,
        event_id: str,
        *,
        at: datetime | None = None,
    ) -> None:
        """Record one recall of one event."""
        ts = (at or datetime.now(timezone.utc)).isoformat()
        conn = self._connect()
        try:
            conn.execute(
                "INSERT OR IGNORE INTO recalls (event_id, recalled_at) "
                "VALUES (?, ?)",
                (event_id, ts),
            )
            conn.commit()
        finally:
            conn.close()

    def record_recalls(
        self,
        event_ids: Iterable[str],
        *,
        at: datetime | None = None,
    ) -> None:
        """Batch of recalls sharing one timestamp (one retrieval call)."""
        ts = (at or datetime.now(timezone.utc)).isoformat()
        rows = [(eid, ts) for eid in event_ids]
        if not rows:
            return
        conn = self._connect()
        try:
            conn.executemany(
                "INSERT OR IGNORE INTO recalls (event_id, recalled_at) "
                "VALUES (?, ?)",
                rows,
            )
            conn.commit()
        finally:
            conn.close()

    def history(self, event_id: str) -> list[datetime]:
        """Ordered ascending list of recall timestamps for one event."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT recalled_at FROM recalls WHERE event_id = ? "
                "ORDER BY recalled_at ASC",
                (event_id,),
            ).fetchall()
        finally:
            conn.close()
        return [datetime.fromisoformat(row[0]) for row in rows]

    def count(self, event_id: str) -> int:
        """Number of times this event has been recalled."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM recalls WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        finally:
            conn.close()
        return int(row[0]) if row else 0
