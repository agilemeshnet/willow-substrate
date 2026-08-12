"""Prosoche: freshness monitoring for named data sources.

Ported from Scout's cwb.py: every source that feeds a context window
carries an expected refresh interval; the monitor stamps a band on
each source (fresh / amber / stale / dead) based on how long since its
last touch. A stale bundle looks EXACTLY like a fresh one; this module
is how we tell the difference.

Sources live in a sidecar SQLite file (``prosoche.db``, separate from
the event ledger so the ledger stays hash-chained). Registration is
declarative: caller names the source and the expected refresh interval;
``touch()`` records a heartbeat; ``bands()`` returns the current colour
per source.

Additive-law honoring: no events are modified; ``prosoche.db`` can be
deleted without touching the ledger. The whole thing is idempotent.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    name TEXT PRIMARY KEY,
    expected_interval_s REAL NOT NULL,
    amber_multiplier REAL NOT NULL DEFAULT 1.5,
    stale_multiplier REAL NOT NULL DEFAULT 3.0,
    dead_multiplier REAL NOT NULL DEFAULT 10.0
);
CREATE TABLE IF NOT EXISTS touches (
    source TEXT NOT NULL,
    touched_at TEXT NOT NULL,
    PRIMARY KEY (source, touched_at)
);
CREATE INDEX IF NOT EXISTS idx_touches_source ON touches(source);
"""


@dataclass(frozen=True)
class SourceBand:
    """One source's current freshness reading."""

    name: str
    band: str  # 'fresh' | 'amber' | 'stale' | 'dead' | 'unknown'
    last_touched: datetime | None
    seconds_since: float | None


class ProsocheMonitor:
    """Freshness tracker for named data sources.

    Not thread-safe. Uses UTC-aware timestamps throughout, matching the
    ledger's convention.
    """

    BAND_FRESH = "fresh"
    BAND_AMBER = "amber"
    BAND_STALE = "stale"
    BAND_DEAD = "dead"
    BAND_UNKNOWN = "unknown"

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

    def register(
        self,
        name: str,
        *,
        expected_interval_s: float,
        amber_multiplier: float = 1.5,
        stale_multiplier: float = 3.0,
        dead_multiplier: float = 10.0,
    ) -> None:
        """Register (or update) a source.

        Bands fire when time-since-touch exceeds ``expected_interval_s *
        multiplier``: amber at 1.5x, stale at 3x, dead at 10x by default.
        """
        if expected_interval_s <= 0:
            raise ValueError("expected_interval_s must be positive")
        if not (amber_multiplier < stale_multiplier < dead_multiplier):
            raise ValueError(
                "multipliers must satisfy amber < stale < dead"
            )
        conn = self._connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO sources "
                "(name, expected_interval_s, amber_multiplier, "
                " stale_multiplier, dead_multiplier) VALUES (?, ?, ?, ?, ?)",
                (
                    name, expected_interval_s,
                    amber_multiplier, stale_multiplier, dead_multiplier,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def touch(self, name: str, *, at: datetime | None = None) -> None:
        """Record a heartbeat for a source.

        Idempotent: repeated touches at the same instant don't error.
        """
        ts = (at or datetime.now(timezone.utc)).isoformat()
        conn = self._connect()
        try:
            conn.execute(
                "INSERT OR IGNORE INTO touches (source, touched_at) "
                "VALUES (?, ?)",
                (name, ts),
            )
            conn.commit()
        finally:
            conn.close()

    def bands(
        self, *, now: datetime | None = None
    ) -> tuple[SourceBand, ...]:
        """Current band for every registered source, in name order."""
        current = now or datetime.now(timezone.utc)
        conn = self._connect()
        try:
            source_rows = conn.execute(
                "SELECT name, expected_interval_s, amber_multiplier, "
                "  stale_multiplier, dead_multiplier "
                "FROM sources ORDER BY name ASC"
            ).fetchall()
            results: list[SourceBand] = []
            for name, expected, amber_m, stale_m, dead_m in source_rows:
                last = conn.execute(
                    "SELECT MAX(touched_at) FROM touches WHERE source = ?",
                    (name,),
                ).fetchone()[0]
                if last is None:
                    results.append(
                        SourceBand(
                            name=name,
                            band=self.BAND_UNKNOWN,
                            last_touched=None,
                            seconds_since=None,
                        )
                    )
                    continue
                last_touched = datetime.fromisoformat(last)
                seconds_since = (current - last_touched).total_seconds()
                results.append(
                    SourceBand(
                        name=name,
                        band=self._band_for(
                            seconds_since,
                            expected,
                            amber_m,
                            stale_m,
                            dead_m,
                        ),
                        last_touched=last_touched,
                        seconds_since=seconds_since,
                    )
                )
            return tuple(results)
        finally:
            conn.close()

    @staticmethod
    def _band_for(
        seconds_since: float,
        expected: float,
        amber_m: float,
        stale_m: float,
        dead_m: float,
    ) -> str:
        """Green until amber_m x expected; amber until stale_m; stale
        until dead_m; dead beyond that."""
        if seconds_since >= expected * dead_m:
            return ProsocheMonitor.BAND_DEAD
        if seconds_since >= expected * stale_m:
            return ProsocheMonitor.BAND_STALE
        if seconds_since >= expected * amber_m:
            return ProsocheMonitor.BAND_AMBER
        return ProsocheMonitor.BAND_FRESH
