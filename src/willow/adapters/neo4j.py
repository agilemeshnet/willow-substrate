"""Neo4j / AuraDB graph projection of the local event ledger.

The willow-substrate event store is the source of truth. This adapter
mirrors it into a Neo4j (typically AuraDB) graph so federated agents can
share a query surface, audits can traverse relationships, and
graph-native queries can join the ledger with other knowledge.

Install with:

    pip install "willow-substrate[neo4j]"

Environment (or constructor args):

    NEO4J_URI       e.g. neo4j+s://<db-id>.databases.neo4j.io
    NEO4J_USER      e.g. neo4j
    NEO4J_PASSWORD  the password

Usage:

    from willow.store import EventStore
    from willow.adapters.neo4j import Neo4jGraphAdapter

    store = EventStore()
    with Neo4jGraphAdapter() as adapter:
        adapter.mirror_all(store)                    # one-shot full mirror
        adapter.mirror_event(store.append("...", ...))  # per-event

The graph shape:

- (:Event {id, timestamp, kind, content, hash}) one node per Willow event.
  MERGE on id, so re-runs are idempotent and never duplicate.
- (:Session {id}) grouping node; one per unique session_id.
- (:Actor {name}) actor node; one per unique actor.
- (:Event)-[:IN_SESSION]->(:Session)
- (:Event)-[:AUTHORED_BY]->(:Actor)
- (:Event)-[:SUPERSEDES]->(:Event) for corrections; the newer event
  points at the one it supersedes. The old one keeps its hash chain.
- (:Event)-[:DERIVES_FROM]->(:Event) for meditations, engrams, and
  any other event whose derived_from list is non-empty.

The adapter is deliberately narrow: it does not attempt to write the
graph BACK into the store. The local ledger stays the single writer.
This is the honest one-way pattern the additive law prefers.
"""
from __future__ import annotations

import os
from contextlib import AbstractContextManager
from typing import Iterable

# Guarded imports; installing [neo4j] adds these.
try:
    from neo4j import GraphDatabase
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "willow.adapters.neo4j requires the [neo4j] extra. "
        "Install with: pip install \"willow-substrate[neo4j]\""
    ) from exc

try:  # optional; adapter still works if dotenv missing
    from dotenv import load_dotenv

    _HAS_DOTENV = True
except ImportError:  # pragma: no cover
    _HAS_DOTENV = False


from willow.events import Event
from willow.store import EventStore


DEFAULT_MERGE_BATCH = 500


class Neo4jConfigError(RuntimeError):
    """Missing or invalid Neo4j connection parameters."""


class Neo4jGraphAdapter(AbstractContextManager):
    """Mirrors the local event ledger into a Neo4j / AuraDB graph.

    Configuration precedence:
    1. Explicit constructor arguments (uri, user, password).
    2. Environment variables (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD).
    3. .env file in the process working directory (if python-dotenv is
       importable; dotenv is a soft dep).
    """

    def __init__(
        self,
        *,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
        load_dotenv_file: bool = True,
    ):
        if load_dotenv_file and _HAS_DOTENV:
            load_dotenv()  # populate os.environ from .env if present
        self.uri = uri or os.environ.get("NEO4J_URI")
        self.user = user or os.environ.get("NEO4J_USER", "neo4j")
        self.password = password or os.environ.get("NEO4J_PASSWORD")
        self.database = database or os.environ.get("NEO4J_DATABASE") or None

        missing = [
            name for name, val in (
                ("NEO4J_URI", self.uri),
                ("NEO4J_PASSWORD", self.password),
            )
            if not val
        ]
        if missing:
            raise Neo4jConfigError(
                "Neo4jGraphAdapter is missing required config: "
                + ", ".join(missing)
                + ". Pass explicit args or set the env variables."
            )
        self._driver = GraphDatabase.driver(
            self.uri, auth=(self.user, self.password)
        )

    def __enter__(self) -> "Neo4jGraphAdapter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def _session(self):
        kwargs: dict[str, str] = {}
        if self.database:
            kwargs["database"] = self.database
        return self._driver.session(**kwargs)

    def ensure_constraints(self) -> None:
        """Idempotent: create the uniqueness constraints Willow expects."""
        with self._session() as session:
            for statement in (
                "CREATE CONSTRAINT willow_event_id IF NOT EXISTS "
                "FOR (e:Event) REQUIRE e.id IS UNIQUE",
                "CREATE CONSTRAINT willow_event_hash IF NOT EXISTS "
                "FOR (e:Event) REQUIRE e.hash IS UNIQUE",
                "CREATE CONSTRAINT willow_session_id IF NOT EXISTS "
                "FOR (s:Session) REQUIRE s.id IS UNIQUE",
                "CREATE CONSTRAINT willow_actor_name IF NOT EXISTS "
                "FOR (a:Actor) REQUIRE a.name IS UNIQUE",
            ):
                session.run(statement)

    def mirror_event(self, event: Event) -> None:
        """Mirror one event. Idempotent (MERGE on id)."""
        with self._session() as session:
            session.execute_write(_write_one_event, event)

    def mirror_events(self, events: Iterable[Event]) -> int:
        """Batch-mirror a stream of events; returns count actually written."""
        count = 0
        with self._session() as session:
            for event in events:
                session.execute_write(_write_one_event, event)
                count += 1
        return count

    def mirror_all(
        self,
        store: EventStore,
        *,
        batch: int = DEFAULT_MERGE_BATCH,
    ) -> int:
        """One-shot mirror of every active event in the store."""
        events = store.events(
            limit=100_000, active_only=False, include_expired=True
        )
        return self.mirror_events(events)


def _write_one_event(tx, event: Event) -> None:
    """Cypher transaction: MERGE the event and its related actor/session,
    plus any SUPERSEDES / DERIVES_FROM edges the event carries."""
    tx.run(
        """
        MERGE (e:Event {id: $id})
        SET e.timestamp = $timestamp,
            e.kind = $kind,
            e.content = $content,
            e.hash = $hash,
            e.seq = $seq
        MERGE (s:Session {id: $session_id})
        MERGE (a:Actor {name: $actor})
        MERGE (e)-[:IN_SESSION]->(s)
        MERGE (e)-[:AUTHORED_BY]->(a)
        """,
        id=event.id,
        timestamp=event.timestamp,
        kind=event.kind,
        content=event.content,
        hash=event.hash,
        seq=event.seq,
        session_id=event.session_id,
        actor=event.actor,
    )
    if event.supersedes:
        tx.run(
            """
            MATCH (newer:Event {id: $newer_id})
            MERGE (older:Event {id: $older_id})
            MERGE (newer)-[:SUPERSEDES]->(older)
            """,
            newer_id=event.id,
            older_id=event.supersedes,
        )
    for source_id in event.derived_from or ():
        tx.run(
            """
            MATCH (derived:Event {id: $derived_id})
            MERGE (source:Event {id: $source_id})
            MERGE (derived)-[:DERIVES_FROM]->(source)
            """,
            derived_id=event.id,
            source_id=source_id,
        )
