from __future__ import annotations

import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from willow.store import EventStore


class EventStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.home = Path(self.temp.name)
        self.store = EventStore(self.home)

    def tearDown(self):
        self.temp.cleanup()

    def test_two_store_instances_share_events(self):
        first = self.store.append(
            "Research started in terminal A",
            actor="peter",
            session_id="terminal-a",
        )
        second_store = EventStore(self.home)
        found = second_store.search("research terminal", limit=5)
        self.assertEqual(found[0].event.id, first.id)

    def test_correction_preserves_original_but_filters_it(self):
        original = self.store.append(
            "The specimen has twelve segments",
            actor="willow",
            session_id="lab",
        )
        correction = self.store.correct(
            original.id,
            "Correction: the specimen has thirteen segments",
            actor="peter",
            session_id="lab",
        )

        active_ids = {event.id for event in self.store.events(limit=20)}
        historical_ids = {
            event.id
            for event in self.store.events(limit=20, active_only=False)
        }
        self.assertNotIn(original.id, active_ids)
        self.assertIn(correction.id, active_ids)
        self.assertIn(original.id, historical_ids)
        self.assertIn(correction.id, historical_ids)

    def test_database_rejects_update_and_delete(self):
        event = self.store.append("Immutable observation")
        conn = self.store._connect()
        try:
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "UPDATE events SET content = 'changed' WHERE id = ?",
                    (event.id,),
                )
            conn.rollback()
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute("DELETE FROM events WHERE id = ?", (event.id,))
        finally:
            conn.close()

    def test_concurrent_writers_keep_one_valid_chain(self):
        def append(index: int):
            EventStore(self.home).append(
                f"Concurrent event {index}",
                actor=f"worker-{index % 3}",
                session_id=f"terminal-{index % 4}",
            )

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(append, range(32)))

        valid, count, error = self.store.verify()
        self.assertTrue(valid, error)
        self.assertEqual(count, 32)

    def test_derivation_requires_existing_sources(self):
        with self.assertRaises(KeyError):
            self.store.append(
                "Derived from something absent",
                kind="meditation",
                derived_from=["evt-missing"],
            )


if __name__ == "__main__":
    unittest.main()
