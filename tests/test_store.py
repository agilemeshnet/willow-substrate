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

    def test_append_rejects_non_iso_timestamp(self):
        """append() must reject caller-supplied timestamps that are not
        ISO-8601 parseable. The timestamp gets hashed into the event payload
        and the sample suite's temporal claims rest on the field; nothing
        else validates it."""
        with self.assertRaises(ValueError) as ctx:
            self.store.append("a", session_id="s", timestamp="banana")
        self.assertIn("iso-8601", str(ctx.exception).lower())

        # Empty string is not a valid timestamp either.
        with self.assertRaises(ValueError):
            self.store.append("b", session_id="s", timestamp="   ")

    def test_append_accepts_valid_iso_timestamps(self):
        """Legitimate ISO-8601 forms (with and without timezone, with Z, with
        offset) must still succeed and preserve the exact string that was
        provided so its hash is not disturbed."""
        for ts in [
            "2026-08-07T15:30:00+00:00",
            "2026-08-07T15:30:00Z",
            "2026-08-07T15:30:00",
            "2026-08-07 15:30:00",
        ]:
            event = self.store.append(
                f"observation at {ts}", session_id="s", timestamp=ts
            )
            self.assertEqual(
                event.timestamp, ts,
                "validator must preserve the caller's exact string",
            )

    def test_append_still_auto_generates_when_no_timestamp_given(self):
        """The unchanged code path: timestamp=None still produces UTC now."""
        event = self.store.append("no timestamp given", session_id="s")
        # Round-trip parseable; presence of tz or 'T' is what matters here.
        from datetime import datetime as _dt
        _dt.fromisoformat(event.timestamp)  # must not raise

    def test_fts_suppression_is_detected_by_verify(self):
        """events_fts has no immutability trigger. A DELETE against it leaves
        the ledger honest but the search index censored, and the two live in
        different tables. verify() has to reconcile the two so silent
        unfindability cannot pass as a green integrity check."""
        first = self.store.append("headache begins", session_id="s")
        second = self.store.append("migraine worsens", session_id="s")
        third = self.store.append("nausea joins in", session_id="s")

        # Deleting from events_fts requires no privileged access; there is no
        # trigger to bypass. This is the exact attack the reviewer described.
        conn = sqlite3.connect(self.store.db_path)
        try:
            conn.execute(
                "DELETE FROM events_fts WHERE event_id = ?", (second.id,)
            )
            conn.commit()
        finally:
            conn.close()

        # Search silently drops the middle event; that is the harm surface.
        search_ids = {hit.event.id for hit in self.store.search("migraine", limit=10)}
        self.assertNotIn(
            second.id, search_ids,
            "the audit's scenario: the deleted event is now unfindable",
        )

        # And verify() must call it out.
        valid, count, error = self.store.verify()
        self.assertFalse(
            valid,
            "verify() must catch FTS suppression; without this the ledger "
            "reads honest while retrieval is censored",
        )
        self.assertEqual(count, 3, "count reflects the actual ledger rows")
        self.assertIn("search index", (error or "").lower())

    def test_tail_truncation_is_detected_via_anchored_head(self):
        """A DELETE of the highest-seq row leaves a shorter but self-consistent
        chain that a plain hash walk cannot flag. The anchored-head sentinels
        in willow_meta (head_hash, event_count) close that gap."""
        self.store.append("first", session_id="s")
        self.store.append("second", session_id="s")
        third = self.store.append("third", session_id="s")

        # Drop the immutability trigger to simulate an attacker with local
        # write access (the same threat model the trigger targets).
        conn = sqlite3.connect(self.store.db_path)
        try:
            conn.execute("DROP TRIGGER events_are_immutable_delete")
            conn.execute("DELETE FROM events WHERE id = ?", (third.id,))
            conn.commit()
        finally:
            conn.close()

        valid, count, error = self.store.verify()
        self.assertFalse(
            valid, "tail truncation must be detected by verify()"
        )
        self.assertEqual(count, 2, "verify reports the actual remaining rows")
        self.assertIn("tail", (error or "").lower())

    def test_anchored_head_tracks_appends(self):
        """head_hash and event_count in willow_meta advance on every append."""
        conn = sqlite3.connect(self.store.db_path)
        try:
            row = conn.execute(
                "SELECT value FROM willow_meta WHERE key = 'event_count'"
            ).fetchone()
            self.assertEqual(row[0], "0")
        finally:
            conn.close()

        first = self.store.append("a", session_id="s")
        second = self.store.append("b", session_id="s")

        conn = sqlite3.connect(self.store.db_path)
        try:
            meta = {
                r[0]: r[1]
                for r in conn.execute("SELECT key, value FROM willow_meta")
            }
        finally:
            conn.close()

        self.assertEqual(meta["event_count"], "2")
        self.assertEqual(meta["head_hash"], second.hash)
        # Verify still passes for the honest history.
        valid, count, error = self.store.verify()
        self.assertTrue(valid, error)
        self.assertEqual(count, 2)


if __name__ == "__main__":
    unittest.main()
