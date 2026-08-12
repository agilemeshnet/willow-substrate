"""Tests for the Context Window Builder + Prosoche freshness monitor.

Pins the layered assembly semantics (standing / foreground / vista /
wave / prosoche), the salience ranking, the additive-law behaviour
(no events modified), and the prosoche freshness bands.
"""
from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from willow_substrate.cwb import ContextWindow, ContextWindowBuilder
from willow_substrate.prosoche import ProsocheMonitor, SourceBand
from willow_substrate.store import EventStore
from willow_substrate.vista import VistaBackend


class ProsocheMonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp.name) / "prosoche.db"
        self.monitor = ProsocheMonitor(self.db_path)

    def tearDown(self):
        self.temp.cleanup()

    def test_unknown_band_when_never_touched(self):
        self.monitor.register("cache", expected_interval_s=3600.0)
        bands = self.monitor.bands()
        self.assertEqual(len(bands), 1)
        self.assertEqual(bands[0].name, "cache")
        self.assertEqual(bands[0].band, ProsocheMonitor.BAND_UNKNOWN)
        self.assertIsNone(bands[0].last_touched)

    def test_bands_progress_fresh_amber_stale_dead(self):
        # Register a source expected every 60s. Touch it, then
        # inspect bands with a moving 'now'.
        self.monitor.register("cache", expected_interval_s=60.0)
        touched = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        self.monitor.touch("cache", at=touched)

        fresh = self.monitor.bands(now=touched + timedelta(seconds=30))
        amber = self.monitor.bands(now=touched + timedelta(seconds=100))
        stale = self.monitor.bands(now=touched + timedelta(seconds=200))
        dead = self.monitor.bands(now=touched + timedelta(seconds=1000))

        self.assertEqual(fresh[0].band, ProsocheMonitor.BAND_FRESH)
        self.assertEqual(amber[0].band, ProsocheMonitor.BAND_AMBER)
        self.assertEqual(stale[0].band, ProsocheMonitor.BAND_STALE)
        self.assertEqual(dead[0].band, ProsocheMonitor.BAND_DEAD)

    def test_register_rejects_bad_multipliers(self):
        with self.assertRaises(ValueError):
            self.monitor.register(
                "x", expected_interval_s=60.0,
                amber_multiplier=5.0,
                stale_multiplier=3.0,  # smaller than amber
                dead_multiplier=10.0,
            )

    def test_register_rejects_non_positive_interval(self):
        with self.assertRaises(ValueError):
            self.monitor.register("x", expected_interval_s=0.0)

    def test_touch_is_idempotent_at_same_instant(self):
        self.monitor.register("cache", expected_interval_s=60.0)
        at = datetime(2024, 6, 1, tzinfo=timezone.utc)
        self.monitor.touch("cache", at=at)
        self.monitor.touch("cache", at=at)  # must not raise
        bands = self.monitor.bands(now=at + timedelta(seconds=1))
        self.assertEqual(bands[0].band, ProsocheMonitor.BAND_FRESH)


class ContextWindowBuilderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.home = Path(self.temp.name)
        self.store = EventStore(self.home)
        # Three standing events (identity + rules); five ordinary
        # events with varying salience.
        self.store.append(
            "IDENTITY: I am Peter and this is my ledger",
            actor="peter", kind="identity", session_id="standing",
            metadata={"standing": True, "salience": 100.0},
        )
        self.store.append(
            "RULE: additive only, no deletion",
            actor="peter", kind="rule", session_id="standing",
            metadata={"standing": True, "salience": 100.0},
        )
        # Five ordinary events; salience 10, 20, 30, 5, 15.
        for i, sal in enumerate([10, 20, 30, 5, 15]):
            self.store.append(
                f"Ordinary event {i} about cats and databases",
                actor="peter", kind="observation", session_id="daily",
                metadata={"salience": float(sal)},
            )
        self.backend = VistaBackend(self.store)
        self.cwb = ContextWindowBuilder(
            self.store, retrieval_backend=self.backend
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_standing_layer_contains_only_pinned_events(self):
        window = self.cwb.build("cats", foreground_k=3)
        # Two standing events registered above.
        self.assertEqual(len(window.standing), 2)
        for event in window.standing:
            self.assertTrue(event.metadata.get("standing"))

    def test_foreground_ranked_by_salience_descending(self):
        window = self.cwb.build("cats", foreground_k=3, include_standing=False)
        # Top 3 by salience among the ordinary events would be sal=30, 20, 15.
        # BUT the standing events also carry salience=100; they'll dominate
        # foreground unless deliberately filtered. Our current spec: standing
        # events ARE eligible for foreground ranking too (they're just also
        # pinned). Verify the top score is the 100-salience event.
        salience_values = [
            float(event.metadata.get("salience", 0.0))
            for event in window.foreground
        ]
        self.assertEqual(salience_values, sorted(salience_values, reverse=True))
        self.assertEqual(salience_values[0], 100.0)

    def test_vista_layer_populated_with_evidence(self):
        window = self.cwb.build("cats and databases", foreground_k=3)
        self.assertIsNotNone(window.vista)
        self.assertGreater(len(window.vista.evidence), 0)

    def test_wave_layer_included_by_default_can_be_skipped(self):
        window_with = self.cwb.build("cats", foreground_k=3)
        window_without = self.cwb.build(
            "cats", foreground_k=3, include_wave=False
        )
        self.assertIsNotNone(window_with.wave)
        self.assertIsNone(window_without.wave)

    def test_prosoche_field_empty_when_no_monitor_configured(self):
        window = self.cwb.build("cats", foreground_k=3)
        self.assertEqual(window.prosoche, ())

    def test_prosoche_field_populated_when_monitor_present(self):
        monitor = ProsocheMonitor(self.home / "prosoche.db")
        monitor.register("cache", expected_interval_s=60.0)
        cwb_with = ContextWindowBuilder(
            self.store,
            retrieval_backend=self.backend,
            prosoche=monitor,
        )
        window = cwb_with.build("cats", foreground_k=3)
        self.assertEqual(len(window.prosoche), 1)
        self.assertEqual(window.prosoche[0].name, "cache")

    def test_all_event_ids_dedupes_across_layers(self):
        window = self.cwb.build("cats", foreground_k=3)
        ids = window.all_event_ids
        self.assertEqual(len(ids), len(set(ids)))

    def test_trace_records_each_layer(self):
        window = self.cwb.build("cats", foreground_k=3)
        joined = " | ".join(window.trace)
        for word in ("banks", "standing", "foreground", "vista", "wave"):
            self.assertIn(word, joined)

    def test_banks_field_populated_from_home_files(self):
        """Cohesion: ContextWindow surfaces banks/*.md alongside the four
        event layers so consumers see the same constitutional floor
        ContextBuilder.boot() delivers."""
        (self.home / "IDENTITY.md").write_text(
            "I am Peter's ledger.", encoding="utf-8"
        )
        (self.home / "GROUND.md").write_text(
            "Additive only.", encoding="utf-8"
        )
        window = self.cwb.build("cats", foreground_k=3)
        self.assertEqual(
            [bank.name for bank in window.banks], ["identity", "ground"]
        )

    def test_banks_field_empty_without_files(self):
        window = self.cwb.build("cats", foreground_k=3)
        self.assertEqual(window.banks, ())

    def test_foreground_uses_five_signal_scorer_when_metadata_salience_absent(self):
        """Cohesion: even without metadata.salience floats, the CWB
        ranks foreground by score_events (which combines recency +
        standing + citation + reflection + query). Standing material
        should be at the top regardless of query."""
        # Fresh CWB with events that carry NO salience metadata, only
        # a standing flag on one of them.
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            store = EventStore(home)
            store.append(
                "Ordinary A", actor="peter",
                kind="observation", session_id="s",
            )
            store.append(
                "Ordinary B", actor="peter",
                kind="observation", session_id="s",
            )
            standing = store.append(
                "Rule: measure first",
                actor="peter", kind="observation", session_id="s",
                metadata={"standing": True},
            )
            cwb = ContextWindowBuilder(
                store,
                retrieval_backend=VistaBackend(store),
            )
            window = cwb.build("unrelated query", foreground_k=3)
            # The standing event outranks the two ordinaries on the
            # multi-signal scorer even though the query does not match it.
            self.assertEqual(window.foreground[0].id, standing.id)

    def test_ledger_untouched_after_many_builds(self):
        before = [
            e.hash for e in self.store.events(limit=100, active_only=True)
        ]
        for _ in range(5):
            self.cwb.build("cats", foreground_k=3)
        after = [
            e.hash for e in self.store.events(limit=100, active_only=True)
        ]
        self.assertEqual(before, after)

    def test_foreground_k_capped_on_small_corpora_leaves_room_for_vista(self):
        """On a tiny store (5 events), foreground_k=15 would seed every
        event and give vista nothing distinct to return. The cap keeps
        foreground <= corpus_size // 3 so vista/wave stay useful.
        Regression against the small-corpus degeneracy caught in the
        end-to-end tryout."""
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            store = EventStore(home)
            for i in range(5):
                store.append(
                    f"Event {i} about connectomes and databases",
                    actor="peter", kind="observation", session_id="s",
                )
            cwb = ContextWindowBuilder(
                store, retrieval_backend=VistaBackend(store)
            )
            # Ask for 15 foreground; corpus has 5. Cap should give
            # max(1, 5 // 3) = 1 foreground event, leaving vista candidates.
            window = cwb.build("connectome", foreground_k=15)
            self.assertLessEqual(len(window.foreground), max(1, 5 // 3))
            # Vista must return SOMETHING now that seeds don't cover the corpus.
            self.assertIsNotNone(window.vista)
        valid, count, error = self.store.verify()
        self.assertTrue(valid, error)


if __name__ == "__main__":
    unittest.main()
