from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from willow_substrate.connections import find_connections
from willow_substrate.samples import evaluate_temporal_sample, load_temporal_sample
from willow_substrate.store import EventStore


SAMPLE = Path(__file__).parents[1] / "examples" / "temporal-bird-study.json"


class TemporalSampleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = EventStore(Path(self.temp.name))

    def tearDown(self):
        self.temp.cleanup()

    def test_sample_replays_idempotently_and_passes_declared_ground_truth(self):
        first = load_temporal_sample(self.store, SAMPLE)
        second = load_temporal_sample(self.store, SAMPLE)

        self.assertEqual(first.created, 9)
        self.assertEqual(first.reused, 0)
        self.assertEqual(second.created, 0)
        self.assertEqual(second.reused, 9)
        self.assertEqual(self.store.count(), 9)

        evaluation = evaluate_temporal_sample(self.store, SAMPLE)
        failures = [check.detail for check in evaluation.checks if not check.passed]
        self.assertTrue(evaluation.passed, failures)

    def test_correction_shape_false_friend_and_privacy_are_separate_signals(self):
        report = load_temporal_sample(self.store, SAMPLE)
        old_count = report.events["week-01-initial-count"]
        corrected = report.events["week-04-corrected-count"]
        active_ids = {
            event.id
            for event in self.store.events(
                limit=20,
                active_only=True,
                include_expired=True,
            )
        }
        self.assertNotIn(old_count.id, active_ids)
        self.assertIn(corrected.id, active_ids)

        predation = report.events["week-03-predation-pressure"]
        construction = report.events["week-02-construction-pressure"]
        shape_matches = find_connections(
            self.store,
            seed_event_id=predation.id,
            limit=20,
        )
        shape_match = next(
            item for item in shape_matches if item.event.id == construction.id
        )
        self.assertEqual(shape_match.channels, ("shape",))

        false_friend = report.events["week-05-lexical-false-friend"]
        lexical_matches = find_connections(
            self.store,
            seed_event_id=construction.id,
            limit=20,
        )
        lexical_match = next(
            item for item in lexical_matches if item.event.id == false_friend.id
        )
        self.assertEqual(lexical_match.channels, ("words",))
        self.assertEqual(lexical_match.shared_shapes, ())

        all_text = " ".join(
            event.content.lower()
            for event in self.store.events(
                limit=20,
                active_only=False,
                include_expired=True,
            )
        )
        self.assertNotIn("latitude", all_text)
        self.assertNotIn("longitude", all_text)
        self.assertIn("intentionally withheld", all_text)

    def test_changed_sample_cannot_reuse_an_idempotency_key(self):
        load_temporal_sample(self.store, SAMPLE)
        changed = json.loads(SAMPLE.read_text(encoding="utf-8"))
        changed["events"][0]["metadata"]["evidence_class"] = "silently-changed"
        changed_path = Path(self.temp.name) / "changed.json"
        changed_path.write_text(json.dumps(changed), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "idempotency key"):
            load_temporal_sample(self.store, changed_path)


if __name__ == "__main__":
    unittest.main()
