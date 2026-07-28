from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from willow.facts import FactLedger
from willow.store import EventStore


class FactLedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = EventStore(Path(self.temp.name))
        self.facts = FactLedger(self.store)
        self.now = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)

    def tearDown(self):
        self.temp.cleanup()

    def test_claim_becomes_due_on_its_own_schedule(self):
        claim = self.facts.add_claim(
            "The research corpus contains 42 papers",
            ttl_days=7,
            source="corpus manifest",
            checked_at=self.now,
        )

        self.assertEqual(self.facts.due(now=self.now), [])
        due = self.facts.due(now=self.now + timedelta(days=8))
        self.assertEqual([state.claim.id for state in due], [claim.id])

    def test_unknown_never_refreshes_last_verified(self):
        claim = self.facts.add_claim(
            "A newly reported result remains reproducible",
            ttl_days=1,
            source="paper",
            checked_at=self.now,
        )
        result = self.facts.record_check(
            claim.id,
            outcome="unknown",
            evidence_kind="none",
            retry_days=2,
            checked_at=self.now + timedelta(days=1),
        )

        state = self.facts.states()[0]
        self.assertEqual(state.status, "unknown")
        self.assertEqual(
            state.last_verified_at,
            self.now.isoformat(),
        )
        self.assertEqual(
            state.next_check_at,
            (self.now + timedelta(days=3)).isoformat(),
        )
        self.assertNotIn("verified_at", result.check.metadata)

    def test_model_memory_cannot_confirm_a_fact(self):
        claim = self.facts.add_claim(
            "The laboratory changed its protocol",
            source="research note",
            checked_at=self.now,
        )
        with self.assertRaisesRegex(ValueError, "model-only memory"):
            self.facts.record_check(
                claim.id,
                outcome="confirmed",
                evidence="The model remembers seeing this.",
                source="local model",
                evidence_kind="model",
                checked_at=self.now + timedelta(days=180),
            )

    def test_update_appends_replacement_and_preserves_history(self):
        original = self.facts.add_claim(
            "The project has twelve samples",
            ttl_days=30,
            source="lab notebook",
            checked_at=self.now,
        )
        result = self.facts.record_check(
            original.id,
            outcome="updated",
            replacement="The project has thirteen samples",
            evidence="Sample 13 appears in the signed manifest.",
            source="signed sample manifest",
            evidence_kind="primary",
            checked_at=self.now + timedelta(days=30),
        )

        self.assertIsNotNone(result.replacement)
        active = self.facts.states()
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].claim.content, "The project has thirteen samples")
        historical = self.store.events(
            limit=20,
            kind="claim",
            active_only=False,
            include_expired=True,
        )
        self.assertEqual(len(historical), 2)
        self.assertEqual(result.replacement.supersedes, original.id)
        self.assertIn(original.id, result.check.derived_from)
        self.assertIn(result.replacement.id, result.check.derived_from)

    def test_contradiction_filters_claim_without_deleting_it(self):
        claim = self.facts.add_claim(
            "The archive opens on Sundays",
            source="old visitor page",
            checked_at=self.now,
        )
        self.facts.record_check(
            claim.id,
            outcome="contradicted",
            evidence="The current official hours say Sunday closed.",
            source="official opening-hours page",
            evidence_kind="primary",
            checked_at=self.now + timedelta(days=2),
        )

        self.assertEqual(self.facts.states(), [])
        historical_ids = {
            event.id
            for event in self.store.events(
                limit=20,
                active_only=False,
                include_expired=True,
            )
        }
        self.assertIn(claim.id, historical_ids)


if __name__ == "__main__":
    unittest.main()
