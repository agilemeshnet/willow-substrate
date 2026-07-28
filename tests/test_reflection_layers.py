from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from willow.dreaming import dream
from willow.engrams import crystallize_retroactive_engrams
from willow.store import EventStore


class ReflectionLayerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.home = Path(self.temp.name)
        self.store = EventStore(self.home)

    def tearDown(self):
        self.temp.cleanup()

    def test_dreams_use_active_evidence_and_are_idempotent(self):
        earlier = self.store.append(
            "Drosophila connectome recurrent motif stabilises navigation",
            actor="peter",
            session_id="terminal-a",
        )
        incorrect = self.store.append(
            "Drosophila connectome recurrent motif predicts weather",
            actor="willow",
            session_id="terminal-b",
        )
        self.store.correct(
            incorrect.id,
            "Correction: that weather interpretation was invalid",
            actor="peter",
            session_id="terminal-b",
        )
        later = self.store.append(
            "Recurrent connectome motif may stabilise memory navigation",
            actor="willow",
            session_id="terminal-c",
        )

        first = dream(self.store, query="connectome recurrent motif", limit=2)
        second = dream(self.store, query="connectome recurrent motif", limit=2)

        self.assertEqual(len(first), 1)
        self.assertEqual(second, [])
        self.assertEqual(set(first[0].derived_from), {earlier.id, later.id})
        self.assertNotIn(incorrect.id, first[0].derived_from)
        self.assertEqual(
            first[0].metadata["epistemic_status"],
            "associative-proposal",
        )

    def test_retroactive_importance_advances_without_mutation(self):
        source = self.store.append(
            "A correction should remain visible as historical evidence",
            actor="peter",
            session_id="research",
        )
        first_reflection = self.store.append(
            "Later meditation returned to historical correction evidence",
            actor="willow",
            kind="meditation",
            session_id="reflection",
            derived_from=(source.id,),
        )
        first = crystallize_retroactive_engrams(self.store, limit=1)
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0].metadata["surprise_weight"], 1)

        second_reflection = self.store.append(
            "A dream again revisited historical correction evidence",
            actor="willow",
            kind="dream",
            session_id="dreams",
            derived_from=(source.id,),
        )
        second = crystallize_retroactive_engrams(self.store, limit=1)
        self.assertEqual(len(second), 1)
        self.assertEqual(second[0].metadata["surprise_weight"], 2)
        self.assertEqual(second[0].supersedes, first[0].id)
        self.assertIn(first_reflection.id, second[0].derived_from)
        self.assertIn(second_reflection.id, second[0].derived_from)

        active_engrams = self.store.events(
            limit=20,
            kind="engram",
            active_only=True,
        )
        self.assertEqual([event.id for event in active_engrams], [second[0].id])
        historical_engrams = self.store.events(
            limit=20,
            kind="engram",
            active_only=False,
            include_expired=True,
        )
        self.assertEqual(len(historical_engrams), 2)


if __name__ == "__main__":
    unittest.main()
