from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from willow_substrate.connections import find_connections
from willow_substrate.facts import FactLedger
from willow_substrate.reflection import meditate
from willow_substrate.research import ResearchLedger
from willow_substrate.store import EventStore


class ConnectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = EventStore(Path(self.temp.name))

    def tearDown(self):
        self.temp.cleanup()

    def test_shape_finds_connection_when_words_do_not(self):
        source = self.store.append(
            "The flock vanishes when the birds land.",
            kind="message",
            session_id="field-notes",
        )
        meditation = meditate(
            self.store,
            "field-notes",
            text="Identity belongs to the relation rather than the member.",
            shapes=["motion:distributed-to-local", "identity:relational"],
        )
        research = ResearchLedger(self.store)
        commission = research.commission("Study temporary institutions")
        result = research.complete(
            commission.id,
            summary="A temporary institution exists only while participation continues.",
            writeup="Its vocabulary shares nothing with the bird observation.",
            provider="test",
            shapes=["identity:relational", "motion:distributed-to-local"],
        )

        matches = find_connections(
            self.store,
            seed_event_id=meditation.id,
        )
        match = next(item for item in matches if item.event.id == result.id)
        self.assertEqual(match.channels, ("shape",))
        self.assertIn("identity:relational", match.shared_shapes)
        self.assertNotIn(source.id, {item.event.id for item in matches})

    def test_words_remain_an_independent_connection_channel(self):
        first = self.store.append(
            "Recurrent connectome motifs may stabilize memory.",
            kind="meditation",
            session_id="one",
        )
        second = self.store.append(
            "The connectome study reports recurrent circuit motifs.",
            kind="research_result",
            session_id="two",
        )
        matches = find_connections(self.store, seed_event_id=first.id)
        match = next(item for item in matches if item.event.id == second.id)
        self.assertEqual(match.channels, ("words",))
        self.assertIn("connectome", match.shared_terms)

    def test_superseded_claim_cannot_seed_normal_connection_results(self):
        facts = FactLedger(self.store)
        claim = facts.add_claim(
            "The archive contains twelve notebooks",
            source="old index",
            shapes=["quantity:collection-size"],
        )
        correction = self.store.correct(
            claim.id,
            "The archive contains thirteen notebooks",
        )
        seed = self.store.append(
            "A collection changed size.",
            kind="meditation",
            metadata={"idea_shape": ["quantity:collection-size"]},
        )
        matches = find_connections(self.store, seed_event_id=seed.id)
        ids = {item.event.id for item in matches}
        self.assertNotIn(claim.id, ids)
        self.assertNotIn(correction.id, ids)


if __name__ == "__main__":
    unittest.main()
