from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from willow.facts import FactLedger
from willow.research import Citation, ResearchLedger
from willow.store import EventStore


class ResearchLedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = EventStore(Path(self.temp.name))
        self.research = ResearchLedger(self.store)

    def tearDown(self):
        self.temp.cleanup()

    def test_commission_is_safe_staged_by_default(self):
        event = self.research.commission(
            "What has changed in connectome reconstruction?"
        )
        state = self.research.states()[0]
        self.assertEqual(state.commission.id, event.id)
        self.assertEqual(state.status, "staged_for_review")
        self.assertTrue(event.metadata["approval_required"])

    def test_result_retains_citations_and_provenance(self):
        commission = self.research.commission(
            "Review evidence for recurrent circuit motifs",
            approval_required=False,
        )
        result = self.research.complete(
            commission.id,
            summary="Several studies report recurrent motifs.",
            writeup="The evidence is promising but method-dependent.",
            citations=[
                Citation(
                    title="Example paper",
                    location="https://example.test/paper",
                    excerpt="Recurrent motifs were measured.",
                )
            ],
            provider="test-provider",
        )

        state = self.research.states()[0]
        self.assertEqual(state.status, "done")
        self.assertEqual(state.result.id, result.id)
        self.assertEqual(result.derived_from, (commission.id,))
        self.assertEqual(result.metadata["citation_count"], 1)

    def test_research_result_can_provenance_a_fact(self):
        commission = self.research.commission("Count the corpus")
        result = self.research.complete(
            commission.id,
            summary="The manifest lists 42 papers.",
            writeup="Counted from the versioned manifest.",
            citations=[
                {
                    "title": "Corpus manifest",
                    "location": "local://corpus/manifest.json",
                }
            ],
            provider="local-manifest",
        )
        claim = FactLedger(self.store).add_claim(
            "The corpus contains 42 papers",
            ttl_days=7,
            source="Corpus manifest",
            derived_from=(result.id,),
        )
        self.assertEqual(claim.derived_from, (result.id,))

    def test_failure_does_not_erase_or_complete_commission(self):
        commission = self.research.commission("Check an unavailable archive")
        failure = self.research.fail(
            commission.id,
            error="Archive timed out",
            provider="test-provider",
        )
        state = self.research.states()[0]
        self.assertEqual(state.status, "error")
        self.assertEqual(state.failure.id, failure.id)
        self.assertIsNone(state.result)


if __name__ == "__main__":
    unittest.main()
