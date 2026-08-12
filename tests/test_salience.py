from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from willow_substrate.context import ContextBuilder
from willow_substrate.salience import (
    rank_selection,
    score_event,
    score_events,
    wikilink_in_degrees,
)
from willow_substrate.store import EventStore


class SalienceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.home = Path(self.temp.name)
        self.store = EventStore(self.home)
        self.now = datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc)

    def tearDown(self):
        self.temp.cleanup()

    def _at(self, days_ago: float) -> str:
        return (self.now - timedelta(days=days_ago)).isoformat()

    def test_standing_material_outranks_fresh_chatter(self):
        standing = self.store.append(
            "Measure before changing anything.",
            kind="note",
            metadata={"standing": True, "name": "measure-first"},
            timestamp=self._at(180),
        )
        chatter = self.store.append(
            "Opened the terminal.",
            timestamp=self._at(0),
        )
        scores = score_events([standing, chatter], now=self.now)
        self.assertGreater(
            scores[standing.id].total,
            scores[chatter.id].total,
        )

    def test_recency_decays_rather_than_cutting_off(self):
        fresh = self.store.append("Fresh work", timestamp=self._at(0))
        older = self.store.append("Older work", timestamp=self._at(28))
        fresh_score = score_event(fresh, now=self.now)
        older_score = score_event(older, now=self.now)
        self.assertGreater(fresh_score.total, older_score.total)
        self.assertGreater(older_score.total, 0.0)

    def test_citation_depth_lifts_material_others_point_at(self):
        cited = self.store.append(
            "The load-bearing decision.",
            kind="note",
            metadata={"name": "load-bearing-decision"},
            timestamp=self._at(10),
        )
        ignored = self.store.append(
            "An unremarkable note.",
            kind="note",
            metadata={"name": "unremarkable"},
            timestamp=self._at(10),
        )
        for index in range(3):
            self.store.append(
                f"Following on from [[load-bearing-decision]] ({index}).",
                kind="note",
                timestamp=self._at(9),
            )
        events = self.store.events(limit=50)
        degrees = wikilink_in_degrees(events)
        self.assertEqual(degrees[cited.id], 3)
        self.assertEqual(degrees[ignored.id], 0)

        scores = score_events(events, now=self.now)
        self.assertGreater(scores[cited.id].total, scores[ignored.id].total)

    def test_self_citation_does_not_count(self):
        event = self.store.append(
            "This note is called [[self-referential]].",
            kind="note",
            metadata={"name": "self-referential"},
        )
        self.assertEqual(wikilink_in_degrees([event])[event.id], 0)

    def test_scores_are_explainable(self):
        event = self.store.append(
            "Connectome recurrent motifs matter here.",
            kind="note",
            metadata={"standing": True},
            timestamp=self._at(1),
        )
        score = score_event(event, query="connectome motifs", now=self.now)
        names = {signal.name for signal in score.signals}
        self.assertIn("standing", names)
        self.assertIn("recency", names)
        self.assertIn("query", names)
        self.assertIn("standing=", score.explain())

    def test_ranking_keeps_peer_engrams_at_the_head(self):
        peer = self.store.append("Peer engram body", kind="engram")
        standing = self.store.append(
            "Standing rule",
            kind="note",
            metadata={"standing": True},
        )
        ordered, _ = rank_selection(
            [("focused", standing), ("hot-peer", peer)],
        )
        self.assertEqual(ordered[0][0], "hot-peer")

    def test_ranking_is_deterministic_for_ties(self):
        first = self.store.append("Alpha", timestamp=self._at(1))
        second = self.store.append("Beta", timestamp=self._at(1))
        selection = [("search", first), ("search", second)]
        once, _ = rank_selection(selection, now=self.now)
        twice, _ = rank_selection(selection, now=self.now)
        self.assertEqual(
            [event.id for _, event in once],
            [event.id for _, event in twice],
        )

    def test_budgeted_context_keeps_standing_material(self):
        self.store.append(
            "STANDING: never delete, only supersede.",
            kind="note",
            metadata={"standing": True},
            session_id="terminal-a",
            timestamp=self._at(120),
        )
        for index in range(25):
            self.store.append(
                f"Routine chatter {index} " + ("filler " * 40),
                session_id="terminal-a",
                timestamp=self._at(0),
            )
        packet = ContextBuilder(self.store).build(
            "what rules apply",
            token_budget=260,
        )
        self.assertIn("never delete, only supersede", packet.markdown)

    def test_salience_can_be_switched_off(self):
        self.store.append("Only message", session_id="terminal-a")
        packet = ContextBuilder(self.store).build(
            "message",
            use_salience=False,
        )
        self.assertEqual(packet.salience, {})
        self.assertIn("Only message", packet.markdown)


if __name__ == "__main__":
    unittest.main()
