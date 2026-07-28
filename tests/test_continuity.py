from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from willow.context import ContextBuilder
from willow.foveation import Foveator
from willow.reflection import meditate
from willow.store import EventStore


class ContinuityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.home = Path(self.temp.name)
        self.store = EventStore(self.home)
        self.first = self.store.append(
            "Investigating the Drosophila connectome and recurrent circuits",
            actor="peter",
            session_id="terminal-a",
            metadata={"topics": ["connectome"]},
        )
        self.second = self.store.append(
            "Next compare recurrent motifs with Willow foveation",
            actor="willow",
            session_id="terminal-a",
            metadata={"topics": ["foveation"]},
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_context_in_another_terminal_contains_shared_work(self):
        packet = ContextBuilder(EventStore(self.home)).build(
            "What were we doing with the connectome?",
            session_id="terminal-b",
            token_budget=800,
        )
        self.assertIn("Drosophila connectome", packet.markdown)
        self.assertIn("terminal-a", packet.markdown)
        self.assertIn(self.first.id, packet.event_ids)

    def test_foveation_has_three_explicit_passes(self):
        result = Foveator(self.store).foveate(
            "connectome recurrent motifs",
            event_limit=6,
        )
        self.assertEqual(
            [item.name for item in result.trace],
            ["peripheral", "parafoveal", "foveal"],
        )
        self.assertIn(self.first.id, result.event_ids)
        self.assertIn(self.second.id, result.event_ids)

    def test_meditation_is_append_only_and_provenanced(self):
        reflection = meditate(self.store, "terminal-a")
        self.assertEqual(reflection.kind, "meditation")
        self.assertEqual(
            set(reflection.derived_from),
            {self.first.id, self.second.id},
        )
        valid, count, error = self.store.verify()
        self.assertTrue(valid, error)
        self.assertEqual(count, 3)

    def test_context_obeys_budget(self):
        for index in range(30):
            self.store.append(
                f"Connectome observation {index} " + ("detail " * 50),
                actor="worker",
                session_id="terminal-a",
            )
        packet = ContextBuilder(self.store).build(
            "connectome observation",
            token_budget=200,
        )
        self.assertLessEqual(len(packet.markdown), 800)


if __name__ == "__main__":
    unittest.main()
