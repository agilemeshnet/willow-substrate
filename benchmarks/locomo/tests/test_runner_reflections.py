"""Tests for the --with-reflections flag path in the runner + the
--expand-derived-from flag path in the scorer.

Ensures:
- Reflections produce meditation events with derived_from links to
  source turns.
- The runner records those links per row so the scorer doesn't need
  to reopen the per-conversation tempdir.
- Score expansion credits meditation retrievals as covering their
  source turns.
- Score without expansion behaves exactly as it did before this branch.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.locomo.adapter import (
    LocomoConversation,
    LocomoQuestion,
    LocomoTurn,
    ingest_into_store,
)
from benchmarks.locomo.run import (
    _derived_from_for,
    _make_reflection_meditator,
    _run_reflections,
)
from benchmarks.locomo.score import (
    _expand_via_derived,
    score_manifest,
)
from willow_substrate.llm import MockMeditator
from willow_substrate.store import EventStore


def _fake_conversation() -> LocomoConversation:
    """Two-session, six-turn conversation with one question.

    Enough turns for meditate() to have material to distill but small
    enough that the test runs fast.
    """
    turns = tuple(
        LocomoTurn(
            turn_id=tid,
            speaker=speaker,
            text=text,
            timestamp="2024-01-01T00:00:00+00:00",
            session_index=idx,
        )
        for idx, tid, speaker, text in [
            (1, "D1:1", "Alice", "Hey Bob, did you finish the Rome trip report?"),
            (1, "D1:2", "Bob", "Yes, wrote it up last night, quite detailed."),
            (1, "D1:3", "Alice", "Great, mind sharing the summary tomorrow?"),
            (2, "D2:1", "Bob", "About that Rome report, I attached photos too."),
            (2, "D2:2", "Alice", "Perfect, let's discuss over coffee."),
            (2, "D2:3", "Bob", "Coffee at ten works for me."),
        ]
    )
    questions = (
        LocomoQuestion(
            question_id="q-0",
            text="When did Bob write the Rome report?",
            gold_turn_ids=("D1:2",),
            category="2",
            gold_answer="last night",
        ),
    )
    return LocomoConversation(
        conversation_id="conv-fake",
        turns=turns,
        questions=questions,
    )


class ReflectionsRunnerTests(unittest.TestCase):
    def test_reflections_produce_meditations_with_derived_from(self):
        conv = _fake_conversation()
        with tempfile.TemporaryDirectory() as tmp:
            store = EventStore(Path(tmp))
            ingest_into_store(store, conv)
            stats = _run_reflections(store, conv)

            # Two sessions -> two meditations. Dreams may or may not fire
            # on such a small fixture; the meditation count is what we pin.
            self.assertEqual(stats["meditations"], 2)
            self.assertGreaterEqual(stats["dreams"], 0)

            meditations = [
                e
                for e in store.events(limit=100, active_only=True)
                if e.kind == "meditation"
            ]
            self.assertEqual(len(meditations), 2)
            for meditation in meditations:
                self.assertTrue(
                    meditation.derived_from,
                    "meditation must carry derived_from links",
                )

    def test_reflections_with_meditator_uses_the_meditator(self):
        """A supplied Meditator drafts each per-session meditation and
        the resulting event's generator tag names the meditator class."""
        conv = _fake_conversation()
        with tempfile.TemporaryDirectory() as tmp:
            store = EventStore(Path(tmp))
            ingest_into_store(store, conv)
            meditator = MockMeditator()
            stats = _run_reflections(store, conv, meditator=meditator)

            # Two sessions -> two meditations, all drafted by the mock.
            self.assertEqual(stats["meditations"], 2)
            self.assertEqual(len(meditator.calls), 2)

            meditations = [
                e
                for e in store.events(limit=100, active_only=True)
                if e.kind == "meditation"
            ]
            self.assertEqual(len(meditations), 2)
            for meditation in meditations:
                self.assertEqual(
                    meditation.metadata.get("generator"),
                    "MockMeditator",
                )
                self.assertIn("MOCK:", meditation.content)
                self.assertTrue(
                    meditation.derived_from,
                    "LLM-drafted meditation must still carry derived_from links",
                )

    def test_derived_from_for_looks_up_only_meditation_and_dream_kinds(self):
        conv = _fake_conversation()
        with tempfile.TemporaryDirectory() as tmp:
            store = EventStore(Path(tmp))
            ingest_into_store(store, conv)
            _run_reflections(store, conv)

            all_events = store.events(limit=100, active_only=True)
            ids = [e.id for e in all_events]
            derived = _derived_from_for(store, ids)

            # Plain messages have no derived_from; only meditations/dreams do.
            for eid in ids:
                event = next(e for e in all_events if e.id == eid)
                if event.kind == "message":
                    self.assertNotIn(eid, derived)
                else:
                    self.assertIn(eid, derived)
                    self.assertTrue(derived[eid])


class MakeReflectionMeditatorTests(unittest.TestCase):
    def test_extractive_returns_none(self):
        """'extractive' means fall back to the deterministic summariser."""
        self.assertIsNone(
            _make_reflection_meditator(
                "extractive", timeout_s=1, max_tokens=1,
            )
        )

    def test_unknown_kind_raises(self):
        with self.assertRaises(ValueError):
            _make_reflection_meditator(
                "not-a-real-meditator", timeout_s=1, max_tokens=1,
            )


class ScoreExpandTests(unittest.TestCase):
    def test_expand_via_derived_expands_after_the_original_id(self):
        """A meditation id followed by its derived_from turns, in order.
        The meditation id stays in the list (idempotent under coincidental
        match to a gold id)."""
        expanded = _expand_via_derived(
            ["med-1", "raw-2"],
            {"med-1": ["src-a", "src-b"], "raw-2": []},
        )
        self.assertEqual(expanded, ["med-1", "src-a", "src-b", "raw-2"])

    def test_expand_deduplicates(self):
        expanded = _expand_via_derived(
            ["med-1", "src-a"],
            {"med-1": ["src-a", "src-b"]},
        )
        self.assertEqual(expanded, ["med-1", "src-a", "src-b"])

    def test_expand_is_identity_when_no_map(self):
        self.assertEqual(
            _expand_via_derived(["a", "b", "c"], {}),
            ["a", "b", "c"],
        )

    def test_score_manifest_expand_credits_meditation_covering_gold(self):
        """A meditation of a session containing the gold turn, retrieved
        in top-K, credits the gold turn when expand_derived_from is on.
        Without the flag, it doesn't."""
        manifest = {
            "config": {"name": "test", "backend": "sparse"},
            "rows": [
                {
                    "conversation_id": "conv-1",
                    "question_id": "q-0",
                    "question_text": "when did Bob write it?",
                    "category": "2",
                    "gold_event_ids": ["gold-src-1"],
                    "retrieved_event_ids": ["med-1", "other-1", "other-2"],
                    "retrieval_latency_ms": 1.0,
                    "per_budget": [],
                    "derived_from": {"med-1": ["gold-src-1", "src-2"]},
                }
            ],
        }
        # Without expansion: gold is NOT among the raw retrieved ids.
        summary_raw = score_manifest(manifest, expand_derived_from=False)
        self.assertEqual(summary_raw["evidence_recall_at_5"]["mean"], 0.0)
        # With expansion: the meditation covers the gold; recall is 1.
        summary_exp = score_manifest(manifest, expand_derived_from=True)
        self.assertEqual(summary_exp["evidence_recall_at_5"]["mean"], 1.0)

    def test_score_manifest_default_is_no_expansion(self):
        """No behaviour change for pre-existing manifests that carry no
        derived_from map."""
        manifest = {
            "config": {"name": "legacy", "backend": "sparse"},
            "rows": [
                {
                    "conversation_id": "conv-1",
                    "question_id": "q-0",
                    "question_text": "?",
                    "category": "1",
                    "gold_event_ids": ["g-1"],
                    "retrieved_event_ids": ["g-1", "x"],
                    "retrieval_latency_ms": 1.0,
                    "per_budget": [],
                }
            ],
        }
        # No expansion, no derived_from field present -> works identically
        # to the pre-branch code.
        summary = score_manifest(manifest)
        self.assertEqual(summary["evidence_recall_at_5"]["mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
