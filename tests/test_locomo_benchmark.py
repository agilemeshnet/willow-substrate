"""Smoke tests for the LoCoMo benchmark scaffold.

Skip when the real dataset is absent (default state); exercise the
adapter, runner, and scorer end-to-end against a tiny synthetic
LoCoMo-shaped fixture so the code paths stay honest even before the
real dataset lands.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.locomo.adapter import (
    LocomoConversation,
    LocomoQuestion,
    LocomoTurn,
    ingest_into_store,
    load_locomo_conversations,
)
from benchmarks.locomo.score import score_manifest
from willow_substrate.store import EventStore


_FIXTURE = {
    "conversations": [
        {
            "id": "conv-1",
            "turns": [
                {
                    "id": "t1",
                    "speaker": "alice",
                    "text": "I moved to Berlin in 2022.",
                    "timestamp": "2022-03-04T09:00:00+00:00",
                },
                {
                    "id": "t2",
                    "speaker": "bob",
                    "text": "Berlin has great transit.",
                    "timestamp": "2022-03-04T09:01:00+00:00",
                },
                {
                    "id": "t3",
                    "speaker": "alice",
                    "text": "I bought a bike last month.",
                    "timestamp": "2023-11-01T09:00:00+00:00",
                },
            ],
            "questions": [
                {
                    "id": "q1",
                    "question": "When did Alice move to Berlin?",
                    "evidence": ["t1"],
                    "category": "temporal",
                    "answer": "2022",
                },
            ],
        }
    ]
}


class AdapterTests(unittest.TestCase):
    def test_load_conversations_from_combined_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.json"
            path.write_text(json.dumps(_FIXTURE))
            conversations = load_locomo_conversations(path)
            self.assertEqual(len(conversations), 1)
            self.assertEqual(conversations[0].conversation_id, "conv-1")
            self.assertEqual(len(conversations[0].turns), 3)
            self.assertEqual(len(conversations[0].questions), 1)
            self.assertEqual(
                conversations[0].questions[0].gold_turn_ids, ("t1",)
            )

    def test_ingest_populates_turn_id_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = EventStore(Path(tmp))
            conversation = LocomoConversation(
                conversation_id="conv-x",
                turns=(
                    LocomoTurn(
                        turn_id="t-a",
                        speaker="alice",
                        text="test one",
                        timestamp="2024-01-01T00:00:00+00:00",
                    ),
                ),
                questions=(),
            )
            ingest_into_store(store, conversation)
            self.assertIn("t-a", conversation.turn_id_to_event_id)


class ScorerTests(unittest.TestCase):
    def test_score_manifest_computes_perfect_recall_when_retrieval_matches(self):
        manifest = {
            "config": {"name": "test", "backend": "test"},
            "willow_commit": "test",
            "wall_seconds": 0.1,
            "rows": [
                {
                    "conversation_id": "c1",
                    "question_id": "q1",
                    "question_text": "?",
                    "category": "test",
                    "gold_event_ids": ["e1", "e2"],
                    "retrieved_event_ids": ["e1", "e2", "e3"],
                    "retrieval_latency_ms": 1.0,
                    "per_budget": [],
                }
            ],
        }
        summary = score_manifest(manifest)
        self.assertAlmostEqual(summary["evidence_recall_at_5"]["mean"], 1.0)
        self.assertAlmostEqual(summary["mrr"]["mean"], 1.0)

    def test_score_manifest_handles_missing_gold_gracefully(self):
        manifest = {
            "config": {"name": "test", "backend": "test"},
            "willow_commit": "test",
            "wall_seconds": 0.1,
            "rows": [
                {
                    "conversation_id": "c1",
                    "question_id": "q1",
                    "question_text": "?",
                    "category": "test",
                    "gold_event_ids": [],
                    "retrieved_event_ids": ["x"],
                    "retrieval_latency_ms": 1.0,
                    "per_budget": [],
                }
            ],
        }
        summary = score_manifest(manifest)
        self.assertEqual(summary["evidence_recall_at_5"]["mean"], 0.0)


class DatasetSkipTests(unittest.TestCase):
    """The real dataset is not vendored. This test just confirms the
    adapter's error is friendly when data_dir is missing so a first-time
    runner knows what to do."""

    def test_missing_dataset_dir_raises_helpful(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            load_locomo_conversations(REPO_ROOT / "nonexistent-locomo")
        message = str(ctx.exception)
        self.assertIn("BENCHMARK_LOCOMO", message)


if __name__ == "__main__":
    unittest.main()
