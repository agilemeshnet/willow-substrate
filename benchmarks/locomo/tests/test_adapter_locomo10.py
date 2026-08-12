"""Regression tests pinning the LoCoMo10 shape parser.

The upstream LoCoMo10 release nests turns under ``conversation.session_N``
keys paired with ``session_N_date_time`` strings ("1:56 pm on 8 May, 2023").
An earlier adapter version only handled 'turns' / 'sessions' / 'dialog'
keys, so a merge silently dropped every turn and every ingested store came
back empty. These tests keep that from happening again.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.locomo.adapter import (
    _normalise_locomo_timestamp,
    _parse_evidence,
    ingest_into_store,
    load_locomo_conversations,
    session_ids_for,
)
from willow_substrate.store import EventStore


# A hand-written LoCoMo10 fixture. Two sessions, three turns each, one
# question. Fits in-line so nobody has to fetch the real 40MB dataset
# just to run the suite.
FIXTURE = [
    {
        "sample_id": "conv-test-1",
        "conversation": {
            "speaker_a": "Alice",
            "speaker_b": "Bob",
            "session_1": [
                {"speaker": "Alice", "dia_id": "D1:1", "text": "Hi Bob."},
                {"speaker": "Bob", "dia_id": "D1:2", "text": "Hi Alice."},
                {"speaker": "Alice", "dia_id": "D1:3", "text": "How was Rome?"},
            ],
            "session_1_date_time": "1:56 pm on 8 May, 2023",
            "session_2": [
                {"speaker": "Bob", "dia_id": "D2:1", "text": "Rome was hot."},
                {"speaker": "Alice", "dia_id": "D2:2", "text": ""},
                {"speaker": "Bob", "dia_id": "D2:3", "text": "Also crowded."},
            ],
            "session_2_date_time": "9:15 am on 12 May, 2023",
        },
        "qa": [
            {
                "question": "When did Alice ask about Rome?",
                "answer": "8 May 2023",
                "evidence": "['D1:3']",
                "category": "2",
            },
        ],
    },
]


class Locomo10ParserTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "locomo10.json"
        with self.path.open("w") as fh:
            json.dump(FIXTURE, fh)

    def tearDown(self):
        self.tmp.cleanup()

    def test_conversation_id_from_sample_id(self):
        convs = load_locomo_conversations(self.path)
        self.assertEqual(len(convs), 1)
        self.assertEqual(convs[0].conversation_id, "conv-test-1")

    def test_turns_walk_session_keys_in_order(self):
        """session_1 comes before session_2 even if the dict order shuffles.
        Empty-text turns (D2:2 here) are dropped so the ledger is clean."""
        convs = load_locomo_conversations(self.path)
        turn_ids = [t.turn_id for t in convs[0].turns]
        self.assertEqual(turn_ids, ["D1:1", "D1:2", "D1:3", "D2:1", "D2:3"])

    def test_session_timestamp_inherits_to_every_turn(self):
        convs = load_locomo_conversations(self.path)
        by_id = {t.turn_id: t for t in convs[0].turns}
        # session_1 stamps for D1:*: 1:56 pm on 8 May 2023 UTC = 13:56.
        self.assertEqual(by_id["D1:1"].timestamp, "2023-05-08T13:56:00+00:00")
        self.assertEqual(by_id["D1:2"].timestamp, "2023-05-08T13:56:00+00:00")
        self.assertEqual(by_id["D1:3"].timestamp, "2023-05-08T13:56:00+00:00")
        # session_2 stamps for D2:*: 9:15 am on 12 May 2023 UTC = 09:15.
        self.assertEqual(by_id["D2:1"].timestamp, "2023-05-12T09:15:00+00:00")
        self.assertEqual(by_id["D2:3"].timestamp, "2023-05-12T09:15:00+00:00")

    def test_question_evidence_parses_from_string(self):
        """LoCoMo10 stores evidence as a Python-repr-like string, not a
        real JSON array. The parser must unstring it."""
        convs = load_locomo_conversations(self.path)
        q = convs[0].questions[0]
        self.assertEqual(q.gold_turn_ids, ("D1:3",))
        self.assertEqual(q.gold_answer, "8 May 2023")

    def test_ingest_writes_a_row_per_non_empty_turn(self):
        """The whole pipeline: parse -> ingest. Store must end up with the
        same turn count the parser reported, and the mapping table must be
        populated so the runner can translate turn ids to event ids."""
        convs = load_locomo_conversations(self.path)
        conv = convs[0]
        with tempfile.TemporaryDirectory() as tmp:
            store = EventStore(Path(tmp))
            ingest_into_store(store, conv)
            events = store.events(limit=100, active_only=True)
            self.assertEqual(len(events), 5)
            self.assertEqual(len(conv.turn_id_to_event_id), 5)
            gold = conv.turn_id_to_event_id["D1:3"]
            self.assertTrue(gold.startswith("evt-"))

    def test_ingest_scopes_session_ids_per_locomo_session(self):
        """Willow session_id is ``{conv}:s{index}`` so per-session
        operations (meditate, session summation) can scope. Without this
        every turn would share one session_id and meditate would have to
        summarise 419 turns as if they belonged to one conversation
        chunk, defeating the layer's purpose."""
        convs = load_locomo_conversations(self.path)
        conv = convs[0]
        with tempfile.TemporaryDirectory() as tmp:
            store = EventStore(Path(tmp))
            ingest_into_store(store, conv)
            events = store.events(limit=100, active_only=True)
            session_ids = {e.session_id for e in events}
            self.assertEqual(
                session_ids, {"conv-test-1:s1", "conv-test-1:s2"}
            )
            # locomo_session_index preserved in metadata for downstream
            # tooling that wants the raw LoCoMo session number.
            for event in events:
                self.assertIn(
                    event.metadata.get("locomo_session_index"), (1, 2)
                )

    def test_session_ids_for_returns_ordered_unique_ids(self):
        """The helper used by the runner to iterate sessions."""
        convs = load_locomo_conversations(self.path)
        conv = convs[0]
        self.assertEqual(
            session_ids_for(conv),
            ["conv-test-1:s1", "conv-test-1:s2"],
        )


class TimestampNormaliserTests(unittest.TestCase):
    def test_pm_wraps_to_24h(self):
        self.assertEqual(
            _normalise_locomo_timestamp("1:56 pm on 8 May, 2023"),
            "2023-05-08T13:56:00+00:00",
        )

    def test_am_stays_below_noon(self):
        self.assertEqual(
            _normalise_locomo_timestamp("9:15 am on 12 May, 2023"),
            "2023-05-12T09:15:00+00:00",
        )

    def test_midnight_and_noon_edges(self):
        # 12 am is 00:00; 12 pm is 12:00. Trips off-by-one implementations.
        self.assertEqual(
            _normalise_locomo_timestamp("12:00 am on 1 January, 2024"),
            "2024-01-01T00:00:00+00:00",
        )
        self.assertEqual(
            _normalise_locomo_timestamp("12:30 pm on 1 January, 2024"),
            "2024-01-01T12:30:00+00:00",
        )

    def test_missing_comma_before_year_ok(self):
        self.assertEqual(
            _normalise_locomo_timestamp("3:00 pm on 4 July 2024"),
            "2024-07-04T15:00:00+00:00",
        )

    def test_iso_input_passes_through_and_gets_utc_if_naive(self):
        # Round-two reviewer rule: naive stamps get UTC stamped.
        self.assertEqual(
            _normalise_locomo_timestamp("2024-01-01T00:00:00"),
            "2024-01-01T00:00:00+00:00",
        )
        self.assertEqual(
            _normalise_locomo_timestamp("2024-01-01T00:00:00Z"),
            "2024-01-01T00:00:00+00:00",
        )

    def test_unknown_month_raises_visibly(self):
        with self.assertRaises(ValueError):
            _normalise_locomo_timestamp("1:00 pm on 5 Smarch, 2024")

    def test_empty_and_wrong_type_raise(self):
        with self.assertRaises(ValueError):
            _normalise_locomo_timestamp("   ")
        with self.assertRaises(TypeError):
            _normalise_locomo_timestamp(1234)  # type: ignore[arg-type]


class EvidenceParserTests(unittest.TestCase):
    def test_locomo10_python_repr_string(self):
        self.assertEqual(_parse_evidence("['D1:3']"), ("D1:3",))
        self.assertEqual(
            _parse_evidence("['D1:3', 'D2:5']"), ("D1:3", "D2:5")
        )

    def test_plain_json_string_also_ok(self):
        self.assertEqual(_parse_evidence('["D1:3"]'), ("D1:3",))

    def test_list_input(self):
        self.assertEqual(_parse_evidence(["D1:3", "D2:5"]), ("D1:3", "D2:5"))

    def test_dict_with_turn_ids(self):
        self.assertEqual(
            _parse_evidence({"turn_ids": ["D1:3"]}), ("D1:3",)
        )

    def test_none_and_empty_return_empty(self):
        self.assertEqual(_parse_evidence(None), ())
        self.assertEqual(_parse_evidence(""), ())
        self.assertEqual(_parse_evidence("   "), ())


if __name__ == "__main__":
    unittest.main()
