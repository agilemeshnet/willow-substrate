from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from willow_substrate.adapters.claude_code import capture_transcript
from willow_substrate.engrams import peer_engrams
from willow_substrate.hooks import handle_claude_hook
from willow_substrate.store import EventStore


def _write_transcript(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class HookTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.home = Path(self.temp.name)
        self.store = EventStore(self.home)

    def tearDown(self):
        self.temp.cleanup()

    def test_claude_transcript_capture_is_idempotent(self):
        transcript = self.home / "session-a.jsonl"
        _write_transcript(
            transcript,
            [
                {
                    "type": "user",
                    "uuid": "user-1",
                    "timestamp": "2026-07-28T10:00:00+00:00",
                    "message": {
                        "role": "user",
                        "content": "Investigate recurrent connectome motifs",
                    },
                },
                {
                    "type": "assistant",
                    "uuid": "assistant-1",
                    "timestamp": "2026-07-28T10:00:01+00:00",
                    "message": {
                        "role": "assistant",
                        "model": "example-model",
                        "usage": {
                            "input_tokens": 10,
                            "cache_read_input_tokens": 90,
                        },
                        "content": [
                            {
                                "type": "text",
                                "text": "## Recurrent motif map\nThe first pass is ready.",
                            },
                            {"type": "tool_use", "name": "Read"},
                        ],
                    },
                },
                {
                    "type": "user",
                    "uuid": "tool-result",
                    "sourceToolAssistantUUID": "assistant-1",
                    "message": {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "content": "ignored"}
                        ],
                    },
                },
            ],
        )

        first = capture_transcript(
            self.store,
            transcript,
            session_id="terminal-a",
        )
        second = capture_transcript(
            self.store,
            transcript,
            session_id="terminal-a",
        )

        self.assertEqual(first.created, 2)
        self.assertEqual(first.recent_input_tokens, 100)
        self.assertEqual(second.created, 0)
        self.assertEqual(second.skipped, 2)
        self.assertEqual(self.store.count(), 2)
        self.assertEqual(first.last_assistant.metadata["tools"], ["Read"])

    def test_prompt_capture_and_later_transcript_row_are_same_event(self):
        transcript = self.home / "prompt.jsonl"
        _write_transcript(transcript, [])
        payload = {
            "session_id": "terminal-a",
            "transcript_path": str(transcript),
            "prompt": "Remember this exact prompt",
        }
        first = handle_claude_hook(
            self.store,
            payload,
            phase="prompt",
        )
        self.assertEqual(first.captured, 1)

        _write_transcript(
            transcript,
            [
                {
                    "type": "user",
                    "uuid": "provider-user-id",
                    "message": {
                        "role": "user",
                        "content": "Remember this exact prompt",
                    },
                }
            ],
        )
        report = capture_transcript(
            self.store,
            transcript,
            session_id="terminal-a",
        )
        self.assertEqual(report.created, 0)
        self.assertEqual(report.skipped, 1)
        self.assertEqual(self.store.count(), 1)

    def test_peer_engram_crosses_terminal_boundary(self):
        transcript = self.home / "session-a.jsonl"
        _write_transcript(
            transcript,
            [
                {
                    "type": "user",
                    "uuid": "a-user",
                    "message": {
                        "role": "user",
                        "content": "Compare Drosophila recurrent connectome motifs",
                    },
                },
                {
                    "type": "assistant",
                    "uuid": "a-assistant",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": "## Connectome comparison underway",
                            }
                        ],
                    },
                },
            ],
        )
        stopped = handle_claude_hook(
            self.store,
            {
                "session_id": "terminal-a",
                "transcript_path": str(transcript),
            },
            phase="stop",
        )
        self.assertEqual(stopped.captured, 3)
        self.assertEqual(len(peer_engrams(self.store)), 1)

        injected = handle_claude_hook(
            self.store,
            {
                "session_id": "terminal-b",
                "prompt": "Where are we with the connectome?",
            },
            phase="prompt",
        )
        self.assertIn("terminal-a", injected.output)
        self.assertIn("Drosophila", injected.output)
        self.assertIn("hot-peer", injected.output)
        self.assertNotIn("terminal-b | user/message", injected.output)

    def test_end_hook_appends_reflection_once(self):
        transcript = self.home / "session-end.jsonl"
        _write_transcript(
            transcript,
            [
                {
                    "type": "user",
                    "uuid": "end-user",
                    "message": {
                        "role": "user",
                        "content": "Investigate temporal graph coherence",
                    },
                },
                {
                    "type": "assistant",
                    "uuid": "end-assistant",
                    "message": {
                        "role": "assistant",
                        "content": "The temporal graph test is coherent.",
                    },
                },
            ],
        )
        payload = {
            "session_id": "terminal-end",
            "transcript_path": str(transcript),
        }
        first = handle_claude_hook(self.store, payload, phase="end")
        second = handle_claude_hook(self.store, payload, phase="end")

        self.assertGreaterEqual(first.captured, 4)
        self.assertEqual(second.captured, 0)
        active_kinds = {
            event.kind
            for event in self.store.events(
                limit=100,
                session_id="terminal-end",
            )
        }
        self.assertIn("summation", active_kinds)
        self.assertIn("meditation", active_kinds)
        valid, _, error = self.store.verify()
        self.assertTrue(valid, error)


if __name__ == "__main__":
    unittest.main()
