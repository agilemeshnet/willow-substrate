from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class MultiProcessContinuityTests(unittest.TestCase):
    def test_concurrent_processes_can_initialize_an_empty_store(self):
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp) / "new-store"
            base = [
                sys.executable,
                "-m",
                "willow_substrate.cli",
                "--home",
                str(home),
                "record",
            ]
            processes = [
                subprocess.Popen(
                    [
                        *base,
                        f"First-run event {index}",
                        "--session",
                        f"terminal-{index}",
                    ],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                for index in range(6)
            ]
            results = [process.communicate(timeout=20) for process in processes]
            for process, (_, stderr) in zip(processes, results):
                self.assertEqual(process.returncode, 0, stderr)

            verified = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "willow_substrate.cli",
                    "--home",
                    str(home),
                    "verify",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(verified.returncode, 0, verified.stderr)
            self.assertIn("6 events", verified.stdout)

    def test_independent_hook_processes_share_continuity(self):
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp)
            transcript = home / "terminal-a.jsonl"
            transcript.write_text(
                "\n".join(
                    [
                        json.dumps({
                            "type": "user",
                            "uuid": "process-user",
                            "message": {
                                "role": "user",
                                "content": (
                                    "Map recurrent Drosophila connectome motifs"
                                ),
                            },
                        }),
                        json.dumps({
                            "type": "assistant",
                            "uuid": "process-assistant",
                            "message": {
                                "role": "assistant",
                                "content": "The connectome map is underway.",
                            },
                        }),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            base = [
                sys.executable,
                "-m",
                "willow_substrate.cli",
                "--home",
                str(home),
            ]

            first = subprocess.run(
                [*base, "hook", "stop"],
                input=json.dumps({
                    "session_id": "terminal-a",
                    "transcript_path": str(transcript),
                }),
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(first.returncode, 0, first.stderr)

            second = subprocess.run(
                [*base, "hook", "prompt"],
                input=json.dumps({
                    "session_id": "terminal-b",
                    "prompt": "What are we doing with the connectome?",
                }),
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertIn("terminal-a", second.stdout)
            self.assertIn("Drosophila", second.stdout)

            verified = subprocess.run(
                [*base, "verify"],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(verified.returncode, 0, verified.stderr)
            self.assertIn("global hash chain intact", verified.stdout)


if __name__ == "__main__":
    unittest.main()
