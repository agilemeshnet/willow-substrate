"""Smoke test the recall benchmark end-to-end.

The benchmark itself is the deliverable; this test just proves it runs and
produces the expected shape. It does NOT pin numeric thresholds here (those
live in the JSON output and get pinned by CI when it lands); tightening this
test into a numeric-regression test would tie unrelated PRs to benchmark
tuning that belongs in the benchmark file.
"""
from __future__ import annotations

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class BenchmarkSmokeTests(unittest.TestCase):
    def test_run_returns_expected_shape(self):
        import sys
        sys.path.insert(0, str(REPO_ROOT))
        from benchmarks.recall.harness import run

        results = run(limit=5)
        self.assertIn("backends", results)
        self.assertIn("per_query", results)
        self.assertGreaterEqual(len(results["backends"]), 2)  # sparse + bm25 at minimum

        names = {row["name"] for row in results["backends"]}
        self.assertIn("sparse", names)
        self.assertIn("bm25", names)

        for row in results["backends"]:
            self.assertIn("recall_at_3", row)
            self.assertIn("recall_at_5", row)
            self.assertIn("mrr", row)
            self.assertIn("median_latency_ms", row)
            # Basic sanity: recall in [0, 1] and MRR in [0, 1].
            self.assertGreaterEqual(row["recall_at_5"], 0.0)
            self.assertLessEqual(row["recall_at_5"], 1.0)
            self.assertGreaterEqual(row["mrr"], 0.0)
            self.assertLessEqual(row["mrr"], 1.0)

    def test_markdown_table_renders(self):
        import sys
        sys.path.insert(0, str(REPO_ROOT))
        from benchmarks.recall.harness import format_markdown_table, run

        results = run(limit=5)
        table = format_markdown_table(results)
        self.assertIn("| Backend |", table)
        self.assertIn("sparse", table)
        self.assertIn("bm25", table)


if __name__ == "__main__":
    unittest.main()
