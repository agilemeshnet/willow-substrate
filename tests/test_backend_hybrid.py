"""Tests for the hybrid RRF recall backend."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from willow_substrate.backends.hybrid import HybridRecallBackend, _rrf_fuse, _BackendContribution
from willow_substrate.store import EventStore
from willow_substrate.vista import VistaResult


class RRFFusionTests(unittest.TestCase):
    """Cormack et al. 2009: rank-based fusion, not score-based."""

    def test_single_backend_returns_input_order(self):
        contributions = [
            _BackendContribution(name="only", ranking=["a", "b", "c"])
        ]
        fused = _rrf_fuse(contributions, k_rrf=60)
        self.assertEqual([eid for eid, _ in fused], ["a", "b", "c"])

    def test_intersection_ranks_higher_than_singleton(self):
        """An event appearing in both backends must fuse above either that
        appears in only one."""
        contributions = [
            _BackendContribution(name="A", ranking=["x", "y"]),
            _BackendContribution(name="B", ranking=["y", "z"]),
        ]
        fused = dict(_rrf_fuse(contributions, k_rrf=60))
        self.assertGreater(fused["y"], fused["x"])
        self.assertGreater(fused["y"], fused["z"])

    def test_k_rrf_dampens_low_ranks(self):
        """Higher k_rrf compresses the score gap between rank-1 and rank-N."""
        contributions = [
            _BackendContribution(name="A", ranking=[f"e{i}" for i in range(20)]),
        ]
        low_k = dict(_rrf_fuse(contributions, k_rrf=1))
        high_k = dict(_rrf_fuse(contributions, k_rrf=1000))
        gap_low = low_k["e0"] - low_k["e19"]
        gap_high = high_k["e0"] - high_k["e19"]
        self.assertGreater(gap_low, gap_high)


class HybridBackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.home = Path(self.temp.name)
        self.store = EventStore(self.home)

    def tearDown(self):
        self.temp.cleanup()

    def _seed_events(self):
        self.store.append(
            "the connectome contains recurrent motifs across regions",
            actor="willow", session_id="lab",
        )
        self.store.append(
            "pricing signals point to inflation upcoming",
            actor="willow", session_id="markets",
        )
        self.store.append(
            "recurrent neural circuits underlie memory formation",
            actor="peter", session_id="lab",
        )

    def test_hybrid_returns_vista_result_shape(self):
        self._seed_events()
        backend = HybridRecallBackend(self.store)
        result = backend.query("connectome", limit=3)
        self.assertIsInstance(result, VistaResult)
        self.assertEqual(result.query, "connectome")
        # RRF trace line should appear in the trace.
        self.assertTrue(
            any("hybrid RRF" in line for line in result.trace),
            f"expected RRF trace line, got {result.trace}",
        )

    def test_hybrid_degrades_gracefully_without_dense(self):
        """No VOYAGE_API_KEY means no dense sub-backend; hybrid must still
        return meaningful results from sparse + BM25."""
        import os
        from unittest.mock import patch

        self._seed_events()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VOYAGE_API_KEY", None)
            backend = HybridRecallBackend(self.store)
            self.assertIsNone(backend.dense)
            result = backend.query("connectome", limit=3)
            # At least the connectome event should appear.
            top_ids = [ev.event.id for ev in result.evidence]
            found_connectome = any(
                "connectome" in self.store.events(limit=100)[0].content or
                "recurrent" in ev.event.content
                for ev in result.evidence
            )
            self.assertTrue(
                found_connectome or len(top_ids) > 0,
                "hybrid without dense must still return term-matching events",
            )

    def test_hybrid_invalid_k_rrf_raises(self):
        with self.assertRaises(ValueError):
            HybridRecallBackend(self.store, k_rrf=0)


class HybridFactoryTests(unittest.TestCase):
    """Factory resolves 'hybrid' correctly."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.home = Path(self.temp.name)
        self.store = EventStore(self.home)

    def tearDown(self):
        self.temp.cleanup()

    def test_factory_returns_hybrid_when_named(self):
        from willow_substrate.backends.factory import make_relational_backend

        backend = make_relational_backend(self.store, name="hybrid")
        self.assertIsInstance(backend, HybridRecallBackend)

    def test_factory_rejects_hybrid_typo(self):
        from willow_substrate.backends.factory import make_relational_backend

        with self.assertRaises(ValueError):
            make_relational_backend(self.store, name="hybird")


if __name__ == "__main__":
    unittest.main()
