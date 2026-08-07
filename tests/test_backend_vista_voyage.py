"""Tests for the [vista] extra's dense-embedding backend.

Skip gracefully when the extra is not installed; run for real (with a mock
embedder so no real Voyage API calls happen) when it is. Same pattern as
docs/EXTRAS.md documents for contributors.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np  # noqa: F401
    from willow.backends.vista_voyage import (  # noqa: F401
        VoyageEmbedder,
        VoyageVistaBackend,
        _cluster_by_hdbscan,
        _cosine_matrix,
        _canonical_event_text,
    )
    HAS_VISTA_EXTRA = True
except ImportError:
    HAS_VISTA_EXTRA = False


from willow.store import EventStore
from willow.vista import VistaResult


class _MockEmbedder:
    """Deterministic embedder for tests; never hits Voyage.

    Produces 8-dim vectors from a simple hash so semantically-related
    strings can be seeded to cluster together. Signature matches what
    VoyageVistaBackend.embed() calls, so the backend is happy.
    """

    def __init__(self, seed_map: dict[str, "np.ndarray"] | None = None):
        if not HAS_VISTA_EXTRA:
            return
        self._seed_map = seed_map or {}
        self._cache: dict[str, "np.ndarray"] = {}

    def embed(self, items):
        result = {}
        for key, text in items:
            if key in self._cache:
                result[key] = self._cache[key]
                continue
            # First seed by explicit override, then fall back to a deterministic
            # 8-dim vector derived from the text hash.
            if key in self._seed_map:
                vec = self._seed_map[key]
            else:
                # Hash the text into 8 stable floats in [-1, 1].
                seed = abs(hash(text)) % (2**32)
                rng = np.random.default_rng(seed)
                vec = rng.standard_normal(8).astype(np.float32)
            self._cache[key] = vec
            result[key] = vec
        return result


@unittest.skipUnless(
    HAS_VISTA_EXTRA,
    "install with pip install \"willow-substrate[vista]\" to run these tests",
)
class VoyageBackendCoreTests(unittest.TestCase):
    """Algorithmic surface: cosine, cluster, canonical text."""

    def test_canonical_event_text_stems_actor_and_kind(self):
        # Round-trip through a temporary store so we get a real Event object.
        with tempfile.TemporaryDirectory() as tmp:
            store = EventStore(Path(tmp))
            event = store.append(
                "the connectome contains recurrent motifs",
                actor="peter",
                kind="observation",
                session_id="s",
            )
            text = _canonical_event_text(event)
            self.assertTrue(text.startswith("[peter/observation]"))
            self.assertIn("connectome", text)

    def test_cosine_matrix_matches_direct_dot(self):
        a = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        b = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        sim = _cosine_matrix(a, b)
        # a[0]·b[0] normalised = 1.0; a[0]·b[1] normalised = 1/sqrt(2).
        self.assertAlmostEqual(float(sim[0, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(sim[0, 1]), 1.0 / np.sqrt(2), places=5)

    def test_cluster_by_hdbscan_falls_back_when_below_min_size(self):
        vectors = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
        )
        labels = _cluster_by_hdbscan(vectors, min_cluster_size=3)
        # Not enough points for HDBSCAN's minimum; all-zero single-cluster fallback.
        self.assertEqual(list(labels), [0, 0])


@unittest.skipUnless(
    HAS_VISTA_EXTRA,
    "install with pip install \"willow-substrate[vista]\" to run these tests",
)
class VoyageBackendQueryTests(unittest.TestCase):
    """End-to-end query using a mock embedder so no Voyage call happens."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.home = Path(self.temp.name)
        self.store = EventStore(self.home)

    def tearDown(self):
        self.temp.cleanup()

    def _seed_two_clusters(self):
        # Cluster A: connectome events lie along [1, 0].
        # Cluster B: pricing events lie along [0, 1].
        # 8-dim vectors so HDBSCAN has room; two clear clusters of 4 each.
        cluster_a = np.array(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
        )
        cluster_b = np.array(
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
        )

        seed_map = {}
        events = []
        for i in range(4):
            e = self.store.append(
                f"connectome observation {i}",
                actor="willow",
                kind="observation",
                session_id="lab",
            )
            events.append(e)
            seed_map[e.id] = cluster_a + np.random.default_rng(i).standard_normal(8).astype(np.float32) * 0.05
        for i in range(4):
            e = self.store.append(
                f"pricing signal {i}",
                actor="willow",
                kind="observation",
                session_id="markets",
            )
            events.append(e)
            seed_map[e.id] = cluster_b + np.random.default_rng(100 + i).standard_normal(8).astype(np.float32) * 0.05
        # Query vector aligned with cluster A.
        seed_map["__query__:connectome"] = cluster_a
        return events, seed_map

    def test_query_returns_semantic_neighbourhood(self):
        events, seed_map = self._seed_two_clusters()
        embedder = _MockEmbedder(seed_map=seed_map)
        backend = VoyageVistaBackend(
            self.store,
            embedder=embedder,
            min_cluster_size=3,
        )

        result = backend.query("connectome", limit=6)

        self.assertIsInstance(result, VistaResult)
        self.assertEqual(result.query, "connectome")
        # At least one match should have populated semantic_score > 0.
        self.assertTrue(
            any(m.semantic_score > 0.0 for m in result.matches),
            f"expected semantic-score-bearing matches, got {result.matches}",
        )
        # The top-ranked evidence event should be a connectome event, not a
        # pricing one (query is aligned with cluster A).
        top_event_id = result.evidence[0].event.id if result.evidence else None
        connectome_ids = {e.id for e in events[:4]}
        self.assertIn(
            top_event_id, connectome_ids,
            "top evidence must come from the semantically-matching cluster",
        )
        # Reference beams should include only positive-weight events.
        for beam in result.reference_beams:
            self.assertGreater(beam.weight, 0.0)
            self.assertEqual(beam.source, "voyage-cosine")
        # Trace records the projection size for debuggability.
        self.assertTrue(
            any("corpus=8" in line for line in result.trace),
            f"trace should report corpus size, got {result.trace}",
        )

    def test_query_returns_empty_result_on_empty_corpus(self):
        embedder = _MockEmbedder()
        backend = VoyageVistaBackend(self.store, embedder=embedder)
        result = backend.query("anything", limit=4)
        self.assertEqual(result.query, "anything")
        self.assertEqual(result.evidence, ())
        self.assertEqual(result.matches, ())


class NoExtrasImportGuardTests(unittest.TestCase):
    """The extra-guarded module should either import cleanly (extras present)
    or fail with a message naming the fix. This test is meant to run in both
    environments; it is a sanity check on the guard message rather than
    exercising the backend."""

    def test_import_guard_message_names_the_extras_install(self):
        if HAS_VISTA_EXTRA:
            # In the [vista]-installed environment, the module imports fine.
            import willow.backends.vista_voyage  # noqa: F401
        else:
            # Without the extra, the import raises ImportError whose message
            # names the fix. Assert both properties.
            with self.assertRaises(ImportError) as ctx:
                import willow.backends.vista_voyage  # noqa: F401
            message = str(ctx.exception)
            self.assertIn("[vista]", message)
            self.assertIn("pip install", message)


if __name__ == "__main__":
    unittest.main()
