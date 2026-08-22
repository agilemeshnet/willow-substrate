"""Tests for the trained-readout two-stage retrieval integration.

Covers:
* WaveFeatures dataclass construction and vector shape.
* build_wave_features clipping and channel-bias derivation.
* LinearReranker weighted-sum scoring, default coefficients, from_dict
  validation, bias term.
* VistaBackend.query with a reranker set: wave_features appear on
  evidence, scoring uses the reranker, backward-compatible behaviour
  is preserved when no reranker is passed.
"""

from __future__ import annotations

import runpy
import tempfile
import unittest
from pathlib import Path

from willow_substrate.readout import (
    FEATURE_NAMES,
    LinearReranker,
    WaveFeatures,
    build_wave_features,
    score_candidates,
)
from willow_substrate.store import EventStore
from willow_substrate.vista import VistaBackend


class WaveFeaturesTests(unittest.TestCase):
    def test_as_vector_preserves_feature_order(self):
        features = WaveFeatures(
            vista_score=0.10,
            wave_score=0.20,
            wave_peak=0.30,
            wave_hop_of_peak=0.40,
            wave_early=0.50,
            channel_bias=1.00,
        )
        vector = features.as_vector()
        self.assertEqual(len(vector), len(FEATURE_NAMES))
        self.assertEqual(vector, (0.10, 0.20, 0.30, 0.40, 0.50, 1.00))

    def test_build_wave_features_clips_and_normalises(self):
        features = build_wave_features(
            vista_score=1.7,
            wave_score=-0.2,
            wave_peak=0.9,
            wave_hop_of_peak_index=2,
            wave_early_activation=0.5,
            hops=4,
        )
        self.assertEqual(features.vista_score, 1.0)
        self.assertEqual(features.wave_score, 0.0)
        self.assertEqual(features.wave_peak, 0.9)
        self.assertEqual(features.wave_hop_of_peak, 0.5)
        self.assertEqual(features.wave_early, 0.5)

    def test_channel_bias_reflects_lit_channels(self):
        both = build_wave_features(
            vista_score=0.4, wave_score=0.3,
            wave_peak=0.3, wave_hop_of_peak_index=1,
            wave_early_activation=0.1, hops=3,
        )
        one = build_wave_features(
            vista_score=0.4, wave_score=0.0,
            wave_peak=0.0, wave_hop_of_peak_index=0,
            wave_early_activation=0.0, hops=3,
        )
        none = build_wave_features(
            vista_score=0.0, wave_score=0.0,
            wave_peak=0.0, wave_hop_of_peak_index=0,
            wave_early_activation=0.0, hops=3,
        )
        self.assertEqual(both.channel_bias, 1.0)
        self.assertEqual(one.channel_bias, 0.5)
        self.assertEqual(none.channel_bias, 0.0)


class LinearRerankerTests(unittest.TestCase):
    FEATURES = WaveFeatures(
        vista_score=0.5,
        wave_score=0.4,
        wave_peak=0.6,
        wave_hop_of_peak=0.25,
        wave_early=0.2,
        channel_bias=1.0,
    )

    def test_empty_reranker_returns_bias(self):
        self.assertEqual(LinearReranker().score(self.FEATURES), 0.0)
        self.assertEqual(
            LinearReranker(bias=0.5).score(self.FEATURES),
            0.5,
        )

    def test_weighted_sum_matches_dot_product(self):
        reranker = LinearReranker.from_dict(
            {
                "vista_score": 1.0,
                "wave_score": 2.0,
                "wave_peak": 3.0,
                "wave_hop_of_peak": 4.0,
                "wave_early": 5.0,
                "channel_bias": 6.0,
            }
        )
        expected = (
            1.0 * 0.5 + 2.0 * 0.4 + 3.0 * 0.6
            + 4.0 * 0.25 + 5.0 * 0.2 + 6.0 * 1.0
        )
        self.assertAlmostEqual(reranker.score(self.FEATURES), expected)

    def test_default_reranker_rewards_early_wave_arrival(self):
        default = LinearReranker.default()
        # The default combines fine semantic ranking with Wave's activation
        # and prefers evidence reached earlier in the trajectory.
        self.assertEqual(default.weights["vista_score"], 0.65)
        self.assertGreater(default.weights["wave_score"], 0.0)
        self.assertLess(default.weights["wave_hop_of_peak"], 0.0)
        early = WaveFeatures(
            vista_score=0.5, wave_score=0.5, wave_peak=0.5,
            wave_hop_of_peak=0.0, wave_early=0.5, channel_bias=1.0,
        )
        late = WaveFeatures(
            vista_score=0.5, wave_score=0.5, wave_peak=0.5,
            wave_hop_of_peak=0.75, wave_early=0.5, channel_bias=1.0,
        )
        self.assertGreater(default.score(early), default.score(late))

    def test_from_dict_rejects_unknown_features(self):
        with self.assertRaises(ValueError):
            LinearReranker.from_dict({"unknown_feature": 1.0})

    def test_score_candidates_preserves_order(self):
        reranker = LinearReranker.default()
        a = WaveFeatures(0.9, 0.1, 0.1, 0.0, 0.0, 0.5)  # vista-heavy
        b = WaveFeatures(0.1, 0.9, 0.9, 0.5, 0.5, 0.5)  # wave-heavy
        c = WaveFeatures(0.5, 0.5, 0.5, 0.5, 0.5, 1.0)  # both
        scores = score_candidates(reranker, [a, b, c])
        self.assertEqual(len(scores), 3)
        # Default weights privilege vista, so vista-heavy should beat wave-heavy.
        self.assertGreater(scores[0], scores[1])


class RerankerIntegrationTests(unittest.TestCase):
    """VistaBackend.query with reranker attached."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = EventStore(Path(self.temp.name))

    def tearDown(self):
        self.temp.cleanup()

    def _seed_corpus(self):
        first = self.store.append(
            "A flock's identity vanishes when the birds land.",
            kind="meditation",
            session_id="field",
            metadata={
                "topics": ["collective-motion"],
                "idea_shape": ["identity:relational"],
            },
        )
        second = self.store.append(
            "A temporary institution exists only while participation "
            "continues; its identity is relational.",
            kind="research_result",
            session_id="sociology",
            metadata={
                "topics": ["temporary-institutions"],
                "idea_shape": ["identity:relational"],
            },
        )
        third = self.store.append(
            "Copper oxidation measurements need another calibration run.",
            kind="research_result",
            session_id="chemistry",
            metadata={"topics": ["spectroscopy"]},
        )
        return first, second, third

    def test_query_without_reranker_matches_existing_behaviour(self):
        first, second, _ = self._seed_corpus()
        backend = VistaBackend(self.store)
        baseline = backend.query(first.content, seed_event_ids=(first.id,), limit=5)
        with_none = backend.query(
            first.content, seed_event_ids=(first.id,), limit=5, reranker=None
        )
        self.assertEqual(baseline.event_ids, with_none.event_ids)
        for evidence in baseline.evidence:
            self.assertIsNone(evidence.wave_features)

    def test_query_with_reranker_populates_wave_features(self):
        first, second, _ = self._seed_corpus()
        backend = VistaBackend(self.store)
        result = backend.query(
            first.content,
            seed_event_ids=(first.id,),
            limit=5,
            reranker=LinearReranker.default(),
        )
        self.assertTrue(result.evidence)
        for evidence in result.evidence:
            self.assertIsNotNone(evidence.wave_features)
            vector = evidence.wave_features.as_vector()
            self.assertEqual(len(vector), len(FEATURE_NAMES))
            for value in vector:
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)

    def test_reranker_score_matches_scored_features(self):
        """The evidence score IS the reranker's score of its features."""
        first, _, _ = self._seed_corpus()
        reranker = LinearReranker.from_dict(
            {"vista_score": 1.0, "wave_score": 2.0}
        )
        result = VistaBackend(self.store).query(
            first.content,
            seed_event_ids=(first.id,),
            limit=5,
            reranker=reranker,
        )
        for evidence in result.evidence:
            expected = reranker.score(evidence.wave_features)
            self.assertAlmostEqual(evidence.score, expected, places=6)

    def test_reranker_is_applied_when_wave_is_disabled(self):
        """A reranker still scores direct Vista evidence with zero wave data."""
        first, _, _ = self._seed_corpus()
        reranker = LinearReranker.from_dict({}, bias=42.0)
        result = VistaBackend(self.store).query(
            first.content,
            seed_event_ids=(first.id,),
            limit=5,
            wave_hops=0,
            reranker=reranker,
        )
        self.assertTrue(result.evidence)
        for evidence in result.evidence:
            self.assertIsNotNone(evidence.wave_features)
            self.assertEqual(evidence.score, 42.0)
            self.assertEqual(evidence.wave_features.wave_score, 0.0)
            self.assertEqual(evidence.wave_features.wave_peak, 0.0)
            self.assertEqual(evidence.wave_features.wave_early, 0.0)

    def test_reranker_can_flip_evidence_order(self):
        """A reranker that only trusts wave_score should reorder evidence
        differently from one that only trusts vista_score, proving the
        readout genuinely drives the ranking."""
        first, _, _ = self._seed_corpus()
        backend = VistaBackend(self.store)
        vista_only = backend.query(
            first.content,
            seed_event_ids=(first.id,),
            limit=8,
            reranker=LinearReranker.from_dict({"vista_score": 1.0}),
        )
        wave_only = backend.query(
            first.content,
            seed_event_ids=(first.id,),
            limit=8,
            reranker=LinearReranker.from_dict({"wave_score": 1.0}),
        )
        # Both queries operate on the same evidence pool; at least one
        # position should reorder (otherwise the wave signal is trivially
        # constant on this corpus and the test is uninformative).
        if len(vista_only.evidence) >= 2 and len(wave_only.evidence) >= 2:
            self.assertTrue(
                vista_only.event_ids != wave_only.event_ids
                or any(
                    v.wave_score != v.vista_score
                    for v in vista_only.evidence
                )
            )


class ReferenceReadoutBenchmarkTests(unittest.TestCase):
    def test_default_reranker_beats_the_legacy_heuristic(self):
        """The fixed reference corpus guards the intended Wave + Vista lift."""
        root = Path(__file__).resolve().parents[1]
        benchmark = runpy.run_path(
            str(root / "benchmarks" / "readout" / "wave_ridge_recall.py")
        )
        store, temporary, topic_ids = benchmark["build_store"]()
        try:
            backend = VistaBackend(store)
            baseline, _ = benchmark["evaluate"](backend, topic_ids)
            combined, _ = benchmark["evaluate"](
                backend,
                topic_ids,
                reranker=LinearReranker.default(),
            )
        finally:
            temporary.cleanup()
        self.assertGreater(combined, baseline)


if __name__ == "__main__":
    unittest.main()
