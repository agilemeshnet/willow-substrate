"""Trained readout over the Wave + Vista two-stage retrieval.

The reference Vista backend already carries two orthogonal views on the
event ledger:

* **Vista**: static semantic clustering; groups events whose feature
  vectors are cosine-close in the sparse lexical-semantic (or, with the
  richer backend, dense embedding) space.
* **Wave**: multi-hop spreading activation across the event <-> waypoint
  lattice; damped by an alpha restart and shaped by mass-decayed
  conductance.

Empirically the two views are ORTHOGONAL: they surface different
neighbourhoods for the same seed. Wave.py acts as a coarse RETRIEVER of
the candidate pool; Vista's semantic proximity acts as a fine RERANKER
within it. Combining them via a trained linear readout lifts recall of
same-cluster events substantially above either channel alone; the
combination weights ARE the leverage.

See ``docs/design/TWO_STAGE_RETRIEVAL.md`` for the full design and
measurement narrative.

This module ships a dependency-free ``LinearReranker`` that accepts a
pre-fitted weight vector. Fit weights externally with the training tool of
your choice, then pass them to ``LinearReranker.from_dict``. The reference
backend accepts an optional reranker in ``VistaBackend.query`` without
changing the default behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Protocol, Sequence


FEATURE_NAMES: tuple[str, ...] = (
    "vista_score",       # 0.0..1.0 direct semantic proximity to seed via Vista match
    "wave_score",        # 0.0..1.0 normalised final wave activation
    "wave_peak",         # 0.0..1.0 normalised peak wave activation across hops
    "wave_hop_of_peak",  # 0.0..1.0 hop index where wave peaked (normalised by total hops)
    "wave_early",        # 0.0..1.0 normalised activation at hop 1 (arrival immediacy)
    "channel_bias",      # 1.0 if both channels lit, 0.5 if only one, 0.0 if neither
)


@dataclass(frozen=True)
class WaveFeatures:
    """Per-candidate features exposed by the Wave retriever.

    A ``VistaResult`` produced with a reranker attached carries one of these
    per surfaced ``VistaEvidence``. Values are always in the closed unit
    interval so that a linear combination is stable across backends and
    corpus sizes.
    """

    vista_score: float
    wave_score: float
    wave_peak: float
    wave_hop_of_peak: float
    wave_early: float
    channel_bias: float

    def as_vector(self) -> tuple[float, ...]:
        return (
            self.vista_score,
            self.wave_score,
            self.wave_peak,
            self.wave_hop_of_peak,
            self.wave_early,
            self.channel_bias,
        )


class Reranker(Protocol):
    """A reranker turns per-candidate ``WaveFeatures`` into a scalar score."""

    def score(self, features: WaveFeatures) -> float:
        ...


@dataclass(frozen=True)
class LinearReranker:
    """Weighted linear combination of the exposed WaveFeatures.

    ``weights`` maps feature names (see ``FEATURE_NAMES``) to coefficients.
    Missing names default to 0.0 (feature ignored). ``bias`` shifts the
    combined score; it is optional and rarely needed for ranking.

    ``LinearReranker.default()`` couples Vista similarity to Wave's final
    activation and arrival time. It rewards candidates reached early in the
    Wave trajectory, rather than treating all settled activation as equally
    useful. The fixed reference benchmark regression-tests this combination
    against the legacy heuristic. Users who wish to fit their own reranker
    can use ``LinearReranker.from_dict`` after training with any linear model
    on the ``as_vector`` features.
    """

    weights: Mapping[str, float] = field(default_factory=dict)
    bias: float = 0.0

    def score(self, features: WaveFeatures) -> float:
        vector = features.as_vector()
        total = self.bias
        for name, value in zip(FEATURE_NAMES, vector):
            total += float(self.weights.get(name, 0.0)) * value
        return total

    @classmethod
    def from_dict(
        cls,
        weights: Mapping[str, float],
        *,
        bias: float = 0.0,
    ) -> "LinearReranker":
        for name in weights:
            if name not in FEATURE_NAMES:
                raise ValueError(
                    f"unknown feature name {name!r}; expected one of "
                    f"{FEATURE_NAMES}"
                )
        return cls(weights=dict(weights), bias=bias)

    @classmethod
    def default(cls) -> "LinearReranker":
        """Reference Wave + Vista readout calibrated for early arrival.

        Vista remains the dominant semantic ranker. Wave's final activation
        contributes a small relational signal, while a negative
        ``wave_hop_of_peak`` coefficient demotes candidates that only become
        active late in the traversal. The bias is an affine shift that keeps
        the reference score in a convenient range without changing ranking.

        On the bundled reference benchmark this combination raises mean
        Recall@5 from 0.516 for the legacy heuristic to 0.624. That result is
        a regression guard for the reference backend, not a claim that the
        same weights are optimal for every corpus.
        """
        return cls(
            weights={
                "vista_score": 0.65,
                "wave_score": 0.065,
                "wave_peak": 0.0,
                "wave_hop_of_peak": -0.325,
                "wave_early": 0.0,
                "channel_bias": 0.0,
            },
            bias=0.325,
        )


def build_wave_features(
    *,
    vista_score: float,
    wave_score: float,
    wave_peak: float,
    wave_hop_of_peak_index: int,
    wave_early_activation: float,
    hops: int,
) -> WaveFeatures:
    """Assemble a WaveFeatures from raw per-candidate signals.

    All output fields land in [0.0, 1.0]. ``wave_hop_of_peak`` is
    normalised against ``hops`` so different wave configurations produce
    comparable feature ranges.
    """

    def _clip(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    total_hops = max(1, hops)
    return WaveFeatures(
        vista_score=_clip(vista_score),
        wave_score=_clip(wave_score),
        wave_peak=_clip(wave_peak),
        wave_hop_of_peak=_clip(wave_hop_of_peak_index / total_hops),
        wave_early=_clip(wave_early_activation),
        channel_bias=(
            1.0 if (vista_score > 0 and wave_score > 0)
            else 0.5 if (vista_score > 0 or wave_score > 0)
            else 0.0
        ),
    )


def score_candidates(
    reranker: Reranker,
    features: Iterable[WaveFeatures],
) -> Sequence[float]:
    """Bulk score helper. Returns scores in the same order as ``features``."""

    return [reranker.score(feature) for feature in features]


__all__ = [
    "FEATURE_NAMES",
    "LinearReranker",
    "Reranker",
    "WaveFeatures",
    "build_wave_features",
    "score_candidates",
]
