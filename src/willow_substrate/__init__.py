"""Willow: continuity for long-running human-AI work."""

from willow_substrate.connections import ConnectionCandidate, find_connections
from willow_substrate.context import ContextBuilder, ContextPacket
from willow_substrate.events import Event, EventHit
from willow_substrate.facts import FactCheckResult, FactLedger, FactState
from willow_substrate.foveation import FoveationResult, Foveator
from willow_substrate.readout import (
    FEATURE_NAMES,
    LinearReranker,
    Reranker,
    WaveFeatures,
    build_wave_features,
    score_candidates,
)
from willow_substrate.research import Citation, ResearchLedger, ResearchState
from willow_substrate.samples import (
    EvaluationCheck,
    SampleEvaluation,
    SampleLoadReport,
    evaluate_temporal_sample,
    load_temporal_sample,
)
from willow_substrate.store import EventStore
from willow_substrate.vista import (
    ReferenceBeam,
    RelationalBackend,
    Vista,
    VistaBackend,
    VistaEvidence,
    VistaMatch,
    VistaResult,
    Waypoint,
)

__all__ = [
    "Citation",
    "ConnectionCandidate",
    "ContextBuilder",
    "ContextPacket",
    "Event",
    "EventHit",
    "EventStore",
    "EvaluationCheck",
    "FactCheckResult",
    "FactLedger",
    "FactState",
    "FEATURE_NAMES",
    "FoveationResult",
    "Foveator",
    "LinearReranker",
    "Reranker",
    "ResearchLedger",
    "ResearchState",
    "ReferenceBeam",
    "RelationalBackend",
    "SampleEvaluation",
    "SampleLoadReport",
    "Vista",
    "VistaBackend",
    "VistaEvidence",
    "VistaMatch",
    "VistaResult",
    "WaveFeatures",
    "Waypoint",
    "build_wave_features",
    "evaluate_temporal_sample",
    "find_connections",
    "load_temporal_sample",
    "score_candidates",
]

__version__ = "0.2.2"
