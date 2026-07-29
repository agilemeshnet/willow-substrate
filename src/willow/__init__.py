"""Willow: continuity for long-running human-AI work."""

from willow.banks import Bank, load_banks, scaffold_banks
from willow.connections import ConnectionCandidate, find_connections
from willow.context import ContextBuilder, ContextPacket
from willow.corpus import ImportedFile, ImportReport, import_markdown
from willow.events import Event, EventHit
from willow.facts import FactCheckResult, FactLedger, FactState
from willow.foveation import FoveationResult, Foveator
from willow.research import Citation, ResearchLedger, ResearchState
from willow.salience import (
    SalienceScore,
    SalienceSignal,
    rank_selection,
    score_event,
    score_events,
)
from willow.samples import (
    EvaluationCheck,
    SampleEvaluation,
    SampleLoadReport,
    evaluate_temporal_sample,
    load_temporal_sample,
)
from willow.store import EventStore
from willow.vista import (
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
    "Bank",
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
    "FoveationResult",
    "Foveator",
    "ImportReport",
    "ImportedFile",
    "ResearchLedger",
    "ResearchState",
    "ReferenceBeam",
    "RelationalBackend",
    "SalienceScore",
    "SalienceSignal",
    "SampleEvaluation",
    "SampleLoadReport",
    "Vista",
    "VistaBackend",
    "VistaEvidence",
    "VistaMatch",
    "VistaResult",
    "Waypoint",
    "evaluate_temporal_sample",
    "find_connections",
    "import_markdown",
    "load_banks",
    "load_temporal_sample",
    "rank_selection",
    "scaffold_banks",
    "score_event",
    "score_events",
]

__version__ = "0.2.0"
