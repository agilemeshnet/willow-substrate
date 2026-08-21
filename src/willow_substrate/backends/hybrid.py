"""Hybrid recall via Reciprocal Rank Fusion (RRF).

Runs multiple RelationalBackend implementations against the same query and
fuses their rankings with Cormack et al. 2009 (SIGIR) Reciprocal Rank
Fusion: score(doc) = sum over backends of 1 / (k + rank_in_backend(doc)),
default k=60.

RRF is the strong practical default for combining retrieval signals of
different provenance (sparse term-frequency, dense semantic, graph-hop)
because it fuses ranks rather than raw scores. Raw scores from different
backends are on incomparable scales; ranks are always in the same unit.
Empirically RRF matches or beats weighted-score fusion across a wide
range of tasks with no tuning.

This backend produces the same VistaResult shape as any other
RelationalBackend, so downstream context / foveation / CLI code reads it
unchanged. It is available whenever willow-substrate is installed; the
dense component is included automatically when the [vista] extra is also
present.

Design notes:

- Sub-backends compose sparse VistaBackend + inline BM25 + optional dense
  Voyage. If Voyage is not available or not configured, hybrid degrades
  gracefully to sparse+BM25.
- Each sub-backend contributes only its ranked event ids to the fusion.
  Their internal scores are used for tie-breaking within a single
  backend's ranking but not across backends (that would defeat RRF).
- The returned VistaResult borrows the sparse backend's Vistas and
  ReferenceBeams for the surround; the evidence list carries the fused
  RRF order. The trace records which backends contributed and their
  respective top-hit event ids for debuggability.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from willow_substrate.events import Event
from willow_substrate.store import EventStore
from willow_substrate.vista import (
    ReferenceBeam,
    Vista,
    VistaBackend,
    VistaEvidence,
    VistaMatch,
    VistaResult,
)


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenise(text: str) -> list[str]:
    return [tok.lower() for tok in _TOKEN_RE.findall(text) if len(tok) > 1]


@dataclass
class _BackendContribution:
    """One backend's ranked list of event ids for a single query."""

    name: str
    ranking: list[str] = field(default_factory=list)


class _BM25Recall:
    """Inline Okapi BM25 recall, k1=1.5, b=0.75.

    Fits over the active event corpus at construction; refits on demand
    if the corpus changes (see refit()). Deliberately self-contained so
    hybrid recall has a strong term-frequency baseline without pulling
    an external dep.
    """

    def __init__(self, store: EventStore, k1: float = 1.5, b: float = 0.75):
        self.store = store
        self.k1 = k1
        self.b = b
        self._events: list[Event] = []
        self._tokens: list[list[str]] = []
        self._doc_freq: Counter[str] = Counter()
        self._avg_len: float = 0.0
        self._fit_signature: tuple[str, ...] = ()
        self.refit()

    def refit(self) -> None:
        events = self.store.events(limit=10_000, active_only=True)
        signature = tuple(event.id for event in events)
        if signature == self._fit_signature and self._events:
            return
        self._events = events
        self._tokens = [_tokenise(event.content) for event in events]
        self._doc_freq = Counter()
        for tokens in self._tokens:
            for term in set(tokens):
                self._doc_freq[term] += 1
        total_len = sum(len(t) for t in self._tokens)
        self._avg_len = total_len / max(1, len(self._tokens))
        self._fit_signature = signature

    def rank(self, text: str, *, limit: int) -> list[str]:
        self.refit()
        n = len(self._events)
        if n == 0:
            return []
        q_terms = _tokenise(text)
        if not q_terms:
            return []
        scores = [0.0] * n
        for i, tokens in enumerate(self._tokens):
            counts = Counter(tokens)
            doc_len = len(tokens)
            for term in q_terms:
                if term not in counts:
                    continue
                df = self._doc_freq.get(term, 0)
                if df == 0:
                    continue
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
                tf = counts[term]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * doc_len / max(1e-9, self._avg_len)
                )
                scores[i] += idf * (numerator / denominator)
        ranked = sorted(enumerate(scores), key=lambda kv: kv[1], reverse=True)
        return [
            self._events[i].id for i, score in ranked if score > 0.0
        ][:limit]


def _sparse_ranking(
    backend: VistaBackend,
    query: str,
    *,
    limit: int,
    seeds: tuple[str, ...],
    reranker=None,
) -> list[str]:
    result = backend.query(
        query, seed_event_ids=seeds, limit=max(limit, 8), reranker=reranker
    )
    return [item.event.id for item in result.evidence][:limit]


def _rrf_fuse(
    contributions: Iterable[_BackendContribution],
    *,
    k_rrf: int = 60,
) -> list[tuple[str, float]]:
    """Standard Cormack-Clarke-Buettcher (2009) Reciprocal Rank Fusion.

    Returns [(event_id, rrf_score), ...] sorted best-first. Event ids that
    appear in more than one backend's ranking accumulate score from each.
    k=60 is the constant the original paper recommends; not sensitive
    enough to warrant tuning under normal use.
    """
    fused: dict[str, float] = defaultdict(float)
    for contribution in contributions:
        for rank, event_id in enumerate(contribution.ranking, start=1):
            fused[event_id] += 1.0 / (k_rrf + rank)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


class HybridRecallBackend:
    """RelationalBackend that fuses sparse + BM25 + optional dense via RRF.

    Substitutes for VistaBackend and VoyageVistaBackend at the read side;
    produces the same VistaResult shape so context / foveation / CLI
    consume it unchanged.

    Instantiate the dense sub-backend explicitly (via ``dense=``) or let
    the constructor auto-detect the Voyage backend when the [vista]
    extra is installed and VOYAGE_API_KEY is set. Without dense, hybrid
    falls back to fusing sparse + BM25.
    """

    def __init__(
        self,
        store: EventStore,
        *,
        max_events: int | None = None,
        wave_damping: float | None = None,
        dense=None,
        k_rrf: int = 60,
        sparse: VistaBackend | None = None,
        bm25: _BM25Recall | None = None,
        reranker=None,
        use_readout: bool = False,
    ):
        if k_rrf < 1:
            raise ValueError("k_rrf must be positive")
        self.store = store
        self.k_rrf = k_rrf
        if reranker is None and use_readout:
            from willow_substrate.readout import LinearReranker
            reranker = LinearReranker.default()
        self.reranker = reranker
        sparse_kwargs = {}
        if max_events is not None:
            sparse_kwargs["max_events"] = max_events
        if wave_damping is not None:
            sparse_kwargs["wave_damping"] = wave_damping
        self.sparse = sparse or VistaBackend(store, **sparse_kwargs)
        self.bm25 = bm25 or _BM25Recall(store)
        if dense is not None:
            self.dense = dense
        else:
            self.dense = _auto_dense(store)

    def query(
        self,
        query: str,
        *,
        seed_event_ids: Iterable[str] = (),
        limit: int = 8,
        wave_hops: int = 4,
    ) -> VistaResult:
        seeds = tuple(seed_event_ids)
        # Ask each sub-backend for a wider ranking than the requested limit
        # so RRF has room to fuse; downstream we take the top-limit after
        # fusion. The 5x factor is a practical default; smaller corpora
        # bottom out at the corpus size anyway.
        wide = max(limit * 5, 20)

        contributions: list[_BackendContribution] = []

        sparse_ranking = _sparse_ranking(
            self.sparse, query, limit=wide, seeds=seeds,
            reranker=self.reranker,
        )
        contributions.append(
            _BackendContribution(name="sparse", ranking=sparse_ranking)
        )

        bm25_ranking = self.bm25.rank(query, limit=wide)
        contributions.append(
            _BackendContribution(name="bm25", ranking=bm25_ranking)
        )

        if self.dense is not None:
            try:
                dense_result = self.dense.query(
                    query,
                    seed_event_ids=seeds,
                    limit=wide,
                    wave_hops=wave_hops,
                )
                dense_ranking = [
                    item.event.id for item in dense_result.evidence
                ]
                contributions.append(
                    _BackendContribution(name="dense", ranking=dense_ranking)
                )
            except Exception as exc:  # pragma: no cover
                # Never let a broken sub-backend deny recall; hybrid degrades
                # to sparse + BM25 and records the failure in the trace.
                contributions.append(
                    _BackendContribution(
                        name=f"dense-error:{type(exc).__name__}", ranking=[]
                    )
                )

        fused = _rrf_fuse(contributions, k_rrf=self.k_rrf)[:limit]

        # Fetch full events for the fused top-K.
        active_events = {
            event.id: event
            for event in self.store.events(limit=10_000, active_only=True)
        }
        evidence = [
            VistaEvidence(
                event=active_events[eid],
                score=score,
                vista_score=score,
                wave_score=0.0,
                vista_slugs=(),
                waypoints=(),
                channels=tuple(
                    contrib.name
                    for contrib in contributions
                    if eid in contrib.ranking[:limit * 2]
                ),
            )
            for eid, score in fused
            if eid in active_events
        ]

        # Reuse the sparse backend's projection for beams/matches/trace so
        # the read-side surround (waypoint reasoning, cluster identity) is
        # not lost by the fusion.
        sparse_full = self.sparse.query(
            query, seed_event_ids=seeds, limit=limit, wave_hops=wave_hops
        )

        trace_lines = [
            f"hybrid RRF k={self.k_rrf}",
            *(
                f"{c.name}: {len(c.ranking)} results"
                + (f" (top: {c.ranking[0]})" if c.ranking else "")
                for c in contributions
            ),
            f"fused: {len(fused)} unique events; returning top {len(evidence)}",
            *sparse_full.trace,
        ]

        return VistaResult(
            query=query,
            seed_event_ids=seeds,
            reference_beams=sparse_full.reference_beams,
            matches=sparse_full.matches,
            evidence=tuple(evidence),
            trace=tuple(trace_lines),
        )


def _auto_dense(store: EventStore):
    """Return an initialised dense backend if [vista] extra + key both present."""
    import os

    if not os.environ.get("VOYAGE_API_KEY"):
        return None
    try:
        from willow_substrate.backends.vista_voyage import VoyageVistaBackend
    except ImportError:
        return None
    try:
        return VoyageVistaBackend(store)
    except Exception:  # pragma: no cover
        return None
