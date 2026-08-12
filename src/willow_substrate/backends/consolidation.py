"""Time-decay + recall-frequency consolidation wrapper for any RelationalBackend.

Ports Hou, Tamoto & Miyashita (CHI EA '24, arXiv 2404.00573v1) into a
composable RelationalBackend wrapper. The paper's core insight: human
memory retention combines relevance, elapsed time, and recall
frequency; frequently-recalled memories decay slower than never-
recalled ones.

Given a wrapped backend's ranked result, this wrapper reweights each
event by the paper's normalised recall probability:

    p(t) = [1 - exp(-r * exp(-t / (tau * g_n)))] / [1 - exp(-1)]

where
    r     = the wrapped backend's score for the event (treated as relevance)
    t     = elapsed time since the event, in seconds
    tau   = configurable half-life scale (default 30 days), so t/tau
            normalises the raw seconds into the paper's dimensionless
            time unit
    g_n   = the paper's consolidation factor:
            g_0 = 1, g_n = g_{n-1} + S(t_i) where t_i is the time
            elapsed at the i-th prior recall and S(t) = (1 - e^-t) / (1 + e^-t)

After reweighting, events are re-sorted by the new p(t) score and each
returned event's recall is recorded to the sidecar so future queries
see updated consolidation.

The wrapper is composable: it takes any object with a ``query(text,
...)`` method returning a VistaResult and returns a VistaResult with
the same evidence events but updated scores and channels. The wrapped
backend is untouched; the store is untouched; recall statistics live
in a separate sidecar file so the ledger stays hash-chained and pure.
"""
from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from willow_substrate.events import Event
from willow_substrate.recall_stats import RecallStats
from willow_substrate.store import EventStore
from willow_substrate.vista import VistaEvidence, VistaResult


DEFAULT_TAU_S = 30.0 * 86400.0  # 30 days: a sensible-scale half-life
# Denominator normalising the paper's p(t) so p at t=0 with r=1 = 1.
_NORMALISATION = 1.0 - math.exp(-1.0)


def _consolidation_sigmoid(delta_s: float) -> float:
    """S(t) = (1 - e^-t) / (1 + e^-t), evaluated on a dimensionless t.

    The paper uses t as a normalised time unit; here we divide the
    real elapsed seconds by tau so the argument is comparable in
    magnitude to the p(t) argument.
    """
    if delta_s <= 0.0:
        return 0.0
    # Use tanh identity: (1-e^-t)/(1+e^-t) = tanh(t/2). Numerically
    # stable and avoids exp overflow at large t.
    return math.tanh(delta_s / 2.0)


def _g_n_from_history(
    recalled_at: list[datetime],
    tau_s: float,
    now: datetime,
) -> float:
    """Compute the paper's g_n from a full recall history.

    g_0 = 1. Each prior recall at time t_i contributes S((now - t_i)/tau)
    (the wider the gap since the recall, the closer S is to 1; a recall
    right now adds ~0). This means old recalls contribute more
    consolidation than very recent ones, which matches the paper's
    "spaced repetition helps" intuition.
    """
    g = 1.0
    for at in recalled_at:
        delta_s = (now - at).total_seconds()
        g += _consolidation_sigmoid(delta_s / tau_s)
    return g


def _recall_probability(
    relevance: float,
    elapsed_s: float,
    g_n: float,
    tau_s: float,
) -> float:
    """The paper's p(t), normalised so p(t=0, r=1) = 1.

    Clips relevance into [0, +inf) so a negative RRF score does not
    invert the exponential.
    """
    r = max(0.0, relevance)
    if r == 0.0:
        return 0.0
    a = 1.0 / g_n
    # Exponent bounded below by -700 to avoid math.exp underflow-to-zero
    # producing NaN downstream when relevance is tiny.
    inner_arg = -a * elapsed_s / tau_s
    inner = math.exp(max(inner_arg, -700.0))
    p_raw = 1.0 - math.exp(-r * inner)
    return p_raw / _NORMALISATION


class ConsolidationBackend:
    """RelationalBackend wrapper: applies time-decay + recall-frequency scoring.

    Composes with any object that exposes ``query(text, *, seed_event_ids=(),
    limit=..., wave_hops=...) -> VistaResult`` so the sparse, dense, and
    hybrid backends are all wrappable.

    Recall counts are recorded per returned event on each query. To
    reproduce the paper's evaluation semantics: the SAME memory
    retrieved multiple times gets a growing g_n across queries, which
    slows its decay.
    """

    def __init__(
        self,
        inner,
        store: EventStore,
        *,
        recall_stats: RecallStats | None = None,
        tau_s: float = DEFAULT_TAU_S,
        now_fn: Callable[[], datetime] | None = None,
    ):
        if tau_s <= 0:
            raise ValueError("tau_s must be positive")
        self.inner = inner
        self.store = store
        self.tau_s = tau_s
        self.now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        if recall_stats is None:
            recall_stats = RecallStats(store.home / "recall_stats.db")
        self.recall_stats = recall_stats

    def query(
        self,
        text: str,
        *,
        seed_event_ids=(),
        limit: int = 8,
        wave_hops: int = 4,
    ) -> VistaResult:
        # Widen the inner search so consolidation reordering has room
        # to move events into the top-K. The paper's mechanism moves
        # frequently-recalled events UP the ranking; if we only see the
        # top-K from the inner backend, we cannot promote anything from
        # below the cut. 5x is a defensive multiplier; still bounded.
        inner_limit = max(limit * 5, limit)
        raw = self.inner.query(
            text,
            seed_event_ids=seed_event_ids,
            limit=inner_limit,
            wave_hops=wave_hops,
        )
        if not raw.evidence:
            return raw

        now = self.now_fn()
        reweighted: list[tuple[float, VistaEvidence]] = []
        for evidence in raw.evidence:
            event = evidence.event
            elapsed_s = _elapsed_seconds(event, now)
            history = self.recall_stats.history(event.id)
            g_n = _g_n_from_history(history, self.tau_s, now)
            new_score = _recall_probability(
                relevance=evidence.score,
                elapsed_s=elapsed_s,
                g_n=g_n,
                tau_s=self.tau_s,
            )
            new_channels = tuple(evidence.channels) + ("consolidation",)
            reweighted.append(
                (
                    new_score,
                    replace(
                        evidence,
                        score=new_score,
                        channels=new_channels,
                    ),
                )
            )

        # Sort by the consolidated score, descending; truncate to the
        # caller's requested limit. Recall bookkeeping only for what we
        # actually return, to keep the paper's semantics honest.
        reweighted.sort(key=lambda pair: pair[0], reverse=True)
        top = [ev for _, ev in reweighted[:limit]]
        self.recall_stats.record_recalls(
            (ev.event.id for ev in top), at=now
        )
        return VistaResult(
            query=raw.query,
            seed_event_ids=raw.seed_event_ids,
            reference_beams=raw.reference_beams,
            matches=raw.matches,
            evidence=tuple(top),
            trace=raw.trace + (
                f"consolidation reweighted {len(reweighted)} candidates, "
                f"kept top {len(top)}",
            ),
        )


def _elapsed_seconds(event: Event, now: datetime) -> float:
    """Seconds between an event's timestamp and now.

    Willow's timestamps are timezone-aware ISO-8601. Naive timestamps
    are unexpected here (the store validator rejects them) but if one
    slips through we treat it as UTC to avoid a TypeError.
    """
    try:
        ts = datetime.fromisoformat(event.timestamp.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    delta = (now - ts).total_seconds()
    return max(0.0, delta)
