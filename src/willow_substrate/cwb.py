"""Context Window Builder: layered per-query context assembly.

Ports the shape of Scout's ``icu/cwb.py`` into willow-substrate as a
first-class composable primitive. Where the existing ``ContextBuilder``
assembles ONE thing (a token-budgeted evidence packet), the
``ContextWindowBuilder`` assembles a LAYERED context window with five
distinct fields:

    standing      -- durable identity/rules the caller has pinned
                     (events with metadata.standing == True)
    foreground    -- salience-ranked pointers to what is most-attended now
                     (events ranked by metadata.salience, then recency)
    vista         -- single-hop retrieval seeded from the query + the
                     foreground beams (VistaResult)
    wave          -- damped multi-hop spreading activation from the same
                     seeds (VistaResult with wave_hops > 0)
    prosoche      -- freshness bands stamped on registered data sources
                     via the ProsocheMonitor sidecar

The four surfaces layer additively; the token budget is honored last so
the layers survive the budget in importance order (standing > foreground
> vista > wave). Consumers see each layer separately so they can decide
what to render, what to hide, and what to hand to an LLM.

Nothing here is Scout-specific; the primitives are portable. Willow's
own cwb.py stays personal; this module is the public shape.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from willow_substrate.banks import Bank, load_banks
from willow_substrate.events import Event
from willow_substrate.prosoche import ProsocheMonitor, SourceBand
from willow_substrate.salience import (
    RECENCY_HALF_LIFE_DAYS,
    is_standing,
    score_events,
)
from willow_substrate.store import EventStore
from willow_substrate.vista import VistaResult


class _RelBackend(Protocol):
    def query(
        self,
        text: str,
        *,
        seed_event_ids=(),
        limit: int = 8,
        wave_hops: int = 4,
    ) -> VistaResult: ...


@dataclass(frozen=True)
class ContextWindow:
    """The layered context surface for one query.

    Every field is separately inspectable so callers can decide what
    survives the token budget and what gets handed to the LLM.
    """

    query: str
    banks: tuple[Bank, ...]
    standing: tuple[Event, ...]
    foreground: tuple[Event, ...]
    vista: VistaResult | None
    wave: VistaResult | None
    prosoche: tuple[SourceBand, ...]
    trace: tuple[str, ...] = field(default_factory=tuple)

    @property
    def all_event_ids(self) -> tuple[str, ...]:
        """Deduplicated event ids across all EVENT layers, in importance order.

        Banks are files, not events, so they do not appear here; callers who
        want them should read ``window.banks`` directly.
        """
        seen: dict[str, None] = {}
        for event in self.standing:
            seen[event.id] = None
        for event in self.foreground:
            seen.setdefault(event.id, None)
        if self.vista is not None:
            for ev in self.vista.evidence:
                seen.setdefault(ev.event.id, None)
        if self.wave is not None:
            for ev in self.wave.evidence:
                seen.setdefault(ev.event.id, None)
        return tuple(seen.keys())


class ContextWindowBuilder:
    """Assemble a layered ContextWindow for a query.

    Composable with any RelationalBackend that produces a VistaResult
    (sparse, hybrid, voyage, consolidation-wrapped, whatever). The
    prosoche monitor is optional; if omitted, the window's ``prosoche``
    field is empty.
    """

    def __init__(
        self,
        store: EventStore,
        *,
        retrieval_backend: _RelBackend,
        prosoche: ProsocheMonitor | None = None,
        foreground_default_k: int = 15,
        vista_limit_default: int = 8,
        wave_hops_default: int = 4,
    ):
        self.store = store
        self.retrieval_backend = retrieval_backend
        self.prosoche = prosoche
        self.foreground_default_k = foreground_default_k
        self.vista_limit_default = vista_limit_default
        self.wave_hops_default = wave_hops_default

    def build(
        self,
        query: str,
        *,
        foreground_k: int | None = None,
        vista_limit: int | None = None,
        wave_hops: int | None = None,
        include_standing: bool = True,
        include_wave: bool = True,
        include_banks: bool = True,
    ) -> ContextWindow:
        """Assemble the five-layer context window for one query.

        Layers, most-durable to most-ephemeral:
        - banks: files loaded from WILLOW_HOME (identity.md, ground.md,
          optional banks/*.md); included whole, cost paid before flow.
        - standing: events flagged as standing/foundational; retrieved
          regardless of query relevance.
        - foreground: top-salience events by the five-signal scorer.
        - vista: single-hop retrieval seeded by query text + foreground ids.
        - wave: multi-hop damped spreading activation from same seeds.
        - prosoche: freshness colours for registered data sources.
        """
        trace: list[str] = []

        banks = load_banks(self.store.home) if include_banks else ()
        trace.append(
            f"banks: {len(banks)} constitutional files "
            f"({sum(b.estimated_tokens for b in banks)} tokens floor)"
        )

        standing = (
            self._standing_events() if include_standing else ()
        )
        trace.append(f"standing: {len(standing)} pinned events")

        k = foreground_k if foreground_k is not None else self.foreground_default_k
        foreground = self._foreground_events(k, query=query)
        trace.append(
            f"foreground: {len(foreground)} top-salience events "
            f"(via five-signal scorer)"
        )

        # Vista layer: seed from foreground so the retrieval reflects
        # 'what I am attending to now' as well as 'what matches the query'.
        seed_ids = tuple(event.id for event in foreground)
        vista_k = vista_limit if vista_limit is not None else self.vista_limit_default
        vista = self.retrieval_backend.query(
            query,
            seed_event_ids=seed_ids,
            limit=vista_k,
            wave_hops=0,
        )
        trace.append(f"vista: {len(vista.evidence)} evidence events")

        # Wave layer: same seeds, this time with multi-hop spreading.
        wave: VistaResult | None = None
        if include_wave:
            hops = wave_hops if wave_hops is not None else self.wave_hops_default
            wave = self.retrieval_backend.query(
                query,
                seed_event_ids=seed_ids,
                limit=vista_k,
                wave_hops=hops,
            )
            trace.append(
                f"wave: {len(wave.evidence)} evidence events, hops={hops}"
            )

        prosoche_bands: tuple[SourceBand, ...] = ()
        if self.prosoche is not None:
            prosoche_bands = self.prosoche.bands()
            trace.append(
                f"prosoche: {len(prosoche_bands)} sources monitored"
            )

        return ContextWindow(
            query=query,
            banks=banks,
            standing=standing,
            foreground=foreground,
            vista=vista,
            wave=wave,
            prosoche=prosoche_bands,
            trace=tuple(trace),
        )

    def _standing_events(self) -> tuple[Event, ...]:
        """Return every active event flagged as standing / foundational.

        Uses ``willow_substrate.salience.is_standing`` so the SAME rule
        that decides standing for the retrieval-time scorer decides it
        here. Keeps the two surfaces in agreement without duplicating
        the flag-check logic.
        """
        return tuple(
            event
            for event in self.store.events(limit=10_000, active_only=True)
            if is_standing(event.metadata)
        )

    def _foreground_events(
        self, k: int, *, query: str = ""
    ) -> tuple[Event, ...]:
        """Top-k events by the five-signal salience scorer.

        Delegates to ``willow_substrate.salience.score_events`` so
        ranking is explainable (standing + citation + reflection +
        recency + query) rather than a single opaque signal. A stored
        ``metadata.salience`` float still contributes when present via
        the standing branch of the scorer, but is no longer the sole
        input, so callers who never set that field also get a
        principled ranking.
        """
        active = list(self.store.events(limit=10_000, active_only=True))
        if not active:
            return ()
        scores = score_events(active, query=query)
        active.sort(key=lambda ev: -scores[ev.id].total)
        return tuple(active[:k])
