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

from willow_substrate.events import Event
from willow_substrate.prosoche import ProsocheMonitor, SourceBand
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
    standing: tuple[Event, ...]
    foreground: tuple[Event, ...]
    vista: VistaResult | None
    wave: VistaResult | None
    prosoche: tuple[SourceBand, ...]
    trace: tuple[str, ...] = field(default_factory=tuple)

    @property
    def all_event_ids(self) -> tuple[str, ...]:
        """Deduplicated ids across all layers, in importance order."""
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
    ) -> ContextWindow:
        """Assemble the four-plus-one layered context for one query."""
        trace: list[str] = []

        standing = (
            self._standing_events() if include_standing else ()
        )
        trace.append(f"standing: {len(standing)} pinned events")

        k = foreground_k if foreground_k is not None else self.foreground_default_k
        foreground = self._foreground_events(k)
        trace.append(f"foreground: {len(foreground)} top-salience events")

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
            standing=standing,
            foreground=foreground,
            vista=vista,
            wave=wave,
            prosoche=prosoche_bands,
            trace=tuple(trace),
        )

    def _standing_events(self) -> tuple[Event, ...]:
        """Return every active event whose metadata sets standing=True."""
        return tuple(
            event
            for event in self.store.events(limit=10_000, active_only=True)
            if bool(event.metadata.get("standing"))
        )

    def _foreground_events(self, k: int) -> tuple[Event, ...]:
        """Top-k events ranked by metadata.salience, then by recency.

        Events with no salience metadata default to 0.0. Ties break
        towards the more recent event (higher seq).
        """
        active = self.store.events(limit=10_000, active_only=True)
        scored = [
            (float(event.metadata.get("salience", 0.0)), event.seq, event)
            for event in active
        ]
        scored.sort(key=lambda triple: (triple[0], triple[1]), reverse=True)
        return tuple(event for _, _, event in scored[:k])
