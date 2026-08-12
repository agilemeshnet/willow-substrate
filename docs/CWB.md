# Context Window Builder

Layered per-query context assembly. Where `willow_substrate.context.ContextBuilder`
assembles ONE thing (a token-budgeted evidence packet), the
`willow_substrate.cwb.ContextWindowBuilder` assembles a LAYERED context
window with five distinct fields, so downstream consumers can decide
what to render, what to hide, and what to hand to an LLM.

Ports the shape of Scout's internal `icu/cwb.py` (developed for the
Willow project's own agent architecture) into willow-substrate as a
first-class composable primitive. The shape is general; the
implementation is not Scout-specific.

## The five layers

| Layer | Where it comes from | What it is |
|---|---|---|
| **standing** | events with `metadata.standing = True` | Durable identity / rules / commitments the caller has pinned. Always present. |
| **foreground** | events ranked by `metadata.salience` desc, then recency | What is most-attended right now. The "reference beams" for the vista layer. |
| **vista** | `RetrievalBackend.query(seed_event_ids=foreground_ids, wave_hops=0)` | Single-hop retrieval seeded from both the query text AND the foreground beams. |
| **wave** | `RetrievalBackend.query(seed_event_ids=foreground_ids, wave_hops>0)` | Multi-hop damped spreading activation from the same seeds. |
| **prosoche** | `ProsocheMonitor.bands()` | Freshness colours for registered data sources: fresh / amber / stale / dead. Tells you when your context is built from stale substrate. |

Layers surface additively. If a token budget matters, the caller
decides layer-priority (standing > foreground > vista > wave is the
suggested importance order).

## Quick start

```python
from willow_substrate.store import EventStore
from willow_substrate.backends.factory import make_relational_backend
from willow_substrate.cwb import ContextWindowBuilder
from willow_substrate.prosoche import ProsocheMonitor

store = EventStore("~/.willow")
backend = make_relational_backend(store)

# Optional: register data sources and their expected refresh cadence
prosoche = ProsocheMonitor(store.home / "prosoche.db")
prosoche.register("email_sync", expected_interval_s=900.0)  # every 15 min
prosoche.register("calendar_sync", expected_interval_s=3600.0)  # every hour

cwb = ContextWindowBuilder(
    store,
    retrieval_backend=backend,
    prosoche=prosoche,
    foreground_default_k=15,
)

# Register a standing event (identity / rule)
store.append(
    "IDENTITY: I am Peter and this is my ledger",
    actor="peter", kind="identity", session_id="standing",
    metadata={"standing": True, "salience": 100.0},
)

# Every time a source refreshes, touch it
prosoche.touch("email_sync")

# Assemble a context window for a query
window = cwb.build("what did I discuss with Sarah about roofing?")

# Layered inspection
for event in window.standing:
    print("STANDING:", event.content)
for event in window.foreground:
    print("FOREGROUND:", event.content)
for ev in window.vista.evidence:
    print("VISTA:", ev.event.content, "(score:", ev.score, ")")
for ev in window.wave.evidence:
    print("WAVE:", ev.event.content)
for band in window.prosoche:
    print(f"PROSOCHE: {band.name} = {band.band}")
```

## Standing events

Any event with `metadata.standing = True` lands in the standing layer.
Standing events are also eligible for the foreground salience ranking,
so a high-salience standing event (identity, guiding principle) shows
up in both layers, which is the intended shape: it is present as
durable ground AND currently top-of-mind.

Standing status can be revoked by superseding the event (via
`store.correct(event_id, ...)`) with a supersession whose metadata does
NOT set `standing = True`. Additive law: the original standing event
stays in history; only its active successor loses the flag.

## Foreground salience

Rank order:

1. `metadata.salience` (float) descending
2. `event.seq` (recency) descending as tie-breaker

Events with no `salience` metadata default to `0.0`. Callers who want a
richer salience computation (weighted by recall frequency, degree in
the connection graph, freshness) should compute it externally and stamp
it on events at append-time, or supersede events with re-scored
metadata over time.

## Vista and Wave layers

Both use the caller's `retrieval_backend`. The vista layer runs the
backend with `wave_hops=0` (single-hop, cosine + BM25 + optional dense
via the hybrid + Voyage stack). The wave layer runs it with `wave_hops
> 0` (multi-hop damped spreading activation across the event-waypoint
lattice). Both are seeded with the foreground event IDs so retrieval
reflects "what am I attending to right now" alongside "what matches
the query text."

Compose with `ConsolidationBackend` (Hou et al. 2024) to add time-decay
+ recall-frequency scoring to both layers automatically.

## Prosoche

A stale bundle looks identical to a fresh one; prosoche is how you
tell the difference. Register each source that feeds your context with
an expected refresh interval; touch it when it refreshes; ask for
`bands()` to get current freshness colours per source. Default bands:

- `fresh`: less than 1.5x expected
- `amber`: 1.5x to 3x expected
- `stale`: 3x to 10x expected
- `dead`: more than 10x expected

Multipliers are per-source configurable. Bands are `unknown` for
registered sources that have never been touched.

Sidecar SQLite at `store.home / prosoche.db`. Separate from the
event ledger, so the ledger stays hash-chained and immutable.

## Design intent

The public substrate should let anyone build the same shape of
per-user active context assembly that Scout's own cwb.py builds for
Scout. Scout's cwb.py stays personal; `willow_substrate.cwb` is the
generic version, wired to any backend, composable with the reflection
layer (meditate + dream) and the consolidation wrapper (Hou 2024) that
already ship.
