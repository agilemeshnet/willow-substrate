# Extras: layered fidelity for the full Willow substrate

The base install is dependency-free and runs the continuity demo. Extras add
richer memory-and-recall backends from the full internal Willow substrate.

The package is not yet on PyPI. Install from a local checkout:

```bash
git clone https://github.com/agilemeshnet/willow-substrate.git
cd willow-substrate
python3 -m venv .venv && source .venv/bin/activate
pip install -e .                       # zero-dep floor (continuity demo)
pip install -e ".[vista]"              # dense-embedding Vista + Wave recall
pip install -e ".[neo4j]"              # optional AuraDB graph projection
pip install -e ".[full]"               # the whole memory-and-recall stack
```

## Design principles

**Zero-dep floor stays load-bearing.** Every backend added by an extra is a
substitute for a component in the base install, not a required addition.
Someone who installs with no extras still gets a working continuity layer.

**Backends fail loud on missing deps.** Each backend module imports its
third-party dependencies at module scope. If the extra is not installed, the
resulting `ImportError` names the exact `pip install` incantation to fix it,
so a caller never gets a confusing runtime failure ten function-calls deep.

**Same evidence contract.** Every Vista/Wave-shaped backend implements the
`RelationalBackend` Protocol in `src/willow/vista.py`. The rest of Willow
(context composer, foveation, CLI, hooks) reads the same `VistaResult` shape
whichever backend is in use. Swap the backend, keep the reader.

**Optional-dependencies over adapters.** Where an adapter would need a
credential (Neo4j URL, Voyage API key), the backend reads it from the
environment or an explicit argument at construction. The bundle never ships
secrets.

## Activation

Installing an extra is necessary but not, on its own, sufficient. The read
paths (CLI commands `context`, `foveate`, `vista`; the connections finder;
the context composer) resolve their backend through the factory in
`willow_substrate.backends.factory.make_relational_backend(store, name=...)`.

The factory's resolution rules:

1. Explicit `name=` argument (or `--backend` CLI flag) wins.
2. Else the `WILLOW_BACKEND` env var (`voyage`, `sparse`, or `auto`).
3. Else `auto`: return the Voyage dense backend when the `[vista]` extra is
   importable AND `VOYAGE_API_KEY` is set; fall back to the dependency-free
   sparse backend otherwise.

Auto only activates the dense backend when a working configuration exists,
so `pip install -e ".[vista]"` does not turn a working sparse
read into a construction-time failure. To force the dense backend
regardless, pass `--backend voyage` or set `WILLOW_BACKEND=voyage`. To
force the sparse floor, pass `--backend sparse`.

`willow_substrate.backends.factory.active_backend_name()` reports what the factory
would resolve to, for logging or CI.

## Currently available extras

### `[vista]` — dense-embedding Vista + Wave recall

Substitutes for the dependency-free `VistaBackend`. Turns TF-IDF-style sparse
retrieval into dense semantic recall over the full event corpus, then computes
a damped multi-hop wave over the co-occurrence graph.

Dependencies: `numpy`, `scikit-learn`, `hdbscan`, `requests`.

Runtime environment: `VOYAGE_API_KEY` for embeddings (Voyage-4 by default;
free tier of 200M tokens across the voyage-4 family covers typical personal
corpora effectively free).

Embeddings are cached to `willow_voyage_embeddings.npz` next to the event
store; repeat calls do not re-embed. A corpus that grew by ten events only
pays for those ten. Model mismatch invalidates the cache so different
embedding dimensions never collide.

Concrete backend module: `willow_substrate.backends.vista_voyage.VoyageVistaBackend`.
Ports the internal Willow `thought-buckets/wave.py` + `mint_buckets.py` +
`voyage_embed.py` patterns. Produces the same `VistaResult` shape as the
dependency-free `VistaBackend` so downstream readers do not care which
backend produced the surround.

### `hybrid` — Reciprocal Rank Fusion of sparse + BM25 + optional dense

Always available; no extras required for the sparse + BM25 half. When
`[vista]` is installed and `VOYAGE_API_KEY` is set, the dense sub-backend
joins the fusion. Cormack et al. 2009 (SIGIR) RRF with k=60. Concrete
module: `willow_substrate.backends.hybrid.HybridRecallBackend`.

Fuses ranks, not scores, so the fusion is stable across sub-backends that
produce incomparable score scales. Any event that appears in more than one
sub-backend's top-K rises above events that appear in only one. Activate
via `--backend hybrid` on the CLI or `WILLOW_BACKEND=hybrid`.

Minimal example (mock embedder in the test suite; real usage sets
`VOYAGE_API_KEY`):

```python
from willow_substrate.store import EventStore
from willow_substrate.backends.vista_voyage import VoyageVistaBackend

store = EventStore()
backend = VoyageVistaBackend(store)  # reads VOYAGE_API_KEY from env
result = backend.query("recurrent connectome motifs", limit=8)
for item in result.evidence:
    print(item.event.short_id, item.score, item.channels)
```

### `[neo4j]` — AuraDB graph projection

Mirror the local event ledger into a Neo4j / AuraDB graph so federated
agents share a query surface, audits can traverse relationships, and
graph-native queries can join the ledger with other knowledge.

Dependencies: `neo4j`, `python-dotenv`.

Environment (or constructor args): `NEO4J_URI`, `NEO4J_USER`,
`NEO4J_PASSWORD`. Optional: `NEO4J_DATABASE`.

Concrete adapter: `willow_substrate.adapters.neo4j.Neo4jGraphAdapter`.

Graph shape:

- `(:Event {id, timestamp, kind, content, hash, seq})` one node per
  Willow event; MERGE on id so re-runs are idempotent.
- `(:Session {id})` grouping node per session.
- `(:Actor {name})` node per unique actor.
- `(:Event)-[:IN_SESSION]->(:Session)`
- `(:Event)-[:AUTHORED_BY]->(:Actor)`
- `(:Event)-[:SUPERSEDES]->(:Event)` for corrections.
- `(:Event)-[:DERIVES_FROM]->(:Event)` for meditations, engrams, and
  any event whose `derived_from` list is non-empty.

Minimal usage:

```python
from willow_substrate.store import EventStore
from willow_substrate.adapters.neo4j import Neo4jGraphAdapter

store = EventStore()
with Neo4jGraphAdapter() as adapter:
    adapter.ensure_constraints()   # one-time; idempotent
    adapter.mirror_all(store)      # one-shot full mirror
```

The adapter is deliberately one-way. The local ledger stays the single
writer; the graph is a projection audit and federation can read.

## Coming extras

The internal Willow substrate contains several more capabilities that will
land as their own extras as they are ported and tested against the public
event contract:

- **`[cognee]`** — optional Cognee ingestion of the `.md` corpus into
  AuraDB via the shared community ETL.
- **`[fable]`** — Anthropic Fable-5 specialist classifier with a
  substrate-aware verifier layer around it (the fable-specialist pattern
  used internally for narrow yes/no gates that a general-purpose model
  handles worse than a small specialist plus verification).
- **`[federation]`** — cypher-RETURN reply protocol and curriculum
  onboarding for sovereign Willow instances sharing an AuraDB.

Additional adapters land as they earn their place.

## Choosing between the zero-dep floor and `[vista]`

| Question | Zero-dep floor | `[vista]` (Voyage) |
|---|---|---|
| Runs offline? | Yes | Requires internet for embed calls |
| Requires API key? | No | Yes (VOYAGE_API_KEY, free tier available) |
| Retrieval quality on small corpora (< 200 events)? | Comparable | Comparable, with better recall on paraphrase |
| Retrieval quality on large corpora (> 1000 events)? | Degrades as TF-IDF thins out | Holds up (dense semantic space) |
| Cost per 1000 events (Voyage-4)? | Zero | ~$0.00015 |
| Reproduces the internal Willow substrate? | Partial | Yes |

The zero-dep floor is the honest starting point. `[vista]` is the recall
power-up that lets the substrate scale to the corpus sizes internal Willow
actually runs (~4k events).

## For contributors

To add a new extra:

1. Declare the extra in `pyproject.toml` under `[project.optional-dependencies]`
   with a short comment naming the backend it enables.
2. Add the backend module under `src/willow/backends/` (or extend the
   `src/willow/adapters/` package for outward-facing integrations).
3. Import the third-party dependencies at module scope so a missing extra
   surfaces immediately with a clear message.
4. Implement the appropriate Protocol from `src/willow/vista.py`,
   `src/willow/context.py`, or wherever the substitution point lives.
5. Add tests that skip gracefully when the extra is not installed and run
   for real when it is. Pattern:

    ```python
    import unittest
    try:
        import willow_substrate.backends.vista_voyage  # noqa: F401
        HAS_VISTA_EXTRA = True
    except ImportError:
        HAS_VISTA_EXTRA = False

    @unittest.skipUnless(HAS_VISTA_EXTRA, "install [vista] to run this test")
    class VoyageVistaTests(unittest.TestCase):
        ...
    ```

6. Update this document with the extra's row and any user-facing setup notes.
