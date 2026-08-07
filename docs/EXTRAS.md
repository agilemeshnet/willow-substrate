# Extras: layered fidelity for the full Willow substrate

The base install is dependency-free and runs the continuity demo. Extras add
richer memory-and-recall backends from the full internal Willow substrate.

```bash
pip install willow-substrate               # zero-dep floor (continuity demo)
pip install "willow-substrate[vista]"      # dense-embedding Vista + Wave recall
pip install "willow-substrate[neo4j]"      # optional AuraDB graph projection
pip install "willow-substrate[full]"       # the whole memory-and-recall stack
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

Concrete backend module: `willow.backends.vista_voyage.VoyageVistaBackend`
(coming in the next foundation follow-up PR; ports the internal Willow
`thought-buckets/wave.py` + `mint_buckets.py` + `voyage_embed.py` patterns).

### `[neo4j]` — AuraDB graph projection (coming)

Optional graph projection of the event ledger for federation and audit.
Reads `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` from the environment or
takes them as constructor arguments.

Dependencies: `neo4j`, `python-dotenv`.

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
        import willow.backends.vista_voyage  # noqa: F401
        HAS_VISTA_EXTRA = True
    except ImportError:
        HAS_VISTA_EXTRA = False

    @unittest.skipUnless(HAS_VISTA_EXTRA, "install [vista] to run this test")
    class VoyageVistaTests(unittest.TestCase):
        ...
    ```

6. Update this document with the extra's row and any user-facing setup notes.
