# Recall benchmark

`benchmarks/recall/` compares each installed backend against a fixed set of
ground-truth-labelled queries on the same corpus, reporting recall@k, mean
reciprocal rank, and per-query wall-clock latency.

Closes the recall-benchmark gap that ChatGPT's audit called out
(finding #4).

## Run it

```bash
# from the repo root
python -m benchmarks.recall

# JSON for CI regression pinning
python -m benchmarks.recall --json

# top-k depth
python -m benchmarks.recall --limit 8

# add a real Voyage run (charges Voyage; requires VOYAGE_API_KEY)
python -m benchmarks.recall --real-voyage
```

Zero-dep runs (sparse + BM25) execute against the built-in corpus without
any extras. Install `[vista]` to add the Voyage-mock row.

## Corpus and ground truth

- **Corpus**: `examples/temporal-bird-study.json` (9 events, ~1200 words),
  the same synthetic longitudinal dataset the sample suite already ships.
  Small on purpose: it fits in a test-run wall clock and its ground-truth
  labels are easy to inspect. Larger corpora will get their own benchmark
  files as they earn their place.
- **Ground truth**: `benchmarks/recall/queries.json`. Each query names a
  `topic` tag; `relevant_keys` is the set of event keys whose
  `metadata.topics` contain that tag. Six queries: three literal (single
  keyword phrases that a term-frequency backend can match by surface form)
  and three paraphrase (synonyms and rephrasings that a term-frequency
  backend cannot match by surface form, so dense semantic recall should
  score them higher).

## Backends compared

### Backends under test

| Backend | Availability | What it is |
|---|---|---|
| `sparse` | Always | The zero-dep `VistaBackend`. TF-IDF-style sparse features with reference-beam ranking; the current default. |
| `bm25` | Always | Inline Okapi BM25 (k1=1.5, b=0.75), included as the canonical term-frequency baseline. |
| `hybrid` | Always | Reciprocal Rank Fusion (Cormack et al. 2009) of sparse + BM25. `willow_substrate.backends.hybrid.HybridRecallBackend`. Does NOT include the topic-oracle dense sub-backend even when `[vista]` is installed; that mix would make the row uninterpretable. |
| `voyage-real` | Opt-in with `--real-voyage` + `VOYAGE_API_KEY` | The real Voyage-4 API. Charges tokens (typical bird-study run < $0.0001). This is the honest dense-retrieval measurement. |

### Ceilings (upper bounds, NOT backends)

| Ceiling | Availability | What it is |
|---|---|---|
| `topic-oracle` | `[vista]` extra | Seeds each vector directly from the same `metadata.topics` field that defines `relevant_keys` in `queries.json`, then runs those vectors through `VoyageVistaBackend`'s clustering + wave pipeline. Not a measurement of dense retrieval; a measurement of what the pipeline can do given labels. |

Ceilings answer 'is the algorithmic scaffolding the bottleneck?' A backend
that matches the ceiling means the pipeline is not the limit; the embedder
is. A backend well under the ceiling means the pipeline needs work.
Ceilings are NOT interpretable as comparisons against other backends; the
harness prints them in a separate table with a warning line for that reason.

## Sample output (from a fresh install with `[vista]`)

```
## Backends under test

| Backend | Recall@3 | Recall@5 | MRR   | Median latency (ms) |
|---------|----------|----------|-------|---------------------|
| sparse  |    0.417 |    0.556 | 0.589 |                0.55 |
| bm25    |    0.389 |    0.500 | 0.625 |                0.01 |
| hybrid  |    0.500 |    0.500 | 0.667 |                2.09 |

## Ceilings (upper bounds, not backends)

| Ceiling      | Recall@3 | Recall@5 | MRR   | Median latency (ms) |
|--------------|----------|----------|-------|---------------------|
| topic-oracle |    0.722 |    0.722 | 1.000 |                0.62 |
```

Read this as:

- **`sparse` and `bm25`** are close on this corpus. BM25 is a hair
  faster (tighter formula, smaller feature set) and slightly better on
  MRR. Neither handles paraphrase queries well by design.
- **`hybrid`** improves on either component through Reciprocal Rank
  Fusion at MRR 0.667 vs 0.589/0.625, but the effect is small at this
  corpus size and cannot be quoted as a comparative result. This
  benchmark has 9 events, 6 queries, and 18 relevance judgments; a
  one-judgment change is bigger than most of the gaps in the table.
  Use [LoCoMo](BENCHMARK_LOCOMO.md) for comparative claims.
- **`topic-oracle`** in the ceilings table is the pipeline's upper
  bound given label-recovering vectors. Any real backend at or near
  this line has hit the algorithmic scaffolding's limit; a real
  backend well under it (as `hybrid` is here) has room the embedder
  could take up.

The competitive-retrieval story on this corpus is a smoke test, not a
result. `--real-voyage` returns the honest dense-retrieval row.
Everything comparative belongs on LoCoMo.

## Methodology notes

- **Recall@k**: fraction of ground-truth-relevant events that appear in
  the top-k results. Averaged over queries.
- **MRR (mean reciprocal rank)**: mean of 1/rank of the first relevant
  event per query. Zero if no relevant event appears in the ranking.
- **Median latency**: median wall-clock per-query time across the six
  queries, in milliseconds. Not a hardware-normalised number; use as a
  relative comparison, not an absolute claim.
- The corpus is loaded into a fresh in-tempdir `EventStore` per run so
  no caches leak between runs. The Voyage backends do share embeddings
  across queries within one run (the whole point of the .npz cache), so
  per-query latency is inflated on the first query and lower thereafter.
  Median is reported specifically to be robust to this.

## Regressions

The JSON output of `python -m benchmarks.recall --json` is stable across
releases. CI (once wired) can pin lower bounds on
`backends[i].mean_recall_at_5` and `backends[i].mrr` and flag any drop.
Reasonable thresholds today (with `[vista]` installed): sparse Recall@5
>= 0.40, BM25 Recall@5 >= 0.40, topic-oracle Recall@5 >= 0.60. Retune as
the corpus and query set grow.

## Adding a query

Edit `benchmarks/recall/queries.json`. Each entry has:

- `id`: stable identifier for CI diff readability.
- `text`: the query string sent to each backend.
- `topic`: the `metadata.topics` tag the query targets.
- `relevant_keys`: the set of event keys whose topics contain the tag
  (these become the ground-truth relevant event ids at run time).
- `paraphrase`: `true` for paraphrase queries; `false` for literal ones.
  Reported separately in the per-query JSON so you can see whether a
  backend's gain comes from paraphrase handling specifically.

## Adding a backend

Add a runner class in `benchmarks/recall/harness.py` next to
`SparseVistaRunner`, `BM25Runner`, `TopicOracleRunner`, `VoyageRealRunner`.
The contract is:

```python
class MyBackendRunner:
    name = "my-backend"

    def __init__(self, store: EventStore, ...): ...

    def query(self, text: str, *, limit: int) -> list[tuple[str, float]]:
        """Return [(event_id, score), ...] sorted best-first, truncated to limit."""
```

Register it in `run()` guarded by whatever extras or env vars it needs so
callers without those still get a working benchmark for the other rows.
