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

| Backend | Availability | What it is |
|---|---|---|
| `sparse` | Always | The zero-dep `VistaBackend`. TF-IDF-style sparse features with reference-beam ranking; the current default. |
| `bm25` | Always | Inline Okapi BM25 (k1=1.5, b=0.75), included as the canonical term-frequency baseline. |
| `voyage-mock` | `[vista]` extra | The `VoyageVistaBackend` from `willow.backends.vista_voyage`, driven by a deterministic mock embedder that seeds per-topic 32-dim vectors so no API calls are needed. See the honesty note below. |
| `voyage-real` | Opt-in with `--real-voyage` + `VOYAGE_API_KEY` | The same backend against the actual Voyage-4 API. Charges tokens; typical bird-study run is < $0.0001. |

## Honesty about `voyage-mock`

The mock embedder in this benchmark **is not a fair simulation of real
Voyage-4**. It gets the topic labels for both events and queries at
construction time and seeds its vectors from those labels, so
same-topic strings end up cosine-adjacent by design. Its numbers are
useful because they answer this: **if the embedding step correctly
recovers the topic each string is about, what recall does the rest of
the pipeline (HDBSCAN clustering + wave-recall + evidence ranking)
achieve?** That upper bound tells you whether the algorithmic
scaffolding around the embeddings can turn good vectors into good
recall.

Real Voyage-4 will not always recover the topic correctly. Its real-run
numbers on this corpus should sit somewhere between `bm25`/`sparse` and
`voyage-mock`; the gap between real-voyage and voyage-mock measures
embedding quality on your specific corpus and query mix. Users who run
with `--real-voyage` will see that gap on their own numbers.

## Sample output (from a fresh install with `[vista]`)

```
| Backend | Recall@3 | Recall@5 | MRR | Median latency (ms) |
|---|---|---|---|---|
| sparse | 0.417 | 0.556 | 0.589 | 0.55 |
| bm25 | 0.389 | 0.500 | 0.625 | 0.01 |
| voyage-mock | 0.722 | 0.722 | 1.000 | 0.66 |
```

Read this as: the `sparse` and `bm25` backends are close to each other
on this corpus, with BM25 a hair faster (a well-tuned formula and a
smaller feature set) and slightly better on MRR. The `voyage-mock`
upper bound shows the algorithmic pipeline (HDBSCAN + wave-recall +
evidence ranking) can turn topic-recovering embeddings into
substantially better recall on paraphrase queries. What the real-voyage
row actually looks like on your corpus is the honest question, which is
why `--real-voyage` is a flag.

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
>= 0.40, BM25 Recall@5 >= 0.40, voyage-mock Recall@5 >= 0.60. Retune as
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
`SparseVistaRunner`, `BM25Runner`, `VoyageMockRunner`, `VoyageRealRunner`.
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
