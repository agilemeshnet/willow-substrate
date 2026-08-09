# LoCoMo evidence-retrieval benchmark

The bird-study benchmark (see [BENCHMARK.md](BENCHMARK.md)) stays as a
fast unit-scale regression test; it does not carry comparative claims.
For those, we run against LoCoMo (`snap-research/locomo`): ten
substantial timestamped conversations with **gold evidence turn ids**
per question, so we can measure objectively whether Willow placed the
right memories into the retrieval result AND into the final context
packet, not whether an LLM happened to guess correctly.

Methodology follows the reviewer's spec (see PR description that
introduced this benchmark).

## The split that matters

For every question, the runner records:

- **raw retrieval**: which event ids the backend returned, in rank order.
- **final context**: which event ids the `ContextBuilder` actually placed
  into the assembled packet at each token budget.

We report metrics on both surfaces. Any config where raw retrieval finds
the gold event but the final context drops it is a **context-assembly
regression**; a class of bug that pure retrieval numbers cannot see.

## Fetching the dataset

The dataset is not vendored in this repo. Fetch it from
[snap-research/locomo](https://github.com/snap-research/locomo):

```bash
mkdir -p benchmarks/locomo/data
cd benchmarks/locomo/data
# Follow the upstream README for the current download instructions.
```

The adapter accepts either the upstream flat layout (one JSON per
conversation) or a single combined file. See
`benchmarks/locomo/adapter.py` for the shape it expects.

## Configurations

Seven configurations, each declared in `benchmarks/locomo/configs/`:

| Config | Backend name | What it tests |
|---|---|---|
| `recent-only` | `recent-only` | Naive recency baseline (return the K newest events, ignore the query). Floor. |
| `sparse` | `sparse` | Zero-dep VistaBackend (TF-IDF-shaped features + reference beams). |
| `hybrid-no-dense` | `hybrid` (VOYAGE_API_KEY unset) | RRF fusion of sparse + BM25 only. Test whether rank fusion alone lifts recall over sparse. |
| `hybrid-voyage` | `hybrid` (with VOYAGE_API_KEY) | Full RRF: sparse + BM25 + real Voyage-4 dense embeddings. |
| `oracle` | `oracle` | Upper bound: always returns the gold turns. Any real backend that beats this is bug-shaped. |
| `bm25` | (see notes) | Pure BM25 baseline for RRF-vs-fusion decomposition. Follow-up: needs a `bm25` factory backend. |
| `contextbuilder` | (default) | Compare current vs proposed global-salience/reranked ContextBuilder. Follow-up. |

Foveation on/off and corrections/superseded ablation live in the
follow-up PR that adds the missing configs.

## Metrics

The reviewer's list, computed per-question then aggregated per-conversation:

- Evidence Recall@5, @10, @20
- MRR
- nDCG@10
- In-context recall by budget (256, 512, 1024, 2048 tokens)
- Context precision by budget
- Retrieval and context-build latency (median + p95)

**Bootstrap CIs by conversation** (not by question), because LoCoMo has
only ten conversations and treating each question as independent
overstates confidence.

Answer-accuracy / F1 / abstention accuracy / knowledge-update accuracy
land in the LongMemEval follow-up (see the reviewer's plan). LoCoMo is
the retrieval-quality lens.

## Running

Once the dataset is at `benchmarks/locomo/data/`:

```bash
# One config
python -m benchmarks.locomo.run \
  --config benchmarks/locomo/configs/hybrid_no_dense.json \
  --output results/hybrid-no-dense.json

# All the always-runnable configs
for c in recent_only sparse hybrid_no_dense oracle; do
  python -m benchmarks.locomo.run \
    --config benchmarks/locomo/configs/$c.json \
    --output results/$c.json
done

# Score everything
python -m benchmarks.locomo.score results/*.json
```

The `hybrid_voyage` config requires `VOYAGE_API_KEY` and will make real
Voyage-4 API calls. Free tier covers a typical LoCoMo run; still, this
is the money-gated row and gets explicit go before firing in CI.

## Output shape

`run.py` writes one JSON manifest per config with:

- `schema` version tag
- `config`: snapshot of the config used (name, backend, kwargs, budgets)
- `top_k` used for retrieval
- `willow_commit` (git rev-parse of the running tree)
- `n_conversations`, `n_questions`, `wall_seconds`
- `rows[]`: one per question with `conversation_id`, `question_id`,
  `question_text`, `category`, `gold_event_ids`, `retrieved_event_ids`,
  `retrieval_latency_ms`, and `per_budget[]` (each with
  `context_tokens`, `final_context_ids`, `context_latency_ms`).

The full row-level shape is deliberate: aggregate scores hide
category-specific regressions (temporal, multi-hop, knowledge-update).
`score.py` reports the aggregates; the JSON supports arbitrary
slicing later.

## Reproducibility

Every manifest carries the Willow commit hash and the config snapshot.
Add the dataset revision (snap-research/locomo commit) to the config
file when running for the record. Dependency versions come from
`pyproject.toml` + `[vista]` extra pins.

## After LoCoMo

Per the reviewer:

1. Cleaned **LongMemEval** set (500 questions across extraction,
   multi-session reasoning, knowledge updates, temporal reasoning,
   abstention) with a fixed reader model.
2. **BEAM** (100 conversations, ~2000 questions, up to 10M tokens).
3. **LongMemEval-V2** (451 agentic-memory questions, haystacks up to
   115M tokens).

Each lands as its own benchmark package once LoCoMo is stable.
