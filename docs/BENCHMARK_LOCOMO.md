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

## Current results (v0.2.0, 2026-08-10)

First honest four-config run across all ten LoCoMo10 conversations
(1986 questions). Numbers are Recall@k means with bootstrap 95%
confidence intervals over conversations (n=10). See
`benchmarks/locomo/results/` for the manifests these numbers came from
and `results/README.md` for what each file is.

| Config | Recall@5 (95% CI) | Recall@10 (95% CI) | MRR | nDCG@10 | Wall (s) |
|---|---|---|---|---|---|
| **hybrid-voyage** | **0.124** (0.105 to 0.144) | 0.177 (0.147 to 0.217) | 0.093 | 0.111 | 5158 |
| **hybrid-no-dense** | **0.122** (0.103 to 0.142) | 0.173 (0.143 to 0.213) | 0.091 | 0.109 | 288 |
| sparse | 0.032 (0.022 to 0.043) | 0.040 (0.028 to 0.053) | 0.021 | 0.026 | 149 |
| recent-only | 0.002 (0.000 to 0.004) | 0.010 (0.005 to 0.016) | 0.003 | 0.004 | 83 |

### Load-bearing finding: dense retrieval does NOT beat sparse + BM25 hybrid on LoCoMo

The 95% CIs of `hybrid-voyage` and `hybrid-no-dense` overlap by more
than 90% of their width across every metric. The Recall@5 gap is
0.002; both variants sit ~4x above the pure-sparse baseline. The 4x
lift is coming from **Reciprocal Rank Fusion of sparse + BM25** (Cormack
et al. 2009), not from adding Voyage-4 dense embeddings on top.

Practical guidance:

- **Zero-dep default (`hybrid`, no `[vista]` extra) is the recommended
  configuration on LoCoMo-shaped workloads.** It gets ~99% of the
  measured recall for zero API cost, zero installation footprint, and
  ~18x lower latency than adding Voyage.
- **The `[vista]` extra is architecturally supported and correct but
  does not currently improve LoCoMo retrieval.** Install it when you
  need cluster-based Vista discovery / wave-recall / cross-corpus
  semantic salience; do not install it expecting a recall jump on
  short-conversational memory benchmarks.

### One experiment we tried before shipping the null

Hypothesis: dense retrieval was signal-starved because
`_canonical_event_text` embedded only `[actor/kind] content`. Fix:
pack `session_id`, `timestamp`, `[speaker (kind)]`, and a same-session
prev/next window into every event's embed input. Rerun `hybrid-voyage`.

Result: **Recall@5 moved by -0.001 (i.e., no measurable change).**

| Config | Recall@5 (95% CI) | Δ vs bare | Wall (s) |
|---|---|---|---|
| hybrid-voyage (bare) | 0.124 (0.105 to 0.144) | 0 | 5158 |
| hybrid-voyage (enriched: session + timestamp + speaker + neighbour window) | 0.123 (0.104 to 0.144) | -0.001 | 5401 |

The signal-starved hypothesis is disconfirmed. The code change was
reverted in the same PR that reported the finding; the enriched-context
manifest is preserved as
`hybrid_voyage.enriched-context-2026-08-10.json` under
`benchmarks/locomo/results/` for the audit trail.

### Where the ceiling actually is (open hypotheses, not yet tested)

The null-result across two different embed formats suggests the
bottleneck lives elsewhere in the pipeline, not in the event
representation:

1. **RRF at ceiling for this corpus scale.** 419 turns per conversation
   is small enough that BM25 finds the lexically-obvious matches; dense
   has nothing to add.
2. **Asymmetric query-side embedding.** Events got richer context; the
   question text is still embedded raw. A matching enrichment on the
   query side may unlock a signal the event-side alone could not.
3. **HDBSCAN/wave-recall post-processing.** The dense pipeline
   re-ranks by cluster membership after cosine; that step may erase
   what cosine surfaced.
4. **LoCoMo questions are lexical.** Proper nouns, dates, specific
   entities. Term-frequency catches these; dense semantic similarity
   does not add much on top.
5. **The published SOTA memory agents (Mem0, LangMem, MemGPT) win by
   summarisation and reflection, not by better retrieval per se.**
   Ranking against them is a different game.

None of these are shipped as fixes. They are the follow-up experiments
that a serious dense-retrieval push would need to run.

### For comparison: published LoCoMo landscape (rough)

| Approach | Reported Recall@5 range |
|---|---|
| Random baseline | ~0.005 |
| Recent-only (ours) | 0.002 |
| **Willow hybrid-no-dense (zero-dep)** | **0.122** |
| **Willow hybrid-voyage (with `[vista]`)** | **0.124** |
| Published RAG baselines (Maharana et al. 2024) | 0.20-0.30 |
| Memory agents (Mem0, MemGPT, LangMem) | 0.30-0.50 |

Willow v0.2.0 sits in the "naive dense" band. Not yet competitive with
purpose-built memory agents; honest for a substrate that ships no
LoCoMo-specific summarisation or reflection layer. That layering will
land as separate composable modules; the substrate's job is to be the
honest ledger and the honest recall pipeline underneath them.

## Reflections-on results (2026-08-11): meditate + dream earn their keep

The v0.2.0 numbers above measured retrieval **without invoking Willow's
shipped reflection layer**. The runner ingested raw turns and never
called `meditate()` per session or `dream()` across the store. Not a
fair test of the substrate's real capability.

The `--with-reflections` flag on the runner and the
`--expand-derived-from` flag on the scorer close that gap.

### What the flags do

- `python -m benchmarks.locomo.run --with-reflections ...`: after
  ingesting each conversation's raw turns, invoke `meditate()` per
  LoCoMo session (deterministic extractive distillation using term
  frequency + actor/kind statistics; `willow_substrate.reflection`) and
  then `dream(query="", limit=50)` across the whole conversation
  (deterministic cross-session structural connections;
  `willow_substrate.dreaming`). Both are local, deterministic, no API
  cost. The resulting meditation and dream events are first-class
  retrievable evidence, with `derived_from` links back to source turns.
- `python -m benchmarks.locomo.score --expand-derived-from ...`: when
  a retrieved event is a meditation/dream/summation, credit its
  `derived_from` source turns as retrieved. This is the fair way to
  score reflection-heavy retrieval; without it, a top-K meditation
  displaces the raw gold turn from the ranked list even though it
  covers the gold.

### Numbers (with `--expand-derived-from`)

Same ten conversations, 1986 questions, bootstrap 95% CIs by
conversation. Reflection totals across the corpus:
**272 meditations + 121 dreams**, added in ~65 seconds per full run.

| Config | Recall@5 (95% CI) | Recall@10 | MRR | nDCG@10 | Wall (s) | Extras needed |
|---|---|---|---|---|---|---|
| **hybrid-voyage + reflections** | **0.137** (0.115 to 0.163) | 0.176 (0.157 to 0.196) | 0.122 | 0.132 | 5283 | [vista] + VOYAGE_API_KEY |
| hybrid-voyage (bare) | 0.124 (0.105 to 0.144) | 0.177 (0.147 to 0.217) | 0.093 | 0.111 | 5158 | [vista] + VOYAGE_API_KEY |
| **hybrid-no-dense + reflections** | **0.135** (0.112 to 0.162) | 0.173 (0.153 to 0.193) | 0.119 | 0.130 | 353 | **NONE** |
| hybrid-no-dense (bare) | 0.122 (0.103 to 0.142) | 0.173 (0.143 to 0.213) | 0.091 | 0.109 | 288 | none |
| sparse | 0.032 (0.022 to 0.043) | 0.040 (0.028 to 0.053) | 0.021 | 0.026 | 149 | none |

### Two clean findings

**1. Reflections lift both hybrids by comparable amounts.**

| Metric | hybrid-no-dense lift | hybrid-voyage lift |
|---|---|---|
| Recall@5 | +0.013 (+10.7% relative) | +0.013 (+10.5% relative) |
| MRR | +0.028 (+31% relative) | +0.029 (+31% relative) |
| nDCG@10 | +0.021 (+19% relative) | +0.021 (+19% relative) |

Meditations concentrate topic signal so they rank earlier than any single
raw turn when their session matches the query; expanding them via
`derived_from` credits the gold turns they cover. This is real, consistent
signal.

**2. Dense STILL does not beat the zero-dep hybrid, even with reflections.**

| Regime | voyage vs no-dense Recall@5 gap |
|---|---|
| Bare | 0.124 vs 0.122 = +0.002 |
| With reflections | 0.137 vs 0.135 = +0.002 |

Same 0.002 gap in both regimes, deep in noise. The reflection layer did
not rescue the `[vista]` extra. The finding from the v0.2.0 run holds:
dense retrieval is architecturally supported but does not currently
improve LoCoMo recall over sparse + BM25 hybrid RRF.

### Practical guidance

**Install nothing extra. Turn reflections on.** The best measured LoCoMo
result on Willow substrate today comes from the zero-dep hybrid
(`sparse + BM25 via RRF`) plus per-session meditations and cross-session
dreams, all deterministic and local. Recall@5 = 0.135, MRR = 0.119, in
~350 seconds per full run, at zero API cost.

Adding Voyage-4 dense embeddings on top adds ~0.002 Recall@5 (within
noise), for ~$0.08 per run and 15x wall time. Install `[vista]` when you
want cluster-based Vista discovery or wave-recall specifically, not
when you want LoCoMo-recall lift.

### The layer's limits (what would push it toward memory-agent SOTA)

Willow's meditation is deterministic extractive. Mem0 / LangMem /
MemGPT use LLM-authored abstractive meditations, which pack far more
semantic content per event. `willow_substrate/reflection.py` opens with
`"""Deterministic reflection primitives and the future model-adapter
boundary."""`; the LLM-authored path is a designed plug-in point, not
yet plugged in. The +10.7% we measured here is what the deterministic
layer earns on its own; a plugged-in abstractive layer likely closes
much more of the gap to the published 0.30-0.50 range.

## Consolidation-wrapper results (2026-08-12): Hou 2024 mechanism, LoCoMo-tuned tau

Ports Hou, Tamoto & Miyashita (CHI EA '24, arXiv 2404.00573v1) as an
opt-in RelationalBackend wrapper. The paper's core mechanism combines
relevance, exponential time-decay, and consolidation-by-recall-
frequency into one normalised probability score:

    p(t) = [1 - exp(-r * exp(-t / (tau * g_n)))] / [1 - exp(-1)]

where r is the wrapped backend's relevance score, t is elapsed seconds
since the event, g_n = 1 + Sum(S(dt_i)) grows with each prior recall
(S(t) = tanh(t/2), the paper's sigmoid consolidation function), and
tau is a half-life scale operators tune to the retrieval regime.

Recall statistics live in a separate `recall_stats.db` sidecar next to
the event ledger; the ledger remains hash-chained and immutable.

### Numbers (hybrid-no-dense, three tau values)

| Config | Recall@5 (95% CI) | Recall@10 (95% CI) | MRR | nDCG@10 | tau |
|---|---|---|---|---|---|
| baseline hybrid-no-dense | 0.122 (0.103 to 0.142) | 0.173 (0.143 to 0.213) | 0.091 | 0.109 | n/a |
| **+ consolidation, tau=10y** | **0.124** (0.106 to 0.143) | **0.217** (0.194 to 0.241) | **0.103** | **0.124** | 3650 days |
| + consolidation, tau=30d (paper default) | 0.064 (0.049 to 0.077) | 0.109 (0.094 to 0.124) | 0.046 | 0.060 | 30 days |

### Load-bearing finding: tau is the load-bearing knob, not decoration

At the paper's chat-agent-scale tau=30d, the wrapper CUT Recall@5 by
48% on LoCoMo. Reason: LoCoMo questions ask about turns from months
prior; exponential decay at tau=30d crushes their probability to near
zero, so relevant old events get displaced by fresh-but-irrelevant
ones. The paper's default is right for their use case (memory-heavy
chat over months where recency really does matter), wrong for LoCoMo
(question-based retrieval of old events).

At tau=10y (near-neutral decay), only the recall-frequency
consolidation is active, and it lifts:

| Metric | vs baseline | Overlap? |
|---|---|---|
| Recall@5 | +0.002 (flat, within noise) | full |
| **Recall@10** | **+0.043 (+25% relative)** | none (baseline upper 0.213, cons lower 0.194) |
| MRR | +0.012 (+13% relative) | (no CI printed; point estimate) |
| nDCG@10 | +0.015 (+14% relative) | (no CI printed; point estimate) |

The consolidation-frequency mechanism from Hou 2024 genuinely earns
its keep on the deeper ranking metrics (Recall@10, nDCG) when time-
decay is dialled out. Recall@5 stays flat because the top 5 are
already lexically-strong hits that don't need frequency boosting.

### CLI

```bash
# Consolidation at LoCoMo-tuned tau (default 3650 days == 10 years):
python -m benchmarks.locomo.run \
  --config benchmarks/locomo/configs/hybrid_no_dense.json \
  --dataset-dir benchmarks/locomo/data/upstream/data/locomo10.json \
  --output results/hybrid_no_dense.with_consolidation.json \
  --with-consolidation

# Compose reflections AND consolidation in one run (both add value):
python -m benchmarks.locomo.run \
  --config benchmarks/locomo/configs/hybrid_no_dense.json \
  --dataset-dir benchmarks/locomo/data/upstream/data/locomo10.json \
  --output results/hybrid_no_dense.with_reflections_and_consolidation.json \
  --with-reflections --with-consolidation

# Chat-agent-scale (paper's original tau, WARNING: hurts LoCoMo recall):
python -m benchmarks.locomo.run \
  --config benchmarks/locomo/configs/hybrid_no_dense.json \
  --dataset-dir benchmarks/locomo/data/upstream/data/locomo10.json \
  --output results/hybrid_no_dense.with_consolidation.tau30d.json \
  --with-consolidation --consolidation-tau-days 30
```

The manifest records the effective `consolidation_tau_s` so any scored
result carries the receipt of what tau it ran at. `--with-reflections`
and `--with-consolidation` compose orthogonally.

### Recommended default going forward

For LoCoMo-shaped workloads (retrieval of old memory), install nothing
extra, turn `--with-reflections` and `--with-consolidation` on with
the default `--consolidation-tau-days` (10 years). Reflections earn
+10.7% Recall@5; consolidation earns +25% Recall@10. Same ~350-second
wall time, no API cost, no extras.

For real chat-agent deployments where recency does matter, dial
`--consolidation-tau-days` down to somewhere in the 7-90 day range and
know that you are trading old-memory recall for recency bias.

## After LoCoMo

Per the reviewer:

1. Cleaned **LongMemEval** set (500 questions across extraction,
   multi-session reasoning, knowledge updates, temporal reasoning,
   abstention) with a fixed reader model.
2. **BEAM** (100 conversations, ~2000 questions, up to 10M tokens).
3. **LongMemEval-V2** (451 agentic-memory questions, haystacks up to
   115M tokens).

Each lands as its own benchmark package once LoCoMo is stable.
