# LoCoMo results (v0.2.0, 2026-08-10)

Six JSON manifests in this directory. Four are the current canonical
per-config results; two are experimental sidecars preserved for the
audit trail (additive law: nothing severed).

## Canonical

Produced by the code at commit b79fa9f + the `--dataset-dir` flag
introduced in PR #28. All ten LoCoMo10 conversations, 1986 questions.

| File | Config | What it measures |
|---|---|---|
| `recent_only.json` | recent-only | Return the K most recent turns, ignore the query. Lower bound. |
| `sparse.json` | sparse | The zero-dep `VistaBackend` (TF-IDF-style sparse features). |
| `hybrid_no_dense.json` | hybrid-no-dense | Reciprocal Rank Fusion of sparse + BM25. Free, zero-dep. |
| `hybrid_no_dense.with_consolidation.tau10y.json` | hybrid-no-dense + Hou 2024 wrapper | Same, wrapped in the ConsolidationBackend (time-decay + recall-frequency scoring per arXiv 2404.00573) at tau=10 years. Lifts Recall@10 by +25% over baseline while keeping Recall@5 flat. Free, zero-dep. |
| `hybrid_voyage.json` | hybrid-voyage | RRF of sparse + BM25 + Voyage-4 dense. Requires `[vista]` extra + VOYAGE_API_KEY. |

Read `docs/BENCHMARK_LOCOMO.md` for the methodology, the honest
recall numbers with bootstrap 95% CIs, and the reviewer's null-result
finding: **dense retrieval does not currently improve Recall@5 over
sparse + BM25 hybrid on this corpus at this pipeline architecture.**

## Experimental sidecars (preserved for audit)

Additive: kept on disk so anyone re-tracing the arc can inspect the
exact JSON the finding rests on. Not shipping guidance; not to be used
for regression pinning.

| File | Purpose |
|---|---|
| `hybrid_voyage.enriched-context-2026-08-10.json` | Result of the one experiment we tried: extending `_canonical_event_text` to pack session_id + timestamp + speaker + same-session prev/next window into every event's embed input. Moved Recall@5 by -0.001 (i.e., nothing). The code that produced this file was reverted; the null result is what shipped as the finding. Kept in-repo because it is genuinely distinct from `hybrid_voyage.json` (different code, different embed inputs). |
| `hybrid_voyage.pre-fix-empty-2026-08-10.json` | Pathological output from a pre-adapter-fix era (locomo10-shape parser missing). Every store came back empty, every retrieval returned 0 events. Preserved as a marker of the bug PR #26 fixed; a future maintainer inspecting this file will immediately see `n_conversations=1, conversation_ids=["unknown"]` and know why. |
| `hybrid_no_dense.with_consolidation.json` | Hou 2024 wrapper run at the paper's default tau (30 days). Recall@5 dropped by 48% because LoCoMo asks about turns from months prior; a 30-day exponential decay crushes those. Kept in-repo as a warning example so `docs/BENCHMARK_LOCOMO.md`'s finding on tau sensitivity has its receipt. |

A local sidecar named `hybrid_voyage.baseline-bare-context-2026-08-10.json`
was created during the experiment as a safety copy of the canonical
bare-context result. It was byte-identical to `hybrid_voyage.json`, so
committing it would have doubled the repo bloat with zero distinct
state to preserve (the additive law preserves prior states, not
duplicates). It lives on the working tree of anyone who reruns the
experiment and is trivial to regenerate; not tracked here.

## Scoring

```bash
# Canonical five-row comparison, including the consolidation wrapper:
python -m benchmarks.locomo.score \
  benchmarks/locomo/results/hybrid_no_dense.with_consolidation.tau10y.json \
  benchmarks/locomo/results/hybrid_voyage.json \
  benchmarks/locomo/results/hybrid_no_dense.json \
  benchmarks/locomo/results/sparse.json \
  benchmarks/locomo/results/recent_only.json
```

Add the tau=30d warning sidecar as a sixth row to reproduce the
"paper's default tau hurts LoCoMo by 48%" finding, and the enriched-
context sidecar as a seventh to reproduce the canonical-text null.
