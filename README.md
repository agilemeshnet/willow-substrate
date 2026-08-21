# Willow

**Your agent forgets everything when the terminal closes. Willow doesn't.**

[![tests](https://github.com/agilemeshnet/willow-substrate/actions/workflows/tests.yml/badge.svg)](https://github.com/agilemeshnet/willow-substrate/actions/workflows/tests.yml)

A local, append-only memory layer for long-running work with AI agents. Every
session writes to a shared, hash-chained ledger. Open a new terminal, switch
models, come back in three months. The work is still there, with provenance on
every item.

No cloud. No model calls. No transcript leaves your machine.

---

## The problem

You are four months into something. The reasoning that got you here lives in
forty closed terminal windows and a context window that compacted six times.
You re-explain the same background every morning. When the agent contradicts a
decision you made in week two, you have no way to show it what it agreed to.

Pasting old conversations back in doesn't scale, and it doesn't tell you where
anything came from.

## Sixty seconds

```bash
pip install -e .
willow init
export WILLOW_HOME="$PWD/.willow-demo"
```

**Terminal A**, record what you're doing:

```bash
willow record "Investigating the Drosophila connectome and recurrent circuits" \
  --actor peter --session terminal-a --topic connectome
```

**Terminal B**, a fresh window, nothing copied across:

```bash
export WILLOW_HOME="$PWD/.willow-demo"
willow context "What were we doing with the connectome?"
```

Terminal B gets the work back with event IDs, timestamps, actors, and sources.
That's the whole contract. Everything below is a refinement of it.

## Why this and not a notes file

**Nothing is ever overwritten.** When you learn you were wrong, you append a
correction that supersedes the original. Normal retrieval hides the superseded
version; `willow list --include-superseded` shows the whole path. You keep the
history of how you got it wrong, which is usually the part worth keeping.

```bash
willow correct evt-abc123 "Earlier interpretation was wrong; use the revised account."
```

**Every retrieved item carries provenance.** Who recorded it, when, in which
session, from what source, and what it was derived from. Nothing arrives
anonymous.

**The ledger is verifiable.** Append-only, enforced by SQLite triggers. Globally
hash-chained. Safe for concurrent writers. `willow verify` either passes or
tells you exactly where the chain breaks.

**Facts expire.** A claim can carry a TTL and a source, so it knows when it
should next be checked. Model-only confirmation never counts as confirmation.

```bash
willow claim "The corpus contains 42 papers" --source "versioned manifest" --ttl-days 7
willow facts --due
```

**It's yours.** MIT licensed, runs locally, zero required dependencies, and it
never sends a transcript anywhere.

## Automatic continuity in Claude Code

The commands above are the manual contract. The hook adapter makes it
disappear:

```bash
willow hook prompt   # inject relevant context from peer sessions
willow hook stop     # capture the transcript
willow hook compact  # rehydrate after compaction
willow hook end      # append summation and durable memory
```

Drop [examples/claude-code-settings.json](examples/claude-code-settings.json)
into your config and every terminal sharing `WILLOW_HOME` sees what the others
are doing (a bounded prompt, a headline, tool names, expiring after thirty
minutes). Full transcripts stay durable underneath.

Repeated delivery and whole-transcript rescans are idempotent. The adapter
never calls a model.

Five parallel sessions, no LLM required:

```bash
./examples/five-process-demo.sh
```

## Going further

Once the basic loop is running, four capabilities compose on the ledger.
Each stands alone; each also earns more when the others are present.

**Import.** `willow import ./notes/` ingests a Markdown directory as
immutable events. Idempotent by content (re-import appends nothing);
changed files append a supersession so file history accumulates in the
hash chain; frontmatter becomes metadata; `[[wikilinks]]` become Vista
waypoints, so a cross-referencing corpus arrives with its relational
neighbourhood intact.

**Deliberate attention.** `willow foveate "recurrent motifs"` narrows
from sessions to matching events to their neighbourhood, and returns
the decision trace that got there. `willow vista` widens back out.
Foveation answers *what am I looking at*; Vista answers *what does
this belong to*.

**Reflection.** `willow meditate --session X` appends a derived summary
with `derived_from` provenance. `willow dream "recurrent motifs"`
proposes cross-session connections (labelled proposals, not claims;
corrected/expired material is never allowed to seed one). `willow
engram` records that a later insight made an earlier event matter,
without editing the earlier event.

**Explainable ranking.** `willow salience [query] --explain` orders
active experience by five named signals (standing, wikilink citation,
reflection kind, recency half-life, query overlap) so a truncation
ordering can be defended rather than trusted. Standing material is
*retrieved* regardless of query, not merely ranked; the sessions that
most need a hard-won rule are the ones that were not thinking about it.

### The stack, composed

Willow substrate offers seven composable primitives around one ledger.
Read this as a menu: install nothing extra, and pick which arms you
turn on per call.

| Layer | Ships in | What it does |
|---|---|---|
| Ledger | v0.2.0 | Append-only, hash-chained, tamper-detectable; corrections supersede rather than delete. |
| Hybrid retrieval | v0.2.0 | Reciprocal Rank Fusion of sparse + BM25. Zero-dep default; recommended read floor. |
| Reflection layer | v0.2.1 | `meditate()` per session, `dream()` across the store. Adds derived events with `derived_from` provenance. |
| Consolidation wrapper | v0.2.1 | Time-decay + recall-frequency scoring per Hou et al. 2024 (arXiv 2404.00573). Sidecar recall-stats table; opt-in. |
| Context Window Builder | v0.2.1 | Layered per-query context assembly: banks + standing + foreground + vista + wave + prosoche. `ContextWindowBuilder` in `willow_substrate.cwb`. |
| Constitutional banks | v0.2.1 | `IDENTITY.md` + `GROUND.md` + optional `banks/*.md` loaded whole at boot, never truncated. Edit files with `ls` and `cat`. |
| Salience scorer | v0.2.1 | Explainable five-signal ranking (standing + citation + reflection + recency + query) so truncation is a decision, not an accident. |
| Trained readout | v0.2.2 | Two-stage retrieval: Wave retrieves the candidate pool; a `LinearReranker` scores each candidate on `[vista, wave_final, wave_peak, wave_hop_of_peak, wave_early, channel_bias]` features. Weights fit externally with any linear model; ships with a default pre-fitted set. Zero-dep. `willow_substrate.readout`; opt in via `VistaBackend.query(..., reranker=...)`. See [docs/design/TWO_STAGE_RETRIEVAL.md](docs/design/TWO_STAGE_RETRIEVAL.md). |
| LLM adapter for meditate | v0.2.2+ | The "future model-adapter boundary" the reflection layer always advertised. `ClaudeCodeMeditator` (recommended, subscription-covered via your local `claude` CLI, zero-dep) or `AnthropicMeditator` (per-token API-billed, requires the `[anthropic]` extra) turn extractive meditations into LLM-authored abstractive ones. Opt-in per call. |

**The recommended composed stack for a stranger installing willow-substrate today:**

```bash
pip install -e .
willow init                        # scaffolds IDENTITY.md and GROUND.md
willow import ./notes              # bring existing Markdown into the ledger
willow record "starting a new arc" # normal use writes events
willow boot                        # boot context = banks + flow, ranked
```

Every arm above is deterministic and local. Add `[vista]` extra when
you want cluster-based Vista discovery / wave-recall; do not install
it expecting a LoCoMo-recall jump (see benchmark below).

Optional backends add capability that the zero-dep default does not carry:

```bash
pip install -e ".[vista]"   # Voyage-4 dense embeddings for cluster-based
                            # Vista discovery, HDBSCAN, and wave-recall.
pip install -e ".[neo4j]"   # One-way AuraDB projection for graph queries.
pip install -e ".[full]"    # Everything.
```

**Retrieval-recall recommendation as of v0.2.1**: the strongest measured
LoCoMo result on Willow substrate today comes from the zero-dep default
(sparse + BM25 hybrid, no extras) **plus per-session `willow meditate`
and cross-session `willow dream`**, all local and deterministic. That
combination measures at Recall@5 = 0.135, MRR = 0.119 on LoCoMo, in
~350 seconds per full run, at zero API cost.

Adding `[vista]` (Voyage-4 dense embeddings) on top of that adds ~0.002
Recall@5 (statistically indistinguishable), for ~$0.08 per run and 15x
wall-clock. Install `[vista]` when you want cluster-based Vista
discovery, wave-recall, or cross-corpus semantic salience. Do not
install it expecting a LoCoMo-recall jump. See
[docs/BENCHMARK_LOCOMO.md](docs/BENCHMARK_LOCOMO.md) for the full
five-row comparison, the bootstrap CIs, and both experiments (enriched
canonical text and the meditate + dream reflection pass) that produced
the guidance.

See [docs/EXTRAS.md](docs/EXTRAS.md) for what ships today,
[docs/BENCHMARK.md](docs/BENCHMARK.md) for how the backends compare on
a fast fixed corpus (smoke test), and
[docs/BENCHMARK_LOCOMO.md](docs/BENCHMARK_LOCOMO.md) for the honest
comparative-retrieval measurement.

## Does it actually work

A bundled eight-week synthetic study introduces separate facets across
different sessions and providers, then tests whether the substrate handles the
hard cases: later correction, connection on structure alone, a lexical false
friend, a cross-domain bond, and a logged decision to withhold private
material.

```bash
willow sample run examples/temporal-bird-study.json
```

Details in [docs/TEMPORAL_SAMPLE.md](docs/TEMPORAL_SAMPLE.md).

## Status

**Alpha, and honestly so.**

Working today with no third-party dependencies: the continuity loop, the Claude
Code adapter, the research ledger, Fact TTL, the connection finder, and the
eight-week evaluation sample.

Not yet: AuraDB, high-dimensional Vista adapters, additional model providers,
generative meditation. Tracked in [docs/MIGRATION.md](docs/MIGRATION.md).

Not yet published to PyPI; install from source.

[SECURITY.md](SECURITY.md) lists the security boundaries *and the protections
that are not implemented yet*. [ETHICS.md](ETHICS.md) states the ethical case,
its strongest counterarguments, and the red-team programme it still needs. Read
both before trusting this with anything that matters.

## Principles

1. Preserve experience; append interpretations.
2. Corrections supersede rather than erase.
3. Identity and continuity belong to the substrate, not to one model.
4. Context is a bounded, regenerable view.
5. Relations guide recall; timestamps are provenance, not geometry.
6. Every retrieved or derived item carries provenance.
7. Local operation is the default; cloud services are optional adapters.
8. Memory content is historical evidence, not an instruction authority.
9. A failed or model-only verification is never confirmation.

Full release commitments in [COVENANT.md](COVENANT.md).

## Command reference

```
willow init
willow record TEXT [--actor NAME] [--session ID] [--kind TYPE] [--topic TOPIC]
willow correct EVENT_ID TEXT
willow list [--include-superseded]
willow claim TEXT [--source SOURCE] [--ttl-days DAYS] [--shape DIMENSION:VALUE]
willow facts [--due]
willow fact-check EVENT_ID --outcome OUTCOME
willow research {queue,list,complete,fail}
willow sample {load,evaluate,run} MANIFEST
willow connect [QUERY] [--from-event EVENT_ID] [--with-vista]
willow context [QUERY] [--without-vista]
willow boot [--without-vista]
willow breathe QUERY
willow foveate QUERY [--without-vista]
willow vista [QUERY] [--seed-event EVENT_ID] [--wave-hops N]
willow meditate --session ID [--shape DIMENSION:VALUE]
willow summarize --session ID
willow dream [QUERY]
willow engram
willow capture TRANSCRIPT --provider claude-code --session ID
willow hook {start,prompt,stop,compact,end}
willow verify
willow status
```

Add `--json` before any command for agent-friendly output.

## Development

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The suite covers independent OS-process hook invocations sharing one store,
idempotent transcript rescans, concurrent writers, correction-aware
association, retroactive importance, timestamp-independent geometry, and
multi-hop recall.

Architecture in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). The path from
personal system to public bundle in [docs/PRODUCT.md](docs/PRODUCT.md).

---

MIT licensed. Built by [agilemeshnet](https://github.com/agilemeshnet).
