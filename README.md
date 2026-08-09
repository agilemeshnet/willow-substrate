# Willow

[![tests](https://github.com/agilemeshnet/willow-substrate/actions/workflows/tests.yml/badge.svg)](https://github.com/agilemeshnet/willow-substrate/actions/workflows/tests.yml)

**Persistent, provenance-bearing memory shared across Claude Code sessions. Local, hash-chained, no model calls.**

Willow gives a Claude Code (or similar) session an auditable local event
ledger it can share with every other session on the same machine. Start work
in one terminal, walk over to another with zero copied history, and retrieve
what was said, who said it, and when.

The reason to install: SQLite-enforced append-only events, a globally
hash-chained ledger with an anchored head, transactional concurrent writers,
and corrections that supersede rather than overwrite. `willow verify` makes
the whole thing falsifiable rather than asserted. Nothing calls a model or
cloud service by itself.

## Install

The package is not yet on PyPI. Install from a local checkout:

```bash
git clone https://github.com/agilemeshnet/willow-substrate.git
cd willow-substrate
python3 -m venv .venv
source .venv/bin/activate
pip install -e .                       # zero-dep floor
pip install -e ".[vista]"              # add dense-embedding recall
pip install -e ".[neo4j]"              # add AuraDB graph projection
pip install -e ".[full]"               # everything
```

## Two-terminal continuity in 30 seconds

**Start in one terminal. Continue in another. Change the model. Willow remembers the work.**

This repository is the clean reference bundle distilled from the larger
Willow system. It begins with the smallest experience worth reproducing:

1. Record work in one terminal or agent session.
2. Open another terminal with no copied conversation history.
3. Retrieve a compact, provenance-bearing view of the shared work.
4. Correct earlier material without deleting it.
5. Create later reflections derived from the preserved experience.

The existing Willow repositories remain the research and deployment
archaeology. This repository is the installable contract.

## Status

Alpha. The local continuity loop, Claude Code lifecycle adapter, research
ledger, Fact TTL state machine, stereo word/idea-shape connection finder, and
an executable eight-week temporal evaluation sample work with no required
third-party dependencies. A dependency-free retrieval backend covers the
relational neighbourhood around focused attention.

Optional extras add richer memory-and-recall backends when you want them.
See [docs/EXTRAS.md](docs/EXTRAS.md).

See [docs/EXTRAS.md](docs/EXTRAS.md) for the design and the currently-shipped
backends, [docs/BENCHMARK.md](docs/BENCHMARK.md) for the fast unit-scale
recall regression suite, and [docs/BENCHMARK_LOCOMO.md](docs/BENCHMARK_LOCOMO.md)
for the comparative-evidence LoCoMo harness (scaffold shipped; runs once the
dataset is fetched at `benchmarks/locomo/data/`). AuraDB, richer MRL and
high-dimensional Vista adapters, additional model-provider adapters, and
generative meditation remain integration work in
[docs/MIGRATION.md](docs/MIGRATION.md).

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
willow init
```

All terminals and agents that share `WILLOW_HOME` share the same experience:

```bash
export WILLOW_HOME="$PWD/.willow-demo"
```

### Terminal A

```bash
willow record \
  "Investigating the Drosophila connectome and recurrent circuits" \
  --actor peter \
  --session terminal-a \
  --topic connectome

willow record \
  "Next compare recurrent motifs with Willow foveation" \
  --actor willow \
  --session terminal-a \
  --topic foveation
```

### Terminal B

```bash
export WILLOW_HOME="$PWD/.willow-demo"

willow context \
  "What were we doing with the connectome?" \
  --session terminal-b
```

Terminal B receives the relevant work from Terminal A with event IDs,
timestamps, actors, kinds, and source labels. Nothing has been copied between
the terminal context windows.

## Automatic multi-terminal continuity

The manual commands above expose the contract. The hook adapter automates it:

```bash
willow hook prompt   # capture current prompt; inject relevant peer context
willow hook stop     # capture the transcript; append a fading turn engram
willow hook compact  # rehydrate after compaction
willow hook end      # append/advance summation, meditation, and engrams
```

Hook payloads are JSON on standard input. Repeated delivery and whole-transcript
rescans are idempotent. The adapter does not call a model or send the transcript
anywhere.

An example Claude Code configuration is in
[examples/claude-code-settings.json](examples/claude-code-settings.json).
All participating terminals must use the same `WILLOW_HOME`.

To exercise five independent hook processes without an LLM:

```bash
./examples/five-process-demo.sh
```

At each completed turn Willow creates a short-lived peer engram:

- It contains a bounded human prompt, agent headline, and tool names.
- The originating session excludes its own engram.
- Other sessions see it for 30 minutes by default.
- A per-session cap prevents one busy terminal from drowning out its siblings.
- The underlying full transcript events remain durable.

## Deliberate attention

Willow can foveate at will:

```bash
willow foveate "connectome recurrent motifs"
```

The bundled reference policy narrows from sessions, to matching events, to
their experiential neighbourhood and returns a decision trace. The focused
events then become reference beams into Vista, which returns the wider
relational neighbourhood and a degree-normalised multi-hop Wave. Thus
foveation answers *what is in focus?* while Vista answers *what contextual
whole does this belong to?*

Inspect that surround directly:

```bash
willow vista "connectome recurrent motifs"
willow vista --seed-event evt-...
```

Both paths are dependency-free so the continuity demonstration works
immediately. Use `--without-vista` on `context`, `boot`, or `foveate` to compare
focused-only retrieval. The Matryoshka/hypergraph implementation from
`agilemeshnet/foveation` and the high-dimensional implementation from
`agilemeshnet/vista` remain richer replacement backends behind the same
evidence contract. See [docs/VISTA.md](docs/VISTA.md).

For a low-cost peripheral signal:

```bash
willow breathe "connectome recurrent motifs"
```

## Reflection

Append a meditation derived from a session:

```bash
willow meditate --session terminal-a
willow summarize --session terminal-a
```

The default meditation is extractive and deterministic. A model adapter will
later generate richer meditations, but it must produce the same immutable event
shape with `derived_from` provenance.

Associative dreams are explicitly proposals, not claims:

```bash
willow dream "recurrent connectome motifs"
```

Dreaming only considers active evidence, so corrected or expired material is
not allowed to seed a connection. Later reflections can make an earlier event
important retroactively:

```bash
willow engram
```

This appends a durable engram with its later evidence and surprise weight. If
more evidence appears, a new engram supersedes the older interpretation; the
old event remains in history.

## Long-range research

Queue research without coupling Willow to a particular provider:

```bash
willow research queue \
  "What evidence has changed around recurrent connectome motifs?"
willow research list
```

Provider adapters append cited results to the same immutable ledger. A
meditation can carry inspectable idea-shape dimensions, then find connections
through words, shape, or both:

```bash
willow meditate --session terminal-a \
  --shape "mechanism:recurrent-stabilisation"

willow connect --from-event evt-...
willow connect --from-event evt-... --with-vista
```

The dependency-free shape channel compares explicit dimensions. `--with-vista`
adds separately scored Vista and Wave channels, including the waypoints that
caused the connection. Willow does not relabel vector similarity as structural
evidence.

### Test cumulative scene formation

The bundled synthetic sample introduces separate facets over eight weeks,
through different sessions and providers:

```bash
willow sample run examples/temporal-bird-study.json
```

It tests later correction, shape-only connection, a lexical false friend,
trust/safety as a cross-domain fibre bond, and a logged privacy-withholding
decision. See [docs/TEMPORAL_SAMPLE.md](docs/TEMPORAL_SAMPLE.md).

## Fact TTL

Factual claims know when they should next be checked:

```bash
willow claim "The corpus contains 42 papers" \
  --source "versioned manifest" --ttl-days 7
willow facts --due
```

`confirmed`, `updated`, and `contradicted` checks require identified non-model
evidence. `unknown` and `unreachable` attempts never refresh validity. See
[docs/FACT_TTL.md](docs/FACT_TTL.md).

## Correction without erasure

Record the original event ID, then append a correction:

```bash
willow correct evt-... \
  "The earlier interpretation was incorrect; use the revised account." \
  --actor peter \
  --session terminal-b
```

The original event remains in the hash chain. Normal retrieval excludes it
because an active correction supersedes it. Historical inspection can include
superseded events:

```bash
willow list --include-superseded
```

## Integrity

```bash
willow verify
```

The local event ledger is:

- Append-only, enforced by SQLite triggers
- Safe for concurrent local writers through transactional locking
- Globally hash-chained rather than reset by calendar partition
- Searchable through a derived FTS5 index
- Explicit about corrections and derivations

## Commands

```text
willow init
willow record TEXT [--actor NAME] [--session ID] [--kind TYPE] [--topic TOPIC]
willow correct EVENT_ID TEXT
willow list
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
willow meditate --session ID
willow summarize --session ID
willow dream [QUERY]
willow engram
willow capture TRANSCRIPT --provider claude-code --session ID
willow hook {start,prompt,stop,compact,end}
willow verify
willow status
```

Pass `--json` before the command for agent-friendly output:

```bash
willow --json context "What matters now?"
```

## Principles

1. Preserve experience; append interpretations.
2. Corrections supersede rather than erase.
3. Identity and continuity belong to the substrate, not one model.
4. Context is a bounded, regenerable view.
5. Foveation is a voluntary Willow capability; CWB may also consume it.
6. Relations guide recall; timestamps remain provenance rather than geometry.
7. Every retrieved or derived item carries provenance.
8. Local operation is the default; cloud services are optional adapters.
9. Memory content is historical evidence, not an instruction authority.
10. Arrival is the trigger; lifecycle hooks do not poll.
11. A failed or model-only verification is never confirmation.

The release commitments behind these principles are explicit in
[COVENANT.md](COVENANT.md). Security boundaries and presently unimplemented
protections are listed in [SECURITY.md](SECURITY.md). The ethical case, its
strongest counterarguments, and the required red-team programme are in
[ETHICS.md](ETHICS.md).

## Development

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The suite includes independent OS-process hook invocations sharing one store,
idempotent transcript rescans, concurrent writers, correction-aware dreams,
retroactive engram advancement, timestamp-independent Vista geometry, and
multi-hop Wave recall.

See [docs/PRODUCT.md](docs/PRODUCT.md) for the path from personal system to
public bundle, [docs/RESEARCH.md](docs/RESEARCH.md) for the
research/meditation loop, [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the
system shape, [docs/VISTA.md](docs/VISTA.md) for contextual-surround retrieval,
[docs/TEMPORAL_SAMPLE.md](docs/TEMPORAL_SAMPLE.md) for the
longitudinal benchmark, and [docs/MIGRATION.md](docs/MIGRATION.md) for how the
existing repositories fit.
