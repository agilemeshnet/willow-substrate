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

Once the basic loop is running, three capabilities build on it.

**Deliberate attention.** `willow foveate "recurrent motifs"` narrows from
sessions to matching events to their neighbourhood, and returns the decision
trace that got there. `willow vista` widens back out to the surrounding
context. Foveation answers *what am I looking at*; Vista answers *what does
this belong to*.

**Reflection.** `willow meditate --session X` appends a derived summary with
`derived_from` provenance. `willow engram` records that a later insight made an
earlier event matter, without editing the earlier event.

**Association.** `willow dream "recurrent motifs"` proposes connections across
distant material. These are labelled proposals, not claims. Corrected or
expired material is never allowed to seed one, and vector similarity is never
relabelled as structural evidence.

Optional backends add richer recall:

```bash
pip install -e ".[vista]"   # dense-embedding Vista + Wave recall
pip install -e ".[neo4j]"   # optional AuraDB graph projection
pip install -e ".[full]"    # the whole stack
```

See [docs/EXTRAS.md](docs/EXTRAS.md) for what ships today and
[docs/BENCHMARK.md](docs/BENCHMARK.md) for how the backends compare on a fixed
corpus.

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
