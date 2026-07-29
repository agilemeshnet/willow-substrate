# Willow as a product

Willow began as personal infrastructure for memory, attention, interruption
recovery, and long-running thought. That origin is a design constraint, not a
marketing anecdote.

The public project should help a person:

- resume an intention after an interruption or a change of model;
- work across several terminals without manually copying context;
- preserve conversations, sources, decisions, and later reflections;
- encounter relevant earlier experience without having to remember to search;
- correct the record without erasing how an earlier belief arose;
- maintain research claims as the world changes.

Willow is not a diagnosis, medical device, or replacement for professional
care. It is user-owned cognitive infrastructure. Evaluation should measure
whether it reduces resumption cost and improves research integrity, not whether
it makes clinical claims.

## One substrate, replaceable organs

The cohesive repository is not a merger of every historical Willow folder. It
is a stable event and evidence contract with replaceable modules:

```text
willow
├── core
│   ├── immutable events and corrections
│   ├── local projections and integrity
│   └── context packets
├── continuity
│   ├── lifecycle hooks
│   ├── peer engrams
│   └── boot and compaction recovery
├── reflection
│   ├── summation and meditation
│   ├── word/shape connections
│   └── dreams and retroactive importance
├── research
│   ├── safe-staged commissions
│   ├── cited results
│   └── Fact TTL
├── adapters
│   ├── model and terminal providers
│   ├── Markdown and graph projections
│   ├── Vista/Wave and Foveation
│   └── federation and voice
└── surfaces
    ├── CLI
    ├── local API
    └── optional web application
```

The current alpha implements the core, Claude Code lifecycle adapter, local
reflection, research ledger, Fact TTL, and dependency-free word/shape
connection finder. It also includes a dependency-free Vista/Wave contextual
surround over active Willow events. AuraDB, the richer high-dimensional Vista
implementation, model-generated meditation, federation, voice, and a safe
remote API remain adapters.

## Three installation experiences

### 1. Continuity

The fifteen-minute first win:

1. Install Willow locally.
2. Connect two terminal sessions to one `WILLOW_HOME`.
3. Discuss a topic in one terminal.
4. Ask about it in the other terminal.
5. See bounded, provenance-bearing context arrive without copying a chat.

This is the smallest useful bundle and requires no cloud service.

### 2. Long-range research

Add:

- research commissions and cited results;
- meditations carrying words and explicit idea-shapes;
- connection proposals across prior meditations, claims, and results;
- Fact TTL with honest confirm/update/contradict/unknown outcomes;
- optional research providers chosen by the user.

The research provider is disposable. The question, evidence, interpretation,
claims, corrections, and future re-checks belong to Willow.

### 3. Relational and distributed Willow

Add optional relational and homelab organs:

- AuraDB or local Neo4j as a graph projection;
- richer high-dimensional Vista and MRL Foveation backends;
- signed federation envelopes between trusted nodes;
- local voice, browser, and actuator capabilities.

The reference Vista/Foveation attention cycle works locally without these
organs. This profile increases scale and retrieval quality; it must not be
required to experience Willow's core benefit.

## User state is not the package

Peter's diary, people, research archive, identity, accumulated character,
standing rules, credentials, and homelab addresses are user state. They must
never become defaults in the public package.

An installation may begin with a small optional seed:

- the user's preferred name;
- a project or question they care about;
- privacy and model-provider choices;
- desired capture surfaces;
- an optional tone or identity note.

Character should be allowed to emerge from interaction. A Willow installation
does not need to impersonate Peter's Willow.

## Safety and ownership requirements

Before a public beta, Willow needs:

- local-only operation as the default;
- explicit disclosure before sending memory to a model provider;
- capability-scoped tools rather than unrestricted remote shell access;
- export, backup, migration, and whole-store deletion;
- secret scanning and configuration templates containing no credentials;
- a clear distinction between immutable history and rebuildable indexes;
- redaction or selective-withholding controls for retrieved context;
- consent, refusal, and revocation as first-class, audience- and
  purpose-bearing events;
- a disclosure audit that distinguishes substrate access from permission to
  repeat personal information;
- provenance on every retrieved or derived item.

Append-only means Willow does not silently rewrite history. It must not mean a
user is trapped in the system. The user owns the store and may export or destroy
it.

## How to find out whether it helps anyone else

Do not begin with a personality benchmark. Begin with observable tasks:

1. **Interruption recovery:** after a day away, how long until the person can
   resume the right task?
2. **Cross-session continuity:** does a second process surface the correct
   earlier work without copied history?
3. **Unprompted recall precision:** when Willow introduces old material, is it
   relevant and welcome?
4. **Research integrity:** can every important claim be traced to sources and
   freshness state?
5. **Correction safety:** does corrected material stop influencing ordinary
   recall and dreams?
6. **Cognitive burden:** does Willow reduce repeated explaining and searching,
   or create more maintenance work?

A useful first study is five to ten consenting users over four weeks, with a
local-only installation, weekly interviews, and opt-in anonymous metrics. The
goal is to discover who benefits and where Willow distracts—not to prove a
medical effect.
