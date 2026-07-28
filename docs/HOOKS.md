# Lifecycle hooks

The installed Willow experience is event-driven. A temporary model process
does not own memory; it participates in a shared substrate at lifecycle
boundaries.

| Phase | Input | Local effect | Output to the model |
|---|---|---|---|
| `start` | Provider hook JSON | Capture any existing transcript | Stable boot packet |
| `prompt` | Prompt + session/transcript metadata | Capture prompt and unseen transcript records | Relevant peer and durable context |
| `stop` | Completed transcript | Capture new messages; append a 30-minute peer engram | Silent |
| `compact` | Transcript + compaction summary | Capture new messages | Recovery packet |
| `end` | Completed transcript | Advance summation and meditation; discover retroactive engrams | Silent |

## Claude Code

Copy or merge the hook entries from
`examples/claude-code-settings.json`. The provider sends its hook payload on
standard input and treats standard output as additional context.

All processes must resolve the same home:

```bash
export WILLOW_HOME="$HOME/.willow"
```

The example wires the phases Claude Code exposes in the current Willow
deployment:

- `UserPromptSubmit` to `willow hook prompt`
- `Stop` to `willow hook stop`
- `PostCompact` to `willow hook compact`

Invoke `willow hook end` from a provider's true session-end event, a deliberate
shutdown command, or a scheduler. Do not map it to every `Stop`: `Stop` is a
turn boundary in Claude Code, not necessarily a session boundary.

## Retry and concurrency behaviour

Provider record IDs and stable prompt occurrence keys become deterministic
Willow event IDs. Rescanning a transcript returns the existing event. If two
processes race to append the same provider record, one wins the SQLite
transaction and the other reads the same immutable result.

Turn engrams use their source event IDs as their idempotency key. Summations,
meditations, dreams, and retroactive engrams follow the same rule.

## Context pressure

When the provider reports recent prompt usage and a context limit, the injected
budget shrinks in four phases:

| Window use | Phase | Injected budget |
|---|---|---|
| Below 50% | `open` | 100% |
| 50–80% | `filling` | 70% |
| 80–95% | `near-compact` | 50% |
| Above 95% | `compact-imminent` | 30% |

After compaction, the `compact` phase rebuilds from the preserved substrate
rather than asking the person to reconstruct the conversation.

## Provider contract

Additional providers need only produce:

- A stable session ID
- Message role, content, and preferably a provider record ID
- An optional UTC timestamp, model name, and tool names
- Lifecycle events equivalent to prompt, completed turn, compaction, and end

Provider adapters must not invoke a model. Generative reflection belongs behind
a separate explicit adapter and still has to append provenance-complete events.

Hooks fail open by default so a memory-side fault does not block the host
conversation. Use `willow hook PHASE --strict` during development and testing.

## Privacy

The bundled hook adapter reads local transcript files and writes local SQLite.
It does not make network requests. Context packets mark recalled content as
historical evidence so an instruction found inside old memory is not silently
treated as current authority.
