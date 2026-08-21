# LLM adapter: the plug for abstractive reflection

Willow's reflection layer (`willow_substrate.reflection.meditate`)
ships with a **deterministic extractive summariser** as the zero-dep
default: it counts words, lists actors and kinds, and returns a
mechanical inventory. That layer was always designed to be **swappable
for an LLM-authored abstractive summariser**, and this module
(`willow_substrate.llm`) is the plug.

Nothing about the substrate calls an LLM unless you EXPLICITLY hand
one in. No hidden costs. No auto-configuration. No opinion imposed on
users about which model to run.

## The Meditator contract

Any callable that satisfies this shape works as a meditator:

```python
class Meditator(Protocol):
    def draft(self, session_id: str, events: list[Event]) -> str: ...
```

That is the whole boundary. Own your own auth, own your own retries,
own your own cost accounting. The substrate treats the returned string
as opaque and stores it as the meditation event's content, with
`derived_from` pointing at the source turns (identical chain to the
extractive path).

## Three implementations ship

### ClaudeCodeMeditator (recommended, subscription-covered)

The path Anthropic already gives you if you use Claude Code.
`ClaudeCodeMeditator` shells out to the local `claude` binary, which
authenticates with your existing `claude login` session, so each
meditation is covered by your Claude Code subscription rather than
billed per token against an API key.

Zero Python dependencies beyond the standard library. The only external
requirement is that the `claude` binary be on `PATH`.

```bash
# Install Claude Code once, sign in once.
# https://claude.com/claude-code
claude login
```

```python
from willow_substrate.store import EventStore
from willow_substrate.reflection import meditate
from willow_substrate.llm import ClaudeCodeMeditator

store = EventStore("~/.willow")
meditator = ClaudeCodeMeditator()  # discovers `claude` on PATH

event = meditate(store, "terminal-a", meditator=meditator)
print(event.content)
# LLM-authored abstractive meditation, subscription-covered.
print(event.metadata["generator"])
# "ClaudeCodeMeditator"
```

Or from the CLI:

```bash
willow meditate --session terminal-a --llm claude-code
```

**Model selection:** the default is `sonnet` (Claude Code accepts short
aliases: `sonnet`, `opus`, `haiku`, etc., as well as full ids like
`claude-sonnet-4-5`). Override with `--llm-timeout-s N` for the timeout,
or `WILLOW_CLAUDE_MODEL=opus` for the model.

**System prompt:** the default framing tells the model to name what
the session was about, who spoke, what participants moved toward, and
any commitments or corrections. It also tells it not to invent facts.
Override via the `system_prompt=` constructor argument, or append via
`extra_args=["--append-system-prompt", "..."]` if you want to layer
extra framing on top of the default.

**Cost:** each call spends one Claude Code turn against your plan. No
per-token invoice; no incremental $ line item. If you exceed your plan
limit, `claude` itself surfaces that; the meditator will raise a
`RuntimeError` with the CLI's stderr tail included.

**Auth:** entirely on the CLI's side. Willow never touches your
credentials. Same auth story you already trust with `claude` day to
day.

### AnthropicMeditator (per-token API-billed)

```bash
pip install "willow-substrate[anthropic]"
export ANTHROPIC_API_KEY="sk-ant-..."
```

```python
from willow_substrate.store import EventStore
from willow_substrate.reflection import meditate
from willow_substrate.llm import AnthropicMeditator

store = EventStore("~/.willow")
meditator = AnthropicMeditator()  # reads ANTHROPIC_API_KEY

event = meditate(store, "terminal-a", meditator=meditator)
print(event.content)
# LLM-authored abstractive meditation, not the extractive inventory.
print(event.metadata["generator"])
# "AnthropicMeditator"
```

Or from the CLI:

```bash
willow meditate --session terminal-a --llm anthropic
```

**Model selection:** the default is `claude-sonnet-4-5`. Override with
`--llm-max-tokens N` on the CLI or by setting
`WILLOW_ANTHROPIC_MODEL=claude-haiku-4-5-20251001` (or any model id
your account has access to) in the environment.

**System prompt:** the default framing tells the model to name what
the session was about, who spoke, what participants moved toward, and
any commitments or corrections. It also tells it not to invent facts.
Override via the `system_prompt=` constructor argument if you want a
different voice.

**Cost:** each call makes ONE Anthropic API request. A typical LoCoMo-
sized session (~20 turns, ~2k input tokens, ~200 output tokens) at
Sonnet-4.5 rates costs well under one US cent. Users sensitive to
spending should pick a smaller model (Haiku), cap session sizes before
calling `meditate`, or write their own local-LLM meditator.

### MockMeditator (tests)

```python
from willow_substrate.llm import MockMeditator

m = MockMeditator(template="TEST: {session} with {n} events")
event = meditate(store, "terminal-a", meditator=m)
assert event.content == "TEST: terminal-a with 5 events"
```

Deterministic, free, never calls a network. Used throughout
`tests/test_llm_adapter.py` so the substrate's own CI never spends
tokens.

## Writing your own adapter

Just implement `draft`. Here is a sketch for OpenAI:

```python
from openai import OpenAI
from willow_substrate.events import Event

class OpenAIMeditator:
    def __init__(self, model="gpt-5-mini"):
        self.client = OpenAI()  # reads OPENAI_API_KEY
        self.model = model

    def draft(self, session_id: str, events: list[Event]) -> str:
        transcript = "\n".join(
            f"- [{e.actor}] {e.content}" for e in events
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Summarise this session."},
                {"role": "user", "content": transcript},
            ],
        )
        return response.choices[0].message.content
```

Or a local one using Ollama:

```python
import ollama
from willow_substrate.events import Event

class OllamaMeditator:
    def __init__(self, model="llama3.1:8b"):
        self.model = model

    def draft(self, session_id: str, events: list[Event]) -> str:
        transcript = "\n".join(
            f"- [{e.actor}] {e.content}" for e in events
        )
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": transcript}],
        )
        return response["message"]["content"]
```

Neither of these ships in the substrate; they are examples. Add
yours as another optional extra if you want it discoverable by
default; otherwise just import your own module and pass instances to
`meditate(..., meditator=X)`.

## Priority order in meditate()

Three ways to produce the meditation text, in this order:

1. **`text=str` (supplied)**: use the caller's exact string; generator
   tag `"supplied"`. Highest priority; a supplied text short-circuits
   the LLM (and its cost) even if a meditator is also passed.
2. **`meditator=Meditator` (LLM-backed)**: call `meditator.draft(...)`;
   generator tag names the meditator class (e.g.,
   `"ClaudeCodeMeditator"`, `"AnthropicMeditator"`, or whatever you
   plugged in). Downstream tooling can filter by generator to keep only
   the extractive floor when a deterministic answer is required.
3. **Neither**: fall back to the deterministic extractive summariser;
   generator tag `"extractive-v1"`. Ships zero-dep, always works.

The `derived_from` chain is identical across all three paths; only
the `content` and `generator` differ.

## What this changes about willow-substrate

Before this plug, the ledger was intelligent about **structure**
(hash chain, corrections, provenance, retrieval) and deterministic
about **content** (extractive summaries, no meaning). Now a caller
who opts in gets intelligent content too, without any change to the
structural guarantees. The substrate stays honest; the smartness is
whatever you plug in.

The Recall@5 = 0.135 measured on LoCoMo with the extractive layer is
the honest floor. Published memory agents (Mem0, LangMem, MemGPT)
reach 0.30-0.50 by drafting LLM-authored meditations; running this
adapter on a LoCoMo-scale corpus should close much of that gap.
Measuring that lift is future work; the plug is now in.
