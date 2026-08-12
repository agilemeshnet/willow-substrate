"""LLM adapter boundary: the plug for abstractive reflection.

The deterministic reflection primitives in ``willow_substrate.reflection``
count words and list actor/kind statistics; they inventory rather than
summarise. This module is the *plug-in point* the reflection module has
always advertised: give it an LLM-backed ``Meditator`` and ``meditate()``
becomes abstractive.

Nothing here calls a model unless the caller EXPLICITLY hands one in.
No default meditator, no auto-configuration, no hidden API calls. The
willow-substrate ledger stays deterministic and local; the LLM is a
choice the operator makes.

## Contract

A ``Meditator`` is anything with this shape::

    class Meditator(Protocol):
        def draft(self, session_id: str, events: list[Event]) -> str: ...

Implementations own their own auth, their own retries, their own cost
accounting. This module ships two:

- ``AnthropicMeditator``: wraps the ``anthropic`` SDK. Requires the
  ``[anthropic]`` extra (``pip install "willow-substrate[anthropic]"``)
  and an ``ANTHROPIC_API_KEY``. Ships as the reference implementation
  because Anthropic is a common substrate; nothing in the design
  privileges it. The default model comes from
  ``WILLOW_ANTHROPIC_MODEL`` env var (falls back to
  ``claude-sonnet-4-5`` if unset).
- ``MockMeditator``: returns a canned string. Used by tests so the
  suite never hits a live API and never spends tokens.

Anyone can add an OpenAI/Ollama/local adapter following the same
Protocol; none is privileged.

## Cost gate (important)

Calling ``draft()`` on ``AnthropicMeditator`` makes a real API call. The
operator (not this module, not the substrate) is responsible for cost.
For a typical LoCoMo-sized session (20 turns, ~2k tokens in / ~200 out)
the cost at ``claude-sonnet-4-5`` rates is well under one cent. Users
sensitive to spending should pick a smaller model, cap session sizes
before calling meditate, or write their own local-LLM Meditator.
"""
from __future__ import annotations

import os
from typing import Protocol

from willow_substrate.events import Event


DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-5"
DEFAULT_MAX_TOKENS = 400
DEFAULT_SYSTEM_PROMPT = (
    "You are the reflection layer of a memory substrate. Given a series "
    "of turns from one conversation session, draft a concise, honest "
    "meditation (150 words or fewer) that names: what the session was "
    "about, who spoke, what the participants moved toward, and any "
    "commitments or corrections that stand out. Do not invent facts. Do "
    "not use em dashes. Return prose, no bullet lists."
)


class Meditator(Protocol):
    """Anything that turns session events into a meditation string.

    Implementations MUST NOT modify the events or the store; they are a
    read-side transform only. Failure modes should raise exceptions;
    callers of ``meditate(..., meditator=X)`` will surface them.
    """

    def draft(self, session_id: str, events: list[Event]) -> str: ...


class MockMeditator:
    """Deterministic canned Meditator for tests.

    Returns a string that includes the session id and event count so
    test assertions can verify the whole pipeline (meditate -> store
    -> retrieval) without any live API call.
    """

    def __init__(self, template: str = "MOCK: session={session} events={n}"):
        self.template = template
        self.calls: list[tuple[str, int]] = []

    def draft(self, session_id: str, events: list[Event]) -> str:
        self.calls.append((session_id, len(events)))
        return self.template.format(session=session_id, n=len(events))


class AnthropicMeditator:
    """Anthropic-backed abstractive Meditator.

    Import fails at construction time (not at module import) if the
    ``[anthropic]`` extra is missing, so users who never enable this
    adapter never see the missing-dep error.

    Config:
    - ``api_key``: passed through to the SDK; defaults to
      ``ANTHROPIC_API_KEY`` env var.
    - ``model``: Anthropic model id. Defaults to
      ``WILLOW_ANTHROPIC_MODEL`` env var or ``claude-sonnet-4-5``.
    - ``max_tokens``: hard cap on the meditation length (default 400).
    - ``system_prompt``: overrideable framing (default in
      ``DEFAULT_SYSTEM_PROMPT``).
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "AnthropicMeditator requires the [anthropic] extra. "
                'Install with: pip install "willow-substrate[anthropic]"'
            ) from exc

        from anthropic import Anthropic  # local import; only when used

        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set; pass api_key= or export the env var"
            )
        self.model = (
            model
            or os.environ.get("WILLOW_ANTHROPIC_MODEL")
            or DEFAULT_ANTHROPIC_MODEL
        )
        self.max_tokens = max(1, int(max_tokens))
        self.system_prompt = system_prompt
        self._client = Anthropic(api_key=self.api_key)

    def draft(self, session_id: str, events: list[Event]) -> str:
        """Draft an abstractive meditation for one session's events."""
        if not events:
            raise ValueError(
                f"session has no events to meditate over: {session_id}"
            )
        prompt = self._render_prompt(session_id, events)
        message = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        # Concatenate all text blocks; the SDK returns a list of blocks.
        chunks: list[str] = []
        for block in getattr(message, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                chunks.append(text)
        return " ".join(chunks).strip() or "(model returned no text)"

    @staticmethod
    def _render_prompt(session_id: str, events: list[Event]) -> str:
        """Render the session as a short transcript for the LLM.

        Keeps the ordering and speaker labels the events themselves
        carry; nothing invented, nothing paraphrased on the way in.
        """
        lines = [f"Session id: {session_id}", ""]
        for event in events:
            actor = event.actor or "unknown"
            kind = event.kind or "message"
            content = " ".join(event.content.split())
            lines.append(f"- [{actor} ({kind})] {content}")
        lines.append("")
        lines.append("Draft the meditation now.")
        return "\n".join(lines)
