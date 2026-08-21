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
accounting. This module ships three:

- ``ClaudeCodeMeditator``: shells out to a locally-installed ``claude``
  CLI (Anthropic Claude Code). Uses the user's existing ``claude
  login`` credentials, so LLM-authored meditations are covered by the
  Claude Code subscription and do NOT incur per-token API cost. Zero
  Python dependencies beyond the standard library; requires only that
  the ``claude`` binary be on ``PATH``. This is the recommended path
  for anyone already using Claude Code.
- ``AnthropicMeditator``: wraps the ``anthropic`` SDK. Requires the
  ``[anthropic]`` extra (``pip install "willow-substrate[anthropic]"``)
  and an ``ANTHROPIC_API_KEY``. This is the *per-token API-billed*
  path; useful for callers without a Claude Code subscription or who
  want to route around one.
- ``MockMeditator``: returns a canned string. Used by tests so the
  suite never hits a live API and never spends tokens or subscription
  budget.

Anyone can add an OpenAI/Ollama/local adapter following the same
Protocol; none is privileged.

## Cost gate (important)

- ``ClaudeCodeMeditator``: each call spends one turn of the operator's
  Claude Code subscription. No per-token invoice; the plan absorbs it.
- ``AnthropicMeditator``: each call makes ONE per-token-billed API
  request. For a typical LoCoMo-sized session (20 turns, ~2k tokens in
  / ~200 out) the cost at ``claude-sonnet-4-5`` rates is well under
  one cent. Cost-sensitive callers should pick a smaller model, cap
  session sizes before calling meditate, or use the subscription path
  above.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from typing import Protocol

from willow_substrate.events import Event


DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-5"
DEFAULT_CLAUDE_CODE_MODEL = "sonnet"  # Claude Code CLI accepts short aliases
DEFAULT_MAX_TOKENS = 400
DEFAULT_CLAUDE_CODE_TIMEOUT_S = 120
DEFAULT_SYSTEM_PROMPT = (
    "You are the reflection layer of a memory substrate. Given a series "
    "of turns from one conversation session, draft a concise, honest "
    "meditation (150 words or fewer) that names: what the session was "
    "about, who spoke, what the participants moved toward, and any "
    "commitments or corrections that stand out. Do not invent facts. Do "
    "not use em dashes. Return prose, no bullet lists."
)


def _render_transcript_prompt(session_id: str, events: list[Event]) -> str:
    """Render the session as a short transcript for the LLM.

    Keeps the ordering and speaker labels the events themselves carry;
    nothing invented, nothing paraphrased on the way in. Shared by
    every Meditator implementation in this module so the prompt shape
    is uniform across adapters.
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


class ClaudeCodeMeditator:
    """Subscription-routed abstractive Meditator that shells out to the
    Claude Code CLI (``claude``).

    Uses the operator's existing ``claude login`` credentials. LLM-authored
    meditations therefore consume Claude Code subscription budget rather
    than incurring per-token API cost. Zero Python dependencies beyond the
    standard library; the only external requirement is that the ``claude``
    binary be on ``PATH``.

    Install Claude Code from https://claude.com/claude-code and run
    ``claude login`` once. From that point every ``ClaudeCodeMeditator``
    call is covered by your plan.

    Config:
    - ``cli_path``: absolute path to the ``claude`` binary. Defaults to
      ``shutil.which("claude")``.
    - ``model``: model alias or full id passed to ``claude --model``.
      Defaults to ``WILLOW_CLAUDE_MODEL`` env var, else ``"sonnet"``.
    - ``system_prompt``: framing sent via ``claude --system-prompt``.
    - ``timeout_s``: subprocess timeout in seconds (default 120).
    - ``extra_args``: additional arguments forwarded to ``claude``, in
      case the operator wants to pin ``--append-system-prompt`` or
      similar. Never overrides the four flags this class sets itself
      (``-p``, ``--model``, ``--system-prompt``, ``--output-format``).

    Failure modes:
    - Missing binary at construction time raises ``RuntimeError`` so no
      substrate meditation ever fails silently at first draft.
    - Non-zero exit or a timeout raises ``RuntimeError`` at ``draft()``
      time with the CLI's stderr tail included.
    - Empty stdout returns the literal string ``"(model returned no
      text)"`` so the meditation event still writes with clear provenance.
    """

    def __init__(
        self,
        *,
        cli_path: str | None = None,
        model: str | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        timeout_s: int = DEFAULT_CLAUDE_CODE_TIMEOUT_S,
        extra_args: list[str] | None = None,
    ):
        resolved = cli_path or shutil.which("claude")
        if not resolved:
            raise RuntimeError(
                "`claude` CLI not found on PATH. Install Claude Code from "
                "https://claude.com/claude-code and run `claude login`."
            )
        self.cli_path = resolved
        self.model = (
            model or os.environ.get("WILLOW_CLAUDE_MODEL") or DEFAULT_CLAUDE_CODE_MODEL
        )
        self.system_prompt = system_prompt
        self.timeout_s = max(1, int(timeout_s))
        self.extra_args = list(extra_args) if extra_args else []

    def draft(self, session_id: str, events: list[Event]) -> str:
        """Draft an abstractive meditation for one session's events.

        Sends the transcript to ``claude -p`` on stdin so long sessions
        do not run into any argv length limit. Prints one turn's worth
        of assistant response to stdout, which is what we capture.
        """
        if not events:
            raise ValueError(
                f"session has no events to meditate over: {session_id}"
            )
        prompt = _render_transcript_prompt(session_id, events)
        cmd = [
            self.cli_path,
            "-p",
            "--model", self.model,
            "--system-prompt", self.system_prompt,
            "--output-format", "text",
            *self.extra_args,
        ]
        try:
            proc = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"claude CLI timed out after {self.timeout_s}s "
                f"drafting session {session_id}"
            ) from exc
        if proc.returncode != 0:
            tail = (proc.stderr or "").strip()[-400:]
            raise RuntimeError(
                f"claude CLI exited {proc.returncode} for session "
                f"{session_id}: {tail}"
            )
        return proc.stdout.strip() or "(model returned no text)"


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
