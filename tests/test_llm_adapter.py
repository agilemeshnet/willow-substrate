"""Tests for the LLM adapter boundary (willow_substrate.llm).

Uses a MockMeditator throughout so the suite never hits a live API
and never spends tokens. Real Anthropic/OpenAI/Ollama adapters are
tested at deployment time by the operator, not in the substrate's
own CI.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from willow_substrate.llm import MockMeditator
from willow_substrate.reflection import meditate
from willow_substrate.store import EventStore


class MockMeditatorTests(unittest.TestCase):
    """The mock itself is deterministic and free; verify its contract."""

    def test_draft_returns_formatted_string(self):
        m = MockMeditator()
        # Fake events; MockMeditator doesn't care about the shape as
        # long as it can measure len().
        got = m.draft("chat", [object()] * 3)
        self.assertEqual(got, "MOCK: session=chat events=3")

    def test_draft_records_calls_for_inspection(self):
        m = MockMeditator()
        m.draft("a", [1, 2])
        m.draft("b", [1, 2, 3])
        self.assertEqual(m.calls, [("a", 2), ("b", 3)])

    def test_custom_template_wins(self):
        m = MockMeditator(template="stub[{session},{n}]")
        self.assertEqual(m.draft("s", [1]), "stub[s,1]")


class MeditateWithLLMAdapterTests(unittest.TestCase):
    """meditate() must accept a Meditator and use its output verbatim
    (after strip()), tagging the generator with the class name so
    downstream tooling can distinguish extractive vs abstractive events."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.home = Path(self.temp.name)
        self.store = EventStore(self.home)
        # Two events in one session so meditate has material to draft over.
        self.store.append(
            "Talked about the connectome and recurrent motifs.",
            actor="peter", session_id="chat",
        )
        self.store.append(
            "Decided to measure before drawing conclusions.",
            actor="peter", session_id="chat",
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_meditator_output_lands_as_event_content(self):
        m = MockMeditator(template="ABSTRACTIVE: {session} ({n} events)")
        event = meditate(self.store, "chat", meditator=m)
        self.assertEqual(event.content, "ABSTRACTIVE: chat (2 events)")
        self.assertEqual(event.kind, "meditation")

    def test_generator_tag_names_the_meditator_class(self):
        m = MockMeditator()
        event = meditate(self.store, "chat", meditator=m)
        self.assertEqual(event.metadata["generator"], "MockMeditator")

    def test_derived_from_still_links_to_source_events(self):
        """The abstractive path must preserve the same provenance chain
        the extractive path does; the ledger contract does not change
        because the drafting engine did."""
        source_ids = [
            e.id for e in self.store.events(session_id="chat")
        ]
        m = MockMeditator()
        event = meditate(self.store, "chat", meditator=m)
        self.assertEqual(set(event.derived_from), set(source_ids))

    def test_supplied_text_still_wins_when_both_provided(self):
        """text= is priority 1; meditator= is priority 2; extractive
        is the fallback. If a caller supplies text explicitly, the
        meditator is not invoked (and doesn't cost anything)."""
        m = MockMeditator()
        event = meditate(
            self.store, "chat", text="hand-authored", meditator=m,
        )
        self.assertEqual(event.content, "hand-authored")
        self.assertEqual(event.metadata["generator"], "supplied")
        self.assertEqual(m.calls, [])  # meditator never called

    def test_extractive_fallback_when_no_meditator_and_no_text(self):
        """The zero-dep behaviour must be unchanged: no meditator, no
        text, extractive summariser runs, generator tag is
        'extractive-v1'."""
        event = meditate(self.store, "chat")
        self.assertEqual(event.metadata["generator"], "extractive-v1")
        # Sanity: extractive output mentions the session id.
        self.assertIn("chat", event.content)

    def test_empty_meditator_output_raises_visibly(self):
        """If the LLM returned an empty string (whitespace-only), that
        is a real problem downstream; fail loudly rather than write a
        blank meditation event."""
        empty = MockMeditator(template="   \n  ")
        with self.assertRaises(ValueError) as ctx:
            meditate(self.store, "chat", meditator=empty)
        self.assertIn("empty", str(ctx.exception).lower())


class AnthropicAdapterImportGuardTests(unittest.TestCase):
    """AnthropicMeditator must fail at construction (not at module
    import) if the extra is missing. Users who never enable it must
    never see the missing-dep error."""

    def test_module_imports_without_the_extra(self):
        # This test file itself imports willow_substrate.llm without
        # the [anthropic] extra required. If this import failed, the
        # test module would never load. Simple assertion that we got
        # here at all is the check.
        import willow_substrate.llm as llm_mod
        self.assertTrue(hasattr(llm_mod, "AnthropicMeditator"))

    def test_construction_without_extra_raises_clear_import_error(self):
        """Attempt construction; if anthropic SDK is installed the test
        skips (we can't easily un-import it). If it is not, we assert
        the correct ImportError message."""
        try:
            import anthropic  # noqa: F401
            self.skipTest("anthropic SDK is installed; cannot test the guard")
        except ImportError:
            pass
        from willow_substrate.llm import AnthropicMeditator
        with self.assertRaises(ImportError) as ctx:
            AnthropicMeditator(api_key="fake-not-used")
        self.assertIn("[anthropic]", str(ctx.exception))


class ClaudeCodeMeditatorTests(unittest.TestCase):
    """ClaudeCodeMeditator shells out to the `claude` binary. Tests mock
    subprocess.run so the suite never invokes the real CLI (which would
    spend the operator's subscription budget). Coverage: binary discovery,
    command construction, stdin transcript delivery, stdout capture,
    error paths (missing binary, non-zero exit, timeout)."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.home = Path(self.temp.name)
        self.store = EventStore(self.home)
        self.store.append(
            "First turn: the ledger boots.",
            actor="peter", session_id="chat",
        )
        self.store.append(
            "Second turn: we decide to measure before drawing conclusions.",
            actor="peter", session_id="chat",
        )
        self.events = self.store.events(session_id="chat")

    def tearDown(self):
        self.temp.cleanup()

    def test_missing_binary_raises_at_construction(self):
        """No `claude` on PATH => RuntimeError at __init__, not at draft().
        Users without Claude Code installed find out immediately."""
        from unittest.mock import patch
        from willow_substrate.llm import ClaudeCodeMeditator
        with patch("shutil.which", return_value=None):
            with self.assertRaises(RuntimeError) as ctx:
                ClaudeCodeMeditator()
        msg = str(ctx.exception)
        self.assertIn("claude", msg)
        self.assertIn("PATH", msg)

    def test_explicit_cli_path_bypasses_discovery(self):
        """Passing cli_path= directly skips shutil.which; useful for tests
        and for operators with a non-standard install."""
        from willow_substrate.llm import ClaudeCodeMeditator
        m = ClaudeCodeMeditator(cli_path="/opt/whatever/claude")
        self.assertEqual(m.cli_path, "/opt/whatever/claude")

    def test_draft_invokes_claude_with_expected_flags(self):
        """The subprocess call must include -p, --model, --system-prompt,
        --output-format text, and pipe the transcript on stdin."""
        from unittest.mock import patch, MagicMock
        from willow_substrate.llm import ClaudeCodeMeditator
        completed = MagicMock(returncode=0, stdout="A drafted meditation.\n", stderr="")
        with patch("subprocess.run", return_value=completed) as run_mock:
            m = ClaudeCodeMeditator(cli_path="/x/claude", model="opus")
            got = m.draft("chat", self.events)
        self.assertEqual(got, "A drafted meditation.")
        args, kwargs = run_mock.call_args
        cmd = args[0]
        self.assertEqual(cmd[0], "/x/claude")
        self.assertIn("-p", cmd)
        self.assertIn("--model", cmd)
        self.assertIn("opus", cmd)
        self.assertIn("--system-prompt", cmd)
        self.assertIn("--output-format", cmd)
        self.assertIn("text", cmd)
        # Transcript must reach the CLI via stdin, not argv.
        self.assertIn("Session id: chat", kwargs["input"])
        self.assertIn("[peter (message)] First turn", kwargs["input"])
        self.assertTrue(kwargs.get("capture_output"))
        self.assertTrue(kwargs.get("text"))

    def test_model_falls_back_to_env_var_then_default(self):
        """Precedence: constructor arg > WILLOW_CLAUDE_MODEL env var >
        the module default ('sonnet')."""
        import os
        from unittest.mock import patch
        from willow_substrate.llm import ClaudeCodeMeditator, DEFAULT_CLAUDE_CODE_MODEL
        # Env var wins when arg omitted.
        with patch.dict(os.environ, {"WILLOW_CLAUDE_MODEL": "haiku-4-5"}):
            m = ClaudeCodeMeditator(cli_path="/x/claude")
        self.assertEqual(m.model, "haiku-4-5")
        # Default wins when neither given.
        env_no_var = {
            k: v for k, v in os.environ.items() if k != "WILLOW_CLAUDE_MODEL"
        }
        with patch.dict(os.environ, env_no_var, clear=True):
            m = ClaudeCodeMeditator(cli_path="/x/claude")
        self.assertEqual(m.model, DEFAULT_CLAUDE_CODE_MODEL)

    def test_nonzero_exit_raises_runtimeerror_with_stderr_tail(self):
        """A crashed CLI must surface, not silently emit an empty
        meditation. The error carries the last bit of stderr for triage."""
        from unittest.mock import patch, MagicMock
        from willow_substrate.llm import ClaudeCodeMeditator
        completed = MagicMock(
            returncode=1, stdout="", stderr="authentication failed: run `claude login`",
        )
        with patch("subprocess.run", return_value=completed):
            m = ClaudeCodeMeditator(cli_path="/x/claude")
            with self.assertRaises(RuntimeError) as ctx:
                m.draft("chat", self.events)
        msg = str(ctx.exception)
        self.assertIn("exited 1", msg)
        self.assertIn("authentication failed", msg)

    def test_timeout_raises_runtimeerror(self):
        """A hung CLI must not block the substrate; surface the timeout
        with the session id so the caller knows which meditation failed."""
        import subprocess as sp
        from unittest.mock import patch
        from willow_substrate.llm import ClaudeCodeMeditator
        with patch(
            "subprocess.run",
            side_effect=sp.TimeoutExpired(cmd="claude", timeout=1.0),
        ):
            m = ClaudeCodeMeditator(cli_path="/x/claude", timeout_s=1)
            with self.assertRaises(RuntimeError) as ctx:
                m.draft("chat", self.events)
        self.assertIn("timed out", str(ctx.exception).lower())
        self.assertIn("chat", str(ctx.exception))

    def test_empty_stdout_falls_back_to_placeholder_not_empty_string(self):
        """If the model returned nothing, we still write a meditation event
        with a legible marker so a downstream reader can see the model was
        silent, distinct from an outright failure."""
        from unittest.mock import patch, MagicMock
        from willow_substrate.llm import ClaudeCodeMeditator
        completed = MagicMock(returncode=0, stdout="   \n  ", stderr="")
        with patch("subprocess.run", return_value=completed):
            m = ClaudeCodeMeditator(cli_path="/x/claude")
            got = m.draft("chat", self.events)
        self.assertEqual(got, "(model returned no text)")

    def test_empty_session_raises_before_shelling_out(self):
        """Calling draft() with no events must fail before spawning the
        subprocess; nothing to summarise, nothing to spend budget on."""
        from unittest.mock import patch
        from willow_substrate.llm import ClaudeCodeMeditator
        with patch("subprocess.run") as run_mock:
            m = ClaudeCodeMeditator(cli_path="/x/claude")
            with self.assertRaises(ValueError):
                m.draft("chat", [])
        run_mock.assert_not_called()

    def test_extra_args_are_appended(self):
        """extra_args land at the end of the command list so operators can
        pass --append-system-prompt or similar without losing the core flags."""
        from unittest.mock import patch, MagicMock
        from willow_substrate.llm import ClaudeCodeMeditator
        completed = MagicMock(returncode=0, stdout="ok", stderr="")
        with patch("subprocess.run", return_value=completed) as run_mock:
            m = ClaudeCodeMeditator(
                cli_path="/x/claude",
                extra_args=["--append-system-prompt", "extra"],
            )
            m.draft("chat", self.events)
        cmd = run_mock.call_args[0][0]
        self.assertEqual(cmd[-2:], ["--append-system-prompt", "extra"])

    def test_generator_tag_flows_through_meditate(self):
        """When wired via meditate(..., meditator=ClaudeCodeMeditator), the
        stored event's generator metadata must name the class so the
        provenance chain can distinguish subscription-routed from API-routed."""
        from unittest.mock import patch, MagicMock
        from willow_substrate.llm import ClaudeCodeMeditator
        completed = MagicMock(
            returncode=0, stdout="A subscription-routed meditation.", stderr="",
        )
        with patch("subprocess.run", return_value=completed):
            adapter = ClaudeCodeMeditator(cli_path="/x/claude")
            event = meditate(self.store, "chat", meditator=adapter)
        self.assertEqual(event.content, "A subscription-routed meditation.")
        self.assertEqual(event.metadata["generator"], "ClaudeCodeMeditator")


if __name__ == "__main__":
    unittest.main()
