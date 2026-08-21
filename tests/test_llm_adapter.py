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


if __name__ == "__main__":
    unittest.main()
