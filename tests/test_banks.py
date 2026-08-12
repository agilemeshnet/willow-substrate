from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from willow_substrate.banks import load_banks, scaffold_banks
from willow_substrate.context import ContextBuilder
from willow_substrate.store import EventStore


class BankTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.home = Path(self.temp.name)
        self.store = EventStore(self.home)

    def tearDown(self):
        self.temp.cleanup()

    def test_no_banks_is_not_an_error(self):
        self.assertEqual(load_banks(self.home), ())
        packet = ContextBuilder(self.store).boot()
        self.assertEqual(packet.banks, ())
        self.assertIn("# Willow boot", packet.markdown)

    def test_core_banks_load_in_fixed_order(self):
        (self.home / "GROUND.md").write_text("Additive only.", encoding="utf-8")
        (self.home / "identity.md").write_text(
            "I am the checking region.",
            encoding="utf-8",
        )
        banks = load_banks(self.home)
        self.assertEqual([bank.name for bank in banks], ["identity", "ground"])

    def test_filename_case_does_not_change_meaning(self):
        (self.home / "IDENTITY.md").write_text("Upper case.", encoding="utf-8")
        banks = load_banks(self.home)
        self.assertEqual([bank.name for bank in banks], ["identity"])
        self.assertEqual(banks[0].text, "Upper case.")

    def test_empty_bank_file_is_skipped(self):
        (self.home / "identity.md").write_text("   \n", encoding="utf-8")
        self.assertEqual(load_banks(self.home), ())

    def test_extra_banks_directory_is_appended_in_filename_order(self):
        (self.home / "identity.md").write_text("Region.", encoding="utf-8")
        extra = self.home / "banks"
        extra.mkdir()
        (extra / "b-second.md").write_text("Second.", encoding="utf-8")
        (extra / "a-first.md").write_text("First.", encoding="utf-8")
        banks = load_banks(self.home)
        self.assertEqual(
            [bank.name for bank in banks],
            ["identity", "a-first", "b-second"],
        )

    def test_banks_are_included_whole_at_boot(self):
        constitution = "Rule one. " * 40
        (self.home / "identity.md").write_text(constitution, encoding="utf-8")
        (self.home / "ground.md").write_text("Additive only.", encoding="utf-8")
        self.store.append("Ordinary working message", session_id="terminal-a")
        packet = ContextBuilder(self.store).boot(token_budget=400)
        self.assertIn("Additive only.", packet.markdown)
        self.assertIn(constitution.strip(), packet.markdown)
        self.assertEqual(
            [bank.name for bank in packet.banks],
            ["identity", "ground"],
        )

    def test_banks_survive_a_budget_too_small_for_them(self):
        # The floor is paid first. A budget that cannot afford the constitution
        # loses experience, never identity.
        (self.home / "identity.md").write_text(
            "Identity line. " * 200,
            encoding="utf-8",
        )
        for index in range(20):
            self.store.append(
                f"Working message {index} " + ("detail " * 40),
                session_id="terminal-a",
            )
        packet = ContextBuilder(self.store).boot(token_budget=150)
        self.assertIn("Identity line.", packet.markdown)
        self.assertIn("## Identity bank", packet.markdown)

    def test_scaffold_writes_templates_once_and_never_overwrites(self):
        written = scaffold_banks(self.home)
        self.assertEqual(len(written), 2)
        (self.home / "IDENTITY.md").write_text("Mine.", encoding="utf-8")
        again = scaffold_banks(self.home)
        self.assertEqual(again, ())
        self.assertEqual(
            (self.home / "IDENTITY.md").read_text(encoding="utf-8"),
            "Mine.",
        )

    def test_boot_labels_banks_as_constitutional(self):
        (self.home / "ground.md").write_text("Additive only.", encoding="utf-8")
        packet = ContextBuilder(self.store).boot()
        self.assertIn("never", packet.markdown)
        self.assertIn("## Constitutional ground", packet.markdown)


if __name__ == "__main__":
    unittest.main()
