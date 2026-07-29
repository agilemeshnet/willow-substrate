from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from willow.corpus import import_markdown, parse_frontmatter
from willow.store import EventStore
from willow.vista import VistaBackend


class FrontmatterTests(unittest.TestCase):
    def test_scalars_lists_and_one_level_of_nesting(self):
        data, body = parse_frontmatter(
            "---\n"
            "name: measure-first\n"
            "description: Measure before changing anything\n"
            "standing: true\n"
            "topics: [method, discipline]\n"
            "metadata:\n"
            "  type: feedback\n"
            "---\n"
            "The body survives.\n"
        )
        self.assertEqual(data["name"], "measure-first")
        self.assertTrue(data["standing"])
        self.assertEqual(data["topics"], ["method", "discipline"])
        self.assertEqual(data["metadata"], {"type": "feedback"})
        self.assertEqual(body.strip(), "The body survives.")

    def test_dash_lists(self):
        data, _ = parse_frontmatter(
            "---\ntopics:\n  - alpha\n  - beta\n---\nbody\n"
        )
        self.assertEqual(data["topics"], ["alpha", "beta"])

    def test_document_without_frontmatter_is_unchanged(self):
        data, body = parse_frontmatter("# Plain\n\nNo frontmatter here.\n")
        self.assertEqual(data, {})
        self.assertIn("No frontmatter here.", body)


class CorpusImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.home = Path(self.temp.name) / "home"
        self.corpus = Path(self.temp.name) / "corpus"
        self.corpus.mkdir(parents=True)
        self.store = EventStore(self.home)

    def tearDown(self):
        self.temp.cleanup()

    def _write(self, name: str, text: str) -> Path:
        path = self.corpus / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_import_creates_events_with_metadata(self):
        self._write(
            "measure-first.md",
            "---\n"
            "name: measure-first\n"
            "standing: true\n"
            "topics: [method]\n"
            "metadata:\n"
            "  type: feedback\n"
            "---\n"
            "Measure before changing anything.\n",
        )
        report = import_markdown(self.store, self.corpus)
        self.assertEqual(len(report.created), 1)
        event = report.created[0].event
        self.assertEqual(event.metadata["name"], "measure-first")
        self.assertTrue(event.metadata["standing"])
        self.assertEqual(event.metadata["topics"], ["method"])
        # A nested metadata block is lifted so retrieval can read it flat.
        self.assertEqual(event.metadata["type"], "feedback")
        self.assertEqual(event.metadata["source_path"], "measure-first.md")
        self.assertIn("Measure before changing", event.content)

    def test_reimport_of_unchanged_corpus_appends_nothing(self):
        self._write("one.md", "First note.\n")
        self._write("nested/two.md", "Second note.\n")
        import_markdown(self.store, self.corpus)
        before = self.store.count()

        report = import_markdown(self.store, self.corpus)
        self.assertEqual(len(report.unchanged), 2)
        self.assertEqual(len(report.created), 0)
        self.assertEqual(self.store.count(), before)

    def test_changed_file_supersedes_rather_than_overwrites(self):
        self._write("one.md", "First version.\n")
        first = import_markdown(self.store, self.corpus).created[0].event

        self._write("one.md", "Second version.\n")
        report = import_markdown(self.store, self.corpus)
        self.assertEqual(len(report.superseded), 1)
        second = report.superseded[0].event

        self.assertEqual(second.supersedes, first.id)
        self.assertIn(first.id, second.derived_from)
        # The original is still in the chain, just no longer active.
        self.assertIsNotNone(self.store.get(first.id))
        active = [event.id for event in self.store.events(limit=50)]
        self.assertIn(second.id, active)
        self.assertNotIn(first.id, active)
        valid, _, error = self.store.verify()
        self.assertTrue(valid, error)

    def test_wikilinks_in_a_corpus_become_waypoints(self):
        self._write(
            "anchor.md",
            "---\nname: anchor\n---\nThe anchor note.\n",
        )
        self._write(
            "pointer.md",
            "---\nname: pointer\n---\nBuilding on [[anchor]] directly.\n",
        )
        import_markdown(self.store, self.corpus)
        result = VistaBackend(self.store).query("anchor", limit=5)
        waypoints = {
            waypoint
            for evidence in result.evidence
            for waypoint in evidence.waypoints
        }
        self.assertTrue(
            any(name == "memory:anchor" for name in waypoints),
            waypoints,
        )

    def test_unreadable_and_oversized_files_are_skipped_not_fatal(self):
        self._write("fine.md", "Readable.\n")
        self._write("huge.md", "x" * 4000)
        report = import_markdown(self.store, self.corpus, max_bytes=1000)
        self.assertEqual(len(report.created), 1)
        self.assertEqual(len(report.skipped), 1)
        self.assertIn("larger than", report.skipped[0][1])

    def test_missing_root_is_reported_clearly(self):
        with self.assertRaises(NotADirectoryError):
            import_markdown(self.store, self.corpus / "absent")


if __name__ == "__main__":
    unittest.main()
