from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from willow_substrate.connections import find_connections
from willow_substrate.context import ContextBuilder
from willow_substrate.samples import load_temporal_sample
from willow_substrate.store import EventStore
from willow_substrate.vista import VistaBackend


SAMPLE = Path(__file__).parents[1] / "examples" / "temporal-bird-study.json"


class VistaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = EventStore(Path(self.temp.name))

    def tearDown(self):
        self.temp.cleanup()

    def test_reference_beam_restores_shape_neighbour_without_shared_words(self):
        flock = self.store.append(
            "The flock vanishes when its birds land.",
            kind="meditation",
            session_id="field",
            metadata={
                "topics": ["collective-motion"],
                "idea_shape": ["identity:relational"],
            },
        )
        institution = self.store.append(
            "A temporary institution exists only while participation continues.",
            kind="research_result",
            session_id="sociology",
            metadata={
                "topics": ["temporary-institutions"],
                "idea_shape": ["identity:relational"],
            },
        )
        self.store.append(
            "Copper oxidation measurements need another calibration run.",
            kind="research_result",
            session_id="chemistry",
            metadata={"topics": ["spectroscopy"]},
        )

        result = VistaBackend(self.store).query(
            flock.content,
            seed_event_ids=(flock.id,),
            limit=10,
        )
        evidence = result.evidence_for(institution.id)

        self.assertIsNotNone(evidence)
        assert evidence is not None
        self.assertIn("vista", evidence.channels)
        self.assertIn("shape:identity:relational", evidence.waypoints)
        self.assertTrue(result.matches)
        self.assertIn(
            "shape:identity:relational",
            {beam.name for beam in result.reference_beams},
        )

    def test_wave_reaches_a_multi_hop_relational_neighbour(self):
        first = self.store.append(
            "Amber observation.",
            session_id="one",
            metadata={"topics": ["alpha"]},
        )
        self.store.append(
            "Cobalt observation.",
            session_id="two",
            metadata={"topics": ["alpha", "beta"]},
        )
        third = self.store.append(
            "Violet observation.",
            session_id="three",
            metadata={"topics": ["beta", "gamma"]},
        )

        result = VistaBackend(
            self.store,
            cluster_threshold=0.95,
        ).query(
            first.content,
            seed_event_ids=(first.id,),
            limit=10,
            wave_hops=4,
        )
        evidence = result.evidence_for(third.id)

        self.assertIsNotNone(evidence)
        assert evidence is not None
        self.assertGreater(evidence.wave_score, 0)
        self.assertIn("wave", evidence.channels)
        self.assertIn("topic:beta", evidence.waypoints)

    def test_superseded_events_are_absent_from_the_projection(self):
        old = self.store.append(
            "The colony count is forty-two.",
            metadata={"topics": ["colony-count"]},
        )
        correction = self.store.correct(
            old.id,
            "Image review establishes twenty-four occupied nests.",
            metadata={"topics": ["colony-count"]},
        )

        projection = VistaBackend(self.store).project()

        self.assertNotIn(old.id, projection.events)
        self.assertIn(correction.id, projection.events)
        self.assertTrue(
            all(old.id not in vista.member_ids for vista in projection.vistas)
        )

    def test_timestamps_do_not_change_vista_geometry(self):
        first_home = tempfile.TemporaryDirectory()
        second_home = tempfile.TemporaryDirectory()
        self.addCleanup(first_home.cleanup)
        self.addCleanup(second_home.cleanup)
        left = EventStore(Path(first_home.name))
        right = EventStore(Path(second_home.name))
        rows = [
            (
                "evt-a",
                "A flock retains identity through changing members.",
                {"idea_shape": ["identity:relational"]},
            ),
            (
                "evt-b",
                "An institution persists through participant turnover.",
                {"idea_shape": ["identity:relational"]},
            ),
            (
                "evt-c",
                "The telescope mirror needs recoating.",
                {"topics": ["optics"]},
            ),
        ]
        early = "2030-01-01T00:00:00+00:00"
        late = "2030-03-01T00:00:00+00:00"
        for index, (event_id, content, metadata) in enumerate(rows):
            left.append(
                content,
                event_id=event_id,
                metadata=metadata,
                timestamp=early if index != 1 else late,
            )
            right.append(
                content,
                event_id=event_id,
                metadata=metadata,
                timestamp=late if index != 1 else early,
            )

        left_projection = VistaBackend(left).project()
        right_projection = VistaBackend(right).project()
        left_shape = sorted(
            (vista.member_ids, round(vista.sigma, 9))
            for vista in left_projection.vistas
        )
        right_shape = sorted(
            (vista.member_ids, round(vista.sigma, 9))
            for vista in right_projection.vistas
        )

        self.assertEqual(left_shape, right_shape)

    def test_context_packet_carries_vista_trace_and_surround(self):
        report = load_temporal_sample(self.store, SAMPLE)
        withholding = report.events["week-06-withholding-decision"]
        trust = report.events["week-07-trust-meditation"]

        packet = ContextBuilder(self.store).build(
            "How did selective disclosure preserve trust?",
            token_budget=1200,
        )

        self.assertIsNotNone(packet.vista)
        assert packet.vista is not None
        self.assertTrue(packet.vista.trace)
        self.assertIn(
            trust.id,
            set(packet.event_ids) | set(packet.vista.event_ids),
        )
        self.assertIn(
            withholding.id,
            set(packet.event_ids) | set(packet.vista.seed_event_ids),
        )

    def test_connection_finder_reports_vista_and_wave_as_separate_channels(self):
        report = load_temporal_sample(self.store, SAMPLE)
        trust = report.events["week-07-trust-meditation"]
        construction = report.events["week-02-construction-pressure"]

        matches = find_connections(
            self.store,
            seed_event_id=trust.id,
            include_vista=True,
            limit=20,
        )
        match = next(
            item for item in matches
            if item.event.id == construction.id
        )

        self.assertIn("shape", match.channels)
        self.assertIn("vista", match.channels)
        self.assertIn("wave", match.channels)
        self.assertGreater(match.vista_score, 0)
        self.assertGreater(match.wave_score, 0)
        self.assertIn(
            "shape:boundary:constrains-and-enables",
            match.relational_waypoints,
        )

    def test_projection_cap_is_visible_in_the_decision_trace(self):
        for index in range(3):
            self.store.append(
                f"Observation {index} about a distinct subject.",
                metadata={"topics": [f"subject-{index}"]},
            )

        backend = VistaBackend(self.store, max_events=2)
        projection = backend.project()
        result = backend.query("subject 2")

        self.assertEqual(len(projection.events), 2)
        self.assertTrue(projection.truncated)
        self.assertTrue(
            any("2-event reference backend cap" in line for line in result.trace)
        )

    def test_high_degree_actor_cannot_outweigh_a_specific_shape_beam(self):
        for index in range(224):
            self.store.append(
                f"Independent ledger observation {index}.",
                actor="peter",
            )
        seed = self.store.append(
            "The membership of a flock changes while its identity persists.",
            actor="peter",
            metadata={"idea_shape": ["identity:relational"]},
        )

        result = VistaBackend(self.store).query(
            seed.content,
            seed_event_ids=(seed.id,),
        )
        weights = {beam.name: beam.weight for beam in result.reference_beams}

        self.assertIn("actor:peter", weights)
        self.assertGreater(
            weights["shape:identity:relational"],
            weights["actor:peter"],
        )

    def test_wave_damping_is_configurable_and_validated(self):
        backend = VistaBackend(self.store, wave_damping=0.25)
        self.assertEqual(backend.wave_damping, 0.25)
        with self.assertRaises(ValueError):
            VistaBackend(self.store, wave_damping=-0.01)
        with self.assertRaises(ValueError):
            VistaBackend(self.store, wave_damping=1.01)


if __name__ == "__main__":
    unittest.main()
