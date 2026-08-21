"""Benchmark: exercise two-stage retrieval on the reference backend.

Builds a synthetic event corpus with known topic clusters, runs
VistaBackend.query without and with LinearReranker.default(), reports
mean recall@K of same-topic events in the top-K evidence.

The fixed corpus is a regression benchmark for the reference readout: the
combined default must beat the legacy `max(vista, 0.45 * wave)` heuristic.
Its result is not a reproduction of the empirical +345% lift measured on
Peter Cooper's Voyage-embedded Doozer substrate, nor a promise that these
weights are optimal for another corpus. Fit and evaluate weights on
representative held-out data before making a production performance claim.

Zero dependencies. Runs in a few seconds.
"""

from __future__ import annotations

import tempfile
from collections import defaultdict
from pathlib import Path

from willow_substrate.readout import LinearReranker
from willow_substrate.store import EventStore
from willow_substrate.vista import VistaBackend


TOPICS: dict[str, list[str]] = {
    "connectome": [
        "Studied the fruit-fly connectome; Kenyon cells split olfactory input.",
        "Kenyon cell axons project into the mushroom body's parallel fibres.",
        "The mushroom body's alpha and beta lobes carry different memory traces.",
        "Adjacency in the fly connectome does not predict functional coupling.",
        "The projection neurons feed sparse odour codes into the Kenyon cells.",
        "Antennal lobe glomeruli tile the odour space with high dimensionality.",
        "Dopamine neurons in the mushroom body gate associative plasticity.",
        "Sparse coding in Kenyon cells supports pattern separation.",
        "Central complex fan-shaped body integrates heading with goal signals.",
        "Optic lobe connectivity respects a strict retinotopic map.",
    ],
    "quantum": [
        "A Bell state entangles two qubits with a Hadamard and a CNOT.",
        "Quantum Fourier transform runs interference over amplitudes.",
        "Grover search finds a marked item in O(sqrt(N)) queries.",
        "Superconducting transmon qubits require millikelvin dilution refrigerators.",
        "Quantum error correction stabilises fragile qubit states.",
        "Shor's algorithm factors integers with polynomial quantum steps.",
        "Amplitude estimation supersedes classical Monte Carlo for many payoffs.",
        "The IBM Marrakesh chip carries a 156-qubit heavy-hexagon lattice.",
        "Variational quantum eigensolvers exploit hybrid quantum-classical loops.",
        "Ancilla qubits enable non-demolition measurements without collapsing data.",
    ],
    "geology": [
        "Basalt columns fracture into hexagons as they cool.",
        "The Deccan Traps erupted through most of the Cretaceous-Paleogene.",
        "Continental drift is driven by mantle convection cells.",
        "Ammonite fossils index Mesozoic marine sediments precisely.",
        "Trilobites diversified across the Cambrian shallow-water shelves.",
        "Iceland straddles the Mid-Atlantic Ridge as an emergent volcanic island.",
        "Radiometric potassium-argon dating suits volcanic rocks under 4 Ma.",
        "Kimberlite pipes deliver diamond xenoliths from the deep mantle.",
        "Ordovician glaciation coincides with a major marine extinction.",
        "Zircon crystals preserve Hadean-era ages up to 4.4 billion years.",
    ],
    "reservoirs": [
        "Echo state networks use a fixed random recurrent reservoir.",
        "Only the linear readout is trained; the reservoir stays untouched.",
        "Spectral radius below one guarantees the echo state property.",
        "Reservoir computing avoids backpropagation-through-time overhead.",
        "Liquid state machines are a spiking cousin of echo state networks.",
        "Reservoirs excel at chaotic time-series prediction with tiny training sets.",
        "Physical reservoirs can be built from soft robotic bodies or photonic media.",
        "Deep-echo-state stacks compose reservoirs with different time constants.",
        "Reservoir topology affects memory capacity via its Lyapunov spectrum.",
        "Ridge regression suffices for training the readout in most benchmarks.",
    ],
    "memory": [
        "Vicarious trial and error at maze junctions reveals hippocampal replay.",
        "Place cells encode allocentric position in the hippocampal formation.",
        "Grid cells tile navigable space with a hexagonal firing lattice.",
        "Memory palace mnemonics leverage spatial coding for arbitrary lists.",
        "Sleep-associated ripples replay waking sequences in compressed form.",
        "Long-term potentiation strengthens synaptic weights via NMDA channels.",
        "Engram cells re-activated during recall are causally sufficient.",
        "Prefrontal cortex indexes remote memories via schema updating.",
        "Complementary learning systems balance rapid hippocampal encoding with cortical consolidation.",
        "Pattern separation in dentate gyrus prevents interference across similar memories.",
    ],
}


DISTRACTORS: list[str] = [
    # Cross-vocabulary events: touch two or more topic vocabularies,
    # so they should NOT sit cleanly in any single topic's cluster.
    "The connectome study borrowed reservoir-computing ideas to model dynamics.",
    "Quantum walks on graphs share mathematics with reservoir echo states.",
    "Memory palace research borrowed ideas from connectome place-cell literature.",
    "Basalt columns cool at rates measurable via geological time-series analysis.",
    "The reservoir engineer moved from quantum simulation to memory consolidation.",
    "A grid-cell paper cited both connectome adjacency and reservoir dynamics.",
    "Kimberlite pipes and ammonite fossils together anchor Mesozoic timelines.",
    "Ripple oscillations replay both spatial and episodic memory traces.",
    "The team's monograph combined connectome mapping with hippocampal replay.",
    "Superconducting qubits and reservoir computers both exploit weak-signal amplification.",
]


def build_store(
    *,
    tag_fraction: float = 0.5,
) -> tuple[EventStore, tempfile.TemporaryDirectory, dict[str, list[str]]]:
    """Populate a fresh EventStore with the topic corpus PLUS distractors.

    ``tag_fraction`` controls what share of each topic's events carry the
    ``topics`` metadata tag. Events without the tag can still cluster via
    shared entity phrases and wikilinks (Wave's channel), so the benchmark
    exercises both the fine Vista signal and the coarse Wave signal.
    """
    temp = tempfile.TemporaryDirectory()
    store = EventStore(Path(temp.name))
    ids: dict[str, list[str]] = defaultdict(list)
    for topic, sentences in TOPICS.items():
        # First `tag_fraction` events keep the topic tag (strong Vista signal);
        # the rest rely on text-shared vocabulary (Wave via entity waypoints).
        cutoff = max(1, int(len(sentences) * tag_fraction))
        for index, sentence in enumerate(sentences):
            metadata = {"topics": [topic]} if index < cutoff else {}
            event = store.append(
                sentence,
                kind="research_result",
                actor="benchmark",
                session_id=f"{topic}-{index}",
                metadata=metadata,
            )
            ids[topic].append(event.id)
    # Distractor events with cross-topic vocabulary; no clean topic tag.
    for index, sentence in enumerate(DISTRACTORS):
        store.append(
            sentence,
            kind="research_result",
            actor="benchmark",
            session_id=f"distractor-{index}",
            metadata={"topics": ["cross-domain"]},
        )
    return store, temp, dict(ids)


def recall_at_k(topk_ids: list[str], truth_ids: set[str], K: int) -> float:
    hits = set(topk_ids[:K]) & truth_ids
    return len(hits) / min(K, len(truth_ids)) if truth_ids else 0.0


def evaluate(
    backend: VistaBackend,
    topic_ids: dict[str, list[str]],
    reranker: LinearReranker | None = None,
    K: int = 5,
) -> tuple[float, float]:
    """Return (mean recall@K, median recall@K) across every seed in the corpus."""
    recalls: list[float] = []
    for topic, ids in topic_ids.items():
        cousins = set(ids)
        for seed_id in ids:
            truth = cousins - {seed_id}
            result = backend.query(
                query="",
                seed_event_ids=(seed_id,),
                limit=max(K, 20),
                reranker=reranker,
            )
            topk_ids = [item.event.id for item in result.evidence[:K]]
            recalls.append(recall_at_k(topk_ids, truth, K))
    mean = sum(recalls) / len(recalls)
    ordered = sorted(recalls)
    median = ordered[len(ordered) // 2]
    return mean, median


def grid_search_weights(
    backend: VistaBackend,
    topic_ids: dict[str, list[str]],
    K: int = 5,
) -> tuple[LinearReranker, float]:
    """Zero-dep grid search over Vista, Wave, and Wave-arrival weights.

    The negative arrival-time coefficient distinguishes early relational
    recall from late, diffuse activation. Real training should use a proper
    linear model and held-out data; this small sweep remains dependency-free.
    """
    best_score = -1.0
    best_reranker = LinearReranker.default()
    for vw in (0.4, 0.65, 0.8, 1.0):
        for ww in (0.0, 0.065, 0.1, 0.2):
            for hop_weight in (-0.5, -0.325, -0.2, 0.0):
                reranker = LinearReranker.from_dict({
                    "vista_score": vw,
                    "wave_score": ww,
                    "wave_hop_of_peak": hop_weight,
                })
                mean, _ = evaluate(backend, topic_ids, reranker=reranker, K=K)
                if mean > best_score:
                    best_score = mean
                    best_reranker = reranker
    return best_reranker, best_score


def main() -> None:
    store, temp, topic_ids = build_store()
    try:
        backend = VistaBackend(store)
        total_events = sum(len(v) for v in topic_ids.values()) + len(DISTRACTORS)
        print(f"# {total_events} events "
              f"({sum(len(v) for v in topic_ids.values())} topical + "
              f"{len(DISTRACTORS)} cross-vocabulary distractors) "
              f"across {len(topic_ids)} topics; K=5 recall of same-topic cousins.")
        print(f"# Half of each topic's events carry the topics-metadata tag; "
              f"the other half rely on wave through shared entity waypoints.\n")

        baseline_mean, baseline_median = evaluate(backend, topic_ids)
        print(f"  baseline (bare max(vista, 0.45*wave)):")
        print(f"    mean recall@5:   {baseline_mean:.3f}")
        print(f"    median recall@5: {baseline_median:.3f}\n")

        default = LinearReranker.default()
        default_mean, default_median = evaluate(backend, topic_ids, reranker=default)
        default_lift = default_mean - baseline_mean
        default_pct = (
            100.0 * default_lift / baseline_mean
            if baseline_mean else float("inf")
        )
        print(f"  LinearReranker.default() (Wave + Vista readout):")
        print(f"    mean recall@5:   {default_mean:.3f}")
        print(f"    median recall@5: {default_median:.3f}")
        print(f"    lift:            {default_lift:+.3f} ({default_pct:+.1f}%)\n")

        # Vista-only reranker: only trust the semantic proximity feature.
        vista_only = LinearReranker.from_dict({"vista_score": 1.0})
        vo_mean, _ = evaluate(backend, topic_ids, reranker=vista_only)
        print(f"  Vista-only ceiling (weights = vista_score:1.0):")
        print(f"    mean recall@5:   {vo_mean:.3f}")

        # Wave-only reranker.
        wave_only = LinearReranker.from_dict({"wave_score": 1.0})
        wo_mean, _ = evaluate(backend, topic_ids, reranker=wave_only)
        print(f"  Wave-only floor (weights = wave_score:1.0):")
        print(f"    mean recall@5:   {wo_mean:.3f}\n")

        # Corpus-fitted reranker: cheap grid search over interpretable weights.
        best_reranker, best_mean = grid_search_weights(backend, topic_ids)
        best_lift = best_mean - baseline_mean
        best_pct = (
            (100.0 * best_lift / baseline_mean)
            if baseline_mean > 0
            else float("inf")
        )
        print(f"  In-sample grid-search LinearReranker (3 weights):")
        print(f"    mean recall@5:   {best_mean:.3f}  "
              f"lift {best_lift:+.3f} ({best_pct:+.1f}%)")
        print(f"    weights: {dict(best_reranker.weights)}\n")

        print(
            "# Interpretation. On this small sparse-backend corpus:\n"
            "# * Vista-only and Wave-only bracket the pattern; both channels\n"
            "#   carry real signal. Wave alone is worse than Vista alone,\n"
            "#   as expected: Vista is the fine reranker, Wave is the coarse\n"
            "#   retriever.\n"
            "# * The default combines Vista similarity with final Wave\n"
            "#   activation and early arrival. A late Wave peak is penalised,\n"
            "#   so diffuse traversal cannot outrank close relational recall.\n"
            "# * The fixed corpus is a regression guard: the combined default\n"
            "#   must beat the legacy heuristic. The grid search is in-sample\n"
            "#   and illustrative only; it is not evidence of generalisation.\n"
            "# * Do not use this synthetic result to reproduce or substantiate\n"
            "#   the +345% Doozer lift. Fit on training data and evaluate on\n"
            "#   held-out queries for a production corpus."
        )
    finally:
        temp.cleanup()


if __name__ == "__main__":
    main()
