"""Per-config LoCoMo runner.

Loads a config, iterates conversations, and for each question records
BOTH the raw backend retrieval AND what the ContextBuilder actually
placed into the final context packet. The split is deliberate: it
exposes cases where retrieval located the correct memory but assembly
discarded it. See docs/BENCHMARK_LOCOMO.md.

Output is a self-describing JSON manifest (per-query rows, config
snapshot, dataset revision, willow commit) suitable for score.py and
for pinning in CI regression thresholds later.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from willow_substrate.store import EventStore  # noqa: E402
from willow_substrate.context import ContextBuilder  # noqa: E402
from willow_substrate.backends.factory import make_relational_backend  # noqa: E402

from benchmarks.locomo.adapter import (  # noqa: E402
    LocomoConversation,
    ingest_into_store,
    load_locomo_conversations,
    session_ids_for,
)
from willow_substrate.reflection import meditate  # noqa: E402
from willow_substrate.dreaming import dream  # noqa: E402


@dataclass(frozen=True)
class RunConfig:
    """One benchmark configuration."""

    name: str
    backend: str  # 'sparse' | 'bm25' | 'hybrid' | 'voyage' | 'recent-only' | 'oracle'
    backend_kwargs: dict[str, Any]
    context_tokens: list[int]
    include_final_context: bool = True
    dataset_dir: str = "benchmarks/locomo/data"

    @classmethod
    def load(cls, path: Path) -> "RunConfig":
        with path.open() as fh:
            payload = json.load(fh)
        return cls(
            name=payload["name"],
            backend=payload["backend"],
            backend_kwargs=payload.get("backend_kwargs", {}),
            context_tokens=list(
                payload.get("context_tokens", [256, 512, 1024, 2048])
            ),
            include_final_context=bool(
                payload.get("include_final_context", True)
            ),
            dataset_dir=payload.get("dataset_dir", "benchmarks/locomo/data"),
        )


def _willow_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _make_backend(
    store: EventStore,
    config: RunConfig,
    conversation: LocomoConversation | None = None,
    *,
    with_consolidation: bool = False,
    consolidation_tau_s: float | None = None,
):
    """Build the retrieval callable for a config.

    Special-case configs:
    - 'recent-only': ignore the query; return the most recent K turns
    - 'oracle': return the gold turns for the current question
    - anything else: dispatch through the willow backend factory

    ``with_consolidation``: when True, wrap the returned backend in a
    ConsolidationBackend that applies time-decay + recall-frequency
    reweighting per Hou et al. 2024. Uses the conversation's most-
    recent-turn timestamp as "now" so LoCoMo's absolute times land in a
    sensible decay window.
    """
    if config.backend == "recent-only":
        base = _RecentOnly(store)
    elif config.backend == "oracle":
        if conversation is None:
            raise ValueError("oracle backend requires a conversation")
        base = _Oracle(store, conversation)
    else:
        base = make_relational_backend(
            store, name=config.backend, **config.backend_kwargs
        )
    if not with_consolidation:
        return base
    if config.backend in ("recent-only", "oracle"):
        # These are baseline / upper-bound rows; leaving them
        # unwrapped keeps the comparison points honest.
        return base
    from willow_substrate.backends.consolidation import (
        ConsolidationBackend, DEFAULT_TAU_S,
    )
    now_fn = _make_now_fn(store)
    return ConsolidationBackend(
        base, store, now_fn=now_fn,
        tau_s=consolidation_tau_s if consolidation_tau_s is not None else DEFAULT_TAU_S,
    )


def _make_now_fn(store: EventStore):
    """Return a ``now_fn`` that reports the LATEST event timestamp in
    the store. This is the fair 'now' for a LoCoMo conversation whose
    events span months of simulated time; using wall-clock ``datetime.now``
    would put every event ~2 years in the past and squash the decay.
    """
    def _now():
        latest_ts = ""
        for event in store.events(limit=10_000, active_only=True):
            if event.timestamp > latest_ts:
                latest_ts = event.timestamp
        if not latest_ts:
            from datetime import datetime, timezone
            return datetime.now(timezone.utc)
        from datetime import datetime
        return datetime.fromisoformat(latest_ts.replace("Z", "+00:00"))
    return _now


class _RecentOnly:
    """Baseline: rank by recency, ignore the query."""

    def __init__(self, store: EventStore):
        self.store = store

    def query(
        self, text: str, *, seed_event_ids=(), limit: int = 8, wave_hops: int = 4
    ):
        events = self.store.events(
            limit=limit, active_only=True, ascending=False
        )
        return _pack_result(events, [1.0] * len(events))


class _Oracle:
    """Upper bound: always returns the gold turns (mapped to event ids)."""

    def __init__(self, store: EventStore, conversation: LocomoConversation):
        self.store = store
        self.conversation = conversation
        self._current_question_gold: tuple[str, ...] = ()

    def set_question(self, gold_turn_ids: tuple[str, ...]) -> None:
        self._current_question_gold = gold_turn_ids

    def query(
        self, text: str, *, seed_event_ids=(), limit: int = 8, wave_hops: int = 4
    ):
        mapping = self.conversation.turn_id_to_event_id
        gold_event_ids = [
            mapping[t] for t in self._current_question_gold if t in mapping
        ]
        events = [
            e
            for e in self.store.events(limit=10_000, active_only=True)
            if e.id in set(gold_event_ids)
        ]
        return _pack_result(events, [1.0] * len(events))


def _pack_result(events, scores):
    """Wrap a plain list of events into a VistaResult-shaped surface."""
    from willow_substrate.vista import VistaEvidence, VistaResult

    evidence = tuple(
        VistaEvidence(
            event=event,
            score=float(score),
            vista_score=float(score),
            wave_score=0.0,
            vista_slugs=(),
            waypoints=(),
            channels=("baseline",),
        )
        for event, score in zip(events, scores)
    )
    return VistaResult(
        query="",
        seed_event_ids=(),
        reference_beams=(),
        matches=(),
        evidence=evidence,
        trace=(),
    )


def _run_reflections(store: EventStore, conversation: LocomoConversation) -> dict[str, int]:
    """Invoke Willow's shipped reflection layer over one conversation's store.

    Per-session meditate() collapses each LoCoMo session into a
    meditation event with derived_from pointing at that session's turns.
    Cross-session dream() proposes structural connections between
    distant material; those dreams' derived_from tuples point at the two
    connected source events, so a retrieval that lands a dream still
    ties back to gold via the derived-from chain when the scorer opts in.

    Both are local, deterministic, no API cost. Returns a small telemetry
    dict recorded in the run manifest.
    """
    meditations = 0
    for session_id in session_ids_for(conversation):
        try:
            meditate(store, session_id)
            meditations += 1
        except ValueError:
            # Empty session (all turns dropped as blank); skip cleanly.
            pass
    # limit=50 lets dream propose enough cross-session bridges to matter on
    # LoCoMo's ~19-session conversations without exploding into every pair.
    dreams = len(dream(store, query="", limit=50))
    return {"meditations": meditations, "dreams": dreams}


def _derived_from_for(store: EventStore, event_ids: list[str]) -> dict[str, list[str]]:
    """Look up derived_from tuples for a list of event ids.

    Recorded per row in the manifest so the scorer can expand
    meditation/dream retrievals to their source turns without needing to
    re-open the store (which is a per-conversation tempdir and is gone by
    the time scoring runs).
    """
    if not event_ids:
        return {}
    ids = set(event_ids)
    lookup: dict[str, list[str]] = {}
    for event in store.events(limit=10_000, active_only=True):
        if event.id in ids and event.derived_from:
            lookup[event.id] = list(event.derived_from)
    return lookup


def run_conversation(
    conversation: LocomoConversation,
    config: RunConfig,
    *,
    top_k: int,
    with_reflections: bool = False,
    with_consolidation: bool = False,
    consolidation_tau_s: float | None = None,
    use_cwb: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Score every question in one conversation. Returns per-query rows
    and a telemetry dict about the reflection pass (empty if disabled).

    ``with_reflections`` and ``with_consolidation`` compose orthogonally:
    reflections adds meditation + dream events to the store BEFORE the
    backend is built; consolidation wraps the resulting backend in a
    time-decay + recall-frequency scoring layer per Hou et al. 2024.

    ``use_cwb``: when True, assemble the final context via
    ``willow_substrate.cwb.ContextWindowBuilder`` instead of the flat
    ``ContextBuilder``. The CWB layers (standing + foreground + vista +
    wave, plus banks and prosoche if configured) each contribute event
    ids; ``final_context_ids`` for scoring is ``window.all_event_ids``.
    Cross-checks whether the layered assembler recovers evidence at a
    different rate than the flat packet.
    """
    rows: list[dict[str, Any]] = []
    reflection_stats: dict[str, int] = {}
    with tempfile.TemporaryDirectory() as tmp:
        store = EventStore(Path(tmp))
        ingest_into_store(store, conversation)
        if with_reflections:
            reflection_stats = _run_reflections(store, conversation)
        backend = _make_backend(
            store, config, conversation,
            with_consolidation=with_consolidation,
            consolidation_tau_s=consolidation_tau_s,
        )

        # Choose the composer. Old flat packet vs new layered window.
        cwb_composer = None
        composer = None
        if config.include_final_context:
            if use_cwb:
                from willow_substrate.cwb import ContextWindowBuilder
                cwb_composer = ContextWindowBuilder(
                    store, retrieval_backend=backend,
                )
            else:
                composer = ContextBuilder(store, relational_backend=backend)

        for q in conversation.questions:
            if isinstance(backend, _Oracle):
                backend.set_question(q.gold_turn_ids)

            gold_event_ids = tuple(
                conversation.turn_id_to_event_id[t]
                for t in q.gold_turn_ids
                if t in conversation.turn_id_to_event_id
            )

            t0 = time.perf_counter()
            retrieved = backend.query(q.text, limit=top_k)
            retrieval_ms = (time.perf_counter() - t0) * 1000.0
            retrieved_ids = [ev.event.id for ev in retrieved.evidence]

            context_ms = 0.0
            per_budget: list[dict[str, Any]] = []
            in_context_ids_all: set[str] = set()
            if composer is not None:
                # Old path: flat ContextBuilder, token-budgeted per budget.
                for budget in config.context_tokens:
                    ct0 = time.perf_counter()
                    packet = composer.build(q.text, token_budget=budget)
                    context_ms = (time.perf_counter() - ct0) * 1000.0
                    in_context_ids = list(packet.event_ids)
                    in_context_ids_all.update(in_context_ids)
                    per_budget.append(
                        {
                            "context_tokens": budget,
                            "final_context_ids": in_context_ids,
                            "context_latency_ms": round(context_ms, 2),
                        }
                    )
            elif cwb_composer is not None:
                # New path: layered ContextWindowBuilder. The window has
                # no token budget of its own; we take the deduplicated
                # all_event_ids and truncate proportionally per budget
                # (~4 chars/token, 500 chars/event heuristic like the
                # flat composer's).
                for budget in config.context_tokens:
                    ct0 = time.perf_counter()
                    window = cwb_composer.build(
                        q.text, foreground_k=min(15, budget // 200),
                    )
                    context_ms = (time.perf_counter() - ct0) * 1000.0
                    max_events_for_budget = max(1, budget // 60)
                    in_context_ids = list(
                        window.all_event_ids[:max_events_for_budget]
                    )
                    in_context_ids_all.update(in_context_ids)
                    per_budget.append(
                        {
                            "context_tokens": budget,
                            "final_context_ids": in_context_ids,
                            "context_latency_ms": round(context_ms, 2),
                            "cwb_layer_sizes": {
                                "standing": len(window.standing),
                                "foreground": len(window.foreground),
                                "vista": (
                                    len(window.vista.evidence)
                                    if window.vista else 0
                                ),
                                "wave": (
                                    len(window.wave.evidence)
                                    if window.wave else 0
                                ),
                            },
                        }
                    )

            # Record derived_from for anything the scorer may need to
            # expand: retrieved ids + any id that landed in the final
            # context. Only meditations/dreams/summations carry a
            # non-empty tuple, so this stays cheap.
            interesting_ids = list(set(retrieved_ids) | in_context_ids_all)
            derived_map = _derived_from_for(store, interesting_ids)

            rows.append(
                {
                    "conversation_id": conversation.conversation_id,
                    "question_id": q.question_id,
                    "question_text": q.text,
                    "category": q.category,
                    "gold_event_ids": list(gold_event_ids),
                    "retrieved_event_ids": retrieved_ids,
                    "retrieval_latency_ms": round(retrieval_ms, 2),
                    "per_budget": per_budget,
                    "derived_from": derived_map,
                }
            )
    return rows, reflection_stats


def run(
    config_path: Path,
    *,
    output_path: Path,
    top_k: int = 20,
    limit_conversations: int | None = None,
    dataset_dir_override: str | None = None,
    with_reflections: bool = False,
    with_consolidation: bool = False,
    consolidation_tau_s: float | None = None,
    use_cwb: bool = False,
) -> dict[str, Any]:
    """Run one config against the LoCoMo corpus.

    ``dataset_dir_override``, when set, replaces ``config.dataset_dir``
    at call time. This is how the ``--dataset-dir`` CLI flag reaches
    the run; it keeps the shipped configs generic (they point at the
    default corpus dir) while letting operators aim a run at, say, the
    combined locomo10 file without editing anything in-repo.

    ``with_reflections``, when True, runs Willow's shipped meditate()
    per session and dream() across the store after ingest. The reflection
    events become first-class retrievable evidence; the scorer's
    ``--expand-derived-from`` flag then credits meditation/dream
    retrievals to their source turns when computing recall.

    ``with_consolidation``, when True, wraps the backend in a
    ConsolidationBackend applying time-decay + recall-frequency scoring
    per Hou et al. 2024 (arXiv 2404.00573). Recall statistics live in
    a per-conversation sidecar file (tempdir; disposed after each
    conversation), so recall counting is within-conversation only in
    this benchmark, which matches the paper's evaluation semantics.

    The two flags compose: use both at once to measure the substrate's
    full reflection + consolidation stack.
    """
    config = RunConfig.load(config_path)
    effective_dataset_dir = dataset_dir_override or config.dataset_dir
    conversations = load_locomo_conversations(
        REPO_ROOT / effective_dataset_dir
    )
    if limit_conversations is not None:
        conversations = conversations[:limit_conversations]

    all_rows: list[dict[str, Any]] = []
    reflection_totals = {"meditations": 0, "dreams": 0}
    t0 = time.perf_counter()
    for conversation in conversations:
        rows, stats = run_conversation(
            conversation, config, top_k=top_k,
            with_reflections=with_reflections,
            with_consolidation=with_consolidation,
            consolidation_tau_s=consolidation_tau_s,
            use_cwb=use_cwb,
        )
        all_rows.extend(rows)
        for key, value in stats.items():
            reflection_totals[key] = reflection_totals.get(key, 0) + value
    wall_seconds = time.perf_counter() - t0

    manifest = {
        "schema": "willow.locomo-benchmark/v1",
        "config": {
            "name": config.name,
            "backend": config.backend,
            "backend_kwargs": config.backend_kwargs,
            "context_tokens": config.context_tokens,
            "include_final_context": config.include_final_context,
        },
        "top_k": top_k,
        "willow_commit": _willow_commit(),
        "dataset_dir": effective_dataset_dir,
        "with_reflections": with_reflections,
        "reflection_totals": reflection_totals if with_reflections else None,
        "with_consolidation": with_consolidation,
        "consolidation_tau_s": consolidation_tau_s if with_consolidation else None,
        "use_cwb": use_cwb,
        "n_conversations": len(conversations),
        "n_questions": len(all_rows),
        "wall_seconds": round(wall_seconds, 2),
        "rows": all_rows,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument(
        "--limit-conversations",
        type=int,
        default=None,
        help="Cap conversations processed (for smoke tests).",
    )
    ap.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help=(
            "Override the config's dataset_dir. Point at a directory "
            "(walked recursively for *.json) or a single JSON file. "
            "Path is resolved relative to the repo root."
        ),
    )
    ap.add_argument(
        "--with-reflections",
        action="store_true",
        help=(
            "Run Willow's shipped meditate() per session and dream() "
            "across the store after ingesting each conversation. The "
            "meditation and dream events become first-class retrievable "
            "evidence with derived_from links back to source turns."
        ),
    )
    ap.add_argument(
        "--with-consolidation",
        action="store_true",
        help=(
            "Wrap the backend in a ConsolidationBackend (time-decay + "
            "recall-frequency scoring per Hou et al. 2024, arXiv "
            "2404.00573). Free, local, deterministic; adds a per-"
            "conversation recall_stats sidecar."
        ),
    )
    ap.add_argument(
        "--consolidation-tau-days",
        type=float,
        default=3650.0,
        help=(
            "Time-decay half-life in days for the consolidation wrapper. "
            "Default 3650 days (10 years), which makes time-decay a "
            "near-no-op and lets the recall-frequency signal dominate. "
            "Reduce to the paper's 30-day scale for chat-agent use "
            "cases where recency matters; LoCoMo shows a 48% Recall@5 "
            "drop at tau=30d because questions ask about old events."
        ),
    )
    ap.add_argument(
        "--use-cwb",
        action="store_true",
        help=(
            "Assemble final context via ContextWindowBuilder (layered: "
            "standing + foreground + vista + wave) instead of the flat "
            "ContextBuilder. Records per-budget layer sizes in each row "
            "so cwb_layer_sizes can be inspected downstream."
        ),
    )
    args = ap.parse_args(argv)
    tau_s = (
        args.consolidation_tau_days * 86400.0
        if args.with_consolidation
        else None
    )
    manifest = run(
        args.config,
        output_path=args.output,
        top_k=args.top_k,
        limit_conversations=args.limit_conversations,
        dataset_dir_override=args.dataset_dir,
        with_reflections=args.with_reflections,
        with_consolidation=args.with_consolidation,
        consolidation_tau_s=tau_s,
        use_cwb=args.use_cwb,
    )
    reflection_note = (
        f" reflections={manifest['reflection_totals']}"
        if manifest.get("with_reflections") and manifest.get("reflection_totals")
        else ""
    )
    consolidation_note = (
        f" consolidation_tau_days={args.consolidation_tau_days}"
        if manifest.get("with_consolidation")
        else ""
    )
    print(
        f"wrote {args.output}: "
        f"config={manifest['config']['name']} "
        f"conversations={manifest['n_conversations']} "
        f"questions={manifest['n_questions']} "
        f"wall_seconds={manifest['wall_seconds']}"
        f"{reflection_note}{consolidation_note}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
