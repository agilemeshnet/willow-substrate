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
)


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


def run_conversation(
    conversation: LocomoConversation,
    config: RunConfig,
    *,
    top_k: int,
    with_consolidation: bool = False,
    consolidation_tau_s: float | None = None,
) -> list[dict[str, Any]]:
    """Score every question in one conversation. Returns per-query rows."""
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        store = EventStore(Path(tmp))
        ingest_into_store(store, conversation)
        backend = _make_backend(
            store, config, conversation,
            with_consolidation=with_consolidation,
            consolidation_tau_s=consolidation_tau_s,
        )

        # ContextBuilder is optional per config (skip for pure retrieval
        # rows). When on, always uses the SAME backend the raw retrieval
        # measured, so the split truly reflects assembly's contribution.
        composer = (
            ContextBuilder(store, relational_backend=backend)
            if config.include_final_context
            else None
        )

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
            if composer is not None:
                for budget in config.context_tokens:
                    ct0 = time.perf_counter()
                    packet = composer.build(q.text, token_budget=budget)
                    context_ms = (time.perf_counter() - ct0) * 1000.0
                    in_context_ids = list(packet.event_ids)
                    per_budget.append(
                        {
                            "context_tokens": budget,
                            "final_context_ids": in_context_ids,
                            "context_latency_ms": round(context_ms, 2),
                        }
                    )

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
                }
            )
    return rows


def run(
    config_path: Path,
    *,
    output_path: Path,
    top_k: int = 20,
    limit_conversations: int | None = None,
    dataset_dir_override: str | None = None,
    with_consolidation: bool = False,
    consolidation_tau_s: float | None = None,
) -> dict[str, Any]:
    """Run one config against the LoCoMo corpus.

    ``dataset_dir_override``, when set, replaces ``config.dataset_dir``
    at call time. This is how the ``--dataset-dir`` CLI flag reaches
    the run; it keeps the shipped configs generic (they point at the
    default corpus dir) while letting operators aim a run at, say, the
    combined locomo10 file without editing anything in-repo.

    ``with_consolidation``: when True, wrap the backend in a
    ConsolidationBackend applying time-decay + recall-frequency scoring
    per Hou et al. 2024 (arXiv 2404.00573). Recall statistics live in
    a per-conversation sidecar file (tempdir; disposed after each
    conversation), so recall counting is within-conversation only in
    this benchmark, which matches the paper's evaluation semantics.
    """
    config = RunConfig.load(config_path)
    effective_dataset_dir = dataset_dir_override or config.dataset_dir
    conversations = load_locomo_conversations(
        REPO_ROOT / effective_dataset_dir
    )
    if limit_conversations is not None:
        conversations = conversations[:limit_conversations]

    all_rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for conversation in conversations:
        all_rows.extend(
            run_conversation(
                conversation, config, top_k=top_k,
                with_consolidation=with_consolidation,
                consolidation_tau_s=consolidation_tau_s,
            )
        )
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
        "with_consolidation": with_consolidation,
        "consolidation_tau_s": consolidation_tau_s if with_consolidation else None,
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
        with_consolidation=args.with_consolidation,
        consolidation_tau_s=tau_s,
    )
    print(
        f"wrote {args.output}: "
        f"config={manifest['config']['name']} "
        f"conversations={manifest['n_conversations']} "
        f"questions={manifest['n_questions']} "
        f"wall_seconds={manifest['wall_seconds']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
