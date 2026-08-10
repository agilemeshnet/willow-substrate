"""Recall benchmark harness for willow-substrate backends.

Runs a fixed set of ground-truth queries against the same corpus through
each available backend and reports recall@k, MRR, and per-query latency.
Zero-dep; if the [vista] extra is installed, the Voyage backend joins the
comparison (with a mock embedder by default so no API key is needed, or
with real Voyage when VOYAGE_API_KEY is set and --real-voyage is passed).

Invoke with:

    python -m benchmarks.recall

Or as a module:

    from benchmarks.recall.harness import run
    results = run()

The returned structure is JSON-serialisable and stable across releases so
CI can pin regression thresholds against it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from willow_substrate.samples import load_temporal_sample  # noqa: E402
from willow_substrate.store import EventStore  # noqa: E402
from willow_substrate.vista import VistaBackend  # noqa: E402


# --------------------------------------------------------------------------- #
# Corpus + ground truth                                                       #
# --------------------------------------------------------------------------- #


def load_corpus(store: EventStore, corpus_path: Path) -> dict[str, str]:
    """Load the temporal-sample corpus into the store; return {key: event_id}."""
    report = load_temporal_sample(store, corpus_path)
    return {key: event.id for key, event in report.events.items()}


def load_queries(queries_path: Path) -> list[dict[str, Any]]:
    with queries_path.open() as fh:
        payload = json.load(fh)
    return payload["queries"]


# --------------------------------------------------------------------------- #
# Baselines                                                                   #
# --------------------------------------------------------------------------- #


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenise(text: str) -> list[str]:
    return [tok.lower() for tok in _TOKEN_RE.findall(text) if len(tok) > 1]


class BM25Backend:
    """Small, well-commented BM25 baseline.

    Deliberately inline (no external dependency) so the benchmark runs on
    a fresh install; the formula is the standard Okapi BM25 with the
    conventional k1=1.5, b=0.75. If you would prefer the canonical
    rank-bm25 implementation, install with the [bench] extra and swap
    this class out; the results should agree to within rounding.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._events: list[Any] = []
        self._tokens: list[list[str]] = []
        self._doc_freq: Counter[str] = Counter()
        self._avg_len: float = 0.0

    def fit(self, events: Iterable[Any]) -> "BM25Backend":
        self._events = list(events)
        self._tokens = [_tokenise(event.content) for event in self._events]
        self._doc_freq = Counter()
        for tokens in self._tokens:
            for term in set(tokens):
                self._doc_freq[term] += 1
        total_len = sum(len(t) for t in self._tokens)
        self._avg_len = total_len / max(1, len(self._tokens))
        return self

    def query(self, text: str, *, limit: int = 8) -> list[tuple[str, float]]:
        n = len(self._events)
        if n == 0:
            return []
        q_terms = _tokenise(text)
        scores: list[float] = [0.0] * n
        for i, tokens in enumerate(self._tokens):
            counts = Counter(tokens)
            doc_len = len(tokens)
            for term in q_terms:
                if term not in counts:
                    continue
                df = self._doc_freq.get(term, 0)
                if df == 0:
                    continue
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
                tf = counts[term]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * doc_len / max(1e-9, self._avg_len)
                )
                scores[i] += idf * (numerator / denominator)
        ranked = sorted(
            enumerate(scores), key=lambda kv: kv[1], reverse=True
        )
        return [
            (self._events[i].id, float(score))
            for i, score in ranked
            if score > 0.0
        ][:limit]


# --------------------------------------------------------------------------- #
# Deterministic mock Voyage embedder                                          #
# --------------------------------------------------------------------------- #


def _deterministic_topic_vector(text: str, topics: list[str], dim: int = 32):
    """A mock semantic vector.

    Real Voyage embeds each string into a learned dense vector. The mock
    stands in for that behaviour without hitting the paid API: it seeds
    a per-topic 32-dim direction, then adds a small per-string jitter.
    Strings that carry the same topic (or paraphrase queries that name
    the same concept) end up cosine-adjacent.

    Requires numpy; only called from the [vista]-guarded code paths.
    """
    import numpy as np

    vec = np.zeros(dim, dtype=np.float32)
    for topic in topics:
        seed = int(hashlib.sha256(topic.encode("utf-8")).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        vec += rng.standard_normal(dim).astype(np.float32)
    if not topics:
        # Query without a known topic: derive a direction from the tokens.
        for token in _tokenise(text):
            seed = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % (2**32)
            rng = np.random.default_rng(seed)
            vec += 0.5 * rng.standard_normal(dim).astype(np.float32)
    # Per-string jitter so equal-topic strings are close but not identical.
    string_seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(string_seed)
    vec += 0.05 * rng.standard_normal(dim).astype(np.float32)
    return vec


class _MockVoyageEmbedder:
    """Substitute for VoyageEmbedder that uses topic-seeded mock vectors.

    Sees the corpus context to know what topic each event carries; queries
    are given topic hints via the caller passing them through the harness.
    """

    def __init__(self, topic_map: dict[str, list[str]]):
        self.topic_map = topic_map  # id_or_query -> topics
        self._cache: dict[str, Any] = {}

    def embed(self, items):
        result = {}
        for key, text in items:
            if key in self._cache:
                result[key] = self._cache[key]
                continue
            topics = self.topic_map.get(key, [])
            vec = _deterministic_topic_vector(text, topics)
            self._cache[key] = vec
            result[key] = vec
        return result


# --------------------------------------------------------------------------- #
# Backend adapters (produce ranked [(event_id, score), ...] from a query)     #
# --------------------------------------------------------------------------- #


class SparseVistaRunner:
    name = "sparse"

    def __init__(self, store: EventStore):
        self.backend = VistaBackend(store)

    def query(self, text: str, *, limit: int) -> list[tuple[str, float]]:
        result = self.backend.query(text, limit=limit)
        return [(ev.event.id, float(ev.score)) for ev in result.evidence][:limit]


class VoyageMockRunner:
    """VoyageVistaBackend driven by _MockVoyageEmbedder (no API calls)."""

    name = "voyage-mock"

    def __init__(
        self,
        store: EventStore,
        *,
        event_key_to_id: dict[str, str],
        corpus_topics: dict[str, list[str]],
        queries: list[dict[str, Any]],
    ):
        from willow_substrate.backends.vista_voyage import VoyageVistaBackend

        # Build the topic map keyed by event_id + query id string.
        topic_map: dict[str, list[str]] = {}
        for key, event_id in event_key_to_id.items():
            topic_map[event_id] = corpus_topics.get(key, [])
        for query in queries:
            topic_map[f"__query__:{query['text']}"] = [query["topic"]]
        embedder = _MockVoyageEmbedder(topic_map)
        self.backend = VoyageVistaBackend(store, embedder=embedder, min_cluster_size=2)

    def query(self, text: str, *, limit: int) -> list[tuple[str, float]]:
        result = self.backend.query(text, limit=limit)
        return [(ev.event.id, float(ev.score)) for ev in result.evidence][:limit]


class VoyageRealRunner:
    """VoyageVistaBackend with the real Voyage API. Opt-in with --real-voyage."""

    name = "voyage-real"

    def __init__(self, store: EventStore, *, api_key: str):
        from willow_substrate.backends.vista_voyage import VoyageVistaBackend

        self.backend = VoyageVistaBackend(store, api_key=api_key)

    def query(self, text: str, *, limit: int) -> list[tuple[str, float]]:
        result = self.backend.query(text, limit=limit)
        return [(ev.event.id, float(ev.score)) for ev in result.evidence][:limit]


class BM25Runner:
    name = "bm25"

    def __init__(self, store: EventStore):
        events = store.events(limit=10_000, active_only=True)
        self.backend = BM25Backend().fit(events)

    def query(self, text: str, *, limit: int) -> list[tuple[str, float]]:
        return self.backend.query(text, limit=limit)


class HybridRunner:
    """Hybrid RRF: fuses sparse + BM25 + optional dense at query time."""

    name = "hybrid"

    def __init__(
        self,
        store: EventStore,
        *,
        event_key_to_id: dict[str, str] | None = None,
        corpus_topics: dict[str, list[str]] | None = None,
        queries: list[dict[str, Any]] | None = None,
        include_dense: bool = False,
    ):
        from willow_substrate.backends.hybrid import HybridRecallBackend

        dense = None
        if include_dense and event_key_to_id and corpus_topics and queries:
            # Use the mock-embedder Voyage backend so no API call is required.
            from willow_substrate.backends.vista_voyage import VoyageVistaBackend

            topic_map: dict[str, list[str]] = {}
            for key, event_id in event_key_to_id.items():
                topic_map[event_id] = corpus_topics.get(key, [])
            for query in queries:
                topic_map[f"__query__:{query['text']}"] = [query["topic"]]
            embedder = _MockVoyageEmbedder(topic_map)
            dense = VoyageVistaBackend(
                store, embedder=embedder, min_cluster_size=2
            )
        self.backend = HybridRecallBackend(store, dense=dense)

    def query(self, text: str, *, limit: int) -> list[tuple[str, float]]:
        result = self.backend.query(text, limit=limit)
        return [(ev.event.id, float(ev.score)) for ev in result.evidence][:limit]


# --------------------------------------------------------------------------- #
# Metrics                                                                     #
# --------------------------------------------------------------------------- #


@dataclass
class QueryResult:
    query_id: str
    query_text: str
    paraphrase: bool
    relevant: list[str]
    ranked: list[str]
    latency_ms: float

    def recall_at(self, k: int) -> float:
        if not self.relevant:
            return 0.0
        top = set(self.ranked[:k])
        hit = sum(1 for r in self.relevant if r in top)
        return hit / len(self.relevant)

    def reciprocal_rank(self) -> float:
        rels = set(self.relevant)
        for i, eid in enumerate(self.ranked, 1):
            if eid in rels:
                return 1.0 / i
        return 0.0


@dataclass
class BackendReport:
    name: str
    per_query: list[QueryResult]

    def mean_recall(self, k: int) -> float:
        vals = [q.recall_at(k) for q in self.per_query]
        return sum(vals) / max(1, len(vals))

    def mrr(self) -> float:
        vals = [q.reciprocal_rank() for q in self.per_query]
        return sum(vals) / max(1, len(vals))

    def median_latency_ms(self) -> float:
        vals = [q.latency_ms for q in self.per_query]
        return statistics.median(vals) if vals else 0.0

    def summary_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "recall_at_3": round(self.mean_recall(3), 3),
            "recall_at_5": round(self.mean_recall(5), 3),
            "mrr": round(self.mrr(), 3),
            "median_latency_ms": round(self.median_latency_ms(), 2),
            "queries": len(self.per_query),
        }


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #


def _corpus_topic_map(corpus_path: Path) -> dict[str, list[str]]:
    data = json.load(corpus_path.open())
    return {
        event["key"]: list(event.get("metadata", {}).get("topics", []))
        for event in data["events"]
    }


def _run_one(
    runner: Any,
    queries: list[dict[str, Any]],
    event_key_to_id: dict[str, str],
    limit: int,
) -> BackendReport:
    per_query: list[QueryResult] = []
    for q in queries:
        t0 = time.perf_counter()
        ranked = runner.query(q["text"], limit=limit)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        ranked_ids = [rid for rid, _score in ranked]
        relevant_ids = [event_key_to_id[key] for key in q["relevant_keys"]]
        per_query.append(
            QueryResult(
                query_id=q["id"],
                query_text=q["text"],
                paraphrase=bool(q.get("paraphrase")),
                relevant=relevant_ids,
                ranked=ranked_ids,
                latency_ms=latency_ms,
            )
        )
    return BackendReport(name=runner.name, per_query=per_query)


def run(
    *,
    limit: int = 5,
    real_voyage: bool = False,
    corpus_path: Path | None = None,
    queries_path: Path | None = None,
) -> dict[str, Any]:
    corpus_path = corpus_path or REPO_ROOT / "examples" / "temporal-bird-study.json"
    queries_path = queries_path or Path(__file__).parent / "queries.json"

    with tempfile.TemporaryDirectory() as tmp:
        store = EventStore(Path(tmp))
        event_key_to_id = load_corpus(store, corpus_path)
        queries = load_queries(queries_path)
        corpus_topics = _corpus_topic_map(corpus_path)

        reports: list[BackendReport] = []

        # Sparse floor
        reports.append(
            _run_one(SparseVistaRunner(store), queries, event_key_to_id, limit)
        )
        # BM25 baseline
        reports.append(
            _run_one(BM25Runner(store), queries, event_key_to_id, limit)
        )
        # Voyage-mock and hybrid (both use the [vista] extra when available)
        has_vista = False
        try:
            import willow_substrate.backends.vista_voyage  # noqa: F401

            has_vista = True
        except ImportError:
            pass

        if has_vista:
            reports.append(
                _run_one(
                    VoyageMockRunner(
                        store,
                        event_key_to_id=event_key_to_id,
                        corpus_topics=corpus_topics,
                        queries=queries,
                    ),
                    queries,
                    event_key_to_id,
                    limit,
                )
            )

        # Hybrid: always available. Includes dense sub-backend when [vista]
        # is installed (with the mock embedder), else fuses sparse + BM25.
        reports.append(
            _run_one(
                HybridRunner(
                    store,
                    event_key_to_id=event_key_to_id,
                    corpus_topics=corpus_topics,
                    queries=queries,
                    include_dense=has_vista,
                ),
                queries,
                event_key_to_id,
                limit,
            )
        )

        # Real Voyage (only when explicitly requested)
        if real_voyage:
            api_key = os.environ.get("VOYAGE_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "--real-voyage requested but VOYAGE_API_KEY is not set"
                )
            reports.append(
                _run_one(
                    VoyageRealRunner(store, api_key=api_key),
                    queries,
                    event_key_to_id,
                    limit,
                )
            )

    return {
        "corpus": str(corpus_path),
        "queries_path": str(queries_path),
        "k_at_recall_summary": [3, 5],
        "limit": limit,
        "backends": [report.summary_dict() for report in reports],
        "per_query": [
            {
                "backend": report.name,
                "results": [
                    {
                        "query_id": q.query_id,
                        "paraphrase": q.paraphrase,
                        "recall_at_3": round(q.recall_at(3), 3),
                        "recall_at_5": round(q.recall_at(5), 3),
                        "reciprocal_rank": round(q.reciprocal_rank(), 3),
                        "latency_ms": round(q.latency_ms, 2),
                    }
                    for q in report.per_query
                ],
            }
            for report in reports
        ],
    }


def format_markdown_table(results: dict[str, Any]) -> str:
    lines = [
        "| Backend | Recall@3 | Recall@5 | MRR | Median latency (ms) |",
        "|---|---|---|---|---|",
    ]
    for row in results["backends"]:
        lines.append(
            f"| {row['name']} | {row['recall_at_3']:.3f} | "
            f"{row['recall_at_5']:.3f} | {row['mrr']:.3f} | "
            f"{row['median_latency_ms']:.2f} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Recall benchmark for willow-substrate backends.",
    )
    ap.add_argument("--limit", type=int, default=5, help="top-k depth")
    ap.add_argument(
        "--real-voyage",
        action="store_true",
        help="Also run the Voyage backend with the real API (requires VOYAGE_API_KEY)",
    )
    ap.add_argument("--json", action="store_true", help="emit JSON")
    args = ap.parse_args(argv)

    results = run(limit=args.limit, real_voyage=args.real_voyage)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"Corpus: {results['corpus']}")
        print(f"Queries: {results['queries_path']}")
        print(f"Top-k limit: {results['limit']}")
        print()
        print(format_markdown_table(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
