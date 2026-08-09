"""Score one or more LoCoMo run manifests.

Reads the JSON produced by run.py and reports evidence recall@k, MRR,
nDCG@10, in-context recall (when the config recorded final-context ids),
context precision, and bootstrap confidence intervals BY CONVERSATION
(LoCoMo only has ten, so per-question bootstrap would exaggerate
certainty; conversation-level is the reviewer's recommendation).

Emits both a compact per-config table and a JSON summary suitable for
further slicing.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class QueryScore:
    conversation_id: str
    category: str
    recall_at: dict[int, float]
    reciprocal_rank: float
    ndcg_at_10: float
    in_context_recall: dict[int, float]
    context_precision: dict[int, float]


def _recall_at(retrieved: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return 0.0
    top = set(retrieved[:k])
    hit = sum(1 for g in gold if g in top)
    return hit / len(gold)


def _reciprocal_rank(retrieved: list[str], gold: list[str]) -> float:
    gold_set = set(gold)
    for i, r in enumerate(retrieved, start=1):
        if r in gold_set:
            return 1.0 / i
    return 0.0


def _dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def _ndcg_at(retrieved: list[str], gold: list[str], k: int) -> float:
    gold_set = set(gold)
    gains = [1.0 if r in gold_set else 0.0 for r in retrieved[:k]]
    dcg = _dcg(gains)
    ideal = sorted(gains, reverse=True)
    idcg = _dcg(ideal)
    return dcg / idcg if idcg > 0 else 0.0


def _bootstrap_ci(
    samples: list[float],
    *,
    ci: float = 0.95,
    n_boot: int = 1000,
    seed: int = 20260809,
) -> tuple[float, float]:
    """Bootstrap CI over conversation-level means."""
    if not samples:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means: list[float] = []
    n = len(samples)
    for _ in range(n_boot):
        resample = [samples[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    lo_idx = int(((1 - ci) / 2) * n_boot)
    hi_idx = int((1 - (1 - ci) / 2) * n_boot) - 1
    return means[lo_idx], means[hi_idx]


def score_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Compute per-config aggregate metrics + bootstrap CIs by conversation."""
    per_query: list[QueryScore] = []
    for row in manifest["rows"]:
        retrieved = list(row["retrieved_event_ids"])
        gold = list(row["gold_event_ids"])
        per_budget_context: dict[int, tuple[float, float]] = {}
        for entry in row.get("per_budget", []):
            budget = int(entry["context_tokens"])
            final_ids = list(entry.get("final_context_ids", []))
            recall = _recall_at(final_ids, gold, k=len(final_ids) or 1)
            precision = (
                sum(1 for i in final_ids if i in set(gold)) / len(final_ids)
                if final_ids
                else 0.0
            )
            per_budget_context[budget] = (recall, precision)
        per_query.append(
            QueryScore(
                conversation_id=row["conversation_id"],
                category=row["category"],
                recall_at={
                    k: _recall_at(retrieved, gold, k) for k in (5, 10, 20)
                },
                reciprocal_rank=_reciprocal_rank(retrieved, gold),
                ndcg_at_10=_ndcg_at(retrieved, gold, 10),
                in_context_recall={
                    k: v[0] for k, v in per_budget_context.items()
                },
                context_precision={
                    k: v[1] for k, v in per_budget_context.items()
                },
            )
        )

    # Aggregate by conversation for bootstrap stability.
    by_conv: dict[str, list[QueryScore]] = defaultdict(list)
    for q in per_query:
        by_conv[q.conversation_id].append(q)

    def _conv_means(getter: Callable[[QueryScore], float]) -> list[float]:
        return [
            sum(getter(q) for q in queries) / len(queries)
            for queries in by_conv.values()
        ]

    def _bucket(getter: Callable[[QueryScore], float]) -> dict[str, Any]:
        conv_means = _conv_means(getter)
        overall_mean = sum(conv_means) / len(conv_means) if conv_means else 0.0
        lo, hi = _bootstrap_ci(conv_means)
        return {
            "mean": round(overall_mean, 4),
            "ci_low_95": round(lo, 4),
            "ci_high_95": round(hi, 4),
            "n_conversations": len(conv_means),
        }

    context_budgets = sorted(
        {b for q in per_query for b in q.in_context_recall}
    )

    return {
        "config_name": manifest["config"]["name"],
        "backend": manifest["config"]["backend"],
        "willow_commit": manifest.get("willow_commit", "unknown"),
        "n_conversations": len(by_conv),
        "n_questions": len(per_query),
        "wall_seconds": manifest.get("wall_seconds"),
        "evidence_recall_at_5": _bucket(lambda q: q.recall_at.get(5, 0.0)),
        "evidence_recall_at_10": _bucket(lambda q: q.recall_at.get(10, 0.0)),
        "evidence_recall_at_20": _bucket(lambda q: q.recall_at.get(20, 0.0)),
        "mrr": _bucket(lambda q: q.reciprocal_rank),
        "ndcg_at_10": _bucket(lambda q: q.ndcg_at_10),
        "in_context_recall_by_budget": {
            str(b): _bucket(lambda q, bud=b: q.in_context_recall.get(bud, 0.0))
            for b in context_budgets
        },
        "context_precision_by_budget": {
            str(b): _bucket(lambda q, bud=b: q.context_precision.get(bud, 0.0))
            for b in context_budgets
        },
    }


def _format_table(summaries: list[dict[str, Any]]) -> str:
    header = (
        "| Config | Backend | Recall@5 | Recall@10 | MRR | nDCG@10 | Wall (s) |"
    )
    sep = "|" + "|".join(["---"] * 7) + "|"
    lines = [header, sep]
    for s in summaries:
        lines.append(
            f"| {s['config_name']} | {s['backend']} | "
            f"{s['evidence_recall_at_5']['mean']:.3f} "
            f"({s['evidence_recall_at_5']['ci_low_95']:.3f} to {s['evidence_recall_at_5']['ci_high_95']:.3f}) | "
            f"{s['evidence_recall_at_10']['mean']:.3f} "
            f"({s['evidence_recall_at_10']['ci_low_95']:.3f} to {s['evidence_recall_at_10']['ci_high_95']:.3f}) | "
            f"{s['mrr']['mean']:.3f} | "
            f"{s['ndcg_at_10']['mean']:.3f} | "
            f"{s['wall_seconds'] if s['wall_seconds'] is not None else '-'} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifests", nargs="+", type=Path)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    summaries: list[dict[str, Any]] = []
    for path in args.manifests:
        with path.open() as fh:
            manifest = json.load(fh)
        summaries.append(score_manifest(manifest))

    if args.json:
        print(json.dumps({"summaries": summaries}, indent=2))
    else:
        print(_format_table(summaries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
