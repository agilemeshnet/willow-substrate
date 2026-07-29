"""Command-line interface for the reference Willow bundle."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

from willow.adapters.claude_code import capture_transcript
from willow.connections import find_connections
from willow.context import ContextBuilder
from willow.dreaming import dream
from willow.engrams import crystallize_retroactive_engrams
from willow.events import Event
from willow.facts import CHECK_OUTCOMES, EVIDENCE_KINDS, FactLedger
from willow.foveation import Foveator
from willow.hooks import handle_claude_hook
from willow.reflection import meditate, summarize_session
from willow.research import Citation, ResearchLedger
from willow.samples import evaluate_temporal_sample, load_temporal_sample
from willow.store import EventStore
from willow.vista import VistaBackend, VistaResult


def _event_dict(event: Event) -> dict[str, Any]:
    return {
        "seq": event.seq,
        "id": event.id,
        "timestamp": event.timestamp,
        "session_id": event.session_id,
        "actor": event.actor,
        "kind": event.kind,
        "content": event.content,
        "metadata": event.metadata,
        "supersedes": event.supersedes,
        "derived_from": list(event.derived_from),
        "hash": event.hash,
    }


def _print_event(event: Event, as_json: bool) -> None:
    if as_json:
        print(json.dumps(_event_dict(event), ensure_ascii=False))
        return
    print(
        f"{event.id}  {event.timestamp[:19]}  "
        f"{event.session_id}  {event.actor}/{event.kind}"
    )
    print(event.content)


def _vista_dict(result: VistaResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "query": result.query,
        "seed_event_ids": list(result.seed_event_ids),
        "reference_beams": [
            {
                "name": beam.name,
                "kind": beam.kind,
                "weight": round(beam.weight, 6),
                "source": beam.source,
            }
            for beam in result.reference_beams
        ],
        "vistas": [
            {
                "slug": match.vista.slug,
                "score": round(match.score, 6),
                "semantic_score": round(match.semantic_score, 6),
                "gaussian_score": round(match.gaussian_score, 6),
                "waypoint_score": round(match.waypoint_score, 6),
                "alpha": round(match.vista.alpha, 6),
                "sigma": round(match.vista.sigma, 6),
                "member_ids": list(match.vista.member_ids),
                "shared_waypoints": list(match.shared_waypoints),
            }
            for match in result.matches
        ],
        "evidence": [
            {
                "event": _event_dict(item.event),
                "score": round(item.score, 6),
                "vista_score": round(item.vista_score, 6),
                "wave_score": round(item.wave_score, 6),
                "vista_slugs": list(item.vista_slugs),
                "waypoints": list(item.waypoints),
                "channels": list(item.channels),
            }
            for item in result.evidence
        ],
        "trace": list(result.trace),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="willow",
        description="Model-independent continuity for long-running human-AI work.",
    )
    parser.add_argument(
        "--home",
        type=Path,
        default=None,
        help="Shared Willow directory (default: WILLOW_HOME or ~/.willow)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit structured JSON where supported",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="Initialize the shared local substrate")

    record = sub.add_parser("record", help="Append a message, action, or observation")
    record.add_argument("text")
    record.add_argument("--actor", default="user")
    record.add_argument("--kind", default="message")
    record.add_argument("--session", default="default")
    record.add_argument("--topic", action="append", default=[])

    correct = sub.add_parser("correct", help="Append a correction to an earlier event")
    correct.add_argument("event_id")
    correct.add_argument("text")
    correct.add_argument("--actor", default="user")
    correct.add_argument("--session", default="default")

    listing = sub.add_parser("list", help="List recent events")
    listing.add_argument("--limit", type=int, default=20)
    listing.add_argument("--session")
    listing.add_argument("--kind")
    listing.add_argument("--include-superseded", action="store_true")

    claim = sub.add_parser(
        "claim",
        help="Append a TTL-bearing factual claim with provenance",
    )
    claim.add_argument("text")
    claim.add_argument("--ttl-days", type=float, default=180)
    claim.add_argument("--domain", default="research")
    claim.add_argument("--source", default="user")
    claim.add_argument("--actor", default="user")
    claim.add_argument("--session", default="default")
    claim.add_argument("--topic", action="append", default=[])
    claim.add_argument("--shape", action="append", default=[])
    claim.add_argument("--derived-from", action="append", default=[])

    facts = sub.add_parser(
        "facts",
        help="List current facts or only facts due for verification",
    )
    facts.add_argument("--due", action="store_true")
    facts.add_argument("--domain")
    facts.add_argument("--limit", type=int, default=20)

    fact_check = sub.add_parser(
        "fact-check",
        help="Append an evidence-bearing fact verification attempt",
    )
    fact_check.add_argument("event_id")
    fact_check.add_argument(
        "--outcome",
        required=True,
        choices=sorted(CHECK_OUTCOMES),
    )
    fact_check.add_argument("--evidence", default="")
    fact_check.add_argument("--source", default="")
    fact_check.add_argument(
        "--evidence-kind",
        choices=sorted(EVIDENCE_KINDS),
        default="none",
    )
    fact_check.add_argument("--replacement")
    fact_check.add_argument("--retry-days", type=float, default=1)
    fact_check.add_argument("--actor", default="willow")
    fact_check.add_argument("--session", default="fact-ttl")

    research = sub.add_parser(
        "research",
        help="Queue and record provider-neutral research work",
    )
    research_sub = research.add_subparsers(
        dest="research_command",
        required=True,
    )
    research_queue = research_sub.add_parser(
        "queue",
        help="Append a safe-staged research commission",
    )
    research_queue.add_argument("query")
    research_queue.add_argument("--lane", default="auto")
    research_queue.add_argument("--topic", action="append", default=[])
    research_queue.add_argument("--actor", default="user")
    research_queue.add_argument("--session", default="research")
    research_queue.add_argument(
        "--approved",
        action="store_true",
        help="Mark provider spending/execution as already approved",
    )

    research_list = research_sub.add_parser(
        "list",
        help="List research commissions and their current state",
    )
    research_list.add_argument("--limit", type=int, default=20)

    research_complete = research_sub.add_parser(
        "complete",
        help="Append a cited result produced by any research adapter",
    )
    research_complete.add_argument("commission_id")
    research_complete.add_argument("writeup_file", type=Path)
    research_complete.add_argument("--summary", default="")
    research_complete.add_argument("--provider", required=True)
    research_complete.add_argument("--shape", action="append", default=[])
    research_complete.add_argument(
        "--citation",
        action="append",
        default=[],
        metavar="TITLE|LOCATION",
    )
    research_complete.add_argument("--actor", default="willow")
    research_complete.add_argument("--session", default="research")

    research_fail = research_sub.add_parser(
        "fail",
        help="Append an honest failed research attempt",
    )
    research_fail.add_argument("commission_id")
    research_fail.add_argument("error")
    research_fail.add_argument("--provider", required=True)
    research_fail.add_argument("--not-retryable", action="store_true")
    research_fail.add_argument("--actor", default="willow")
    research_fail.add_argument("--session", default="research")

    sample = sub.add_parser(
        "sample",
        help="Load or evaluate an inspectable temporal data sample",
    )
    sample_sub = sample.add_subparsers(
        dest="sample_command",
        required=True,
    )
    sample_load = sample_sub.add_parser(
        "load",
        help="Replay a temporal sample into the shared immutable store",
    )
    sample_load.add_argument("manifest", type=Path)
    sample_evaluate = sample_sub.add_parser(
        "evaluate",
        help="Evaluate a previously loaded sample against declared ground truth",
    )
    sample_evaluate.add_argument("manifest", type=Path)
    sample_run = sample_sub.add_parser(
        "run",
        help="Idempotently load and then evaluate a temporal sample",
    )
    sample_run.add_argument("manifest", type=Path)

    context = sub.add_parser("context", help="Build a bounded CWB-style context")
    context.add_argument("query", nargs="?", default="")
    context.add_argument("--tokens", type=int, default=2000)
    context.add_argument("--session")
    context.add_argument(
        "--mode",
        choices=["ambient", "recovery", "voluntary"],
        default="ambient",
    )
    context.add_argument("--without-foveation", action="store_true")
    context.add_argument("--without-vista", action="store_true")

    boot = sub.add_parser("boot", help="Build stable grounding for a new process")
    boot.add_argument("--agent", default="willow")
    boot.add_argument("--tokens", type=int, default=4000)
    boot.add_argument("--without-vista", action="store_true")

    breathe = sub.add_parser("breathe", help="Return a peripheral grounding signal")
    breathe.add_argument("query")

    foveate = sub.add_parser("foveate", help="Deliberately focus shared experience")
    foveate.add_argument("query")
    foveate.add_argument("--limit", type=int, default=12)
    foveate.add_argument("--without-vista", action="store_true")

    vista = sub.add_parser(
        "vista",
        help="Inspect the contextual surround and Wave from a query or event",
    )
    vista.add_argument("query", nargs="?", default="")
    vista.add_argument("--seed-event", action="append", default=[])
    vista.add_argument("--limit", type=int, default=8)
    vista.add_argument("--wave-hops", type=int, default=4)

    meditation = sub.add_parser(
        "meditate",
        help="Append a meditation derived from one session",
    )
    meditation.add_argument("--session", required=True)
    meditation.add_argument("--text")
    meditation.add_argument("--actor", default="willow")
    meditation.add_argument("--shape", action="append", default=[])

    connect = sub.add_parser(
        "connect",
        help="Find connections by words, idea-shape, and optional Vista/Wave",
    )
    connect.add_argument("query", nargs="?", default="")
    connect.add_argument("--from-event")
    connect.add_argument("--shape", action="append", default=[])
    connect.add_argument("--limit", type=int, default=10)
    connect.add_argument(
        "--with-vista",
        action="store_true",
        help="Add relational Vista/Wave evidence as separate channels",
    )

    summation = sub.add_parser(
        "summarize",
        help="Append or advance a session summation",
    )
    summation.add_argument("--session", required=True)
    summation.add_argument("--actor", default="willow")

    dreaming = sub.add_parser(
        "dream",
        help="Append associative proposals across active experience",
    )
    dreaming.add_argument("query", nargs="?", default="")
    dreaming.add_argument("--limit", type=int, default=3)
    dreaming.add_argument("--minimum-shared-terms", type=int, default=2)
    dreaming.add_argument("--actor", default="willow")

    engram = sub.add_parser(
        "engram",
        help="Crystallize experience whose importance emerged later",
    )
    engram.add_argument("--limit", type=int, default=3)
    engram.add_argument("--minimum-later-reflections", type=int, default=1)

    capture = sub.add_parser(
        "capture",
        help="Idempotently capture a provider transcript",
    )
    capture.add_argument("transcript", type=Path)
    capture.add_argument("--provider", choices=["claude-code"], default="claude-code")
    capture.add_argument("--session", required=True)
    capture.add_argument("--user-actor", default="user")
    capture.add_argument("--assistant-actor", default="willow")

    hook = sub.add_parser(
        "hook",
        help="Handle a provider lifecycle event from JSON on stdin",
    )
    hook.add_argument(
        "phase",
        choices=["start", "prompt", "stop", "compact", "end"],
    )
    hook.add_argument("--provider", choices=["claude-code"], default="claude-code")
    hook.add_argument("--tokens", type=int, default=1600)
    hook.add_argument("--context-limit", type=int, default=0)
    hook.add_argument("--user-actor", default="user")
    hook.add_argument("--assistant-actor", default="willow")
    hook.add_argument("--engram-ttl", type=int, default=1800)
    hook.add_argument(
        "--strict",
        action="store_true",
        help="Return a failure instead of silently allowing the host turn",
    )

    sub.add_parser("verify", help="Verify the immutable event hash chain")
    sub.add_parser("status", help="Show substrate location and event count")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        store = EventStore(args.home)
    except (OSError, sqlite3.Error) as exc:
        if args.command == "hook" and not args.strict:
            if args.json:
                print(json.dumps({
                    "phase": args.phase,
                    "error": str(exc),
                    "output": "",
                }))
            return 0
        if args.json:
            print(json.dumps({"error": str(exc)}))
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 2

    try:
        if args.command == "init":
            result = {
                "home": str(store.home),
                "database": str(store.db_path),
                "events": store.count(),
            }
            if args.json:
                print(json.dumps(result))
            else:
                print(f"Willow initialized at {store.home}")
                print(f"Shared event store: {store.db_path}")
            return 0

        if args.command == "record":
            metadata = {"topics": args.topic} if args.topic else {}
            event = store.append(
                args.text,
                actor=args.actor,
                kind=args.kind,
                session_id=args.session,
                metadata=metadata,
            )
            _print_event(event, args.json)
            return 0

        if args.command == "correct":
            event = store.correct(
                args.event_id,
                args.text,
                actor=args.actor,
                session_id=args.session,
            )
            _print_event(event, args.json)
            return 0

        if args.command == "list":
            events = store.events(
                limit=args.limit,
                session_id=args.session,
                kind=args.kind,
                active_only=not args.include_superseded,
            )
            if args.json:
                print(json.dumps([_event_dict(event) for event in events]))
            else:
                for event in events:
                    marker = f" -> supersedes {event.supersedes}" if event.supersedes else ""
                    print(
                        f"{event.short_id}  {event.timestamp[:19]}  "
                        f"{event.session_id:<16} {event.actor}/{event.kind}{marker}"
                    )
                    print(f"  {' '.join(event.content.split())[:180]}")
            return 0

        if args.command == "claim":
            event = FactLedger(store).add_claim(
                args.text,
                ttl_days=args.ttl_days,
                domain=args.domain,
                source=args.source,
                actor=args.actor,
                session_id=args.session,
                topics=args.topic,
                shapes=args.shape,
                derived_from=args.derived_from,
            )
            _print_event(event, args.json)
            return 0

        if args.command == "facts":
            ledger = FactLedger(store)
            if args.due:
                states = ledger.due(
                    domain=args.domain,
                    limit=args.limit,
                )
            else:
                states = ledger.states()
                if args.domain:
                    states = [
                        state
                        for state in states
                        if state.claim.metadata.get("domain") == args.domain
                    ]
                states = states[: max(0, args.limit)]
            rows = [
                {
                    "claim": _event_dict(state.claim),
                    "status": state.status,
                    "next_check_at": state.next_check_at,
                    "last_verified_at": state.last_verified_at,
                    "check_count": state.check_count,
                    "last_check_id": (
                        state.last_check.id if state.last_check else None
                    ),
                }
                for state in states
            ]
            if args.json:
                print(json.dumps(rows, ensure_ascii=False))
            elif not rows:
                print("No facts matched.")
            else:
                for row in rows:
                    claim_event = row["claim"]
                    print(
                        f"{claim_event['id'][:16]}  {row['status']:<11} "
                        f"next={row['next_check_at'] or 'never'}"
                    )
                    print(f"  {claim_event['content']}")
                    print(
                        "  "
                        f"domain={claim_event['metadata'].get('domain')} "
                        f"source={claim_event['metadata'].get('source')} "
                        f"checks={row['check_count']}"
                    )
            return 0

        if args.command == "fact-check":
            result = FactLedger(store).record_check(
                args.event_id,
                outcome=args.outcome,
                evidence=args.evidence,
                source=args.source,
                evidence_kind=args.evidence_kind,
                replacement=args.replacement,
                actor=args.actor,
                session_id=args.session,
                retry_days=args.retry_days,
            )
            if args.json:
                print(json.dumps({
                    "check": _event_dict(result.check),
                    "replacement": (
                        _event_dict(result.replacement)
                        if result.replacement
                        else None
                    ),
                }, ensure_ascii=False))
            else:
                if result.replacement:
                    print("Replacement claim:")
                    _print_event(result.replacement, False)
                print("Fact check:")
                _print_event(result.check, False)
            return 0

        if args.command == "research":
            ledger = ResearchLedger(store)
            if args.research_command == "queue":
                event = ledger.commission(
                    args.query,
                    actor=args.actor,
                    session_id=args.session,
                    lane=args.lane,
                    topics=args.topic,
                    approval_required=not args.approved,
                )
                _print_event(event, args.json)
                return 0

            if args.research_command == "list":
                states = ledger.states()[-max(0, args.limit):]
                rows = [
                    {
                        "commission": _event_dict(state.commission),
                        "status": state.status,
                        "result_id": (
                            state.result.id if state.result else None
                        ),
                        "failure_id": (
                            state.failure.id if state.failure else None
                        ),
                    }
                    for state in states
                ]
                if args.json:
                    print(json.dumps(rows, ensure_ascii=False))
                elif not rows:
                    print("No research commissions.")
                else:
                    for row in reversed(rows):
                        event = row["commission"]
                        print(
                            f"{event['id'][:16]}  {row['status']:<18} "
                            f"lane={event['metadata'].get('lane')}"
                        )
                        print(f"  {' '.join(event['content'].split())[:180]}")
                return 0

            if args.research_command == "complete":
                citation_values = []
                for value in args.citation:
                    title, separator, location = value.partition("|")
                    if not separator:
                        location = title
                    citation_values.append(
                        Citation(
                            title=title.strip() or location.strip(),
                            location=location.strip(),
                        )
                    )
                event = ledger.complete(
                    args.commission_id,
                    summary=args.summary,
                    writeup=args.writeup_file.read_text(encoding="utf-8"),
                    citations=citation_values,
                    shapes=args.shape,
                    provider=args.provider,
                    actor=args.actor,
                    session_id=args.session,
                )
                _print_event(event, args.json)
                return 0

            if args.research_command == "fail":
                event = ledger.fail(
                    args.commission_id,
                    error=args.error,
                    provider=args.provider,
                    retryable=not args.not_retryable,
                    actor=args.actor,
                    session_id=args.session,
                )
                _print_event(event, args.json)
                return 0

        if args.command == "sample":
            load_report = None
            if args.sample_command in {"load", "run"}:
                load_report = load_temporal_sample(store, args.manifest)
                if args.sample_command == "load":
                    result = {
                        "sample_id": load_report.sample_id,
                        "created": load_report.created,
                        "reused": load_report.reused,
                        "event_ids": {
                            key: event.id
                            for key, event in load_report.events.items()
                        },
                    }
                    if args.json:
                        print(json.dumps(result, ensure_ascii=False))
                    else:
                        print(
                            f"Loaded {load_report.sample_id}: "
                            f"{load_report.created} created, "
                            f"{load_report.reused} already present."
                        )
                    return 0

            evaluation = evaluate_temporal_sample(store, args.manifest)
            result = {
                "sample_id": evaluation.sample_id,
                "passed": evaluation.passed,
                "loaded": (
                    {
                        "created": load_report.created,
                        "reused": load_report.reused,
                    }
                    if load_report is not None
                    else None
                ),
                "checks": [
                    {
                        "name": check.name,
                        "passed": check.passed,
                        "detail": check.detail,
                    }
                    for check in evaluation.checks
                ],
            }
            if args.json:
                print(json.dumps(result, ensure_ascii=False))
            else:
                print(
                    f"Temporal sample {evaluation.sample_id}: "
                    f"{'PASS' if evaluation.passed else 'FAIL'}"
                )
                for check in evaluation.checks:
                    marker = "✓" if check.passed else "✗"
                    print(f"  {marker} {check.name}: {check.detail}")
            return 0 if evaluation.passed else 1

        if args.command == "context":
            packet = ContextBuilder(store).build(
                args.query,
                token_budget=args.tokens,
                session_id=args.session,
                mode=args.mode,
                use_foveation=not args.without_foveation,
                use_vista=not args.without_vista,
            )
            if args.json:
                print(json.dumps({
                    "query": packet.query,
                    "mode": packet.mode,
                    "estimated_tokens": packet.estimated_tokens,
                    "event_ids": packet.event_ids,
                    "context": packet.markdown,
                    "vista": _vista_dict(packet.vista),
                }))
            else:
                print(packet.markdown)
            return 0

        if args.command == "boot":
            packet = ContextBuilder(store).boot(
                agent=args.agent,
                token_budget=args.tokens,
                use_vista=not args.without_vista,
            )
            if args.json:
                print(json.dumps({
                    "mode": packet.mode,
                    "estimated_tokens": packet.estimated_tokens,
                    "event_ids": packet.event_ids,
                    "context": packet.markdown,
                    "vista": _vista_dict(packet.vista),
                }))
            else:
                print(packet.markdown)
            return 0

        if args.command == "breathe":
            print(ContextBuilder(store).breathe(args.query))
            return 0

        if args.command == "foveate":
            result = Foveator(store).foveate(
                args.query,
                mode="voluntary",
                event_limit=args.limit,
            )
            vista_result = (
                VistaBackend(store).query(
                    args.query,
                    seed_event_ids=result.event_ids[:5],
                    limit=args.limit,
                )
                if not args.without_vista
                else None
            )
            if args.json:
                print(json.dumps({
                    "query": result.query,
                    "mode": result.mode,
                    "event_ids": result.event_ids,
                    "trace": [
                        {
                            "name": item.name,
                            "candidates_examined": item.candidates_examined,
                            "selected": item.selected,
                            "rationale": item.rationale,
                        }
                        for item in result.trace
                    ],
                    "hits": [
                        {
                            "event": _event_dict(hit.event),
                            "score": hit.score,
                            "source": hit.source,
                        }
                        for hit in result.hits
                    ],
                    "vista": _vista_dict(vista_result),
                }))
            else:
                print(result.to_markdown())
                if vista_result is not None:
                    print("\n\n" + vista_result.to_markdown())
            return 0

        if args.command == "vista":
            result = VistaBackend(store).query(
                args.query,
                seed_event_ids=args.seed_event,
                limit=args.limit,
                wave_hops=args.wave_hops,
            )
            if args.json:
                print(json.dumps(_vista_dict(result), ensure_ascii=False))
            else:
                print(result.to_markdown())
            return 0

        if args.command == "meditate":
            event = meditate(
                store,
                args.session,
                text=args.text,
                actor=args.actor,
                shapes=args.shape,
            )
            _print_event(event, args.json)
            return 0

        if args.command == "connect":
            candidates = find_connections(
                store,
                seed_event_id=args.from_event,
                query=args.query,
                shapes=args.shape,
                limit=args.limit,
                include_vista=args.with_vista,
            )
            rows = [
                {
                    "event": _event_dict(candidate.event),
                    "score": round(candidate.score, 4),
                    "lexical_score": round(candidate.lexical_score, 4),
                    "shape_score": round(candidate.shape_score, 4),
                    "vista_score": round(candidate.vista_score, 4),
                    "wave_score": round(candidate.wave_score, 4),
                    "shared_terms": list(candidate.shared_terms),
                    "shared_shapes": list(candidate.shared_shapes),
                    "shared_dimensions": list(
                        candidate.shared_dimensions
                    ),
                    "relational_waypoints": list(
                        candidate.relational_waypoints
                    ),
                    "vista_slugs": list(candidate.vista_slugs),
                    "channels": list(candidate.channels),
                }
                for candidate in candidates
            ]
            if args.json:
                print(json.dumps(rows, ensure_ascii=False))
            elif not rows:
                print("No active research connection met the threshold.")
            else:
                for row in rows:
                    event = row["event"]
                    print(
                        f"{event['id'][:16]}  score={row['score']:.3f}  "
                        f"via={'+'.join(row['channels'])}  "
                        f"{event['actor']}/{event['kind']}"
                    )
                    print(f"  {' '.join(event['content'].split())[:220]}")
                    if row["shared_shapes"]:
                        print(
                            "  shapes: "
                            + ", ".join(row["shared_shapes"])
                        )
                    elif row["shared_dimensions"]:
                        print(
                            "  shape dimensions: "
                            + ", ".join(row["shared_dimensions"])
                        )
                    if row["shared_terms"]:
                        print(
                            "  words: "
                            + ", ".join(row["shared_terms"])
                        )
                    if row["relational_waypoints"]:
                        print(
                            "  waypoints: "
                            + ", ".join(row["relational_waypoints"])
                        )
            return 0

        if args.command == "summarize":
            event, created = summarize_session(
                store,
                args.session,
                actor=args.actor,
            )
            if args.json:
                result = _event_dict(event)
                result["created"] = created
                print(json.dumps(result, ensure_ascii=False))
            else:
                _print_event(event, False)
                if not created:
                    print("(already current)")
            return 0

        if args.command == "dream":
            events = dream(
                store,
                query=args.query,
                limit=args.limit,
                minimum_shared_terms=args.minimum_shared_terms,
                actor=args.actor,
            )
            if args.json:
                print(json.dumps(
                    [_event_dict(event) for event in events],
                    ensure_ascii=False,
                ))
            elif not events:
                print("No new active cross-session connection met the threshold.")
            else:
                for event in events:
                    _print_event(event, False)
            return 0

        if args.command == "engram":
            events = crystallize_retroactive_engrams(
                store,
                limit=args.limit,
                minimum_later_reflections=args.minimum_later_reflections,
            )
            if args.json:
                print(json.dumps(
                    [_event_dict(event) for event in events],
                    ensure_ascii=False,
                ))
            elif not events:
                print("No new retroactive importance was discovered.")
            else:
                for event in events:
                    _print_event(event, False)
            return 0

        if args.command == "capture":
            report = capture_transcript(
                store,
                args.transcript,
                session_id=args.session,
                user_actor=args.user_actor,
                assistant_actor=args.assistant_actor,
            )
            result = {
                "provider": args.provider,
                "session_id": report.session_id,
                "created": report.created,
                "skipped": report.skipped,
                "event_ids": [event.id for event in report.created_events],
                "recent_input_tokens": report.recent_input_tokens,
            }
            if args.json:
                print(json.dumps(result))
            else:
                print(
                    f"Captured {report.created} event(s); "
                    f"skipped {report.skipped} existing event(s)."
                )
            return 0

        if args.command == "hook":
            try:
                payload = json.loads(sys.stdin.read() or "{}")
            except json.JSONDecodeError:
                payload = {}
            if not isinstance(payload, dict):
                payload = {}
            try:
                result = handle_claude_hook(
                    store,
                    payload,
                    phase=args.phase,
                    token_budget=args.tokens,
                    context_limit=args.context_limit,
                    user_actor=args.user_actor,
                    assistant_actor=args.assistant_actor,
                    engram_ttl_seconds=args.engram_ttl,
                )
            except (KeyError, ValueError, OSError, sqlite3.Error) as exc:
                if args.strict:
                    raise
                if args.json:
                    print(json.dumps({
                        "phase": args.phase,
                        "error": str(exc),
                        "output": "",
                    }))
                return 0
            if args.json:
                print(json.dumps({
                    "phase": result.phase,
                    "captured": result.captured,
                    "skipped": result.skipped,
                    "event_ids": result.event_ids,
                    "output": result.output,
                }))
            elif result.output:
                print(result.output)
            return 0

        if args.command == "verify":
            valid, count, error = store.verify()
            result = {"valid": valid, "events": count, "error": error}
            if args.json:
                print(json.dumps(result))
            elif valid:
                print(f"OK: {count} events; global hash chain intact")
            else:
                print(f"FAIL: {error}", file=sys.stderr)
            return 0 if valid else 1

        if args.command == "status":
            valid, count, error = store.verify()
            result = {
                "home": str(store.home),
                "database": str(store.db_path),
                "events": count,
                "chain_valid": valid,
                "chain_error": error,
            }
            if args.json:
                print(json.dumps(result))
            else:
                print(f"Home: {store.home}")
                print(f"Database: {store.db_path}")
                print(f"Events: {count}")
                print(f"Chain: {'valid' if valid else error}")
            return 0

    except (KeyError, ValueError, OSError, sqlite3.Error) as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}))
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 2

    parser.error(f"unknown command: {args.command}")
    return 2
if __name__ == "__main__":
    raise SystemExit(main())
