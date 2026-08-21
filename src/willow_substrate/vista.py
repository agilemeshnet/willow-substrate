"""Vista contextual-surround retrieval over Willow's immutable event ledger.

Vista is a derived, rebuildable projection.  It never mutates source events and
does not treat timestamps as coordinates.  The dependency-free reference
backend uses a sparse lexical-semantic field so that a clean Willow install can
exercise the complete contract:

* active events become normalised points in a feature space;
* softly bounded Gaussian neighbourhoods form Vistas;
* explicit shapes, topics, entities, links, and derivations form waypoints;
* foveated events become reference beams;
* Wave spreads activation across the event <-> waypoint lattice.

The richer ``agilemeshnet/vista`` embedding implementation can replace this
backend without changing the evidence/result contract defined here.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Mapping, Protocol

from willow_substrate.events import Event, EventHit
from willow_substrate.readout import (
    Reranker,
    WaveFeatures,
    build_wave_features,
)
from willow_substrate.store import EventStore


STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "and",
    "are",
    "because",
    "been",
    "before",
    "being",
    "but",
    "can",
    "could",
    "did",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "into",
    "its",
    "just",
    "more",
    "not",
    "our",
    "should",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "will",
    "with",
    "would",
    "you",
    "your",
}

WAYPOINT_METADATA = {
    "topic": "topic",
    "topics": "topic",
    "project": "project",
    "projects": "project",
    "person": "person",
    "people": "person",
    "entity": "entity",
    "entities": "entity",
    "place": "place",
    "places": "place",
    "location": "place",
    "locations": "place",
    "idea_shape": "shape",
    "shapes": "shape",
}

GENERIC_ACTORS = {
    "agent",
    "assistant",
    "model",
    "system",
    "user",
    "willow",
}


SparseVector = dict[str, float]


def _normalise(value: str) -> str:
    lowered = re.sub(r"\s+", "-", value.strip().lower())
    return re.sub(r"[^a-z0-9_:+.-]", "", lowered)


def _words(text: str) -> list[str]:
    return [
        word.lower()
        for word in re.findall(r"[\w'-]{3,}", text, flags=re.UNICODE)
        if word.lower() not in STOPWORDS
    ]


def _metadata_values(value: Any) -> Iterable[str]:
    if value is None:
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if isinstance(nested, (Mapping, list, tuple, set)):
                yield from _metadata_values(nested)
            else:
                yield f"{key}:{nested}"
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _metadata_values(item)
        return
    yield str(value)


def _proper_phrases(text: str) -> set[str]:
    phrases = set()
    for raw in re.findall(
        r"\b[A-Z][\w'-]+(?:\s+[A-Z][\w'-]+){0,2}\b",
        text,
        flags=re.UNICODE,
    ):
        normalised = _normalise(raw)
        if len(normalised) >= 4 and normalised not in {"the", "this", "that"}:
            phrases.add(normalised)
    return phrases


def _wikilinks(text: str) -> set[str]:
    return {
        _normalise(value)
        for value in re.findall(r"\[\[([^\]]+)\]\]", text)
        if _normalise(value)
    }


def _dot(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(key, 0.0) for key, value in left.items())


def _normalise_vector(vector: Mapping[str, float]) -> SparseVector:
    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm <= 1e-12:
        return {}
    return {key: value / norm for key, value in vector.items() if value}


def _mean_vector(vectors: Iterable[Mapping[str, float]]) -> SparseVector:
    values = list(vectors)
    if not values:
        return {}
    total: dict[str, float] = defaultdict(float)
    for vector in values:
        for key, value in vector.items():
            total[key] += value
    return _normalise_vector(
        {key: value / len(values) for key, value in total.items()}
    )


def _parse_timestamp(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo is not None else None


@dataclass(frozen=True)
class Waypoint:
    """One relational landmark in the Vista lattice."""

    name: str
    kind: str
    mass: float
    degree: int
    member_ids: tuple[str, ...]


@dataclass(frozen=True)
class Vista:
    """One soft semantic neighbourhood."""

    slug: str
    mu: Mapping[str, float]
    sigma: float
    alpha: float
    member_ids: tuple[str, ...]
    time_start: str
    time_end: str
    waypoints: tuple[str, ...]


@dataclass(frozen=True)
class ReferenceBeam:
    """A waypoint illuminated by focused attention or the current query."""

    name: str
    kind: str
    weight: float
    source: str


@dataclass(frozen=True)
class VistaMatch:
    """A Vista returned at the intersection of the current beams."""

    vista: Vista
    score: float
    semantic_score: float
    gaussian_score: float
    waypoint_score: float
    shared_waypoints: tuple[str, ...]


@dataclass(frozen=True)
class VistaEvidence:
    """One provenance-bearing event surfaced by Vista and/or Wave."""

    event: Event
    score: float
    vista_score: float
    wave_score: float
    vista_slugs: tuple[str, ...]
    waypoints: tuple[str, ...]
    channels: tuple[str, ...]
    wave_features: WaveFeatures | None = None

    def as_event_hit(self) -> EventHit:
        return EventHit(
            event=self.event,
            score=self.score,
            source="-".join(self.channels),
        )


@dataclass(frozen=True)
class VistaResult:
    """The contextual surround and its inspectable decision trace."""

    query: str
    seed_event_ids: tuple[str, ...]
    reference_beams: tuple[ReferenceBeam, ...]
    matches: tuple[VistaMatch, ...]
    evidence: tuple[VistaEvidence, ...]
    trace: tuple[str, ...]

    @property
    def event_ids(self) -> tuple[str, ...]:
        return tuple(item.event.id for item in self.evidence)

    @property
    def hits(self) -> tuple[EventHit, ...]:
        return tuple(item.as_event_hit() for item in self.evidence)

    def evidence_for(self, event_id: str) -> VistaEvidence | None:
        return next(
            (item for item in self.evidence if item.event.id == event_id),
            None,
        )

    def to_markdown(self) -> str:
        lines = [
            f"# Vista contextual surround: {self.query or 'current attention'}",
            "",
        ]
        if self.reference_beams:
            lines.append("## Reference beams")
            for beam in self.reference_beams:
                lines.append(
                    f"- `{beam.name}` ({beam.kind}, {beam.source}, "
                    f"weight {beam.weight:.3f})"
                )
            lines.append("")
        lines.append("## Intersection Vistas")
        if not self.matches:
            lines.append("- No Vista met the current beam.")
        for match in self.matches:
            lines.append(
                f"- `{match.vista.slug}` score={match.score:.3f} "
                f"semantic={match.semantic_score:.3f} "
                f"gaussian={match.gaussian_score:.3f} "
                f"waypoint={match.waypoint_score:.3f} "
                f"members={len(match.vista.member_ids)}"
            )
        lines.extend(["", "## Contextual evidence"])
        if not self.evidence:
            lines.append("- No additional active evidence was surfaced.")
        for item in self.evidence:
            snippet = " ".join(item.event.content.split())
            if len(snippet) > 180:
                snippet = snippet[:177] + "..."
            lines.append(
                f"- [{item.event.short_id}] score={item.score:.3f} "
                f"via={','.join(item.channels)} "
                f"waypoints={','.join(item.waypoints[:3]) or 'semantic'}: "
                f"{snippet}"
            )
        lines.extend(["", "## Decision trace"])
        lines.extend(f"- {item}" for item in self.trace)
        return "\n".join(lines)


@dataclass
class VistaProjection:
    """A derived in-memory field over one active-event snapshot."""

    events: dict[str, Event]
    vectors: dict[str, SparseVector]
    idf: dict[str, float]
    event_waypoints: dict[str, dict[str, float]]
    waypoints: dict[str, Waypoint]
    vistas: tuple[Vista, ...]
    truncated: bool = False


class RelationalBackend(Protocol):
    """Replaceable evidence contract for relational/contextual retrieval."""

    def query(
        self,
        query: str,
        *,
        seed_event_ids: Iterable[str] = (),
        limit: int = 8,
        wave_hops: int = 4,
    ) -> VistaResult:
        ...


class VistaBackend:
    """Dependency-free reference implementation of Vista and Wave."""

    def __init__(
        self,
        store: EventStore,
        *,
        max_events: int = 2000,
        cluster_threshold: float = 0.30,
        half_life_days: float = 60.0,
        max_beams: int = 8,
        wave_damping: float = 0.5,
    ):
        if max_events < 1:
            raise ValueError("max_events must be positive")
        if not 0.0 < cluster_threshold < 1.0:
            raise ValueError("cluster_threshold must be between zero and one")
        if half_life_days <= 0:
            raise ValueError("half_life_days must be positive")
        if not 0.0 <= wave_damping <= 1.0:
            raise ValueError("wave_damping must be between zero and one")
        self.store = store
        self.max_events = max_events
        self.cluster_threshold = cluster_threshold
        self.half_life_days = half_life_days
        self.max_beams = max_beams
        self.wave_damping = wave_damping
        self._snapshot_key: tuple[str, ...] = ()
        self._projection: VistaProjection | None = None

    def project(self) -> VistaProjection:
        events = self.store.events(
            limit=self.max_events + 1,
            active_only=True,
            include_expired=False,
            ascending=False,
        )
        truncated = len(events) > self.max_events
        if truncated:
            events = events[: self.max_events]
        events.reverse()
        snapshot_key = (
            f"truncated:{truncated}",
            *(event.id for event in events),
        )
        if self._projection is not None and snapshot_key == self._snapshot_key:
            return self._projection

        event_map = {event.id: event for event in events}
        raw_features = {
            event.id: self._event_features(event)
            for event in events
        }
        document_frequency: Counter[str] = Counter()
        for features in raw_features.values():
            document_frequency.update(features)
        count = max(1, len(events))
        idf = {
            key: math.log((1.0 + count) / (1.0 + frequency)) + 1.0
            for key, frequency in document_frequency.items()
        }
        vectors = {
            event_id: self._tfidf_vector(features, idf)
            for event_id, features in raw_features.items()
        }

        event_waypoints = self._extract_waypoints(events)
        decays = self._recency_decays(events)
        waypoint_members: dict[str, dict[str, float]] = defaultdict(dict)
        for event_id, edges in event_waypoints.items():
            for name, prominence in edges.items():
                waypoint_members[name][event_id] = prominence
        waypoints: dict[str, Waypoint] = {}
        for name, members in waypoint_members.items():
            kind = name.partition(":")[0]
            mass = sum(
                prominence * decays.get(event_id, 1.0)
                for event_id, prominence in members.items()
            )
            waypoints[name] = Waypoint(
                name=name,
                kind=kind,
                mass=max(0.01, mass),
                degree=len(members),
                member_ids=tuple(sorted(members)),
            )

        components = self._cluster(vectors)
        vistas: list[Vista] = []
        for member_ids in components:
            member_vectors = [vectors[event_id] for event_id in member_ids]
            mu = _mean_vector(member_vectors)
            distances = [max(0.0, 1.0 - _dot(mu, vector)) for vector in member_vectors]
            rms = math.sqrt(
                sum(distance * distance for distance in distances)
                / max(1, len(distances))
            )
            sigma_floor = 0.30 if len(member_ids) == 1 else 0.08
            sigma = max(sigma_floor, rms)
            alpha = max(
                0.05,
                math.log1p(
                    sum(decays.get(event_id, 1.0) for event_id in member_ids)
                ),
            )
            timestamps = sorted(event_map[event_id].timestamp for event_id in member_ids)
            inherited = sorted(
                {
                    name
                    for event_id in member_ids
                    for name in event_waypoints.get(event_id, {})
                }
            )
            identity = "\0".join(sorted(member_ids)).encode("utf-8")
            slug = "vista-" + hashlib.sha256(identity).hexdigest()[:12]
            vistas.append(
                Vista(
                    slug=slug,
                    mu=mu,
                    sigma=sigma,
                    alpha=alpha,
                    member_ids=tuple(sorted(member_ids)),
                    time_start=timestamps[0] if timestamps else "",
                    time_end=timestamps[-1] if timestamps else "",
                    waypoints=tuple(inherited),
                )
            )
        vistas.sort(key=lambda item: (-item.alpha, item.slug))

        projection = VistaProjection(
            events=event_map,
            vectors=vectors,
            idf=idf,
            event_waypoints=event_waypoints,
            waypoints=waypoints,
            vistas=tuple(vistas),
            truncated=truncated,
        )
        self._snapshot_key = snapshot_key
        self._projection = projection
        return projection

    def query(
        self,
        query: str,
        *,
        seed_event_ids: Iterable[str] = (),
        limit: int = 8,
        wave_hops: int = 4,
        reranker: Reranker | None = None,
    ) -> VistaResult:
        """Vista query. When ``reranker`` is set, evidence is scored via the
        trained two-stage-retrieval readout (see
        :mod:`willow_substrate.readout`); otherwise the existing ad-hoc
        combination is used unchanged.
        """
        projection = self.project()
        seeds = tuple(
            event_id
            for event_id in dict.fromkeys(seed_event_ids)
            if event_id in projection.events
        )
        if not projection.events:
            return VistaResult(
                query=query,
                seed_event_ids=(),
                reference_beams=(),
                matches=(),
                evidence=(),
                trace=("The active event projection is empty.",),
            )

        query_vector = self._query_vector(query, projection.idf)
        if not query_vector and seeds:
            query_vector = _mean_vector(
                projection.vectors[event_id] for event_id in seeds
            )
        beams = self._reference_beams(query, seeds, projection)
        total_beam_weight = sum(beam.weight for beam in beams) or 1.0

        matches: list[VistaMatch] = []
        for vista in projection.vistas:
            proximity = (
                max(0.0, _dot(query_vector, vista.mu))
                if query_vector
                else 0.0
            )
            distance = max(0.0, 1.0 - proximity)
            gaussian = (
                math.exp(
                    -(distance * distance)
                    / (2.0 * vista.sigma * vista.sigma)
                )
                if query_vector
                else 0.0
            )
            semantic = 0.75 * proximity + 0.25 * gaussian

            shared: list[str] = []
            waypoint_score = 0.0
            for beam in beams:
                touching = sum(
                    1
                    for event_id in vista.member_ids
                    if beam.name
                    in projection.event_waypoints.get(event_id, {})
                )
                if not touching:
                    continue
                shared.append(beam.name)
                waypoint_score += beam.weight * (
                    touching / len(vista.member_ids)
                )
            waypoint_score /= total_beam_weight

            if query_vector and beams:
                combined = 0.45 * semantic + 0.55 * waypoint_score
            elif query_vector:
                combined = semantic
            else:
                combined = waypoint_score
            # Gaussian support is intentionally soft and never exactly zero.
            # A reporting floor prevents numerical tail dust from being
            # presented as meaningful recall while preserving the field maths.
            if combined < 0.05:
                continue
            score = combined * (1.0 + 0.3 * vista.alpha)
            matches.append(
                VistaMatch(
                    vista=vista,
                    score=score,
                    semantic_score=semantic,
                    gaussian_score=gaussian,
                    waypoint_score=waypoint_score,
                    shared_waypoints=tuple(shared),
                )
            )
        matches.sort(key=lambda item: (-item.score, item.vista.slug))
        selected_matches = tuple(matches[: max(0, limit)])

        direct: dict[str, dict[str, Any]] = {}
        for match in selected_matches:
            for event_id in match.vista.member_ids:
                if event_id in seeds:
                    continue
                member_semantic = (
                    max(
                        0.0,
                        _dot(query_vector, projection.vectors[event_id]),
                    )
                    if query_vector
                    else 0.0
                )
                score = 0.80 * match.score + 0.20 * member_semantic
                current = direct.setdefault(
                    event_id,
                    {
                        "score": 0.0,
                        "slugs": set(),
                        "waypoints": set(),
                    },
                )
                current["score"] = max(current["score"], score)
                current["slugs"].add(match.vista.slug)
                current["waypoints"].update(
                    name
                    for name in match.shared_waypoints
                    if name
                    in projection.event_waypoints.get(event_id, {})
                )

        wave_seeds = seeds
        if not wave_seeds:
            wave_seeds = tuple(
                event_id
                for event_id, _ in sorted(
                    (
                        (event_id, values["score"])
                        for event_id, values in direct.items()
                    ),
                    key=lambda item: (-item[1], item[0]),
                )[:3]
            )
        wave_hop_count = max(0, wave_hops)
        if reranker is not None:
            wave_scores, wave_carriers, wave_trajectory = self._wave(
                projection,
                wave_seeds,
                hops=wave_hop_count,
                damping=self.wave_damping,
                return_trajectory=True,
            )
        else:
            wave_scores, wave_carriers = self._wave(
                projection,
                wave_seeds,
                hops=wave_hop_count,
                damping=self.wave_damping,
            )
            wave_trajectory = []
        non_seed_wave = {
            event_id: score
            for event_id, score in wave_scores.items()
            if event_id not in wave_seeds and score > 0
        }
        wave_peak = max(non_seed_wave.values(), default=0.0)
        # Per-hop peak used to normalise the per-hop feature into [0, 1] when
        # a trained reranker consumes the wave's transient shape.
        hop_peaks = [
            max((score for score in snapshot.values()), default=0.0)
            for snapshot in wave_trajectory
        ]

        evidence_ids = set(direct) | set(non_seed_wave)
        evidence: list[VistaEvidence] = []
        for event_id in evidence_ids:
            if event_id in seeds:
                continue
            direct_values = direct.get(
                event_id,
                {"score": 0.0, "slugs": set(), "waypoints": set()},
            )
            vista_score = float(direct_values["score"])
            wave_score = (
                non_seed_wave.get(event_id, 0.0) / wave_peak
                if wave_peak
                else 0.0
            )
            wave_features: WaveFeatures | None = None
            if reranker is not None:
                per_hop = [
                    (
                        snapshot.get(event_id, 0.0) / hop_peaks[index]
                        if hop_peaks[index]
                        else 0.0
                    )
                    for index, snapshot in enumerate(wave_trajectory)
                ]
                peak = max(per_hop) if per_hop else 0.0
                hop_of_peak = per_hop.index(peak) if per_hop and peak > 0 else 0
                early = per_hop[0] if per_hop else 0.0
                wave_features = build_wave_features(
                    vista_score=vista_score,
                    wave_score=wave_score,
                    wave_peak=peak,
                    wave_hop_of_peak_index=hop_of_peak,
                    wave_early_activation=early,
                    hops=wave_hop_count,
                )
                score = float(reranker.score(wave_features))
            else:
                score = max(vista_score, 0.45 * wave_score)
            channels: list[str] = []
            if vista_score:
                channels.append("vista")
            if wave_score:
                channels.append("wave")
            carrier = wave_carriers.get(event_id)
            waypoint_names = set(direct_values["waypoints"])
            if carrier:
                waypoint_names.add(carrier)
            evidence.append(
                VistaEvidence(
                    event=projection.events[event_id],
                    score=score,
                    vista_score=vista_score,
                    wave_score=wave_score,
                    vista_slugs=tuple(sorted(direct_values["slugs"])),
                    waypoints=tuple(sorted(waypoint_names)),
                    channels=tuple(channels),
                    wave_features=wave_features,
                )
            )
        evidence.sort(key=lambda item: (-item.score, -item.event.seq))
        selected_evidence = tuple(evidence[: max(0, limit)])

        trace = [
            f"Projected {len(projection.events)} active events into "
            f"{len(projection.vistas)} soft Vistas.",
            f"Resolved {len(beams)} relational reference beams from "
            f"{len(seeds)} focused seed events and the query.",
            "Ranked Vistas using semantic/Gaussian proximity and "
            "waypoint-member overlap; timestamps did not enter geometry.",
            f"Propagated a degree-normalised, conductance-weighted Wave "
            f"for {max(0, wave_hops)} hops with damping "
            f"{self.wave_damping:.2f} from {len(wave_seeds)} seeds.",
        ]
        if projection.truncated:
            trace.insert(
                1,
                f"Projection reached the {self.max_events}-event reference "
                "backend cap; older active events were not evaluated.",
            )
        return VistaResult(
            query=query,
            seed_event_ids=seeds,
            reference_beams=beams,
            matches=selected_matches,
            evidence=selected_evidence,
            trace=tuple(trace),
        )

    def _event_features(self, event: Event) -> Counter[str]:
        features: Counter[str] = Counter()
        words = _words(event.content)
        features.update(f"word:{word}" for word in words)
        features.update(
            {
                f"pair:{left}_{right}": 0.65
                for left, right in zip(words, words[1:])
            }
        )
        for key, kind in WAYPOINT_METADATA.items():
            if key not in event.metadata:
                continue
            for value in _metadata_values(event.metadata[key]):
                normalised = _normalise(value)
                if normalised:
                    weight = 3.0 if kind == "shape" else 2.2
                    features[f"{kind}:{normalised}"] += weight
        for value in _wikilinks(event.content):
            features[f"memory:{value}"] += 3.0
        for phrase in _proper_phrases(event.content):
            features[f"entity:{phrase}"] += 1.8
        return features

    @staticmethod
    def _tfidf_vector(
        features: Mapping[str, float],
        idf: Mapping[str, float],
    ) -> SparseVector:
        weighted = {
            key: (1.0 + math.log(max(value, 1e-9))) * idf.get(key, 1.0)
            for key, value in features.items()
            if value > 0
        }
        return _normalise_vector(weighted)

    def _query_vector(
        self,
        query: str,
        idf: Mapping[str, float],
    ) -> SparseVector:
        words = _words(query)
        features: Counter[str] = Counter(f"word:{word}" for word in words)
        features.update(
            {
                f"pair:{left}_{right}": 0.65
                for left, right in zip(words, words[1:])
            }
        )
        for phrase in _proper_phrases(query):
            features[f"entity:{phrase}"] += 1.8
        return self._tfidf_vector(
            {
                key: value
                for key, value in features.items()
                if key in idf
            },
            idf,
        )

    def _extract_waypoints(
        self,
        events: Iterable[Event],
    ) -> dict[str, dict[str, float]]:
        event_list = list(events)
        edges: dict[str, dict[str, float]] = {
            event.id: {}
            for event in event_list
        }
        event_ids = set(edges)
        phrase_frequency: Counter[str] = Counter()
        event_phrases: dict[str, set[str]] = {}
        for event in event_list:
            phrases = _proper_phrases(event.content)
            event_phrases[event.id] = phrases
            phrase_frequency.update(phrases)

            actor = _normalise(event.actor)
            if actor and actor not in GENERIC_ACTORS:
                edges[event.id][f"actor:{actor}"] = 0.40
            for key, kind in WAYPOINT_METADATA.items():
                if key not in event.metadata:
                    continue
                prominence = 1.50 if kind == "shape" else 1.20
                for value in _metadata_values(event.metadata[key]):
                    normalised = _normalise(value)
                    if normalised:
                        edges[event.id][
                            f"{kind}:{normalised}"
                        ] = prominence
            for value in _wikilinks(event.content):
                edges[event.id][f"memory:{value}"] = 1.60

        for event in event_list:
            for phrase in event_phrases[event.id]:
                if phrase_frequency[phrase] >= 2:
                    edges[event.id][f"entity:{phrase}"] = 1.10

        for event in event_list:
            linked_ids = list(event.derived_from)
            if event.supersedes:
                linked_ids.append(event.supersedes)
            for source_id in linked_ids:
                if source_id not in event_ids:
                    continue
                name = f"memory:{source_id}"
                edges[event.id][name] = 1.80
                edges[source_id][name] = 1.80
        return edges

    def _recency_decays(self, events: Iterable[Event]) -> dict[str, float]:
        event_list = list(events)
        parsed = {
            event.id: _parse_timestamp(event.timestamp)
            for event in event_list
        }
        newest = max(
            (value for value in parsed.values() if value is not None),
            default=None,
        )
        if newest is None:
            return {event.id: 1.0 for event in event_list}
        decays = {}
        for event in event_list:
            timestamp = parsed[event.id]
            if timestamp is None:
                decays[event.id] = 1.0
                continue
            age_days = max(0.0, (newest - timestamp).total_seconds() / 86400.0)
            decays[event.id] = 0.5 ** (age_days / self.half_life_days)
        return decays

    def _cluster(
        self,
        vectors: Mapping[str, Mapping[str, float]],
    ) -> list[tuple[str, ...]]:
        event_ids = sorted(vectors)
        parent = {event_id: event_id for event_id in event_ids}

        def find(value: str) -> str:
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        def union(left: str, right: str) -> None:
            left_root, right_root = find(left), find(right)
            if left_root == right_root:
                return
            if left_root < right_root:
                parent[right_root] = left_root
            else:
                parent[left_root] = right_root

        postings: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for event_id, vector in vectors.items():
            for feature, value in vector.items():
                postings[feature].append((event_id, value))
        pair_scores: dict[tuple[str, str], float] = defaultdict(float)
        high_frequency_limit = max(64, int(len(event_ids) * 0.20))
        for members in postings.values():
            if len(members) > high_frequency_limit:
                continue
            for index, (left_id, left_value) in enumerate(members):
                for right_id, right_value in members[index + 1 :]:
                    pair = (
                        (left_id, right_id)
                        if left_id < right_id
                        else (right_id, left_id)
                    )
                    pair_scores[pair] += left_value * right_value
        for (left_id, right_id), similarity in pair_scores.items():
            if similarity >= self.cluster_threshold:
                union(left_id, right_id)

        components: dict[str, list[str]] = defaultdict(list)
        for event_id in event_ids:
            components[find(event_id)].append(event_id)
        return [
            tuple(members)
            for _, members in sorted(components.items())
        ]

    def _reference_beams(
        self,
        query: str,
        seeds: tuple[str, ...],
        projection: VistaProjection,
    ) -> tuple[ReferenceBeam, ...]:
        scores: dict[str, float] = defaultdict(float)
        sources: dict[str, set[str]] = defaultdict(set)
        for event_id in seeds:
            for name, prominence in projection.event_waypoints.get(
                event_id,
                {},
            ).items():
                waypoint = projection.waypoints[name]
                specificity = waypoint.mass / max(1, waypoint.degree)
                scores[name] += prominence * specificity
                sources[name].add("focus")

        query_terms = set(_words(query))
        normalised_query = _normalise(query)
        for name, waypoint in projection.waypoints.items():
            label = name.partition(":")[2]
            label_terms = set(re.split(r"[-_:]+", label))
            if (
                label
                and (
                    label in normalised_query
                    or bool(query_terms & label_terms)
                )
            ):
                scores[name] += waypoint.mass / max(1, waypoint.degree)
                sources[name].add("query")

        ranked = sorted(
            scores.items(),
            key=lambda item: (-item[1], item[0]),
        )[: self.max_beams]
        return tuple(
            ReferenceBeam(
                name=name,
                kind=projection.waypoints[name].kind,
                weight=score,
                source="+".join(sorted(sources[name])),
            )
            for name, score in ranked
        )

    @staticmethod
    def _wave(
        projection: VistaProjection,
        seeds: Iterable[str],
        *,
        hops: int,
        damping: float = 0.5,
        return_trajectory: bool = False,
    ):
        """Degree-normalised, conductance-weighted, damped spreading activation.

        Returns ``(activation, carriers)`` by default. When
        ``return_trajectory`` is true the wave's per-hop activation snapshots
        are also returned: ``(activation, carriers, trajectory)`` where
        ``trajectory`` is a list of length ``hops`` whose entries are
        dictionaries mapping event id to the settled activation after that
        hop. The trajectory feeds trained readouts (see
        :mod:`willow_substrate.readout`) so a reranker can use per-hop
        wave shape features, not only the final scalar.
        """
        seed_ids = tuple(
            event_id
            for event_id in dict.fromkeys(seeds)
            if event_id in projection.events
        )
        if not seed_ids or hops <= 0:
            if return_trajectory:
                return {}, {}, []
            return {}, {}
        base = {
            event_id: 1.0 / len(seed_ids)
            for event_id in seed_ids
        }
        activation = dict(base)
        peak_waypoint_flow: dict[str, float] = defaultdict(float)
        trajectory: list[dict[str, float]] = []

        for _ in range(hops):
            waypoint_activation: dict[str, float] = defaultdict(float)
            for event_id, amount in activation.items():
                edges = projection.event_waypoints.get(event_id, {})
                total = sum(
                    projection.waypoints[name].mass * prominence
                    for name, prominence in edges.items()
                ) or 1.0
                for name, prominence in edges.items():
                    flow = (
                        amount
                        * projection.waypoints[name].mass
                        * prominence
                        / total
                    )
                    waypoint_activation[name] += flow
            for name, amount in waypoint_activation.items():
                peak_waypoint_flow[name] = max(
                    peak_waypoint_flow[name],
                    amount,
                )

            memory_activation: dict[str, float] = defaultdict(float)
            for name, amount in waypoint_activation.items():
                waypoint = projection.waypoints[name]
                share = amount / max(1, waypoint.degree)
                for event_id in waypoint.member_ids:
                    memory_activation[event_id] += share
            activation = {
                event_id: (
                    (1.0 - damping)
                    * memory_activation.get(event_id, 0.0)
                    + damping
                    * base.get(event_id, 0.0)
                )
                for event_id in set(memory_activation) | set(base)
            }
            if return_trajectory:
                trajectory.append(dict(activation))

        carriers: dict[str, str] = {}
        for event_id in activation:
            candidates = []
            for name, prominence in projection.event_waypoints.get(
                event_id,
                {},
            ).items():
                if name not in peak_waypoint_flow:
                    continue
                candidates.append(
                    (
                        peak_waypoint_flow[name]
                        * projection.waypoints[name].mass
                        * prominence,
                        name,
                    )
                )
            if candidates:
                carriers[event_id] = max(candidates)[1]
        if return_trajectory:
            return activation, carriers, trajectory
        return activation, carriers
