"""Dense-embedding Vista + Wave backend.

Substitutes for the dependency-free VistaBackend in src/willow/vista.py via
the RelationalBackend Protocol. Turns TF-IDF-style sparse retrieval into
dense semantic recall over the full event corpus, then computes a damped
multi-hop wave over the co-occurrence graph derived from cluster membership.

Ported from the internal Willow thought-buckets pipeline
(mint_buckets.py + wave.py + voyage_embed.py), adapted to willow-substrate's
event ledger. The dependency-free VistaBackend remains the floor; installing
[vista] activates this richer backend without changing the read-side
evidence contract.

Install with:

    pip install -e ".[vista]"

Usage:

    from willow.store import EventStore
    from willow.backends.vista_voyage import VoyageVistaBackend

    store = EventStore()
    backend = VoyageVistaBackend(store, api_key=os.environ["VOYAGE_API_KEY"])
    result = backend.query("recurrent connectome motifs", limit=8)

The returned VistaResult has the same shape as the dependency-free
VistaBackend's; downstream code (context composer, foveation, CLI, hooks)
reads it unchanged. Vistas populated by this backend leave `mu` as an empty
sparse dict because the semantic centroid lives in a dense space; the score
gates use `semantic_score` populated directly by cosine similarity.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

# These imports intentionally fail at module scope if the [vista] extra is
# missing. The ImportError message directs the caller to install the extra.
try:
    import numpy as np
    from sklearn.cluster import HDBSCAN
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "willow.backends.vista_voyage requires the [vista] extra. "
        "Install with: pip install \"willow-substrate[vista]\""
    ) from exc

from willow.events import Event
from willow.store import EventStore
from willow.vista import (
    ReferenceBeam,
    Vista,
    VistaEvidence,
    VistaMatch,
    VistaResult,
)


VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"
DEFAULT_VOYAGE_MODEL = "voyage-4"
DEFAULT_VOYAGE_DIM = 1024
DEFAULT_BATCH = 64
DEFAULT_RATE_LIMIT_S = 22.0  # Voyage free tier: 3 RPM safe cadence
DEFAULT_CACHE_NAME = "willow_voyage_embeddings.npz"


def _canonical_event_text(event: Event) -> str:
    """The string an event contributes to embedding.

    Kept short and stable: an actor/kind stem plus the content. Metadata is
    intentionally not folded in; keeping the input surface small stops noise
    from swamping the semantic signal.
    """
    stem = f"[{event.actor}/{event.kind}]"
    return f"{stem} {event.content.strip()}"


class VoyageEmbedder:
    """Voyage-4 REST client with on-disk .npz cache.

    Caches by event id so a single event never re-embeds; a corpus that grew
    by ten events only pays for those ten. Model name is stamped into the
    cache header so different embedding dimensions never silently collide.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_VOYAGE_MODEL,
        dim: int = DEFAULT_VOYAGE_DIM,
        cache_path: Path | str,
        batch_size: int = DEFAULT_BATCH,
        rate_limit_s: float = DEFAULT_RATE_LIMIT_S,
    ):
        self.api_key = api_key or os.environ.get("VOYAGE_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "VOYAGE_API_KEY not set; pass api_key= or export the env var"
            )
        self.model = model
        self.dim = dim
        self.cache_path = Path(cache_path)
        self.batch_size = max(1, batch_size)
        self.rate_limit_s = max(0.0, rate_limit_s)
        self._cache: dict[str, "np.ndarray"] = self._load_cache()

    def _load_cache(self) -> dict[str, "np.ndarray"]:
        if not self.cache_path.exists():
            return {}
        try:
            data = np.load(self.cache_path, allow_pickle=True)
            keys = list(data["keys"])
            vecs = data["vecs"]
            stored_model = str(data.get("model", ""))
            if stored_model and stored_model != self.model:
                # Model mismatch: do not silently reuse different-dimensional
                # vectors. Return empty and let the caller re-embed.
                return {}
            return {str(k): vecs[i] for i, k in enumerate(keys)}
        except Exception:
            return {}

    def _save_cache(self) -> None:
        if not self._cache:
            return
        keys = list(self._cache.keys())
        matrix = np.stack([self._cache[k] for k in keys], axis=0)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.cache_path,
            keys=np.array(keys, dtype=object),
            vecs=matrix,
            model=self.model,
            dim=self.dim,
            embedded_at=datetime.now(timezone.utc).isoformat(),
        )

    def _post_batch(self, texts: Sequence[str]) -> "np.ndarray":
        body = json.dumps(
            {"input": list(texts), "model": self.model}
        ).encode("utf-8")
        req = urllib.request.Request(
            VOYAGE_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover
            raise RuntimeError(
                f"Voyage embed failed: HTTP {exc.code} {exc.reason}"
            ) from exc
        return np.array(
            [item["embedding"] for item in payload["data"]],
            dtype=np.float32,
        )

    def embed(self, items: Iterable[tuple[str, str]]) -> dict[str, "np.ndarray"]:
        """Return a {key: vector} map for every requested (key, text) pair.

        Cached keys are served from memory; only cache misses hit Voyage.
        """
        pending: list[tuple[str, str]] = []
        result: dict[str, "np.ndarray"] = {}
        for key, text in items:
            cached = self._cache.get(key)
            if cached is not None:
                result[key] = cached
            else:
                pending.append((key, text))

        if pending:
            for start in range(0, len(pending), self.batch_size):
                batch = pending[start : start + self.batch_size]
                vectors = self._post_batch([text for _, text in batch])
                for (key, _), vec in zip(batch, vectors):
                    self._cache[key] = vec
                    result[key] = vec
                if start + self.batch_size < len(pending):
                    time.sleep(self.rate_limit_s)
            self._save_cache()
        return result


def _cluster_by_hdbscan(
    vectors: "np.ndarray",
    *,
    min_cluster_size: int = 3,
) -> "np.ndarray":
    """Cluster labels via HDBSCAN over cosine distance."""
    n = vectors.shape[0]
    if n < min_cluster_size * 2:
        # Not enough points to cluster meaningfully; single-cluster fallback.
        return np.zeros(n, dtype=int)
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="cosine",
        cluster_selection_method="eom",
    )
    return clusterer.fit_predict(vectors)


def _cosine_matrix(a: "np.ndarray", b: "np.ndarray") -> "np.ndarray":
    """Cosine similarity between rows of a and rows of b."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T


@dataclass
class _DenseProjection:
    events: tuple[Event, ...]
    event_ids: tuple[str, ...]
    matrix: "np.ndarray"
    labels: "np.ndarray"
    truncated: bool

    @property
    def event_by_id(self) -> dict[str, Event]:
        return {event.id: event for event in self.events}


class VoyageVistaBackend:
    """RelationalBackend implementation using dense Voyage-4 embeddings.

    Substitutable for the dependency-free VistaBackend. Produces the same
    VistaResult shape so downstream context, foveation, and CLI code is
    unchanged.

    The algorithm, at a glance:
    1. Read active events from the store.
    2. Embed them via Voyage-4 (cached per-event id).
    3. Cluster via HDBSCAN in the embedding space; each cluster is a Vista.
    4. Embed the query text; cosine-rank matches against the corpus.
    5. Wave-recall: from top matches, run a damped multi-hop spread over the
       co-occurrence graph (edges between events sharing a Vista) so the
       returned neighbourhood reflects the whole cluster, not just the seed.

    Vistas populated by this backend have `mu = {}` because the semantic
    centroid lives in a dense space that VistaBackend's sparse-vector API
    does not carry; the score gates use `semantic_score` populated directly
    by dense cosine similarity. `alpha` is set to a size-and-cohesion
    salience so downstream consumers can still rank by it.
    """

    def __init__(
        self,
        store: EventStore,
        *,
        api_key: str | None = None,
        max_events: int = 2000,
        min_cluster_size: int = 3,
        wave_damping: float = 0.5,
        cache_path: Path | str | None = None,
        embedder: VoyageEmbedder | None = None,
    ):
        if max_events < 1:
            raise ValueError("max_events must be positive")
        if not 0.0 <= wave_damping <= 1.0:
            raise ValueError("wave_damping must be between zero and one")
        self.store = store
        self.max_events = max_events
        self.min_cluster_size = min_cluster_size
        self.wave_damping = wave_damping
        if embedder is not None:
            self.embedder = embedder
        else:
            resolved_cache = (
                Path(cache_path)
                if cache_path is not None
                else store.home / DEFAULT_CACHE_NAME
            )
            self.embedder = VoyageEmbedder(
                api_key=api_key, cache_path=resolved_cache
            )
        self._projection_key: tuple[str, ...] = ()
        self._projection: _DenseProjection | None = None

    def _project(self) -> _DenseProjection:
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
        key = tuple(event.id for event in events)
        if self._projection is not None and key == self._projection_key:
            return self._projection

        if not events:
            projection = _DenseProjection(
                events=(),
                event_ids=(),
                matrix=np.zeros((0, DEFAULT_VOYAGE_DIM), dtype=np.float32),
                labels=np.zeros(0, dtype=int),
                truncated=truncated,
            )
        else:
            vectors_map = self.embedder.embed(
                (event.id, _canonical_event_text(event)) for event in events
            )
            ordered_ids = tuple(
                event.id for event in events if event.id in vectors_map
            )
            matrix = (
                np.stack([vectors_map[eid] for eid in ordered_ids], axis=0)
                if ordered_ids
                else np.zeros((0, DEFAULT_VOYAGE_DIM), dtype=np.float32)
            )
            labels = (
                _cluster_by_hdbscan(matrix, min_cluster_size=self.min_cluster_size)
                if ordered_ids
                else np.zeros(0, dtype=int)
            )
            projection = _DenseProjection(
                events=tuple(events),
                event_ids=ordered_ids,
                matrix=matrix,
                labels=labels,
                truncated=truncated,
            )

        self._projection = projection
        self._projection_key = key
        return projection

    def query(
        self,
        query: str,
        *,
        seed_event_ids: Iterable[str] = (),
        limit: int = 8,
        wave_hops: int = 4,
    ) -> VistaResult:
        projection = self._project()
        seed_tuple = tuple(seed_event_ids)
        trace: list[str] = [
            f"corpus={len(projection.events)}",
            f"vectorised={len(projection.event_ids)}",
        ]

        if not projection.events or projection.matrix.size == 0:
            trace.append("empty corpus or embedder returned no vectors")
            return VistaResult(
                query=query,
                seed_event_ids=seed_tuple,
                reference_beams=(),
                matches=(),
                evidence=(),
                trace=tuple(trace),
            )

        # 1. Embed the query and rank events by cosine similarity.
        query_text = query.strip()
        query_similarity: dict[str, float] = {}
        if query_text:
            query_map = self.embedder.embed(
                [(f"__query__:{query_text}", query_text)]
            )
            if query_map:
                q = next(iter(query_map.values())).reshape(1, -1)
                sims = _cosine_matrix(q, projection.matrix)[0]
                for eid, sim in zip(projection.event_ids, sims):
                    query_similarity[eid] = float(max(0.0, sim))
                trace.append(
                    f"query cosine top1={max(query_similarity.values()):.3f}"
                    if query_similarity
                    else "query embed returned empty"
                )

        # 2. Merge in explicit seeds (weight 1.0).
        merged_scores: dict[str, float] = dict(query_similarity)
        event_ids_set = set(projection.event_ids)
        for eid in seed_tuple:
            if eid in event_ids_set:
                merged_scores[eid] = max(merged_scores.get(eid, 0.0), 1.0)

        # 3. Build Vistas from HDBSCAN clusters.
        vistas = self._build_vistas(projection)
        vistas_by_slug = {vista.slug: vista for vista in vistas}
        event_to_vistas = defaultdict(list)
        for vista in vistas:
            for eid in vista.member_ids:
                event_to_vistas[eid].append(vista.slug)
        trace.append(f"vistas={len(vistas)}")

        # 4. Reference beams: top-K query-similar events as attention pointers.
        beams: list[ReferenceBeam] = []
        for eid, score in sorted(
            query_similarity.items(), key=lambda kv: kv[1], reverse=True
        )[:limit]:
            event = projection.event_by_id.get(eid)
            if event is None or score <= 0.0:
                continue
            beams.append(
                ReferenceBeam(
                    name=event.id,
                    kind="event",
                    weight=score,
                    source="voyage-cosine",
                )
            )

        # 5. Rank Vistas by best-member semantic score.
        vista_matches: list[VistaMatch] = []
        for vista in vistas:
            if not vista.member_ids:
                continue
            member_scores = [
                query_similarity.get(eid, 0.0) for eid in vista.member_ids
            ]
            semantic = max(member_scores) if member_scores else 0.0
            if semantic <= 0.0:
                continue
            vista_matches.append(
                VistaMatch(
                    vista=vista,
                    score=semantic * vista.alpha,
                    semantic_score=semantic,
                    gaussian_score=0.0,
                    waypoint_score=0.0,
                    shared_waypoints=(),
                )
            )
        vista_matches.sort(key=lambda m: m.score, reverse=True)
        vista_matches = vista_matches[:limit]
        trace.append(f"vista_matches={len(vista_matches)}")

        # 6. Wave-recall: damped multi-hop spread over the co-occurrence graph.
        wave_scores = self._wave_expand(
            projection, merged_scores, hops=wave_hops
        )
        trace.append(
            f"wave_hops={wave_hops} damping={self.wave_damping}"
        )

        # 7. Combine into VistaEvidence, ranked by aggregate score.
        evidence = self._build_evidence(
            projection,
            query_similarity=query_similarity,
            wave_scores=wave_scores,
            event_to_vistas=event_to_vistas,
            limit=limit,
        )

        return VistaResult(
            query=query,
            seed_event_ids=seed_tuple,
            reference_beams=tuple(beams),
            matches=tuple(vista_matches),
            evidence=tuple(evidence),
            trace=tuple(trace),
        )

    def _build_vistas(
        self, projection: _DenseProjection
    ) -> list[Vista]:
        by_label: dict[int, list[int]] = defaultdict(list)
        for idx, label in enumerate(projection.labels):
            if label < 0:
                continue
            by_label[int(label)].append(idx)

        vistas: list[Vista] = []
        for label, indices in by_label.items():
            member_ids = tuple(
                sorted(projection.event_ids[i] for i in indices)
            )
            block = projection.matrix[indices]
            centroid = block.mean(axis=0)
            centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-12)
            per_member_sim = (block @ centroid_norm.reshape(-1, 1)).flatten()
            cohesion = float(per_member_sim.mean()) if len(indices) else 0.0
            alpha = float(cohesion * math.log1p(len(indices)))
            # sigma: 1 - cohesion, floored to 0.08 (matches VistaBackend floor).
            sigma = max(0.08, 1.0 - cohesion)
            timestamps = sorted(
                projection.event_by_id[eid].timestamp for eid in member_ids
            )
            # Slug derived from sorted member ids (stable across re-projections).
            identity = "\0".join(member_ids).encode("utf-8")
            slug = "vista-voyage-" + hashlib.sha256(identity).hexdigest()[:12]
            vistas.append(
                Vista(
                    slug=slug,
                    mu={},  # sparse-dict representation not used; see class docstring
                    sigma=sigma,
                    alpha=alpha,
                    member_ids=member_ids,
                    time_start=timestamps[0] if timestamps else "",
                    time_end=timestamps[-1] if timestamps else "",
                    waypoints=(),
                )
            )
        vistas.sort(key=lambda v: (-v.alpha, v.slug))
        return vistas

    def _wave_expand(
        self,
        projection: _DenseProjection,
        seed_scores: dict[str, float],
        *,
        hops: int,
    ) -> dict[str, float]:
        """Damped multi-hop spreading activation over the cluster graph.

        Edge weights are the pairwise cosine similarities within each
        cluster (non-cluster nodes contribute no edges). Per-hop:
        activation_next = damping * degree_normalised_spread(activation)
                        + (1 - damping) * seed_activation.
        Degree normalisation keeps a dense cluster from swamping a smaller
        one (the anti-seizure governor from thought-buckets/wave.py).
        """
        n = len(projection.event_ids)
        if n == 0 or not seed_scores:
            return {}

        id_to_idx = {eid: i for i, eid in enumerate(projection.event_ids)}
        seed_activation = np.zeros(n, dtype=np.float32)
        for eid, score in seed_scores.items():
            i = id_to_idx.get(eid)
            if i is not None:
                seed_activation[i] = float(score)

        cluster_members: dict[int, list[int]] = defaultdict(list)
        for idx, label in enumerate(projection.labels):
            if label < 0:
                continue
            cluster_members[int(label)].append(idx)

        current = seed_activation.copy()
        for _ in range(max(0, hops)):
            spread = np.zeros(n, dtype=np.float32)
            for members in cluster_members.values():
                if len(members) < 2:
                    continue
                block = projection.matrix[members]
                block_norm = block / (
                    np.linalg.norm(block, axis=1, keepdims=True) + 1e-12
                )
                sim = block_norm @ block_norm.T
                np.fill_diagonal(sim, 0.0)
                local_activation = current[members]
                spread_local = sim @ local_activation
                degree_norm = sim.sum(axis=1) + 1e-12
                spread_local = spread_local / degree_norm
                for i, m in enumerate(members):
                    spread[m] += spread_local[i]
            current = (
                self.wave_damping * spread
                + (1 - self.wave_damping) * seed_activation
            )

        return {
            projection.event_ids[i]: float(current[i])
            for i in range(n)
            if current[i] > 0.0
        }

    def _build_evidence(
        self,
        projection: _DenseProjection,
        *,
        query_similarity: dict[str, float],
        wave_scores: dict[str, float],
        event_to_vistas: dict[str, list[str]],
        limit: int,
    ) -> list[VistaEvidence]:
        candidates: dict[str, tuple[float, float, tuple[str, ...]]] = {}
        for eid in set(query_similarity) | set(wave_scores):
            event = projection.event_by_id.get(eid)
            if event is None:
                continue
            vista_score = float(query_similarity.get(eid, 0.0))
            wave_score = float(wave_scores.get(eid, 0.0))
            channels: list[str] = []
            if vista_score > 0.0:
                channels.append("semantic")
            if wave_score > 0.0:
                channels.append("wave")
            if not channels:
                continue
            candidates[eid] = (
                vista_score,
                wave_score,
                tuple(channels),
            )

        # Composite score: max of the two channels (they carry different scales
        # so summing would over-weight wave when many clusters contribute).
        evidence = [
            VistaEvidence(
                event=projection.event_by_id[eid],
                score=max(vista_score, wave_score),
                vista_score=vista_score,
                wave_score=wave_score,
                vista_slugs=tuple(event_to_vistas.get(eid, ())),
                waypoints=(),
                channels=channels,
            )
            for eid, (vista_score, wave_score, channels) in candidates.items()
        ]
        evidence.sort(key=lambda item: item.score, reverse=True)
        return evidence[:limit]
