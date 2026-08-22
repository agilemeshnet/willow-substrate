# Two-stage retrieval: Wave retrieves, Vista ranks

**Status.** Ships in v0.2.2 as a zero-dep primitive.
**Modules.** `willow_substrate.readout`, `willow_substrate.vista.VistaBackend.query(..., reranker=...)`.
**Empirical anchor.** +345% recall@K lift, measured on Peter Cooper's
1440-memory / 400-waypoint Doozer substrate (2026-08-21). The fixed
zero-dependency reference benchmark also guards a positive Wave + Vista lift:
0.516 to 0.624 mean Recall@5 (+20.9%).

## What the substrate has been doing all along

Willow's `VistaBackend` already carries two independent views over the
event ledger. They were built for different reasons and were never
formally connected:

* **Vista** is a static semantic clustering. Events with cosine-close
  feature vectors form soft Gaussian neighbourhoods. Query proximity to
  a Vista's centroid, weighted by beam overlap, gives a `vista_score`
  per candidate event.
* **Wave** is dynamic spreading activation. From a seed event, activation
  hops through the shared-waypoint lattice, damped by an alpha restart
  and weighted by mass-decayed conductance. The settled activation
  after N hops gives a `wave_score` per candidate event.

Until v0.2.2, `VistaBackend.query` combined the two via a fixed
heuristic:

```python
score = max(vista_score, 0.45 * wave_score)
```

## The measurement that named the shape

Instrumented on the Doozer substrate (2026-08-21), the two views turn
out to be **orthogonal**, not redundant:

* Wave's top-K activation includes the seed's Vista mates only **12%**
  of the time on average (median 0.000; 75x above random baseline).
* But Wave's top-100 reaches **60%** of the seed's Vista mates, and
  its full activation vector reaches **92%**.

So Wave is a coarse RETRIEVER of the right pool; Vista is a fine RANKER
within it. The two channels see genuinely different structure. Wave
finds cross-Vista bridges via shared entities/wikilinks; Vista clusters
by embedding cosine. Combining them at the readout layer beats either
signal alone:

| Readout | mean recall@K | median | lift over bare wave sort |
|---|---|---|---|
| Baseline (bare wave sort) | 0.117 | 0.000 | - |
| Ridge (wave + Vista features) | 0.505 | 0.500 | **+333%** |
| Logistic (wave + Vista features) | 0.518 | 0.500 | **+345%** |
| Vista cosine alone | 0.518 | 0.500 | +345% |

The trained readout matches Vista-alone on this specific task because
the Vista cosine feature carries the discriminating signal. What the
learned readout adds is architectural: it makes the two-stage pattern
EXPLICIT, so richer feature sets (per-hop wave shape, external
image/audio embeddings) can compose additively without redesign.

## Biology

Rats at a maze junction integrate at least two neural signals when they
choose. Hippocampal REPLAY sweeps down candidate corridors (dynamic,
waypoint-like, this is Wave); PLACE CELLS and GRID CELLS carry a static
embedding of the environment (similarity-like, this is Vista). The
decision uses both. The rat's "vicarious trial and error" is the
transient the two-signal integration is settling through.

Willow's two-stage retrieval has the same architecture with the same
combination move. The Wave and Vista mechanisms were not designed as
biological analogues; the measurement revealed the shape.

## Ancestry: reservoir computing

Wave is functionally an Echo State Network with a dumb readout: fixed
random topology (the waypoint graph), spectral-radius-bounded
recurrence (the alpha damping), a driven input (the seed), a transient
that settles (the wave), and a linear readout on the settled state
(currently a sort by final activation). The reservoir-computing
literature makes the general principle clear: don't retrain the
substrate; retrain the READOUT.

`LinearReranker` is that readout. Its weight vector is the ONE thing
you fit; the underlying substrate never changes.

## What ships

### `WaveFeatures` (per-candidate)

Six normalised features exposed to the reranker per surfaced event:

```
vista_score        [0..1] direct semantic proximity from Vista matching
wave_score         [0..1] final wave activation, normalised by peak
wave_peak          [0..1] peak wave activation across hops, per-hop normalised
wave_hop_of_peak   [0..1] hop index where wave peaked, normalised by total hops
wave_early         [0..1] arrival immediacy: activation at hop 1
channel_bias       {0, 0.5, 1.0} indicator of both/one/no channel lit
```

All six land in the closed unit interval so that a linear combination is
stable across corpus sizes and backends.

### `LinearReranker` (zero-dep readout)

```python
from willow_substrate import LinearReranker, VistaBackend

reranker = LinearReranker.default()          # pre-fitted weights
result = VistaBackend(store).query(
    "recurrent motifs",
    seed_event_ids=(seed.id,),
    limit=8,
    reranker=reranker,                       # opt-in; default is None
)
for evidence in result.evidence:
    # `wave_features` is populated only when a reranker is passed
    print(evidence.score, evidence.wave_features)
```

Custom weights via `LinearReranker.from_dict({feature_name: coef, ...})`.

Fit your own coefficients externally with any linear model
(Ridge, Logistic Regression, LARS, or a hand-rolled least-squares
solver over the six-column feature matrix) and pass them into
`from_dict`.

### Preserved defaults

`VistaBackend.query` without a `reranker` argument keeps the existing
ad-hoc `max(vista_score, 0.45 * wave_score)` combination, so no
existing call site changes behaviour. Every previously-shipped test
still passes on v0.2.2 unchanged.

## Exercising the reference implementation

See `benchmarks/readout/wave_ridge_recall.py`. It builds a synthetic event
corpus with known cluster structure on the reference backend, runs Wave
without and with `LinearReranker.default()`, and reports the Recall@K delta.
The fixed benchmark is a regression guard: the default's early-arrival Wave
signal raises mean Recall@5 from 0.516 to 0.624 (+20.9%) over the legacy
heuristic. It is not a reproduction of the external +345% Doozer result or a
guarantee for another corpus; fit on representative training data and evaluate
against held-out queries before making production performance claims. Runs in
a few seconds; zero dependencies.

## Extending the readout to non-text channels

`WaveFeatures` is deliberately a fixed six-tuple in v0.2.2 so the
weight vector is small and stable. When a caller wants richer
features (CLIP-image cosine for image seeds, CLAP-audio for audio
seeds, temporal decay signals), the pattern is: subclass
`WaveFeatures` or extend the feature schema in a follow-up minor
release, and register a corresponding reranker that consumes the
enlarged vector. The retrieve-then-rerank split stays the same.

The biology reading suggests one field, N channels: eventually the
reranker's input should be a single unified vector integrating all
sensory channels. See `docs/BENCHMARK_LOCOMO.md` for empirical
evaluation of when adding channels lifts task-specific metrics vs
when it does not.

## Provenance

* Discovery arc, 2026-08-20 to 2026-08-21, in Peter Cooper's Scout
  workspace. The counterfactual-clustering literature paste of
  2026-08-20 named the substrate's counterfactual geometry; the
  reservoir-computing paste of 2026-08-20 named `wave.py` as an ESN
  with a dumb readout; the measurement of orthogonality and the +345%
  lift landed 2026-08-21.
* Doozer substrate is 1440 memories, 400 waypoints, using Voyage-4
  dense embeddings. The pattern is expected to generalise to the
  zero-dep reference backend (sparse lexical-semantic); this is what
  the benchmark script verifies.
* Related memory entries in the internal corpus:
  `project_wave_and_vista_two_orthogonal_eyes_2026_08_21`,
  `feedback_wave_py_is_an_echo_state_network_2026_08_20`,
  `feedback_wave_retrieves_voyage_ranks_2026_08_21`.
