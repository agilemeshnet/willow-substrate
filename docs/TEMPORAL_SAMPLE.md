# Temporal sample: cumulative scene formation

Willow should be evaluated on experience accumulated over time, not on one
large prompt that already contains the complete story.

The public sample in
[`examples/temporal-bird-study.json`](../examples/temporal-bird-study.json)
contains nine synthetic fragments staged over eight weeks. It uses no
Peter-specific stories or private source material.

```bash
export WILLOW_HOME="$PWD/.willow-sample"
willow sample run examples/temporal-bird-study.json
```

The command idempotently replays the dated fragments and evaluates the sample's
declared ground truth. A successful run checks:

- all fragments are present in one valid append-only chain;
- the record spans at least eight weeks, four providers, and five sessions;
- a later correction filters the provisional count without deleting it;
- construction and predation connect through explicit shape despite having no
  useful words in common;
- a physical bridge and mathematical bridge function are reported as a
  word-only near-match, not silently promoted to structural evidence;
- a trust-and-safety connection crosses domains through
  `boundary:constrains-and-enables`;
- the supported reconstruction uses the correction, respects withholding, and
  labels the lexical false friend.

## Timeline

| Week | Fragment | Purpose |
|---:|---|---|
| 1 | Provisional visual count | Fallible initial observation |
| 2 | Visitor works and redistribution | First external-pressure shape |
| 3 | Predation and clustering | Shape-only parallel |
| 4 | Image-based count correction | Supersession without erasure |
| 5 | Mathematical bridge function | Ambiguous lexical near-match |
| 6 | Coordinate-withholding decision | Privacy boundary as positive agency |
| 7 | Selective-disclosure meditation | Trust/safety fibre bond |
| 8 | Pressure reinterpretation | Later integration across causes |
| 9 | Supported summation | Cohesive, provenance-bearing scene |

The timestamps are in 2030 so consent-expiry metadata remains inspectable
during current development. Replaying the sample does not wait eight weeks; it
preserves the original event times so retrieval can be tested immediately.

## What this does and does not prove

Passing the deterministic checks proves that the reference substrate preserves
the necessary temporal, correction, provenance, word, and shape signals. It
does not prove that every model will narrate them faithfully.

The manifest therefore includes a human rubric. A model-facing evaluation
should ask an unfamiliar temporary agent to reconstruct the study and score
whether it:

1. retains supported facts and corrections;
2. distinguishes evidence from meditation;
3. finds the shape-only connections;
4. rejects the lexical false friend as causal evidence;
5. surfaces the disclosure boundary before sensitive retrieval;
6. names provenance and uncertainty;
7. avoids invented details.

This is the beginning of a benchmark format. Additional samples should vary
domain, duration, language, correction type, privacy policy, and distractor
density while retaining an inspectable answer key.

## Privacy boundary

The sample never contains precise nesting coordinates. It records that
withholding occurred, its audience and purpose, and the aggregate projection
that remains permitted.

That distinction is intentional:

> Append-only integrity protects what happened. Consent determines what may be
> disclosed now.

The alpha can preserve such a boundary event and present it as context.
Capability-enforced consent, revocation, audience projection, and disclosure
auditing remain required before the public bundle can claim comprehensive
privacy enforcement.
