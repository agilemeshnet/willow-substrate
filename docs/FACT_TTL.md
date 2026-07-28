# Fact TTL

**Every factual claim has a shelf life appropriate to how its world changes.**

Fact TTL prevents a long-running research substrate from quietly filling with
confidently obsolete claims. Each claim carries:

- its source and domain;
- a positive TTL, or `-1` for an intentionally immutable claim;
- `last_verified_at`;
- its own `next_check_at`;
- optional topics and idea-shapes.

The claim is the clock. A scheduler merely asks which claims are due.

## Reference commands

```bash
willow claim \
  "The corpus contains 42 papers" \
  --source "versioned corpus manifest" \
  --domain research \
  --ttl-days 7

willow facts
willow facts --due
```

Record a confirmation only with identified evidence:

```bash
willow fact-check evt-... \
  --outcome confirmed \
  --evidence-kind primary \
  --source "signed corpus manifest v3" \
  --evidence "The manifest still contains 42 unique paper IDs."
```

## Outcomes

| Outcome | Effect |
|---|---|
| `confirmed` | Advances `last_verified_at` and computes a new `next_check_at` |
| `updated` | Appends a replacement claim and a check linking old and new |
| `contradicted` | Appends a dispute/check and filters the unsupported claim from ordinary retrieval |
| `unknown` | Records the attempt and a retry time; does not refresh validity |
| `unreachable` | Records provider failure and a retry time; does not refresh validity |

Confirming, updating, or contradicting requires an evidence note, an identified
source, and evidence classified as primary, secondary, user, or sensor.
Model-only recollection is explicitly rejected as confirmation.

## Preservation and filtering

No claim or check is deleted. Updated or contradicted claims stop participating
in normal retrieval because a later event supersedes them. Historical
inspection can still reconstruct:

- the original statement;
- what supported it;
- when it was challenged;
- the evidence used;
- the replacement, if any.

This is the same additive correction ethic used throughout Willow.

## What is not implemented yet

The reference core provides the state machine, due projection, CLI, and
epistemic guardrails. It does not yet ship:

- a background scheduler;
- web, scholarly, or domain-API verifiers;
- adaptive TTL based on observed volatility;
- vector-neighbour contagion;
- a freshness-field visualization;
- automatic human-review queues.

Those are adapters over this contract. A verifier must return evidence, not
merely a verdict.
