# Salience: what survives the budget

Retrieval decides what is *eligible*. Salience decides what is *kept* when the
budget runs out before the candidates do.

Before this layer, truncation was decided by whichever retrieval arm happened to
run first. A standing rule learned six months ago lost its place to an
incidental message from this morning, and nothing in the output showed that it
had happened.

## Two mechanisms, not one

**Retrieval of standing material.** A rule that was learned by something going
wrong must not have to match the query to be eligible. Relevance is precisely
the thing such a rule cannot rely on: the sessions that most need it are the
ones that were not thinking about it. Active events marked `standing` are pulled
into the candidate set regardless of the query, bounded by `standing_limit`.

**Ranking.** Candidates are then ordered by an explainable score, and the budget
truncates the tail rather than an arbitrary middle.

## The signals

| Signal | Weight | What it captures |
|---|---|---|
| `standing` | +8.0 | Explicitly marked standing or foundational |
| `citation` | +0.5 each, capped +4.0 | Wikilink in-degree: material others keep pointing at |
| `reflection` | +1.0 | Meditations, summations, dreams, engrams are already distilled |
| `recency` | up to +3.0 | Half-life of 14 days, a decay rather than a cliff |
| `query` | up to +2.0 | Lexical overlap with the current intent |

The score is a sum of named parts rather than one opaque number, so any ranking
can be explained:

```bash
willow salience "what rules apply" --explain
```

```text
 11.83  evt-9f2c1a...  STANDING: measure before changing anything.
        11.83 = standing=+8.00 (marked standing) citation=+1.50 (cited by 3) recency=+2.33 (4.1d old)
```

These weights are a policy, not a discovery. They are deliberately blunt and are
expected to be tuned. What matters is that they are inspectable.

## Time is a voice, not a coordinate

Recency is one signal among five and never a geometric axis. This is the same
commitment Vista makes: timestamps are provenance. A six-month-old standing rule
outranks this morning's chatter, which is the entire point.

## Marking material as standing

Any of these in event metadata will do, because a corpus written by a human
should not have to match one exact key to keep its own rules:

```yaml
standing: true
foundational: yes
type: standing
tags: [foundational]
```

[Corpus import](CORPUS.md) reads these from Markdown frontmatter.

## Citation depth

Wikilink in-degree is the dependency-free analogue of a Vista waypoint's
accumulated mass: material that later material keeps pointing at has been
reinforced by use rather than by assertion. Self-citation does not count. An
event that mentions its own name is labelled, not important.

## Turning it off

```bash
willow context "..." --without-salience
```

Peer engrams are exempt from ranking and keep their position at the head of the
context. They are a live coordination signal with their own short expiry, so
they are not asked to compete with durable material on durability.
