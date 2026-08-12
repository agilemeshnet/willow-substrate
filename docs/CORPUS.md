# Corpus: Markdown as the source, the store as the index

The event store is the right shape for retrieval and the wrong shape for a total
software failure. A directory of Markdown files is the reverse. Keeping both is
not redundancy for its own sake: when every script is broken, `ls` and `cat`
still work, and the store can be rebuilt from the files.

```bash
willow import ./notes
willow import ./notes --kind note --session corpus --pattern "*.md"
```

## Import is idempotent by content

Re-importing an unchanged file appends nothing. Re-importing a *changed* file
appends a correction that supersedes the previous version of that path, so file
history accumulates in the hash chain instead of overwriting itself.

```text
created    measure-first.md  evt-3f9a0c21b7d4
updated    ground-rules.md   evt-88b1e0aa4c02
1 new, 1 updated, 12 unchanged, 0 skipped
```

The superseded version stays in the chain and is excluded from normal retrieval,
exactly like any other correction. A file deleted from disk simply stops
receiving new versions; nothing is removed from history.

## Frontmatter becomes metadata

```markdown
---
name: measure-first
description: Measure before changing anything
standing: true
topics: [method, discipline]
metadata:
  type: feedback
---

Instrument the system before mutating it. Defer the change until an external
measurement exists, otherwise the fix and the evidence for it arrive together
and neither can be trusted. Related: [[verification-before-completion]].
```

- `topics`, `person`, `place`, `project`, and `idea_shape` become
  [Vista](VISTA.md) waypoints.
- `standing` is read by [salience](SALIENCE.md).
- A nested `metadata:` block is lifted to the top level, so a corpus that nests
  its type does not have to restructure itself to be understood.
- `name` defaults to the filename stem and is what other files cite.

## Wikilinks arrive as structure

`[[other-note]]` in the body is already a Vista waypoint and a citation edge. A
corpus that cross-references itself therefore arrives with its relational
neighbourhood intact, and in-degree feeds the citation signal in salience. No
separate link-extraction pass is required.

## Frontmatter parsing is deliberately small

Scalars, bracketed lists, dash lists, and one level of nesting. A corpus that
needs more than this is better served by an explicit adapter than by a parser
that guesses. Files that are unreadable, empty, or larger than `--max-bytes` are
reported as skipped rather than failing the import.

## Known limit

Matching a file to its previous version is a bounded scan over active events of
the given kind, not an index lookup. This is honest about being a reference
implementation: a production adapter should carry a `source_path` index instead
of paying that cost per file.
