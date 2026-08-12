# Banks: the floor a boot cannot lose

A Willow boot has two layers, and they obey different rules.

**Banks are constitutional.** They are the identity of the region that boots
here and the standing rules of the substrate it operates in. They are included
whole, at the top, on every boot. They are never truncated to make room for
experience, and they do not change between wakes, which also makes them stable
enough for a provider to cache.

**Flow is everything else.** Events, peer engrams, Vista surround, research,
facts. Flow is ranked and budgeted, and its budget is the residual after the
banks are paid.

Without this separation, a process with a perfect event ledger still boots as a
stranger holding somebody else's notes. It can tell you what happened. It cannot
tell you who it is, what it is responsible for, or which corrections it has
already been given.

## Where banks live

```
$WILLOW_HOME/
  IDENTITY.md      # who this region is
  GROUND.md        # the standing rules of this substrate
  banks/           # optional further documents, filename order
    house-style.md
```

Filename case does not change meaning: `IDENTITY.md` and `identity.md` are the
same bank. Empty or unreadable files are skipped rather than producing an empty
section.

They are plain Markdown on purpose. A person can read and edit them without the
tool, and a total software failure still leaves the constitution recoverable
with `cat`.

## Commands

```bash
willow init                 # writes bank templates if none exist
willow init --without-banks # skip scaffolding
willow banks                # show which banks load, and their token cost
willow banks --full         # print their contents
willow status               # includes the size of the floor
willow boot                 # banks whole, then ranked flow
```

`init` never overwrites an existing bank. An empty constitution somebody forgot
to write is a recoverable problem; a constitution silently replaced by a
template is not.

## Ordering

Identity comes before ground, because a region needs to know who it is before
the rules of where it lives mean anything. Additional `banks/*.md` documents
follow in filename order, so an operator can add a third document without
editing code.

## What belongs in a bank

Banks are expensive: they are paid in full on every boot. That cost is the
discipline. Material earns a place in a bank when a fresh process cannot infer
it and would act wrongly without it:

- what this region is responsible for, and what belongs to someone else;
- how the people it works with want to be addressed and answered;
- constraints that were learned by something going wrong;
- rules whose violation is not detectable from the event history alone.

Material that is merely *important* belongs in the corpus, marked `standing`,
where [salience](SALIENCE.md) will rank it up without charging every boot for
it.

## Relationship to the rest of the system

Banks answer *who is booting*. Foveation answers *what is in focus*. Vista
answers *what contextual whole this belongs to*. Salience answers *what survives
the budget*. Only the first is exempt from ranking.
