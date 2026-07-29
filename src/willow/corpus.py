"""Import a Markdown corpus as immutable events.

The event store is the index.  A directory of Markdown files can be the
*source*.  Keeping both is not redundancy for its own sake: a database is the
right shape for retrieval and the wrong shape for a total software failure.
When every script is broken, a directory of Markdown files is still readable
with ``ls`` and ``cat``, and the store can be rebuilt from it.

Import is idempotent by content.  Re-importing an unchanged file returns the
existing event.  Re-importing a *changed* file appends a correction that
supersedes the previous version of that path, so file history accumulates in the
hash chain instead of overwriting itself.  Nothing is ever deleted; a file
removed from disk simply stops receiving new versions.

Frontmatter keys land in event metadata, where the Vista projection already
turns ``topic``/``topics``/``person``/``shape`` into waypoints, and where
``standing`` is read by :mod:`willow.salience`.  ``[[wikilinks]]`` in the body
are already waypoints and citation edges, so a corpus that cross-references
itself arrives with its relational structure intact.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from willow.events import Event
from willow.store import EventStore

SOURCE_PATH_KEY = "source_path"
SOURCE_SHA_KEY = "source_sha256"
DEFAULT_KIND = "note"

# Frontmatter is intentionally a small, predictable subset rather than full
# YAML: a scalar, a bracketed list, a dash list, or one level of nesting. A
# corpus that needs more than this is better served by an explicit adapter than
# by a parser that guesses.
_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
_SCALAR_TRUE = {"true", "yes", "on"}
_SCALAR_FALSE = {"false", "no", "off"}


@dataclass(frozen=True)
class ImportedFile:
    """The outcome of importing one Markdown file."""

    path: Path
    relative_path: str
    event: Event
    status: str  # created | unchanged | superseded


@dataclass(frozen=True)
class ImportReport:
    """Aggregate outcome of a corpus import."""

    root: Path
    files: tuple[ImportedFile, ...]
    skipped: tuple[tuple[Path, str], ...] = ()

    @property
    def created(self) -> tuple[ImportedFile, ...]:
        return tuple(item for item in self.files if item.status == "created")

    @property
    def unchanged(self) -> tuple[ImportedFile, ...]:
        return tuple(item for item in self.files if item.status == "unchanged")

    @property
    def superseded(self) -> tuple[ImportedFile, ...]:
        return tuple(item for item in self.files if item.status == "superseded")

    def summary(self) -> str:
        return (
            f"{len(self.created)} new, {len(self.superseded)} updated, "
            f"{len(self.unchanged)} unchanged, {len(self.skipped)} skipped"
        )


def _coerce(value: str) -> Any:
    text = value.strip()
    if not text:
        return ""
    if text[0] in "\"'" and text[-1] == text[0] and len(text) >= 2:
        return text[1:-1]
    lowered = text.lower()
    if lowered in _SCALAR_TRUE:
        return True
    if lowered in _SCALAR_FALSE:
        return False
    if text.startswith("[") and text.endswith("]"):
        return [
            item.strip().strip("\"'")
            for item in text[1:-1].split(",")
            if item.strip()
        ]
    return text


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split a Markdown document into (frontmatter, body)."""

    match = _FRONTMATTER.match(text)
    if not match:
        return {}, text
    block = match.group(1)
    body = text[match.end() :]

    data: dict[str, Any] = {}
    # A key with no inline value is *undecided*: the next line reveals whether
    # it opened a list (dash items) or a mapping (indented key/value pairs).
    open_key: str | None = None
    nested_open_key: str | None = None

    for raw_line in block.splitlines():
        if not raw_line.strip() or raw_line.strip().startswith("#"):
            continue
        indented = raw_line[:1].isspace()
        line = raw_line.strip()

        if line.startswith("- "):
            item = _coerce(line[2:])
            if (
                nested_open_key is not None
                and isinstance(data.get(open_key), dict)
            ):
                container: dict[str, Any] = data[open_key]
                key = nested_open_key
            elif open_key is not None:
                container = data
                key = open_key
            else:
                continue
            existing = container.get(key)
            if isinstance(existing, list):
                existing.append(item)
            else:
                container[key] = [item]
            continue

        key, separator, value = line.partition(":")
        if not separator:
            continue
        key = key.strip()
        if not key:
            continue

        if indented and open_key is not None:
            if not isinstance(data.get(open_key), dict):
                data[open_key] = {}
            if value.strip():
                data[open_key][key] = _coerce(value)
                nested_open_key = None
            else:
                nested_open_key = key
            continue

        if value.strip():
            data[key] = _coerce(value)
            open_key = None
        else:
            data[key] = None
            open_key = key
        nested_open_key = None

    # A key that opened and never received content is an empty list.
    for key, value in list(data.items()):
        if value is None:
            data[key] = []
    return data, body


def _flatten_metadata(frontmatter: dict[str, Any]) -> dict[str, Any]:
    """Lift a nested ``metadata:`` block to the top level.

    Vista and salience read flat keys.  A corpus that nests its type under
    ``metadata:`` should not have to restructure itself to be understood.
    """

    flat: dict[str, Any] = {}
    for key, value in frontmatter.items():
        if key == "metadata" and isinstance(value, dict):
            for nested_key, nested_value in value.items():
                flat.setdefault(nested_key, nested_value)
            continue
        flat[key] = value
    return flat


def _markdown_files(root: Path, pattern: str) -> list[Path]:
    return sorted(
        path
        for path in root.rglob(pattern)
        if path.is_file() and not path.name.startswith(".")
    )


def _latest_for_path(
    store: EventStore,
    relative_path: str,
    *,
    kind: str,
    scan_limit: int,
) -> Event | None:
    """Find the active event representing a corpus path.

    This is a bounded scan rather than an index lookup.  It is honest about
    being a reference implementation: a production adapter should carry a
    ``source_path`` index instead of paying this cost per file.
    """

    for event in store.events(limit=scan_limit, kind=kind, active_only=True):
        if event.metadata.get(SOURCE_PATH_KEY) == relative_path:
            return event
    return None


def import_markdown(
    store: EventStore,
    root: str | Path,
    *,
    actor: str = "corpus",
    kind: str = DEFAULT_KIND,
    session_id: str = "corpus",
    pattern: str = "*.md",
    max_bytes: int = 200_000,
    scan_limit: int = 20_000,
    extra_metadata: dict[str, Any] | None = None,
) -> ImportReport:
    """Import a directory of Markdown files as immutable events."""

    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise NotADirectoryError(f"corpus root is not a directory: {root_path}")

    imported: list[ImportedFile] = []
    skipped: list[tuple[Path, str]] = []

    for path in _markdown_files(root_path, pattern):
        relative_path = path.relative_to(root_path).as_posix()
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            skipped.append((path, f"unreadable: {exc}"))
            continue
        if len(raw.encode("utf-8")) > max_bytes:
            skipped.append((path, f"larger than {max_bytes} bytes"))
            continue
        if not raw.strip():
            skipped.append((path, "empty"))
            continue

        frontmatter, body = parse_frontmatter(raw)
        content = body.strip() or raw.strip()
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()

        metadata: dict[str, Any] = dict(extra_metadata or {})
        metadata.update(_flatten_metadata(frontmatter))
        metadata[SOURCE_PATH_KEY] = relative_path
        metadata[SOURCE_SHA_KEY] = digest
        metadata.setdefault("name", path.stem)

        previous = _latest_for_path(
            store,
            relative_path,
            kind=kind,
            scan_limit=scan_limit,
        )
        if previous is not None and previous.metadata.get(SOURCE_SHA_KEY) == digest:
            imported.append(
                ImportedFile(
                    path=path,
                    relative_path=relative_path,
                    event=previous,
                    status="unchanged",
                )
            )
            continue

        event, created = store.append_idempotent(
            content,
            idempotency_key=f"corpus:{relative_path}:{digest}",
            actor=actor,
            kind=kind,
            session_id=session_id,
            metadata=metadata,
            supersedes=previous.id if previous is not None else None,
            derived_from=(previous.id,) if previous is not None else None,
        )
        imported.append(
            ImportedFile(
                path=path,
                relative_path=relative_path,
                event=event,
                status=(
                    "superseded"
                    if previous is not None
                    else ("created" if created else "unchanged")
                ),
            )
        )

    return ImportReport(
        root=root_path,
        files=tuple(imported),
        skipped=tuple(skipped),
    )


def corpus_events(store: EventStore, *, kind: str = DEFAULT_KIND, limit: int = 20_000) -> Iterable[Event]:
    """Active events that came from a Markdown corpus."""

    for event in store.events(limit=limit, kind=kind, active_only=True):
        if SOURCE_PATH_KEY in event.metadata:
            yield event
