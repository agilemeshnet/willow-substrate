"""Constitutional banks: the part of grounding that is never ranked away.

A Willow boot has two layers with different physics.

*Banks* are constitutional.  They are the identity of the region and the rules
of the substrate it operates in.  They are included **whole**, at the top, on
every boot, and they are never truncated to make room for experience.  Because
they do not change between wakes they are also stable enough for a provider to
cache.

*Flow* is everything else: events, engrams, Vista surround, research, facts.
Flow is ranked and budgeted, and the budget it gets is the residual left after
the banks are paid.

Separating the two is what distinguishes a process that *remembers the work*
from a process that *is a particular region doing the work*.  A store with
perfect recall and no banks boots as a stranger holding somebody else's notes.

Banks are plain Markdown files inside ``WILLOW_HOME`` so that a human can read
and edit them without the tool, and so that a total software failure still
leaves the constitution recoverable with ``cat``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

BANK_DIRECTORY = "banks"

# Ordered: identity before ground, because a region needs to know who it is
# before the rules of where it lives mean anything.
CORE_BANKS: tuple[tuple[str, str, str], ...] = (
    ("identity", "identity.md", "Identity bank"),
    ("ground", "ground.md", "Constitutional ground"),
)

IDENTITY_TEMPLATE = """# Identity

Replace this file with the identity of the region that boots here.

Useful things to state, because a fresh process cannot infer them:

- Who this region is, and what it is *not*.
- What it is responsible for, and what belongs to someone else.
- How the people it works with want to be addressed and answered.
- The failure modes it has already been corrected for.

This file is included whole on every boot and is never truncated.
Keep it short enough that being included whole is affordable.
"""

GROUND_TEMPLATE = """# Ground

Replace this file with the standing rules of this substrate.

These are the constraints every session inherits, as distinct from the identity
of any one region. Rules that have earned their place here are usually the ones
that were learned by something going wrong.

This file is included whole on every boot and is never truncated.
"""


@dataclass(frozen=True)
class Bank:
    """One constitutional document included whole at boot."""

    name: str
    heading: str
    path: Path
    text: str

    @property
    def size_bytes(self) -> int:
        return len(self.text.encode("utf-8"))

    @property
    def estimated_tokens(self) -> int:
        return max(1, len(self.text) // 4)


def _read(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None
    return text or None


def _resolve_case_insensitive(home: Path, filename: str) -> Path | None:
    """Find ``filename`` in ``home`` regardless of case.

    ``IDENTITY.md`` and ``identity.md`` are the same intent, and which one a
    person reaches for is a matter of habit rather than meaning.
    """

    direct = home / filename
    if direct.exists():
        return direct
    lowered = filename.lower()
    try:
        entries = sorted(home.iterdir())
    except OSError:
        return None
    for entry in entries:
        if entry.is_file() and entry.name.lower() == lowered:
            return entry
    return None


def _heading_for(path: Path) -> str:
    stem = path.stem.replace("-", " ").replace("_", " ").strip()
    return stem[:1].upper() + stem[1:] if stem else path.name


def load_banks(home: str | Path) -> tuple[Bank, ...]:
    """Load the constitutional banks for a Willow home.

    Core banks (identity, ground) come first in a fixed order.  Any further
    ``banks/*.md`` files follow in filename order, so an operator can add
    a third document without editing code.  Empty or unreadable files are
    skipped rather than emitting an empty section.
    """

    home_path = Path(home).expanduser()
    banks: list[Bank] = []
    for name, filename, heading in CORE_BANKS:
        path = _resolve_case_insensitive(home_path, filename)
        if path is None:
            continue
        text = _read(path)
        if text is None:
            continue
        banks.append(Bank(name=name, heading=heading, path=path, text=text))

    extra_directory = home_path / BANK_DIRECTORY
    if extra_directory.is_dir():
        try:
            entries = sorted(extra_directory.iterdir())
        except OSError:
            entries = []
        for entry in entries:
            if not entry.is_file() or entry.suffix.lower() != ".md":
                continue
            text = _read(entry)
            if text is None:
                continue
            banks.append(
                Bank(
                    name=entry.stem.lower(),
                    heading=_heading_for(entry),
                    path=entry,
                    text=text,
                )
            )
    return tuple(banks)


def render_banks(banks: tuple[Bank, ...]) -> list[str]:
    """Render banks as boot Markdown sections."""

    lines: list[str] = []
    for bank in banks:
        lines.extend(["", f"## {bank.heading}", "", bank.text])
    return lines


def scaffold_banks(home: str | Path) -> tuple[Path, ...]:
    """Write bank templates for any core bank that does not exist yet.

    Existing files are never overwritten.  An empty constitution that a person
    forgot to write is a recoverable problem; a constitution silently replaced
    by a template is not.
    """

    home_path = Path(home).expanduser()
    home_path.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    templates = {
        "identity.md": IDENTITY_TEMPLATE,
        "ground.md": GROUND_TEMPLATE,
    }
    for filename, template in templates.items():
        if _resolve_case_insensitive(home_path, filename) is not None:
            continue
        path = home_path / filename.upper().replace(".MD", ".md")
        path.write_text(template, encoding="utf-8")
        written.append(path)
    return tuple(written)
