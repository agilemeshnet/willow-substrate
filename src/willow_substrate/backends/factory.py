"""Backend factory: pick the right RelationalBackend for the caller's environment.

The audit that motivated this module found that ``pip install
"willow-substrate[vista]"`` did not activate the dense backend for ordinary
users: the CLI, hooks, context composer, and connection finder all
constructed the dependency-free ``VistaBackend`` directly. Installing the
extra was necessary but not sufficient. This module is the sufficient half:
one factory that every construction site calls, one env var and one CLI
flag that pick the backend.

Usage from calling code:

    from willow_substrate.backends.factory import make_relational_backend
    backend = make_relational_backend(store)                 # auto
    backend = make_relational_backend(store, name="voyage")  # force
    backend = make_relational_backend(store, name="sparse")  # force floor

Resolution order for the default ``auto`` mode:

1. If the caller passes ``name=`` explicitly, honour it.
2. Else if ``WILLOW_BACKEND`` env is set, use that.
3. Else if the ``[vista]`` extra is importable AND ``VOYAGE_API_KEY`` is
   set, return a ``VoyageVistaBackend``.
4. Otherwise return the dependency-free ``VistaBackend``.

Rationale for the ``VOYAGE_API_KEY`` gate: constructing
``VoyageVistaBackend`` without an API key raises ``RuntimeError`` at
construction time. Requiring the key present keeps auto-selection from
turning a working sparse read into a hard failure the moment someone
pip installs the extra.

The dependency-free floor stays load-bearing. Never returning ``None``,
never raising for the ``auto`` path, and never silently swapping to a
backend that would error at first use: those three invariants are the
contract of this module.
"""
from __future__ import annotations

import os
from typing import Any

from willow_substrate.store import EventStore
from willow_substrate.vista import VistaBackend


VALID_NAMES = ("auto", "voyage", "sparse", "default", "hybrid")


class BackendNotAvailable(RuntimeError):
    """Requested backend is not installed or not configured for use."""


def make_relational_backend(
    store: EventStore,
    *,
    name: str | None = None,
    max_events: int | None = None,
    wave_damping: float | None = None,
    **extra: Any,
):
    """Return a RelationalBackend selected by name / env / availability.

    The signature accepts optional ``max_events`` and ``wave_damping`` so
    callers that used to pass them to ``VistaBackend`` directly can keep
    doing so without changes. Any unknown kwargs are passed through to
    the concrete backend.
    """
    resolved = _resolve_name(name)

    if resolved == "voyage":
        try:
            from willow_substrate.backends.vista_voyage import VoyageVistaBackend
        except ImportError as exc:
            raise BackendNotAvailable(
                "Backend 'voyage' requires the [vista] extra. Install with: "
                "pip install \"willow-substrate[vista]\""
            ) from exc
        kwargs: dict[str, Any] = dict(extra)
        if max_events is not None:
            kwargs.setdefault("max_events", max_events)
        if wave_damping is not None:
            kwargs.setdefault("wave_damping", wave_damping)
        return VoyageVistaBackend(store, **kwargs)

    if resolved in ("sparse", "default"):
        return _sparse_backend(
            store, max_events=max_events, wave_damping=wave_damping, **extra
        )

    if resolved == "hybrid":
        from willow_substrate.backends.hybrid import HybridRecallBackend

        kwargs = dict(extra)
        if max_events is not None:
            kwargs.setdefault("max_events", max_events)
        if wave_damping is not None:
            kwargs.setdefault("wave_damping", wave_damping)
        return HybridRecallBackend(store, **kwargs)

    # 'auto' resolution: try Voyage if extras + key are both present.
    if _voyage_available_and_configured():
        try:
            from willow_substrate.backends.vista_voyage import VoyageVistaBackend

            kwargs = dict(extra)
            if max_events is not None:
                kwargs.setdefault("max_events", max_events)
            if wave_damping is not None:
                kwargs.setdefault("wave_damping", wave_damping)
            return VoyageVistaBackend(store, **kwargs)
        except Exception:
            # If activation fails at construction time for any reason, do
            # not deny recall; fall back to the sparse floor.
            pass

    return _sparse_backend(
        store, max_events=max_events, wave_damping=wave_damping, **extra
    )


def _sparse_backend(
    store: EventStore,
    *,
    max_events: int | None,
    wave_damping: float | None,
    **extra: Any,
) -> VistaBackend:
    kwargs = dict(extra)
    if max_events is not None:
        kwargs.setdefault("max_events", max_events)
    if wave_damping is not None:
        kwargs.setdefault("wave_damping", wave_damping)
    return VistaBackend(store, **kwargs)


def _resolve_name(name: str | None) -> str:
    if name is not None:
        candidate = name.strip().lower()
    else:
        candidate = os.environ.get("WILLOW_BACKEND", "auto").strip().lower()
    if candidate == "":
        candidate = "auto"
    if candidate not in VALID_NAMES:
        raise ValueError(
            f"unknown backend {candidate!r}; expected one of {VALID_NAMES}"
        )
    return candidate


def _voyage_available_and_configured() -> bool:
    """True when both the [vista] extra and a VOYAGE_API_KEY are present."""
    if not os.environ.get("VOYAGE_API_KEY"):
        return False
    try:
        import willow_substrate.backends.vista_voyage  # noqa: F401
    except ImportError:
        return False
    return True


def active_backend_name(name: str | None = None) -> str:
    """What ``make_relational_backend`` would resolve to, for logging + CLI."""
    resolved = _resolve_name(name)
    if resolved == "auto":
        return "voyage" if _voyage_available_and_configured() else "sparse"
    if resolved == "default":
        return "sparse"
    return resolved
