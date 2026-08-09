"""Tests for the backend factory that wires [vista] into the defaults.

The factory exists because ``pip install "willow-substrate[vista]"`` alone
did not activate the dense backend: the CLI, context composer, and
connections default construction sites all created ``VistaBackend``
directly. This test suite pins the factory's resolution rules so a
regression cannot silently return the sparse backend when the caller
explicitly asked for Voyage.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from willow.store import EventStore
from willow.vista import VistaBackend
from willow.backends.factory import (
    BackendNotAvailable,
    active_backend_name,
    make_relational_backend,
)


try:
    import willow.backends.vista_voyage  # noqa: F401
    HAS_VISTA_EXTRA = True
except ImportError:
    HAS_VISTA_EXTRA = False


class FactoryResolutionTests(unittest.TestCase):
    """Env / kwarg resolution rules for the factory."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.home = Path(self.temp.name)
        self.store = EventStore(self.home)

    def tearDown(self):
        self.temp.cleanup()

    def test_explicit_sparse_returns_vistabackend(self):
        backend = make_relational_backend(self.store, name="sparse")
        self.assertIsInstance(backend, VistaBackend)

    def test_default_alias_is_sparse(self):
        backend = make_relational_backend(self.store, name="default")
        self.assertIsInstance(backend, VistaBackend)

    def test_auto_without_key_returns_sparse_even_when_extra_installed(self):
        """VOYAGE_API_KEY missing must NOT auto-activate the Voyage backend;
        otherwise a working sparse read would flip to a construction-time
        failure the moment a user pip-installed the extra."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VOYAGE_API_KEY", None)
            os.environ.pop("WILLOW_BACKEND", None)
            backend = make_relational_backend(self.store)
            self.assertIsInstance(backend, VistaBackend)
            self.assertEqual(active_backend_name(), "sparse")

    def test_unknown_name_raises(self):
        with self.assertRaises(ValueError):
            make_relational_backend(self.store, name="hallucinated-backend")

    def test_env_var_forces_sparse_even_with_key_present(self):
        with patch.dict(
            os.environ,
            {"WILLOW_BACKEND": "sparse", "VOYAGE_API_KEY": "fake"},
            clear=False,
        ):
            backend = make_relational_backend(self.store)
            self.assertIsInstance(backend, VistaBackend)
            self.assertEqual(active_backend_name(), "sparse")

    def test_explicit_name_overrides_env(self):
        with patch.dict(
            os.environ, {"WILLOW_BACKEND": "voyage"}, clear=False
        ):
            backend = make_relational_backend(self.store, name="sparse")
            self.assertIsInstance(backend, VistaBackend)

    def test_kwargs_passthrough_to_sparse(self):
        # max_events + wave_damping should reach the underlying VistaBackend.
        backend = make_relational_backend(
            self.store,
            name="sparse",
            max_events=99,
            wave_damping=0.25,
        )
        self.assertEqual(backend.max_events, 99)
        self.assertAlmostEqual(backend.wave_damping, 0.25)


@unittest.skipUnless(
    HAS_VISTA_EXTRA,
    "install with pip install \"willow-substrate[vista]\" to run these tests",
)
class FactoryVoyagePathTests(unittest.TestCase):
    """Voyage resolution when the extra is present."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.home = Path(self.temp.name)
        self.store = EventStore(self.home)

    def tearDown(self):
        self.temp.cleanup()

    def test_forced_voyage_without_key_raises_backendnotavailable(self):
        """name='voyage' must NOT silently fall back to sparse; the caller
        explicitly asked for the dense backend and deserves the error."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VOYAGE_API_KEY", None)
            with self.assertRaises((BackendNotAvailable, RuntimeError)):
                make_relational_backend(self.store, name="voyage")

    def test_auto_with_key_selects_voyage(self):
        """When the extra is installed AND VOYAGE_API_KEY is set, auto picks
        the dense backend. Constructing it does not call the API."""
        with patch.dict(
            os.environ, {"VOYAGE_API_KEY": "test-key-not-real"}, clear=False
        ):
            os.environ.pop("WILLOW_BACKEND", None)
            self.assertEqual(active_backend_name(), "voyage")
            backend = make_relational_backend(self.store)
            # Duck-type: has the query() method that the RelationalBackend
            # Protocol requires. Instance-check would couple to the class.
            self.assertTrue(hasattr(backend, "query"))
            # And it is not a bare VistaBackend.
            self.assertNotIsInstance(backend, VistaBackend)


class ActiveBackendNameTests(unittest.TestCase):
    def test_active_name_defaults_to_sparse_without_extras_or_key(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VOYAGE_API_KEY", None)
            os.environ.pop("WILLOW_BACKEND", None)
            self.assertEqual(active_backend_name(), "sparse")

    def test_active_name_reports_explicit_choice(self):
        self.assertEqual(active_backend_name("sparse"), "sparse")
        self.assertEqual(active_backend_name("default"), "sparse")


if __name__ == "__main__":
    unittest.main()
