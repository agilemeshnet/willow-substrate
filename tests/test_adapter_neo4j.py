"""Tests for the [neo4j] extra's graph adapter.

Skip gracefully when the extra is not installed. When it IS installed but
NEO4J_URI is not set, the constructor still needs to raise Neo4jConfigError
with a specific message; that path is testable without a live database.

A live-database test lives behind a WILLOW_NEO4J_LIVE=1 gate so ordinary
runs never touch a real Neo4j instance.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from willow_substrate.adapters.neo4j import (  # noqa: F401
        Neo4jConfigError,
        Neo4jGraphAdapter,
    )
    HAS_NEO4J_EXTRA = True
except ImportError:
    HAS_NEO4J_EXTRA = False


from willow_substrate.store import EventStore


@unittest.skipUnless(
    HAS_NEO4J_EXTRA,
    "install with pip install \"willow-substrate[neo4j]\" to run these tests",
)
class Neo4jAdapterConfigTests(unittest.TestCase):
    def test_missing_uri_raises_configerror(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NEO4J_URI", None)
            os.environ.pop("NEO4J_PASSWORD", None)
            with self.assertRaises(Neo4jConfigError) as ctx:
                Neo4jGraphAdapter(load_dotenv_file=False)
            message = str(ctx.exception)
            self.assertIn("NEO4J_URI", message)

    def test_missing_password_raises_configerror(self):
        with patch.dict(
            os.environ,
            {"NEO4J_URI": "neo4j+s://example.databases.neo4j.io"},
            clear=False,
        ):
            os.environ.pop("NEO4J_PASSWORD", None)
            with self.assertRaises(Neo4jConfigError) as ctx:
                Neo4jGraphAdapter(load_dotenv_file=False)
            self.assertIn("NEO4J_PASSWORD", str(ctx.exception))

    def test_explicit_args_win_over_env(self):
        """Explicit args must override environment; no attempt to open a
        real connection here since the URI is not reachable, but the
        constructor's arg-precedence contract can be validated up to the
        driver-construction call."""
        with patch.dict(
            os.environ,
            {
                "NEO4J_URI": "neo4j+s://from-env",
                "NEO4J_PASSWORD": "env-secret",
            },
            clear=False,
        ):
            # The neo4j driver's construction accepts these strings without
            # opening a socket; the URI check happens lazily on first query.
            adapter = Neo4jGraphAdapter(
                uri="neo4j+s://from-arg",
                user="explicit-user",
                password="arg-secret",
                load_dotenv_file=False,
            )
            self.assertEqual(adapter.uri, "neo4j+s://from-arg")
            self.assertEqual(adapter.user, "explicit-user")
            self.assertEqual(adapter.password, "arg-secret")
            adapter.close()


@unittest.skipUnless(
    HAS_NEO4J_EXTRA and os.environ.get("WILLOW_NEO4J_LIVE") == "1",
    "set WILLOW_NEO4J_LIVE=1 (plus NEO4J_URI/USER/PASSWORD) to run "
    "against a real Neo4j instance",
)
class Neo4jAdapterLiveTests(unittest.TestCase):
    """Full write-path against a live Neo4j / AuraDB. Opt-in only."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = EventStore(Path(self.temp.name))
        self.adapter = Neo4jGraphAdapter()
        self.adapter.ensure_constraints()

    def tearDown(self):
        try:
            self.adapter.close()
        finally:
            self.temp.cleanup()

    def test_mirror_all_creates_event_and_actor_and_session_nodes(self):
        first = self.store.append("live test 1", actor="peter", session_id="test")
        second = self.store.append("live test 2", actor="willow", session_id="test")
        count = self.adapter.mirror_all(self.store)
        self.assertGreaterEqual(count, 2)


class NoExtraImportGuardTests(unittest.TestCase):
    def test_guard_message_names_the_extras_install(self):
        if HAS_NEO4J_EXTRA:
            import willow_substrate.adapters.neo4j  # noqa: F401
        else:
            with self.assertRaises(ImportError) as ctx:
                import willow_substrate.adapters.neo4j  # noqa: F401
            message = str(ctx.exception)
            self.assertIn("[neo4j]", message)
            self.assertIn("pip install", message)


if __name__ == "__main__":
    unittest.main()
