"""Shared pytest fixtures.

The consent-boundary test runs against a real Postgres but uses:
- the signed-cookie session store (no Redis needed)
- LLM_ENABLED=false (no Anthropic needed)

Set TEST_PGDATABASE in your env to a throwaway DB. The fixtures truncate
the relevant tables before each test.
"""
from __future__ import annotations

import os
import secrets

# Force test config BEFORE importing the app.
os.environ.setdefault("SESSION_STORE", "cookie")
os.environ.setdefault("SESSION_COOKIE_SECRET", secrets.token_urlsafe(32))
os.environ.setdefault("LLM_ENABLED", "false")
os.environ.setdefault("ANTHROPIC_API_KEY", "")
os.environ.setdefault("PGDATABASE", os.environ.get("TEST_PGDATABASE", "skillbridge_test"))
# Admin auth: deterministic test key, always required so the route guard
# test can confirm 401 for unauthenticated admin calls.
os.environ.setdefault("REQUIRE_ADMIN_AUTH", "true")
os.environ.setdefault("ADMIN_API_KEY", "test-admin-key-do-not-use-in-prod")

import pytest
from fastapi.testclient import TestClient

import api as api_module
from skillbridge.db import sync_cursor


CONSENT_GUARDED_TABLES = [
    "profile.user_profile",
    "profile.user_skill",
    "interaction.chat_event",
    "interaction.recommendation_feedback",
    "analytics.job_match",
    "analytics.job_match_skill",
    "analytics.training_recommendation",
]


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "nodb: test does not need a database. Skips the autouse DB truncate, "
        "so the test runs even when Postgres is unreachable.",
    )


@pytest.fixture(autouse=True)
def _clean_db(request):
    """Truncate every consent-guarded table before each test.

    Tests that don't touch the DB at all (pure string/logic checks) can
    opt out with `@pytest.mark.nodb` or a module-level
    `pytestmark = pytest.mark.nodb`.
    """
    if request.node.get_closest_marker("nodb"):
        yield
        return
    with sync_cursor() as cur:
        cur.execute(
            "TRUNCATE "
            + ", ".join(CONSENT_GUARDED_TABLES)
            + " RESTART IDENTITY CASCADE"
        )
    yield


@pytest.fixture
def client() -> TestClient:
    return TestClient(api_module.app)


def count(table: str) -> int:
    with sync_cursor() as cur:
        cur.execute(f"SELECT COUNT(*) AS n FROM {table}")
        return int(cur.fetchone()["n"])
