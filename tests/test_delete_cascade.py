"""Profile deletion contract — must clear every PII-bearing row.

After PR 3, DELETE operates on `/v1/profiles/me` with a bearer token.
The endpoint never accepts a profile_id from the path; identity is
resolved from the token.

Tests:

1. test_delete_clears_all_pii — seed rows in every profile-linked table,
   call DELETE /v1/profiles/me with the consent-grant token, assert
   hard-deleted rows are gone and the profile shell has all PII / consent /
   token fields cleared.

2. test_delete_without_token_returns_401 — DELETE without auth must be
   blocked by require_profile, not reach the handler.

3. test_token_invalidated_after_delete — after a successful delete, the
   same token must no longer authenticate. require_profile filters by
   deleted_at IS NULL, so the token resolves to no row and returns 401.

4. test_no_orphan_profile_id_fks — schema scan that asserts every foreign
   key referencing profile.user_profile is in the known delete-chain set.
"""
from __future__ import annotations

import pytest

from skillbridge.db import sync_cursor
from tests.conftest import count


# ---------------------------------------------------------------- Fixtures
@pytest.fixture
def clean_synthetic_core():
    """Tests that insert synthetic rows into core.* clean them up here.

    PR 7A: source must be in core.approved_job_source, so tests use
    'partner_csv' (approved) and identify their rows by a source_job_id
    pattern instead of a synthetic source value.
    """
    yield
    with sync_cursor() as cur:
        cur.execute("DELETE FROM core.job_posting WHERE source_job_id LIKE 'test_%'")
        cur.execute("DELETE FROM core.training_resource WHERE provider = 'Test Delete Provider'")


def _consented_profile(client) -> tuple[str, str]:
    """Drive the chat + consent flow; return (profile_id, raw token)."""
    r1 = client.post(
        "/v1/chat/messages",
        json={"message": "I have customer service, cash handling and inventory experience."},
    )
    sid = r1.json()["data"]["session_id"]
    r2 = client.post(
        "/v1/consent",
        json={
            "session_id": sid,
            "consent_purposes": ["profile_storage", "job_recommendation",
                                 "training_recommendation"],
        },
    )
    body = r2.json()["data"]
    return body["profile_id"], body["session_token"]


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _seed_all_profile_linked_rows(profile_id: str) -> None:
    """Insert at least one row into every profile-linked table.

    Uses direct SQL so the test does not depend on the match engine or LLM
    to populate dependent tables. Also sets the PR 10 intake fields on
    the profile row so the delete assertions can prove they got scrubbed
    (not just that they were never set).
    """
    with sync_cursor() as cur:
        # PR 10 + Sprint 1 fields aren't filled by the consent flow in
        # LLM-off tests, so set them here. If delete forgets to scrub one,
        # the assertion below catches it.
        cur.execute(
            """
            UPDATE profile.user_profile SET
                salary_expectation_text = 'around $20/hr',
                shift_preference        = 'days',
                transportation_text     = 'own car',
                availability_text       = 'start immediately',
                resume_text             = 'Software Engineer at Acme...',
                resume_filename         = 'resume.pdf',
                resume_parsed_at        = NOW(),
                resume_facts_json       = '{"version": 1, "skills": []}'::jsonb
             WHERE profile_id = %s
            """,
            (profile_id,),
        )
        cur.execute(
            """
            INSERT INTO core.job_posting (source, source_job_id, title, posted_date, is_active)
            VALUES ('partner_csv', %s, 'Test Job', CURRENT_DATE, TRUE)
            ON CONFLICT (source, source_job_id) DO UPDATE SET is_active = TRUE
            RETURNING job_id
            """,
            (f"test_{profile_id}",),
        )
        job_id = cur.fetchone()["job_id"]
        cur.execute(
            """
            INSERT INTO core.training_resource (provider, title, is_active)
            VALUES ('Test Delete Provider', %s, TRUE)
            ON CONFLICT (provider, title) DO UPDATE SET is_active = TRUE
            RETURNING resource_id
            """,
            (f"Test Resource {profile_id}",),
        )
        resource_id = cur.fetchone()["resource_id"]
        cur.execute(
            "INSERT INTO profile.user_skill (profile_id, skill_name, source, confidence) "
            "VALUES (%s, 'manual test skill', 'manual_update', 0.9) "
            "ON CONFLICT (profile_id, skill_name) DO NOTHING",
            (profile_id,),
        )
        cur.execute(
            "INSERT INTO interaction.chat_event (profile_id, role, message_text) "
            "VALUES (%s, 'user', 'seeded message')",
            (profile_id,),
        )
        cur.execute(
            """
            INSERT INTO analytics.job_match
                (profile_id, job_id, match_score, match_band, match_eligible,
                 matched_skills_count, required_skills_count, missing_skills_count,
                 engine_version)
            VALUES (%s, %s, 0.75, 'strong', TRUE, 3, 4, 1, 'test-v1')
            RETURNING match_id
            """,
            (profile_id, job_id),
        )
        match_id = cur.fetchone()["match_id"]
        cur.execute(
            "INSERT INTO analytics.job_match_skill (match_id, skill_name, status) "
            "VALUES (%s, 'manual test skill', 'matched')",
            (match_id,),
        )
        cur.execute(
            "INSERT INTO interaction.recommendation_feedback (profile_id, job_id, action) "
            "VALUES (%s, %s, 'saved')",
            (profile_id, job_id),
        )
        cur.execute(
            """
            INSERT INTO analytics.training_recommendation
                (profile_id, resource_id, skill_name, engine_version)
            VALUES (%s, %s, 'manual test skill', 'test-v1')
            """,
            (profile_id, resource_id),
        )


def _count_for_profile(table: str, profile_id: str, column: str = "profile_id") -> int:
    with sync_cursor() as cur:
        cur.execute(f"SELECT COUNT(*) AS n FROM {table} WHERE {column} = %s", (profile_id,))
        return int(cur.fetchone()["n"])


# ============================================================== TESTS
def test_delete_clears_all_pii(client, clean_synthetic_core):
    """End-to-end: seed PII in every table, DELETE /me with token, assert clean."""
    pid, token = _consented_profile(client)
    _seed_all_profile_linked_rows(pid)

    # Sanity: every profile-linked table has at least one row.
    assert _count_for_profile("profile.user_skill", pid) >= 1
    assert _count_for_profile("interaction.chat_event", pid) >= 1
    assert _count_for_profile("interaction.recommendation_feedback", pid) >= 1
    assert _count_for_profile("analytics.job_match", pid) >= 1
    assert _count_for_profile("analytics.training_recommendation", pid) >= 1
    with sync_cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) AS n FROM analytics.job_match_skill s "
            "JOIN analytics.job_match m ON m.match_id = s.match_id "
            "WHERE m.profile_id = %s",
            (pid,),
        )
        assert cur.fetchone()["n"] >= 1

    # ---- Delete via /me + bearer token ----
    resp = client.delete("/v1/profiles/me", headers=_auth(token))
    assert resp.status_code == 200, resp.text

    # ---- All protected tables must be empty for this profile ----
    assert _count_for_profile("profile.user_skill", pid) == 0
    assert _count_for_profile("interaction.chat_event", pid) == 0
    assert _count_for_profile("interaction.recommendation_feedback", pid) == 0
    assert _count_for_profile("analytics.job_match", pid) == 0
    assert _count_for_profile("analytics.training_recommendation", pid) == 0
    # job_match_skill should cascade away when job_match rows are gone.
    with sync_cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) AS n FROM analytics.job_match_skill s "
            "JOIN analytics.job_match m ON m.match_id = s.match_id "
            "WHERE m.profile_id = %s",
            (pid,),
        )
        assert cur.fetchone()["n"] == 0

    # ---- Profile shell remains, with all PII fields cleared ----
    with sync_cursor() as cur:
        cur.execute(
            """
            SELECT profile_id, created_at, deleted_at,
                   session_id, preferred_location, target_role_text,
                   education_text, experience_text, skills_text,
                   work_type_preference, language_preferences,
                   salary_expectation_text, shift_preference,
                   transportation_text, availability_text,
                   resume_text, resume_filename, resume_parsed_at,
                   resume_facts_json,
                   consent_version, consent_purposes, consent_granted_at,
                   session_token_hash, session_token_expires_at
              FROM profile.user_profile WHERE profile_id = %s
            """,
            (pid,),
        )
        row = cur.fetchone()

    assert row is not None, "Profile shell must be retained for audit"
    assert row["created_at"] is not None
    assert row["deleted_at"] is not None
    assert row["session_id"] is None
    assert row["preferred_location"] is None
    assert row["target_role_text"] is None
    assert row["education_text"] is None
    assert row["experience_text"] is None
    assert row["skills_text"] is None
    assert row["work_type_preference"] is None
    assert (row["language_preferences"] or []) == []
    # PR 10 intake fields must also be scrubbed.
    assert row["salary_expectation_text"] is None
    assert row["shift_preference"] is None
    assert row["transportation_text"] is None
    assert row["availability_text"] is None
    # Sprint 1 resume fields must also be scrubbed.
    assert row["resume_text"] is None
    assert row["resume_filename"] is None
    assert row["resume_parsed_at"] is None
    assert row["resume_facts_json"] is None
    assert row["consent_version"] is None
    assert (row["consent_purposes"] or []) == []
    assert row["consent_granted_at"] is None
    assert row["session_token_hash"] is None
    assert row["session_token_expires_at"] is None


def test_delete_without_token_returns_401(client):
    """No bearer = no identity = 401, never reaches the delete handler."""
    resp = client.delete("/v1/profiles/me")
    assert resp.status_code == 401


def test_delete_with_invalid_token_returns_401(client):
    """Garbage token = no resolvable profile = 401."""
    resp = client.delete(
        "/v1/profiles/me",
        headers={"Authorization": "Bearer not-a-real-token-at-all"},
    )
    assert resp.status_code == 401


def test_token_invalidated_after_delete(client):
    """After a successful delete, the same token must no longer authenticate.

    require_profile filters by deleted_at IS NULL, so a soft-deleted profile's
    token resolves to nothing — every subsequent call with that token is 401.
    This is the new equivalent of the old "idempotent 404 on second call":
    the user can't accidentally double-delete because the token is invalidated.
    """
    _pid, token = _consented_profile(client)

    r1 = client.delete("/v1/profiles/me", headers=_auth(token))
    assert r1.status_code == 200

    # The token now points at a soft-deleted profile; any /me call must 401.
    r2 = client.delete("/v1/profiles/me", headers=_auth(token))
    assert r2.status_code == 401

    r3 = client.get("/v1/profiles/me", headers=_auth(token))
    assert r3.status_code == 401


# ----------------------------------------- Schema scan: durable guard
def test_no_orphan_profile_id_fks():
    """Every FK referencing profile.user_profile must be in the delete chain.

    If a new table that references profile.user_profile is added in the
    future, this test fails until:
      1. The DELETE endpoint in routes/profiles.py is updated to clear it.
      2. The expected_tables set below is updated.
    """
    expected_tables = {
        "profile.user_skill",
        "interaction.chat_event",
        "interaction.recommendation_feedback",
        "analytics.job_match",
        "analytics.training_recommendation",
    }
    with sync_cursor() as cur:
        cur.execute(
            """
            SELECT n.nspname || '.' || c.relname AS table_name
              FROM pg_constraint con
              JOIN pg_class c ON c.oid = con.conrelid
              JOIN pg_namespace n ON n.oid = c.relnamespace
             WHERE con.contype = 'f'
               AND con.confrelid = 'profile.user_profile'::regclass
            """
        )
        found = {r["table_name"] for r in cur.fetchall()}

    unexpected = found - expected_tables
    assert not unexpected, (
        f"Tables {sorted(unexpected)} reference profile.user_profile but are "
        f"NOT in the delete chain. Update routes/profiles.py:delete_profile "
        f"and tests/test_delete_cascade.py:expected_tables."
    )

    missing = expected_tables - found
    assert not missing, (
        f"Tables {sorted(missing)} are in the expected delete chain but no "
        f"FK to profile.user_profile exists in the live schema. Schema and "
        f"test are out of sync."
    )
