"""The durable consent-boundary test.

Anonymous chat must produce ZERO rows in any consent-guarded table.
Consent grant must produce non-zero rows atomically.

If anyone in the future breaks the consent boundary, this test fails first.
"""
from __future__ import annotations

from tests.conftest import count


# ---------------------------------------------------- ANONYMOUS PATH
def test_anonymous_chat_does_not_write_pii(client):
    """Send 10 anonymous chat messages with realistic skill text.
    No profile/skill/chat/match rows may exist afterwards."""
    session_id: str | None = None
    messages = [
        "I worked in retail for two years and I'm comfortable with cash handling and customer service.",
        "I can use POS machines and I helped manage inventory.",
        "I want full-time work in Sault Ste. Marie.",
        "I drove a forklift in a warehouse before moving here.",
        "I have basic Microsoft Excel and computer skills.",
        "I speak English and some Punjabi.",
        "I'm interested in healthcare or warehouse jobs.",
        "I have a Class G driver's license.",
        "I worked as a cook at a small restaurant.",
        "I'm comfortable working evenings or weekends.",
    ]
    for msg in messages:
        body = {"message": msg}
        if session_id:
            body["session_id"] = session_id
        resp = client.post("/v1/chat/messages", json=body)
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert data["requires_consent"] is True
        assert data["profile_id"] is None
        assert data["session_id"] is not None
        session_id = data["session_id"]

    # ---- the contract ----
    assert count("profile.user_profile") == 0, "profile.user_profile must be empty before consent"
    assert count("profile.user_skill") == 0, "profile.user_skill must be empty before consent"
    assert count("interaction.chat_event") == 0, "interaction.chat_event must be empty before consent"
    assert count("interaction.recommendation_feedback") == 0
    assert count("analytics.job_match") == 0, "analytics.job_match must be empty before consent"
    assert count("analytics.job_match_skill") == 0
    assert count("analytics.training_recommendation") == 0


def test_anonymous_chat_extracts_skills_in_memory(client):
    """Session blob accumulates skills across turns without DB writes."""
    body = {"message": "I have customer service, cash handling, and inventory experience."}
    r1 = client.post("/v1/chat/messages", json=body)
    sid = r1.json()["data"]["session_id"]

    body2 = {"message": "I also use Microsoft Excel.", "session_id": sid}
    r2 = client.post("/v1/chat/messages", json=body2)
    assert r2.status_code == 200

    # Still no DB writes.
    assert count("profile.user_profile") == 0
    assert count("profile.user_skill") == 0


# ---------------------------------------------------- CONSENT GRANT
def test_consent_grant_persists_atomically(client):
    """After /v1/consent, profile + skills exist and a token is returned."""
    r1 = client.post(
        "/v1/chat/messages",
        json={"message": "I have customer service, cash handling, and inventory experience."},
    )
    sid = r1.json()["data"]["session_id"]

    assert count("profile.user_profile") == 0
    assert count("profile.user_skill") == 0

    r2 = client.post(
        "/v1/consent",
        json={
            "session_id": sid,
            "consent_purposes": ["profile_storage", "job_recommendation"],
        },
    )
    assert r2.status_code == 200, r2.text
    body = r2.json()["data"]
    assert body["profile_id"]
    assert body["session_token"]
    assert "profile_storage" in body["consent_purposes"]

    assert count("profile.user_profile") == 1
    # At least one extracted skill should land in the table.
    assert count("profile.user_skill") >= 1


def test_consent_grant_requires_profile_storage(client):
    """consent_purposes must include 'profile_storage'."""
    r1 = client.post(
        "/v1/chat/messages",
        json={"message": "I have basic computer skills."},
    )
    sid = r1.json()["data"]["session_id"]

    r2 = client.post(
        "/v1/consent",
        json={"session_id": sid, "consent_purposes": ["job_recommendation"]},
    )
    assert r2.status_code == 400
    assert count("profile.user_profile") == 0


def test_consent_grant_invalidates_staged_session(client):
    """After consent, the session_id should no longer load staged data."""
    r1 = client.post(
        "/v1/chat/messages",
        json={"message": "I have customer service experience."},
    )
    sid = r1.json()["data"]["session_id"]

    r2 = client.post(
        "/v1/consent",
        json={"session_id": sid, "consent_purposes": ["profile_storage"]},
    )
    assert r2.status_code == 200
    profile_count_after = count("profile.user_profile")

    # Reusing the old session_id without the bearer token should start a new
    # anonymous session — it must NOT load the consented profile's data, and
    # it must NOT create another profile row.
    r3 = client.post(
        "/v1/chat/messages",
        json={"message": "I'm also a quick learner.", "session_id": sid},
    )
    assert r3.status_code == 200
    assert r3.json()["data"]["profile_id"] is None
    assert count("profile.user_profile") == profile_count_after


# ---------------------------------------------------- AUTHENTICATED PATH
def test_authenticated_chat_persists(client):
    """With a bearer token, chat writes to chat_event and user_skill."""
    r1 = client.post(
        "/v1/chat/messages",
        json={"message": "I have customer service and cash handling."},
    )
    sid = r1.json()["data"]["session_id"]
    r2 = client.post(
        "/v1/consent",
        json={"session_id": sid, "consent_purposes": ["profile_storage", "job_recommendation"]},
    )
    token = r2.json()["data"]["session_token"]
    pid = r2.json()["data"]["profile_id"]

    pre_events = count("interaction.chat_event")
    r3 = client.post(
        "/v1/chat/messages",
        json={"message": "I also have warehouse experience."},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r3.status_code == 200
    data = r3.json()["data"]
    assert data["profile_id"] == pid
    assert data["requires_consent"] is False

    # Authenticated chat must have written user + assistant events.
    assert count("interaction.chat_event") >= pre_events + 2


def test_invalid_bearer_token_is_treated_as_anonymous(client):
    """Random bogus token must not grant access — falls through to anonymous."""
    r = client.post(
        "/v1/chat/messages",
        json={"message": "I have customer service skills."},
        headers={"Authorization": "Bearer not-a-real-token"},
    )
    assert r.status_code == 200
    assert r.json()["data"]["profile_id"] is None
    assert count("profile.user_profile") == 0
