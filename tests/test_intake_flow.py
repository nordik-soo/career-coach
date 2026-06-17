"""PR 10 invariant tests — guided intake + evidence-bound extractor.

These tests assume LLM_ENABLED=false (set by conftest). They exercise:
  - Evidence-bound extractor drops ungrounded slots / hallucinated skills.
  - Off-topic messages don't pollute the staged profile.
  - Declined slots are not re-asked.
  - Profile-completeness gating — matches are not shown until enough slots
    are filled (target_role + at least MIN_SLOTS_FOR_MATCH match-ready
    slots).
  - Consent flush persists the four new PR 10 fields.

When LLM_ENABLED is off, the rule-based skill extractor still runs, so
"I have customer service and warehouse experience" still yields skills.
The evidence-bound LLM extractor itself is exercised via monkeypatch of
skillbridge.llm.call_json — see test_extractor_drops_*.
"""
from __future__ import annotations

import pytest

from skillbridge.chat import extractor as chat_extractor
from skillbridge.chat import intake_priority, intake_state
from skillbridge.session.staging import StagedProfile, StagedSkill
from tests.conftest import count


# =========================================================================
# Evidence-bound extractor (unit tests; patch call_json directly)
# =========================================================================
def test_extractor_drops_ungrounded_field(monkeypatch):
    """LLM says 'preferred_location: Sault Ste. Marie' but evidence isn't
    in the user message. Field must be dropped."""
    monkeypatch.setattr(chat_extractor, "is_enabled", lambda: True)

    def fake_call_json(*args, **kwargs):
        return {
            "fields": {
                "preferred_location": {
                    "value": "Sault Ste. Marie",
                    "evidence": "this phrase is not in the message",
                }
            },
            "skills": [],
        }
    monkeypatch.setattr(chat_extractor, "call_json", fake_call_json)

    result = chat_extractor.extract("I am looking for warehouse work.", asked_slots=[])
    assert "preferred_location" not in result.fields
    assert any("ungrounded" in d for d in result.raw_keys_dropped)


def test_extractor_drops_hallucinated_skill(monkeypatch):
    """User said nothing about Python — LLM hallucinates it. Must be dropped."""
    monkeypatch.setattr(chat_extractor, "is_enabled", lambda: True)

    def fake_call_json(*args, **kwargs):
        return {
            "fields": {},
            "skills": [
                {"name": "Python", "evidence": "fictional grounding", "confidence": 0.9},
            ],
        }
    monkeypatch.setattr(chat_extractor, "call_json", fake_call_json)

    result = chat_extractor.extract("I have warehouse experience.", asked_slots=[])
    assert all(s.skill_name.lower() != "python" for s in result.skills)


def test_extractor_keeps_grounded_field(monkeypatch):
    """Evidence appears verbatim in the message — keep the field."""
    monkeypatch.setattr(chat_extractor, "is_enabled", lambda: True)

    def fake_call_json(*args, **kwargs):
        return {
            "fields": {
                "target_role_text": {
                    "value": "warehouse manager",
                    "evidence": "warehouse manager",
                }
            },
            "skills": [],
        }
    monkeypatch.setattr(chat_extractor, "call_json", fake_call_json)

    result = chat_extractor.extract(
        "I'm looking for a warehouse manager job.", asked_slots=[],
    )
    assert result.fields.get("target_role_text") == "warehouse manager"


def test_extractor_short_evidence_rejected(monkeypatch):
    """Evidence shorter than 4 chars cannot ground a field."""
    monkeypatch.setattr(chat_extractor, "is_enabled", lambda: True)

    def fake_call_json(*args, **kwargs):
        return {
            "fields": {"target_role_text": {"value": "anything", "evidence": "I"}},
            "skills": [],
        }
    monkeypatch.setattr(chat_extractor, "call_json", fake_call_json)

    result = chat_extractor.extract("I'm here.", asked_slots=[])
    assert "target_role_text" not in result.fields


# =========================================================================
# Decline detection
# =========================================================================
def test_decline_salary_pattern():
    result = chat_extractor.extract(
        "I'd rather not say my salary expectation.", asked_slots=[],
    )
    assert "salary_expectation_text" in result.declined


def test_blanket_decline_targets_asked_slots():
    """User says 'skip that' after being asked about salary -> salary declined."""
    result = chat_extractor.extract(
        "skip that",
        asked_slots=["salary_expectation_text", "shift_preference"],
    )
    # Both asked slots should be declined since user gave nothing else.
    assert "salary_expectation_text" in result.declined
    assert "shift_preference" in result.declined


def test_blanket_decline_only_uses_last_turn_not_history(client):
    """End-to-end: 'skip that' after several turns of history must only
    decline what the assistant asked on the most recent turn, not every
    slot ever asked about."""
    from skillbridge.session import get_store

    # First turn: send a fresh message, the handler will ask some questions.
    r1 = client.post(
        "/v1/chat/messages",
        json={"message": "I have customer service experience."},
    )
    sid = r1.json()["data"]["session_id"]

    # Manually walk the staged profile through several historical asks so
    # asked_slots and last_asked_slots diverge.
    store = get_store()
    staged = store.load(sid)
    assert staged is not None
    staged.asked_slots = [
        "target_role_text", "work_type_preference", "experience_text",
        "shift_preference", "preferred_location",
    ]
    # Only the LAST turn asked about salary.
    staged.last_asked_slots = ["salary_expectation_text"]
    sid = store.save(staged)

    # User says "skip that" — should ONLY decline salary, not the 5 historical.
    r2 = client.post(
        "/v1/chat/messages",
        json={"message": "skip that", "session_id": sid},
    )
    assert r2.status_code == 200
    sid = r2.json()["data"]["session_id"]

    after = store.load(sid)
    assert after is not None
    assert "salary_expectation_text" in after.declined_slots
    # The historical asks must NOT have been auto-declined.
    for slot in ("target_role_text", "work_type_preference",
                 "experience_text", "shift_preference", "preferred_location"):
        assert slot not in after.declined_slots, (
            f"{slot} should not be auto-declined from blanket 'skip that'"
        )


# =========================================================================
# Off-topic detection — chitchat does not advance the profile
# =========================================================================
def test_off_topic_message_is_flagged():
    """Pure greeting with no profile data: extractor returns off_topic."""
    result = chat_extractor.extract("hello there", asked_slots=[])
    assert result.off_topic is True
    assert not result.fields
    assert not result.skills
    assert not result.declined


# =========================================================================
# Intake priority — declined slots never re-asked, asked slots deprioritised
# =========================================================================
def test_pick_slots_skips_declined():
    staged = StagedProfile.new("sess-1")
    staged.target_role_text = "warehouse manager"
    staged.declined_slots = ["salary_expectation_text"]
    picked = intake_priority.pick_slots_to_ask(staged, max_n=5)
    assert "salary_expectation_text" not in picked


def test_pick_slots_skips_already_filled():
    staged = StagedProfile.new("sess-2")
    staged.target_role_text = "warehouse manager"
    staged.work_type_preference = "full-time"
    picked = intake_priority.pick_slots_to_ask(staged, max_n=5)
    assert "target_role_text" not in picked
    assert "work_type_preference" not in picked


def test_pick_slots_prefers_unasked_over_asked():
    staged = StagedProfile.new("sess-3")
    staged.asked_slots = ["target_role_text"]
    picked = intake_priority.pick_slots_to_ask(staged, max_n=1)
    # Should pick something other than the already-asked slot if possible.
    assert picked[0] != "target_role_text"


# =========================================================================
# State machine — completeness gating
# =========================================================================
def test_state_machine_low_completeness_asks_questions():
    staged = StagedProfile.new("sess-4")
    decision = intake_state.decide(
        staged, off_topic=False, extracted_anything=False,
        declined_this_turn=[], authenticated=False,
    )
    assert decision.action == intake_state.ACTION_ASK_QUESTIONS
    assert decision.show_matches is False


def test_state_machine_ready_completeness_presents_matches():
    """5 match-ready slots filled -> band='ready' -> PRESENT_MATCHES."""
    staged = StagedProfile.new("sess-5")
    staged.target_role_text = "warehouse manager"
    staged.work_type_preference = "full-time"
    staged.experience_text = "3 years warehouse"
    staged.shift_preference = "days"
    staged.preferred_location = "Sault Ste. Marie"
    staged.skills = [StagedSkill(skill_name="forklift operation")]
    decision = intake_state.decide(
        staged, off_topic=False, extracted_anything=True,
        declined_this_turn=[], authenticated=False,
    )
    assert decision.action == intake_state.ACTION_PRESENT_MATCHES
    assert decision.show_matches is True


def test_state_machine_off_topic_does_not_advance():
    """Off-topic preserves state and emits REDIRECT."""
    staged = StagedProfile.new("sess-6")
    staged.intake_state = intake_state.STATE_INTAKE_COLLECTING
    decision = intake_state.decide(
        staged, off_topic=True, extracted_anything=False,
        declined_this_turn=[], authenticated=False,
    )
    assert decision.action == intake_state.ACTION_REDIRECT
    assert decision.next_state == intake_state.STATE_INTAKE_COLLECTING
    assert decision.show_matches is False


def test_state_machine_decline_only_acknowledges():
    staged = StagedProfile.new("sess-7")
    decision = intake_state.decide(
        staged, off_topic=False, extracted_anything=False,
        declined_this_turn=["salary_expectation_text"],
        authenticated=False,
    )
    assert decision.action == intake_state.ACTION_ACKNOWLEDGE_AND_WAIT
    assert decision.show_matches is False


# =========================================================================
# End-to-end: chat does not present matches until profile is ready
# =========================================================================
def test_chat_holds_matches_until_enough_slots(client):
    """First turn (only skills, no target role) must not present matches.
    Even though the rule-based extractor will pick up 'customer service',
    target_role_text isn't filled -> band='low' -> ASK_QUESTIONS."""
    r = client.post(
        "/v1/chat/messages",
        json={"message": "I have customer service experience."},
    )
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["requires_consent"] is True
    # No matches shown — gating works.
    assert data["recommended_jobs"] == []
    assert data["next_action"] == intake_state.ACTION_ASK_QUESTIONS


def test_chat_off_topic_does_not_create_profile_data(client):
    """Pure off-topic chitchat -> no DB writes."""
    r = client.post("/v1/chat/messages", json={"message": "hello"})
    assert r.status_code == 200
    assert count("profile.user_profile") == 0
    assert count("profile.user_skill") == 0


# =========================================================================
# Consent flush persists new PR 10 fields
# =========================================================================
def test_consent_persists_new_fields(client, monkeypatch):
    """Stage a profile with the four new fields and verify they land in
    profile.user_profile on consent."""
    # Use the extractor's grounded path by going through chat. Since LLM is
    # off in tests, manually seed the session via two messages whose
    # rule-based extractor produces skills, then patch the staged blob.
    from skillbridge.session import get_store

    r1 = client.post(
        "/v1/chat/messages",
        json={"message": "I want warehouse manager work, full-time, days shift, in Sault Ste. Marie."},
    )
    sid = r1.json()["data"]["session_id"]

    # Hand-fill the four new fields directly on the staged blob (LLM is off,
    # so the evidence-bound path can't extract them in tests).
    store = get_store()
    staged = store.load(sid)
    assert staged is not None
    staged.target_role_text = "warehouse manager"
    staged.salary_expectation_text = "around $20/hr"
    staged.shift_preference = "days"
    staged.transportation_text = "own car"
    staged.availability_text = "start immediately"
    sid = store.save(staged)

    r2 = client.post(
        "/v1/consent",
        json={"session_id": sid, "consent_purposes": ["profile_storage"]},
    )
    assert r2.status_code == 200, r2.text

    # Confirm the four new fields are on the persisted row.
    from skillbridge.db import sync_cursor
    with sync_cursor() as cur:
        cur.execute(
            """
            SELECT salary_expectation_text, shift_preference,
                   transportation_text, availability_text
              FROM profile.user_profile
             LIMIT 1
            """
        )
        row = cur.fetchone()
    assert row["salary_expectation_text"] == "around $20/hr"
    assert row["shift_preference"] == "days"
    assert row["transportation_text"] == "own car"
    assert row["availability_text"] == "start immediately"
