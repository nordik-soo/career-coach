"""Bug A (part 2) — handler-side registry-gap suppression.

`registry.find_gaps_in_message` is HAS/NEED blind — it matches canonical
and alias text at word boundaries. The original Bug A surfaced in the
chat extractor (Class G ending up in transportation_text instead of
skills[]); the credential patch fixed that. But the registry-gap signal
the router consults runs in parallel and was not corrected.

Repro under Bug A's first-half fix:
  - User types "I have my Class G license, 5 years driving, ..."
  - Credential patch adds "Class G license" to staged.skills as a have-skill
  - But the handler still calls `registry.find_gaps_in_message(user_message)`
    which returns ["Class G driver's license"] because the phrase appears
    verbatim
  - The router sees registry_gap entity + training-action word ("license")
    and fires `rule_2_training_with_entity`
  - Responder narrates a credential gap the user explicitly denied

`_filter_registry_gaps_by_have_skills` is the suppression. These tests
pin its behavior so the same hallucination cannot recur for any
credential the credential patch (or any prior turn) has already landed
on staged.skills.
"""
from __future__ import annotations

import os

os.environ.setdefault("LLM_ENABLED", "false")

import pytest

from skillbridge.chat.handler import _filter_registry_gaps_by_have_skills
from skillbridge.session.staging import StagedSkill

pytestmark = pytest.mark.nodb


# ============================================================================
# §1 — primary suppression
# ============================================================================
def test_class_g_registry_gap_suppressed_when_user_has_it():
    """The exact Bug A part-2 failure case from session d7866f75
    (2026-06-15 15:06): registry surfaces "Class G driver's license"
    because the phrase appears in the user's message; the credential
    patch landed "Class G license" on staged.skills. The two converge
    via canonicalize_skill to "class g license" — the suppression must
    drop the registry gap so the router doesn't fire
    rule_2_training_with_entity."""
    found = ["Class G driver's license"]
    user_skills = [StagedSkill(skill_name="Class G license", source="chat")]
    kept, suppressed = _filter_registry_gaps_by_have_skills(found, user_skills)
    assert kept == []
    assert suppressed == ["Class G driver's license"]


def test_multiple_gaps_some_suppressed_some_kept():
    """User has Class G but NOT Class A. Only Class G should be
    suppressed; Class A remains a legitimate training entity."""
    found = ["Class G driver's license", "commercial driver's license"]
    user_skills = [StagedSkill(skill_name="Class G license", source="chat")]
    kept, suppressed = _filter_registry_gaps_by_have_skills(found, user_skills)
    assert suppressed == ["Class G driver's license"]
    assert "commercial driver's license" in kept


# ============================================================================
# §2 — negation path (no suppression when user doesn't claim the skill)
# ============================================================================
def test_no_suppression_when_user_has_no_class_g_skill():
    """When the credential patch correctly NOT-added Class G (because the
    user said "I don't have a Class G yet"), staged.skills doesn't
    contain it, and the registry gap must remain — producing a true
    training-request intent on that turn."""
    found = ["Class G driver's license"]
    user_skills = [
        StagedSkill(skill_name="customer service", source="chat"),
        StagedSkill(skill_name="defensive driving", source="chat"),
    ]
    kept, suppressed = _filter_registry_gaps_by_have_skills(found, user_skills)
    assert kept == ["Class G driver's license"]
    assert suppressed == []


# ============================================================================
# §3 — canonical equivalence
# ============================================================================
@pytest.mark.parametrize("user_phrasing", [
    "Class G license",
    "Class G",
    "Class G driver's license",
    "G license",
    "class g license",          # lowercase
])
def test_suppression_via_canonical_equivalence(user_phrasing):
    """Different surface forms of the same credential all canonicalize
    to the same form ("class g license") and must suppress the registry
    gap symmetrically — the canonicalize_skill authority is what the
    matcher uses, so the filter must use it too."""
    found = ["Class G driver's license"]
    user_skills = [StagedSkill(skill_name=user_phrasing, source="chat")]
    kept, suppressed = _filter_registry_gaps_by_have_skills(found, user_skills)
    assert kept == []
    assert suppressed == ["Class G driver's license"]


# ============================================================================
# §4 — robustness on malformed input
# ============================================================================
def test_empty_inputs_safe():
    assert _filter_registry_gaps_by_have_skills([], []) == ([], [])


def test_malformed_skill_name_skipped():
    """A StagedSkill with non-string or empty skill_name must not crash
    or produce false suppressions."""
    found = ["Class G driver's license"]
    user_skills = [
        StagedSkill(skill_name="", source="chat"),
        StagedSkill(skill_name="   ", source="chat"),
    ]
    kept, suppressed = _filter_registry_gaps_by_have_skills(found, user_skills)
    assert kept == ["Class G driver's license"]
    assert suppressed == []


def test_malformed_gap_name_skipped():
    """An empty / whitespace-only canonical from the registry must be
    silently dropped — no crash, no addition to either output list."""
    found = ["", "   ", "Class G driver's license"]
    user_skills = []
    kept, suppressed = _filter_registry_gaps_by_have_skills(found, user_skills)
    assert kept == ["Class G driver's license"]
    assert suppressed == []


# ============================================================================
# §5 — preserves order of kept gaps
# ============================================================================
def test_order_of_kept_gaps_matches_input_order():
    """The router and message_understanding pin behavior on first-seen
    order ('the first/strongest entity is the one we narrate'). The
    filter must preserve that order — no sorting, no re-grouping."""
    found = ["WHMIS", "Class G driver's license", "first aid and CPR"]
    user_skills = [StagedSkill(skill_name="Class G license", source="chat")]
    kept, suppressed = _filter_registry_gaps_by_have_skills(found, user_skills)
    assert kept == ["WHMIS", "first aid and CPR"]
    assert suppressed == ["Class G driver's license"]
