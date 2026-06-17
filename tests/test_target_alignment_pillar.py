"""Fresh-intake-on-target-change pillar (2026-06-15) — unit tests.

When a user switches target mid-session (e.g. "I want to be an
accounting clerk" → skills intake → matches → "I want to be a truck
driver"), the prior target's skill/experience evidence still lives on
`staged.skills` / `staged.experience_text`. Without this pillar, the
engine would run on stale-target evidence and surface either:
  - cross-NOC bleed (closed by Bug B's same-NOC-family gate), or
  - a misleading "no opportunity" decision based on incomplete intake.

The pillar:
  - Adds `skills_collected_for_target` and `experience_collected_for_target`
    to StagedProfile, stamped on merge.
  - Computes `skills_aligned_with_target` / `experience_aligned_with_target`
    in build_truth_summary using casefold-normalized comparison.
  - Forces the arbiter pass 1 to `ask_one_clarifying_question` on the
    first misaligned slot (skills first, then experience) before the
    engine can run.

These tests pin the lifecycle, the alignment computation, and the
arbiter gate behaviour.
"""
from __future__ import annotations

import os

os.environ.setdefault("LLM_ENABLED", "false")

import pytest

from skillbridge.chat.arbiter import (
    ArbiterDecision,
    RunEngine,
    validate_planner_intent,
)
from skillbridge.chat.planner import PlannerDecision
from skillbridge.chat.truth_summary import (
    _compute_target_alignment,
    _normalize_target_role,
    build_truth_summary,
)
from skillbridge.session.staging import StagedProfile, StagedSkill

pytestmark = pytest.mark.nodb


# ============================================================================
# §1 — _normalize_target_role
# ============================================================================
@pytest.mark.parametrize("inp,expected", [
    (None, None),
    ("", None),
    ("   ", None),
    ("truck driver", "truck driver"),
    ("Truck Driver", "truck driver"),
    ("TRUCK DRIVER", "truck driver"),
    ("  truck   driver  ", "truck driver"),    # collapse whitespace
    ("Truck-Driver", "truck-driver"),           # hyphen kept (different evidence)
    ("long-haul truck driver", "long-haul truck driver"),  # distinct from "truck driver"
])
def test_normalize_target_role(inp, expected):
    assert _normalize_target_role(inp) == expected


# ============================================================================
# §2 — lifecycle: merge_skills stamps the alignment field
# ============================================================================
def test_merge_skills_stamps_alignment_when_new_skill_added():
    sp = StagedProfile.new("s1")
    sp.target_role_text = "accounting clerk"
    assert sp.skills_collected_for_target is None

    added = sp.merge_skills([
        StagedSkill(skill_name="bank reconciliation", source="chat"),
    ])
    assert added == 1
    assert sp.skills_collected_for_target == "accounting clerk"


def test_merge_skills_no_stamp_when_only_confidence_upgrade():
    """Confidence-only upgrade on an already-present skill is NOT new
    evidence against the new target. The alignment field stays at its
    prior value."""
    sp = StagedProfile.new("s1")
    sp.target_role_text = "accounting clerk"
    sp.merge_skills([StagedSkill(skill_name="excel", confidence=0.5, source="chat")])
    assert sp.skills_collected_for_target == "accounting clerk"

    # Switch target. Field stays pointed at "accounting clerk".
    sp.target_role_text = "truck driver"
    assert sp.skills_collected_for_target == "accounting clerk"

    # Confidence upgrade only — no new key.
    added = sp.merge_skills([StagedSkill(skill_name="excel", confidence=0.9, source="chat")])
    assert added == 0
    # Stamp does NOT shift — user hasn't actually re-affirmed for the new target.
    assert sp.skills_collected_for_target == "accounting clerk"


def test_target_role_change_does_not_clear_alignment_field():
    """The whole point: target_role_text changes but the alignment
    field keeps pointing at the PRIOR target. The mismatch is the
    load-bearing signal."""
    sp = StagedProfile.new("s1")
    sp.target_role_text = "accounting clerk"
    sp.merge_skills([StagedSkill(skill_name="bank reconciliation", source="chat")])

    sp.target_role_text = "truck driver"
    assert sp.skills_collected_for_target == "accounting clerk"
    assert sp.target_role_text == "truck driver"


# ============================================================================
# §3 — lifecycle: merge_fields stamps experience_collected_for_target
# ============================================================================
def test_merge_fields_stamps_experience_alignment():
    sp = StagedProfile.new("s1")
    sp.target_role_text = "accounting clerk"
    assert sp.experience_collected_for_target is None

    sp.merge_fields({"experience_text": "3 years bookkeeping"})
    assert sp.experience_text == "3 years bookkeeping"
    assert sp.experience_collected_for_target == "accounting clerk"


def test_merge_fields_does_not_stamp_when_experience_empty():
    sp = StagedProfile.new("s1")
    sp.target_role_text = "accounting clerk"
    sp.merge_fields({"experience_text": ""})    # empty → skipped by guard
    assert sp.experience_collected_for_target is None


def test_merge_fields_other_slots_dont_stamp_experience():
    """Only experience_text setting flips the experience alignment.
    Setting preferred_location or shift_preference must not touch it."""
    sp = StagedProfile.new("s1")
    sp.target_role_text = "accounting clerk"
    sp.merge_fields({"preferred_location": "Sault Ste. Marie"})
    assert sp.experience_collected_for_target is None


def test_direct_setattr_on_experience_text_stamps_alignment():
    """Live-2026-06-16 repro: fallback_fill / closed_vocab_fill paths
    set staged.experience_text directly via setattr, bypassing
    merge_fields. The stamp lives in __setattr__ so every path covers
    it uniformly. Without this, the alignment gate would keep firing
    even after experience was filled."""
    sp = StagedProfile.new("s1")
    sp.target_role_text = "accounting clerk"
    sp.experience_text = "3 years bookkeeping at a small business"
    assert sp.experience_collected_for_target == "accounting clerk"


def test_setattr_on_empty_experience_text_does_not_stamp():
    """Setting experience_text to None or empty string must not stamp.
    Stamp represents collected evidence — empty isn't evidence."""
    sp = StagedProfile.new("s1")
    sp.target_role_text = "accounting clerk"
    sp.experience_text = ""
    assert sp.experience_collected_for_target is None
    sp.experience_text = None
    assert sp.experience_collected_for_target is None
    sp.experience_text = "   "
    assert sp.experience_collected_for_target is None


# ============================================================================
# §4 — _compute_target_alignment
# ============================================================================
def test_alignment_when_target_unresolved_is_always_true():
    """Cold profile, no target set yet. Alignment is True regardless of
    the field values. The other intake gates handle the cold path."""
    sp = StagedProfile.new("s1")
    # target_role_text=None
    s_ok, e_ok, slot = _compute_target_alignment(sp)
    assert s_ok is True and e_ok is True and slot is None


def test_alignment_when_cold_with_target_set_misaligned_on_skills_first():
    """User states target but has never provided skills or experience.
    Skills get asked first per locked priority."""
    sp = StagedProfile.new("s1")
    sp.target_role_text = "truck driver"
    s_ok, e_ok, slot = _compute_target_alignment(sp)
    assert s_ok is False
    assert e_ok is False
    assert slot == "skills_text"   # skills always asked first


def test_alignment_after_target_switch_skills_misaligned():
    """The Bug-B-deeper repro: user had skills for accounting, switches
    to truck driver, skill alignment now mismatches."""
    sp = StagedProfile.new("s1")
    sp.target_role_text = "accounting clerk"
    sp.merge_skills([StagedSkill(skill_name="bank reconciliation", source="chat")])
    sp.merge_fields({"experience_text": "3 years bookkeeping"})

    sp.target_role_text = "truck driver"
    s_ok, e_ok, slot = _compute_target_alignment(sp)
    assert s_ok is False
    assert e_ok is False
    assert slot == "skills_text"


def test_alignment_skills_aligned_experience_still_misaligned():
    """User switched target, then provided skills for the new target,
    but experience still belongs to the prior target."""
    sp = StagedProfile.new("s1")
    sp.target_role_text = "accounting clerk"
    sp.merge_skills([StagedSkill(skill_name="excel", source="chat")])
    sp.merge_fields({"experience_text": "3 years bookkeeping"})

    sp.target_role_text = "truck driver"
    sp.merge_skills([StagedSkill(skill_name="class g license", source="chat")])

    s_ok, e_ok, slot = _compute_target_alignment(sp)
    assert s_ok is True
    assert e_ok is False
    assert slot == "experience_text"


def test_alignment_both_aligned_after_full_re_intake():
    sp = StagedProfile.new("s1")
    sp.target_role_text = "accounting clerk"
    sp.merge_skills([StagedSkill(skill_name="excel", source="chat")])
    sp.merge_fields({"experience_text": "3 years bookkeeping"})

    sp.target_role_text = "truck driver"
    sp.merge_skills([StagedSkill(skill_name="class g license", source="chat")])
    sp.merge_fields({"experience_text": "5 years driving"})

    s_ok, e_ok, slot = _compute_target_alignment(sp)
    assert s_ok is True
    assert e_ok is True
    assert slot is None


def test_alignment_case_insensitive():
    """Canonical comparison casefolds — 'Truck Driver' vs 'truck driver'
    must be considered the same target."""
    sp = StagedProfile.new("s1")
    sp.target_role_text = "Truck Driver"
    sp.merge_skills([StagedSkill(skill_name="class g license", source="chat")])
    sp.merge_fields({"experience_text": "5 years driving"})

    sp.target_role_text = "truck driver"   # case change only
    s_ok, e_ok, _ = _compute_target_alignment(sp)
    assert s_ok is True
    assert e_ok is True


def test_alignment_distinguishes_specific_variants():
    """`long-haul truck driver` and `truck driver` are different
    evidence per locked design — alignment must reject the match."""
    sp = StagedProfile.new("s1")
    sp.target_role_text = "truck driver"
    sp.merge_skills([StagedSkill(skill_name="class g license", source="chat")])
    sp.merge_fields({"experience_text": "5 years driving"})

    sp.target_role_text = "long-haul truck driver"
    s_ok, e_ok, _ = _compute_target_alignment(sp)
    assert s_ok is False
    assert e_ok is False


# ============================================================================
# §5 — truth summary integration
# ============================================================================
def test_truth_summary_carries_alignment_fields():
    sp = StagedProfile.new("s1")
    sp.target_role_text = "accounting clerk"
    sp.merge_skills([StagedSkill(skill_name="excel", source="chat")])
    sp.merge_fields({"experience_text": "3 years bookkeeping"})

    sp.target_role_text = "truck driver"
    ts = build_truth_summary(staged=sp, user_message="looking for truck driver")
    assert ts.skills_aligned_with_target is False
    assert ts.experience_aligned_with_target is False
    assert ts.target_alignment_ok is False
    assert ts.target_alignment_first_misaligned_slot == "skills_text"


def test_truth_summary_alignment_serializes_to_planner_json():
    sp = StagedProfile.new("s1")
    sp.target_role_text = "truck driver"
    ts = build_truth_summary(staged=sp, user_message="hi")
    payload = ts.to_planner_json()
    assert "skills_aligned_with_target" in payload
    assert "experience_aligned_with_target" in payload
    assert "target_alignment_ok" in payload
    assert "target_alignment_first_misaligned_slot" in payload


# ============================================================================
# §6 — arbiter pass 1 gate
# ============================================================================
def _truth_with_alignment(*, alignment_ok: bool, first_misaligned: str | None):
    """Minimal truth dict shape the arbiter consumes."""
    return {
        "enough_to_match": True,
        "usable_evidence_present": True,
        "target_role_specificity": "specific",
        "target_role_text": "truck driver",
        "scope_violations_detected": [],
        "registry_gaps_in_message": [],
        "user_intent_signal": "neutral",
        "target_alignment_ok": alignment_ok,
        "target_alignment_first_misaligned_slot": first_misaligned,
    }


def _planner_proceed():
    """Builds a planner decision saying 'proceed_to_match'."""
    return PlannerDecision(
        move="proceed_to_match",
        reason_code="user_explicitly_asked_to_match",
        tone="brief_confident",
    )


def test_arbiter_gates_proceed_when_alignment_off():
    """The planner says proceed; truth says alignment is off; the
    arbiter must override to ask_one_clarifying_question on the
    first misaligned slot."""
    truth = _truth_with_alignment(
        alignment_ok=False, first_misaligned="skills_text",
    )
    out = validate_planner_intent(_planner_proceed(), truth)
    assert isinstance(out, ArbiterDecision)
    assert out.final_move == "ask_one_clarifying_question"
    assert out.reason_code == "target_changed_need_fresh_intake"
    assert out.ask_slot == "skills_text"


def test_arbiter_routes_to_experience_when_skills_aligned():
    """When skills are aligned but experience is misaligned, the
    arbiter asks for experience_text next."""
    truth = _truth_with_alignment(
        alignment_ok=False, first_misaligned="experience_text",
    )
    out = validate_planner_intent(_planner_proceed(), truth)
    assert isinstance(out, ArbiterDecision)
    assert out.ask_slot == "experience_text"


def test_arbiter_allows_proceed_when_alignment_ok():
    """Both aligned → RunEngine."""
    truth = _truth_with_alignment(alignment_ok=True, first_misaligned=None)
    out = validate_planner_intent(_planner_proceed(), truth)
    assert isinstance(out, RunEngine)
