"""Resume-upload offer (2026-06-16) — regression tests.

When the engine can't surface a strong/good match AND the user has
thin chat-side skill evidence AND no resume has been uploaded yet
AND the offer hasn't already fired, the responder weaves a
"upload a CV could unlock more matches" offer into the no-match /
low-band response. This corrects the previously misleading "no
postings exist" framing for cases where the underlying truth was
"evidence too thin to score the postings that DO exist."

The four-condition gate lives in `handler._should_offer_resume_upload`.
The responder side lives in `_present_no_match_fallback_v2`. These
tests pin both layers.
"""
from __future__ import annotations

import os

os.environ.setdefault("LLM_ENABLED", "false")

from dataclasses import dataclass, field

import pytest

from skillbridge.chat.handler import _should_offer_resume_upload
from skillbridge.chat.responder import (
    ResponderV2Input,
    _present_no_match_fallback_v2,
)
from skillbridge.chat.arbiter import ArbiterDecision
from skillbridge.session.staging import StagedProfile, StagedSkill

pytestmark = pytest.mark.nodb


# ============================================================================
# §1 — handler-side gate
# ============================================================================
def _staged_with_skills(n: int, *, resume: bool = False, offered: bool = False) -> StagedProfile:
    sp = StagedProfile.new("s1")
    sp.target_role_text = "accounting clerk"
    sp.skills = [
        StagedSkill(skill_name=f"skill_{i}", source="chat", confidence=0.85)
        for i in range(n)
    ]
    if resume:
        sp.resume_facts_json = {"skills": [{"skill_name": "x"}]}
    sp.resume_upload_offered = offered
    return sp


def test_offer_fires_on_no_match_with_thin_evidence():
    sp = _staged_with_skills(3)
    assert _should_offer_resume_upload(
        staged=sp, final_move="present_no_match", band_signal="none",
    ) is True


def test_offer_fires_on_stretch_only_with_thin_evidence():
    sp = _staged_with_skills(3)
    assert _should_offer_resume_upload(
        staged=sp, final_move="present_tiered_matches", band_signal="stretch_only",
    ) is True


def test_offer_fires_on_low_only_with_thin_evidence():
    sp = _staged_with_skills(3)
    assert _should_offer_resume_upload(
        staged=sp, final_move="present_matches", band_signal="low_only",
    ) is True


def test_offer_blocked_when_resume_uploaded():
    sp = _staged_with_skills(3, resume=True)
    assert _should_offer_resume_upload(
        staged=sp, final_move="present_no_match", band_signal="none",
    ) is False


def test_offer_fires_even_when_already_offered():
    """No-final-no-without-resume rule (2026-06-16): the once-per-target
    gate was DROPPED. The offer now fires on every no-match turn until
    a resume is uploaded. The `resume_upload_offered` flag is retained
    on StagedProfile for audit/telemetry but is no longer consulted by
    the gate. The LLM happy-path is responsible for varying its
    phrasing turn-by-turn so the user doesn't feel pestered."""
    sp = _staged_with_skills(3, offered=True)
    assert _should_offer_resume_upload(
        staged=sp, final_move="present_no_match", band_signal="none",
    ) is True


def test_offer_fires_regardless_of_skill_count():
    """User clarification (2026-06-16): "resume upload offer will do
    when resume missing + no strong match, not based on skills
    threshold." Even with 10 chat skills, if the engine can't surface
    a strong match AND no resume is uploaded AND the offer hasn't
    fired for the current target, the offer must fire."""
    sp = StagedProfile.new("s1")
    sp.target_role_text = "accounting clerk"
    sp.merge_skills([
        StagedSkill(skill_name=f"skill_{i}", source="chat", confidence=0.85)
        for i in range(10)
    ])
    assert _should_offer_resume_upload(
        staged=sp, final_move="present_no_match", band_signal="none",
    ) is True


def test_offer_blocked_when_strong_match_found():
    """Strong/good match means the engine surfaced a real fit — no
    need to offer upload."""
    sp = _staged_with_skills(3)
    assert _should_offer_resume_upload(
        staged=sp, final_move="present_matches", band_signal="strong_or_good",
    ) is False
    assert _should_offer_resume_upload(
        staged=sp, final_move="present_tiered_matches", band_signal="strong_or_good",
    ) is False


def test_offer_counts_only_chat_source_skills():
    """Resume-source skills don't count toward the thin-evidence
    threshold — if the user has 4 chat skills + 10 resume skills, the
    resume IS the rich evidence we'd be offering. Don't re-offer it."""
    sp = StagedProfile.new("s1")
    sp.target_role_text = "accounting clerk"
    sp.skills = [
        StagedSkill(skill_name=f"chat_{i}", source="chat", confidence=0.85)
        for i in range(4)
    ] + [
        StagedSkill(skill_name=f"resume_{i}", source="resume", confidence=0.85)
        for i in range(10)
    ]
    sp.resume_facts_json = {"skills": [{"skill_name": "x"}]}
    # Block because resume IS uploaded (resume_facts_json is set).
    assert _should_offer_resume_upload(
        staged=sp, final_move="present_no_match", band_signal="none",
    ) is False


# ============================================================================
# §2 — responder template
# ============================================================================
def _resp_input(
    *, offer: bool, next_skill=(None, 0),
    target_role_text: str | None = None,
) -> ResponderV2Input:
    return ResponderV2Input(
        user_message="hi",
        decision=ArbiterDecision(
            final_move="present_no_match",
            reason_code="zero_matches_in_dataset",
            tone="honest_redirect",
            arbiter_action="resolved_to_no_match",
        ),
        results=[],
        training_by_job={},
        next_skill=next_skill,
        band_signal="none",
        requires_consent=True,
        target_role_text=target_role_text,
        should_offer_resume_upload=offer,
    )


def test_responder_renders_offer_when_flag_set():
    inp = _resp_input(offer=True)
    msg = _present_no_match_fallback_v2(inp)
    # The reframed message acknowledges thin evidence + offers upload
    # BEFORE the SCCC referral.
    assert "couldn't find a strong fit" in msg.lower()
    assert "cv or resume" in msg.lower() or "upload" in msg.lower()
    # SCCC referral is still there, as the final fallback option.
    assert "sault community career centre" in msg.lower()
    # The misleading "no postings exist" framing is GONE.
    assert "i don't see one in today's sault ste. marie postings" not in msg.lower()


def test_responder_uses_legacy_template_when_flag_unset():
    inp = _resp_input(offer=False)
    msg = _present_no_match_fallback_v2(inp)
    # Legacy framing preserved when the offer doesn't fire.
    assert "i don't see one in today's sault ste. marie postings" in msg.lower()
    assert "sault community career centre" in msg.lower()
    # No upload offer.
    assert "cv or resume" not in msg.lower()
    assert "upload" not in msg.lower()


def test_responder_offer_with_next_skill_hint_includes_both():
    """When next_skill is populated, the offer message includes the
    'pick up X to unlock N more jobs' hint as a parenthetical."""
    inp = _resp_input(offer=True, next_skill=("payroll processing", 2))
    msg = _present_no_match_fallback_v2(inp)
    assert "payroll processing" in msg
    assert "2 more current jobs" in msg
    assert "cv or resume" in msg.lower() or "upload" in msg.lower()


# ============================================================================
# §3 — Role-aware substitution (Option A, 2026-06-16)
# ============================================================================
def test_offer_substitutes_target_role_in_text():
    """The literal target_role_text must appear in the message so
    accounting-clerk and truck-driver sessions render DIFFERENT no-
    match offers. Pre-fix, both rendered identical generic text
    ('the roles in your target')."""
    inp_accounting = _resp_input(offer=True, target_role_text="accounting clerk")
    msg_accounting = _present_no_match_fallback_v2(inp_accounting)
    assert "accounting clerk" in msg_accounting.lower()

    inp_truck = _resp_input(offer=True, target_role_text="truck driver")
    msg_truck = _present_no_match_fallback_v2(inp_truck)
    assert "truck driver" in msg_truck.lower()

    # The two messages must not be byte-identical (regression guard
    # for the "canned card" complaint from 2026-06-16).
    assert msg_accounting != msg_truck


def test_offer_uses_role_category_specific_keep_going_examples():
    """The 'or keep going here' line cites role-category-appropriate
    examples — accounting/admin gets Excel/QuickBooks/payroll;
    trucking/trades gets trade tickets/tools; etc. Role-blind generic
    examples ('tools, software, or tasks') only fire as fallback for
    the 'other' category."""
    msg_admin = _present_no_match_fallback_v2(
        _resp_input(offer=True, target_role_text="accounting clerk"),
    )
    # accounting clerk classifies as 'admin' → admin examples.
    assert (
        "excel" in msg_admin.lower()
        or "quickbooks" in msg_admin.lower()
        or "payroll" in msg_admin.lower()
        or "reconciliation" in msg_admin.lower()
    )

    msg_trades = _present_no_match_fallback_v2(
        _resp_input(offer=True, target_role_text="welder"),
    )
    # welder classifies as 'trades' → trade-ticket examples.
    assert (
        "310" in msg_trades  # trade codes like 310T/310S
        or "trade ticket" in msg_trades.lower()
        or "apprenticeship" in msg_trades.lower()
    )


def test_offer_falls_back_gracefully_when_no_target_role():
    """When target_role_text is None or empty, the offer must still
    render — just with the generic 'your target' phrasing and 'other'
    category examples. No crashes; no NoneTypeError."""
    msg = _present_no_match_fallback_v2(_resp_input(offer=True, target_role_text=None))
    assert "your target" in msg.lower()
    assert "cv or resume" in msg.lower() or "upload" in msg.lower()

    msg_empty = _present_no_match_fallback_v2(_resp_input(offer=True, target_role_text=""))
    assert "your target" in msg_empty.lower()


# ============================================================================
# §3 — Gap 3: target-change resets resume_upload_offered
# ============================================================================
def test_target_change_resets_resume_upload_offered():
    """Gap 3 (2026-06-16) live repro: user got the upload offer for
    "accounting clerk" → resume_upload_offered=True. Then switched to
    "truck driver" → flag must reset to False so the offer can
    re-fire if the new target also hits the thin-evidence path."""
    sp = StagedProfile.new("s1")
    sp.target_role_text = "accounting clerk"
    sp.resume_upload_offered = True  # offer already fired

    sp.target_role_text = "truck driver"
    assert sp.resume_upload_offered is False, (
        "Target change must reset resume_upload_offered so the offer "
        "becomes per-target rather than per-session."
    )


def test_target_change_does_not_clear_staged_skills_themselves():
    """Skills are still transferable across targets — the design says
    NOT to wipe staged.skills on target change (per CP3 principle).
    Gap 3+4 fixes touch only the offer flag + counter, NOT the
    actual skill list. Excel, customer service, etc. legitimately
    apply to multiple roles."""
    sp = StagedProfile.new("s1")
    sp.target_role_text = "accounting clerk"
    sp.merge_skills([
        StagedSkill(skill_name="excel", source="chat"),
        StagedSkill(skill_name="customer service", source="chat"),
    ])
    skill_names_before = {s.skill_name for s in sp.skills}

    sp.target_role_text = "truck driver"
    skill_names_after = {s.skill_name for s in sp.skills}
    assert skill_names_before == skill_names_after


def test_post_target_switch_offer_re_fires():
    """Live repro from session 167bf5e1 turn 4 (2026-06-16): user got
    the offer for accounting, switched to truck driver, then provided
    truck skills with no truck postings in the dataset. Pre-fix:
    `resume_upload_offered=True` from the accounting turn blocked the
    offer from re-firing. Post-Gap-3-fix: target switch resets the
    flag, and per the locked rule (no skill-count threshold), the
    offer re-fires whenever there's no resume + no strong match."""
    sp = StagedProfile.new("s1")

    # Turn 1-2: accounting flow → offer fires.
    sp.target_role_text = "accounting clerk"
    sp.merge_skills([
        StagedSkill(skill_name=name, source="chat", confidence=0.85)
        for name in ("accounting", "bookkeeper", "office admin")
    ])
    assert _should_offer_resume_upload(
        staged=sp, final_move="present_no_match", band_signal="none",
    ) is True
    sp.resume_upload_offered = True   # the offer rendered

    # Turn 3: target switch.
    sp.target_role_text = "truck driver"
    assert sp.resume_upload_offered is False   # Gap 3 reset

    # Turn 4: user gives truck skills (any number — threshold removed).
    sp.merge_skills([
        StagedSkill(skill_name=name, source="chat", confidence=0.85)
        for name in (
            "class g license", "defensive driving", "route planning",
            "customer service", "vehicle maintenance", "five years driving",
        )
    ])

    # Offer fires again for the new target, regardless of skill count.
    assert _should_offer_resume_upload(
        staged=sp, final_move="present_no_match", band_signal="none",
    ) is True, (
        "After Gap 3 fix + locked-rule simplification (no skill "
        "threshold), the offer must re-fire on the new target "
        "regardless of how many skills the user accumulated."
    )


def test_target_change_does_not_reset_when_value_unchanged():
    """Idempotent: setting target_role_text to its CURRENT value must
    not reset the offer flag. Only a real change triggers the reset
    (mirrors the existing target_noc invalidation pattern in the same
    __setattr__ override)."""
    sp = StagedProfile.new("s1")
    sp.target_role_text = "accounting clerk"
    sp.merge_skills([StagedSkill(skill_name="excel", source="chat")])
    sp.resume_upload_offered = True

    sp.target_role_text = "accounting clerk"   # same value
    assert sp.resume_upload_offered is True    # NOT reset
