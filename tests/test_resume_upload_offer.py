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


def test_offer_fires_on_strong_or_good_band_when_no_resume():
    """Pattern 1 (closing-matrix v2, 2026-06-17, LOCKED): the upload
    offer fires on ANY no-resume turn — regardless of band — framed
    around "look at related roles" (broadening), not "find a stronger
    match" (terminating). Earlier v1 behavior blocked the offer on
    strong/good band because the no-final-no-without-resume rule only
    cared about closing-on-no. Under the user-always-gets-something
    principle, the offer is now a universal invitation to upload for
    the related-role search (CP5) — strong matches still get a
    surface, but the closing pivots to broadening, not "go apply."
    """
    sp = _staged_with_skills(3)
    assert _should_offer_resume_upload(
        staged=sp, final_move="present_matches", band_signal="strong_or_good",
    ) is True
    assert _should_offer_resume_upload(
        staged=sp, final_move="present_tiered_matches", band_signal="strong_or_good",
    ) is True


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
# §4 — Gap 1: stretch-tier upload offer
# ============================================================================
def test_should_offer_fires_on_stretch_only_band():
    """Gap 1 (2026-06-16 evening): when the engine surfaces ONLY
    stretch-tier matches (band_signal=stretch_only) AND no resume,
    the upload offer must still fire. The user sees stretch-tier
    cards but the LLM weaves "uploading might unlock a stronger
    match" alongside them — completes the no-final-no-without-resume
    rule for the tier-cards path."""
    sp = _staged_with_skills(8)   # rich evidence; still gets offer
    assert _should_offer_resume_upload(
        staged=sp,
        final_move="present_tiered_matches",
        band_signal="stretch_only",
    ) is True


def test_should_offer_fires_on_strong_or_good_band_pattern_1():
    """Pattern 1 (closing-matrix v2, 2026-06-17): same as the §1 test
    above — strong/good band + no resume = offer fires. Was blocked
    under v1 (Gap 1, 2026-06-16). Flipped for the universal upload
    ask. Resume entitles the user to CP5 / CP4 service; without it,
    every turn invites broadening — including strong-band turns."""
    sp = _staged_with_skills(8)
    assert _should_offer_resume_upload(
        staged=sp,
        final_move="present_tiered_matches",
        band_signal="strong_or_good",
    ) is True


def _stub_view():
    """Minimal SanitizedResponderView stub — the four tier iterables
    the user_block builder reads. scoring-v6 (2026-06-17) added
    prompt_tiered_explore_later as the new fourth direct-target slot.
    Built via SimpleNamespace so the test doesn't depend on every
    field of the production dataclass."""
    from types import SimpleNamespace
    return SimpleNamespace(
        prompt_tiered_apply_today=(),
        prompt_tiered_worth_a_try=(),
        prompt_tiered_explore_later=(),
        prompt_tiered_sideways_move=(),
    )


def test_tiered_user_block_includes_resume_upload_offer_flag():
    """Gap 1 wiring: when should_offer_resume_upload=True,
    `_build_user_block_for_tiered_matches` includes RESUME_UPLOAD_OFFER:
    yes in the user block so the LLM has the signal."""
    from skillbridge.chat.responder import (
        _build_user_block_for_tiered_matches,
    )
    inp = _resp_input(offer=True, target_role_text="finance clerk")
    block = _build_user_block_for_tiered_matches(inp, _stub_view())
    assert "RESUME_UPLOAD_OFFER: yes" in block
    assert "TARGET_ROLE: finance clerk" in block


def test_tiered_user_block_omits_offer_when_flag_false():
    """When no offer authorized (e.g. resume uploaded or strong band),
    the user_block must NOT carry RESUME_UPLOAD_OFFER. Otherwise the
    LLM would see the flag and weave the offer language anyway."""
    from skillbridge.chat.responder import (
        _build_user_block_for_tiered_matches,
    )
    inp = _resp_input(offer=False, target_role_text="finance clerk")
    block = _build_user_block_for_tiered_matches(inp, _stub_view())
    assert "RESUME_UPLOAD_OFFER" not in block


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


# ============================================================================
# §5 — REMOVED (2026-06-17): action-closing tests
# ============================================================================
# The action-closing rule shipped on 2026-06-16 ("got your credentials
# ready to apply?" when STRONG_MATCHES tier had records) was rolled
# back on 2026-06-17 — it violated the locked user-always-gets-
# something principle (system never pushes the user toward "go
# apply"; applying is the user's decision). The 4 tests that pinned
# the prompt content for that section were deleted alongside the
# rollback. See project_user_always_gets_something memory file for
# the principle, and prompts.py for the rollback note inside
# COACH_TIERS_RESPONDER_PROMPT.


# ============================================================================
# §5.5 — 4-label heading rename (Step 4, scoring-v6, 2026-06-17)
# ============================================================================
# The prompt was updated to list 5 tier headings (Strong / Good /
# Stretch / Explore later / Sideways) instead of the prior 3 (Apply
# today / Worth a try / Sideways). The EVIDENCE PACKAGE section
# list expanded to 6 tier-related sections (added GOOD_MATCHES and
# EXPLORE_LATER). The responder user_block now serializes
# GOOD_MATCHES (records with strength_claim_text=strongest_current)
# and EXPLORE_LATER (records from prompt_tiered_explore_later).
# These tests pin the prompt content + the new section presence.

def test_coach_tiers_prompt_lists_four_target_headings():
    """The renamed heading list must include the 4 scoring-v6 labels
    + the unchanged Sideways heading."""
    from skillbridge.chat.prompts import COACH_TIERS_RESPONDER_PROMPT

    assert "**Strong match — apply today**" in COACH_TIERS_RESPONDER_PROMPT
    assert "**Good match — solid fit**" in COACH_TIERS_RESPONDER_PROMPT
    assert "**Stretch — reachable with prep**" in COACH_TIERS_RESPONDER_PROMPT
    assert "**Explore later — not your main target**" in COACH_TIERS_RESPONDER_PROMPT
    # Sideways heading (unchanged for now; Step 8 may rename to "related").
    assert "**Sideways move — same skills, different angle**" in COACH_TIERS_RESPONDER_PROMPT


def test_coach_tiers_prompt_lists_new_evidence_sections():
    """The EVIDENCE PACKAGE section list now includes GOOD_MATCHES
    and EXPLORE_LATER — the LLM has to know what raw sections to
    expect in user_block before mapping them to headings."""
    from skillbridge.chat.prompts import COACH_TIERS_RESPONDER_PROMPT

    assert "GOOD_MATCHES" in COACH_TIERS_RESPONDER_PROMPT
    assert "EXPLORE_LATER" in COACH_TIERS_RESPONDER_PROMPT


def test_coach_tiers_prompt_old_apply_today_heading_removed():
    """The pre-v6 "Apply today" heading text must NOT remain in the
    prompt — regression guard against a partial rename leaving the
    old phrase alongside the new ones (which would confuse the LLM
    about which to use)."""
    from skillbridge.chat.prompts import COACH_TIERS_RESPONDER_PROMPT

    # The old heading was: "Apply today — your skills line up"
    # Sub-pattern check ("your skills line up") catches it without
    # depending on em-dash encoding.
    assert "your skills line up" not in COACH_TIERS_RESPONDER_PROMPT
    # The old "Worth a try — close, with gaps to address" sub-pattern.
    assert "close, with gaps to address" not in COACH_TIERS_RESPONDER_PROMPT


def test_tiered_user_block_serializes_good_matches_separately():
    """STRONG_MATCHES section contains only competitive_match records
    (band="strong"); GOOD_MATCHES section contains only
    strongest_current records (band="good"). They must be SEPARATE
    sections in the user_block so the LLM picks the right heading
    for each posting."""
    from skillbridge.chat.responder import (
        _build_user_block_for_tiered_matches,
    )
    from skillbridge.chat.url_views import (
        PromptStrongMatch, JobFacts as ViewJobFacts,
    )
    from types import SimpleNamespace

    # Build minimal PromptStrongMatch items — one strong, one good.
    facts = ViewJobFacts(
        posted_date=None, posted_days_ago=None,
        location=None, employment_type=None, salary_text=None,
    )
    strong_item = PromptStrongMatch(
        job_id="s1", title="Strong Posting", employer="Acme",
        location=None, noc_code="14200", url=None,
        job_facts=facts, skill_alignment=(),
        non_blocking_gaps=(),
        credential_warning_text=None,
        strength_claim_text="competitive_match",
    )
    good_item = PromptStrongMatch(
        job_id="g1", title="Good Posting", employer="BetaCo",
        location=None, noc_code="14200", url=None,
        job_facts=facts, skill_alignment=(),
        non_blocking_gaps=(),
        credential_warning_text=None,
        strength_claim_text="strongest_current",
    )
    view = SimpleNamespace(
        prompt_tiered_apply_today=(strong_item, good_item),
        prompt_tiered_worth_a_try=(),
        prompt_tiered_explore_later=(),
        prompt_tiered_sideways_move=(),
    )
    inp = _resp_input(offer=False, target_role_text="accounting clerk")
    block = _build_user_block_for_tiered_matches(inp, view)

    # Both labelled sections appear.
    assert "STRONG_MATCHES:" in block
    assert "GOOD_MATCHES:" in block
    # Each posting appears under its own section header, not the other.
    strong_idx = block.index("STRONG_MATCHES:")
    good_idx = block.index("GOOD_MATCHES:")
    stretch_idx = block.index("STRETCH_MATCHES:")
    # Strong posting JSON is between STRONG_MATCHES: and GOOD_MATCHES:
    strong_section = block[strong_idx:good_idx]
    assert "Strong Posting" in strong_section
    assert "Good Posting" not in strong_section
    # Good posting JSON is between GOOD_MATCHES: and STRETCH_MATCHES:
    good_section = block[good_idx:stretch_idx]
    assert "Good Posting" in good_section
    assert "Strong Posting" not in good_section


def test_tiered_user_block_includes_explore_later_section():
    """The new EXPLORE_LATER section must appear in the user_block
    whenever prompt_tiered_explore_later has records. Section
    header appears even when empty (uniform structure) so the LLM
    can tell "explored, found nothing" from "section not present at
    all"."""
    from skillbridge.chat.responder import (
        _build_user_block_for_tiered_matches,
    )
    from types import SimpleNamespace

    view = SimpleNamespace(
        prompt_tiered_apply_today=(),
        prompt_tiered_worth_a_try=(),
        prompt_tiered_explore_later=(),
        prompt_tiered_sideways_move=(),
    )
    inp = _resp_input(offer=False, target_role_text="accounting clerk")
    block = _build_user_block_for_tiered_matches(inp, view)
    assert "EXPLORE_LATER:" in block


# ============================================================================
# §5.5.5 — Step 11i: COACH_TIERS closing reorder + forbidden phrases
# ============================================================================
# Live verify on 2026-06-17 showed the LLM defaulting to the action-
# closing ("Ready to put together an application?") even with
# Pattern 2 conditions held. Three contributing factors fixed in
# Step 11i: (a) PATTERN 3 was listed BEFORE PATTERN 2 (reading
# order), (b) Pattern 2's trigger didn't name the user_block signal
# explicitly, (c) no concrete forbidden-phrases list. These tests
# pin all three fixes as regression guards so the live bug can't
# silently come back.

def test_coach_tiers_prompt_pattern_2_before_pattern_3():
    """Step 11i regression guard: PATTERN 2 must appear BEFORE
    PATTERN 3 in the closing section. The LLM reads top-down;
    Pattern 2 is the more common case (any direct-target match +
    resume = Pattern 2; only adjacency-only with empty target =
    Pattern 3). Pattern 3 first absorbed the "related roles"
    example slot in the LLM's attention; reorder fixes it."""
    from skillbridge.chat.prompts import COACH_TIERS_RESPONDER_PROMPT

    p2_idx = COACH_TIERS_RESPONDER_PROMPT.index(
        "PATTERN 2 (closing-matrix v2"
    )
    p3_idx = COACH_TIERS_RESPONDER_PROMPT.index(
        "PATTERN 3 (closing-matrix v2"
    )
    assert p2_idx < p3_idx, (
        "PATTERN 2 must come BEFORE PATTERN 3 in the prompt."
    )


def test_coach_tiers_prompt_has_forbidden_closing_phrases_list():
    """Step 11i regression guard: the closing section MUST include
    a FORBIDDEN CLOSING PHRASES list that names specific action-
    closing clichés. Each named phrase gives the LLM a concrete
    avoid-list entry, not just an abstract rule."""
    from skillbridge.chat.prompts import COACH_TIERS_RESPONDER_PROMPT

    body = COACH_TIERS_RESPONDER_PROMPT
    # Header explicit.
    assert "FORBIDDEN CLOSING PHRASES" in body

    # Each forbidden phrase explicitly listed.
    forbidden = [
        "Ready to apply?",
        "Ready to put together an application?",
        "Ready to make a move on this?",
        "Want to give this a shot?",
        "Want to pull the trigger on this?",
        "Time to apply?",
        "Got your application together?",
        "Got your credentials ready to apply?",
        "Got your cover letter ready?",
        "Want to take the next step on this?",
        "Want to throw your hat in the ring?",
        "Want me to walk you through applying?",
    ]
    for phrase in forbidden:
        assert phrase in body, (
            f"Forbidden phrase {phrase!r} must be explicitly listed "
            f"in COACH_TIERS_RESPONDER_PROMPT."
        )


def test_coach_tiers_prompt_pattern_2_has_explicit_signal_mapping():
    """Step 11i regression guard: Pattern 2's trigger must specify
    the explicit user_block signals (RESUME_UPLOAD_OFFER absent +
    direct-target tier records) rather than implicit conditions.
    The LLM needs to see HOW to detect each precondition from the
    input block."""
    from skillbridge.chat.prompts import COACH_TIERS_RESPONDER_PROMPT

    body = COACH_TIERS_RESPONDER_PROMPT
    p2_idx = body.index("PATTERN 2 (closing-matrix v2")
    p3_idx = body.index("PATTERN 3 (closing-matrix v2")
    section = body[p2_idx:p3_idx]
    # Pattern 2 trigger names the absent RESUME_UPLOAD_OFFER signal.
    assert "RESUME_UPLOAD_OFFER" in section
    assert "absent" in section.lower()
    # Pattern 2 trigger names the 4 direct-target tier sections.
    for tier in ("STRONG_MATCHES", "GOOD_MATCHES",
                 "STRETCH_MATCHES", "EXPLORE_LATER"):
        assert tier in section, (
            f"Pattern 2 trigger must name {tier} explicitly."
        )


def test_coach_tiers_prompt_pattern_3_has_explicit_signal_mapping():
    """Step 11i regression guard: Pattern 3 trigger names the
    explicit signals (all 4 direct-target tiers empty + ADJACENT_JOBS
    has records). Same rationale as Pattern 2."""
    from skillbridge.chat.prompts import COACH_TIERS_RESPONDER_PROMPT

    body = COACH_TIERS_RESPONDER_PROMPT
    p3_idx = body.index("PATTERN 3 (closing-matrix v2")
    end_idx = body.index("GENERIC CLOSE", p3_idx)
    section = body[p3_idx:end_idx]
    assert "RESUME_UPLOAD_OFFER" in section
    assert "ADJACENT_JOBS" in section
    for tier in ("STRONG_MATCHES", "GOOD_MATCHES",
                 "STRETCH_MATCHES", "EXPLORE_LATER"):
        assert tier in section


def test_coach_tiers_prompt_pattern_2_locked_example_present():
    """Pattern 2's canonical example phrasing must remain in the
    prompt — the LLM uses it as the anchor for varying turn-by-turn."""
    from skillbridge.chat.prompts import COACH_TIERS_RESPONDER_PROMPT

    body = COACH_TIERS_RESPONDER_PROMPT
    assert "Want me to also look at related roles your skills fit?" in body


def test_coach_tiers_prompt_forbidden_list_appears_before_patterns():
    """Step 11i: the FORBIDDEN CLOSING PHRASES list MUST appear
    BEFORE both PATTERN 2 and PATTERN 3 in the prompt — the LLM
    should read the forbidden list FIRST so it has the constraint
    in mind when applying any closing rule. Without this ordering,
    the LLM might apply the pattern rule and then think of a
    "natural" close from its training, which is exactly the
    cliché set the forbidden list catches."""
    from skillbridge.chat.prompts import COACH_TIERS_RESPONDER_PROMPT

    body = COACH_TIERS_RESPONDER_PROMPT
    forbidden_idx = body.index("FORBIDDEN CLOSING PHRASES")
    p2_idx = body.index("PATTERN 2 (closing-matrix v2")
    p3_idx = body.index("PATTERN 3 (closing-matrix v2")
    assert forbidden_idx < p2_idx < p3_idx, (
        "FORBIDDEN list must come first, then Pattern 2, then "
        "Pattern 3."
    )


# ============================================================================
# §5.6 — Pattern 1 closing rule (Step 5, closing-matrix v2, 2026-06-17)
# ============================================================================
# Pattern 1 (universal upload ask): when RESUME_UPLOAD_OFFER is "yes"
# the prompt's closing rule MUST frame the upload as broadening into
# "related roles" — NOT as "find a stronger match" (terminating) or
# "go apply" (pushing the user out). These tests pin the prompt
# content + the widened gate behavior so a regression that reverts
# either piece breaks the test.

def test_pattern_1_prompt_frames_upload_around_related_roles():
    """The UPLOAD OFFER section names 'related roles' (broadening) as
    the value of upload — not 'stronger match' (terminating)."""
    from skillbridge.chat.prompts import COACH_TIERS_RESPONDER_PROMPT

    upload_start = COACH_TIERS_RESPONDER_PROMPT.index(
        'UPLOAD OFFER + CLOSING when RESUME_UPLOAD_OFFER is "yes"'
    )
    # Slice forward to the next major prompt section so we only check
    # text inside the UPLOAD OFFER block.
    closing_fallback_idx = COACH_TIERS_RESPONDER_PROMPT.index(
        "CLOSING when RESUME_UPLOAD_OFFER is absent"
    )
    section = COACH_TIERS_RESPONDER_PROMPT[upload_start:closing_fallback_idx]

    # Broadening language present.
    body = section.lower()
    # Both the spec language ("related roles" / "related role") and
    # the example phrasings appear. Substrings stay short so the
    # line-wrapping inside comment blocks doesn't break the check.
    assert "related roles" in body or "related role" in body
    assert "find related roles" in body or "other related" in body
    # The word "broadening" or equivalent value-frame appears.
    assert (
        "broader" in body
        or "broadening" in body
        or "more options" in body
        or "more roles" in body
    )


def test_pattern_1_prompt_forbids_terminating_framing():
    """The UPLOAD OFFER section MUST explicitly forbid the
    terminating-framing phrasings ('stronger match', 'go apply')
    that violate the user-always-gets-something principle."""
    from skillbridge.chat.prompts import COACH_TIERS_RESPONDER_PROMPT

    upload_start = COACH_TIERS_RESPONDER_PROMPT.index(
        'UPLOAD OFFER + CLOSING when RESUME_UPLOAD_OFFER is "yes"'
    )
    closing_fallback_idx = COACH_TIERS_RESPONDER_PROMPT.index(
        "CLOSING when RESUME_UPLOAD_OFFER is absent"
    )
    section = COACH_TIERS_RESPONDER_PROMPT[upload_start:closing_fallback_idx]

    body = section.lower()
    # Forbid-list called out in the section.
    assert "stronger match" in body  # named as a thing to avoid
    assert "credentials ready" in body or "go apply" in body or "applying" in body
    # Structural rules survive (carried over from v1, still locked).
    assert "must end with a question mark" in body
    assert "one sentence" in body
    assert "under 25 words" in body


def test_pattern_1_gate_fires_on_all_no_resume_bands():
    """Pattern 1 gate: fires for EVERY band when no resume. Pin all
    the band/move combinations to lock the universal behavior."""
    sp = _staged_with_skills(5)   # no resume
    for final_move in ("present_matches", "present_tiered_matches",
                       "present_no_match"):
        for band in ("strong_or_good", "stretch_only", "low_only", "none"):
            assert _should_offer_resume_upload(
                staged=sp, final_move=final_move, band_signal=band,
            ) is True, f"Pattern 1 should fire on ({final_move}, {band}, no resume)"


def test_pattern_1_gate_blocked_when_resume_uploaded():
    """Resume uploaded = Pattern 1 doesn't fire (Pattern 2 / 3
    territory). The gate becomes the resume gate — no other factors."""
    sp = _staged_with_skills(3, resume=True)
    for final_move in ("present_matches", "present_tiered_matches",
                       "present_no_match"):
        for band in ("strong_or_good", "stretch_only", "low_only", "none"):
            assert _should_offer_resume_upload(
                staged=sp, final_move=final_move, band_signal=band,
            ) is False, f"Pattern 1 must not fire when resume uploaded ({final_move}, {band})"


# ============================================================================
# §5.7 — Pattern 3 closing rule (Step 6, closing-matrix v2, 2026-06-17)
# ============================================================================
# Pattern 3 (CP5 auto-fire inline): when resume is uploaded AND the
# engine returned ZERO matches in the user's target market AND
# adjacency lookup found related roles, the response frames the
# related roles as the primary surface — an honest pivot, not a
# consolation. The arbiter ALREADY routes this scenario to
# present_tiered_matches with only ADJACENT_JOBS populated (see
# arbiter.py lines 629-652). Step 6 ships the PROMPT RULE so the
# LLM frames the response correctly.

def _pattern_3_section() -> str:
    """Helper: slice the Pattern 3 section from the prompt under the
    Step 11i ordering (Pattern 2 comes BEFORE Pattern 3). Pattern 3
    runs from its header to GENERIC CLOSE."""
    from skillbridge.chat.prompts import COACH_TIERS_RESPONDER_PROMPT
    start = COACH_TIERS_RESPONDER_PROMPT.index("PATTERN 3 (closing-matrix v2")
    end = COACH_TIERS_RESPONDER_PROMPT.index("GENERIC CLOSE", start)
    return COACH_TIERS_RESPONDER_PROMPT[start:end]


def _pattern_2_section() -> str:
    """Helper: slice the Pattern 2 section under Step 11i ordering.
    Pattern 2 runs from its header to Pattern 3's header."""
    from skillbridge.chat.prompts import COACH_TIERS_RESPONDER_PROMPT
    start = COACH_TIERS_RESPONDER_PROMPT.index("PATTERN 2 (closing-matrix v2")
    end = COACH_TIERS_RESPONDER_PROMPT.index("PATTERN 3 (closing-matrix v2")
    return COACH_TIERS_RESPONDER_PROMPT[start:end]


def test_pattern_3_prompt_rule_present():
    """The CLOSING (when RESUME_UPLOAD_OFFER is absent) branch
    includes a PATTERN 3 sub-rule for the CP5 auto-fire inline case."""
    from skillbridge.chat.prompts import COACH_TIERS_RESPONDER_PROMPT

    assert "PATTERN 3" in COACH_TIERS_RESPONDER_PROMPT
    body = COACH_TIERS_RESPONDER_PROMPT.lower()
    assert "auto-fire inline" in body or "auto-fire" in body
    # Trigger conditions named (in the Pattern 3 section specifically).
    section = _pattern_3_section().lower()
    assert "adjacent_jobs" in section
    assert "strong_matches" in section and "good_matches" in section
    assert "stretch_matches" in section and "explore_later" in section


def test_pattern_3_prompt_requires_honest_pivot_framing():
    """Pattern 3 must be framed as an HONEST PIVOT (acknowledge the
    empty target market, then surface the related roles), not as a
    hollow no-match or a consolation."""
    body = _pattern_3_section().lower()

    # Honest acknowledgment of empty target market required.
    assert "nothing in" in body or "empty target" in body
    # Related roles framing (locked v2 vocabulary).
    assert "related role" in body or "related roles" in body
    # Forbidden anti-patterns spelled out.
    assert "do not" in body
    assert "consolation" in body or "downgrade" in body


def test_pattern_3_prompt_forbids_internal_tokens():
    """Pattern 3 closing must NOT quote internal vocabulary tokens
    (same_noc_minor_group, transferable_lane, etc.) — those are
    engine-side discriminators, not user-facing copy."""
    body = _pattern_3_section().lower()
    assert "same_noc_minor_group" in body
    assert "transferable_lane" in body


def test_pattern_3_prompt_example_uses_question_close():
    """The example phrasing in Pattern 3 must end with a question
    mark — matches the policy gate's locked rule."""
    section = _pattern_3_section()
    assert 'End with "?"' in section or "ends with" in section.lower()
    assert "Want to look at" in section
    assert "those?" in section or "these?" in section


# ============================================================================
# §5.8 — Pattern 2 closing rule + state flag (Step 7, 2026-06-17)
# ============================================================================
# Pattern 2 (CP5 offer with two-turn flow): when resume is uploaded
# AND at least one direct-target tier has records, the closing
# OFFERS the related-role search ("want me to also look at related
# roles?"). The user's yes-like reply on Turn N+1 fires CP5; their
# no / other reply clears the flag and routes normally. The flag
# `staged.pending_adjacent_search_offer` (added 2026-06-17) tracks
# the pending consent state. Step 7 ships the declarative pieces:
# field + prompt rule + tests. The handler set/consume wiring is
# Step 7b.

def test_staged_profile_has_pending_adjacent_search_offer_field():
    """Pattern 2's two-turn flow needs a state flag — added 2026-06-17.
    Default is False. The flag tracks "we just asked the user about
    related-role search, awaiting their reply."""
    sp = StagedProfile.new("s1")
    assert hasattr(sp, "pending_adjacent_search_offer")
    assert sp.pending_adjacent_search_offer is False


def test_pending_adjacent_search_offer_resets_on_target_change():
    """Per-target lifecycle: target switch invalidates the pending
    consent (the prior question was about the prior target's
    adjacencies). Same pattern as resume_upload_offered."""
    sp = StagedProfile.new("s1")
    sp.target_role_text = "accounting clerk"
    sp.pending_adjacent_search_offer = True

    sp.target_role_text = "truck driver"   # different target → reset
    assert sp.pending_adjacent_search_offer is False


def test_pending_adjacent_search_offer_not_reset_when_target_unchanged():
    """Idempotent: setting target_role_text to the SAME value does
    not reset the flag (mirrors the resume_upload_offered behavior
    in the same __setattr__ override)."""
    sp = StagedProfile.new("s1")
    sp.target_role_text = "accounting clerk"
    sp.pending_adjacent_search_offer = True

    sp.target_role_text = "accounting clerk"   # same value
    assert sp.pending_adjacent_search_offer is True


def test_pattern_2_prompt_rule_present():
    """The CLOSING branch includes a PATTERN 2 sub-rule (separate
    from Pattern 3) for the CP5-offer two-turn flow."""
    from skillbridge.chat.prompts import COACH_TIERS_RESPONDER_PROMPT

    assert "PATTERN 2" in COACH_TIERS_RESPONDER_PROMPT
    body = COACH_TIERS_RESPONDER_PROMPT.lower()
    # Pattern 2 is named as the CP5 offer / two-turn flow.
    assert "cp5" in body or "related-role search" in body
    assert "two-turn" in body or "user's yes" in body or "next turn" in body


def test_pattern_2_prompt_requires_offer_not_push():
    """Pattern 2 must frame the closing as an OFFER ('want me to
    look at related roles?'), not as a push ('go apply', 'ready
    to apply?'). Uses the Step-11i-ordering helper for slicing."""
    body = _pattern_2_section().lower()
    section = _pattern_2_section()

    # Broadening language present (related roles).
    assert "related role" in body or "related roles" in body
    # Anti-patterns named so the LLM sees and avoids them.
    assert "applying" in body or "go apply" in body or "ready to apply" in body
    # Structural rules carry forward (one sentence, ends with ?, under 25 words).
    assert "one sentence" in body
    assert "under 25 words" in body
    assert "?" in section  # the example uses ?


def test_pattern_2_prompt_forbids_internal_adjacent_vocabulary():
    """Pattern 2 closing must use 'related' not 'adjacent' in user-
    facing copy."""
    body = _pattern_2_section().lower()

    # "related" appears in user-facing examples and rules.
    assert "related" in body
    # "adjacent" is called out as INTERNAL vocab.
    assert "internal" in body or "(not " in body


# ============================================================================
# §5.9 — Pattern 2 classifier helper (Step 7b, 2026-06-17)
# ============================================================================
# `_classify_pattern_2_reply` reads the user's reply to a Pattern 2
# closing question ("want me to also look at related roles?") and
# classifies it as yes / no / other. The yes-set is REUSED from
# adjacent_intent._AFFIRMATIVE_REPLIES (same coach voice = same
# expected user replies); the no-set is a new locked frozenset.
# Anything outside both sets returns "other" (conservative).

from skillbridge.chat.handler import _classify_pattern_2_reply


@pytest.mark.parametrize("reply", [
    # confirming variations — match _CONFIRMING_PATTERNS in truth_summary
    "yes", "Yes", "YES",
    "alright", "looks right", "looks good", "sounds good",
    "that's right", "yep", "yeah", "sure", "ok", "okay",
    # impatient_proceed variations — match _IMPATIENT_PATTERNS
    "go ahead", "let's go",
    # The live repro that broke v1 (Step 11b fix):
    "yes. go ahead",
    "yes, go ahead",
])
def test_pattern_2_classifier_yes(reply):
    """Affirmative replies — both 'confirming' and 'impatient_proceed'
    truth-summary intents map to consent. This includes the live
    repro 'yes. go ahead' that broke the regex-based v1 classifier."""
    assert _classify_pattern_2_reply(reply) == "yes"


@pytest.mark.parametrize("reply", [
    "no thanks", "no thank you", "not now", "not today",
    "not interested", "skip", "skip it",
    "nope",
    "I don't want", "i do not want",
])
def test_pattern_2_classifier_no(reply):
    """Negative replies — declining intent maps to no. NOTE: the
    truth_summary pattern requires "no thanks" / "nope" / "not now"
    — bare "no" returns `neutral` and falls through to "other"
    (see `test_pattern_2_classifier_bare_no_known_limitation`)."""
    assert _classify_pattern_2_reply(reply) == "no"


def test_pattern_2_classifier_bare_no_known_limitation():
    """Known limitation (2026-06-17, Step 11b): the truth_summary
    `_DECLINING_PATTERNS` regex requires accompanying tokens
    ("no thanks" / "not now" / "skip") and does NOT match a bare
    "no". So a user typing just "no" to Pattern 2's offer falls
    to "other" — flag clears, but the message routes through
    normal planner logic rather than being recognized as a
    decline.

    Acceptable cost — most decline replies include politeness
    ("no thanks", "not now") and bare "no" routes through the
    planner anyway. A future fix could add `^no\\b$` to
    `_DECLINING_PATTERNS` in truth_summary, but that touches a
    shared module and could affect other consumers."""
    assert _classify_pattern_2_reply("no") == "other"
    assert _classify_pattern_2_reply("No") == "other"
    assert _classify_pattern_2_reply("NO") == "other"


@pytest.mark.parametrize("reply", [
    # asking_question intent (questions)
    "what about other roles?",
    # asking_about_gap intent (gap questions)
    "how do I get my Class G?",
    # correcting intent
    "actually let me think about that",
    # neutral (no strong signal) — most natural-language replies
    "tell me more about the first one",
    "the second one looks interesting",
    "good",  # bare "good" doesn't match the "sounds good" / "looks good" patterns
    "show me",  # impatient pattern requires "show me jobs/matches/results"
    "", "  ",
])
def test_pattern_2_classifier_other(reply):
    """Anything that isn't a clean affirmative or decline returns
    'other'. The consume hook clears the flag and routes normally,
    treating the message as a fresh turn."""
    assert _classify_pattern_2_reply(reply) == "other"


def test_pattern_2_classifier_handles_non_string():
    """Defensive: a non-string message (theoretically shouldn't
    happen, but guards against bad upstream wiring) returns 'other'
    instead of raising."""
    assert _classify_pattern_2_reply(None) == "other"
    assert _classify_pattern_2_reply(123) == "other"


def test_pattern_2_classifier_delegates_to_truth_summary_intent():
    """Step 11b refactor: the classifier reads `_classify_intent`
    from truth_summary. Pin the integration point so a future
    rewrite of either layer that drops the contract breaks this
    test loudly."""
    from skillbridge.chat.truth_summary import _classify_intent

    # The classifier MUST return whatever the intent layer says for
    # the signals it acts on.
    assert _classify_intent("yes. go ahead") == "impatient_proceed"
    assert _classify_pattern_2_reply("yes. go ahead") == "yes"
    assert _classify_intent("alright") == "confirming"
    assert _classify_pattern_2_reply("alright") == "yes"
    # NOTE: the truth_summary pattern requires "no thanks" / "nope"
    # / "not now" — bare "no" produces neutral, not declining.
    assert _classify_intent("no thanks") == "declining"
    assert _classify_pattern_2_reply("no thanks") == "no"


# ============================================================================
# §5.10 — Pattern 2 yes-consent display projection (Step 8, 2026-06-17)
# ============================================================================
# `_blank_direct_tiers_for_pattern_2` is the surgical change that
# wires Pattern 2's yes-consent into the existing Sideways display
# infrastructure. It replaces a TieredEvidence's direct-target
# slots (apply_today, worth_a_try, explore_later) with empty
# tuples, preserves sideways_move, and lets the rest of the
# pipeline render the result as a related-roles pivot — same
# Sideways-only shape that Pattern 3's auto-fire produces.

from skillbridge.chat.handler import _blank_direct_tiers_for_pattern_2
from skillbridge.chat.tiered_evidence import TieredEvidence


def _make_tier_evidence(
    *, apply_today_count=0, worth_a_try_count=0,
    explore_later_count=0, sideways_count=0,
):
    """Build a TieredEvidence stub with sentinel-shaped tier slots.
    The tuple contents don't matter for the blanking test — only
    the lengths do, since `_blank_direct_tiers_for_pattern_2`
    works at the slot level."""
    sentinel = object()
    return TieredEvidence(
        apply_today=tuple(sentinel for _ in range(apply_today_count)),
        worth_a_try=tuple(sentinel for _ in range(worth_a_try_count)),
        sideways_move=tuple(sentinel for _ in range(sideways_count)),
        explore_later=tuple(sentinel for _ in range(explore_later_count)),
    )


def test_blank_direct_tiers_preserves_sideways():
    """The Sideways tier carries through unchanged — it's the
    related-roles surface the user asked to see."""
    te = _make_tier_evidence(
        apply_today_count=2, worth_a_try_count=1,
        explore_later_count=1, sideways_count=3,
    )
    blanked = _blank_direct_tiers_for_pattern_2(te)
    assert len(blanked.sideways_move) == 3
    # And references the SAME tuple identities (no rebuilding /
    # re-projection — the blanking is pure slot-substitution).
    assert blanked.sideways_move is te.sideways_move


def test_blank_direct_tiers_clears_direct_target_slots():
    """The three direct-target slots (apply_today, worth_a_try,
    explore_later) must be empty after blanking."""
    te = _make_tier_evidence(
        apply_today_count=3, worth_a_try_count=2,
        explore_later_count=1, sideways_count=2,
    )
    blanked = _blank_direct_tiers_for_pattern_2(te)
    assert blanked.apply_today == ()
    assert blanked.worth_a_try == ()
    assert blanked.explore_later == ()


def test_blank_direct_tiers_when_no_sideways_falls_to_empty():
    """If sideways was empty in the original, the blanked
    TieredEvidence is all-empty — `_tier_evidence_has_any_records`
    returns False downstream and the handler falls through to
    present_no_match (which Step 9 enhances with SSM market
    summary). This pins the graceful-degradation behavior so a
    yes-consent + no-related-roles scenario doesn't loop or stall."""
    te = _make_tier_evidence(
        apply_today_count=2, worth_a_try_count=1,
        sideways_count=0,  # no related roles found
    )
    blanked = _blank_direct_tiers_for_pattern_2(te)
    assert blanked.apply_today == ()
    assert blanked.worth_a_try == ()
    assert blanked.explore_later == ()
    assert blanked.sideways_move == ()


def test_blank_direct_tiers_returns_new_instance():
    """The blanking function returns a NEW TieredEvidence (frozen
    dataclass) rather than mutating the input. Mutation would
    break upstream callers that hold the same reference."""
    te = _make_tier_evidence(
        apply_today_count=1, sideways_count=1,
    )
    blanked = _blank_direct_tiers_for_pattern_2(te)
    # Original is untouched.
    assert len(te.apply_today) == 1
    # Blanked is distinct.
    assert blanked is not te


# ============================================================================
# §5.11 — SHAPE 2 enhanced: SSM market summary (Step 9, 2026-06-17)
# ============================================================================
# When the no-match branch fires WITHOUT a resume-upload offer
# (meaning resume IS uploaded AND adjacency returned nothing AND
# we're at the absolute floor of the closing matrix), the response
# now surfaces a Sault Ste. Marie market panorama instead of a bare
# "talk to SCCC" close. The PipelineSnapshot dataclass was extended
# with top_sectors + top_employers (Step 9 / 2026-06-17); the
# fallback renderer reads them when present and substitutes
# graceful coach-voice phrases.

from skillbridge.chat.pipeline_snapshot import (
    PipelineSnapshot, _NOC_BROAD_CATEGORY_NAMES,
)
from skillbridge.chat.responder import (
    _format_top_sectors_phrase,
    _format_top_employers_phrase,
)


def test_pipeline_snapshot_has_default_market_summary_fields():
    """The new market-summary fields default to empty tuples for
    backward compatibility with snapshots constructed pre-Step-9."""
    snap = PipelineSnapshot(total_active_jobs=43, last_publish_at_text=None)
    assert snap.top_sectors == ()
    assert snap.top_employers == ()


def test_pipeline_snapshot_carries_market_summary_when_supplied():
    snap = PipelineSnapshot(
        total_active_jobs=43,
        last_publish_at_text="2026-06-17 09:00 ET",
        top_sectors=("healthcare", "trades and transport", "education and social services"),
        top_employers=("Health Sciences North", "Algoma Family Services", "CMHA"),
    )
    assert len(snap.top_sectors) == 3
    assert "healthcare" in snap.top_sectors
    assert "Health Sciences North" in snap.top_employers


def test_noc_broad_category_names_cover_all_digits():
    """The NOC 2021 broad-category mapping covers digits 0-9
    inclusive. Step 9's SQL groups by LEFT(noc_code, 1) so any
    valid NOC first digit must resolve to a user-facing name."""
    for digit in "0123456789":
        assert digit in _NOC_BROAD_CATEGORY_NAMES
        assert _NOC_BROAD_CATEGORY_NAMES[digit].strip() != ""


@pytest.mark.parametrize("sectors, expected", [
    ((), ""),
    (("healthcare",), "mostly in healthcare"),
    (("healthcare", "trades and transport"),
     "mostly in healthcare and trades and transport"),
    (("healthcare", "trades and transport", "education and social services"),
     "mostly in healthcare, trades and transport, and education and social services"),
    # >3 sectors → truncated to first 3.
    (("a", "b", "c", "d", "e"), "mostly in a, b, and c"),
])
def test_format_top_sectors_phrase(sectors, expected):
    """The sector phrasing handles 0/1/2/3 sectors gracefully (no
    "mostly in " orphan, proper "and" joiners) and caps at 3."""
    assert _format_top_sectors_phrase(sectors) == expected


@pytest.mark.parametrize("employers, expected", [
    ((), ""),
    (("Health Sciences North",),
     "Health Sciences North is actively hiring."),
    (("Health Sciences North", "Algoma Family Services"),
     "Health Sciences North and Algoma Family Services are actively hiring."),
    (("Health Sciences North", "Algoma Family Services", "CMHA"),
     "Health Sciences North, Algoma Family Services, and CMHA are "
     "actively hiring."),
])
def test_format_top_employers_phrase(employers, expected):
    """Employer phrasing parallels sector phrasing — graceful 0/1/2/3
    handling. Singular vs plural verb ("is" vs "are") matches grammar."""
    assert _format_top_employers_phrase(employers) == expected


def test_no_match_fallback_renders_3_movement_structure_when_snapshot_present():
    """Step 11g (2026-06-17): when the no-match fallback fires
    WITHOUT a resume-upload offer (resume IS uploaded → Pattern 1
    doesn't fire → we're at SHAPE 2 / RELATED_ROLES_EXHAUSTED
    territory) AND a populated snapshot is available, the response
    renders the locked 3-movement structure:
      A: acknowledgment ("I checked for related roles...")
      B: market panorama (count + sectors + employers)
      C: training-offer close (abstract, no specific provider)
    Symmetric to the LLM happy path's Step 11f prompt rule."""
    from skillbridge.chat.responder import _present_no_match_fallback_v2

    snap = PipelineSnapshot(
        total_active_jobs=43,
        last_publish_at_text="2026-06-17 09:00 ET",
        top_sectors=("healthcare", "trades and transport", "admin"),
        top_employers=("Health Sciences North", "Algoma Family Services"),
    )
    inp = _resp_input(offer=False, target_role_text="accounting clerk")
    # pipeline_snapshot lives on ResponderV2Input — set it directly.
    inp.pipeline_snapshot = snap
    msg = _present_no_match_fallback_v2(inp)

    # Movement A — acknowledgment (replaces generic "I don't see a fit").
    assert "checked for related roles" in msg
    assert "didn't find" in msg
    # The generic v1 "I don't see a fit" lead-in is REMOVED.
    assert "don't see a fit" not in msg

    # Movement B — market panorama.
    # Active count is named.
    assert "43" in msg
    # Sectors phrase appears.
    assert "mostly in healthcare" in msg
    # Employers phrase appears.
    assert "Health Sciences North and Algoma Family Services" in msg
    assert "are actively hiring" in msg

    # Movement C — training-direction close (Step 11h locked phrasing).
    # The old sectors/SCCC close is REPLACED.
    assert "Want me to look at one of those sectors" not in msg
    assert "Sault Community Career Centre" not in msg
    # The training offer is present (Step 11h locked wording).
    assert "improve your skills gap" in msg
    assert "training directions" in msg
    assert "do you want?" in msg.lower()


def test_no_match_fallback_movement_c2_names_cp4_gap_verbatim():
    """Step 11h: when ResponderV2Input carries `cp4_primary_gap`,
    the deterministic fallback's Movement C2 names it verbatim:
    "The one thing that came up is [GAP]." When absent, C2 is
    skipped (no fabrication)."""
    from skillbridge.chat.responder import _present_no_match_fallback_v2

    snap = PipelineSnapshot(
        total_active_jobs=43,
        last_publish_at_text=None,
        top_sectors=("healthcare",),
        top_employers=("Health Sciences North",),
    )
    inp = _resp_input(offer=False, target_role_text="accounting clerk")
    inp.pipeline_snapshot = snap
    inp.cp4_primary_gap = "confidentiality handling"
    msg = _present_no_match_fallback_v2(inp)
    # C2 fires with the gap verbatim.
    assert "The one thing that came up is confidentiality handling" in msg
    # C3 closes with the locked phrasing.
    assert "improve your skills gap" in msg
    assert "training directions" in msg


def test_no_match_fallback_movement_c2_skipped_when_no_cp4_gap():
    """Step 11h: when cp4_primary_gap is None, the fallback MUST
    NOT fabricate a gap. C2 is silently skipped; C3 still fires."""
    from skillbridge.chat.responder import _present_no_match_fallback_v2

    snap = PipelineSnapshot(
        total_active_jobs=43,
        last_publish_at_text=None,
        top_sectors=("healthcare",),
        top_employers=("Health Sciences North",),
    )
    inp = _resp_input(offer=False, target_role_text="accounting clerk")
    inp.pipeline_snapshot = snap
    inp.cp4_primary_gap = None
    msg = _present_no_match_fallback_v2(inp)
    # No C2.
    assert "The one thing that came up" not in msg
    # C3 still fires.
    assert "improve your skills gap" in msg
    assert "training directions" in msg


def test_no_match_fallback_movement_c_avoids_specific_providers():
    """Step 11g HARD RULE: the training-offer close MUST NOT name
    specific providers (QuickBooks, Sault College, etc.) on this
    turn — the TRAINING block isn't present, so any specific name
    would be ungrounded. The fallback's phrasing stays abstract."""
    from skillbridge.chat.responder import _present_no_match_fallback_v2

    snap = PipelineSnapshot(
        total_active_jobs=43,
        last_publish_at_text="2026-06-17 09:00 ET",
        top_sectors=("healthcare",),
        top_employers=("Health Sciences North",),
    )
    inp = _resp_input(offer=False, target_role_text="accounting clerk")
    inp.pipeline_snapshot = snap
    msg = _present_no_match_fallback_v2(inp)

    # Hallucination guard — no specific training provider names.
    msg_lower = msg.lower()
    assert "quickbooks" not in msg_lower
    assert "sault college" not in msg_lower
    assert "sage" not in msg_lower


def test_no_match_fallback_falls_back_when_no_snapshot():
    """When pipeline_snapshot is None, the legacy SHAPE 2 message
    fires — preserving the safety net for environments where the
    snapshot can't be fetched (test mode, DB outage, etc.)."""
    from skillbridge.chat.responder import _present_no_match_fallback_v2

    inp = _resp_input(offer=False, target_role_text="accounting clerk")
    inp.pipeline_snapshot = None
    msg = _present_no_match_fallback_v2(inp)
    # Legacy lead-in present.
    assert "I don't see one in today's Sault Ste. Marie postings." in msg
    # Legacy SCCC referral present.
    assert "Sault Community Career Centre" in msg
    # No market summary tokens.
    assert "43" not in msg
    assert "mostly in" not in msg


def test_no_match_fallback_falls_back_when_snapshot_has_zero_jobs():
    """If total_active_jobs is 0, the market summary is meaningless
    — fall back to the legacy SHAPE 2 close."""
    from skillbridge.chat.responder import _present_no_match_fallback_v2

    snap = PipelineSnapshot(
        total_active_jobs=0, last_publish_at_text=None,
        top_sectors=(), top_employers=(),
    )
    inp = _resp_input(offer=False, target_role_text="accounting clerk")
    inp.pipeline_snapshot = snap
    msg = _present_no_match_fallback_v2(inp)
    assert "I don't see one in today's Sault Ste. Marie postings." in msg


def test_no_match_fallback_resume_offer_path_unchanged():
    """When should_offer_resume_upload=True, the Pattern 1 message
    branch fires (untouched by Step 9). Verify the market summary
    does NOT leak into that path."""
    from skillbridge.chat.responder import _present_no_match_fallback_v2

    snap = PipelineSnapshot(
        total_active_jobs=43, last_publish_at_text=None,
        top_sectors=("healthcare",), top_employers=("Health Sciences North",),
    )
    inp = _resp_input(offer=True, target_role_text="accounting clerk")
    inp.pipeline_snapshot = snap
    msg = _present_no_match_fallback_v2(inp)
    # Pattern 1 upload-ask phrasing is present.
    assert "upload" in msg.lower()
    # Market summary tokens are NOT in the Pattern 1 message — they
    # belong only to the SHAPE 2 enhanced branch (resume already in).
    assert "43 active" not in msg


# ============================================================================
# §5.12 — Step 11d: LLM happy-path pipes through pipeline_snapshot
# ============================================================================
# Step 9's SHAPE 2 enhanced was incomplete — the deterministic
# fallback (`_present_no_match_fallback_v2`) got the market summary
# but the LLM happy path that drives the actual response had no
# signal. Step 11d threads `inp.pipeline_snapshot` through the
# user_block builder for the LLM AND adds a SHAPE 2 ENHANCED rule
# to OUTCOME_RESPONDER_PROMPT so the LLM weaves the market summary
# when no direct + no adjacency matches are found.

def test_outcome_prompt_has_shape_2_enhanced_rule():
    """OUTCOME_RESPONDER_PROMPT must include the SHAPE 2 ENHANCED
    rule so the LLM weaves the market summary when conditions hold.
    Substrings are kept short to survive the docstring's line-wrap
    indentation."""
    from skillbridge.chat.prompts import OUTCOME_RESPONDER_PROMPT

    assert "SHAPE 2 ENHANCED" in OUTCOME_RESPONDER_PROMPT
    body = OUTCOME_RESPONDER_PROMPT.lower()
    # Trigger conditions named: total_active_jobs > 0 AND
    # top_sectors / top_employers non-empty.
    assert "total_active_jobs" in body
    assert "top_sectors" in body
    assert "top_employers" in body
    # Two-way invitation phrasing is present (fragments — the literal
    # phrase spans line wraps inside the docstring).
    assert "two-way invitation" in body
    assert "one of those sectors" in body
    # SCCC referral framing carried forward (as fallback when
    # snapshot is missing, NOT as the bare close when enhanced fires).
    assert "sault community career centre" in body


def test_outcome_prompt_shape_2_enhanced_locks_verbatim_quoting():
    """SHAPE 2 ENHANCED must instruct the LLM to QUOTE the sector
    and employer names verbatim — no abbreviation, no synonym
    substitution. This pins the grounding rule that prevents
    the LLM from saying "healthcare and trades" when the snapshot
    says "healthcare" and "trades and transport"."""
    from skillbridge.chat.prompts import OUTCOME_RESPONDER_PROMPT

    body = OUTCOME_RESPONDER_PROMPT.lower()
    assert "verbatim" in body
    assert "do not invent sectors" in body or "do not invent" in body


def test_user_block_v2_serializes_pipeline_snapshot_when_present():
    """`_build_user_block_v2` must emit a PIPELINE_SNAPSHOT block
    containing the new market-summary fields when the snapshot is
    set on ResponderV2Input."""
    from skillbridge.chat.responder import _build_user_block_v2
    from skillbridge.chat.url_views import SanitizedResponderView
    from types import MappingProxyType

    snap = PipelineSnapshot(
        total_active_jobs=43,
        last_publish_at_text="2026-06-17 09:00 ET",
        top_sectors=("healthcare", "trades and transport"),
        top_employers=("Health Sciences North", "Algoma Family Services"),
    )
    inp = _resp_input(offer=False, target_role_text="accounting clerk")
    inp.pipeline_snapshot = snap
    # Note: _resp_input already builds ArbiterDecision with
    # final_move="present_no_match" (the no-match shape we're
    # testing). The decision is frozen — cannot mutate; the
    # _resp_input default IS the test condition.
    # Minimal view — only the fields the builder reads on this path.
    view = SanitizedResponderView(
        prompt_results=(),
        fallback_results=(),
        prompt_present_matches_training_flat=(),
        prompt_present_matches_training_groups=(),
        fallback_present_matches_training_by_job=MappingProxyType({}),
        prompt_explain_gap_training_flat=(),
        fallback_explain_gap_training_flat=(),
        prompt_present_near_miss_training_flat=(),
        prompt_explain_remaining_gaps_training_flat=(),
        prompt_adjacent_recommendations=None,
        fallback_adjacent_recommendations=(),
        prompt_adjacent_role=None,
        fallback_adjacent_role=None,
        rejected_source_urls=(),
        prompt_urls=frozenset(),
        fallback_urls=frozenset(),
    )
    block = _build_user_block_v2(inp, view)

    assert "PIPELINE_SNAPSHOT:" in block
    assert "43" in block
    assert "healthcare" in block
    assert "trades and transport" in block
    assert "Health Sciences North" in block
    assert "Algoma Family Services" in block


# ============================================================================
# §5.13 — Step 11e: RELATED_ROLES_EXHAUSTED signal
# ============================================================================
# When a turn lands in present_no_match AND the user has uploaded a
# resume, the engine has by construction already attempted the
# related-role (CP5) adjacency lookup. The handler emits
# RELATED_ROLES_EXHAUSTED: yes into the user_block so the LLM:
#   (a) acknowledges the search ran instead of generic "I don't see"
#   (b) does NOT offer to look at related roles in the closing
# Without this signal the v1 SHAPE 2 "Optional: offer one
# alternative angle" rule fires and the closing reads "want me to
# look at related roles?" — but the engine ALREADY tried and got 0.
# Infinite-offer loop. Surfaced in live verify 2026-06-17.

def test_user_block_v2_emits_related_roles_exhausted_on_no_match_with_resume():
    """When `final_move == present_no_match` AND `resume_facts` has
    content, the user_block MUST include RELATED_ROLES_EXHAUSTED:
    yes — the signal SHAPE 2 uses to know the related-role search
    was already tried."""
    from skillbridge.chat.responder import _build_user_block_v2
    from skillbridge.chat.url_views import SanitizedResponderView
    from types import MappingProxyType

    inp = _resp_input(offer=False, target_role_text="accounting clerk")
    # _resp_input default final_move IS present_no_match.
    # Resume facts with at least one entry so the helper returns True.
    inp.resume_facts = {
        "work_history": [
            {"title": "Bookkeeper", "employer": "Algoma Family Services"},
        ],
    }
    view = SanitizedResponderView(
        prompt_results=(),
        fallback_results=(),
        prompt_present_matches_training_flat=(),
        prompt_present_matches_training_groups=(),
        fallback_present_matches_training_by_job=MappingProxyType({}),
        prompt_explain_gap_training_flat=(),
        fallback_explain_gap_training_flat=(),
        prompt_present_near_miss_training_flat=(),
        prompt_explain_remaining_gaps_training_flat=(),
        prompt_adjacent_recommendations=None,
        fallback_adjacent_recommendations=(),
        prompt_adjacent_role=None,
        fallback_adjacent_role=None,
        rejected_source_urls=(),
        prompt_urls=frozenset(),
        fallback_urls=frozenset(),
    )
    block = _build_user_block_v2(inp, view)
    assert "RELATED_ROLES_EXHAUSTED: yes" in block


def test_user_block_v2_omits_related_roles_exhausted_when_no_resume():
    """When no resume is on file (Pattern 1 territory), the signal
    MUST NOT fire — Pattern 1's upload ask is a different closing
    branch entirely. RELATED_ROLES_EXHAUSTED is specific to SHAPE 2
    (resume uploaded) cases."""
    from skillbridge.chat.responder import _build_user_block_v2
    from skillbridge.chat.url_views import SanitizedResponderView
    from types import MappingProxyType

    inp = _resp_input(offer=False, target_role_text="accounting clerk")
    # No resume facts → Pattern 1 territory.
    inp.resume_facts = None
    view = SanitizedResponderView(
        prompt_results=(),
        fallback_results=(),
        prompt_present_matches_training_flat=(),
        prompt_present_matches_training_groups=(),
        fallback_present_matches_training_by_job=MappingProxyType({}),
        prompt_explain_gap_training_flat=(),
        fallback_explain_gap_training_flat=(),
        prompt_present_near_miss_training_flat=(),
        prompt_explain_remaining_gaps_training_flat=(),
        prompt_adjacent_recommendations=None,
        fallback_adjacent_recommendations=(),
        prompt_adjacent_role=None,
        fallback_adjacent_role=None,
        rejected_source_urls=(),
        prompt_urls=frozenset(),
        fallback_urls=frozenset(),
    )
    block = _build_user_block_v2(inp, view)
    assert "RELATED_ROLES_EXHAUSTED" not in block


def test_outcome_prompt_has_related_roles_exhausted_rule():
    """OUTCOME_RESPONDER_PROMPT SHAPE 2 must teach the LLM what to
    do when RELATED_ROLES_EXHAUSTED is yes: 3-movement structure
    (acknowledgment, market panorama, training-offer close)."""
    from skillbridge.chat.prompts import OUTCOME_RESPONDER_PROMPT

    assert "RELATED_ROLES_EXHAUSTED" in OUTCOME_RESPONDER_PROMPT
    body = OUTCOME_RESPONDER_PROMPT.lower()
    # Movement A: acknowledgment. Substrings kept short to survive
    # docstring line-wrap indentation.
    assert "movement a" in body
    assert "acknowledgment" in body or "acknowledge" in body
    assert "checked for" in body  # "checked for\n   related roles" — fragmented
    # Movement B: market panorama.
    assert "movement b" in body
    assert "panorama" in body or "pipeline_snapshot summary" in body
    # Movement C: 3 sub-movements (Step 11h personalization).
    assert "movement c" in body
    # C1: skill acknowledgment.
    assert "c1" in body
    assert "skill acknowledgment" in body
    assert "resume_facts" in body
    # C2: gap callout.
    assert "c2" in body
    assert "gap callout" in body
    assert "cp4_primary_gap" in body
    # C3: training-direction close.
    assert "c3" in body
    assert "training-offer close" in body or "training-direction" in body
    # Step 11h locked C3 phrasing fragments (survive line wrap).
    assert "improve your skills gap" in body
    assert "training directions" in body
    assert "do you want?" in body
    # Step 11h verbatim-grounding rules.
    assert "verbatim" in body
    # Step 11h HARD RULES.
    assert "must not name" in body  # no specific provider
    # Hard rules — no related-roles offer (carried from 11f).
    assert "must not offer to look at related" in body
    assert "infinite-offer" in body


def test_user_block_v2_serializes_cp4_primary_gap_when_present():
    """Step 11h: when ResponderV2Input.cp4_primary_gap is set, the
    user_block MUST include `CP4_PRIMARY_GAP: <name>` so the LLM
    can quote it verbatim in Movement C2."""
    from skillbridge.chat.responder import _build_user_block_v2
    from skillbridge.chat.url_views import SanitizedResponderView
    from types import MappingProxyType

    inp = _resp_input(offer=False, target_role_text="accounting clerk")
    inp.cp4_primary_gap = "confidentiality handling"
    view = SanitizedResponderView(
        prompt_results=(),
        fallback_results=(),
        prompt_present_matches_training_flat=(),
        prompt_present_matches_training_groups=(),
        fallback_present_matches_training_by_job=MappingProxyType({}),
        prompt_explain_gap_training_flat=(),
        fallback_explain_gap_training_flat=(),
        prompt_present_near_miss_training_flat=(),
        prompt_explain_remaining_gaps_training_flat=(),
        prompt_adjacent_recommendations=None,
        fallback_adjacent_recommendations=(),
        prompt_adjacent_role=None,
        fallback_adjacent_role=None,
        rejected_source_urls=(),
        prompt_urls=frozenset(),
        fallback_urls=frozenset(),
    )
    block = _build_user_block_v2(inp, view)
    assert "CP4_PRIMARY_GAP: confidentiality handling" in block


def test_user_block_v2_omits_cp4_primary_gap_when_none():
    """When cp4_primary_gap is None (CP4 returned no recommendation),
    the user_block MUST NOT include the line at all — preserves
    Step 11h's "skip C2 when absent" contract."""
    from skillbridge.chat.responder import _build_user_block_v2
    from skillbridge.chat.url_views import SanitizedResponderView
    from types import MappingProxyType

    inp = _resp_input(offer=False, target_role_text="accounting clerk")
    inp.cp4_primary_gap = None
    view = SanitizedResponderView(
        prompt_results=(),
        fallback_results=(),
        prompt_present_matches_training_flat=(),
        prompt_present_matches_training_groups=(),
        fallback_present_matches_training_by_job=MappingProxyType({}),
        prompt_explain_gap_training_flat=(),
        fallback_explain_gap_training_flat=(),
        prompt_present_near_miss_training_flat=(),
        prompt_explain_remaining_gaps_training_flat=(),
        prompt_adjacent_recommendations=None,
        fallback_adjacent_recommendations=(),
        prompt_adjacent_role=None,
        fallback_adjacent_role=None,
        rejected_source_urls=(),
        prompt_urls=frozenset(),
        fallback_urls=frozenset(),
    )
    block = _build_user_block_v2(inp, view)
    assert "CP4_PRIMARY_GAP" not in block


def test_outcome_prompt_exhausted_close_replaces_sectors_close():
    """When RELATED_ROLES_EXHAUSTED is on, the prompt MUST instruct
    the LLM that the closing is the training offer, NOT the
    sectors / SCCC two-way ask from sub-rule 3."""
    from skillbridge.chat.prompts import OUTCOME_RESPONDER_PROMPT

    body = OUTCOME_RESPONDER_PROMPT.lower()
    # Training-offer is named as the locked close.
    assert "training-offer" in body or "want to see what" in body
    # The sectors/SCCC close is named as REPLACED.
    assert "replaced" in body or "do not close with" in body


# ============================================================================
# §5.14 — Step 11f: AR-6b soft-offer suppression on RELATED_ROLES_EXHAUSTED
# ============================================================================
# `_maybe_append_soft_offer` is the post-LLM hook that appends the
# AR-6b adjacency soft-offer ("If you'd like, I can also look for
# related roles... just say what other roles?") to present_no_match
# replies. On a Pattern 2 yes-consent / Pattern 3 auto-fire turn
# that hit empty sideways, that soft-offer is exactly the
# infinite-offer loop we just diagnosed. Step 11f adds a suppression
# at the top of the soft-offer helper when:
#   - final.final_move == "present_no_match"
#   - staged.resume_facts_json (resume uploaded — SHAPE 2 territory)

def test_soft_offer_suppressed_on_no_match_with_resume():
    """Step 11f: when the user has a resume on file AND the turn
    landed in present_no_match, the AR-6b soft-offer must NOT be
    appended to the reply — the related-role search was already
    tried by the engine and returned 0."""
    from skillbridge.chat.handler import _maybe_append_soft_offer
    from skillbridge.chat.arbiter import ArbiterDecision

    sp = StagedProfile.new("s1")
    sp.target_role_text = "accounting clerk"
    sp.resume_facts_json = {"skills": [{"skill_name": "bookkeeping"}]}
    decision = ArbiterDecision(
        final_move="present_no_match",
        reason_code="zero_matches_in_dataset",
        tone="honest_redirect",
        arbiter_action="resolved_to_no_match",
    )
    base_reply = "I checked for related roles but didn't find more."
    result = _maybe_append_soft_offer(
        reply=base_reply, staged=sp, final=decision,
        results=[], pending_offer=False, prior_empty_adjacency=False,
    )
    # Reply MUST be unchanged — no soft-offer line appended.
    assert result == base_reply
    # The pending_adjacent_offer flag MUST NOT be set as a side effect.
    assert sp.pending_adjacent_offer is False


def test_soft_offer_suppressed_on_no_match_without_resume():
    """Step 11l (2026-06-18): widen Step 11f's suppression to fire
    on EVERY present_no_match turn, regardless of resume state.

    Rationale: when the LLM goes through OUTCOME_RESPONDER_PROMPT
    SHAPE 1, the closing IS the Pattern 1 upload ask (framed around
    finding related roles per Step 11k). Appending the AR-6b
    soft-offer line ("or say what other roles?") creates two
    competing offers — the "splits attention" anti-pattern
    Pattern 1's structural rules forbid.

    Both no-resume (Pattern 1 territory) AND resume-uploaded (Step
    11f territory) present_no_match turns now suppress AR-6b
    uniformly."""
    from skillbridge.chat.handler import _maybe_append_soft_offer
    from skillbridge.chat.arbiter import ArbiterDecision

    sp = StagedProfile.new("s1")
    sp.target_role_text = "accounting clerk"
    sp.resume_facts_json = None  # no resume → Pattern 1 territory
    decision = ArbiterDecision(
        final_move="present_no_match",
        reason_code="zero_matches_in_dataset",
        tone="honest_redirect",
        arbiter_action="resolved_to_no_match",
    )
    base_reply = (
        "I couldn't find a strong fit yet. Want to upload your CV "
        "so I can find related roles your skills fit?"
    )
    result = _maybe_append_soft_offer(
        reply=base_reply, staged=sp, final=decision,
        results=[], pending_offer=False, prior_empty_adjacency=False,
    )
    # Reply MUST be unchanged — no soft-offer line appended.
    assert result == base_reply
    # The pending_adjacent_offer flag MUST NOT be set as a side effect.
    assert sp.pending_adjacent_offer is False


def test_user_block_v2_omits_pipeline_snapshot_when_none():
    """When `inp.pipeline_snapshot` is None (the default), no
    PIPELINE_SNAPSHOT block appears in the user_block — preserves
    the prior behavior on non-snapshot turns."""
    from skillbridge.chat.responder import _build_user_block_v2
    from skillbridge.chat.url_views import SanitizedResponderView
    from types import MappingProxyType

    inp = _resp_input(offer=False, target_role_text="accounting clerk")
    inp.pipeline_snapshot = None
    # _resp_input default already builds present_no_match decision.
    view = SanitizedResponderView(
        prompt_results=(),
        fallback_results=(),
        prompt_present_matches_training_flat=(),
        prompt_present_matches_training_groups=(),
        fallback_present_matches_training_by_job=MappingProxyType({}),
        prompt_explain_gap_training_flat=(),
        fallback_explain_gap_training_flat=(),
        prompt_present_near_miss_training_flat=(),
        prompt_explain_remaining_gaps_training_flat=(),
        prompt_adjacent_recommendations=None,
        fallback_adjacent_recommendations=(),
        prompt_adjacent_role=None,
        fallback_adjacent_role=None,
        rejected_source_urls=(),
        prompt_urls=frozenset(),
        fallback_urls=frozenset(),
    )
    block = _build_user_block_v2(inp, view)
    assert "PIPELINE_SNAPSHOT:" not in block


# ============================================================================
# §6 — skills_text recovery (2026-06-17)
# ============================================================================
# Live repro (2026-06-17 accounting clerk turn): user explicitly listed 12
# real skills. Per-skill grounding passed 12/12 — staged.skills survived
# in full. But the LLM extractor's slot-level skills_text evidence wasn't
# a verbatim substring (Haiku paraphrased the summary), so the slot was
# dropped with `raw_keys_dropped=['ungrounded:skills_text']`. Change C's
# skills_text_present guard read False, enough_to_match stayed False,
# and the engine refused to run despite 12 grounded skills. The user got
# another clarifying question instead of strong matches.
#
# Fix: `_maybe_recover_skills_text_slot` in handler.py back-fills the slot
# when the extractor explicitly signalled "user was listing skills" (slot
# attempted + dropped) AND >=3 per-skill items grounded from THIS turn's
# message. Phantom-skill protection is preserved because phantom prose
# ("Completed Truck and Coach apprenticeship at Sault College") doesn't
# trigger 'ungrounded:skills_text' — the extractor doesn't claim the user
# was listing skills there.

from skillbridge.chat.handler import _maybe_recover_skills_text_slot
from skillbridge.chat import extractor as chat_extractor


def _extraction(skills_grounded: int, *, ungrounded_skills_text: bool):
    """Build a minimal ExtractionResult shaped like the real one. Only
    raw_keys_dropped and skills are read by the recovery helper."""
    skills_list = [
        chat_extractor.ExtractedSkill(
            skill_name=f"skill_{i}", raw_phrase=f"raw {i}", confidence=0.85,
        )
        for i in range(skills_grounded)
    ]
    dropped = ["ungrounded:skills_text"] if ungrounded_skills_text else []
    return chat_extractor.ExtractionResult(
        fields={}, skills=skills_list, declined=[], off_topic=False,
        raw_keys_dropped=dropped,
    )


def test_skills_text_recovery_fires_on_ungrounded_plus_3_grounded():
    """The exact 2026-06-17 live bug: extractor dropped skills_text
    as ungrounded, but >=3 per-skill items grounded. Recovery MUST
    back-fill staged.skills_text so the engine can run on the user's
    real skill claims."""
    sp = StagedProfile.new("s1")
    sp.target_role_text = "accounting clerk"
    extraction = _extraction(skills_grounded=12, ungrounded_skills_text=True)
    message = (
        "I have bookkeeping and journal entry posting experience. "
        "Strong with account reconciliation, invoice processing, "
        "accounts payable, accounts receivable, bank reconciliation, "
        "payroll. Use QuickBooks and Excel daily."
    )
    fired = _maybe_recover_skills_text_slot(
        staged=sp, extraction=extraction, message=message,
    )
    assert fired is True
    assert sp.skills_text == message[:500]


def test_skills_text_recovery_does_not_fire_without_ungrounded_signal():
    """Phantom-skill protection: when the LLM extractor did NOT claim
    the user was listing skills (no 'ungrounded:skills_text' in
    raw_keys_dropped), recovery must NOT back-fill, even if per-skill
    items grounded. This is the case that Change C was designed to
    block — phantom skills lifted from experience prose."""
    sp = StagedProfile.new("s1")
    sp.target_role_text = "truck driver"
    extraction = _extraction(skills_grounded=5, ungrounded_skills_text=False)
    fired = _maybe_recover_skills_text_slot(
        staged=sp, extraction=extraction,
        message=(
            "Completed Truck and Coach Technician apprenticeship "
            "at Sault College in 2019."
        ),
    )
    assert fired is False
    assert not sp.skills_text


def test_skills_text_recovery_requires_at_least_3_skills():
    """One or two grounded skill items aren't enough evidence to
    treat the message as a skills list. The 3+ threshold matches
    the chat_skills_sufficient gate in _compute_enough_to_match —
    keeping the recovery aligned with the engine-run threshold."""
    sp = StagedProfile.new("s1")
    sp.target_role_text = "finance clerk"
    extraction = _extraction(skills_grounded=2, ungrounded_skills_text=True)
    fired = _maybe_recover_skills_text_slot(
        staged=sp, extraction=extraction,
        message="I know QuickBooks and Excel.",
    )
    assert fired is False
    assert not sp.skills_text


def test_skills_text_recovery_does_not_overwrite_existing_slot():
    """If staged.skills_text already holds a value from a prior turn,
    recovery must NOT overwrite it. The recovery is a one-shot
    correction for the current turn's dropped slot, not a turn-by-
    turn rewrite."""
    sp = StagedProfile.new("s1")
    sp.target_role_text = "accounting clerk"
    sp.skills_text = "previous turn skills statement"
    extraction = _extraction(skills_grounded=10, ungrounded_skills_text=True)
    fired = _maybe_recover_skills_text_slot(
        staged=sp, extraction=extraction,
        message="I have payroll, AP, AR, bank reconciliation, QuickBooks.",
    )
    assert fired is False
    assert sp.skills_text == "previous turn skills statement"


def test_skills_text_recovery_truncates_long_messages():
    """Long messages must be truncated to 500 chars to match the
    existing fallback_fill safety-net behavior (handler.py line ~4140).
    Avoids unbounded slot values flowing downstream."""
    sp = StagedProfile.new("s1")
    sp.target_role_text = "accounting clerk"
    extraction = _extraction(skills_grounded=4, ungrounded_skills_text=True)
    message = "A" * 1200  # well over 500
    fired = _maybe_recover_skills_text_slot(
        staged=sp, extraction=extraction, message=message,
    )
    assert fired is True
    assert len(sp.skills_text) == 500
    assert sp.skills_text == "A" * 500


def test_skills_text_recovery_clears_enough_to_match_after_fix():
    """End-to-end: the bug's symptom was `enough_to_match=False` even
    with 12 grounded skills. After the recovery fills skills_text,
    `_compute_enough_to_match` must return True via the
    chat_skills_sufficient branch.

    This test pins the FULL chain: ungrounded-slot drop → recovery →
    skills_text_present=True → enough_to_match=True. A regression in
    either the recovery or the truth_summary gate breaks this."""
    from skillbridge.chat.truth_summary import _compute_enough_to_match
    from skillbridge.chat.truth_summary import ResumeFactsSummary

    sp = StagedProfile.new("s1")
    sp.target_role_text = "accounting clerk"
    extraction = _extraction(skills_grounded=12, ungrounded_skills_text=True)
    msg = (
        "I have bookkeeping, journal entry posting, account "
        "reconciliation, invoice processing, AP, AR, bank "
        "reconciliation, payroll, QuickBooks, Excel, financial "
        "reporting, attention to detail."
    )
    assert _maybe_recover_skills_text_slot(
        staged=sp, extraction=extraction, message=msg,
    )

    # Simulate the gate after recovery: skills_text_present=True
    # (the slot is filled). chat_skill_count=12 clears the >=3 threshold.
    skills_text_present = (
        isinstance(sp.skills_text, str) and bool(sp.skills_text.strip())
    )
    enough, reason, usable = _compute_enough_to_match(
        target_role_specificity="specific",
        resume_parse_quality="no_resume",
        counts=ResumeFactsSummary(),
        chat_skill_count=12,
        user_intent_signal="neutral",
        skills_text_present=skills_text_present,
    )
    assert enough is True
    assert reason == "chat_skills_sufficient"
    assert usable is True
