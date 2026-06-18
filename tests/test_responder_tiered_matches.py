"""AR-9.feat.coach-tiers CP2 step 3 — responder branch for
`present_tiered_matches`.

Pins:
  - `compose_response_v2` dispatches to the tiered branch when
    `decision.final_move == "present_tiered_matches"`;
  - Tiered branch does NOT consult `results` / `training_by_job` /
    `band_signal` / `next_skill` (legacy fields);
  - When `tier_evidence is None`, falls back to the deterministic
    empty-state body without an LLM call;
  - When LLM is disabled, returns the deterministic fallback;
  - When LLM returns empty / falsy, falls back;
  - When policy gate rejects, falls back;
  - When policy gate passes, returns the LLM reply verbatim;
  - user_block contains the locked EVIDENCE PACKAGE sections;
  - user_block omits salary_text (option B);
  - user_block uses url.raw for every URL field;
  - Tiered policy gate uses `GROUNDED_TERMS` (tier-view) not
    `training_by_job`;
  - Tiered policy gate enforces URL grounding, salary defense-in-depth,
    consent, region patterns;
  - The branch never builds the legacy v2 view.
"""
from __future__ import annotations

import json
from datetime import date
from unittest.mock import patch

import pytest

from skillbridge.chat.arbiter import ArbiterDecision
from skillbridge.chat.pipeline_snapshot import PipelineSnapshot
from skillbridge.chat.responder import (
    ResponderV2Input,
    _build_user_block_for_tiered_matches,
    _compose_tiered_matches_response,
    _empty_tiered_view,
    _policy_ok_tiered_matches,
    compose_response_v2,
)
from skillbridge.chat.tiered_evidence import (
    AdjacentJob,
    JobFacts,
    PrioritizedGap,
    StretchMatch,
    StrongMatch,
    TieredEvidence,
    TrainingOption,
    TransferablePair,
)
from skillbridge.chat.url_policy import Validated, validate
from skillbridge.chat.url_views import (
    build_sanitized_responder_view_for_tiered_matches,
)
from skillbridge.match.alignment import SkillAlignment

pytestmark = pytest.mark.nodb


# =========================================================================
# Fixtures
# =========================================================================
def _validated(raw: str) -> Validated:
    result = validate(raw)
    assert isinstance(result, Validated), raw
    return result


def _alignment(user="QuickBooks", job="QuickBooks") -> SkillAlignment:
    return SkillAlignment(
        user_skill=user, job_requirement=job,
        stage="exact", source="required",
        is_normalized_equal=(user == job),
    )


def _facts() -> JobFacts:
    return JobFacts(
        posted_date=date(2026, 6, 11),
        posted_days_ago=3,
        location="Sault Ste. Marie, ON",
        employment_type="full-time",
        salary_text="$22-24/hr",   # preserved on view; omitted in serialization
    )


def _strong() -> StrongMatch:
    return StrongMatch(
        job_id="s1", title="Accounts Payable Clerk", employer="Diamond J",
        location="Sault Ste. Marie, ON", noc_code="13102",
        url=_validated("https://example.com/strong"),
        job_facts=_facts(),
        skill_alignment=(_alignment(),),
        non_blocking_gaps=(),
        credential_warning_text=None,
        strength_claim_text="competitive_match",
    )


def _stretch() -> StretchMatch:
    return StretchMatch(
        job_id="w1", title="Junior Accountant", employer="Algoma Office",
        location="Sault Ste. Marie, ON", noc_code="11100",
        url=_validated("https://example.com/stretch"),
        job_facts=_facts(),
        skill_alignment=(_alignment(),),
        prioritized_gaps=(
            PrioritizedGap(
                job_requirement="account reconciliation",
                category="required", priority=1, blocker=False,
                training_options=(
                    TrainingOption(
                        provider="Sault College",
                        title="Bookkeeping",
                        url=_validated("https://example.com/train"),
                        format="online",
                        duration_text="6 weeks",
                    ),
                ),
            ),
        ),
        credential_warning_text=None,
        strength_claim_text="close_with_named_gap",
    )


def _adjacent() -> AdjacentJob:
    return AdjacentJob(
        job_id="adj1", title="Payroll Administrator", employer="North Star",
        location="Sault Ste. Marie, ON", noc_code="12102",
        url=_validated("https://example.com/adj"),
        job_facts=_facts(),
        skill_alignment=(_alignment(),),
        transferable_pairs=(
            TransferablePair(
                user_skill="QuickBooks", applies_to="QuickBooks",
                stage="exact",
            ),
        ),
        important_gaps=("ROE filing",),
        credential_warning_text=None,
        why_adjacent="same_noc_minor_group",
        strength_claim_text="transferable_lane",
    )


def _evidence(strong=(), stretch=(), adjacent=()) -> TieredEvidence:
    return TieredEvidence(
        apply_today=tuple(strong),
        worth_a_try=tuple(stretch),
        sideways_move=tuple(adjacent),
    )


def _decision(move="present_tiered_matches") -> ArbiterDecision:
    return ArbiterDecision(
        final_move=move,
        reason_code="tiered_matches_found",
        tone="brief_confident",
        arbiter_action="resolved_to_tiered_matches",
    )


def _input(
    *, tier_evidence=None, pipeline_snapshot=None,
    user_message="show me jobs",
    requires_consent=False,
) -> ResponderV2Input:
    return ResponderV2Input(
        user_message=user_message,
        decision=_decision(),
        results=[],
        training_by_job={},
        next_skill=(None, 0),
        band_signal="strong_or_good",
        requires_consent=requires_consent,
        target_role_text="accounting clerk",
        tier_evidence=tier_evidence,
        pipeline_snapshot=pipeline_snapshot,
    )


# =========================================================================
# Dispatch — compose_response_v2 routes to the tiered branch
# =========================================================================
def test_compose_response_v2_routes_to_tiered_branch():
    """The dispatch in compose_response_v2 must route final_move ==
    'present_tiered_matches' to _compose_tiered_matches_response."""
    ev = _evidence(strong=[_strong()])
    inp = _input(tier_evidence=ev)
    # Mocking is_enabled to False forces the deterministic fallback path.
    with patch("skillbridge.chat.responder.is_enabled", return_value=False):
        reply = compose_response_v2(inp)
    # Fallback rendered the three-tier prose with the Apply today header.
    assert "**Strong match — apply today**" in reply


def test_tiered_branch_does_not_consult_legacy_results_field():
    """The tiered branch must NOT consult inp.results — the dispatch
    must work even when `results=[]`."""
    inp = _input(tier_evidence=_evidence(strong=[_strong()]))
    inp.results = []   # explicit
    inp.training_by_job = {}   # explicit
    with patch("skillbridge.chat.responder.is_enabled", return_value=False):
        reply = compose_response_v2(inp)
    # scoring-v6 (2026-06-17): heading is "Strong match — apply today";
    # substring check uses the new prefix.
    assert "Strong match" in reply


# =========================================================================
# Fallback paths
# =========================================================================
def test_falls_back_to_empty_state_when_tier_evidence_missing():
    """Handler contract violation: dispatched present_tiered_matches
    without supplying tier_evidence. Responder logs the violation and
    renders the deterministic empty-state body — no LLM call."""
    inp = _input(tier_evidence=None)
    with patch(
        "skillbridge.chat.responder.is_enabled", return_value=True,
    ) as mock_is_enabled, patch(
        "skillbridge.chat.responder.call",
    ) as mock_call:
        reply = _compose_tiered_matches_response(inp)
    assert "Nothing on the board matches yet." in reply
    # The LLM was never invoked
    mock_call.assert_not_called()
    # is_enabled was not consulted before the fallback path returned
    # (defense in depth — we want to short-circuit before any LLM gate)
    assert mock_is_enabled.call_count == 0


def test_falls_back_when_llm_disabled():
    inp = _input(tier_evidence=_evidence(strong=[_strong()]))
    with patch(
        "skillbridge.chat.responder.is_enabled", return_value=False,
    ):
        reply = _compose_tiered_matches_response(inp)
    assert "**Strong match — apply today**" in reply
    assert "Diamond J" in reply or "Accounts Payable Clerk" in reply


def test_falls_back_when_llm_returns_empty_string():
    inp = _input(tier_evidence=_evidence(strong=[_strong()]))
    with patch(
        "skillbridge.chat.responder.is_enabled", return_value=True,
    ), patch(
        "skillbridge.chat.responder.call", return_value="",
    ):
        reply = _compose_tiered_matches_response(inp)
    # scoring-v6 (2026-06-17): heading is "Strong match — apply today".
    assert "Strong match" in reply


def test_falls_back_when_policy_rejects():
    """LLM returns a reply that the policy gate rejects (e.g., names a
    salary). The deterministic fallback fires instead."""
    inp = _input(tier_evidence=_evidence(strong=[_strong()]))
    # LLM reply names "$22/hr" — policy must reject (salary defense)
    bad_reply = "Diamond J pays $22/hr."
    with patch(
        "skillbridge.chat.responder.is_enabled", return_value=True,
    ), patch(
        "skillbridge.chat.responder.call", return_value=bad_reply,
    ):
        reply = _compose_tiered_matches_response(inp)
    assert reply != bad_reply
    # scoring-v6 (2026-06-17): heading is "Strong match — apply today".
    assert "Strong match" in reply
    assert "$" not in reply


def test_returns_llm_reply_when_policy_passes():
    """A well-formed reply containing the required tier shape and only
    grounded URLs is returned verbatim. The reply MUST carry the
    Apply-today heading (since Apply-today is populated), the
    authorized strength phrase, and the closing question — the
    structural validator added in the step-3 review enforces this."""
    inp = _input(tier_evidence=_evidence(strong=[_strong()]))
    good_reply = (
        "Here's what's on the board.\n\n"
        "**Strong match — apply today**\n\n"
        "You'd be competitive for this one. Diamond J is hiring an "
        "Accounts Payable Clerk. They ask for QuickBooks, which you "
        "have. https://example.com/strong\n\n"
        "Want me to dig into any of these further?"
    )
    with patch(
        "skillbridge.chat.responder.is_enabled", return_value=True,
    ), patch(
        "skillbridge.chat.responder.call", return_value=good_reply,
    ):
        reply = _compose_tiered_matches_response(inp)
    assert reply == good_reply


# =========================================================================
# user_block shape
# =========================================================================
def test_user_block_includes_locked_evidence_package_sections():
    inp = _input(
        tier_evidence=_evidence(
            strong=[_strong()], stretch=[_stretch()], adjacent=[_adjacent()],
        ),
        pipeline_snapshot=PipelineSnapshot(
            total_active_jobs=43,
            last_publish_at_text="2026-06-14 06:14 ET",
        ),
    )
    view = build_sanitized_responder_view_for_tiered_matches(inp.tier_evidence)
    block = _build_user_block_for_tiered_matches(inp, view)
    assert "USER_MESSAGE:" in block
    assert "TARGET_ROLE:" in block
    assert "STRONG_MATCHES:" in block
    assert "STRETCH_MATCHES:" in block
    assert "ADJACENT_JOBS:" in block
    assert "PIPELINE_SNAPSHOT:" in block


def test_user_block_omits_salary_text_per_option_b():
    """Salary is preserved on the view's job_facts but MUST NOT appear
    in the serialized user_block. Option B."""
    inp = _input(tier_evidence=_evidence(strong=[_strong()]))
    view = build_sanitized_responder_view_for_tiered_matches(inp.tier_evidence)
    block = _build_user_block_for_tiered_matches(inp, view)
    assert "$22-24/hr" not in block
    assert "salary_text" not in block
    # Belt + suspenders: no `$` token anywhere in the serialized payload.
    assert "$" not in block


def test_user_block_uses_url_raw_for_strong_match():
    inp = _input(tier_evidence=_evidence(strong=[_strong()]))
    view = build_sanitized_responder_view_for_tiered_matches(inp.tier_evidence)
    block = _build_user_block_for_tiered_matches(inp, view)
    # The strong job's URL must appear as a literal raw string in JSON.
    assert "https://example.com/strong" in block


def test_user_block_uses_url_raw_for_training_option():
    inp = _input(tier_evidence=_evidence(stretch=[_stretch()]))
    view = build_sanitized_responder_view_for_tiered_matches(inp.tier_evidence)
    block = _build_user_block_for_tiered_matches(inp, view)
    assert "https://example.com/train" in block


def test_user_block_omits_pipeline_snapshot_when_none():
    inp = _input(tier_evidence=_evidence(strong=[_strong()]))
    view = build_sanitized_responder_view_for_tiered_matches(inp.tier_evidence)
    block = _build_user_block_for_tiered_matches(inp, view)
    assert "PIPELINE_SNAPSHOT:" not in block


def test_user_block_does_not_leak_arbiter_action_or_notes():
    """Same V2 prompt discipline applies — operational fields must
    NEVER reach the LLM."""
    inp = _input(tier_evidence=_evidence(strong=[_strong()]))
    view = build_sanitized_responder_view_for_tiered_matches(inp.tier_evidence)
    block = _build_user_block_for_tiered_matches(inp, view)
    assert "arbiter_action" not in block
    assert "resolved_to_tiered_matches" not in block


def test_user_block_strong_match_payload_carries_known_fields():
    inp = _input(tier_evidence=_evidence(strong=[_strong()]))
    view = build_sanitized_responder_view_for_tiered_matches(inp.tier_evidence)
    block = _build_user_block_for_tiered_matches(inp, view)
    # Extract the STRONG_MATCHES JSON line (the line right after the
    # "STRONG_MATCHES:" marker).
    lines = block.splitlines()
    strong_idx = lines.index("STRONG_MATCHES:")
    payload = json.loads(lines[strong_idx + 1])
    assert payload["title"] == "Accounts Payable Clerk"
    assert payload["employer"] == "Diamond J"
    assert payload["strength_claim_text"] == "competitive_match"
    assert payload["url"] == "https://example.com/strong"
    assert payload["job_facts"]["employment_type"] == "full-time"
    assert "salary_text" not in payload["job_facts"]


# =========================================================================
# Policy gate
# =========================================================================
def test_policy_rejects_salary_token():
    """Defense-in-depth: even with salary omitted from prompt, if the
    LLM somehow emits `$`, the policy rejects."""
    inp = _input(tier_evidence=_evidence(strong=[_strong()]))
    view = build_sanitized_responder_view_for_tiered_matches(inp.tier_evidence)
    assert _policy_ok_tiered_matches("They pay $22/hr.", inp, view) is False


def test_policy_rejects_empty_reply():
    inp = _input(tier_evidence=_evidence(strong=[_strong()]))
    view = build_sanitized_responder_view_for_tiered_matches(inp.tier_evidence)
    assert _policy_ok_tiered_matches("", inp, view) is False
    assert _policy_ok_tiered_matches("   ", inp, view) is False


def test_policy_rejects_ungrounded_url():
    """A URL not in view.prompt_urls fails URL-grounding."""
    inp = _input(tier_evidence=_evidence(strong=[_strong()]))
    view = build_sanitized_responder_view_for_tiered_matches(inp.tier_evidence)
    bad = "Try https://ungrounded.example/job"
    assert _policy_ok_tiered_matches(bad, inp, view) is False


def _well_formed_strong_reply(url: str = "https://example.com/strong") -> str:
    """A reply that satisfies the structural validator for a
    Strong-only view of `_strong()`. Used by positive tests that
    isolate a single shared safety rule."""
    return (
        "Here's what's on the board.\n\n"
        "**Strong match — apply today**\n\n"
        "You'd be competitive for this one. Diamond J is hiring an "
        f"Accounts Payable Clerk. They ask for QuickBooks, which you "
        f"have. {url}\n\n"
        "Want me to dig into any of these further?"
    )


def test_policy_accepts_grounded_url():
    inp = _input(tier_evidence=_evidence(strong=[_strong()]))
    view = build_sanitized_responder_view_for_tiered_matches(inp.tier_evidence)
    assert _policy_ok_tiered_matches(
        _well_formed_strong_reply(), inp, view,
    ) is True


def test_policy_uses_tier_view_for_provider_grounding():
    """A provider mention that's grounded ONLY via the tier view's
    skill_alignment (not training_by_job) must be exempted. This is
    the QuickBooks case the step-11 policy fix targeted."""
    inp = _input(tier_evidence=_evidence(strong=[_strong()]))
    view = build_sanitized_responder_view_for_tiered_matches(inp.tier_evidence)
    # The reply mentions "QuickBooks" — a known provider in the deny list.
    # It's grounded via skill_alignment (Strong match has QuickBooks alignment).
    # Policy must NOT reject.
    assert _policy_ok_tiered_matches(
        _well_formed_strong_reply(), inp, view,
    ) is True


def test_policy_rejects_forbidden_phrases():
    inp = _input(tier_evidence=_evidence(strong=[_strong()]))
    view = build_sanitized_responder_view_for_tiered_matches(inp.tier_evidence)
    for forbidden in (
        "Check Job Bank",
        "Statistics Canada has data on this",
        "the national average is $42k",
        "StatCan says",
    ):
        assert _policy_ok_tiered_matches(forbidden, inp, view) is False, (
            forbidden
        )


def test_policy_consent_promise_check():
    """When requires_consent is True, consent-promise phrases reject."""
    inp = _input(
        tier_evidence=_evidence(strong=[_strong()]),
        requires_consent=True,
    )
    view = build_sanitized_responder_view_for_tiered_matches(inp.tier_evidence)
    assert _policy_ok_tiered_matches("I'll remember that.", inp, view) is False


def test_policy_consent_promise_only_checked_when_required():
    """When requires_consent is False, consent-promise wording is OK."""
    inp = _input(
        tier_evidence=_evidence(strong=[_strong()]),
        requires_consent=False,
    )
    view = build_sanitized_responder_view_for_tiered_matches(inp.tier_evidence)
    # Plain reply, no consent promise, no forbidden content.
    assert _policy_ok_tiered_matches(
        _well_formed_strong_reply(), inp, view,
    ) is True


# =========================================================================
# Defense-in-depth: empty view shape
# =========================================================================
def test_empty_tiered_view_has_no_tier_records():
    view = _empty_tiered_view()
    assert view.prompt_tiered_apply_today == ()
    assert view.prompt_tiered_worth_a_try == ()
    assert view.prompt_tiered_sideways_move == ()


def test_empty_view_falls_back_to_pure_empty_state_body():
    """Missing tier evidence + no snapshot → only the generic empty body."""
    inp = _input(tier_evidence=None)
    reply = _compose_tiered_matches_response(inp)
    assert "Nothing on the board matches yet." in reply
    # No tier headers anywhere. scoring-v6 (2026-06-17): heading
    # substrings renamed to the 4-label vocabulary.
    assert "Strong match" not in reply
    assert "Good match" not in reply
    assert "Stretch" not in reply
    assert "Explore later" not in reply
    assert "Sideways move" not in reply


# =========================================================================
# Step-3 review High 1 — required tier structure must be validated
# =========================================================================
def test_policy_accepts_reply_with_full_tier_shape():
    inp = _input(tier_evidence=_evidence(strong=[_strong()]))
    view = build_sanitized_responder_view_for_tiered_matches(inp.tier_evidence)
    reply = (
        "Here's what's on the board.\n\n"
        "**Strong match — apply today**\n\n"
        "You'd be competitive for this one. Diamond J is hiring an "
        "Accounts Payable Clerk. They ask for QuickBooks, which you "
        "have. https://example.com/strong\n\n"
        "Want me to dig into any of these further?"
    )
    assert _policy_ok_tiered_matches(reply, inp, view) is True


def test_policy_rejects_internal_token_leakage():
    inp = _input(tier_evidence=_evidence(strong=[_strong()]))
    view = build_sanitized_responder_view_for_tiered_matches(inp.tier_evidence)
    for leaked in (
        "competitive_match", "strongest_current", "transferable_lane",
        "same_noc_minor_group", "skill_evidence",
        "skill_alignment", "prioritized_gaps", "why_adjacent",
        "strength_claim_text", "job_facts", "is_normalized_equal",
    ):
        reply = (
            "**Strong match — apply today**\n"
            "You'd be competitive for this one. Diamond J is hiring an "
            f"Accounts Payable Clerk. The {leaked} matched. "
            "https://example.com/strong\n\n"
            "Want me to dig into any of these further?"
        )
        assert _policy_ok_tiered_matches(reply, inp, view) is False, leaked


# =========================================================================
# Step-3 review High 2 — shared safety rules incompletely copied
# =========================================================================
def test_policy_rejects_slash_hr_pay_token():
    """The exact failure case from the review: '22/hour' and '22/hr'
    slipped through because the gate only checked '$'."""
    inp = _input(tier_evidence=_evidence(strong=[_strong()]))
    view = build_sanitized_responder_view_for_tiered_matches(inp.tier_evidence)
    for bad in (
        "The role pays 22/hour.",
        "The role pays 22/hr.",
        "Rate: 22/HR.",
    ):
        assert _policy_ok_tiered_matches(bad, inp, view) is False, bad


def test_policy_rejects_credential_equivalence_claims():
    """`_CREDENTIAL_EQUIVALENCE_PATTERNS` parity check."""
    inp = _input(tier_evidence=_evidence(strong=[_strong()]))
    view = build_sanitized_responder_view_for_tiered_matches(inp.tier_evidence)
    for bad in (
        "Your foreign licence is equivalent to a Canadian licence.",
        "Get a Canadian equivalent.",
        "WES certification can convert your degree.",
        "Credential recognition is straightforward.",
        "Your degree counts as a Canadian Bachelor's.",
    ):
        assert _policy_ok_tiered_matches(bad, inp, view) is False, bad


def test_policy_rejects_immigration_legal_advice():
    """`_IMMIGRATION_LEGAL_PATTERNS` parity check."""
    inp = _input(tier_evidence=_evidence(strong=[_strong()]))
    view = build_sanitized_responder_view_for_tiered_matches(inp.tier_evidence)
    for bad in (
        "This job can help your Express Entry application.",
        "Check the RCIP eligibility criteria.",
        "You'll need a work permit first.",
        "You may qualify for Express Entry.",
        "Consult a lawyer for immigration matters.",
    ):
        assert _policy_ok_tiered_matches(bad, inp, view) is False, bad


def test_policy_rejects_operational_leakage_terms():
    """Operational-leakage parity, plus the new tiered move's
    arbiter_action token (resolved_to_tiered_matches)."""
    inp = _input(tier_evidence=_evidence(strong=[_strong()]))
    view = build_sanitized_responder_view_for_tiered_matches(inp.tier_evidence)
    for bad in (
        "The arbiter_action was resolved.",
        "I overrode the planner's choice.",
        "The planner said to ask first.",
        "The arbiter decided this.",
        "Falling back to fallback_to_legacy.",
        "Dispatched resolved_to_tiered_matches.",
    ):
        assert _policy_ok_tiered_matches(bad, inp, view) is False, bad


# =========================================================================
# Step-3 review High/Medium corrections — multi-record, exactly-once,
# case-insensitive token checks.
# =========================================================================
def _second_strong() -> StrongMatch:
    """A second Strong record with a different job_id, title, employer
    and URL but the SAME competitive_match strength claim. Used to
    test that the validator demands per-record evidence (not just
    one mention of the shared phrase) and that strength-phrase
    counts cover the number of records using each phrase."""
    return StrongMatch(
        job_id="s2",
        title="Bookkeeping Assistant",
        employer="North Star",
        location="Sault Ste. Marie, ON",
        noc_code="13102",
        url=_validated("https://example.com/strong2"),
        job_facts=_facts(),
        skill_alignment=(_alignment(),),
        non_blocking_gaps=(),
        credential_warning_text=None,
        strength_claim_text="competitive_match",
    )


def test_policy_accepts_single_canonical_phrase_with_natural_transition():
    """CP2 step 6.3: two records sharing the same strength_claim_text
    token may surface the canonical phrase ONCE and open the second
    record with a natural transition. The original ≥N-times rule is
    relaxed to ≥1 per distinct token used so back-to-back records do
    not have to read robotically. Per-record grounding (title,
    employer, URL, alignment) is still strict."""
    inp = _input(
        tier_evidence=_evidence(strong=[_strong(), _second_strong()]),
    )
    view = build_sanitized_responder_view_for_tiered_matches(inp.tier_evidence)
    natural_reply = (
        "Here's what's on the board.\n\n"
        "**Strong match — apply today**\n\n"
        "You'd be competitive for this one. Diamond J is hiring an "
        "Accounts Payable Clerk. Your QuickBooks aligns with their "
        "QuickBooks requirement. https://example.com/strong\n\n"
        "Here's another solid option for you: North Star is hiring "
        "a Bookkeeping Assistant. Your QuickBooks aligns with their "
        "QuickBooks requirement. https://example.com/strong2\n\n"
        "Want me to dig into any of these further?"
    )
    assert _policy_ok_tiered_matches(natural_reply, inp, view) is True


def test_policy_accepts_reply_describing_both_strong_jobs():
    """Two Strong records each described with title, employer, URL,
    the strength phrase appearing twice, AND a skill_alignment
    mention (final-correction content check)."""
    inp = _input(
        tier_evidence=_evidence(strong=[_strong(), _second_strong()]),
    )
    view = build_sanitized_responder_view_for_tiered_matches(inp.tier_evidence)
    full_reply = (
        "Here's what's on the board.\n\n"
        "**Strong match — apply today**\n\n"
        "You'd be competitive for this one. Diamond J is hiring an "
        "Accounts Payable Clerk. Your QuickBooks aligns with their "
        "QuickBooks requirement. https://example.com/strong\n\n"
        "You'd be competitive for this one. North Star is hiring a "
        "Bookkeeping Assistant. Your QuickBooks aligns with their "
        "QuickBooks requirement. https://example.com/strong2\n\n"
        "Want me to dig into any of these further?"
    )
    assert _policy_ok_tiered_matches(full_reply, inp, view) is True


def test_policy_accepts_stretch_reply_with_full_training_details():
    inp = _input(tier_evidence=_evidence(stretch=[_stretch()]))
    view = build_sanitized_responder_view_for_tiered_matches(inp.tier_evidence)
    full_reply = (
        "Here's what's on the board.\n\n"
        "**Stretch — reachable with prep**\n\n"
        "This one is close — there's a specific gap to close first. "
        "Algoma Office is hiring a Junior Accountant. Your QuickBooks "
        "aligns with their QuickBooks requirement. "
        "The gap is account reconciliation. Sault College offers "
        "Bookkeeping (online, 6 weeks): https://example.com/train "
        "https://example.com/stretch\n\n"
        "Would the prep be doable, or should we look at other options?"
    )
    assert _policy_ok_tiered_matches(full_reply, inp, view) is True


def test_policy_accepts_stretch_reply_with_no_verified_sentence():
    """When training_options has no actionable entry, the no-verified-
    training sentence satisfies the per-gap content check."""
    stretch_no_training = StretchMatch(
        job_id="w2", title="Junior Accountant", employer="Algoma Office",
        location="Sault Ste. Marie, ON", noc_code="11100",
        url=_validated("https://example.com/stretch"),
        job_facts=_facts(),
        skill_alignment=(_alignment(),),
        prioritized_gaps=(
            PrioritizedGap(
                job_requirement="account reconciliation",
                category="required", priority=1, blocker=False,
                training_options=(),
            ),
        ),
        credential_warning_text=None,
        strength_claim_text="close_with_named_gap",
    )
    inp = _input(tier_evidence=_evidence(stretch=[stretch_no_training]))
    view = build_sanitized_responder_view_for_tiered_matches(inp.tier_evidence)
    reply_with_no_verified = (
        "Here's what's on the board.\n\n"
        "**Stretch — reachable with prep**\n\n"
        "This one is close — there's a specific gap to close first. "
        "Algoma Office is hiring a Junior Accountant. Your QuickBooks "
        "aligns with their QuickBooks requirement. "
        "The gap is account reconciliation. "
        "I don't have a verified training option for that gap yet. "
        "https://example.com/stretch\n\n"
        "Would the prep be doable, or should we look at other options?"
    )
    assert _policy_ok_tiered_matches(reply_with_no_verified, inp, view) is True


def test_policy_accepts_sideways_reply_with_full_content():
    inp = _input(tier_evidence=_evidence(adjacent=[_adjacent()]))
    view = build_sanitized_responder_view_for_tiered_matches(inp.tier_evidence)
    full_reply = (
        "Here's what's on the board.\n\n"
        "**Sideways move — same skills, different angle**\n\n"
        "Your skill set carries over here. North Star has a "
        "Payroll Administrator posting. Your QuickBooks carries over "
        "to QuickBooks. The gap is ROE filing. https://example.com/adj\n\n"
        "Which sideways option would you like to explore first?"
    )
    assert _policy_ok_tiered_matches(full_reply, inp, view) is True


def test_policy_accepts_reply_with_credential_warning_text():
    sm_with_warning = StrongMatch(
        job_id="s1", title="Accounts Payable Clerk", employer="Diamond J",
        location="Sault Ste. Marie, ON", noc_code="13102",
        url=_validated("https://example.com/strong"),
        job_facts=_facts(),
        skill_alignment=(_alignment(),),
        non_blocking_gaps=(),
        credential_warning_text=(
            "This occupation may require Canadian/Ontario licensing."
        ),
        strength_claim_text="competitive_match",
    )
    inp = _input(tier_evidence=_evidence(strong=[sm_with_warning]))
    view = build_sanitized_responder_view_for_tiered_matches(inp.tier_evidence)
    reply_with_warning = (
        "Here's what's on the board.\n\n"
        "**Strong match — apply today**\n\n"
        "You'd be competitive for this one. Diamond J is hiring an "
        "Accounts Payable Clerk. Your QuickBooks aligns with their "
        "QuickBooks requirement. "
        "This occupation may require Canadian/Ontario licensing. "
        "https://example.com/strong\n\n"
        "Want me to dig into any of these further?"
    )
    assert _policy_ok_tiered_matches(reply_with_warning, inp, view) is True


# =========================================================================
# Step-3 review (paragraph-scoping + actionable-training tightening)
# =========================================================================
def test_policy_accepts_each_record_with_its_own_paragraph_alignment():
    """Both Strong records have QuickBooks alignment AND each
    paragraph mentions it — both records get coached."""
    inp = _input(
        tier_evidence=_evidence(strong=[_strong(), _second_strong()]),
    )
    view = build_sanitized_responder_view_for_tiered_matches(inp.tier_evidence)
    well_scoped_reply = (
        "Here's what's on the board.\n\n"
        "**Strong match — apply today**\n\n"
        "You'd be competitive for this one. Diamond J is hiring an "
        "Accounts Payable Clerk. Your QuickBooks aligns with their "
        "QuickBooks requirement. https://example.com/strong\n\n"
        "You'd be competitive for this one. North Star is hiring a "
        "Bookkeeping Assistant. Your QuickBooks aligns with their "
        "QuickBooks requirement. https://example.com/strong2\n\n"
        "Want me to dig into any of these further?"
    )
    assert _policy_ok_tiered_matches(well_scoped_reply, inp, view) is True


def test_policy_accepts_same_title_employer_distinct_url_paragraphs():
    """Same shape but each paragraph carries its own URL — both
    records have unambiguous ownership of their paragraphs."""
    s1 = StrongMatch(
        job_id="s1", title="Accounts Payable Clerk", employer="Diamond J",
        location="Sault Ste. Marie, ON", noc_code="13102",
        url=_validated("https://example.com/strong"),
        job_facts=_facts(),
        skill_alignment=(_alignment(),),
        non_blocking_gaps=(),
        credential_warning_text=None,
        strength_claim_text="competitive_match",
    )
    s2 = StrongMatch(
        job_id="s2", title="Accounts Payable Clerk", employer="Diamond J",
        location="Sault Ste. Marie, ON", noc_code="13102",
        url=_validated("https://example.com/strong2"),
        job_facts=_facts(),
        skill_alignment=(_alignment(),),
        non_blocking_gaps=(),
        credential_warning_text=None,
        strength_claim_text="competitive_match",
    )
    inp = _input(tier_evidence=_evidence(strong=[s1, s2]))
    view = build_sanitized_responder_view_for_tiered_matches(inp.tier_evidence)
    well_scoped_reply = (
        "Here's what's on the board.\n\n"
        "**Strong match — apply today**\n\n"
        "You'd be competitive for this one. Diamond J is hiring an "
        "Accounts Payable Clerk. Your QuickBooks aligns with their "
        "QuickBooks requirement. https://example.com/strong\n\n"
        "You'd be competitive for this one. Diamond J is hiring an "
        "Accounts Payable Clerk. Your QuickBooks aligns with their "
        "QuickBooks requirement. https://example.com/strong2\n\n"
        "Want me to dig into any of these further?"
    )
    assert _policy_ok_tiered_matches(well_scoped_reply, inp, view) is True


def test_policy_rejects_internal_tokens_case_insensitively():
    """Token rejection must be case-insensitive: COMPETITIVE_MATCH,
    SKILL_ALIGNMENT, WHY_ADJACENT must all reject."""
    inp = _input(tier_evidence=_evidence(strong=[_strong()]))
    view = build_sanitized_responder_view_for_tiered_matches(inp.tier_evidence)
    for leaked in (
        "COMPETITIVE_MATCH", "SKILL_ALIGNMENT", "WHY_ADJACENT",
        "Strength_Claim_Text", "Same_NOC_Minor_Group",
        "PRIORITIZED_GAPS", "Is_Normalized_Equal",
    ):
        reply = (
            "Here's what's on the board.\n\n"
            "**Strong match — apply today**\n\n"
            "You'd be competitive for this one. Diamond J is hiring an "
            f"Accounts Payable Clerk. The {leaked} matched. "
            "https://example.com/strong\n\n"
            "Want me to dig into any of these further?"
        )
        assert _policy_ok_tiered_matches(reply, inp, view) is False, leaked


# =========================================================================
# CP2 step 6.2 — Option B refined: first prioritized gap + every blocker
# =========================================================================
def _stretch_with_gaps(*gaps: PrioritizedGap) -> StretchMatch:
    """Build a Stretch record with a custom prioritized_gaps tuple.
    Shares the rest of the _stretch() fixture so the only varying
    axis between cases is the gap shape."""
    return StretchMatch(
        job_id="w-gap", title="Junior Accountant", employer="Algoma Office",
        location="Sault Ste. Marie, ON", noc_code="11100",
        url=_validated("https://example.com/stretch"),
        job_facts=_facts(),
        skill_alignment=(_alignment(),),
        prioritized_gaps=tuple(gaps),
        credential_warning_text=None,
        strength_claim_text="close_with_named_gap",
    )


def _stretch_reply(body_fragment: str) -> str:
    """Wrap a Stretch paragraph body in the locked envelope (heading +
    strength phrase + alignment + URL + closing) so the policy gate
    only judges the gap-content section."""
    return (
        "Here's what's on the board.\n\n"
        "**Stretch — reachable with prep**\n\n"
        "This one is close — there's a specific gap to close first. "
        "Algoma Office is hiring a Junior Accountant. Your QuickBooks "
        "aligns with their QuickBooks requirement. "
        f"{body_fragment} "
        "https://example.com/stretch\n\n"
        "Would the prep be doable, or should we look at other options?"
    )


def test_policy_accepts_stretch_reply_mentioning_only_first_gap_no_blocker():
    """Option B refined: a Stretch record with three non-blocker gaps
    is accepted when the LLM surfaces only the first gap. Lower-
    priority non-blocker gaps may be skipped."""
    record = _stretch_with_gaps(
        PrioritizedGap(job_requirement="account reconciliation",
                       category="required", priority=1, blocker=False,
                       training_options=()),
        PrioritizedGap(job_requirement="bank reconciliation",
                       category="required", priority=2, blocker=False,
                       training_options=()),
        PrioritizedGap(job_requirement="financial reporting",
                       category="required", priority=3, blocker=False,
                       training_options=()),
    )
    inp = _input(tier_evidence=_evidence(stretch=[record]))
    view = build_sanitized_responder_view_for_tiered_matches(inp.tier_evidence)
    reply = _stretch_reply(
        "The gap is account reconciliation. "
        "I don't have a verified training option for that gap yet."
    )
    assert _policy_ok_tiered_matches(reply, inp, view) is True


def test_policy_accepts_stretch_reply_surfacing_first_plus_blockers():
    """When the LLM surfaces the first gap plus every blocker, the
    reply passes even if mid-priority non-blocker gaps are omitted."""
    record = _stretch_with_gaps(
        PrioritizedGap(job_requirement="account reconciliation",
                       category="required", priority=1, blocker=False,
                       training_options=()),
        PrioritizedGap(job_requirement="bank reconciliation",
                       category="required", priority=2, blocker=False,
                       training_options=()),
        PrioritizedGap(job_requirement="forklift certification",
                       category="required", priority=3, blocker=True,
                       training_options=()),
    )
    inp = _input(tier_evidence=_evidence(stretch=[record]))
    view = build_sanitized_responder_view_for_tiered_matches(inp.tier_evidence)
    reply = _stretch_reply(
        "The gaps here are account reconciliation and forklift "
        "certification. I don't have verified training options "
        "for those gaps yet."
    )
    assert _policy_ok_tiered_matches(reply, inp, view) is True


def test_policy_training_validated_only_for_surfaced_gaps():
    """Training details are only enforced for the gaps the LLM is
    required to surface. A non-blocker non-first gap that has
    actionable training but is silently omitted is acceptable."""
    record = _stretch_with_gaps(
        PrioritizedGap(
            job_requirement="account reconciliation",
            category="required", priority=1, blocker=False,
            training_options=(),
        ),
        PrioritizedGap(
            job_requirement="bank reconciliation",
            category="required", priority=2, blocker=False,
            training_options=(
                TrainingOption(
                    provider="Sault College",
                    title="Bookkeeping Fundamentals",
                    url=_validated("https://example.com/bookkeeping"),
                    format="online",
                    duration_text="6 weeks",
                ),
            ),
        ),
    )
    inp = _input(tier_evidence=_evidence(stretch=[record]))
    view = build_sanitized_responder_view_for_tiered_matches(inp.tier_evidence)
    reply = _stretch_reply(
        "The gap is account reconciliation. "
        "I don't have a verified training option for that gap yet."
    )
    # The LLM did not surface gap 1 (bank reconciliation), so its
    # training is not required either. Acceptance pins the refined
    # rule: training validation tracks surfaced gaps, not all gaps.
    assert _policy_ok_tiered_matches(reply, inp, view) is True


def test_policy_accepts_stretch_only_reply_ending_with_all_tier_closing():
    """The LLM naturally picks `"Which of these would you like to look
    at first?"` even for Stretch-only surfaces. Closing choice carries
    no grounding or safety signal, so the validator no longer pins
    closing-by-tier-presence."""
    inp = _input(tier_evidence=_evidence(stretch=[_stretch()]))
    view = build_sanitized_responder_view_for_tiered_matches(inp.tier_evidence)
    reply = (
        "Here's what's on the board.\n\n"
        "**Stretch — reachable with prep**\n\n"
        "This one is close — there's a specific gap to close first. "
        "Algoma Office is hiring a Junior Accountant. Your QuickBooks "
        "aligns with their QuickBooks requirement. "
        "The gap is account reconciliation. Sault College offers "
        "Bookkeeping (online, 6 weeks): https://example.com/train "
        "https://example.com/stretch\n\n"
        "Which of these would you like to look at first?"
    )
    assert _policy_ok_tiered_matches(reply, inp, view) is True


def test_policy_rejects_reply_without_any_authorized_closing():
    """Some closing from the closed set must end the reply — an
    invented closing is rejected."""
    inp = _input(tier_evidence=_evidence(stretch=[_stretch()]))
    view = build_sanitized_responder_view_for_tiered_matches(inp.tier_evidence)
    reply = (
        "Here's what's on the board.\n\n"
        "**Stretch — reachable with prep**\n\n"
        "This one is close — there's a specific gap to close first. "
        "Algoma Office is hiring a Junior Accountant. Your QuickBooks "
        "aligns with their QuickBooks requirement. "
        "The gap is account reconciliation. Sault College offers "
        "Bookkeeping (online, 6 weeks): https://example.com/train "
        "https://example.com/stretch\n\n"
        "Let me know what you think."
    )
    assert _policy_ok_tiered_matches(reply, inp, view) is False


