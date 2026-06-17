"""AR-9.feat.coach-tiers CP1 step 9 — view boundary for the
present_tiered_matches move.

Pins:
  - Validated → SanitizedURL is performed at the view boundary
    (no raw Validated, no raw str URL leaks into the view);
  - PromptStrongMatch / PromptStretchMatch / PromptAdjacentJob
    expose only the fields the coach prompt and deterministic
    fallback need (no raw MatchResult, no raw job dict);
  - prompt_urls is the union of every tier-record job URL and
    every training-option URL across all three tiers;
  - fallback_urls covers exactly what the deterministic fallback
    can render (= prompt_urls in step 9);
  - tier order and exclusivity are pass-through from TieredEvidence
    (no re-ordering, no re-filtering at the view);
  - move gating: tier slots are non-empty ONLY when the
    `build_sanitized_responder_view_for_tiered_matches` builder ran;
  - non-tier slots are empty on the tiered view (the view is
    move-gated; the new builder populates tier slots and leaves
    everything else empty).
"""
from __future__ import annotations

from datetime import date

import pytest

from skillbridge.chat.tiered_evidence import (
    AdjacentJob,
    JobFacts,
    NonBlockingGap,
    PrioritizedGap,
    StretchMatch,
    StrongMatch,
    TieredEvidence,
    TrainingOption,
    TransferablePair,
)
from skillbridge.chat.url_policy import Validated, validate
from skillbridge.chat.url_views import (
    PromptAdjacentJob,
    PromptJobFacts,
    PromptNonBlockingGap,
    PromptPrioritizedGap,
    PromptStretchMatch,
    PromptStrongMatch,
    PromptTrainingOption,
    PromptTransferablePair,
    SanitizedResponderView,
    SanitizedURL,
    build_sanitized_responder_view_for_tiered_matches,
)
from skillbridge.match.alignment import SkillAlignment

pytestmark = pytest.mark.nodb


# =========================================================================
# Fixture builders
# =========================================================================
def _validated(raw: str) -> Validated:
    """Run a URL through the structural validator; assume it passes."""
    result = validate(raw)
    assert isinstance(result, Validated), f"fixture URL must validate: {raw}"
    return result


def _alignment(user_skill="QuickBooks", job_requirement="QuickBooks") -> SkillAlignment:
    return SkillAlignment(
        user_skill=user_skill,
        job_requirement=job_requirement,
        stage="exact",
        source="required",
        is_normalized_equal=(user_skill == job_requirement),
    )


def _job_facts(employment_type="full-time") -> JobFacts:
    return JobFacts(
        posted_date=date(2026, 6, 1),
        posted_days_ago=13,
        location="Sault Ste. Marie, ON",
        employment_type=employment_type,
        salary_text="$22-24/hr",
    )


def _strong_match(*, job_id="s1", url="https://example.com/strong/1") -> StrongMatch:
    return StrongMatch(
        job_id=job_id,
        title="Accounts Payable Clerk",
        employer="Diamond J",
        location="Sault Ste. Marie, ON",
        noc_code="13102",
        url=_validated(url),
        job_facts=_job_facts(),
        skill_alignment=(_alignment(),),
        non_blocking_gaps=(),
        credential_warning_text=None,
        strength_claim_text="competitive_match",
    )


def _stretch_match(
    *,
    job_id="w1",
    url="https://example.com/stretch/1",
    training_url="https://example.com/train/1",
) -> StretchMatch:
    return StretchMatch(
        job_id=job_id,
        title="Junior Accountant",
        employer="Algoma Office",
        location="Sault Ste. Marie, ON",
        noc_code="11100",
        url=_validated(url),
        job_facts=_job_facts(),
        skill_alignment=(_alignment(),),
        prioritized_gaps=(
            PrioritizedGap(
                job_requirement="account reconciliation",
                category="required",
                priority=1,
                blocker=False,
                training_options=(
                    TrainingOption(
                        provider="Sault College",
                        title="Bookkeeping Fundamentals",
                        url=_validated(training_url),
                        format="online",
                        duration_text="6 weeks",
                    ),
                ),
            ),
        ),
        credential_warning_text=None,
        strength_claim_text="close_with_named_gap",
    )


def _adjacent_job(*, job_id="adj-1", url="https://example.com/adj/1") -> AdjacentJob:
    return AdjacentJob(
        job_id=job_id,
        title="Payroll Administrator",
        employer="North Star",
        location="Sault Ste. Marie, ON",
        noc_code="12102",
        url=_validated(url),
        job_facts=_job_facts(),
        skill_alignment=(_alignment(),),
        transferable_pairs=(
            TransferablePair(
                user_skill="QuickBooks",
                applies_to="QuickBooks",
                stage="exact",
            ),
        ),
        important_gaps=("ROE filing",),
        credential_warning_text=None,
        why_adjacent="same_noc_minor_group",
        strength_claim_text="transferable_lane",
    )


def _evidence(*, strong=(), stretch=(), adjacent=()) -> TieredEvidence:
    return TieredEvidence(
        apply_today=tuple(strong),
        worth_a_try=tuple(stretch),
        sideways_move=tuple(adjacent),
    )


# =========================================================================
# Validated → SanitizedURL projection
# =========================================================================
def test_strong_url_promoted_to_sanitized_url():
    out = build_sanitized_responder_view_for_tiered_matches(
        _evidence(strong=[_strong_match(url="https://example.com/a")]),
    )
    item = out.prompt_tiered_apply_today[0]
    assert isinstance(item.url, SanitizedURL)
    assert item.url.canonical == "https://example.com/a"


def test_strong_url_none_when_evidence_validated_is_none():
    strong = StrongMatch(
        job_id="s", title="t", employer=None, location=None, noc_code=None,
        url=None,
        job_facts=_job_facts(), skill_alignment=(), non_blocking_gaps=(),
        credential_warning_text=None, strength_claim_text="competitive_match",
    )
    out = build_sanitized_responder_view_for_tiered_matches(
        _evidence(strong=[strong]),
    )
    assert out.prompt_tiered_apply_today[0].url is None


def test_stretch_url_and_training_url_both_promoted():
    out = build_sanitized_responder_view_for_tiered_matches(
        _evidence(stretch=[_stretch_match(
            url="https://example.com/j",
            training_url="https://example.com/t",
        )]),
    )
    item = out.prompt_tiered_worth_a_try[0]
    assert isinstance(item.url, SanitizedURL)
    assert item.url.canonical == "https://example.com/j"
    opt = item.prioritized_gaps[0].training_options[0]
    assert isinstance(opt.url, SanitizedURL)
    assert opt.url.canonical == "https://example.com/t"


def test_adjacent_url_promoted_to_sanitized_url():
    out = build_sanitized_responder_view_for_tiered_matches(
        _evidence(adjacent=[_adjacent_job(url="https://example.com/x")]),
    )
    item = out.prompt_tiered_sideways_move[0]
    assert isinstance(item.url, SanitizedURL)
    assert item.url.canonical == "https://example.com/x"


# =========================================================================
# prompt_urls — every tier URL covered
# =========================================================================
def test_prompt_urls_includes_every_tier_job_url():
    out = build_sanitized_responder_view_for_tiered_matches(
        _evidence(
            strong=[_strong_match(url="https://example.com/s")],
            stretch=[_stretch_match(
                url="https://example.com/w",
                training_url="https://example.com/wt",
            )],
            adjacent=[_adjacent_job(url="https://example.com/a")],
        ),
    )
    canonicals = out.prompt_urls
    assert "https://example.com/s" in canonicals
    assert "https://example.com/w" in canonicals
    assert "https://example.com/a" in canonicals


def test_prompt_urls_includes_all_training_urls_across_stretch():
    out = build_sanitized_responder_view_for_tiered_matches(
        _evidence(
            stretch=[_stretch_match(
                url="https://example.com/j1",
                training_url="https://example.com/t1",
            )],
        ),
    )
    assert "https://example.com/t1" in out.prompt_urls


def test_prompt_urls_excludes_none_urls_silently():
    """An evidence item whose Validated was None contributes nothing
    to the allowlist."""
    strong = StrongMatch(
        job_id="s", title="t", employer=None, location=None, noc_code=None,
        url=None,
        job_facts=_job_facts(), skill_alignment=(), non_blocking_gaps=(),
        credential_warning_text=None, strength_claim_text="competitive_match",
    )
    out = build_sanitized_responder_view_for_tiered_matches(
        _evidence(strong=[strong]),
    )
    assert out.prompt_urls == frozenset()


def test_fallback_urls_equals_prompt_urls_at_step_9():
    """v5 / S1: fallback_urls is what the deterministic fallback may
    render. In step 9 (before the fallback is built in step 10) the
    set is identical to prompt_urls; step 10 may tighten it."""
    out = build_sanitized_responder_view_for_tiered_matches(
        _evidence(
            strong=[_strong_match(url="https://example.com/s")],
            stretch=[_stretch_match(
                url="https://example.com/w",
                training_url="https://example.com/wt",
            )],
        ),
    )
    assert out.fallback_urls == out.prompt_urls


# =========================================================================
# Field-exposure boundary
# =========================================================================
def test_view_carries_no_validated_objects():
    """The view's tier records must not surface any raw Validated.
    All URL slots are either SanitizedURL or None."""
    out = build_sanitized_responder_view_for_tiered_matches(
        _evidence(
            strong=[_strong_match()],
            stretch=[_stretch_match()],
            adjacent=[_adjacent_job()],
        ),
    )
    for item in out.prompt_tiered_apply_today:
        assert item.url is None or isinstance(item.url, SanitizedURL)
    for item in out.prompt_tiered_worth_a_try:
        assert item.url is None or isinstance(item.url, SanitizedURL)
        for g in item.prioritized_gaps:
            for o in g.training_options:
                assert o.url is None or isinstance(o.url, SanitizedURL)
    for item in out.prompt_tiered_sideways_move:
        assert item.url is None or isinstance(item.url, SanitizedURL)


def test_view_carries_no_raw_match_result_or_job_dict():
    """Spot-check exposure: the view's tier items must be the
    Prompt* dataclasses, NOT MatchResult / dict / TieredEvidence."""
    out = build_sanitized_responder_view_for_tiered_matches(
        _evidence(
            strong=[_strong_match()],
            stretch=[_stretch_match()],
            adjacent=[_adjacent_job()],
        ),
    )
    assert all(isinstance(i, PromptStrongMatch)
               for i in out.prompt_tiered_apply_today)
    assert all(isinstance(i, PromptStretchMatch)
               for i in out.prompt_tiered_worth_a_try)
    assert all(isinstance(i, PromptAdjacentJob)
               for i in out.prompt_tiered_sideways_move)


def test_view_tier_records_have_only_known_field_names():
    """The view-side records should expose ONLY the fields the prompt
    and fallback need — not any extra leaks from the evidence layer."""
    expected_strong = {
        "job_id", "title", "employer", "location", "noc_code",
        "url", "job_facts", "skill_alignment", "non_blocking_gaps",
        "credential_warning_text", "strength_claim_text",
    }
    expected_stretch = {
        "job_id", "title", "employer", "location", "noc_code",
        "url", "job_facts", "skill_alignment", "prioritized_gaps",
        "credential_warning_text", "strength_claim_text",
    }
    expected_adjacent = {
        "job_id", "title", "employer", "location", "noc_code",
        "url", "job_facts", "skill_alignment", "transferable_pairs",
        "important_gaps", "credential_warning_text",
        "why_adjacent", "strength_claim_text",
    }
    assert set(PromptStrongMatch.__annotations__.keys()) == expected_strong
    assert set(PromptStretchMatch.__annotations__.keys()) == expected_stretch
    assert set(PromptAdjacentJob.__annotations__.keys()) == expected_adjacent


def test_job_facts_view_carries_only_sourceable_fields():
    expected = {
        "posted_date", "posted_days_ago", "location",
        "employment_type", "salary_text",
    }
    assert set(PromptJobFacts.__annotations__.keys()) == expected


# =========================================================================
# Tier order + exclusivity pass-through
# =========================================================================
def test_view_preserves_apply_today_order_from_evidence():
    out = build_sanitized_responder_view_for_tiered_matches(
        _evidence(strong=[
            _strong_match(job_id="z", url="https://example.com/1"),
            _strong_match(job_id="a", url="https://example.com/2"),
            _strong_match(job_id="m", url="https://example.com/3"),
        ]),
    )
    assert [i.job_id for i in out.prompt_tiered_apply_today] == ["z", "a", "m"]


def test_view_preserves_worth_a_try_order_from_evidence():
    out = build_sanitized_responder_view_for_tiered_matches(
        _evidence(stretch=[
            _stretch_match(job_id="z", url="https://example.com/1",
                           training_url="https://example.com/t1"),
            _stretch_match(job_id="a", url="https://example.com/2",
                           training_url="https://example.com/t2"),
        ]),
    )
    assert [i.job_id for i in out.prompt_tiered_worth_a_try] == ["z", "a"]


def test_view_preserves_sideways_order_from_evidence():
    out = build_sanitized_responder_view_for_tiered_matches(
        _evidence(adjacent=[
            _adjacent_job(job_id="z", url="https://example.com/1"),
            _adjacent_job(job_id="a", url="https://example.com/2"),
        ]),
    )
    assert [i.job_id for i in out.prompt_tiered_sideways_move] == ["z", "a"]


def test_view_is_pass_through_no_reordering_or_filtering():
    """The view does NOT re-filter or re-sort. Evidence-level
    exclusivity is the contract; the view trusts it."""
    out = build_sanitized_responder_view_for_tiered_matches(
        _evidence(
            strong=[_strong_match(job_id="x")],
            stretch=[_stretch_match(job_id="x")],   # would be a tier-exclusivity violation
        ),
    )
    # The view emits both — exclusivity is the evidence builder's
    # responsibility. Step 8 already pins it; the view trusts it.
    assert len(out.prompt_tiered_apply_today) == 1
    assert len(out.prompt_tiered_worth_a_try) == 1


# =========================================================================
# Move gating — tier data appears only via the tiered builder
# =========================================================================
def test_other_builders_leave_tier_slots_empty():
    """Every existing builder constructs a SanitizedResponderView
    WITHOUT touching the new tier slots. Default-empty fields enforce
    move gating."""
    from skillbridge.chat.url_views import _empty_view
    view = _empty_view()
    assert view.prompt_tiered_apply_today == ()
    assert view.prompt_tiered_worth_a_try == ()
    assert view.prompt_tiered_sideways_move == ()


def test_tiered_builder_leaves_non_tier_slots_empty():
    """Symmetric move gating: the tiered builder populates ONLY tier
    slots; all other per-move slots stay empty."""
    out = build_sanitized_responder_view_for_tiered_matches(
        _evidence(strong=[_strong_match()]),
    )
    assert out.prompt_results == ()
    assert out.fallback_results == ()
    assert out.prompt_present_matches_training_flat == ()
    assert out.prompt_present_matches_training_groups == ()
    assert out.prompt_explain_gap_training_flat == ()
    assert out.prompt_present_near_miss_training_flat == ()
    assert out.prompt_explain_remaining_gaps_training_flat == ()
    assert out.prompt_adjacent_recommendations is None
    assert out.fallback_adjacent_recommendations == ()
    assert out.prompt_adjacent_role is None
    assert out.fallback_adjacent_role is None


def test_empty_tiers_yield_empty_view_slots_and_empty_allowlists():
    out = build_sanitized_responder_view_for_tiered_matches(_evidence())
    assert out.prompt_tiered_apply_today == ()
    assert out.prompt_tiered_worth_a_try == ()
    assert out.prompt_tiered_sideways_move == ()
    assert out.prompt_urls == frozenset()
    assert out.fallback_urls == frozenset()


# =========================================================================
# Pass-through field integrity
# =========================================================================
def test_strong_match_pass_through_scalars():
    out = build_sanitized_responder_view_for_tiered_matches(
        _evidence(strong=[_strong_match()]),
    )
    item = out.prompt_tiered_apply_today[0]
    assert item.job_id == "s1"
    assert item.title == "Accounts Payable Clerk"
    assert item.employer == "Diamond J"
    assert item.strength_claim_text == "competitive_match"


def test_stretch_match_prioritized_gap_pass_through():
    out = build_sanitized_responder_view_for_tiered_matches(
        _evidence(stretch=[_stretch_match()]),
    )
    gap = out.prompt_tiered_worth_a_try[0].prioritized_gaps[0]
    assert isinstance(gap, PromptPrioritizedGap)
    assert gap.job_requirement == "account reconciliation"
    assert gap.priority == 1
    assert gap.training_options[0].provider == "Sault College"
    assert gap.training_options[0].format == "online"
    assert gap.training_options[0].duration_text == "6 weeks"


def test_adjacent_job_why_adjacent_pass_through():
    out = build_sanitized_responder_view_for_tiered_matches(
        _evidence(adjacent=[_adjacent_job()]),
    )
    assert out.prompt_tiered_sideways_move[0].why_adjacent == "same_noc_minor_group"


def test_strong_non_blocking_gap_pass_through():
    sm = StrongMatch(
        job_id="s", title="t", employer=None, location=None, noc_code=None,
        url=None,
        job_facts=_job_facts(), skill_alignment=(),
        non_blocking_gaps=(
            NonBlockingGap(job_requirement="Sage 50", material=True),
            NonBlockingGap(job_requirement="French", material=False),
        ),
        credential_warning_text=None, strength_claim_text="strongest_current",
    )
    out = build_sanitized_responder_view_for_tiered_matches(
        _evidence(strong=[sm]),
    )
    gaps = out.prompt_tiered_apply_today[0].non_blocking_gaps
    assert [g.job_requirement for g in gaps] == ["Sage 50", "French"]
    assert [g.material for g in gaps] == [True, False]
    assert all(isinstance(g, PromptNonBlockingGap) for g in gaps)


def test_transferable_pair_pass_through():
    out = build_sanitized_responder_view_for_tiered_matches(
        _evidence(adjacent=[_adjacent_job()]),
    )
    pair = out.prompt_tiered_sideways_move[0].transferable_pairs[0]
    assert isinstance(pair, PromptTransferablePair)
    assert pair.user_skill == "QuickBooks"
    assert pair.applies_to == "QuickBooks"
    assert pair.stage == "exact"


# =========================================================================
# Frozen smoke
# =========================================================================
def test_view_tier_records_are_frozen():
    out = build_sanitized_responder_view_for_tiered_matches(
        _evidence(
            strong=[_strong_match()],
            stretch=[_stretch_match()],
            adjacent=[_adjacent_job()],
        ),
    )
    with pytest.raises((AttributeError, Exception)):
        out.prompt_tiered_apply_today[0].title = "x"  # type: ignore
    with pytest.raises((AttributeError, Exception)):
        out.prompt_tiered_worth_a_try[0].title = "x"  # type: ignore
    with pytest.raises((AttributeError, Exception)):
        out.prompt_tiered_sideways_move[0].title = "x"  # type: ignore
