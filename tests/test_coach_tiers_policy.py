"""AR-9.feat.coach-tiers CP1 step 11 — tiered-matches policy gate.

Pins:
  - GROUNDED_TERMS includes user skills, matched/missing job skills,
    employers, training providers from the tier view's records;
  - Exemption is exact normalized-string match (no fuzzy, no sentence
    classification);
  - Known providers absent from GROUNDED_TERMS are still rejected;
  - Abbreviation-aware grounding (SCCC ↔ "Sault Community Career
    Centre") matches the existing `responder._check_ungrounded_provider`
    behavior;
  - Salary text is OMITTED from the deterministic fallback's facts
    clause (locked decision: option B).
"""
from __future__ import annotations

from datetime import date

import pytest

from skillbridge.chat.coach_tiers_fallback import render_coach_tiers_fallback
from skillbridge.chat.coach_tiers_policy import (
    build_grounded_terms,
    check_ungrounded_provider_for_tiered_matches,
)
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
    build_sanitized_responder_view_for_tiered_matches,
)
from skillbridge.match.alignment import SkillAlignment

pytestmark = pytest.mark.nodb


# =========================================================================
# Fixture builders (light reuse of test_tiered_view shapes)
# =========================================================================
def _validated(raw: str) -> Validated:
    result = validate(raw)
    assert isinstance(result, Validated), raw
    return result


def _alignment(user_skill="QuickBooks", job_requirement="QuickBooks") -> SkillAlignment:
    return SkillAlignment(
        user_skill=user_skill, job_requirement=job_requirement,
        stage="exact", source="required",
        is_normalized_equal=(user_skill == job_requirement),
    )


def _facts(salary_text="$22-24/hr") -> JobFacts:
    return JobFacts(
        posted_date=date(2026, 6, 11),
        posted_days_ago=3,
        location="Sault Ste. Marie, ON",
        employment_type="full-time",
        salary_text=salary_text,
    )


def _strong(*, employer="Diamond J", alignments=(_alignment(),), gaps=()) -> StrongMatch:
    return StrongMatch(
        job_id="s1", title="Accounts Payable Clerk", employer=employer,
        location=None, noc_code=None,
        url=_validated("https://example.com/strong"),
        job_facts=_facts(),
        skill_alignment=alignments,
        non_blocking_gaps=gaps,
        credential_warning_text=None,
        strength_claim_text="competitive_match",
    )


def _stretch(
    *,
    employer="Algoma Office",
    alignments=(_alignment(),),
    training_provider="Sault College",
) -> StretchMatch:
    return StretchMatch(
        job_id="w1", title="Junior Accountant", employer=employer,
        location=None, noc_code=None,
        url=_validated("https://example.com/stretch"),
        job_facts=_facts(),
        skill_alignment=alignments,
        prioritized_gaps=(
            PrioritizedGap(
                job_requirement="account reconciliation",
                category="required", priority=1, blocker=False,
                training_options=(
                    TrainingOption(
                        provider=training_provider, title="Bookkeeping",
                        url=_validated("https://example.com/train"),
                        format="online", duration_text="6 weeks",
                    ),
                ),
            ),
        ),
        credential_warning_text=None,
        strength_claim_text="stretch_with_training_bridge",
    )


def _adjacent(*, employer="North Star") -> AdjacentJob:
    return AdjacentJob(
        job_id="adj-1", title="Payroll Administrator", employer=employer,
        location=None, noc_code=None,
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


def _view(*, strong=(), stretch=(), adjacent=()):
    return build_sanitized_responder_view_for_tiered_matches(
        TieredEvidence(
            apply_today=tuple(strong),
            worth_a_try=tuple(stretch),
            sideways_move=tuple(adjacent),
        ),
    )


# =========================================================================
# build_grounded_terms — coverage of every documented source
# =========================================================================
def test_grounded_terms_includes_user_skills_from_alignment():
    v = _view(strong=[_strong(
        alignments=(_alignment(user_skill="QuickBooks",
                                job_requirement="quickbooks"),),
    )])
    terms = build_grounded_terms(v)
    assert "quickbooks" in terms


def test_grounded_terms_includes_job_requirements_from_alignment():
    v = _view(strong=[_strong(
        alignments=(_alignment(user_skill="bookkeeping",
                                job_requirement="accounts payable"),),
    )])
    terms = build_grounded_terms(v)
    assert "accounts payable" in terms
    assert "bookkeeping" in terms


def test_grounded_terms_includes_employers_across_all_tiers():
    v = _view(
        strong=[_strong(employer="Diamond J")],
        stretch=[_stretch(employer="Algoma Office")],
        adjacent=[_adjacent(employer="North Star")],
    )
    terms = build_grounded_terms(v)
    assert "diamond j" in terms
    assert "algoma office" in terms
    assert "north star" in terms


def test_grounded_terms_includes_training_providers():
    v = _view(stretch=[_stretch(training_provider="Sault College")])
    terms = build_grounded_terms(v)
    assert "sault college" in terms


def test_grounded_terms_includes_non_blocking_gap_names():
    v = _view(strong=[_strong(gaps=(
        NonBlockingGap(job_requirement="Sage 50", material=True),
    ))])
    terms = build_grounded_terms(v)
    assert "sage 50" in terms


def test_grounded_terms_includes_prioritized_gap_names():
    v = _view(stretch=[_stretch()])
    terms = build_grounded_terms(v)
    assert "account reconciliation" in terms


def test_grounded_terms_includes_important_gaps_for_adjacent():
    v = _view(adjacent=[_adjacent()])
    terms = build_grounded_terms(v)
    assert "roe filing" in terms


def test_grounded_terms_includes_transferable_pair_skills_for_adjacent():
    v = _view(adjacent=[_adjacent()])
    terms = build_grounded_terms(v)
    assert "quickbooks" in terms


def test_grounded_terms_empty_for_empty_view():
    v = _view()
    assert build_grounded_terms(v) == frozenset()


def test_grounded_terms_excludes_empty_string():
    """Whitespace-only employer should not pollute the set."""
    v = _view(strong=[_strong(employer="")])
    terms = build_grounded_terms(v)
    assert "" not in terms


# =========================================================================
# check_ungrounded_provider — exact-match exemption
# =========================================================================
def test_quickbooks_mention_grounded_via_user_skill():
    """The locked QuickBooks case. QuickBooks is a known provider in
    `_KNOWN_TRAINING_PROVIDERS`. When the user has QuickBooks as a
    skill (in skill_alignment), an LLM reply mentioning QuickBooks
    must NOT be rejected as ungrounded."""
    v = _view(strong=[_strong(
        alignments=(_alignment(user_skill="QuickBooks",
                                job_requirement="quickbooks"),),
    )])
    reply = "Diamond J is hiring an Accounts Payable Clerk. You have QuickBooks already."
    assert check_ungrounded_provider_for_tiered_matches(reply, v) is None


def test_quickbooks_mention_rejected_when_not_in_user_skills():
    """The other direction: a reply mentioning QuickBooks must be
    rejected when QuickBooks is not grounded anywhere in the view."""
    v = _view(strong=[_strong(
        alignments=(_alignment(user_skill="Python",
                                job_requirement="python"),),
    )])
    reply = "I recommend QuickBooks Online to brush up."
    result = check_ungrounded_provider_for_tiered_matches(reply, v)
    assert result == "quickbooks"


def test_sault_college_mention_grounded_via_training_provider():
    v = _view(stretch=[_stretch(training_provider="Sault College")])
    reply = "Sault College has Bookkeeping training."
    assert check_ungrounded_provider_for_tiered_matches(reply, v) is None


def test_sault_college_mention_rejected_when_not_grounded():
    """Sault College in the deny-list, NOT in the view — must reject."""
    v = _view(strong=[_strong()])
    reply = "Try Sault College for bookkeeping training."
    assert check_ungrounded_provider_for_tiered_matches(reply, v) == "sault college"


def test_known_provider_mention_grounded_via_employer():
    """Algoma University is a known provider; if it's also the
    employer in a tier record, mention is grounded."""
    v = _view(strong=[_strong(employer="Algoma University")])
    reply = "Algoma University has an interesting role open."
    assert check_ungrounded_provider_for_tiered_matches(reply, v) is None


def test_no_provider_mentioned_returns_none():
    v = _view(strong=[_strong()])
    reply = "Worth checking out the listing in detail."
    assert check_ungrounded_provider_for_tiered_matches(reply, v) is None


def test_invented_provider_still_rejected():
    """A reply naming a provider on the deny-list that isn't in the
    view stays rejected — defense against LLM supplementing TRAINING
    with outside-knowledge organizations."""
    v = _view(strong=[_strong()])
    reply = "Transportation Association of Canada certifies this."
    assert check_ungrounded_provider_for_tiered_matches(reply, v) == "transportation association of canada"


def test_abbreviation_exempted_when_canonical_grounded():
    """If 'Sault Community Career Centre' is grounded via employer,
    its abbreviation 'SCCC' is treated as grounded too."""
    v = _view(strong=[_strong(employer="Sault Community Career Centre")])
    reply = "SCCC has a posting that might be worth a look."
    assert check_ungrounded_provider_for_tiered_matches(reply, v) is None


def test_abbreviation_rejected_when_canonical_not_grounded():
    v = _view(strong=[_strong()])
    reply = "SCCC has resources for that."
    assert check_ungrounded_provider_for_tiered_matches(reply, v) == "sccc"


def test_word_boundary_avoids_substring_false_positive():
    """The check uses a word-boundary regex (parity with the existing
    `_check_ungrounded_provider`). 'TAC' inside 'tactical' must NOT
    fire."""
    v = _view(strong=[_strong()])
    reply = "Your tactical experience translates well."
    assert check_ungrounded_provider_for_tiered_matches(reply, v) is None


def test_no_fuzzy_or_sentence_classification():
    """Pure set-membership exemption. A typo in the grounded term
    is NOT exempted (no fuzzy match)."""
    v = _view(strong=[_strong(
        alignments=(_alignment(user_skill="QuickBooks",
                                job_requirement="QuickBooks"),),
    )])
    # The LLM writes "quickbook" (typo) — but it's not on the deny-list,
    # so the check returns None (nothing to evaluate).
    reply = "You have quickbook experience."
    assert check_ungrounded_provider_for_tiered_matches(reply, v) is None

    # The LLM writes "QuickBooks" inside a forbidden context too —
    # since "quickbooks" IS in the grounded set, it remains exempt.
    reply = "QuickBooks is one of your strengths."
    assert check_ungrounded_provider_for_tiered_matches(reply, v) is None


# =========================================================================
# Salary omission in the fallback (option B)
# =========================================================================
def test_salary_text_omitted_from_strong_paragraph():
    """salary_text stays on the view's job_facts for future use, but
    the deterministic fallback does NOT render it. The `$` token must
    not appear in any rendered tier paragraph."""
    v = _view(strong=[_strong()])
    text, _ = render_coach_tiers_fallback(v)
    assert "$" not in text
    assert "22-24" not in text


def test_salary_text_omitted_from_stretch_paragraph():
    v = _view(stretch=[_stretch()])
    text, _ = render_coach_tiers_fallback(v)
    assert "$" not in text


def test_salary_text_omitted_from_sideways_paragraph():
    v = _view(adjacent=[_adjacent()])
    text, _ = render_coach_tiers_fallback(v)
    assert "$" not in text


def test_salary_text_preserved_on_view_for_future_use():
    """The omission is at the renderer; salary_text stays on the
    view's job_facts so future surfaces can opt-in if needed."""
    v = _view(strong=[_strong()])
    item = v.prompt_tiered_apply_today[0]
    assert item.job_facts.salary_text == "$22-24/hr"


def test_employment_type_and_posted_days_still_rendered():
    """The salary omission is narrow — other facts still surface."""
    v = _view(strong=[_strong()])
    text, _ = render_coach_tiers_fallback(v)
    assert "full-time" in text
    assert "posted 3 days ago" in text
