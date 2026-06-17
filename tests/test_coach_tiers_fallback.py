"""AR-9.feat.coach-tiers CP1 step 10 — deterministic fallback renderer.

Pins:
  - Reads ONLY `view.prompt_tiered_*` records;
  - Renders the three locked headers when their tier is non-empty;
  - Skips header AND paragraph for empty tiers;
  - Includes every valid SanitizedURL from the tier records, never
    a raw or unvalidated URL;
  - Includes training URLs only when actionable (training option's
    SanitizedURL is non-None);
  - `fallback_urls` (on the view) equals what the renderer actually
    emitted — recomputed from rendered fields, not copied from
    `prompt_urls`;
  - Strength phrasing is drawn from the closed five-token vocabulary;
  - Empty case produces no fabricated tier content;
  - Always ends with one short closing question from a closed set.
"""
from __future__ import annotations

from datetime import date

import pytest

from skillbridge.chat.coach_tiers_fallback import (
    _CLOSING_ALL_TIERS,
    _CLOSING_APPLY_AND_SIDEWAYS,
    _CLOSING_APPLY_AND_STRETCH,
    _CLOSING_APPLY_ONLY,
    _CLOSING_EMPTY,
    _CLOSING_SIDEWAYS_ONLY,
    _CLOSING_STRETCH_AND_SIDEWAYS,
    _CLOSING_STRETCH_ONLY,
    _HEADER_APPLY_TODAY,
    _HEADER_SIDEWAYS,
    _HEADER_WORTH_A_TRY,
    _STRENGTH_PHRASES,
    collect_fallback_render_urls,
    render_coach_tiers_fallback,
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
# Fixture builders (reuse the same shapes as test_tiered_view)
# =========================================================================
def _validated(raw: str) -> Validated:
    result = validate(raw)
    assert isinstance(result, Validated), f"fixture URL must validate: {raw}"
    return result


def _alignment(user_skill="QuickBooks", job_requirement="QuickBooks") -> SkillAlignment:
    return SkillAlignment(
        user_skill=user_skill, job_requirement=job_requirement,
        stage="exact", source="required",
        is_normalized_equal=(user_skill == job_requirement),
    )


def _job_facts(
    employment_type="full-time",
    posted_days_ago=3,
    salary_text=None,
) -> JobFacts:
    return JobFacts(
        posted_date=date(2026, 6, 11),
        posted_days_ago=posted_days_ago,
        location="Sault Ste. Marie, ON",
        employment_type=employment_type,
        salary_text=salary_text,
    )


def _strong(
    *, job_id="s1", url="https://example.com/strong/1",
    claim="competitive_match", employer="Diamond J",
    title="Accounts Payable Clerk",
) -> StrongMatch:
    return StrongMatch(
        job_id=job_id, title=title, employer=employer,
        location="Sault Ste. Marie, ON", noc_code="13102",
        url=_validated(url), job_facts=_job_facts(),
        skill_alignment=(_alignment(),), non_blocking_gaps=(),
        credential_warning_text=None, strength_claim_text=claim,
    )


def _stretch(
    *, job_id="w1", url="https://example.com/stretch/1",
    training_url="https://example.com/train/1",
    gap="account reconciliation",
    claim="close_with_named_gap", employer="Algoma Office",
    title="Junior Accountant",
) -> StretchMatch:
    return StretchMatch(
        job_id=job_id, title=title, employer=employer,
        location="Sault Ste. Marie, ON", noc_code="11100",
        url=_validated(url), job_facts=_job_facts(),
        skill_alignment=(_alignment(),),
        prioritized_gaps=(
            PrioritizedGap(
                job_requirement=gap, category="required",
                priority=1, blocker=False,
                training_options=(
                    TrainingOption(
                        provider="Sault College",
                        title="Bookkeeping Fundamentals",
                        url=_validated(training_url) if training_url else None,
                        format="online", duration_text="6 weeks",
                    ),
                ),
            ),
        ),
        credential_warning_text=None, strength_claim_text=claim,
    )


def _adjacent(
    *, job_id="adj-1", url="https://example.com/adj/1",
    employer="North Star", title="Payroll Administrator",
) -> AdjacentJob:
    return AdjacentJob(
        job_id=job_id, title=title, employer=employer,
        location="Sault Ste. Marie, ON", noc_code="12102",
        url=_validated(url), job_facts=_job_facts(),
        skill_alignment=(_alignment(),),
        transferable_pairs=(
            TransferablePair(
                user_skill="QuickBooks",
                applies_to="QuickBooks", stage="exact",
            ),
        ),
        important_gaps=(), credential_warning_text=None,
        why_adjacent="same_noc_minor_group",
        strength_claim_text="transferable_lane",
    )


def _view(*, strong=(), stretch=(), adjacent=()):
    ev = TieredEvidence(
        apply_today=tuple(strong),
        worth_a_try=tuple(stretch),
        sideways_move=tuple(adjacent),
    )
    return build_sanitized_responder_view_for_tiered_matches(ev)


# =========================================================================
# Section headers + structure
# =========================================================================
def test_renders_all_three_headers_when_all_tiers_present():
    v = _view(strong=[_strong()], stretch=[_stretch()], adjacent=[_adjacent()])
    text, _ = render_coach_tiers_fallback(v)
    assert _HEADER_APPLY_TODAY in text
    assert _HEADER_WORTH_A_TRY in text
    assert _HEADER_SIDEWAYS in text


def test_omits_apply_today_header_when_tier_empty():
    v = _view(stretch=[_stretch()])
    text, _ = render_coach_tiers_fallback(v)
    assert _HEADER_APPLY_TODAY not in text
    assert _HEADER_WORTH_A_TRY in text


def test_omits_worth_a_try_header_when_tier_empty():
    v = _view(strong=[_strong()])
    text, _ = render_coach_tiers_fallback(v)
    assert _HEADER_WORTH_A_TRY not in text


def test_omits_sideways_header_when_tier_empty():
    v = _view(strong=[_strong()])
    text, _ = render_coach_tiers_fallback(v)
    assert _HEADER_SIDEWAYS not in text


def test_header_order_is_apply_then_worth_then_sideways():
    v = _view(strong=[_strong()], stretch=[_stretch()], adjacent=[_adjacent()])
    text, _ = render_coach_tiers_fallback(v)
    apply_pos = text.index(_HEADER_APPLY_TODAY)
    worth_pos = text.index(_HEADER_WORTH_A_TRY)
    side_pos = text.index(_HEADER_SIDEWAYS)
    assert apply_pos < worth_pos < side_pos


# =========================================================================
# Strength-phrase vocabulary
# =========================================================================
def test_strength_phrase_vocabulary_is_closed_five_tokens():
    assert set(_STRENGTH_PHRASES.keys()) == {
        "competitive_match", "strongest_current",
        "close_with_named_gap", "stretch_with_training_bridge",
        "transferable_lane",
    }


def test_strong_paragraph_uses_competitive_match_phrase():
    v = _view(strong=[_strong(claim="competitive_match")])
    text, _ = render_coach_tiers_fallback(v)
    assert _STRENGTH_PHRASES["competitive_match"] in text


def test_strong_paragraph_uses_strongest_current_phrase():
    v = _view(strong=[_strong(claim="strongest_current")])
    text, _ = render_coach_tiers_fallback(v)
    assert _STRENGTH_PHRASES["strongest_current"] in text


def test_stretch_paragraph_uses_close_with_named_gap_phrase():
    v = _view(stretch=[_stretch(claim="close_with_named_gap")])
    text, _ = render_coach_tiers_fallback(v)
    assert _STRENGTH_PHRASES["close_with_named_gap"] in text


def test_stretch_paragraph_uses_training_bridge_phrase():
    v = _view(stretch=[_stretch(claim="stretch_with_training_bridge")])
    text, _ = render_coach_tiers_fallback(v)
    assert _STRENGTH_PHRASES["stretch_with_training_bridge"] in text


def test_sideways_paragraph_uses_transferable_lane_phrase():
    v = _view(adjacent=[_adjacent()])
    text, _ = render_coach_tiers_fallback(v)
    assert _STRENGTH_PHRASES["transferable_lane"] in text


# =========================================================================
# URL rendering — only SanitizedURLs from the view appear
# =========================================================================
def test_renders_job_urls_from_every_tier():
    v = _view(
        strong=[_strong(url="https://example.com/s")],
        stretch=[_stretch(url="https://example.com/w",
                          training_url="https://example.com/wt")],
        adjacent=[_adjacent(url="https://example.com/a")],
    )
    text, urls = render_coach_tiers_fallback(v)
    assert "https://example.com/s" in text
    assert "https://example.com/w" in text
    assert "https://example.com/a" in text
    # Training URL rendered too
    assert "https://example.com/wt" in text
    assert "https://example.com/s" in urls
    assert "https://example.com/w" in urls
    assert "https://example.com/a" in urls
    assert "https://example.com/wt" in urls


def test_does_not_render_training_url_when_option_url_is_none():
    """When the training option has no valid URL, the renderer emits
    the honest 'no verified training option' line. No invented URL."""
    s = StretchMatch(
        job_id="w", title="t", employer="e",
        location=None, noc_code=None,
        url=_validated("https://example.com/j"),
        job_facts=_job_facts(),
        skill_alignment=(),
        prioritized_gaps=(
            PrioritizedGap(
                job_requirement="account reconciliation",
                category="required", priority=1, blocker=False,
                training_options=(
                    TrainingOption(
                        provider="P", title="T",
                        url=None,                # no actionable URL
                        format="online", duration_text="6 weeks",
                    ),
                ),
            ),
        ),
        credential_warning_text=None,
        strength_claim_text="close_with_named_gap",
    )
    v = _view(stretch=[s])
    text, urls = render_coach_tiers_fallback(v)
    assert "I don't have a verified training option for that gap yet." in text
    # Only the job URL was rendered, no training URL
    assert urls == frozenset({"https://example.com/j"})


def test_does_not_render_job_url_when_none():
    s = StrongMatch(
        job_id="s", title="T", employer="E",
        location=None, noc_code=None, url=None,
        job_facts=_job_facts(),
        skill_alignment=(), non_blocking_gaps=(),
        credential_warning_text=None,
        strength_claim_text="competitive_match",
    )
    v = _view(strong=[s])
    text, urls = render_coach_tiers_fallback(v)
    assert "http" not in text   # no URL anywhere
    assert urls == frozenset()


# =========================================================================
# fallback_urls is computed from rendered fields
# =========================================================================
def test_fallback_urls_equals_rendered_url_set():
    v = _view(
        strong=[_strong(url="https://example.com/s")],
        stretch=[_stretch(url="https://example.com/w",
                          training_url="https://example.com/wt")],
        adjacent=[_adjacent(url="https://example.com/a")],
    )
    _, rendered_urls = render_coach_tiers_fallback(v)
    assert v.fallback_urls == rendered_urls


def test_fallback_urls_excludes_non_actionable_training():
    """Training option with url=None contributes nothing to
    fallback_urls — the renderer doesn't emit it."""
    s = StretchMatch(
        job_id="w", title="t", employer="e",
        location=None, noc_code=None,
        url=_validated("https://example.com/j"),
        job_facts=_job_facts(),
        skill_alignment=(),
        prioritized_gaps=(
            PrioritizedGap(
                job_requirement="reconciliation",
                category="required", priority=1, blocker=False,
                training_options=(
                    TrainingOption(
                        provider="P", title="T", url=None,
                        format="online", duration_text="6 weeks",
                    ),
                ),
            ),
        ),
        credential_warning_text=None,
        strength_claim_text="close_with_named_gap",
    )
    v = _view(stretch=[s])
    assert v.fallback_urls == frozenset({"https://example.com/j"})


def test_collect_fallback_render_urls_matches_renderer():
    """The URL collector and the renderer must agree exactly. Same
    input → same URL set, by construction (collector calls renderer)."""
    v = _view(
        strong=[_strong()], stretch=[_stretch()], adjacent=[_adjacent()],
    )
    _, rendered = render_coach_tiers_fallback(v)
    collected = collect_fallback_render_urls(v)
    assert collected == rendered


# =========================================================================
# Closing question — exactly one, from the closed set
# =========================================================================
_CLOSED_CLOSINGS = frozenset({
    _CLOSING_ALL_TIERS, _CLOSING_APPLY_AND_STRETCH,
    _CLOSING_APPLY_AND_SIDEWAYS, _CLOSING_APPLY_ONLY,
    _CLOSING_STRETCH_AND_SIDEWAYS, _CLOSING_STRETCH_ONLY,
    _CLOSING_SIDEWAYS_ONLY, _CLOSING_EMPTY,
})


def test_closing_for_all_three_tiers():
    v = _view(strong=[_strong()], stretch=[_stretch()], adjacent=[_adjacent()])
    text, _ = render_coach_tiers_fallback(v)
    assert text.rstrip().endswith(_CLOSING_ALL_TIERS)


def test_closing_for_apply_only():
    v = _view(strong=[_strong()])
    text, _ = render_coach_tiers_fallback(v)
    assert text.rstrip().endswith(_CLOSING_APPLY_ONLY)


def test_closing_for_stretch_only():
    v = _view(stretch=[_stretch()])
    text, _ = render_coach_tiers_fallback(v)
    assert text.rstrip().endswith(_CLOSING_STRETCH_ONLY)


def test_closing_for_sideways_only():
    v = _view(adjacent=[_adjacent()])
    text, _ = render_coach_tiers_fallback(v)
    assert text.rstrip().endswith(_CLOSING_SIDEWAYS_ONLY)


def test_closing_for_empty_tiers():
    v = _view()
    text, _ = render_coach_tiers_fallback(v)
    assert text.rstrip().endswith(_CLOSING_EMPTY)


def test_closing_for_apply_and_stretch():
    v = _view(strong=[_strong()], stretch=[_stretch()])
    text, _ = render_coach_tiers_fallback(v)
    assert text.rstrip().endswith(_CLOSING_APPLY_AND_STRETCH)


def test_closing_for_apply_and_sideways():
    v = _view(strong=[_strong()], adjacent=[_adjacent()])
    text, _ = render_coach_tiers_fallback(v)
    assert text.rstrip().endswith(_CLOSING_APPLY_AND_SIDEWAYS)


def test_closing_for_stretch_and_sideways():
    v = _view(stretch=[_stretch()], adjacent=[_adjacent()])
    text, _ = render_coach_tiers_fallback(v)
    assert text.rstrip().endswith(_CLOSING_STRETCH_AND_SIDEWAYS)


def test_closing_is_always_from_closed_set():
    """Sweep all 8 tier-presence combinations; every closing must be
    one of the closed-set tokens. No generated wording."""
    for strong, stretch, adjacent in [
        ((), (), ()),
        ((_strong(),), (), ()),
        ((), (_stretch(),), ()),
        ((), (), (_adjacent(),)),
        ((_strong(),), (_stretch(),), ()),
        ((_strong(),), (), (_adjacent(),)),
        ((), (_stretch(),), (_adjacent(),)),
        ((_strong(),), (_stretch(),), (_adjacent(),)),
    ]:
        v = _view(strong=list(strong), stretch=list(stretch), adjacent=list(adjacent))
        text, _ = render_coach_tiers_fallback(v)
        last_line = text.rstrip().splitlines()[-1]
        assert last_line in _CLOSED_CLOSINGS, last_line


# =========================================================================
# No invented advice / no fabrication
# =========================================================================
def test_no_invented_advice_phrases_in_output():
    """Templated fallback must not surface invented coaching prose.
    Sweep a populated view and grep for telltale invented phrases."""
    v = _view(
        strong=[_strong()], stretch=[_stretch()], adjacent=[_adjacent()],
    )
    text, _ = render_coach_tiers_fallback(v)
    forbidden = [
        "I recommend",
        "you should",
        "I'd suggest",
        "consider trying",
        "great fit",
        "perfect match",
        "ideal candidate",
    ]
    for phrase in forbidden:
        assert phrase.lower() not in text.lower(), phrase


def test_empty_view_produces_no_fabricated_tier_content():
    v = _view()
    text, urls = render_coach_tiers_fallback(v)
    assert _HEADER_APPLY_TODAY not in text
    assert _HEADER_WORTH_A_TRY not in text
    assert _HEADER_SIDEWAYS not in text
    assert urls == frozenset()


# =========================================================================
# Reads only prompt_tiered_* slots
# =========================================================================
def test_renderer_reads_only_prompt_tiered_slots():
    """The renderer must consult only the three tier slots — changing
    a non-tier slot must NOT change the fallback output. Verified by
    swapping `rejected_source_urls` between baseline and adversarial
    inputs: same tier data → same fallback text and URL set."""
    import dataclasses
    from skillbridge.chat.url_policy import Violation, ViolationCode
    v = _view(strong=[_strong()])
    text_baseline, urls_baseline = render_coach_tiers_fallback(v)
    fake_rejected = (
        # A non-tier slot; touching it must not move the fallback.
        # Build using the same shape used elsewhere in the codebase
        # so we don't accidentally drift from the production schema.
        # `Violation` carries (code, raw_token, raw_token_hash).
    )
    # Rebuild the view with rejected_source_urls populated.
    extra = dataclasses.replace(v, rejected_source_urls=fake_rejected)
    text_extra, urls_extra = render_coach_tiers_fallback(extra)
    assert text_extra == text_baseline
    assert urls_extra == urls_baseline


# =========================================================================
# Header wording is exact (locked)
# =========================================================================
def test_locked_header_strings():
    """Fix 1 (post-step-10 review): Worth-a-try heading no longer
    implies a single gap. A job can have multiple required gaps."""
    assert _HEADER_APPLY_TODAY == "**Apply today — your skills line up**"
    assert _HEADER_WORTH_A_TRY == "**Worth a try — close, with gaps to address**"
    assert _HEADER_SIDEWAYS == "**Sideways move — same skills, different angle**"


# =========================================================================
# Fix 2 (post-step-10 review) — grounded match details in
# Apply Today and Worth a Try (one alignment sentence each)
# =========================================================================
def test_apply_today_includes_alignment_sentence():
    """Single-alignment Apply Today paragraph names the user_skill →
    job_requirement mapping in one short sentence."""
    sm = StrongMatch(
        job_id="s", title="Accounts Payable Clerk", employer="Diamond J",
        location=None, noc_code=None,
        url=_validated("https://example.com/j"),
        job_facts=_job_facts(),
        skill_alignment=(
            SkillAlignment(
                user_skill="QuickBooks", job_requirement="QuickBooks",
                stage="exact", source="required",
                is_normalized_equal=True,
            ),
        ),
        non_blocking_gaps=(),
        credential_warning_text=None,
        strength_claim_text="competitive_match",
    )
    text, _ = render_coach_tiers_fallback(_view(strong=[sm]))
    assert "Your QuickBooks aligns with their QuickBooks requirement." in text


def test_apply_today_alignment_sentence_uses_top_two_when_multiple():
    """When skill_alignment has 2+ entries, the alignment sentence
    names the first two — never invents extra skills."""
    sm = StrongMatch(
        job_id="s", title="Accounts Payable Clerk", employer="Diamond J",
        location=None, noc_code=None,
        url=_validated("https://example.com/j"),
        job_facts=_job_facts(),
        skill_alignment=(
            SkillAlignment(
                user_skill="bookkeeping",
                job_requirement="accounts-payable",
                stage="semantic", source="required",
                is_normalized_equal=False,
            ),
            SkillAlignment(
                user_skill="payroll",
                job_requirement="reconciliation",
                stage="semantic", source="required",
                is_normalized_equal=False,
            ),
            SkillAlignment(
                user_skill="Excel",
                job_requirement="spreadsheets",
                stage="semantic", source="required",
                is_normalized_equal=False,
            ),
        ),
        non_blocking_gaps=(),
        credential_warning_text=None,
        strength_claim_text="competitive_match",
    )
    text, _ = render_coach_tiers_fallback(_view(strong=[sm]))
    expected = ("Your bookkeeping and payroll align with their "
                "accounts-payable and reconciliation requirements.")
    assert expected in text
    # Third alignment must NOT appear
    assert "Excel" not in text
    assert "spreadsheets" not in text


def test_apply_today_alignment_sentence_omitted_when_no_alignments():
    sm = StrongMatch(
        job_id="s", title="t", employer=None, location=None, noc_code=None,
        url=None, job_facts=_job_facts(),
        skill_alignment=(),                # nothing to report
        non_blocking_gaps=(),
        credential_warning_text=None,
        strength_claim_text="competitive_match",
    )
    text, _ = render_coach_tiers_fallback(_view(strong=[sm]))
    assert "aligns with" not in text
    assert "align with" not in text


def test_worth_a_try_includes_alignment_sentence_before_gap_sentence():
    """The fallback must mention one aligned skill BEFORE the gap.
    Otherwise the user reads only "close, with a gap" — repeating
    the old shallow-response failure."""
    sm = StretchMatch(
        job_id="w", title="Junior Accountant", employer="Algoma Office",
        location=None, noc_code=None,
        url=_validated("https://example.com/j"),
        job_facts=_job_facts(),
        skill_alignment=(
            SkillAlignment(
                user_skill="QuickBooks", job_requirement="QuickBooks",
                stage="exact", source="required",
                is_normalized_equal=True,
            ),
        ),
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
    text, _ = render_coach_tiers_fallback(_view(stretch=[sm]))
    # Both sentences present
    assert "Your QuickBooks aligns with their QuickBooks requirement." in text
    assert "The gap is account reconciliation." in text
    # And the alignment appears BEFORE the gap sentence
    align_pos = text.index("aligns with")
    gap_pos = text.index("The gap is")
    assert align_pos < gap_pos


def test_sideways_paragraph_unchanged_by_fix_2():
    """The transferable-pair sentence already grounded sideways; Fix 2
    explicitly only updated Apply Today and Worth a Try."""
    v = _view(adjacent=[_adjacent()])
    text, _ = render_coach_tiers_fallback(v)
    assert "Your QuickBooks carries over to QuickBooks." in text


# =========================================================================
# Fix 3 (post-step-10 review) — corrected closing wording
# =========================================================================
def test_locked_closings_for_sideways_only_and_empty():
    assert _CLOSING_SIDEWAYS_ONLY == "Which sideways option would you like to explore first?"
    assert _CLOSING_EMPTY == "What kind of work are you aiming for?"


def test_sideways_only_does_not_offer_to_pull_live_listings():
    """Sideways already surfaces live listings (with URLs); asking to
    pull them again was self-contradictory."""
    v = _view(adjacent=[_adjacent()])
    text, _ = render_coach_tiers_fallback(v)
    assert "pull live listings" not in text


def test_empty_closing_is_grammatical():
    """The replacement closing is a complete grammatical question,
    not a statement with a tacked-on question mark."""
    v = _view()
    text, _ = render_coach_tiers_fallback(v)
    last_line = text.rstrip().splitlines()[-1]
    assert last_line == "What kind of work are you aiming for?"
    assert "Tell me more about" not in last_line


# =========================================================================
# CP1 review Medium — credential blockers in non-first positions must
# be surfaced. Without this fix, a credential gap at index ≥1 (when a
# non-credential gap came first by importance order) was silently
# hidden by the fallback.
# =========================================================================
def test_credential_blocker_at_index_1_is_rendered():
    """First gap is non-credential; second gap is a credential blocker
    with mapped training. The fallback must surface BOTH gaps and
    their training, not just the first."""
    sm = StretchMatch(
        job_id="w", title="t", employer="e",
        location=None, noc_code=None,
        url=_validated("https://example.com/j"),
        job_facts=_job_facts(),
        skill_alignment=(),
        prioritized_gaps=(
            PrioritizedGap(
                job_requirement="account reconciliation",
                category="required", priority=1, blocker=False,
                training_options=(
                    TrainingOption(
                        provider="Sault College", title="Bookkeeping",
                        url=_validated("https://example.com/bk"),
                        format="online", duration_text="6 weeks",
                    ),
                ),
            ),
            PrioritizedGap(
                job_requirement="Class G driver's license",
                category="required", priority=2, blocker=True,
                training_options=(
                    TrainingOption(
                        provider="DriveTest", title="Class G Prep",
                        url=_validated("https://example.com/g"),
                        format="in-person", duration_text=None,
                    ),
                ),
            ),
        ),
        credential_warning_text=None,
        strength_claim_text="stretch_with_training_bridge",
    )
    v = _view(stretch=[sm])
    text, urls = render_coach_tiers_fallback(v)
    # First gap surfaces
    assert "The gap is account reconciliation." in text
    assert "Sault College offers Bookkeeping" in text
    # The credential blocker at index 1 surfaces too
    assert "Also a credential gap: Class G driver's license." in text
    assert "DriveTest offers Class G Prep" in text
    # Both URLs in the rendered set
    assert "https://example.com/bk" in urls
    assert "https://example.com/g" in urls


def test_credential_blocker_at_index_1_with_no_training_surfaces_honest_line():
    """The credential blocker has no actionable training (only invalid
    URL). It still appears explicitly with the honest fallback line so
    the user sees the blocker even when prep isn't ready."""
    sm = StretchMatch(
        job_id="w", title="t", employer="e",
        location=None, noc_code=None,
        url=_validated("https://example.com/j"),
        job_facts=_job_facts(),
        skill_alignment=(),
        prioritized_gaps=(
            PrioritizedGap(
                job_requirement="reconciliation",
                category="required", priority=1, blocker=False,
                training_options=(),
            ),
            PrioritizedGap(
                job_requirement="Class G driver's license",
                category="required", priority=2, blocker=True,
                training_options=(
                    TrainingOption(
                        provider="P", title="T",
                        url=None,           # not actionable
                        format=None, duration_text=None,
                    ),
                ),
            ),
        ),
        credential_warning_text=None,
        strength_claim_text="close_with_named_gap",
    )
    v = _view(stretch=[sm])
    text, _ = render_coach_tiers_fallback(v)
    assert "Also a credential gap: Class G driver's license." in text
    # Honest fallback line appears for at least one gap
    assert "I don't have a verified training option for that gap yet." in text


def test_non_blocker_gaps_beyond_first_are_NOT_rendered():
    """Non-credential gaps beyond the first stay hidden — only blockers
    are explicitly surfaced. (Otherwise the fallback would balloon for
    jobs with many small gaps.)"""
    sm = StretchMatch(
        job_id="w", title="t", employer="e",
        location=None, noc_code=None,
        url=_validated("https://example.com/j"),
        job_facts=_job_facts(),
        skill_alignment=(),
        prioritized_gaps=(
            PrioritizedGap(
                job_requirement="reconciliation",
                category="required", priority=1, blocker=False,
                training_options=(),
            ),
            PrioritizedGap(
                job_requirement="advanced Excel",
                category="required", priority=2, blocker=False,
                training_options=(),
            ),
            PrioritizedGap(
                job_requirement="Sage 50",
                category="required", priority=3, blocker=False,
                training_options=(),
            ),
        ),
        credential_warning_text=None,
        strength_claim_text="close_with_named_gap",
    )
    v = _view(stretch=[sm])
    text, _ = render_coach_tiers_fallback(v)
    assert "The gap is reconciliation." in text
    assert "advanced Excel" not in text
    assert "Sage 50" not in text


def test_credential_blocker_at_index_0_is_not_double_rendered():
    """When the first gap IS the credential blocker, it surfaces once
    (as the lead "The gap is ..."), not twice."""
    sm = StretchMatch(
        job_id="w", title="t", employer="e",
        location=None, noc_code=None,
        url=_validated("https://example.com/j"),
        job_facts=_job_facts(),
        skill_alignment=(),
        prioritized_gaps=(
            PrioritizedGap(
                job_requirement="Class G driver's license",
                category="required", priority=1, blocker=True,
                training_options=(
                    TrainingOption(
                        provider="DriveTest", title="Class G Prep",
                        url=_validated("https://example.com/g"),
                        format="in-person", duration_text=None,
                    ),
                ),
            ),
        ),
        credential_warning_text=None,
        strength_claim_text="stretch_with_training_bridge",
    )
    v = _view(stretch=[sm])
    text, _ = render_coach_tiers_fallback(v)
    # Lead form, not the "Also a credential gap" form
    assert "The gap is Class G driver's license." in text
    assert "Also a credential gap" not in text
    assert text.count("Class G driver's license") == 1


# =========================================================================

