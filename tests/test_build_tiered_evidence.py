"""AR-9.feat.coach-tiers CP1 step 8 — `build_tiered_evidence` deterministic
three-tier partition.

Pins:
  Apply today:
    band in {"strong","good"} AND required_missing == [];
  Worth a try:
    band in {"good","stretch"} AND at least one required gap;
    credential-only gap profile included ONLY when at least one
    credential gap has mapped training;
  Sideways move:
    accepted_adjacent enriched, with job_ids in direct tiers excluded;
  Caps:
    Strong=3, Stretch=2, Adjacent=3 (defaults);
  Order:
    direct-tier order follows the caller's `results` order;
    Sideways order follows the caller's `accepted_adjacent` order;
  Tier exclusivity:
    a job_id appears in at most one tier;
  material flag:
    True for at most one preferred miss per Apply-today record (the
    first, per importance order);
  strength_claim_text:
    closed enum — never any other string;
  No prompt / presentation / prose in step 8.
"""
from __future__ import annotations

from datetime import date

import pytest

from skillbridge.chat import tiered_evidence as te
from skillbridge.chat.tiered_evidence import (
    AdjacentJob,
    JobFacts,
    NonBlockingGap,
    PrioritizedGap,
    StretchMatch,
    StrongMatch,
    TieredEvidence,
    TrainingOption,
    build_tiered_evidence,
)
from skillbridge.match.alignment import UserSkillRow
from skillbridge.match.engine import MatchResult

pytestmark = pytest.mark.nodb


@pytest.fixture(autouse=True)
def _no_db_regulated_lookup(monkeypatch):
    """Sideways calls `_credential_warning_text` which queries the DB
    via `_regulated`. Short-circuit for nodb runs (same pattern as
    test_enrich_adjacency)."""
    monkeypatch.setattr(te, "_regulated", lambda noc, target: None)


# =========================================================================
# Fixture builders
# =========================================================================
def _row(text, *, sid=None):
    from skillbridge.match.aliases import canonicalize_skill
    stripped = text.strip()
    return UserSkillRow(
        skill_id=sid,
        text=stripped,
        name=stripped.lower(),
        canon=canonicalize_skill(stripped) or "",
    )


def _sets(rows):
    ids = {r.skill_id for r in rows if r.skill_id}
    names = {r.name for r in rows}
    canons = {r.canon for r in rows if r.canon}
    return ids, names, canons


def _result(
    *,
    job_id="j1",
    title="Job One",
    band="strong",
    eligible=True,
    matched=None,
    missing=None,
    required_missing=None,
    employer="Acme",
    url="https://example.com/job/1",
    location="Sault Ste. Marie, ON",
    noc_code="12102",
    employment_type="full-time",
    salary_text="$22-24/hr",
    posted_date=None,
    credential_warning=None,
) -> MatchResult:
    """Build a MatchResult fixture with minimum-viable defaults."""
    matched = matched or []
    missing = missing or []
    required_missing = required_missing if required_missing is not None else []
    return MatchResult(
        job_id=job_id,
        profile_id="p1",
        title=title,
        employer=employer,
        url=url,
        location=location,
        match_score=0.80 if band == "strong" else 0.65 if band == "good" else 0.45,
        match_band=band,
        match_eligible=eligible,
        ineligibility_reason=None,
        matched_skills=matched,
        missing_skills=missing,
        matched_skill_ids=[None] * len(matched),
        missing_skill_ids=[None] * len(missing),
        required_skills_count=len(matched) + len(missing),
        credential_warning=credential_warning,
        posted_date=posted_date,
        noc_code=noc_code,
        score_explanation={"required_missing": required_missing},
        employment_type=employment_type,
        salary_text=salary_text,
    )


def _accepted_adj(*, job_id="adj-1", title="Sideways Job", noc_code="73402"):
    return {
        "job_id": job_id,
        "title": title,
        "employer": "Adj Co",
        "url": "https://example.com/adj/1",
        "location": "Sault Ste. Marie, ON",
        "noc_code": noc_code,
        "employment_type": "full-time",
        "salary_text": None,
        "posted_date": None,
        "skills": [{"skill_id": None, "skill_name": "QuickBooks", "skill_type": "required"}],
    }


def _base_inputs():
    rows = [_row("QuickBooks"), _row("Excel")]
    ids, names, canons = _sets(rows)
    return rows, ids, names, canons


# =========================================================================
# Empty / shape
# =========================================================================
def test_returns_tiered_evidence_instance():
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence([], [], rows, ids, names, canons)
    assert isinstance(out, TieredEvidence)


def test_all_empty_when_no_results_or_adjacent():
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence([], [], rows, ids, names, canons)
    assert out.apply_today == ()
    assert out.worth_a_try == ()
    assert out.sideways_move == ()


def test_empty_tiers_stay_empty_no_placeholders():
    """Apply-today empty does NOT fabricate a synthetic placeholder
    record; tiers are empty tuples."""
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [_result(band="low")], [], rows, ids, names, canons,
    )
    assert out.apply_today == ()
    assert out.worth_a_try == ()
    assert out.sideways_move == ()


# =========================================================================
# Apply-today filter
# =========================================================================
def test_strong_band_no_gaps_is_apply_today():
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [_result(job_id="s1", band="strong", required_missing=[])],
        [], rows, ids, names, canons,
    )
    assert len(out.apply_today) == 1
    assert out.apply_today[0].job_id == "s1"


def test_good_band_no_gaps_is_apply_today():
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [_result(job_id="g1", band="good", required_missing=[])],
        [], rows, ids, names, canons,
    )
    assert len(out.apply_today) == 1


def test_strong_band_with_required_gap_lands_in_worth_a_try():
    """CP3 step 2 (2026-06-15): a strong-band match with a real
    required gap belongs in Worth a try. Apply today still requires
    `required_missing == []`, so a strong-with-gap record cannot be
    Apply today, but it must not fall off the tier surface either —
    it lands in Worth a try (there's a specific gap to close first,
    regardless of how well the overall score lined up).

    Inverts the earlier "falls out of direct tiers entirely" pin: the
    old behavior dropped these records to the legacy `present_matches`
    card surface; the new behavior keeps them in the tier prose."""
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [_result(job_id="x", band="strong",
                 missing=["Sage 50"], required_missing=["Sage 50"])],
        [], rows, ids, names, canons,
    )
    assert out.apply_today == ()
    assert len(out.worth_a_try) == 1
    assert out.worth_a_try[0].job_id == "x"


def test_stretch_band_no_gaps_not_in_apply_today():
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [_result(band="stretch", required_missing=[])],
        [], rows, ids, names, canons,
    )
    assert out.apply_today == ()


def test_low_band_excluded_from_all_direct_tiers():
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [_result(band="low", required_missing=["x"])],
        [], rows, ids, names, canons,
    )
    assert out.apply_today == ()
    assert out.worth_a_try == ()


def test_ineligible_excluded_from_all_direct_tiers():
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [_result(band="strong", eligible=False, required_missing=[])],
        [], rows, ids, names, canons,
    )
    assert out.apply_today == ()
    assert out.worth_a_try == ()


# =========================================================================
# Worth-a-try filter
# =========================================================================
def test_good_with_non_credential_gap_is_worth_a_try():
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [_result(job_id="w1", band="good",
                 missing=["invoice processing"],
                 required_missing=["invoice processing"])],
        [], rows, ids, names, canons,
    )
    assert len(out.worth_a_try) == 1
    assert out.worth_a_try[0].job_id == "w1"


def test_stretch_with_non_credential_gap_is_worth_a_try():
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [_result(band="stretch",
                 missing=["account reconciliation"],
                 required_missing=["account reconciliation"])],
        [], rows, ids, names, canons,
    )
    assert len(out.worth_a_try) == 1


def test_credential_only_gap_without_training_is_excluded():
    """A job whose ONLY required gap is a credential AND there's no
    mapped training cannot be coached forward — excluded entirely."""
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [_result(band="good",
                 missing=["Class G driver's license"],
                 required_missing=["Class G driver's license"])],
        [], rows, ids, names, canons,
        training_by_job={},
    )
    assert out.worth_a_try == ()
    assert out.apply_today == ()


def test_credential_only_gap_with_training_is_included():
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [_result(job_id="wc", band="good",
                 missing=["Class G driver's license"],
                 required_missing=["Class G driver's license"])],
        [], rows, ids, names, canons,
        training_by_job={
            "wc": [{
                "provider": "Sault College",
                "title": "Driver Prep",
                "url": "https://example.com/train/g",
                "for_skill": "Class G driver's license",
                "format": "in-person",
                "duration_text": "6 weeks",
            }],
        },
    )
    assert len(out.worth_a_try) == 1


def test_mixed_gaps_without_credential_training_excluded():
    """Fix 2 (step-8 review): a mixed gap profile (credential +
    non-credential) is NOT Worth a Try when the credential lacks
    training. The non-credential gap is addressable but the
    credential is a blocker — the job is not actionable end-to-end."""
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [_result(band="good",
                 missing=["Class G driver's license", "invoice processing"],
                 required_missing=["Class G driver's license", "invoice processing"])],
        [], rows, ids, names, canons,
        training_by_job={},
    )
    assert out.worth_a_try == ()


def test_mixed_gaps_with_credential_training_included():
    """Fix 2: same mixed gap is Worth a Try when the credential
    has mapped training."""
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [_result(job_id="m1", band="good",
                 missing=["Class G driver's license", "invoice processing"],
                 required_missing=["Class G driver's license", "invoice processing"])],
        [], rows, ids, names, canons,
        training_by_job={
            "m1": [{
                "provider": "Sault College",
                "title": "Driver Prep",
                "url": "https://example.com/g",
                "for_skill": "Class G driver's license",
            }],
        },
    )
    assert len(out.worth_a_try) == 1


# =========================================================================
# Sideways move
# =========================================================================
def test_sideways_includes_adjacent_not_in_direct_tiers():
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [], [_accepted_adj(job_id="adj-1")],
        rows, ids, names, canons,
    )
    assert len(out.sideways_move) == 1
    assert out.sideways_move[0].job_id == "adj-1"
    assert isinstance(out.sideways_move[0], AdjacentJob)


def test_sideways_excludes_apply_today_job_ids():
    """If an adjacent job's job_id coincides with an Apply-today
    job_id, the adjacent entry is filtered out by tier exclusivity."""
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [_result(job_id="overlap", band="strong", required_missing=[])],
        [_accepted_adj(job_id="overlap")],
        rows, ids, names, canons,
    )
    assert len(out.apply_today) == 1
    assert out.sideways_move == ()


def test_sideways_excludes_worth_a_try_job_ids():
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [_result(job_id="overlap", band="good",
                 missing=["invoice processing"],
                 required_missing=["invoice processing"])],
        [_accepted_adj(job_id="overlap")],
        rows, ids, names, canons,
    )
    assert len(out.worth_a_try) == 1
    assert out.sideways_move == ()


# =========================================================================
# Caps
# =========================================================================
def test_apply_today_capped_at_three():
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [
            _result(job_id=f"s{i}", band="strong", required_missing=[])
            for i in range(5)
        ],
        [], rows, ids, names, canons,
    )
    assert len(out.apply_today) == 3
    assert [m.job_id for m in out.apply_today] == ["s0", "s1", "s2"]


def test_worth_a_try_capped_at_two():
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [
            _result(job_id=f"w{i}", band="good",
                    missing=["x"], required_missing=["x"])
            for i in range(4)
        ],
        [], rows, ids, names, canons,
    )
    assert len(out.worth_a_try) == 2


def test_sideways_capped_at_three():
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [], [_accepted_adj(job_id=f"adj-{i}") for i in range(5)],
        rows, ids, names, canons,
    )
    assert len(out.sideways_move) == 3


def test_caps_overridable_via_kwargs():
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [
            _result(job_id=f"s{i}", band="strong", required_missing=[])
            for i in range(5)
        ],
        [], rows, ids, names, canons,
        strong_cap=1,
    )
    assert len(out.apply_today) == 1


# =========================================================================
# Order preservation
# =========================================================================
def test_apply_today_preserves_input_order():
    """Caller has already sorted by score; builder MUST NOT re-sort."""
    rows, ids, names, canons = _base_inputs()
    # Intentionally NOT sorted by score; builder preserves input order.
    results = [
        _result(job_id="z", band="strong", required_missing=[]),
        _result(job_id="a", band="good", required_missing=[]),
        _result(job_id="m", band="strong", required_missing=[]),
    ]
    out = build_tiered_evidence(results, [], rows, ids, names, canons)
    assert [m.job_id for m in out.apply_today] == ["z", "a", "m"]


def test_worth_a_try_preserves_input_order():
    rows, ids, names, canons = _base_inputs()
    results = [
        _result(job_id="z", band="good",
                missing=["x"], required_missing=["x"]),
        _result(job_id="a", band="stretch",
                missing=["y"], required_missing=["y"]),
    ]
    out = build_tiered_evidence(results, [], rows, ids, names, canons)
    assert [m.job_id for m in out.worth_a_try] == ["z", "a"]


def test_sideways_preserves_input_order():
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [], [
            _accepted_adj(job_id="zz"),
            _accepted_adj(job_id="aa"),
            _accepted_adj(job_id="mm"),
        ], rows, ids, names, canons,
    )
    assert [a.job_id for a in out.sideways_move] == ["zz", "aa", "mm"]


# =========================================================================
# Tier exclusivity (centrally enforced)
# =========================================================================
def test_job_id_in_apply_today_excluded_from_worth_a_try():
    """Strong-band match with NO required gaps belongs to Apply-today;
    it must NOT also be considered for Worth-a-try."""
    rows, ids, names, canons = _base_inputs()
    # Try to coerce a job into BOTH tiers by making it Apply-today-eligible
    # but also passing Worth-a-try band/gap filters — but a job with no
    # required gaps cannot satisfy Worth-a-try anyway. So pin the
    # complementary case: a good-band job WITH required gaps is Worth-a-try,
    # not Apply-today.
    out = build_tiered_evidence(
        [
            _result(job_id="a1", band="strong", required_missing=[]),
            _result(job_id="b1", band="good",
                    missing=["x"], required_missing=["x"]),
        ],
        [], rows, ids, names, canons,
    )
    apply_ids = {m.job_id for m in out.apply_today}
    worth_ids = {m.job_id for m in out.worth_a_try}
    assert apply_ids & worth_ids == set()


def test_no_job_id_appears_in_two_tiers():
    """Final exclusivity assertion across all three tiers."""
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [
            _result(job_id="a", band="strong", required_missing=[]),
            _result(job_id="b", band="good",
                    missing=["x"], required_missing=["x"]),
            _result(job_id="c", band="strong", required_missing=[]),
        ],
        [_accepted_adj(job_id="b"),    # overlap with worth-a-try
         _accepted_adj(job_id="d")],   # genuinely new
        rows, ids, names, canons,
    )
    apply_ids = {m.job_id for m in out.apply_today}
    worth_ids = {m.job_id for m in out.worth_a_try}
    side_ids = {a.job_id for a in out.sideways_move}
    assert apply_ids.isdisjoint(worth_ids)
    assert apply_ids.isdisjoint(side_ids)
    assert worth_ids.isdisjoint(side_ids)
    # "d" is the only sideways survivor.
    assert side_ids == {"d"}


# =========================================================================
# material flag (locked preferred-gap rule)
# =========================================================================
def test_first_preferred_miss_is_material():
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [_result(band="strong",
                 missing=["nice-to-have-1", "nice-to-have-2"],
                 required_missing=[])],
        [], rows, ids, names, canons,
    )
    gaps = out.apply_today[0].non_blocking_gaps
    assert len(gaps) == 2
    assert gaps[0].material is True
    assert gaps[1].material is False


def test_at_most_one_material_per_apply_today_record():
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [_result(band="strong",
                 missing=["a", "b", "c", "d"], required_missing=[])],
        [], rows, ids, names, canons,
    )
    gaps = out.apply_today[0].non_blocking_gaps
    materials = [g for g in gaps if g.material]
    assert len(materials) == 1


def test_no_material_when_no_preferred_misses():
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [_result(band="strong", missing=[], required_missing=[])],
        [], rows, ids, names, canons,
    )
    assert out.apply_today[0].non_blocking_gaps == ()


# =========================================================================
# strength_claim_text (closed enum)
# =========================================================================
_CLOSED_CLAIMS = frozenset({
    "competitive_match", "strongest_current",
    "close_with_named_gap", "stretch_with_training_bridge",
    "transferable_lane",
})


def test_strong_band_apply_today_uses_competitive_match():
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [_result(band="strong", required_missing=[])],
        [], rows, ids, names, canons,
    )
    assert out.apply_today[0].strength_claim_text == "competitive_match"


def test_good_band_apply_today_uses_strongest_current():
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [_result(band="good", required_missing=[])],
        [], rows, ids, names, canons,
    )
    assert out.apply_today[0].strength_claim_text == "strongest_current"


def test_stretch_non_credential_gap_uses_close_with_named_gap():
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [_result(band="stretch",
                 missing=["account reconciliation"],
                 required_missing=["account reconciliation"])],
        [], rows, ids, names, canons,
    )
    assert out.worth_a_try[0].strength_claim_text == "close_with_named_gap"


def test_stretch_credential_with_training_uses_stretch_with_training_bridge():
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [_result(job_id="wc", band="good",
                 missing=["Class G driver's license"],
                 required_missing=["Class G driver's license"])],
        [], rows, ids, names, canons,
        training_by_job={
            "wc": [{
                "provider": "P", "title": "T",
                "url": "https://example.com/t",
                "for_skill": "Class G driver's license",
            }],
        },
    )
    assert out.worth_a_try[0].strength_claim_text == "stretch_with_training_bridge"


def test_sideways_strength_claim_is_transferable_lane():
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [], [_accepted_adj()], rows, ids, names, canons,
    )
    assert out.sideways_move[0].strength_claim_text == "transferable_lane"


def test_every_emitted_strength_claim_is_in_closed_enum():
    """Sweep all three tiers across multiple shapes; every record's
    strength_claim_text must be a known token."""
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [
            _result(job_id="s1", band="strong", required_missing=[]),
            _result(job_id="g1", band="good", required_missing=[]),
            _result(job_id="w1", band="stretch",
                    missing=["x"], required_missing=["x"]),
        ],
        [_accepted_adj()], rows, ids, names, canons,
    )
    claims = [m.strength_claim_text for m in out.apply_today]
    claims += [m.strength_claim_text for m in out.worth_a_try]
    claims += [a.strength_claim_text for a in out.sideways_move]
    for c in claims:
        assert c in _CLOSED_CLAIMS, c


# =========================================================================
# Prioritized gap fields (Worth-a-try)
# =========================================================================
def test_prioritized_gap_priority_is_one_indexed():
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [_result(band="good",
                 missing=["g1", "g2", "g3"],
                 required_missing=["g1", "g2", "g3"])],
        [], rows, ids, names, canons,
    )
    gaps = out.worth_a_try[0].prioritized_gaps
    assert [g.priority for g in gaps] == [1, 2, 3]


def test_prioritized_gap_blocker_only_for_credentials():
    """Per Fix 2 the job must clear the credentials-need-training
    filter before its gaps reach Worth a Try; we supply training for
    the credential so the job is admitted and both gaps are projected."""
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [_result(job_id="b", band="good",
                 missing=["invoice processing", "Class G driver's license"],
                 required_missing=["invoice processing", "Class G driver's license"])],
        [], rows, ids, names, canons,
        training_by_job={
            "b": [{
                "provider": "P", "title": "Driver Prep",
                "url": "https://example.com/g",
                "for_skill": "Class G driver's license",
            }],
        },
    )
    gaps = out.worth_a_try[0].prioritized_gaps
    by_name = {g.job_requirement: g for g in gaps}
    assert by_name["invoice processing"].blocker is False
    assert by_name["Class G driver's license"].blocker is True


def test_prioritized_gap_training_options_mapped_by_for_skill():
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [_result(job_id="w", band="good",
                 missing=["account reconciliation"],
                 required_missing=["account reconciliation"])],
        [], rows, ids, names, canons,
        training_by_job={
            "w": [{
                "provider": "Sault College",
                "title": "Bookkeeping Fundamentals",
                "url": "https://example.com/t",
                "for_skill": "account reconciliation",
                "format": "online",
                "duration_text": "6 weeks",
            }],
        },
    )
    gap = out.worth_a_try[0].prioritized_gaps[0]
    assert len(gap.training_options) == 1
    t = gap.training_options[0]
    assert isinstance(t, TrainingOption)
    assert t.provider == "Sault College"
    assert t.format == "online"
    assert t.duration_text == "6 weeks"


def test_prioritized_gap_training_empty_when_no_mapping():
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [_result(band="good",
                 missing=["unmapped skill"],
                 required_missing=["unmapped skill"])],
        [], rows, ids, names, canons,
    )
    gap = out.worth_a_try[0].prioritized_gaps[0]
    assert gap.training_options == ()


def test_prioritized_gap_invalid_format_becomes_none():
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [_result(job_id="w", band="good",
                 missing=["skill x"], required_missing=["skill x"])],
        [], rows, ids, names, canons,
        training_by_job={
            "w": [{
                "provider": "P", "title": "T",
                "url": "https://example.com/t",
                "for_skill": "skill x",
                "format": "self-paced-virtual",   # not in closed set
            }],
        },
    )
    gap = out.worth_a_try[0].prioritized_gaps[0]
    assert gap.training_options[0].format is None


# =========================================================================
# Projection fields (StrongMatch / StretchMatch)
# =========================================================================
def test_strong_match_carries_job_facts():
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [_result(band="strong", required_missing=[],
                 posted_date=date(2026, 6, 14),
                 employment_type="full-time", salary_text="$22-24/hr")],
        [], rows, ids, names, canons,
    )
    f = out.apply_today[0].job_facts
    assert isinstance(f, JobFacts)
    assert f.employment_type == "full-time"
    assert f.salary_text == "$22-24/hr"


def test_strong_match_carries_skill_alignment_passthrough():
    """skill_alignment on the StrongMatch is the tuple from MatchResult."""
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [_result(band="strong", required_missing=[])],
        [], rows, ids, names, canons,
    )
    # Default fixture has no skill_alignment on the MatchResult, so
    # the projection is empty too — just verify field-type.
    assert isinstance(out.apply_today[0].skill_alignment, tuple)


def test_strong_match_credential_warning_text_is_passthrough():
    """credential_warning on MatchResult flows through unchanged."""
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [_result(band="strong", required_missing=[],
                 credential_warning="Some occupational notice.")],
        [], rows, ids, names, canons,
    )
    assert out.apply_today[0].credential_warning_text == "Some occupational notice."


# =========================================================================
# Dataclass frozen smoke
# =========================================================================
def test_strong_match_is_frozen():
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [_result(band="strong", required_missing=[])],
        [], rows, ids, names, canons,
    )
    with pytest.raises((AttributeError, Exception)):
        out.apply_today[0].title = "x"  # type: ignore


def test_stretch_match_is_frozen():
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [_result(band="good",
                 missing=["x"], required_missing=["x"])],
        [], rows, ids, names, canons,
    )
    with pytest.raises((AttributeError, Exception)):
        out.worth_a_try[0].title = "x"  # type: ignore


def test_tiered_evidence_is_frozen():
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence([], [], rows, ids, names, canons)
    with pytest.raises((AttributeError, Exception)):
        out.apply_today = ()  # type: ignore


def test_non_blocking_gap_and_prioritized_gap_are_frozen():
    g = NonBlockingGap(job_requirement="x", material=True)
    with pytest.raises((AttributeError, Exception)):
        g.material = False  # type: ignore
    p = PrioritizedGap(
        job_requirement="x", category="required",
        priority=1, blocker=False, training_options=(),
    )
    with pytest.raises((AttributeError, Exception)):
        p.priority = 2  # type: ignore


# =========================================================================
# Step-8 review Fix 1 — fail closed on missing explanations
# =========================================================================
def _result_no_explanation(*, band="strong", eligible=True):
    """A MatchResult whose score_explanation is None — i.e. the matcher
    did NOT report required-missing data."""
    return MatchResult(
        job_id="ne", profile_id="p1", title="T", employer=None, url=None,
        location=None, match_score=0.8, match_band=band,
        match_eligible=eligible, ineligibility_reason=None,
        matched_skills=[], missing_skills=[],
        matched_skill_ids=[], missing_skill_ids=[],
        required_skills_count=0, credential_warning=None,
        posted_date=None, noc_code=None,
        score_explanation=None,
    )


def _result_explanation_no_required_missing(*, band="strong"):
    """score_explanation is a dict but lacks the required_missing key."""
    return MatchResult(
        job_id="nrm", profile_id="p1", title="T", employer=None, url=None,
        location=None, match_score=0.8, match_band=band,
        match_eligible=True, ineligibility_reason=None,
        matched_skills=[], missing_skills=[],
        matched_skill_ids=[], missing_skill_ids=[],
        required_skills_count=0, credential_warning=None,
        posted_date=None, noc_code=None,
        score_explanation={"some_other_field": "x"},
    )


def _result_required_missing_none(*, band="strong"):
    """score_explanation['required_missing'] is explicitly None."""
    return MatchResult(
        job_id="rmn", profile_id="p1", title="T", employer=None, url=None,
        location=None, match_score=0.8, match_band=band,
        match_eligible=True, ineligibility_reason=None,
        matched_skills=[], missing_skills=[],
        matched_skill_ids=[], missing_skill_ids=[],
        required_skills_count=0, credential_warning=None,
        posted_date=None, noc_code=None,
        score_explanation={"required_missing": None},
    )


def test_fix1_absent_score_explanation_blocks_apply_today():
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [_result_no_explanation(band="strong")],
        [], rows, ids, names, canons,
    )
    assert out.apply_today == ()


def test_fix1_absent_required_missing_key_blocks_apply_today():
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [_result_explanation_no_required_missing(band="strong")],
        [], rows, ids, names, canons,
    )
    assert out.apply_today == ()


def test_fix1_required_missing_none_blocks_apply_today():
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [_result_required_missing_none(band="strong")],
        [], rows, ids, names, canons,
    )
    assert out.apply_today == ()


def test_fix1_absent_score_explanation_blocks_worth_a_try():
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [_result_no_explanation(band="good")],
        [], rows, ids, names, canons,
    )
    assert out.worth_a_try == ()


def test_fix1_absent_required_missing_key_blocks_worth_a_try():
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [_result_explanation_no_required_missing(band="stretch")],
        [], rows, ids, names, canons,
    )
    assert out.worth_a_try == ()


def test_fix1_explicit_empty_required_missing_still_admits_apply_today():
    """Sanity: an EXPLICIT empty list is still 'no required gaps' —
    the fail-closed rule only rejects on absence / wrong-type."""
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [_result(band="strong", required_missing=[])],
        [], rows, ids, names, canons,
    )
    assert len(out.apply_today) == 1


# =========================================================================
# Step-8 review Fix 2 — every credential gap must have training
# (mixed-gap cases pinned above; this block covers multi-credential
# and pure-credential variants in detail)
# =========================================================================
def test_fix2_two_credentials_some_with_training_excluded():
    """Two credential gaps; only one has mapped training. The other
    credential remains a blocker — the job is excluded entirely."""
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [_result(job_id="t2", band="good",
                 missing=["Class G driver's license", "WHMIS"],
                 required_missing=["Class G driver's license", "WHMIS"])],
        [], rows, ids, names, canons,
        training_by_job={
            "t2": [{
                "provider": "P", "title": "T",
                "url": "https://example.com/g",
                "for_skill": "Class G driver's license",
            }],   # no WHMIS training
        },
    )
    assert out.worth_a_try == ()


def test_fix2_two_credentials_both_with_training_included():
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [_result(job_id="t2", band="good",
                 missing=["Class G driver's license", "WHMIS"],
                 required_missing=["Class G driver's license", "WHMIS"])],
        [], rows, ids, names, canons,
        training_by_job={
            "t2": [
                {"provider": "P1", "title": "Driver Prep",
                 "url": "https://example.com/g",
                 "for_skill": "Class G driver's license"},
                {"provider": "P2", "title": "WHMIS Course",
                 "url": "https://example.com/whmis",
                 "for_skill": "WHMIS"},
            ],
        },
    )
    assert len(out.worth_a_try) == 1


def test_fix2_strength_claim_credential_path_only_when_training_exists():
    """`stretch_with_training_bridge` requires the job to have passed
    the Fix 2 filter — which means all credentials had training. So
    the credential-path strength claim ONLY fires for admitted jobs
    with a credential gap, never for jobs that snuck through some
    other way."""
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [_result(job_id="w1", band="good",
                 missing=["Class G driver's license"],
                 required_missing=["Class G driver's license"])],
        [], rows, ids, names, canons,
        training_by_job={
            "w1": [{
                "provider": "P", "title": "T",
                "url": "https://example.com/g",
                "for_skill": "Class G driver's license",
            }],
        },
    )
    assert out.worth_a_try[0].strength_claim_text == "stretch_with_training_bridge"


# =========================================================================
# Step-8 review Fix 3 — within-tier dedup by job_id
# =========================================================================
def test_fix3_duplicate_result_rows_dedup_in_apply_today():
    """Two MatchResult rows with the same job_id can come from
    upstream — e.g. a join or a pipeline that emits duplicates.
    Within-tier dedup keeps only the first."""
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [
            _result(job_id="dup", band="strong", required_missing=[]),
            _result(job_id="dup", band="strong", required_missing=[],
                    title="DIFFERENT TITLE"),
        ],
        [], rows, ids, names, canons,
    )
    assert len(out.apply_today) == 1
    assert out.apply_today[0].title == "Job One"   # the first row's title


def test_fix3_duplicate_result_rows_dedup_in_worth_a_try():
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [
            _result(job_id="dup", band="good",
                    missing=["x"], required_missing=["x"]),
            _result(job_id="dup", band="good",
                    missing=["x"], required_missing=["x"],
                    title="DIFFERENT TITLE"),
        ],
        [], rows, ids, names, canons,
    )
    assert len(out.worth_a_try) == 1
    assert out.worth_a_try[0].title == "Job One"


def test_fix3_duplicate_accepted_adjacent_dedup_in_sideways():
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [],
        [
            _accepted_adj(job_id="adj-dup", title="A"),
            _accepted_adj(job_id="adj-dup", title="B"),
            _accepted_adj(job_id="adj-other", title="C"),
        ],
        rows, ids, names, canons,
    )
    assert len(out.sideways_move) == 2
    assert [a.title for a in out.sideways_move] == ["A", "C"]


# =========================================================================
# CP1 review High — invalid training URL must NOT satisfy a credential
# blocker. Non-string `for_skill` must not raise.
# =========================================================================
def test_invalid_url_does_not_satisfy_credential_blocker():
    """A credential gap with only a non-https training URL (e.g. ftp://)
    is NOT covered. The job must NOT be admitted to Worth a Try
    and must NOT be projected with `stretch_with_training_bridge`."""
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [_result(job_id="c1", band="good",
                 missing=["Class G driver's license"],
                 required_missing=["Class G driver's license"])],
        [], rows, ids, names, canons,
        training_by_job={
            "c1": [{
                "provider": "P", "title": "T",
                "url": "ftp://example.com/g",   # not actionable
                "for_skill": "Class G driver's license",
            }],
        },
    )
    assert out.worth_a_try == ()


def test_malformed_url_does_not_satisfy_credential_blocker():
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [_result(job_id="c1", band="good",
                 missing=["Class G driver's license"],
                 required_missing=["Class G driver's license"])],
        [], rows, ids, names, canons,
        training_by_job={
            "c1": [{
                "provider": "P", "title": "T",
                "url": "not a url",   # validator rejects
                "for_skill": "Class G driver's license",
            }],
        },
    )
    assert out.worth_a_try == ()


def test_missing_url_does_not_satisfy_credential_blocker():
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [_result(job_id="c1", band="good",
                 missing=["Class G driver's license"],
                 required_missing=["Class G driver's license"])],
        [], rows, ids, names, canons,
        training_by_job={
            "c1": [{
                "provider": "P", "title": "T",
                # no "url" key at all
                "for_skill": "Class G driver's license",
            }],
        },
    )
    assert out.worth_a_try == ()


def test_one_actionable_alongside_one_invalid_covers_blocker():
    """Two training entries for the same credential — one with an
    actionable URL, one without. The actionable one covers the
    blocker, so the job IS admitted."""
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [_result(job_id="c1", band="good",
                 missing=["Class G driver's license"],
                 required_missing=["Class G driver's license"])],
        [], rows, ids, names, canons,
        training_by_job={
            "c1": [
                {"provider": "P1", "title": "T1",
                 "url": "ftp://bad",   # not actionable
                 "for_skill": "Class G driver's license"},
                {"provider": "P2", "title": "T2",
                 "url": "https://example.com/g",   # actionable
                 "for_skill": "Class G driver's license"},
            ],
        },
    )
    assert len(out.worth_a_try) == 1


def test_non_string_for_skill_is_skipped_without_error():
    """A training entry whose `for_skill` is not a string (forged input
    or upstream corruption) must NOT raise AttributeError; it's silently
    skipped."""
    rows, ids, names, canons = _base_inputs()
    # Should not raise.
    out = build_tiered_evidence(
        [_result(job_id="c1", band="good",
                 missing=["Class G driver's license"],
                 required_missing=["Class G driver's license"])],
        [], rows, ids, names, canons,
        training_by_job={
            "c1": [
                {"provider": "P", "title": "T",
                 "url": "https://example.com/g",
                 "for_skill": 12345},                      # int, not str
                {"provider": "P", "title": "T",
                 "url": "https://example.com/g",
                 "for_skill": ["Class G driver's license"]},  # list
            ],
        },
    )
    # No actionable string for_skill → not admitted.
    assert out.worth_a_try == ()


def test_fix3_dedup_preserves_first_occurrence_order():
    """Across both within-tier and cross-tier dedup, the first
    occurrence wins. Combined with order preservation, that means
    later duplicates simply drop out."""
    rows, ids, names, canons = _base_inputs()
    out = build_tiered_evidence(
        [
            _result(job_id="a", band="strong", required_missing=[]),
            _result(job_id="b", band="strong", required_missing=[]),
            _result(job_id="a", band="strong", required_missing=[]),  # dup
            _result(job_id="c", band="strong", required_missing=[]),
        ],
        [], rows, ids, names, canons,
    )
    assert [m.job_id for m in out.apply_today] == ["a", "b", "c"]
