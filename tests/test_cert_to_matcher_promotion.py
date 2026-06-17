"""Round-29: certification-to-matcher promotion + alias additions.

Pins the contract:

 1. `derive_staged_slots` promotes `facts["certifications"]` into the
    StagedSkill list with `source="resume"` and `confidence=0.9`.
 2. A credential dual-listed in `facts["skills"]` AND `facts["certifications"]`
    appears ONCE in the StagedSkill list (dedupe by canonical alias).
 3. `skills_text` (the user-visible summary) is UNAFFECTED -- it
    remains a readable list of skills only; credentials stay in
    structured `resume_facts_json["certifications"]`.
 4. The matcher alias map (skillbridge/match/aliases.py) folds 310S
    and G2/G variants to canonical forms so the matcher's
    `canonicalize_skill` lookup matches job-skill rows.
 5. End-to-end on the perfect_310s resume: after promotion + alias
    additions, a user who has Class G + 310S in their resume can
    match a 310S Honda job WITHOUT triggering
    `band_capped_by_credential`.
"""
from __future__ import annotations

import os

os.environ.setdefault("LLM_ENABLED", "false")

import pytest

from skillbridge.match.aliases import canonicalize_skill
from skillbridge.resume.derive import derive_staged_slots

pytestmark = pytest.mark.nodb


def _facts(*, skills=None, certifications=None, work_history=None,
           education=None, projects=None, languages=None):
    """Minimal resume_facts_json fixture."""
    return {
        "version": 1,
        "extracted_at": "2026-06-09T00:00:00Z",
        "extractor_version": "test",
        "skills": skills or [],
        "work_history": work_history or [],
        "education": education or [],
        "certifications": certifications or [],
        "projects": projects or [],
        "languages": languages or [],
        "summary_signals": {},
    }


# ============================================================================
# 1. Promotion: certifications become StagedSkill entries
# ============================================================================
def test_certification_appears_in_staged_skills_list():
    facts = _facts(
        certifications=[
            {"fact_id": "f_cert_001",
             "name": "Class G driver's license",
             "issuer": "MTO",
             "evidence": "Valid Ontario Class G driver's license",
             "source": "resume"},
        ],
    )
    out = derive_staged_slots(facts)
    names = [s["skill_name"] for s in out["skills"]]
    assert "Class G driver's license" in names


def test_promoted_certification_has_fixed_confidence_09():
    facts = _facts(
        certifications=[
            {"name": "WHMIS 2015", "evidence": "WHMIS 2015"},
        ],
    )
    out = derive_staged_slots(facts)
    assert len(out["skills"]) == 1
    assert out["skills"][0]["confidence"] == 0.9


def test_promoted_certification_uses_resume_source_tag():
    """`source="resume_cert"` would fall through to "chat" at the
    handler's persistence boundary (chat/handler.py:3137 -- only
    {"chat","resume","form","manual_update"} are accepted). The
    promotion MUST use the existing "resume" tag so persistence
    classifies it correctly."""
    facts = _facts(
        certifications=[
            {"name": "Class G driver's license",
             "evidence": "Class G driver's license"},
        ],
    )
    out = derive_staged_slots(facts)
    assert all(s["source"] == "resume" for s in out["skills"])


def test_promoted_certification_carries_evidence_as_raw_phrase():
    facts = _facts(
        certifications=[
            {"name": "Class G driver's license",
             "evidence": "Valid Ontario Class G driver's license with a clean record"},
        ],
    )
    out = derive_staged_slots(facts)
    assert out["skills"][0]["raw_phrase"] == \
        "Valid Ontario Class G driver's license with a clean record"


def test_certification_without_name_is_skipped():
    """Defensive: a certification entry missing `name` cannot be
    matched against. Skip rather than emit a nameless StagedSkill."""
    facts = _facts(
        certifications=[
            {"evidence": "some evidence", "source": "resume"},   # no name
        ],
    )
    out = derive_staged_slots(facts)
    assert out["skills"] == []


# ============================================================================
# 2. Dedupe: credential dual-listed in skills + certifications appears once
# ============================================================================
def test_dedupe_credential_listed_in_both_skills_and_certifications():
    """If the LLM put Class G into BOTH lists (which a strict reading
    of the prompt would discourage but is observed), the StagedSkill
    list emits ONE entry -- the skills-pass entry wins (its confidence
    is whatever the LLM emitted; the cert promotion would otherwise
    overwrite with a hard 0.9)."""
    facts = _facts(
        skills=[
            {"name": "Class G driver's license",
             "evidence": "Class G",
             "confidence": 0.85},
        ],
        certifications=[
            {"name": "Class G driver's license",
             "evidence": "Valid Ontario Class G driver's license"},
        ],
    )
    out = derive_staged_slots(facts)
    names = [s["skill_name"] for s in out["skills"]]
    assert names.count("Class G driver's license") == 1
    # Skills-pass entry survived; its confidence (0.85) was preserved.
    assert out["skills"][0]["confidence"] == 0.85


def test_dedupe_handles_alias_variants_of_same_credential():
    """The skills entry says "G license" and the certification says
    "Class G driver's license". Both canonicalize to the same alias
    key (`class g license`) -- they MUST dedupe."""
    facts = _facts(
        skills=[
            {"name": "G license", "evidence": "G license", "confidence": 0.85},
        ],
        certifications=[
            {"name": "Class G driver's license",
             "evidence": "Class G driver's license"},
        ],
    )
    out = derive_staged_slots(facts)
    # Both fold to "class g license" canonical -> only the skill survives
    assert len(out["skills"]) == 1
    assert out["skills"][0]["skill_name"] == "G license"


def test_two_distinct_credentials_both_promoted():
    """Two certifications with different canonicals BOTH emit
    StagedSkill entries -- dedupe only collapses same-canonical ones."""
    facts = _facts(
        certifications=[
            {"name": "Class G driver's license",
             "evidence": "Class G"},
            {"name": "WHMIS 2015", "evidence": "WHMIS 2015"},
        ],
    )
    out = derive_staged_slots(facts)
    names = [s["skill_name"] for s in out["skills"]]
    assert "Class G driver's license" in names
    assert "WHMIS 2015" in names


# ============================================================================
# 3. skills_text excludes certifications (round-29 contract)
# ============================================================================
def test_skills_text_does_not_include_certifications():
    """`skills_text` is the user-visible "your skills" summary --
    credentials stay in structured `resume_facts_json["certifications"]`
    for the matcher's StagedSkill consumption but should not appear in
    the readable summary."""
    facts = _facts(
        skills=[
            {"name": "automotive diagnostics",
             "evidence": "automotive diagnostics",
             "confidence": 0.9},
        ],
        certifications=[
            {"name": "Class G driver's license",
             "evidence": "Class G driver's license"},
        ],
    )
    out = derive_staged_slots(facts)
    assert out["skills_text"] is not None
    assert "automotive diagnostics" in out["skills_text"]
    # Credential NOT in user-facing skills summary
    assert "Class G" not in out["skills_text"]
    assert "license" not in out["skills_text"]


# ============================================================================
# 4. Matcher alias coverage: 310S + G2/G variants fold correctly
# ============================================================================
@pytest.mark.parametrize("variant,expected", [
    # Class G -- explicit "Class G" wording folds to "class g license"
    ("Class G driver's license", "class g license"),
    ("Valid Ontario Class G driver's license", "class g license"),
    ("Class G licence", "class g license"),
    ("Class G", "class g license"),
    ("G license", "class g license"),
    # 310S family -- all fold to "310s automotive technician"
    ("310S", "310s automotive technician"),
    ("310S license", "310s automotive technician"),
    ("310S licence", "310s automotive technician"),
    ("310S certification", "310s automotive technician"),
    ("310S Automotive Service Technician License", "310s automotive technician"),
    ("Valid Ontario 310S Automotive Service Technician License",
     "310s automotive technician"),
])
def test_alias_map_folds_credential_variants(variant, expected):
    """The matcher alias map MUST fold every observed credential
    phrasing to a single canonical so user-side and job-side terms
    match without depending on the fuzzy fallback."""
    assert canonicalize_skill(variant) == expected


# ============================================================================
# 4b. G2 must NOT canonicalize to Class G (round-30 reviewer correction)
# ============================================================================
# G2 is Ontario's intermediate graduated licence; Class G is the full
# licence. They are DIFFERENT levels. Folding G2 to "class g license"
# would let a G2-only candidate silently pass a job's mandatory Class G
# requirement.
@pytest.mark.parametrize("g2_variant", [
    "G2",
    "G2 license",
    "G2 licence",
    "Ontario G2 license",
    "Ontario G2 licence",
])
def test_g2_alone_does_NOT_canonicalize_to_class_g(g2_variant):
    """A G2-only credential MUST NOT match a Class G requirement."""
    assert canonicalize_skill(g2_variant) != "class g license", (
        f"G2 must not fold to Class G; {g2_variant!r} did. This would "
        f"let a G2-only candidate silently pass a Class G credential "
        f"requirement."
    )


def test_g2_g_phrase_has_its_own_canonical_not_class_g():
    """"G2/G driver's license" is ambiguous between "graduated through
    G2 to G" and "currently G2 working toward G". The conservative
    default: it gets its OWN canonical ("g2 g license") and does NOT
    silently claim Class G. A candidate who actually has Class G lists
    it separately."""
    for variant in [
        "G2/G driver's license",
        "G2/G",
        "Valid G2/G driver's license",
    ]:
        canon = canonicalize_skill(variant)
        assert canon != "class g license", (
            f"G2/G must not silently fold to Class G; {variant!r} did."
        )
        # And it should be a stable canonical (any G2/G variant folds
        # to the same thing).
        assert canon == "g2 g license", (
            f"G2/G should have a single shared canonical; "
            f"{variant!r} -> {canon!r}"
        )


# ============================================================================
# 5. End-to-end regression on the meeting_01 perfect resume:
#    promotion + aliases together must produce a StagedSkill list that
#    canonically covers 310S and Class G.
# ============================================================================
def test_perfect_310s_resume_promotes_credentials_into_matchable_skills():
    """End-to-end: a payload shaped like a competent LLM extraction
    of the perfect_310s resume produces a StagedSkill list whose
    canonicalized forms include `310s automotive technician` AND
    `class g license`. These are the two credential canonicals the
    Honda 310S job's required-credential rows expect. Without round-29
    the StagedSkill list would have NEITHER (certs were never
    promoted)."""
    facts = _facts(
        certifications=[
            {"name": "310S Automotive Service Technician License",
             "evidence": "Valid Ontario 310S Automotive Service Technician License"},
            {"name": "Class G driver's license",
             "evidence": "Valid Ontario Class G driver's license"},
            {"name": "G2/G driver's license",
             "evidence": "Valid G2/G driver's license"},
            {"name": "WHMIS 2015", "evidence": "WHMIS 2015"},
            {"name": "Standard First Aid and CPR",
             "evidence": "Standard First Aid and CPR"},
        ],
    )
    out = derive_staged_slots(facts)
    canonicals = {canonicalize_skill(s["skill_name"]) for s in out["skills"]}
    # The two load-bearing matches for the 310S Honda job
    assert "310s automotive technician" in canonicals, (
        f"310S credential did not canonicalize; got {canonicals}"
    )
    assert "class g license" in canonicals, (
        f"Class G credential did not canonicalize; got {canonicals}"
    )
    # Bonus: other certs come through with their canonicals
    assert "whmis" in canonicals
    assert "first aid" in canonicals


def test_perfect_310s_resume_keeps_g2_g_and_class_g_as_distinct_entries():
    """Round-30 correction: G2/G and Class G have DISTINCT canonicals
    (`g2 g license` vs `class g license`). The promoted StagedSkill
    list keeps both entries -- the matcher will count Class G as the
    actual licence-level match and treat G2/G as orthogonal context.
    A regression that re-folded them would silently equate the two
    licence levels."""
    facts = _facts(
        certifications=[
            {"name": "Class G driver's license",
             "evidence": "Valid Ontario Class G driver's license"},
            {"name": "G2/G driver's license",
             "evidence": "Valid G2/G driver's license"},
        ],
    )
    out = derive_staged_slots(facts)
    canonicals = [canonicalize_skill(s["skill_name"]) for s in out["skills"]]
    assert "class g license" in canonicals
    assert "g2 g license" in canonicals
    assert canonicals.count("class g license") == 1
    assert canonicals.count("g2 g license") == 1


# ============================================================================
# 6. Existing behaviors unchanged
# ============================================================================
def test_skills_only_facts_emit_skills_unchanged():
    """No certifications -> no behavior change. The skills pass works
    exactly as before round-29."""
    facts = _facts(
        skills=[
            {"name": "automotive diagnostics",
             "evidence": "automotive diagnostics",
             "confidence": 0.9},
            {"name": "preventive maintenance",
             "evidence": "preventive maintenance",
             "confidence": 0.8},
        ],
    )
    out = derive_staged_slots(facts)
    assert len(out["skills"]) == 2
    assert out["skills"][0]["skill_name"] == "automotive diagnostics"
    assert out["skills"][0]["confidence"] == 0.9
    assert out["skills"][0]["source"] == "resume"


def test_below_floor_skills_still_dropped():
    """`MIN_EXTRACTION_CONFIDENCE` floor preserved for skills."""
    facts = _facts(
        skills=[
            {"name": "guessed skill",
             "evidence": "guessed",
             "confidence": 0.4},   # below floor
            {"name": "real skill",
             "evidence": "real",
             "confidence": 0.9},
        ],
    )
    out = derive_staged_slots(facts)
    names = [s["skill_name"] for s in out["skills"]]
    assert "real skill" in names
    assert "guessed skill" not in names


# ============================================================================
# 7. Matcher-level regression: round-30 reviewer's required pin
# ============================================================================
# The end-to-end derive test only proves canonicalization; it does NOT
# exercise the matcher itself. This test wires:
#   1. derive_staged_slots(perfect_310s facts)
#   2. -> StagedSkill list
#   3. -> _score_one_job against a Honda 310S job whose required skills
#         include the credential rows
#   4. Assertions: 310S + Class G matched; neither in missing; no
#      band_capped_by_credential.
# ============================================================================
def _build_user_skill_sets_from_derived(derived_skills):
    """Mirror what `compute_matches_in_memory` does at engine.py:1439."""
    user_skill_ids: set[str] = set()    # no DB-resolved skill_id in tests
    user_skill_names = {(s.get("skill_name") or "").lower() for s in derived_skills}
    user_skill_names_canon = {canonicalize_skill(n) for n in user_skill_names}
    return user_skill_ids, user_skill_names, user_skill_names_canon


def _honda_310s_job_and_skills():
    """A Honda 310S Licensed Automotive Technician job with required-
    skill rows shaped like the engine's `_filter_eligible_skills`
    expects. Confidence above MIN_EXTRACTION_CONFIDENCE so they survive
    the eligibility filter; importance_rank low so they're top-N."""
    job = {
        "job_id": "honda-1",
        "title": "310S Licensed Automotive Technician",
        "employer": "Great Lakes Honda",
        "url": None,
        "location": "Sault Ste. Marie, ON",
        "region_code": "SSM",
        "noc_code": "72410",            # auto technician
        "employment_type": "full-time",
        "posted_date": None,
    }
    job_skills = [
        # The two load-bearing credentials. job_skill rows from the
        # engine's extraction can phrase credentials many ways; use
        # forms aliases.py knows how to fold.
        {"skill_name": "310S automotive technician license",
         "importance_rank": 1, "confidence": 0.95, "required": True},
        {"skill_name": "Class G driver's license",
         "importance_rank": 2, "confidence": 0.95, "required": True},
        # Two general skills the perfect resume claims explicitly so
        # the eligible-job branch fires (>=3 required).
        {"skill_name": "automotive diagnostics",
         "importance_rank": 3, "confidence": 0.9, "required": True},
        {"skill_name": "preventive maintenance",
         "importance_rank": 4, "confidence": 0.8, "required": True},
    ]
    return job, job_skills


def _perfect_310s_resume_facts():
    """The 310S/Honda-perfect resume in the facts-JSON shape the
    extractor produces. Mirrors the LICENSES + CORE SKILLS sections
    from `docs/test-resumes/meeting_01_310s_automotive_perfect.txt`."""
    return _facts(
        skills=[
            {"name": "automotive diagnostics",
             "evidence": "automotive diagnostics",
             "confidence": 0.9},
            {"name": "preventive maintenance",
             "evidence": "preventive maintenance",
             "confidence": 0.9},
            {"name": "brake service",
             "evidence": "brake service",
             "confidence": 0.9},
            {"name": "Honda vehicle experience",
             "evidence": "Honda vehicle experience",
             "confidence": 0.9},
        ],
        certifications=[
            {"name": "310S Automotive Service Technician License",
             "evidence": "Valid Ontario 310S Automotive Service Technician License"},
            {"name": "Class G driver's license",
             "evidence": "Valid Ontario Class G driver's license"},
            {"name": "G2/G driver's license",
             "evidence": "Valid G2/G driver's license"},
            {"name": "WHMIS 2015", "evidence": "WHMIS 2015"},
        ],
        work_history=[
            {"title": "Licensed Automotive Service Technician",
             "employer": "Algoma Motor Service Honda",
             "start_year": 2021, "is_current": True,
             "evidence": "Algoma Motor Service Honda"},
        ],
    )


def test_matcher_scores_310s_honda_job_without_credential_cap(monkeypatch):
    """Round-30 R-2 required pin: after derive + canonicalize, the
    perfect 310S resume matches 310S + Class G on a Honda job AND
    does NOT receive band_capped_by_credential.

    Pre-round-29 the certs never reached staged.skills -> both
    credentials were in missing_skills -> band_capped_by_credential
    fired -> match demoted to stretch. The regression locks the new
    path end to end through `_score_one_job`."""
    from skillbridge.match import engine as engine_mod
    from skillbridge.match.engine import _score_one_job
    # _score_one_job calls _regulated() which hits Postgres. Stub it
    # for this nodb test -- the regulated lookup is for the warning
    # message, not for credential matching.
    monkeypatch.setattr(engine_mod, "_regulated", lambda noc, role: None)

    derived = derive_staged_slots(_perfect_310s_resume_facts())
    user_skill_ids, user_skill_names, user_skill_names_canon = \
        _build_user_skill_sets_from_derived(derived["skills"])
    profile = {
        "profile_id": "test-perfect-310s",
        "target_role_text": "automotive technician",
        "experience_text": derived["experience_text"]
            or "7 years automotive dealership experience",
        "preferred_location": "Sault Ste. Marie",
        "target_noc": "72410",
        "work_type_preference": "full-time",
    }
    job, job_skills = _honda_310s_job_and_skills()
    result = _score_one_job(
        job=job, job_skills=job_skills,
        user_skill_ids=user_skill_ids,
        user_skill_names=user_skill_names,
        user_skill_names_canon=user_skill_names_canon,
        profile=profile,
    )
    assert result is not None
    matched_lower = {s.lower() for s in (result.matched_skills or [])}
    missing_lower = {s.lower() for s in (result.missing_skills or [])}

    # ---- The two load-bearing credentials are matched ----
    assert "310s automotive technician license" in matched_lower, (
        f"310S credential not matched after promotion. matched={matched_lower!r} "
        f"missing={missing_lower!r}"
    )
    assert "class g driver's license" in matched_lower, (
        f"Class G credential not matched after promotion. matched={matched_lower!r} "
        f"missing={missing_lower!r}"
    )

    # ---- Neither credential is in missing ----
    assert "310s automotive technician license" not in missing_lower
    assert "class g driver's license" not in missing_lower

    # ---- band_capped_by_credential MUST be absent ----
    expl = result.score_explanation or {}
    caps = expl.get("caps_applied") or []
    assert "band_capped_by_credential" not in caps, (
        f"Credential cap fired despite user having 310S + Class G. "
        f"caps_applied={caps!r} credential_gap_skills="
        f"{expl.get('credential_gap_skills')!r}"
    )
    # And `credential_gap_skills`, when present, MUST NOT contain
    # either credential.
    cred_gaps = [s.lower() for s in (expl.get("credential_gap_skills") or [])]
    assert "310s automotive technician license" not in cred_gaps
    assert "class g driver's license" not in cred_gaps


def test_g2_only_user_does_not_equate_to_class_g_via_alias_map():
    """Round-30 alias-layer defense: the alias map MUST NOT fold G2-
    only or G2/G phrasings to `class g license`. This is the layer
    the reviewer specifically asked be corrected.

    Note: the broader matcher has a lexical token-overlap fallback in
    `_skill_match_strength` (engine.py:359) that can still register a
    fuzzy match between "G2/G driver's license" and "class g driver's
    license" via shared tokens {g, drivers, license}. That's pre-
    existing matcher behavior outside this slice's scope. The alias
    layer no longer SILENTLY equates the two licence levels, which is
    the contract the reviewer asked to lock."""
    from skillbridge.match.aliases import canonicalize_skill
    # G2-alone canonical is NOT class g license
    for variant in ("G2", "G2 license", "G2 licence"):
        assert canonicalize_skill(variant) != "class g license"
    # G2/G phrase has its OWN canonical
    assert canonicalize_skill("G2/G driver's license") == "g2 g license"
    assert canonicalize_skill("G2/G driver's license") != "class g license"


def test_matcher_still_caps_when_user_has_nothing_close_to_class_g(monkeypatch):
    """Cap-fires regression: a user with NEITHER 310S nor any G-class
    licence MUST still receive band_capped_by_credential on the Honda
    310S job. Pins that round-29 promotion didn't accidentally bypass
    the cap."""
    from skillbridge.match import engine as engine_mod
    from skillbridge.match.engine import _score_one_job
    monkeypatch.setattr(engine_mod, "_regulated", lambda noc, role: None)
    # User has NO credentials at all -- just two general skills. The
    # Honda job requires 310S AND Class G; both must remain in missing.
    facts = _facts(
        skills=[
            {"name": "general handywork",
             "evidence": "general handywork", "confidence": 0.9},
            {"name": "punctuality",
             "evidence": "punctuality", "confidence": 0.9},
        ],
        certifications=[],
        work_history=[
            {"title": "Helper", "employer": "Shop",
             "start_year": 2021, "is_current": True,
             "evidence": "Helper"},
        ],
    )
    derived = derive_staged_slots(facts)
    user_skill_ids, user_skill_names, user_skill_names_canon = \
        _build_user_skill_sets_from_derived(derived["skills"])
    profile = {
        "profile_id": "test-no-creds",
        "target_role_text": "automotive technician",
        "experience_text": "helper at a small shop",
        "preferred_location": "Sault Ste. Marie",
        "target_noc": "72410",
        "work_type_preference": "full-time",
    }
    job, job_skills = _honda_310s_job_and_skills()
    result = _score_one_job(
        job=job, job_skills=job_skills,
        user_skill_ids=user_skill_ids,
        user_skill_names=user_skill_names,
        user_skill_names_canon=user_skill_names_canon,
        profile=profile,
    )
    assert result is not None
    missing_lower = {s.lower() for s in (result.missing_skills or [])}
    expl = result.score_explanation or {}
    # The Class G requirement IS missing because G2/G has its own
    # canonical and does NOT fold to class g license.
    assert "class g driver's license" in missing_lower, (
        f"G2-only user must NOT silently pass a Class G requirement. "
        f"missing={missing_lower!r}"
    )
    caps = expl.get("caps_applied") or []
    assert "band_capped_by_credential" in caps, (
        f"Credential cap must fire when Class G is missing. "
        f"caps_applied={caps!r}"
    )


# ============================================================================
# 8. Round-31: credential-strict matching path
# ============================================================================
# After cert promotion (round-29), the matcher's existing fuzzy / token-
# overlap / semantic fallbacks become credential-safety bugs: G2/G
# would token-match Class G via {g, drivers, license}, "drivers
# license" generic would match any class licence requirement, etc.
# Round-31 gates credential-class job skills behind the first three
# rungs only (exact skill_id / exact name / canonical alias).
# ============================================================================
def _score_one_job_with_user_names(monkeypatch, user_names: set[str],
                                    job_skills: list[dict]):
    """Helper: score the Honda 310S job against a hand-crafted user
    skill set. Bypasses the cert-promotion / derive layer so the test
    isolates `_skill_match_strength` behavior."""
    from skillbridge.match import engine as engine_mod
    from skillbridge.match.engine import _score_one_job
    monkeypatch.setattr(engine_mod, "_regulated", lambda noc, role: None)
    job, _ = _honda_310s_job_and_skills()
    profile = {
        "profile_id": "credential-strict-test",
        "target_role_text": "automotive technician",
        "experience_text": "shop experience",
        "preferred_location": "Sault Ste. Marie",
        "target_noc": "72410",
        "work_type_preference": "full-time",
    }
    user_skill_names = {n.lower() for n in user_names}
    user_skill_names_canon = {canonicalize_skill(n) for n in user_skill_names}
    return _score_one_job(
        job=job, job_skills=job_skills,
        user_skill_ids=set(),
        user_skill_names=user_skill_names,
        user_skill_names_canon=user_skill_names_canon,
        profile=profile,
    )


def _class_g_only_job_skills():
    """Minimal eligible job: Class G + 3 ordinary skills so the
    `min_required_skills_for_eligibility=3` threshold passes."""
    return [
        {"skill_name": "Class G driver's license",
         "importance_rank": 1, "confidence": 0.95, "required": True},
        {"skill_name": "preventive maintenance",
         "importance_rank": 2, "confidence": 0.9, "required": True},
        {"skill_name": "brake service",
         "importance_rank": 3, "confidence": 0.9, "required": True},
        {"skill_name": "automotive diagnostics",
         "importance_rank": 4, "confidence": 0.9, "required": True},
    ]


# ----- G2 only -> MUST NOT satisfy Class G; cap MUST fire -----
@pytest.mark.parametrize("g2_phrasing", [
    "G2",
    "G2 license",
    "G2 licence",
    "Ontario G2 license",
])
def test_g2_only_does_not_satisfy_class_g_requirement(monkeypatch, g2_phrasing):
    """A user whose only driver credential is G2 MUST NOT match a
    Class G requirement on the job. The credential cap MUST fire."""
    result = _score_one_job_with_user_names(
        monkeypatch,
        user_names={
            g2_phrasing,
            "preventive maintenance", "brake service", "automotive diagnostics",
        },
        job_skills=_class_g_only_job_skills(),
    )
    assert result is not None
    missing_lower = {s.lower() for s in (result.missing_skills or [])}
    matched_lower = {s.lower() for s in (result.matched_skills or [])}
    assert "class g driver's license" in missing_lower, (
        f"G2-only user falsely satisfied Class G via {g2_phrasing!r}; "
        f"matched={matched_lower!r} missing={missing_lower!r}"
    )
    caps = (result.score_explanation or {}).get("caps_applied") or []
    assert "band_capped_by_credential" in caps, (
        f"Credential cap must fire for G2-only against Class G "
        f"requirement; caps_applied={caps!r}"
    )


@pytest.mark.parametrize("g2_g_phrasing", [
    "G2/G",
    "G2/G driver's license",
    "Valid G2/G driver's license",
])
def test_g2_g_alone_does_not_satisfy_class_g(monkeypatch, g2_g_phrasing):
    """G2/G phrasing alone (without explicit Class G elsewhere on the
    candidate) MUST NOT satisfy a Class G requirement. The lexical
    token-overlap that previously let "G2/G driver's license" match
    "Class G driver's license" via shared tokens is now blocked by
    the credential-strict gate."""
    result = _score_one_job_with_user_names(
        monkeypatch,
        user_names={
            g2_g_phrasing,
            "preventive maintenance", "brake service", "automotive diagnostics",
        },
        job_skills=_class_g_only_job_skills(),
    )
    assert result is not None
    missing_lower = {s.lower() for s in (result.missing_skills or [])}
    assert "class g driver's license" in missing_lower, (
        f"G2/G-only user falsely satisfied Class G via {g2_g_phrasing!r}; "
        f"missing={missing_lower!r}"
    )
    caps = (result.score_explanation or {}).get("caps_applied") or []
    assert "band_capped_by_credential" in caps


def test_generic_drivers_license_does_not_satisfy_class_g(monkeypatch):
    """A generic "driver's license" (no class specified) MUST NOT
    satisfy Class G. Same protection mechanism; different failure
    mode."""
    result = _score_one_job_with_user_names(
        monkeypatch,
        user_names={
            "driver's license",
            "preventive maintenance", "brake service", "automotive diagnostics",
        },
        job_skills=_class_g_only_job_skills(),
    )
    assert result is not None
    missing_lower = {s.lower() for s in (result.missing_skills or [])}
    assert "class g driver's license" in missing_lower, (
        f"Generic driver's license falsely satisfied Class G; "
        f"missing={missing_lower!r}"
    )
    caps = (result.score_explanation or {}).get("caps_applied") or []
    assert "band_capped_by_credential" in caps


# ----- Real Class G + 310S aliases MUST still match -----
@pytest.mark.parametrize("class_g_alias", [
    "Class G driver's license",
    "Class G driver license",   # apostrophe-stripped variant
    "G license",
    "Class G licence",
    "Valid Ontario Class G driver's license",
])
def test_real_class_g_aliases_satisfy_class_g_requirement(monkeypatch, class_g_alias):
    """Legitimate Class G phrasings -- every form the alias map maps
    to `class g license` -- MUST still satisfy the requirement. The
    credential-strict gate stops AFTER the canonical-alias rung, so
    nothing in the alias map gets discarded."""
    result = _score_one_job_with_user_names(
        monkeypatch,
        user_names={
            class_g_alias,
            "preventive maintenance", "brake service", "automotive diagnostics",
        },
        job_skills=_class_g_only_job_skills(),
    )
    assert result is not None
    matched_lower = {s.lower() for s in (result.matched_skills or [])}
    missing_lower = {s.lower() for s in (result.missing_skills or [])}
    assert "class g driver's license" in matched_lower, (
        f"Legitimate Class G alias {class_g_alias!r} failed to satisfy "
        f"the requirement; matched={matched_lower!r} missing={missing_lower!r}"
    )
    caps = (result.score_explanation or {}).get("caps_applied") or []
    assert "band_capped_by_credential" not in caps


@pytest.mark.parametrize("user_310s_alias,job_phrasing", [
    ("310S Automotive Service Technician License",
     "310S automotive technician license"),
    ("310S license",
     "310S automotive technician license"),
    ("310S certification",
     "310S automotive technician certification"),
    ("Valid Ontario 310S Automotive Service Technician License",
     "310S automotive technician license"),
])
def test_real_310s_aliases_satisfy_310s_requirement(
    monkeypatch, user_310s_alias, job_phrasing,
):
    """Legitimate 310S phrasings (varied between user-side and job-side)
    MUST still match via the alias map. The credential-strict gate
    must not discard the alias-resolved equality path."""
    job_skills = [
        {"skill_name": job_phrasing,
         "importance_rank": 1, "confidence": 0.95, "required": True},
        {"skill_name": "preventive maintenance",
         "importance_rank": 2, "confidence": 0.9, "required": True},
        {"skill_name": "brake service",
         "importance_rank": 3, "confidence": 0.9, "required": True},
        {"skill_name": "automotive diagnostics",
         "importance_rank": 4, "confidence": 0.9, "required": True},
    ]
    result = _score_one_job_with_user_names(
        monkeypatch,
        user_names={
            user_310s_alias,
            "preventive maintenance", "brake service", "automotive diagnostics",
        },
        job_skills=job_skills,
    )
    assert result is not None
    matched_lower = {s.lower() for s in (result.matched_skills or [])}
    assert job_phrasing.lower() in matched_lower, (
        f"Legitimate 310S alias {user_310s_alias!r} failed to satisfy "
        f"the job's {job_phrasing!r} requirement; matched={matched_lower!r}"
    )


# ============================================================================
# 8b. Round-32: bare trade codes (310S / 310T / 433A / ...) are credentials
# ============================================================================
# `is_credential_skill_name()` previously checked only keyword
# substrings ("licence", "certification", "class g", ...). A job
# requirement listed as the bare trade code -- "310S" rather than
# "310S license" -- fell through to fuzzy/substring matching, letting
# non-credential prose like "310S automotive experience" silently
# satisfy the requirement. Round-32 adds the `\b\d{3}[A-Za-z]\b`
# regex to the credential classifier.
# ============================================================================
def _trade_code_job_skills(bare_code: str):
    """Honda-style job with the credential listed as the bare trade
    code rather than spelled-out. The minimum-eligibility floor of 3
    required skills is still met."""
    return [
        {"skill_name": bare_code,
         "importance_rank": 1, "confidence": 0.95, "required": True},
        {"skill_name": "preventive maintenance",
         "importance_rank": 2, "confidence": 0.9, "required": True},
        {"skill_name": "brake service",
         "importance_rank": 3, "confidence": 0.9, "required": True},
        {"skill_name": "automotive diagnostics",
         "importance_rank": 4, "confidence": 0.9, "required": True},
    ]


def test_is_credential_skill_name_catches_bare_trade_codes():
    """Direct check of the classifier: bare trade codes must register
    as credentials so the matcher's strict gate applies."""
    from skillbridge.match.engine import is_credential_skill_name
    # Ontario apprenticeship trade codes
    assert is_credential_skill_name("310S") is True
    assert is_credential_skill_name("310T") is True
    assert is_credential_skill_name("433A") is True
    assert is_credential_skill_name("309A") is True
    assert is_credential_skill_name("442A") is True
    assert is_credential_skill_name("313A") is True
    # Lowercase letter forms (matcher passes through lowercased)
    assert is_credential_skill_name("310s") is True
    # Inside a longer string too
    assert is_credential_skill_name("310S automotive technician") is True


def test_is_credential_skill_name_rejects_non_trade_code_numerics():
    """The pattern must NOT false-positive on common version / product
    strings -- otherwise a non-credential skill would be wrongly
    locked into the credential-strict matching path."""
    from skillbridge.match.engine import is_credential_skill_name
    # Version strings: insufficient digits or missing trailing letter
    assert is_credential_skill_name("Python 3") is False
    assert is_credential_skill_name("Windows 10") is False
    assert is_credential_skill_name("Java 11") is False
    # ISO / product codes: 4+ digits, no trailing letter
    assert is_credential_skill_name("ISO 9001") is False
    assert is_credential_skill_name("ISO 14001") is False
    # Trailing letter on too-few digits
    assert is_credential_skill_name("AWS S3") is False
    assert is_credential_skill_name("MS-365") is False
    # Pure words with no digit/letter pairing
    assert is_credential_skill_name("automotive diagnostics") is False
    assert is_credential_skill_name("customer service") is False


def test_bare_310s_job_requirement_rejects_loosely_related_user_prose(monkeypatch):
    """The headline reviewer repro: a job listed as bare `310S` must
    NOT match a user who has "310S automotive experience" (a non-
    credential prose phrase that happens to share the substring).
    Pre-round-32 the credential-strict gate didn't fire for the bare
    code, so the substring path matched -- silently passing a
    mandatory credential requirement."""
    result = _score_one_job_with_user_names(
        monkeypatch,
        user_names={
            "310S automotive experience",          # NOT a credential
            "preventive maintenance", "brake service", "automotive diagnostics",
        },
        job_skills=_trade_code_job_skills("310S"),
    )
    assert result is not None
    matched_lower = {s.lower() for s in (result.matched_skills or [])}
    missing_lower = {s.lower() for s in (result.missing_skills or [])}
    assert "310s" in missing_lower, (
        f"Bare 310S requirement falsely satisfied by 310S-prose; "
        f"matched={matched_lower!r} missing={missing_lower!r}"
    )
    caps = (result.score_explanation or {}).get("caps_applied") or []
    assert "band_capped_by_credential" in caps, (
        f"Credential cap must fire for bare 310S requirement when "
        f"the user has only related prose; caps_applied={caps!r}"
    )


def test_bare_310t_job_requirement_rejects_loosely_related_user_prose(monkeypatch):
    """Same defense for 310T (truck and coach technician)."""
    result = _score_one_job_with_user_names(
        monkeypatch,
        user_names={
            "310T heavy-equipment background",     # NOT a credential
            "preventive maintenance", "brake service", "automotive diagnostics",
        },
        job_skills=_trade_code_job_skills("310T"),
    )
    assert result is not None
    missing_lower = {s.lower() for s in (result.missing_skills or [])}
    assert "310t" in missing_lower
    caps = (result.score_explanation or {}).get("caps_applied") or []
    assert "band_capped_by_credential" in caps


@pytest.mark.parametrize("user_alias", [
    "310S",                              # exact bare code
    "310S license",                      # alias-mapped form
    "310S certification",                # alias-mapped form
    "310S Automotive Service Technician License",
    "Valid Ontario 310S Automotive Service Technician License",
])
def test_explicit_310s_aliases_still_match_bare_310s_requirement(
    monkeypatch, user_alias,
):
    """Round-32 does not regress legitimate matches. Every alias-map
    entry for 310S still satisfies a bare `310S` job requirement via
    the canonical-alias rung (both fold to `310s automotive
    technician`)."""
    result = _score_one_job_with_user_names(
        monkeypatch,
        user_names={
            user_alias,
            "preventive maintenance", "brake service", "automotive diagnostics",
        },
        job_skills=_trade_code_job_skills("310S"),
    )
    assert result is not None
    matched_lower = {s.lower() for s in (result.matched_skills or [])}
    assert "310s" in matched_lower, (
        f"Legitimate 310S alias {user_alias!r} did not satisfy a "
        f"bare 310S requirement; matched={matched_lower!r}"
    )
    caps = (result.score_explanation or {}).get("caps_applied") or []
    assert "band_capped_by_credential" not in caps


def test_ordinary_numeric_skills_still_fuzzy_match(monkeypatch):
    """Round-32 must NOT regress non-credential skills that happen to
    contain digits. "Python 3" / "Windows 10" / "ISO 9001" / "AWS S3"
    are NOT trade codes; they should keep benefiting from the matcher's
    ordinary fuzzy / substring path."""
    job_skills = [
        # Job phrases a non-credential skill with a version number; the
        # user phrases it slightly differently.
        {"skill_name": "Microsoft Excel 2019",
         "importance_rank": 1, "confidence": 0.9, "required": True},
        {"skill_name": "QuickBooks Online",
         "importance_rank": 2, "confidence": 0.9, "required": True},
        {"skill_name": "data entry",
         "importance_rank": 3, "confidence": 0.9, "required": True},
    ]
    result = _score_one_job_with_user_names(
        monkeypatch,
        user_names={
            "Microsoft Excel",       # token-overlap with "Microsoft Excel 2019"
            "QuickBooks",            # token-overlap with "QuickBooks Online"
            "data entry",
        },
        job_skills=job_skills,
    )
    assert result is not None
    matched_lower = {s.lower() for s in (result.matched_skills or [])}
    # All three non-credential job skills should match via the
    # fuzzy / substring paths (none was tagged credential).
    assert "microsoft excel 2019" in matched_lower
    assert "quickbooks online" in matched_lower
    assert "data entry" in matched_lower


# ============================================================================
# 8c. Round-33 -- carve-out covers bare trade codes + 310T aliases
# ============================================================================
def test_filter_eligible_skills_carves_out_low_ranked_bare_trade_code():
    """Round-33 (1): `_filter_eligible_skills` previously used a raw
    keyword check, so a bare trade code like `310S` or `310T` ranked
    below the top-N cutoff fell off the eligible set -- and the
    credential cap downstream had nothing to fire on. The carve-out
    now consults `is_credential_skill_name`, which catches bare
    trade codes via the round-32 regex."""
    from skillbridge.match.engine import _filter_eligible_skills, MATCH
    # Build a job with MORE than top_n_required_skills skills, with
    # the bare trade code ranked LAST so the importance-rank sort
    # pushes it past the cutoff.
    top_n = MATCH.top_n_required_skills
    ordinary = [
        {"skill_name": f"ordinary skill {i}",
         "importance_rank": i + 1, "confidence": 0.9}
        for i in range(top_n + 2)
    ]
    bare_credential = {
        "skill_name": "310T",
        "importance_rank": top_n + 5,    # well below cutoff
        "confidence": 0.9,
    }
    kept = _filter_eligible_skills(ordinary + [bare_credential])
    kept_names = {s["skill_name"] for s in kept}
    assert "310T" in kept_names, (
        f"Low-ranked bare trade code 310T was dropped by the filter; "
        f"the carve-out must catch it via is_credential_skill_name. "
        f"kept_names={kept_names!r}"
    )


def test_filter_eligible_skills_still_carves_out_keyword_credentials():
    """Existing keyword credentials must still survive low-rank
    dropping (no regression on Class G, WHMIS, etc.)."""
    from skillbridge.match.engine import _filter_eligible_skills, MATCH
    top_n = MATCH.top_n_required_skills
    ordinary = [
        {"skill_name": f"ordinary skill {i}",
         "importance_rank": i + 1, "confidence": 0.9}
        for i in range(top_n + 2)
    ]
    class_g = {
        "skill_name": "Class G driver's license",
        "importance_rank": top_n + 5, "confidence": 0.9,
    }
    kept = _filter_eligible_skills(ordinary + [class_g])
    assert any(
        s["skill_name"] == "Class G driver's license" for s in kept
    )


def test_filter_eligible_skills_drops_ordinary_low_ranked_skill():
    """Negative control: a NON-credential low-ranked skill still gets
    dropped. The carve-out is credential-specific, not blanket."""
    from skillbridge.match.engine import _filter_eligible_skills, MATCH
    top_n = MATCH.top_n_required_skills
    ordinary = [
        {"skill_name": f"ordinary skill {i}",
         "importance_rank": i + 1, "confidence": 0.9}
        for i in range(top_n + 2)
    ]
    kept = _filter_eligible_skills(ordinary)
    # Only top_n make it through (the +2 surplus drops)
    assert len(kept) == top_n


# ----- 310T alias coverage -----
@pytest.mark.parametrize("variant", [
    "310T",
    "310T license",
    "310T licence",
    "310T certification",
    "310T cert",
    "310T certificate",
    "310T certificate of qualification",
    "310T COQ",
    "310T truck and coach",
    "310T technician",
    "310T technician license",
    "310T technician certification",
    "Valid Ontario 310T license",
    "Valid 310T certificate of qualification",
])
def test_310t_variants_fold_to_canonical(variant):
    """Round-33 (2): every observed 310T phrasing folds to the same
    canonical so user-side and job-side terms match without depending
    on the (now-disabled-for-credentials) fuzzy fallback."""
    assert canonicalize_skill(variant) == "310t truck and coach technician"


@pytest.mark.parametrize("user_alias,job_phrasing", [
    ("310T", "310T"),
    ("310T license", "310T"),
    ("310T certificate of qualification", "310T"),
    ("310T", "310T certificate of qualification"),
    ("310T license", "310T technician certification"),
    ("Valid Ontario 310T license", "310T"),
])
def test_310t_aliases_match_at_strict_path(monkeypatch, user_alias, job_phrasing):
    """Round-33 (2) end-to-end: any 310T alias on the user side
    satisfies any 310T alias on the job side via the canonical-alias
    rung. The credential-strict gate doesn't block this -- aliases
    are the explicitly-permitted equivalence layer."""
    job_skills = [
        {"skill_name": job_phrasing,
         "importance_rank": 1, "confidence": 0.95, "required": True},
        {"skill_name": "preventive maintenance",
         "importance_rank": 2, "confidence": 0.9, "required": True},
        {"skill_name": "brake service",
         "importance_rank": 3, "confidence": 0.9, "required": True},
        {"skill_name": "automotive diagnostics",
         "importance_rank": 4, "confidence": 0.9, "required": True},
    ]
    result = _score_one_job_with_user_names(
        monkeypatch,
        user_names={
            user_alias,
            "preventive maintenance", "brake service", "automotive diagnostics",
        },
        job_skills=job_skills,
    )
    assert result is not None
    matched_lower = {s.lower() for s in (result.matched_skills or [])}
    assert job_phrasing.lower() in matched_lower, (
        f"User's {user_alias!r} did not satisfy job's {job_phrasing!r}; "
        f"matched={matched_lower!r}"
    )
    caps = (result.score_explanation or {}).get("caps_applied") or []
    assert "band_capped_by_credential" not in caps


def test_310t_truck_experience_prose_does_not_match_bare_310t(monkeypatch):
    """Round-33 (2) negative: non-credential prose that happens to
    contain "310T" (e.g., "310T truck experience" — descriptive, not
    a credential claim) MUST NOT satisfy a bare 310T credential
    requirement. The strict gate blocks all sub-token paths."""
    result = _score_one_job_with_user_names(
        monkeypatch,
        user_names={
            "310T truck experience",     # prose, not a credential
            "preventive maintenance", "brake service", "automotive diagnostics",
        },
        job_skills=_trade_code_job_skills("310T"),
    )
    assert result is not None
    missing_lower = {s.lower() for s in (result.missing_skills or [])}
    assert "310t" in missing_lower, (
        f"310T prose falsely satisfied a bare 310T credential "
        f"requirement; missing={missing_lower!r}"
    )
    caps = (result.score_explanation or {}).get("caps_applied") or []
    assert "band_capped_by_credential" in caps


# ----- Non-credential skills can STILL fuzzy-match (round-31 not regressing) -----
def test_non_credential_skills_still_fuzzy_match(monkeypatch):
    """Round-31 only tightens the CREDENTIAL path. Ordinary skills
    must still benefit from token-overlap and word-bounded substring
    matching as before -- otherwise we'd silently regress legitimate
    skill matches for ordinary roles."""
    job_skills = [
        # Non-credential skill phrased differently between user / job;
        # the token-overlap path matches them.
        {"skill_name": "automotive systems diagnostics",
         "importance_rank": 1, "confidence": 0.9, "required": True},
        {"skill_name": "preventive maintenance",
         "importance_rank": 2, "confidence": 0.9, "required": True},
        {"skill_name": "brake service",
         "importance_rank": 3, "confidence": 0.9, "required": True},
    ]
    result = _score_one_job_with_user_names(
        monkeypatch,
        user_names={
            "automotive diagnostics",         # shares tokens with the job-side phrase
            "preventive maintenance",
            "brake service",
        },
        job_skills=job_skills,
    )
    assert result is not None
    matched_lower = {s.lower() for s in (result.matched_skills or [])}
    # The fuzzy path matches "automotive diagnostics" (user) against
    # "automotive systems diagnostics" (job) -- still legitimate for
    # non-credential skills.
    assert "automotive systems diagnostics" in matched_lower


def test_below_floor_skill_does_not_block_cert_with_same_name():
    """A below-floor SKILL entry should not pre-populate the dedupe
    set; the same name as a certification should still get promoted
    (the cert's 0.9 floor is the definite-claim contract)."""
    facts = _facts(
        skills=[
            {"name": "Class G driver's license",
             "evidence": "Class G",
             "confidence": 0.4},        # below floor -- skipped
        ],
        certifications=[
            {"name": "Class G driver's license",
             "evidence": "Class G driver's license"},
        ],
    )
    out = derive_staged_slots(facts)
    names = [s["skill_name"] for s in out["skills"]]
    assert "Class G driver's license" in names
    # The promoted entry's confidence is the cert's 0.9, not the
    # skill's dropped 0.4.
    assert out["skills"][0]["confidence"] == 0.9
