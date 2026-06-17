"""AR-1b tests: SSM region predicate + adjacency evidence floor +
soft-offer eligibility predicates.

Covers (per docs/adjacent-recommendations-design.md):
  - is_ssm_region_job: dedicated SSM-proper aliases + verified region
    codes; LOCAL_CITIES is NOT consulted; Algoma communities rejected.
  - has_usable_skill_evidence: three non-credential resume/chat skills
    @ ≥ 0.6 confidence; resume-less paths supported.
  - is_credential_only_band_cap: reads score_explanation.caps_applied
    AND score_components.score_pre_caps (NOT result-level caps);
    pre-cap band via engine._band.
  - should_emit_soft_offer_on_matches / on_no_match: composite gate.

All AR-1b modules are dead code; no production caller dispatches into
them until AR-6 activates the wiring. These tests exercise the pure
functions directly.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.nodb


from skillbridge.match.adjacent import (
    ADJACENT_MIN_USER_SKILLS,
    _CRED_CAP_FLAG,
    _is_valid_normalized_score,
    _result_caps,
    has_usable_skill_evidence,
    is_credential_only_band_cap,
    should_emit_soft_offer_on_matches,
    should_emit_soft_offer_on_no_match,
)
from skillbridge.match.engine import MATCH
from skillbridge.match.region import (
    _SSM_PROPER_LOCATION_ALIASES,
    _SSM_PROPER_REGION_CODES,
    is_ssm_region_job,
)
from skillbridge.session.staging import StagedProfile, StagedSkill


# =========================================================================
# is_ssm_region_job
# =========================================================================
def test_region_code_3557011_accepted() -> None:
    """Production fixtures (test_match_strength.py:160, test_hard_gates.py:44,
    etc.) all use the StatsCan CSD code 3557011."""
    assert is_ssm_region_job({"region_code": "3557011"}) is True


def test_region_code_ssm_case_insensitive() -> None:
    """Legacy fixture (test_cert_to_matcher_promotion.py:416) uses
    literal "SSM"; both cases accepted."""
    assert is_ssm_region_job({"region_code": "SSM"}) is True
    assert is_ssm_region_job({"region_code": "ssm"}) is True
    assert is_ssm_region_job({"region_code": "  SsM  "}) is True


def test_region_code_algoma_rejected() -> None:
    """The locked product scope is SSM proper only -- Algoma
    communities never enter the candidate pool even though
    LOCAL_CITIES (.env:64) admits them for the local-boost radius."""
    assert is_ssm_region_job({"region_code": "algoma"}) is False
    assert is_ssm_region_job({"region_code": "wawa"}) is False
    assert is_ssm_region_job({"region_code": "blind_river"}) is False


def test_region_code_toronto_rejected() -> None:
    assert is_ssm_region_job({"region_code": "toronto"}) is False
    assert is_ssm_region_job({"region_code": "3520005"}) is False  # Toronto CSD


def test_region_code_present_takes_precedence_over_location() -> None:
    """Explicit region_code wins even if the location string would
    otherwise admit the job. Prevents a Toronto-coded record from
    leaking in via a city name in the address."""
    assert is_ssm_region_job({
        "region_code": "toronto",
        "location": "Sault Ste. Marie, ON",
    }) is False


def test_missing_region_code_falls_back_to_location() -> None:
    """When region_code is absent, the location string is checked
    against the dedicated SSM-proper aliases."""
    assert is_ssm_region_job({"location": "Sault Ste. Marie, ON"}) is True
    assert is_ssm_region_job({"location": "sault ste marie"}) is True
    assert is_ssm_region_job({"location": "SSM"}) is True


def test_missing_region_code_location_algoma_rejected() -> None:
    """A naive LOCAL_CITIES check would admit Wawa here. The dedicated
    alias set MUST NOT include it."""
    assert is_ssm_region_job({"location": "Wawa, ON"}) is False
    assert is_ssm_region_job({"location": "Blind River"}) is False
    assert is_ssm_region_job({"location": "Chapleau, ON"}) is False
    assert is_ssm_region_job({"location": "Algoma District"}) is False


def test_missing_region_code_and_missing_location_rejected() -> None:
    """Conservative: missing both → non-SSM. Prevents a corrupted /
    legacy record from leaking in."""
    assert is_ssm_region_job({}) is False
    assert is_ssm_region_job({"location": ""}) is False
    assert is_ssm_region_job({"region_code": "", "location": ""}) is False
    assert is_ssm_region_job({"region_code": None, "location": None}) is False


def test_non_string_region_or_location_handled() -> None:
    """Defensive: non-str values must not crash."""
    assert is_ssm_region_job({"region_code": 3557011}) is False  # int, not str
    assert is_ssm_region_job({"location": ["Sault Ste. Marie"]}) is False


# ---- AR-1b round-2: SSM substring false-positive ----
def test_ssm_substring_in_unrelated_word_rejected() -> None:
    """`"ssm" in "rossmore"` is True, but Rossmore is not SSM. The
    predicate matches at WORD BOUNDARIES, so substring leakage like
    this is impossible.

    Failure mode this gates: a posting located in Rossmore, Ontario
    (or any word containing the literal "ssm" substring) would have
    been silently accepted by the v1 implementation."""
    assert is_ssm_region_job({"location": "Rossmore, Ontario"}) is False
    assert is_ssm_region_job({"location": "Mossmoor Drive"}) is False
    assert is_ssm_region_job({"location": "Cossmington"}) is False


def test_ssm_as_standalone_token_still_accepted() -> None:
    """Word-boundary matching still accepts SSM as a real city token."""
    assert is_ssm_region_job({"location": "SSM"}) is True
    assert is_ssm_region_job({"location": "SSM, ON"}) is True
    assert is_ssm_region_job({"location": "SSM Plaza, 123 Main St"}) is True
    assert is_ssm_region_job({"location": "Some place near SSM"}) is True


def test_sault_ste_marie_alias_still_accepted_with_word_boundary() -> None:
    """Sanity: multi-word aliases keep working under the word-boundary
    rule. (`\\b` between "marie" and "," / end-of-string is satisfied.)"""
    assert is_ssm_region_job({"location": "Sault Ste. Marie, ON"}) is True
    assert is_ssm_region_job({"location": "sault ste marie"}) is True
    assert is_ssm_region_job({"location": "downtown Sault Ste. Marie area"}) is True


def test_local_cities_is_not_consulted() -> None:
    """Audit: ensure the SSM predicate doesn't IMPORT config.LOCAL_CITIES
    anywhere. Walks the AST so that docstring mentions (which explain
    *why* the dependency is rejected) don't trip the check. Catches
    `from config import LOCAL_CITIES`, `from skillbridge... import
    LOCAL_CITIES`, `config.LOCAL_CITIES`."""
    import ast
    import inspect

    from skillbridge.match import region

    tree = ast.parse(inspect.getsource(region))

    forbidden_imports = []
    forbidden_attrs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "LOCAL_CITIES":
                    forbidden_imports.append(node.module)
        elif isinstance(node, ast.Attribute):
            if node.attr == "LOCAL_CITIES":
                forbidden_attrs.append(ast.dump(node))
        elif isinstance(node, ast.Name):
            if node.id == "LOCAL_CITIES":
                forbidden_attrs.append(node.id)

    assert not forbidden_imports, (
        f"match/region.py imported LOCAL_CITIES: {forbidden_imports}. "
        f"Use the dedicated _SSM_PROPER_LOCATION_ALIASES set instead."
    )
    assert not forbidden_attrs, (
        f"match/region.py references LOCAL_CITIES at runtime: "
        f"{forbidden_attrs}. Use the dedicated alias set instead."
    )


def test_alias_set_does_not_contain_algoma() -> None:
    """Hard-coded scope check: the alias set never includes Algoma
    communities."""
    forbidden = {"wawa", "blind river", "chapleau", "elliot lake", "algoma"}
    leaked = forbidden & _SSM_PROPER_LOCATION_ALIASES
    assert not leaked, f"SSM alias set leaked Algoma communities: {leaked}"


def test_region_code_set_only_contains_verified_codes() -> None:
    """The region-code set is the exact pair {3557011, ssm}."""
    assert _SSM_PROPER_REGION_CODES == frozenset({"3557011", "ssm"})


# =========================================================================
# has_usable_skill_evidence
# =========================================================================
def _staged_with_skills(skills: list[StagedSkill]) -> StagedProfile:
    s = StagedProfile.new("sess-1")
    s.skills = skills
    return s


def test_evidence_three_resume_skills_at_floor_passes() -> None:
    skills = [
        StagedSkill(skill_name="welding", source="resume", confidence=0.7),
        StagedSkill(skill_name="blueprint reading", source="resume", confidence=0.7),
        StagedSkill(skill_name="forklift operation", source="resume", confidence=0.7),
    ]
    assert has_usable_skill_evidence(_staged_with_skills(skills)) is True


def test_evidence_resume_less_three_chat_skills_passes() -> None:
    """Resume presence is NOT required."""
    skills = [
        StagedSkill(skill_name="welding", source="chat", confidence=0.7),
        StagedSkill(skill_name="customer service", source="chat", confidence=0.7),
        StagedSkill(skill_name="cash handling", source="chat", confidence=0.7),
    ]
    assert has_usable_skill_evidence(_staged_with_skills(skills)) is True


def test_evidence_mixed_resume_and_chat_passes() -> None:
    skills = [
        StagedSkill(skill_name="welding", source="resume", confidence=0.7),
        StagedSkill(skill_name="customer service", source="chat", confidence=0.7),
        StagedSkill(skill_name="cash handling", source="chat", confidence=0.7),
    ]
    assert has_usable_skill_evidence(_staged_with_skills(skills)) is True


def test_evidence_below_count_threshold_fails() -> None:
    skills = [
        StagedSkill(skill_name="welding", source="resume", confidence=0.7),
        StagedSkill(skill_name="blueprint reading", source="resume", confidence=0.7),
    ]
    assert has_usable_skill_evidence(_staged_with_skills(skills)) is False


def test_evidence_only_credentials_fails() -> None:
    """A user with three certificates and no work skills must NOT
    pass the floor."""
    skills = [
        StagedSkill(skill_name="Class G", source="resume", confidence=0.9),
        StagedSkill(skill_name="WHMIS 2015 Certificate", source="resume", confidence=0.9),
        StagedSkill(skill_name="310S", source="resume", confidence=0.9),
    ]
    assert has_usable_skill_evidence(_staged_with_skills(skills)) is False


def test_evidence_sources_outside_allowlist_fail() -> None:
    skills = [
        StagedSkill(skill_name="welding", source="fallback", confidence=0.9),
        StagedSkill(skill_name="blueprint reading", source="synthetic", confidence=0.9),
        StagedSkill(skill_name="forklift operation", source="inferred", confidence=0.9),
    ]
    assert has_usable_skill_evidence(_staged_with_skills(skills)) is False


def test_evidence_confidence_below_floor_fails() -> None:
    skills = [
        StagedSkill(skill_name="welding", source="resume", confidence=0.5),
        StagedSkill(skill_name="blueprint reading", source="resume", confidence=0.5),
        StagedSkill(skill_name="forklift operation", source="resume", confidence=0.5),
    ]
    assert has_usable_skill_evidence(_staged_with_skills(skills)) is False


def test_evidence_at_confidence_floor_passes() -> None:
    """Confidence == 0.6 is the inclusive floor."""
    skills = [
        StagedSkill(skill_name="welding", source="resume", confidence=0.6),
        StagedSkill(skill_name="blueprint reading", source="resume", confidence=0.6),
        StagedSkill(skill_name="forklift operation", source="resume", confidence=0.6),
    ]
    assert has_usable_skill_evidence(_staged_with_skills(skills)) is True


def test_evidence_empty_profile_fails() -> None:
    assert has_usable_skill_evidence(StagedProfile.new("sess-1")) is False


def test_evidence_short_circuits_at_floor_count() -> None:
    """Sanity: the predicate returns True as soon as the third
    distinct canonical is seen."""
    skills = [
        StagedSkill(skill_name="welding", source="resume", confidence=0.7),
        StagedSkill(skill_name="blueprint reading", source="resume", confidence=0.7),
        StagedSkill(skill_name="forklift operation", source="resume", confidence=0.7),
        # Subsequent malformed entries shouldn't matter once we cleared
        # the floor:
        StagedSkill(skill_name="", source="garbage", confidence=-9.0),
    ]
    assert has_usable_skill_evidence(_staged_with_skills(skills)) is True


# ---- AR-1b round-2: duplicate canonicals must not inflate the floor ----
def test_evidence_three_duplicate_records_do_NOT_pass() -> None:
    """Three records for the SAME canonical skill (e.g. resume + two
    chat re-mentions of "forklift") cannot trip the three-skill floor.
    Counts UNIQUE canonical names, not raw record count."""
    skills = [
        StagedSkill(skill_name="forklift", source="resume", confidence=0.9),
        StagedSkill(skill_name="forklift", source="chat", confidence=0.7),
        StagedSkill(skill_name="forklift", source="chat", confidence=0.8),
    ]
    assert has_usable_skill_evidence(_staged_with_skills(skills)) is False


def test_evidence_distinct_canonicals_pass_with_dupes_mixed_in() -> None:
    """Three distinct canonicals pass even when accompanied by
    duplicates."""
    skills = [
        StagedSkill(skill_name="forklift", source="resume", confidence=0.9),
        StagedSkill(skill_name="forklift", source="chat", confidence=0.7),   # dupe
        StagedSkill(skill_name="welding", source="resume", confidence=0.7),
        StagedSkill(skill_name="welding", source="chat", confidence=0.7),    # dupe
        StagedSkill(skill_name="blueprint reading", source="resume", confidence=0.7),
    ]
    assert has_usable_skill_evidence(_staged_with_skills(skills)) is True


def test_evidence_canonicalization_dedupes_phrasings() -> None:
    """Different surface phrasings that canonicalize to the same name
    should NOT inflate the count. Uses canonicalize_skill, which folds
    case and (typically) whitespace."""
    skills = [
        StagedSkill(skill_name="Welding", source="resume", confidence=0.7),
        StagedSkill(skill_name="welding", source="chat", confidence=0.7),
        StagedSkill(skill_name="WELDING", source="chat", confidence=0.7),
    ]
    assert has_usable_skill_evidence(_staged_with_skills(skills)) is False


def test_evidence_empty_canonical_skipped() -> None:
    """A record whose name canonicalizes to empty string is skipped --
    a forged blank record can't be counted toward the floor."""
    skills = [
        StagedSkill(skill_name="welding", source="resume", confidence=0.7),
        StagedSkill(skill_name="blueprint reading", source="resume", confidence=0.7),
        StagedSkill(skill_name="", source="resume", confidence=0.9),
    ]
    assert has_usable_skill_evidence(_staged_with_skills(skills)) is False


# ---- AR-1b round-2: malformed confidence values ----
def test_evidence_nan_confidence_rejected() -> None:
    """A NaN confidence comparison with 0.6 returns False (NaN < 0.6
    is False; NaN >= 0.6 is False), so without an explicit guard a
    NaN-confidence record would have silently passed the floor."""
    skills = [
        StagedSkill(skill_name="welding", source="resume", confidence=float("nan")),
        StagedSkill(skill_name="blueprint reading", source="resume",
                    confidence=float("nan")),
        StagedSkill(skill_name="forklift", source="resume", confidence=float("nan")),
    ]
    assert has_usable_skill_evidence(_staged_with_skills(skills)) is False


def test_evidence_infinity_confidence_rejected() -> None:
    """+inf trivially passes `>= 0.6`. Reject as non-finite."""
    skills = [
        StagedSkill(skill_name="welding", source="resume", confidence=float("inf")),
        StagedSkill(skill_name="blueprint reading", source="resume",
                    confidence=float("inf")),
        StagedSkill(skill_name="forklift", source="resume",
                    confidence=float("inf")),
    ]
    assert has_usable_skill_evidence(_staged_with_skills(skills)) is False


def test_evidence_boolean_confidence_rejected() -> None:
    """`True` is a `bool` (subclass of `int`), so it would pass an
    `isinstance(int, float)` check and act as 1.0. Reject."""
    skills = [
        StagedSkill(skill_name="welding", source="resume", confidence=True),  # type: ignore[arg-type]
        StagedSkill(skill_name="blueprint reading", source="resume",
                    confidence=True),  # type: ignore[arg-type]
        StagedSkill(skill_name="forklift", source="resume",
                    confidence=True),  # type: ignore[arg-type]
    ]
    assert has_usable_skill_evidence(_staged_with_skills(skills)) is False


def test_evidence_negative_confidence_rejected() -> None:
    """Negative confidence is out-of-range; the existing `< 0.6` check
    would have caught most but not -0.0. _is_valid_normalized_score
    enforces the full [0, 1] window."""
    skills = [
        StagedSkill(skill_name="welding", source="resume", confidence=-0.7),
        StagedSkill(skill_name="blueprint reading", source="resume",
                    confidence=-0.7),
        StagedSkill(skill_name="forklift", source="resume", confidence=-0.7),
    ]
    assert has_usable_skill_evidence(_staged_with_skills(skills)) is False


def test_evidence_above_one_confidence_rejected() -> None:
    """Confidence > 1.0 is out-of-range."""
    skills = [
        StagedSkill(skill_name="welding", source="resume", confidence=1.5),
        StagedSkill(skill_name="blueprint reading", source="resume", confidence=1.5),
        StagedSkill(skill_name="forklift", source="resume", confidence=1.5),
    ]
    assert has_usable_skill_evidence(_staged_with_skills(skills)) is False


# ---- AR-1b round-2: _is_valid_normalized_score helper ----
def test_is_valid_normalized_score_accepts_normal_floats() -> None:
    assert _is_valid_normalized_score(0.0) is True
    assert _is_valid_normalized_score(0.5) is True
    assert _is_valid_normalized_score(1.0) is True
    assert _is_valid_normalized_score(0) is True   # int
    assert _is_valid_normalized_score(1) is True


def test_is_valid_normalized_score_rejects_out_of_range() -> None:
    assert _is_valid_normalized_score(-0.01) is False
    assert _is_valid_normalized_score(1.01) is False
    assert _is_valid_normalized_score(-1.0) is False
    assert _is_valid_normalized_score(2.0) is False


def test_is_valid_normalized_score_rejects_non_finite() -> None:
    assert _is_valid_normalized_score(float("nan")) is False
    assert _is_valid_normalized_score(float("inf")) is False
    assert _is_valid_normalized_score(float("-inf")) is False


def test_is_valid_normalized_score_rejects_booleans() -> None:
    assert _is_valid_normalized_score(True) is False
    assert _is_valid_normalized_score(False) is False


def test_is_valid_normalized_score_rejects_non_numeric() -> None:
    assert _is_valid_normalized_score(None) is False
    assert _is_valid_normalized_score("0.5") is False
    assert _is_valid_normalized_score([0.5]) is False
    assert _is_valid_normalized_score({"x": 0.5}) is False


# =========================================================================
# is_credential_only_band_cap / soft-offer predicates
# =========================================================================
def _lead_result(
    *,
    pre_cap_score: float | None,
    caps: list[str] | None,
) -> dict:
    """Build a lead-result dict shaped like engine.py emits at
    engine.py:1042+ / engine.py:1335+."""
    result: dict = {}
    if pre_cap_score is not None:
        result["score_explanation"] = {
            "score_components": {"score_pre_caps": pre_cap_score},
        }
        if caps is not None:
            result["score_explanation"]["caps_applied"] = caps
    return result


def test_credential_only_cap_strong_pre_cap_passes() -> None:
    """Pre-cap band == strong + sole credential cap → True."""
    result = _lead_result(
        pre_cap_score=MATCH.band_strong + 0.01,
        caps=[_CRED_CAP_FLAG],
    )
    assert is_credential_only_band_cap(result) is True


def test_credential_only_cap_good_pre_cap_passes() -> None:
    """Pre-cap band == good + sole credential cap → True."""
    result = _lead_result(
        pre_cap_score=MATCH.band_good + 0.01,
        caps=[_CRED_CAP_FLAG],
    )
    assert is_credential_only_band_cap(result) is True


def test_credential_only_cap_stretch_pre_cap_fails() -> None:
    """Pre-cap band == stretch → no soft offer (the lead match wasn't
    a strong/good match to begin with)."""
    result = _lead_result(
        pre_cap_score=MATCH.band_stretch + 0.01,
        caps=[_CRED_CAP_FLAG],
    )
    assert is_credential_only_band_cap(result) is False


def test_credential_only_cap_low_pre_cap_fails() -> None:
    result = _lead_result(
        pre_cap_score=max(0.0, MATCH.band_stretch - 0.05),
        caps=[_CRED_CAP_FLAG],
    )
    assert is_credential_only_band_cap(result) is False


def test_caps_other_than_credential_fail() -> None:
    """A lead match capped by no-experience OR work-type-mismatch (not
    by credential) doesn't trip the soft offer -- those caps reflect a
    different remediation path."""
    result = _lead_result(
        pre_cap_score=MATCH.band_good + 0.01,
        caps=["band_capped_by_no_experience"],
    )
    assert is_credential_only_band_cap(result) is False


def test_caps_with_extra_flag_fails() -> None:
    """Credential cap PLUS another cap doesn't qualify -- v11 lock:
    ONLY the credential cap may be present."""
    result = _lead_result(
        pre_cap_score=MATCH.band_good + 0.01,
        caps=[_CRED_CAP_FLAG, "band_capped_by_work_type_mismatch"],
    )
    assert is_credential_only_band_cap(result) is False


def test_no_caps_at_all_fails() -> None:
    """An uncapped result -- the lead was already good and isn't being
    blocked by anything -- doesn't need the adjacency offer."""
    result = _lead_result(pre_cap_score=MATCH.band_good + 0.01, caps=[])
    assert is_credential_only_band_cap(result) is False


def test_missing_score_pre_caps_fails() -> None:
    """Missing field → False (no crash)."""
    result = {
        "score_explanation": {
            "score_components": {},
            "caps_applied": [_CRED_CAP_FLAG],
        },
    }
    assert is_credential_only_band_cap(result) is False


def test_missing_score_explanation_fails() -> None:
    """A bare result with no score_explanation → False."""
    result = {"caps_applied": [_CRED_CAP_FLAG]}  # caps at WRONG level
    assert is_credential_only_band_cap(result) is False


def test_does_NOT_consult_top_level_caps_applied() -> None:
    """Result-level caps_applied (engine.py:588 stores caps under
    score_explanation; v6 review). A top-level field must NOT trigger
    the soft offer."""
    result = {
        "caps_applied": [_CRED_CAP_FLAG],   # wrong path
        "score_explanation": {
            "score_components": {"score_pre_caps": MATCH.band_good + 0.01},
            # NO caps_applied here
        },
    }
    assert is_credential_only_band_cap(result) is False


def test_malformed_inputs_handled() -> None:
    """Wrong types throughout must not crash."""
    assert is_credential_only_band_cap(None) is False  # type: ignore[arg-type]
    assert is_credential_only_band_cap("not a dict") is False  # type: ignore[arg-type]
    assert is_credential_only_band_cap({}) is False
    assert is_credential_only_band_cap({"score_explanation": "string"}) is False


# ---- AR-1b round-2: malformed score_pre_caps numeric values ----
def test_credential_cap_rejects_boolean_score_pre_caps() -> None:
    """`True` would pass an isinstance(int, float) check and float() to
    1.0 (a band_strong score), spuriously firing the soft offer. Reject
    via _is_valid_normalized_score."""
    result = _lead_result(pre_cap_score=True, caps=[_CRED_CAP_FLAG])  # type: ignore[arg-type]
    assert is_credential_only_band_cap(result) is False
    result_false = _lead_result(pre_cap_score=False, caps=[_CRED_CAP_FLAG])  # type: ignore[arg-type]
    assert is_credential_only_band_cap(result_false) is False


def test_credential_cap_rejects_infinity_score_pre_caps() -> None:
    """+inf trivially clears any threshold; reject as non-finite."""
    result = _lead_result(pre_cap_score=float("inf"), caps=[_CRED_CAP_FLAG])
    assert is_credential_only_band_cap(result) is False
    result_neg = _lead_result(pre_cap_score=float("-inf"), caps=[_CRED_CAP_FLAG])
    assert is_credential_only_band_cap(result_neg) is False


def test_credential_cap_rejects_nan_score_pre_caps() -> None:
    """NaN comparisons are always False, so `_band(nan)` would fall
    through to 'low' and the predicate would return False -- but only
    after silently calling _band on a NaN. The explicit guard makes the
    rejection clean."""
    result = _lead_result(pre_cap_score=float("nan"), caps=[_CRED_CAP_FLAG])
    assert is_credential_only_band_cap(result) is False


def test_credential_cap_rejects_out_of_range_score_pre_caps() -> None:
    """Scores outside [0, 1] shouldn't reach this predicate (the engine
    clamps), but a forged score_explanation could try."""
    result_high = _lead_result(pre_cap_score=1.5, caps=[_CRED_CAP_FLAG])
    assert is_credential_only_band_cap(result_high) is False
    result_neg = _lead_result(pre_cap_score=-0.1, caps=[_CRED_CAP_FLAG])
    assert is_credential_only_band_cap(result_neg) is False


def test_result_caps_helper_handles_garbage() -> None:
    assert _result_caps({}) == ()
    assert _result_caps({"score_explanation": "string"}) == ()
    assert _result_caps({"score_explanation": {"caps_applied": "string"}}) == ()
    assert _result_caps({"score_explanation": {"caps_applied": [1, "ok", None]}}) == ("ok",)


# =========================================================================
# Composite soft-offer predicates
# =========================================================================
def _good_lead_with_cred_cap() -> dict:
    return _lead_result(
        pre_cap_score=MATCH.band_good + 0.01,
        caps=[_CRED_CAP_FLAG],
    )


def _three_skill_staged() -> StagedProfile:
    return _staged_with_skills([
        StagedSkill(skill_name="welding", source="resume", confidence=0.7),
        StagedSkill(skill_name="blueprint reading", source="resume", confidence=0.7),
        StagedSkill(skill_name="forklift operation", source="resume", confidence=0.7),
    ])


def test_soft_offer_on_matches_fires_when_both_conditions_met() -> None:
    assert should_emit_soft_offer_on_matches(
        _good_lead_with_cred_cap(), _three_skill_staged(),
    ) is True


def test_soft_offer_on_matches_suppressed_when_cap_not_credential_only() -> None:
    result = _lead_result(
        pre_cap_score=MATCH.band_good + 0.01,
        caps=["band_capped_by_no_experience"],
    )
    assert should_emit_soft_offer_on_matches(result, _three_skill_staged()) is False


def test_soft_offer_on_matches_suppressed_when_no_evidence() -> None:
    """Lead is credential-capped but the user has no usable evidence to
    surface adjacent roles against."""
    assert should_emit_soft_offer_on_matches(
        _good_lead_with_cred_cap(),
        _staged_with_skills([]),
    ) is False


def test_soft_offer_on_no_match_fires_when_evidence_present() -> None:
    """A present_no_match outcome with usable evidence is the second
    soft-offer surface (v3 Q-C lock)."""
    assert should_emit_soft_offer_on_no_match(_three_skill_staged()) is True


def test_soft_offer_on_no_match_suppressed_without_evidence() -> None:
    """A present_no_match outcome without usable evidence cannot
    surface adjacency -- we'd have nothing to recommend."""
    assert should_emit_soft_offer_on_no_match(_staged_with_skills([])) is False
