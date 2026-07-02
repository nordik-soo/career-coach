"""Structural validator for the matching eval corpus (Step 4, 2026-07-02).

Loads data/matching_eval_corpus.yaml into typed dataclasses and asserts
structural invariants: unique ids, closed vocabularies, coverage floor,
enum agreement with engine code, no-skip enforcement.

Design (per docs/matching-eval-corpus-schema.md §Validation):
  - Loader (structural, at parse): dataclass construction with type-
    checked field values. Malformed corpus raises CorpusValidationError.
  - Validators (this module's tests): each test asserts one specific
    invariant. Failures are loud and specific -- no test bundles many
    invariants into a single opaque assertion.
  - Zero skips, ever (schema Design Goal 1). No @pytest.mark.skip /
    skipif in this file. Failures stay loud until the referenced
    corpus gap is filled (Step 5+).

Scope of Step 4 (LOADER + VALIDATOR ONLY):
  - No engine invocation
  - No DB access (pytest.mark.nodb)
  - No LLM
  - No mocks
  - No filling coverage from 17 -> 36 (that's Step 5)
  - No first-run engine disagreement resolution (that's Step 6)

Expected state at Step 4 ship:
  16 tests PASS: structural invariants that hold today
  2  tests FAIL: coverage-floor + missing-diagnosis-outcome
                  (both are the exact Step 5 todo list, loud by design)

This file is NOT integrated into CI until Step 8. Between now and then,
`pytest tests/test_matching_eval.py` reports the honest state.
"""
from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml

# Every test in this module reads a frozen YAML file; no engine, no DB.
pytestmark = pytest.mark.nodb


# ---------------------------------------------------------------------------
# Closed vocabularies (copied from schema doc §Vocabularies)
# ---------------------------------------------------------------------------
# Rule from the schema doc: "when the engine grows a new cap reason or
# diagnosis outcome, the vocabulary here is updated in the same PR".
# The corpus's *accepted* vocabulary is what this file defines. Separate
# tests below cross-check against engine code so drift is caught.
_VALID_BAND: frozenset[str] = frozenset(
    {"strong", "good", "stretch", "explore_later", "none"}
)
_VALID_VIA_STAGE: frozenset[str] = frozenset({"exact", "fuzzy", "semantic"})
_VALID_CAP_REASONS: frozenset[str] = frozenset({
    "band_capped_by_credential",
    "band_capped_by_no_experience",
    "band_capped_by_work_type_mismatch",
})
_VALID_DIAGNOSIS: frozenset[str] = frozenset({
    "UNDETERMINED", "MARKET_DATA_UNAVAILABLE", "READY_TO_APPLY",
    "PREPARATION_GAP", "SKILL_ADJACENT_AVAILABLE", "NO_OPPORTUNITY_FOUND",
})
_VALID_CATEGORIES: frozenset[str] = frozenset({
    "credential_gap", "cap_semantics", "no_match", "negative_control",
    "semantic_bridge", "fuzzy_boundary", "adjacent_only", "thin_evidence",
    "work_type", "direct_title", "family_gate", "ready_to_apply",
})
_VALID_REQUIREMENT: frozenset[str] = frozenset({"required", "preferred"})
_VALID_EXPECTATION_STATUS: frozenset[str] = frozenset(
    {"transcribed", "authored"}
)

# Coverage floor per schema §Coverage floor. Each key is a frozenset of
# category tags; the value is the minimum number of cases that must
# match ANY of the categories in the key. Cases may count toward
# multiple groups when they carry multiple categories.
_COVERAGE_FLOOR: dict[frozenset[str], int] = {
    frozenset({"credential_gap"}): 6,
    frozenset({"negative_control"}): 4,
    frozenset({"no_match", "thin_evidence"}): 6,
    frozenset({"semantic_bridge", "fuzzy_boundary"}): 8,
    frozenset({"cap_semantics", "work_type", "family_gate"}): 6,
    frozenset({"ready_to_apply", "direct_title", "adjacent_only"}): 6,
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
class CorpusValidationError(ValueError):
    """Raised by load_corpus when the YAML fails structural validation."""


@dataclass(frozen=True)
class PostingSkill:
    name: str
    requirement: str
    is_credential: bool


@dataclass(frozen=True)
class Posting:
    posting_id: str
    is_synthetic: bool
    transcribed_from_sccc: bool
    title: str
    employer: str
    noc_code: str
    location: str
    region_code: str
    employment_type: str
    posted_days_ago: int
    skills: tuple[PostingSkill, ...]
    embedding_profile: str = "default"
    description_snippet: str | None = None
    retired: bool = False


@dataclass(frozen=True)
class MatchedRequirement:
    requirement: str
    via_stage: str
    user_skill: str


@dataclass(frozen=True)
class JobExpect:
    posting_id: str
    band: str | None = None
    band_at_least: bool = False
    cap_reasons: tuple[str, ...] = ()
    cap_reasons_forbidden: tuple[str, ...] = ()
    matched_required: tuple[MatchedRequirement, ...] = ()
    missing_required_contains: tuple[str, ...] = ()
    blocking_credential: str | None = None


@dataclass(frozen=True)
class Expect:
    diagnosis: str
    jobs: tuple[JobExpect, ...] = ()
    jobs_absent: tuple[str, ...] = ()


@dataclass(frozen=True)
class Profile:
    target_role: str
    skill_phrases: tuple[str, ...] = ()
    experience_text: str | None = None
    education_text: str | None = None
    work_type_preference: str | None = None


@dataclass(frozen=True)
class Case:
    case_id: str
    expectation_status: str
    description: str
    categories: tuple[str, ...]
    profile: Profile
    expect: Expect


@dataclass(frozen=True)
class EmbeddingPair:
    a: str
    b: str
    cosine: float


@dataclass(frozen=True)
class Corpus:
    corpus_version: int
    engine_version_pinned: str
    frozen_today: str
    embedding_fixtures: tuple[EmbeddingPair, ...]
    posting_bank: tuple[Posting, ...]
    cases: tuple[Case, ...]
    corpus_verified_at: str | None = None

    def posting_by_id(self, pid: str) -> Posting | None:
        for p in self.posting_bank:
            if p.posting_id == pid:
                return p
        return None


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
_CORPUS_PATH: Path = (
    Path(__file__).resolve().parents[1] / "data" / "matching_eval_corpus.yaml"
)


def _require(condition: bool, msg: str) -> None:
    if not condition:
        raise CorpusValidationError(msg)


def _parse_posting_skill(raw: Any, ctx: str) -> PostingSkill:
    _require(
        isinstance(raw, dict),
        f"{ctx}: skill entry must be a mapping, got {type(raw).__name__}",
    )
    name = raw.get("name")
    requirement = raw.get("requirement")
    is_credential = raw.get("is_credential")
    _require(
        isinstance(name, str) and name.strip(),
        f"{ctx}: skill missing 'name' (or empty)",
    )
    _require(
        isinstance(requirement, str),
        f"{ctx}: skill '{name}' missing 'requirement'",
    )
    _require(
        isinstance(is_credential, bool),
        f"{ctx}: skill '{name}' 'is_credential' must be bool",
    )
    return PostingSkill(
        name=name, requirement=requirement, is_credential=is_credential,
    )


def _parse_posting(raw: Any, index: int) -> Posting:
    _require(
        isinstance(raw, dict),
        f"posting_bank[{index}] must be a mapping",
    )
    posting_id = raw.get("posting_id")
    _require(
        isinstance(posting_id, str) and posting_id.strip(),
        f"posting_bank[{index}] missing posting_id",
    )
    ctx = f"posting_bank[{posting_id}]"
    for required_field in ("is_synthetic", "transcribed_from_sccc"):
        _require(
            required_field in raw,
            f"{ctx}: required field '{required_field}' missing",
        )
        _require(
            isinstance(raw[required_field], bool),
            f"{ctx}: '{required_field}' must be bool",
        )
    for str_field in (
        "title", "employer", "noc_code", "location",
        "region_code", "employment_type",
    ):
        _require(
            str_field in raw and isinstance(raw[str_field], str),
            f"{ctx}: required field '{str_field}' missing or not str",
        )
    _require(
        isinstance(raw.get("posted_days_ago"), int),
        f"{ctx}: 'posted_days_ago' must be int",
    )
    skills_raw = raw.get("skills") or []
    _require(
        isinstance(skills_raw, list) and skills_raw,
        f"{ctx}: 'skills' must be a non-empty list",
    )
    skills = tuple(
        _parse_posting_skill(s, f"{ctx}.skills[{i}]")
        for i, s in enumerate(skills_raw)
    )
    return Posting(
        posting_id=posting_id,
        is_synthetic=raw["is_synthetic"],
        transcribed_from_sccc=raw["transcribed_from_sccc"],
        title=raw["title"],
        employer=raw["employer"],
        noc_code=raw["noc_code"],
        location=raw["location"],
        region_code=raw["region_code"],
        employment_type=raw["employment_type"],
        posted_days_ago=raw["posted_days_ago"],
        skills=skills,
        embedding_profile=raw.get("embedding_profile", "default"),
        description_snippet=raw.get("description_snippet"),
        retired=bool(raw.get("retired", False)),
    )


def _parse_matched(raw: Any, ctx: str) -> MatchedRequirement:
    _require(
        isinstance(raw, dict),
        f"{ctx}: matched_required entry must be a mapping",
    )
    for f in ("requirement", "via_stage", "user_skill"):
        _require(
            isinstance(raw.get(f), str) and raw[f],
            f"{ctx}: missing string field '{f}'",
        )
    return MatchedRequirement(
        requirement=raw["requirement"],
        via_stage=raw["via_stage"],
        user_skill=raw["user_skill"],
    )


def _parse_job_expect(raw: Any, ctx: str) -> JobExpect:
    _require(
        isinstance(raw, dict),
        f"{ctx}: expect.jobs entry must be a mapping",
    )
    pid = raw.get("posting_id")
    _require(
        isinstance(pid, str) and pid.strip(),
        f"{ctx}: 'posting_id' missing",
    )
    return JobExpect(
        posting_id=pid,
        band=raw.get("band"),
        band_at_least=bool(raw.get("band_at_least", False)),
        cap_reasons=tuple(raw.get("cap_reasons") or ()),
        cap_reasons_forbidden=tuple(raw.get("cap_reasons_forbidden") or ()),
        matched_required=tuple(
            _parse_matched(m, f"{ctx}.matched_required[{i}]")
            for i, m in enumerate(raw.get("matched_required") or ())
        ),
        missing_required_contains=tuple(
            raw.get("missing_required_contains") or ()
        ),
        blocking_credential=raw.get("blocking_credential"),
    )


def _parse_expect(raw: Any, ctx: str) -> Expect:
    _require(
        isinstance(raw, dict),
        f"{ctx}: expect must be a mapping",
    )
    diagnosis = raw.get("diagnosis")
    _require(
        isinstance(diagnosis, str),
        f"{ctx}: 'diagnosis' missing or not str",
    )
    jobs = tuple(
        _parse_job_expect(j, f"{ctx}.jobs[{i}]")
        for i, j in enumerate(raw.get("jobs") or ())
    )
    jobs_absent = tuple(raw.get("jobs_absent") or ())
    for j in jobs_absent:
        _require(
            isinstance(j, str),
            f"{ctx}.jobs_absent: must be strings",
        )
    return Expect(diagnosis=diagnosis, jobs=jobs, jobs_absent=jobs_absent)


def _parse_case(raw: Any, index: int) -> Case:
    _require(
        isinstance(raw, dict),
        f"cases[{index}] must be a mapping",
    )
    cid = raw.get("case_id")
    _require(
        isinstance(cid, str) and cid.strip(),
        f"cases[{index}]: 'case_id' missing",
    )
    ctx = f"cases[{cid}]"
    profile_raw = raw.get("profile") or {}
    _require(
        isinstance(profile_raw, dict),
        f"{ctx}: 'profile' must be a mapping",
    )
    profile = Profile(
        target_role=str(profile_raw.get("target_role") or ""),
        skill_phrases=tuple(profile_raw.get("skill_phrases") or ()),
        experience_text=profile_raw.get("experience_text"),
        education_text=profile_raw.get("education_text"),
        work_type_preference=profile_raw.get("work_type_preference"),
    )
    return Case(
        case_id=cid,
        expectation_status=str(
            raw.get("expectation_status") or ""
        ),
        description=str(raw.get("description") or ""),
        categories=tuple(raw.get("categories") or ()),
        profile=profile,
        expect=_parse_expect(raw.get("expect"), ctx),
    )


def _parse_embedding_pair(raw: Any, index: int) -> EmbeddingPair:
    _require(
        isinstance(raw, dict),
        f"embedding_fixtures[{index}] must be a mapping",
    )
    for f in ("a", "b"):
        _require(
            isinstance(raw.get(f), str),
            f"embedding_fixtures[{index}] missing string field '{f}'",
        )
    _require(
        isinstance(raw.get("cosine"), (int, float)),
        f"embedding_fixtures[{index}] 'cosine' must be numeric",
    )
    return EmbeddingPair(
        a=raw["a"], b=raw["b"], cosine=float(raw["cosine"]),
    )


def load_corpus(path: Path = _CORPUS_PATH) -> Corpus:
    """Parse the eval corpus YAML into typed dataclasses.

    Raises CorpusValidationError on any structural malformation.
    Successful return means the corpus is well-formed at the type
    level. Vocabulary and cross-reference validation happens in
    the tests below, not at load time -- separation of concerns.
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise CorpusValidationError(f"YAML parse failed: {exc}") from exc

    _require(
        isinstance(raw, dict),
        "corpus root must be a mapping",
    )
    for f in ("corpus_version", "engine_version_pinned", "frozen_today"):
        _require(f in raw, f"corpus root missing '{f}'")
    _require(
        isinstance(raw["corpus_version"], int),
        "'corpus_version' must be int",
    )
    _require(
        isinstance(raw["engine_version_pinned"], str),
        "'engine_version_pinned' must be str",
    )

    posting_bank_raw = raw.get("posting_bank") or []
    _require(
        isinstance(posting_bank_raw, list) and posting_bank_raw,
        "'posting_bank' must be a non-empty list",
    )
    postings = tuple(
        _parse_posting(p, i) for i, p in enumerate(posting_bank_raw)
    )

    cases_raw = raw.get("cases") or []
    _require(
        isinstance(cases_raw, list) and cases_raw,
        "'cases' must be a non-empty list",
    )
    cases = tuple(_parse_case(c, i) for i, c in enumerate(cases_raw))

    embedding_fixtures = tuple(
        _parse_embedding_pair(e, i)
        for i, e in enumerate(raw.get("embedding_fixtures") or ())
    )

    return Corpus(
        corpus_version=raw["corpus_version"],
        engine_version_pinned=raw["engine_version_pinned"],
        frozen_today=str(raw["frozen_today"]),
        embedding_fixtures=embedding_fixtures,
        posting_bank=postings,
        cases=cases,
        corpus_verified_at=raw.get("corpus_verified_at"),
    )


# ---------------------------------------------------------------------------
# Module-level fixture: load once, share across tests
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def corpus() -> Corpus:
    return load_corpus()


# ===========================================================================
# 1. Loader
# ===========================================================================
def test_corpus_loads_without_error() -> None:
    """The corpus YAML parses into typed dataclasses without raising.
    This is the load contract; every other test depends on it."""
    c = load_corpus()
    assert c.corpus_version == 1
    assert c.posting_bank, "posting_bank must be non-empty"
    assert c.cases, "cases must be non-empty"


# ===========================================================================
# 2. Uniqueness invariants
# ===========================================================================
def test_all_case_ids_unique(corpus: Corpus) -> None:
    """No two cases may share a case_id. Calibration reports diff by id
    (per schema); duplicates would double-count silently."""
    seen: dict[str, int] = {}
    for c in corpus.cases:
        seen[c.case_id] = seen.get(c.case_id, 0) + 1
    dups = {cid: n for cid, n in seen.items() if n > 1}
    assert not dups, f"duplicate case_ids: {sorted(dups)}"


def test_all_posting_ids_unique(corpus: Corpus) -> None:
    """No two postings may share a posting_id. Cases reference them by
    id; duplicates would resolve ambiguously."""
    seen: dict[str, int] = {}
    for p in corpus.posting_bank:
        seen[p.posting_id] = seen.get(p.posting_id, 0) + 1
    dups = {pid: n for pid, n in seen.items() if n > 1}
    assert not dups, f"duplicate posting_ids: {sorted(dups)}"


# ===========================================================================
# 3. Cross-reference resolution
# ===========================================================================
def test_every_case_job_ref_resolves_to_posting_bank(corpus: Corpus) -> None:
    """Every expect.jobs[i].posting_id must resolve to a posting in the
    bank. Dangling references would fail-open (test asserts nothing
    against a nonexistent posting)."""
    bank_ids = {p.posting_id for p in corpus.posting_bank}
    dangling: list[tuple[str, str]] = []
    for c in corpus.cases:
        for j in c.expect.jobs:
            if j.posting_id not in bank_ids:
                dangling.append((c.case_id, j.posting_id))
        for pid in c.expect.jobs_absent:
            if pid not in bank_ids:
                dangling.append((c.case_id, pid))
    assert not dangling, (
        f"case job references missing from posting_bank: {dangling}"
    )


# ===========================================================================
# 4. Closed-vocabulary validators
# ===========================================================================
def test_all_band_values_in_closed_vocabulary(corpus: Corpus) -> None:
    """Per-job 'band' values must be from the closed set. Schema §Vocabularies."""
    bad: list[tuple[str, str]] = []
    for c in corpus.cases:
        for j in c.expect.jobs:
            if j.band is None:
                continue
            if j.band not in _VALID_BAND:
                bad.append((c.case_id, j.band))
    assert not bad, (
        f"band values outside closed vocabulary {sorted(_VALID_BAND)}: {bad}"
    )


def test_all_via_stage_values_in_closed_vocabulary(corpus: Corpus) -> None:
    """matched_required[].via_stage must be from the closed set. Schema §Vocabularies."""
    bad: list[tuple[str, str]] = []
    for c in corpus.cases:
        for j in c.expect.jobs:
            for m in j.matched_required:
                if m.via_stage not in _VALID_VIA_STAGE:
                    bad.append((c.case_id, m.via_stage))
    assert not bad, (
        f"via_stage values outside closed vocabulary {sorted(_VALID_VIA_STAGE)}: {bad}"
    )


def test_all_cap_reasons_in_closed_vocabulary(corpus: Corpus) -> None:
    """cap_reasons and cap_reasons_forbidden entries must be from the closed set."""
    bad: list[tuple[str, str, str]] = []
    for c in corpus.cases:
        for j in c.expect.jobs:
            for r in j.cap_reasons:
                if r not in _VALID_CAP_REASONS:
                    bad.append((c.case_id, "cap_reasons", r))
            for r in j.cap_reasons_forbidden:
                if r not in _VALID_CAP_REASONS:
                    bad.append((c.case_id, "cap_reasons_forbidden", r))
    assert not bad, (
        f"cap_reasons outside closed vocabulary {sorted(_VALID_CAP_REASONS)}: {bad}"
    )


def test_all_diagnosis_values_in_closed_vocabulary(corpus: Corpus) -> None:
    """expect.diagnosis must be one of the six Outcome literals."""
    bad: list[tuple[str, str]] = []
    for c in corpus.cases:
        if c.expect.diagnosis not in _VALID_DIAGNOSIS:
            bad.append((c.case_id, c.expect.diagnosis))
    assert not bad, (
        f"diagnosis values outside closed vocabulary {sorted(_VALID_DIAGNOSIS)}: {bad}"
    )


def test_all_categories_in_closed_vocabulary(corpus: Corpus) -> None:
    """Every case category tag must be from the closed set."""
    bad: list[tuple[str, str]] = []
    for c in corpus.cases:
        for cat in c.categories:
            if cat not in _VALID_CATEGORIES:
                bad.append((c.case_id, cat))
    assert not bad, (
        f"categories outside closed vocabulary {sorted(_VALID_CATEGORIES)}: {bad}"
    )


# ===========================================================================
# 5. Case-level structural rules
# ===========================================================================
def test_cap_reasons_and_forbidden_dont_overlap(corpus: Corpus) -> None:
    """A reason cannot be simultaneously required and forbidden on the same job."""
    bad: list[tuple[str, str, set[str]]] = []
    for c in corpus.cases:
        for j in c.expect.jobs:
            overlap = set(j.cap_reasons) & set(j.cap_reasons_forbidden)
            if overlap:
                bad.append((c.case_id, j.posting_id, overlap))
    assert not bad, f"cap_reasons ∩ cap_reasons_forbidden non-empty: {bad}"


def test_every_case_has_at_least_one_category(corpus: Corpus) -> None:
    """Coverage floor enforcement depends on category tags; every case
    must carry at least one."""
    empty = [c.case_id for c in corpus.cases if not c.categories]
    assert not empty, f"cases with no categories: {empty}"


# ===========================================================================
# 6. Posting-level structural rules (Step 3 introduced is_synthetic)
# ===========================================================================
def test_every_posting_has_is_synthetic_set(corpus: Corpus) -> None:
    """Step 3 added is_synthetic as a required posting field. Every
    posting must carry the flag; missing means someone added a posting
    without the Step-3-introduced discipline."""
    # Field is required in the loader (raises on missing) -- this test
    # exists as a self-documenting invariant that survives loader
    # refactors.
    for p in corpus.posting_bank:
        assert isinstance(p.is_synthetic, bool), (
            f"posting {p.posting_id}: is_synthetic missing or not bool"
        )


def test_is_synthetic_and_transcribed_are_mutually_exclusive(
    corpus: Corpus,
) -> None:
    """A posting cannot be both is_synthetic: true AND
    transcribed_from_sccc: true. If both are set, the flags contradict:
    synthetic-by-design AND copied-from-live-SCCC can't both be true.
    Documents the semantic contract Step 3 introduced."""
    bad = [
        p.posting_id
        for p in corpus.posting_bank
        if p.is_synthetic and p.transcribed_from_sccc
    ]
    assert not bad, (
        f"postings with both is_synthetic AND transcribed_from_sccc true: {bad}"
    )


# ===========================================================================
# 7. Engine agreement (cross-check corpus vocab against actual code)
# ===========================================================================
def test_is_credential_flag_agrees_with_engine_helper(corpus: Corpus) -> None:
    """Every posting skill's is_credential flag must agree with
    engine.is_credential_skill_name(). Divergence is a real bug in one
    of the two places (per schema §Validation, CI layer).

    This test can catch either a corpus author labeling a normal skill
    as credential, or the engine helper missing a credential keyword."""
    from skillbridge.match.engine import is_credential_skill_name

    disagreements: list[tuple[str, str, bool, bool]] = []
    for p in corpus.posting_bank:
        for s in p.skills:
            engine_says = is_credential_skill_name(s.name)
            if s.is_credential != engine_says:
                disagreements.append(
                    (p.posting_id, s.name, s.is_credential, engine_says)
                )
    assert not disagreements, (
        "is_credential flag disagreement between corpus and "
        f"is_credential_skill_name: {disagreements}"
    )


def test_corpus_diagnosis_vocab_matches_engine_outcome_literal() -> None:
    """The corpus's _VALID_DIAGNOSIS set must equal the engine's
    inventory_diagnosis.Outcome enum. Drift means either the corpus
    is stale against an engine enum change or the engine dropped an
    outcome the corpus still references."""
    from skillbridge.chat.inventory_diagnosis import Outcome
    from typing import get_args

    engine_outcomes = frozenset(get_args(Outcome))
    assert _VALID_DIAGNOSIS == engine_outcomes, (
        f"corpus _VALID_DIAGNOSIS diverges from engine Outcome. "
        f"corpus_only={_VALID_DIAGNOSIS - engine_outcomes} "
        f"engine_only={engine_outcomes - _VALID_DIAGNOSIS}"
    )


def test_corpus_via_stage_vocab_matches_engine_stage_literal() -> None:
    """The corpus's _VALID_VIA_STAGE must equal the engine's
    SkillAlignmentStage. Same drift-catching intent as the diagnosis
    cross-check."""
    from skillbridge.match.alignment import SkillAlignmentStage
    from typing import get_args

    engine_stages = frozenset(get_args(SkillAlignmentStage))
    assert _VALID_VIA_STAGE == engine_stages, (
        f"corpus _VALID_VIA_STAGE diverges from engine SkillAlignmentStage. "
        f"corpus_only={_VALID_VIA_STAGE - engine_stages} "
        f"engine_only={engine_stages - _VALID_VIA_STAGE}"
    )


# ===========================================================================
# 8. Coverage floor (schema §Coverage floor, CI-enforced)
# ===========================================================================
def test_coverage_floor_met(corpus: Corpus) -> None:
    """Each category grouping must have at least its minimum case count.
    A case counts toward every grouping that contains any of its tags.

    Expected to FAIL at Step 4 ship: the 17-case seed corpus does not
    meet floors for negative_control / no_match+thin_evidence /
    semantic_bridge+fuzzy_boundary / ready_to_apply+direct_title+
    adjacent_only. Failures name the exact gaps -- that's the Step 5
    todo list.

    NOT skipped, NOT xfailed. Loud failure until the gaps close."""
    shortfalls: list[tuple[set[str], int, int]] = []
    for group, minimum in _COVERAGE_FLOOR.items():
        count = sum(
            1 for c in corpus.cases
            if set(c.categories) & group
        )
        if count < minimum:
            shortfalls.append((set(group), count, minimum))
    assert not shortfalls, (
        "coverage floor not met (Step 5 fills these):\n  "
        + "\n  ".join(
            f"{sorted(g)}: {n}/{m} (need {m - n} more)"
            for g, n, m in shortfalls
        )
    )


# ===========================================================================
# 9. Enum-coverage floor (every engine enum has ≥1 exercising case)
# ===========================================================================
def test_every_engine_cap_reason_has_exercising_case(corpus: Corpus) -> None:
    """Every cap_reason in the closed vocabulary must be asserted in
    at least one case (either in cap_reasons or cap_reasons_forbidden).
    A cap-reason with zero coverage means we're not testing that gate."""
    exercised: set[str] = set()
    for c in corpus.cases:
        for j in c.expect.jobs:
            exercised.update(j.cap_reasons)
            exercised.update(j.cap_reasons_forbidden)
    missing = _VALID_CAP_REASONS - exercised
    assert not missing, (
        f"engine cap_reasons with zero corpus coverage: {sorted(missing)}"
    )


def test_every_diagnosis_outcome_has_exercising_case(corpus: Corpus) -> None:
    """Every Outcome value must be exercised by at least one case's
    expect.diagnosis. Zero coverage on any outcome means we're not
    testing that inventory-diagnosis branch.

    Expected to FAIL at Step 4 ship: the 17-case seed exercises
    PREPARATION_GAP, READY_TO_APPLY, NO_OPPORTUNITY_FOUND, UNDETERMINED
    (4/6). Missing MARKET_DATA_UNAVAILABLE and SKILL_ADJACENT_AVAILABLE
    -- both need constructed inputs (see schema §Coverage floor,
    no_match/thin_evidence row). That's the Step 5 todo list."""
    exercised = {c.expect.diagnosis for c in corpus.cases}
    missing = _VALID_DIAGNOSIS - exercised
    assert not missing, (
        f"engine diagnosis outcomes with zero corpus coverage: {sorted(missing)}"
    )


# ===========================================================================
# 10. Anti-skip enforcement (Design Goal 1)
# ===========================================================================
def test_no_pytest_skip_in_eval_module() -> None:
    """The eval module must contain zero pytest.skip / skipif calls.
    Structural test: greps this file's own source. Design Goal 1 in
    the schema doc: 'pytest.skip is structurally forbidden in the
    eval module (enforced by CI).'

    The schema says: 'Data drift cannot erode coverage because there
    is no external data.' Skips would erode coverage silently."""
    source = Path(__file__).read_text(encoding="utf-8")
    # Strip strings + comments FIRST so the schema-doc-quoted examples
    # inside this test's own docstring don't trigger a false positive.
    stripped = re.sub(r'"""[\s\S]*?"""', "", source)  # triple-double
    stripped = re.sub(r"'''[\s\S]*?'''", "", stripped)  # triple-single
    stripped = re.sub(r'#.*', "", stripped)  # comments
    forbidden_patterns = [
        r"\bpytest\.skip\b",
        r"\bpytest\.mark\.skip\b",
        r"\bpytest\.mark\.skipif\b",
        r"@pytest\.mark\.xfail",
    ]
    hits: list[str] = []
    for pat in forbidden_patterns:
        if re.search(pat, stripped):
            hits.append(pat)
    assert not hits, (
        f"forbidden skip/xfail markers in eval module: {hits}. "
        "Zero skips, ever (schema Design Goal 1)."
    )
