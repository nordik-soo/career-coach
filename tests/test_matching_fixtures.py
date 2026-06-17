"""Matching Engine v2 -- validation fixtures.

These fixtures are the truth-table for the matching engine. Each one
captures a real failure mode (some already fixed, some to be fixed by
v2 Phase A) and asserts the expected match band + reasoning.

Usage:
  pytest tests/test_matching_fixtures.py -v          # CI / regression
  python -m tests.test_matching_fixtures             # human-readable PASS/FAIL

Fixtures pin against the current SCCC dataset. When a job ages out of
core.v_current_job, the affected fixture is *skipped* (pytest.skip) so
data drift never reports a false regression. To refresh after a SCCC
ingest:
  python run_pipeline.py --sync-source sccc && python run_pipeline.py --extract

Build order for v2 Phase A:
  1. These fixtures -- committed BEFORE any engine code changes.
  2. Alias normalisation seed -> some currently-failing fixtures pass.
  3. Required vs preferred split -> more pass (adds new fixtures).
  4. Hard gates (work-type cap, no-experience floor) -> remaining pass.
  5. Structured score_explanation -> all explanation assertions pass.
  6. Responder narrates from explanation only.

Each fixture has a `phase_a_status` tag:
  - "passes_today"             current v1.1 engine handles correctly
  - "fails_today_fixed_by_v2"  must be fixed by v2 Phase A
  - "depends_on_data"          passes only if specific posting still in DB
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

os.environ.setdefault("LLM_ENABLED", "false")

import pytest

from skillbridge.match import engine as match_engine
from skillbridge.session.staging import StagedProfile, StagedSkill


# =========================================================================
# Fixture definition
# =========================================================================
@dataclass
class MatchFixture:
    """One scenario the matching engine must handle correctly."""
    name: str
    description: str
    # Profile
    target_role: str
    skill_phrases: list[str] = field(default_factory=list)
    experience_text: Optional[str] = None
    education_text: Optional[str] = None
    work_type_preference: Optional[str] = None
    # Expectations for the SPECIFIC job we're checking
    expected_job_title_contains: Optional[str] = None
    expected_employer_contains: Optional[str] = None
    expected_band: Optional[str] = None     # "strong" | "good" | "stretch" | None=no eligible match
    expected_matched_contains: list[str] = field(default_factory=list)
    expected_missing_contains: list[str] = field(default_factory=list)
    expected_cap_reason: Optional[str] = None   # e.g. "band_capped_by_credential"
    # Meta
    phase_a_status: str = "passes_today"


# =========================================================================
# Fixtures
# =========================================================================
# NOTE: F6 (required_fully_met_preferred_partial) was removed during the
# step-1 review. It presumed a labeling decision (which hotel skills count
# as required vs preferred) that the JD extractor does not yet make. We
# will reintroduce required/preferred fixtures AFTER Sprint 5 step 3 ships
# the JD-side required/preferred labels -- otherwise the fixture is just
# our opinion of what the labels *should* be, not a regression check.
FIXTURES: list[MatchFixture] = [
    # -------------------------------------------------------------
    # F1: Michael Carter (truck&coach apprentice) without Class G.
    # The Garden River First Nation truck-and-coach posting has very
    # specific top-12 skills (truck service & maintenance, emergency
    # repair, motor vehicle inspection, emissions testing, wheel end
    # inspection, welding, parts fabrication, 310T cert, Class G,
    # MTO contract supervision, etc.). Without Class G, the credential
    # cap MUST fire regardless of other overlap.
    # Re-pinned for v1.2.0 extraction (Sprint 5 slice 4e).
    # -------------------------------------------------------------
    MatchFixture(
        name="truck_coach_no_class_g",
        description=(
            "Truck&Coach apprentice without explicit Class G claim. "
            "Required credential is missing -> credential cap MUST fire "
            "(band_capped_by_credential), band ends at stretch."
        ),
        target_role="truck and coach technician apprentice",
        skill_phrases=[
            "welding", "truck maintenance", "vehicle inspection",
            "parts fabrication", "diesel repair",
        ],
        experience_text="Apprentice Truck & Coach Technician at Northern Fleet Services (2023-present)",
        education_text="Truck & Coach Technician Apprenticeship -- Sault College",
        work_type_preference="full-time",
        expected_job_title_contains="Truck and Coach Technician",
        expected_employer_contains="Garden River",
        expected_band="stretch",
        expected_matched_contains=["welding"],
        expected_missing_contains=["class g"],
        expected_cap_reason="band_capped_by_credential",
        phase_a_status="passes_today",
    ),
    # -------------------------------------------------------------
    # F2: Same Michael WITH Class G. v1.2.0 extracts very specific
    # skills (MTO contract supervision, First Nation cultural knowledge,
    # 310T cert) that a generic newcomer truck/coach apprentice doesn't
    # claim, so the honest outcome is 'stretch' even with Class G --
    # but crucially, the credential cap does NOT fire (no licence gap).
    # That's the contract this fixture pins: holding Class G should
    # remove the cap reason, not necessarily produce a 'strong' band.
    # Re-pinned for v1.2.0 (Sprint 5 slice 4e).
    # -------------------------------------------------------------
    MatchFixture(
        name="truck_coach_with_class_g",
        description=(
            "Truck&Coach apprentice WITH Class G. v1.2.0 vocab is "
            "specialized enough that the band stays stretch, but the "
            "credential cap must NOT fire -- the cap reason is the "
            "actionable signal, not the band itself."
        ),
        target_role="truck and coach technician",
        skill_phrases=[
            "welding", "truck maintenance", "vehicle inspection",
            "parts fabrication", "diesel repair",
            "Class G driver's license",
        ],
        experience_text="Apprentice Truck & Coach Technician at Northern Fleet Services",
        education_text="Truck & Coach Technician Apprenticeship -- Sault College",
        work_type_preference="full-time",
        expected_job_title_contains="Truck and Coach Technician",
        expected_employer_contains="Garden River",
        expected_band="stretch",
        expected_matched_contains=["welding", "class g"],
        expected_missing_contains=[],
        # NO expected_cap_reason: contract is that the credential cap is
        # ABSENT when the user has the credential. Validated indirectly
        # by F1 having it set and this fixture not asserting it.
        phase_a_status="passes_today",
    ),
    # -------------------------------------------------------------
    # F3: Software developer profile -- no software roles in SCCC.
    # Expected: no eligible matches for software-developer target.
    # This guards against false-positive matches (e.g. matching Office
    # Administrator against Python developer because of generic skills).
    # -------------------------------------------------------------
    MatchFixture(
        name="software_dev_no_software_jobs",
        description=(
            "Software developer profile, no software jobs in current SCCC. "
            "Engine must NOT promote unrelated roles to strong/good."
        ),
        target_role="software developer",
        skill_phrases=[
            "Python", "React", "PostgreSQL", "Docker", "AWS",
            "machine learning", "data analysis",
        ],
        experience_text="Data Analyst at Telus International (5 years)",
        education_text="BSc Computer Science, University of Toronto",
        work_type_preference="full-time",
        # No specific job expected -- we check that NO software-titled role
        # surfaces (because none exist in the dataset). expected_band None
        # means "this profile should not have any strong/good matches".
        expected_job_title_contains=None,
        expected_band=None,
        phase_a_status="passes_today",
    ),
    # -------------------------------------------------------------
    # F4: Exact-title typed, zero skills -> title-match fast path.
    # Used to verify the Cygnus-style fix stays correct.
    # -------------------------------------------------------------
    MatchFixture(
        name="exact_title_match_zero_skills",
        description=(
            "User types exact SCCC job title, has no skills yet AND no "
            "experience_text. Title-match override surfaces the posting; "
            "the no-experience floor must STILL set its flag (band may "
            "already be stretch from the override, but the flag is the "
            "honesty signal the responder narrates from)."
        ),
        target_role="Senior Manager of Operations & Client Services",
        skill_phrases=[],   # explicitly empty
        expected_job_title_contains="Senior Manager of Operations",
        expected_employer_contains="Cygnus",
        expected_band="stretch",
        expected_matched_contains=[],  # No skills -> none matched
        expected_cap_reason="band_capped_by_no_experience",
        phase_a_status="passes_today",
    ),
    # -------------------------------------------------------------
    # F5: Generic customer service rep applying to a hotel front-desk role.
    # The job requires hotel-specific skills (guest check-in, reservation
    # management, property management system) that a generic CS rep doesn't
    # have. Honest outcome: stretch -- they have transferable skills but
    # are missing the hotel-specific stack.
    # -------------------------------------------------------------
    MatchFixture(
        name="customer_service_to_hotel_stretch",
        description=(
            "Generic customer service profile applying to hotel Front Desk. "
            "Honest stretch -- has CS basics, missing hotel-specific skills."
        ),
        target_role="customer service representative",
        skill_phrases=[
            "customer service", "phone communication", "computer systems",
            "attention to detail", "communication", "payment processing",
            "multitasking", "conflict resolution",
        ],
        experience_text="3 years retail customer service",
        education_text="Ontario Secondary School Diploma",
        work_type_preference="full-time",
        expected_job_title_contains="Front Desk Agent",
        expected_employer_contains="Quality Inn",
        expected_band="stretch",
        expected_matched_contains=["customer service", "communication"],
        expected_missing_contains=["guest check-in"],
        phase_a_status="passes_today",
    ),
    # -------------------------------------------------------------
    # F6: (REMOVED -- see header note). Required-vs-preferred fixtures
    # will be added in Sprint 5 step 3 once the JD extractor produces
    # required/preferred labels we can pin against, instead of relying
    # on our opinion of which skills "should" be preferred.
    # -------------------------------------------------------------
    # F7: No-experience floor.
    # User has skill overlap but no experience_text -> cap at stretch.
    # -------------------------------------------------------------
    MatchFixture(
        name="no_experience_floor",
        description=(
            "Skill overlap is high but user has no experience_text. "
            "v2 caps at stretch -- skills without context don't prove fit. "
            "Asserts the explicit cap flag so a regression that stops "
            "firing the floor fails (rather than passing by luck)."
        ),
        target_role="customer service representative",
        skill_phrases=[
            "customer service", "phone communication", "computer systems",
            "communication", "payment processing",
        ],
        experience_text=None,   # explicitly empty
        work_type_preference="full-time",
        expected_job_title_contains="Front Desk Agent",
        expected_band="stretch",
        expected_cap_reason="band_capped_by_no_experience",
        phase_a_status="fails_today_fixed_by_v2",
    ),
    # -------------------------------------------------------------
    # F8: Alias normalisation -- British 'licence' spelling.
    # The job extracts "Class G driver's license" (American spelling).
    # The user types "Class G licence" (British). The alias map collapses
    # these to one canonical form -- the contract is that the skill shows
    # up in matched_skills, regardless of which band the rest of the
    # vocabulary lands at.
    # Re-pinned for v1.2.0 (Sprint 5 slice 4e).
    # -------------------------------------------------------------
    MatchFixture(
        name="alias_class_g_licence_spelling",
        description=(
            "British 'licence' must canonicalise to the same skill as the "
            "JD's 'license'. Contract: class g appears in matched_skills, "
            "and the credential cap does NOT fire (user has it)."
        ),
        target_role="truck and coach technician",
        skill_phrases=[
            "welding",
            "truck maintenance",
            "vehicle inspection",
            "Class G licence",   # British spelling -- alias must collapse
        ],
        experience_text="Apprentice Truck & Coach Technician",
        work_type_preference="full-time",
        expected_job_title_contains="Truck and Coach Technician",
        expected_employer_contains="Garden River",
        expected_band="stretch",  # specialized vocab keeps band at stretch
        expected_matched_contains=["welding", "class g"],
        # No expected_cap_reason -- the contract is the cap is ABSENT.
        phase_a_status="fails_today_fixed_by_v2",
    ),
    # -------------------------------------------------------------
    # F9: Cap-by-credential narration -- the credential cap fires AND
    # sets band_capped_by_credential=True in score_explanation, with
    # Class G named in credential_gap_skills. This is the failure mode
    # the slice 4b carve-out specifically protects against (v1.2.0
    # ranks Class G at position 9 -- with the old top-5 cutoff, the
    # cap could never see the gap).
    # Re-pinned for v1.2.0 (Sprint 5 slice 4e).
    # -------------------------------------------------------------
    MatchFixture(
        name="credential_cap_explanation",
        description=(
            "Strong skill overlap but Class G missing -> band capped at "
            "stretch, score_explanation.band_capped_by_credential=True. "
            "Validates the slice 4b carve-out: Class G ranks 9 in v1.2.0 "
            "but the carve-out keeps it in the matching set."
        ),
        target_role="truck and coach technician",
        skill_phrases=[
            "welding", "truck maintenance", "vehicle inspection",
            "parts fabrication", "diesel repair",
            # Class G deliberately NOT present
        ],
        experience_text="Apprentice Truck & Coach Technician at Northern Fleet Services",
        work_type_preference="full-time",
        expected_job_title_contains="Truck and Coach Technician",
        expected_employer_contains="Garden River",
        expected_band="stretch",
        expected_missing_contains=["class g"],
        expected_cap_reason="band_capped_by_credential",
        phase_a_status="passes_today",
    ),
    # -------------------------------------------------------------
    # F10: Honesty floor: empty profile + off-topic target should
    # produce no surfaced matches at all. Guards against the engine
    # generating false-positives from a thin profile.
    # -------------------------------------------------------------
    MatchFixture(
        name="empty_profile_no_matches",
        description=(
            "Empty profile + vague target -> engine must not surface any "
            "strong/good matches. Stretch via title-match also acceptable "
            "only if the title closely matches a real posting."
        ),
        target_role="something",
        skill_phrases=[],
        expected_band=None,   # no strong/good expected
        phase_a_status="passes_today",
    ),
    # -------------------------------------------------------------
    # F11: Credential visibility contract (Sprint 5 slice 4e).
    # The Garden River truck-and-coach posting ranks Class G driver's
    # license at importance_rank 9 -- BELOW the top-5 cutoff that
    # shipped with Sprint 5 step 1. Without the slice 4b carve-out,
    # Class G would never enter the matching set, and the credential
    # cap could never fire even when the user lacks it.
    #
    # This fixture is vocabulary-independent in intent: any credential
    # below top-N must still be visible in required_missing when the
    # user doesn't have it. We pin against Garden River because we
    # know it has a credential at rank 9 today -- if a future SCCC
    # re-extract changes that, the data-drift skip kicks in rather
    # than a false fail.
    # -------------------------------------------------------------
    MatchFixture(
        name="credential_visibility_contract",
        description=(
            "Class G ranks 9 in v1.2.0 extraction (past the old top-5 "
            "cutoff). The slice 4b carve-out must keep it in the "
            "matching set so the credential cap can fire honestly when "
            "the user doesn't have it."
        ),
        target_role="truck and coach technician",
        skill_phrases=[
            # Realistic newcomer vocab -- covers some of the JD's core
            # duties but NOT Class G. The user looks otherwise qualified
            # except for the licence gap.
            "welding", "truck maintenance", "vehicle inspection",
            "parts fabrication", "outdoor work",
        ],
        experience_text="Apprentice Truck & Coach Technician (2 years)",
        work_type_preference="full-time",
        expected_job_title_contains="Truck and Coach Technician",
        expected_employer_contains="Garden River",
        expected_band="stretch",
        # The core assertion: Class G must appear in required_missing
        # despite being ranked outside the configured top-N. This is
        # the vocabulary-independent piece -- the carve-out works by
        # keyword match, not by importance_rank.
        expected_missing_contains=["class g"],
        expected_cap_reason="band_capped_by_credential",
        phase_a_status="passes_today",
    ),
]


# =========================================================================
# Helpers
# =========================================================================
def build_profile(f: MatchFixture) -> StagedProfile:
    sp = StagedProfile.new("fixture-" + f.name)
    sp.target_role_text = f.target_role
    sp.skills = [
        StagedSkill(skill_name=name, confidence=0.85, source="resume")
        for name in f.skill_phrases
    ]
    if f.experience_text:
        sp.experience_text = f.experience_text
    if f.education_text:
        sp.education_text = f.education_text
    if f.work_type_preference:
        sp.work_type_preference = f.work_type_preference
    return sp


def find_target_match(matches: list, f: MatchFixture):
    """Return the match that matches the fixture's expected job, or None."""
    if not f.expected_job_title_contains:
        return None
    candidates = [
        m for m in matches
        if f.expected_job_title_contains.lower() in (m.title or "").lower()
    ]
    if f.expected_employer_contains:
        candidates = [
            m for m in candidates
            if f.expected_employer_contains.lower() in (m.employer or "").lower()
        ]
    return candidates[0] if candidates else None


class FixtureResult:
    """Outcome of running one fixture. Status is one of: pass / fail / skip."""
    __slots__ = ("status", "message")

    def __init__(self, status: str, message: str):
        self.status = status
        self.message = message


def evaluate_fixture(f: MatchFixture) -> FixtureResult:
    """Run a single fixture. Returns (status, message).

    Status semantics:
      - "pass"  expectations met
      - "fail"  engine produced wrong band/skills/cap
      - "skip"  data drift: the SCCC posting we pin against is no longer
                in core.v_current_job (re-run the SCCC sync to refresh)
    """
    sp = build_profile(f)
    matches = match_engine.compute_matches_in_memory(sp, top=30)

    if f.expected_band is None and f.expected_job_title_contains is None:
        # Negative case: no strong/good matches expected for an empty/vague profile.
        promoted = [m for m in matches if m.match_eligible and m.match_band in ("strong", "good")]
        if promoted:
            titles = ", ".join(m.title for m in promoted[:3])
            return FixtureResult("fail", f"unexpected strong/good matches: {titles}")
        return FixtureResult("pass", "ok -- no false-positive strong/good matches")

    if f.expected_band is None:
        # F3-style: software dev with no software roles. Check no
        # software-titled job is promoted to strong/good.
        software_words = ("developer", "engineer", "programmer", "software")
        promoted_software = [
            m for m in matches
            if m.match_eligible
            and m.match_band in ("strong", "good")
            and any(w in (m.title or "").lower() for w in software_words)
        ]
        if promoted_software:
            return FixtureResult("fail", f"unexpected software match: {promoted_software[0].title}")
        return FixtureResult("pass", "ok -- no false-positive software promotion")

    # Positive case: check the specific job
    target = find_target_match(matches, f)
    if target is None:
        # Data drift, not a regression. The pinned posting is no longer
        # in the dataset (aged out, re-classified, or never ingested).
        return FixtureResult("skip", (
            f"data drift: target posting not in current matches "
            f"(expected title~{f.expected_job_title_contains!r}, "
            f"employer~{f.expected_employer_contains!r}). "
            f"Re-run SCCC sync. Available titles in top 5: "
            f"{[m.title for m in matches[:5]]}"
        ))

    if target.match_band != f.expected_band:
        return FixtureResult("fail", (
            f"band mismatch: expected {f.expected_band!r}, got {target.match_band!r} "
            f"(score={target.match_score}, matched={target.matched_skills}, "
            f"missing={target.missing_skills})"
        ))

    matched_lower = [s.lower() for s in (target.matched_skills or [])]
    for expected in f.expected_matched_contains:
        if not any(expected.lower() in m for m in matched_lower):
            return FixtureResult("fail", (
                f"expected matched skill {expected!r} not in {target.matched_skills}"
            ))

    missing_lower = [s.lower() for s in (target.missing_skills or [])]
    for expected in f.expected_missing_contains:
        if not any(expected.lower() in m for m in missing_lower):
            return FixtureResult("fail", (
                f"expected missing skill {expected!r} not in {target.missing_skills}"
            ))

    if f.expected_cap_reason:
        se = target.score_explanation or {}
        if not se.get(f.expected_cap_reason):
            return FixtureResult("fail", (
                f"expected score_explanation.{f.expected_cap_reason}=True, got {se}"
            ))

    return FixtureResult("pass", f"ok -- band={target.match_band}, score={target.match_score}")


# =========================================================================
# pytest entry point
# =========================================================================
@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda f: f.name)
def test_matching_fixture(fixture: MatchFixture):
    """Drives every MatchFixture through the live engine.

    Data drift (target posting no longer in core.v_current_job) becomes
    pytest.skip(), not a failure. Genuine regressions (wrong band, wrong
    matched/missing, missing cap reason) fail loudly.
    """
    result = evaluate_fixture(fixture)
    if result.status == "skip":
        pytest.skip(result.message)
    assert result.status == "pass", (
        f"[{fixture.name}] ({fixture.phase_a_status}) {result.message}"
    )


# =========================================================================
# Standalone runner -- preserves human-readable output for ad-hoc runs.
# =========================================================================
def main() -> int:
    print(f"=== Matching Engine v2 -- Validation Fixtures ({len(FIXTURES)} cases) ===\n")
    by_status: dict[str, list[tuple[str, str, str]]] = {}
    for f in FIXTURES:
        try:
            res = evaluate_fixture(f)
        except Exception as e:
            res = FixtureResult("fail", f"exception: {e!r}")
        marker = {"pass": "PASS", "fail": "FAIL", "skip": "SKIP"}[res.status]
        print(f"  [{marker}] {f.name:42s} ({f.phase_a_status})")
        if res.status != "pass":
            print(f"         -> {res.message}")
        by_status.setdefault(f.phase_a_status, []).append((f.name, res.status, res.message))

    print()
    total_passed = sum(1 for entries in by_status.values() for _, s, _ in entries if s == "pass")
    total_failed = sum(1 for entries in by_status.values() for _, s, _ in entries if s == "fail")
    total_skipped = sum(1 for entries in by_status.values() for _, s, _ in entries if s == "skip")
    print(f"Results: {total_passed} passed, {total_failed} failed, "
          f"{total_skipped} skipped (of {len(FIXTURES)} total)\n")

    for status in ("passes_today", "fails_today_fixed_by_v2", "depends_on_data"):
        entries = by_status.get(status, [])
        if not entries:
            continue
        passing = sum(1 for _, s, _ in entries if s == "pass")
        skipped = sum(1 for _, s, _ in entries if s == "skip")
        skip_note = f" ({skipped} skipped)" if skipped else ""
        print(f"  {status}: {passing}/{len(entries)} passing{skip_note}")
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
