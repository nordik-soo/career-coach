"""AR-9.feat.coach-tiers CP1 — tier evidence projections.

Step 7 introduces the AdjacentJob projection used by the
"Sideways move — same skills, different angle" tier. Strong-tier and
Stretch-tier projections land in step 8.

Design discipline (locked v5 + adjacency enrichment correction):
  - No new scoring, no new ranking, no new query — the function takes
    the OUTPUT of the existing `accept_candidates` strict-AND gate.
  - No new reason vocabulary — `why_adjacent` reuses the closed set
    {"same_noc_minor_group", "skill_evidence"} derived the same way
    `handler.py` already derives it (target_noc[:4] vs job_noc[:4]).
  - Alignment construction goes through the shared `build_skill_alignment`
    helper. The matcher runs exactly once per job-side requirement.
  - Credential safety is preserved: `accept_candidates` already rejects
    any job where the user fails a required credential, so accepted
    jobs cannot carry missing-credential gaps. The occupational
    `credential_warning_text` is informational only — never treated
    as a gap.
  - Non-credential gaps only.
  - The job URL is validated through the existing `url_policy.validate`
    pipeline before construction; an unvalidated URL becomes None on
    the projection, never a raw string.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

from skillbridge.chat.url_policy import Validated, validate
from skillbridge.match.alignment import SkillAlignment, UserSkillRow
from skillbridge.match.engine import (
    MatchResult,
    _regulated,
    build_skill_alignment,
    is_credential_skill_name,
)


WhyAdjacent = Literal["same_noc_minor_group", "skill_evidence"]


# AR-9.feat.coach-tiers CP1 step 8: closed-vocabulary strength claim
# tokens (v5 lock). The prompt layer reserves specific phrasings per
# token. Builder must NEVER emit anything outside this set.
StrengthClaim = Literal[
    "competitive_match",            # Apply-today, band == strong
    "strongest_current",            # Apply-today, band == good (or strong-with-caveat)
    "close_with_named_gap",         # Worth-a-try, non-credential required gap
    "stretch_with_training_bridge", # Worth-a-try, credential gap with mapped training
    "transferable_lane",            # Sideways-move
]


@dataclass(frozen=True)
class JobFacts:
    """Sourceable job facts (per C3). All fields are pulled raw from
    the job row; no interpretation or paraphrase happens here.
    `posted_days_ago` is the only derived field and is computed
    deterministically from `posted_date`.

    Shared across all three tier records (StrongMatch, StretchMatch,
    AdjacentJob) so the evidence schema stays consistent.
    """
    posted_date: date | None
    posted_days_ago: int | None
    location: str | None
    employment_type: str | None
    salary_text: str | None


@dataclass(frozen=True)
class TransferablePair:
    """One user-skill ↔ adjacent-requirement mapping projected from
    a SkillAlignment record. Used by the prompt's Sideways-move
    paragraph rule.
    """
    user_skill: str
    applies_to: str          # the adjacent job's requirement text
    stage: Literal["exact", "fuzzy", "semantic"]


@dataclass(frozen=True)
class NonBlockingGap:
    """A preferred-skill miss attached to an Apply-today record.

    `material` is True for AT MOST ONE entry per StrongMatch per the
    locked v5 rule: the highest-priority preferred miss when all
    required skills are covered. The engine returns missing skills in
    importance order (job_skills ORDER BY importance_rank), so the
    first preferred miss IS the highest-priority one.
    """
    job_requirement: str
    material: bool


@dataclass(frozen=True)
class TrainingOption:
    """One training resource mapped to a specific gap.

    Sourced from the existing training registry / training_by_job
    payload the responder already receives. URL is stored as Validated
    (not SanitizedURL) for the same dependency-direction reasons as
    AdjacentJob.url — see Fix 1 (post-step-7 review).
    """
    provider: str
    title: str
    url: Validated | None
    format: Literal["online", "in-person", "hybrid"] | None
    duration_text: str | None


@dataclass(frozen=True)
class PrioritizedGap:
    """A required-skill miss on a Worth-a-try record, with priority
    and any training that maps to it.

    `priority` is 1-indexed in the order the matcher emitted misses
    (which IS importance order — see engine.py _fetch_job_skills ORDER
    BY importance_rank).

    `blocker` is True only for credential gaps (is_credential_skill_name).
    Worth-a-try inclusion rule (post-step-8 Fix 2): a job with
    credential gaps is included ONLY when EVERY credential gap has at
    least one actionable (URL-validated) training option. A single
    credential without actionable training keeps the job out of
    Worth a try entirely.
    """
    job_requirement: str
    category: Literal["required", "preferred"]
    priority: int
    blocker: bool
    training_options: tuple[TrainingOption, ...]


@dataclass(frozen=True)
class StrongMatch:
    """Apply-today tier record.

    Eligibility (enforced centrally in build_tiered_evidence):
      - match_eligible is True
      - match_band in {"strong", "good"}
      - required_missing is empty (so no missing required credentials
        either — credentials are a subset of required)
    """
    job_id: str
    title: str
    employer: str | None
    location: str | None
    noc_code: str | None
    url: Validated | None
    job_facts: JobFacts
    skill_alignment: tuple[SkillAlignment, ...]
    non_blocking_gaps: tuple[NonBlockingGap, ...]
    credential_warning_text: str | None
    strength_claim_text: StrengthClaim


@dataclass(frozen=True)
class StretchMatch:
    """Worth-a-try tier record.

    Eligibility (enforced centrally in build_tiered_evidence):
      - match_eligible is True
      - match_band in {"good", "stretch"}
      - has at least one required gap
      - if the ONLY required gaps are credentials, at least one of
        them must have training in `training_by_job` (otherwise the
        job is not actionable and is excluded entirely)
    """
    job_id: str
    title: str
    employer: str | None
    location: str | None
    noc_code: str | None
    url: Validated | None
    job_facts: JobFacts
    skill_alignment: tuple[SkillAlignment, ...]
    prioritized_gaps: tuple[PrioritizedGap, ...]
    credential_warning_text: str | None
    strength_claim_text: StrengthClaim


@dataclass(frozen=True)
class TieredEvidence:
    """The tier evidence package (post scoring-v6, 2026-06-17).

    Now FOUR direct-target tiers + the adjacency tier:
      - apply_today    : Strong + Good labels (top tier under v6 naming
                          is split into "Strong match" + "Good match"
                          but both share the apply_today slot for now;
                          response heading rename happens in Step 4).
      - worth_a_try    : Stretch label
      - explore_later  : Explore-later label (NEW in v6; previously
                          hidden by responder's eligible-only-low
                          branch — now surfaced as a fourth tier so
                          users see a panorama of what's available).
      - sideways_move  : NOC-adjacent matches (unchanged; later
                          slices may wire this as CP5).

    Tier exclusivity is invariant: a job_id appears in at most one
    of these slots. Enforced centrally in `build_tiered_evidence`.
    Empty tiers are empty tuples — never filled with placeholders.

    `explore_later` defaults to () so existing TieredEvidence
    constructors (tests, fixtures, etc.) keep compiling without
    modification.
    """
    apply_today: tuple[StrongMatch, ...]
    worth_a_try: tuple[StretchMatch, ...]
    sideways_move: tuple[AdjacentJob, ...]
    explore_later: tuple[StretchMatch, ...] = ()


@dataclass(frozen=True)
class AdjacentJob:
    """One enriched adjacency record for the Sideways-move tier.

    Constructed only from jobs that survived the existing
    `accept_candidates` strict-AND gate. The four invariants enforced
    by `accept_candidates` carry through here:
      - has_usable_skill_evidence(staged) was True;
      - the job has at least one required non-credential skill;
      - the user satisfies every required credential check;
      - mean coverage and anchor-only transferable strength both
        cleared their floors.

    Fields:
      job_id, title, employer, location, noc_code
                — raw values from the job row.
      url        — Validated (raw structural validation result from
                   `url_policy.validate`). None when validation
                   rejected the raw value. The view layer projects
                   Validated → SanitizedURL at render time; storing
                   the raw Validated here keeps the matcher/evidence
                   layer free of any url_views.py dependency, so
                   url_views.py can later import AdjacentJob without
                   a circular import.
      job_facts  — sourceable subset of the job row (see JobFacts).
      skill_alignment
                — per-requirement attribution, built via the shared
                   helper. Drives the prompt's Sideways paragraph.
      transferable_pairs
                — UI/prompt projection of skill_alignment.
      important_gaps
                — required NON-CREDENTIAL job_skill names not in
                   skill_alignment. Credentials are never here:
                   accept_candidates already rejected any job where
                   the user fails a required credential; the
                   `credential_warning_text` field is the occupational
                   licensing note instead.
      credential_warning_text
                — None unless the JOB'S OWN noc_code maps to a
                   regulated occupation row. NEVER derived from the
                   user's target_role_text — a target-role-based
                   fallback would attach the wrong licensing warning
                   to an adjacent job whose NOC differs from the
                   target. Always informational; the prompt surfaces
                   it as a licensing note, never as a gap.
      why_adjacent
                — reused closed vocabulary from handler.py's existing
                   computation. "same_noc_minor_group" when the
                   target_noc minor group equals the job's;
                   "skill_evidence" otherwise.
      strength_claim_text
                — always "transferable_lane" for AdjacentJob. Kept
                   here so the three tier records share the same
                   strength_claim_text shape; the prompt layer reads
                   it uniformly across tiers.
    """
    job_id: str
    title: str
    employer: str | None
    location: str | None
    noc_code: str | None
    url: Validated | None
    job_facts: JobFacts
    skill_alignment: tuple[SkillAlignment, ...]
    transferable_pairs: tuple[TransferablePair, ...]
    important_gaps: tuple[str, ...]
    credential_warning_text: str | None
    why_adjacent: WhyAdjacent
    strength_claim_text: StrengthClaim


# ----------------------------------------------------------------- helpers
def _validate_url(raw: object) -> Validated | None:
    """Run a raw URL string through the existing structural validator.
    Returns Validated when the URL passes; None on any failure (None
    input, non-str input, empty string, or any Violation).

    Dependency-direction note: this module deliberately stops at
    Validated and does NOT promote to SanitizedURL. Storing Validated
    here keeps the evidence layer free of any chat/url_views.py
    import, so step 9 can have url_views.py consume AdjacentJob
    without creating a circular dependency. The view layer projects
    Validated → SanitizedURL at render time.
    """
    if not isinstance(raw, str) or not raw:
        return None
    result = validate(raw)
    if isinstance(result, Validated):
        return result
    return None


def _why_adjacent(target_noc: str | None, job_noc: str | None) -> WhyAdjacent:
    """Reuse handler.py's existing computation byte-for-byte:
    same_noc_minor_group when target_noc[:4] == job_noc[:4] (both
    non-empty), else skill_evidence.
    """
    target_minor = (target_noc or "")[:4]
    job_minor = (job_noc or "")[:4]
    if target_minor and job_minor and target_minor == job_minor:
        return "same_noc_minor_group"
    return "skill_evidence"


def _posted_days_ago(posted_date: object) -> int | None:
    if not isinstance(posted_date, date):
        return None
    delta = (date.today() - posted_date).days
    return max(0, delta)


def _important_non_credential_gaps(
    job_skills: list[dict],
    matched_requirement_names: set[str],
) -> tuple[str, ...]:
    """Return required, non-credential job_skill names that did NOT
    appear in the alignment (i.e. weren't matched). Order preserves
    the input job_skills order so prompt prose is reproducible.
    """
    out: list[str] = []
    for s in job_skills:
        if not isinstance(s, dict):
            continue
        # "required" bucket only: credentials sit in credential_warning_text,
        # preferred-skill gaps are not surfaced as "important".
        raw = (s.get("skill_type") or "").strip().lower()
        if raw == "preferred":
            continue
        name = s.get("skill_name")
        if not isinstance(name, str) or not name.strip():
            continue
        if is_credential_skill_name(name):
            continue
        if name in matched_requirement_names:
            continue
        out.append(name)
    return tuple(out)


def _credential_warning_text(job_noc: object) -> str | None:
    """Look up the regulated-occupation row by THE ADJACENT JOB'S OWN
    NOC and project the existing warning text. Same wording the engine
    produces.

    Adjacency-specific rule (Fix 2, post-step-7 review): this helper
    does NOT fall back to the user's `target_role_text`. The engine's
    `_regulated(noc, target_role)` falls back to a similarity search
    against the regulated_occupation table when noc is absent — that
    fallback is correct for the lead-result path (it's the user's own
    target) but WRONG for adjacency, where it would attach the user's
    target occupation's licensing warning to a different adjacent job
    whose NOC differs. The fix is structural: never pass
    target_role_text through here. If `job_noc` is absent, return None.

    Graceful: catches DB-unavailable exceptions and returns None;
    credential warnings are informational, not load-bearing.
    """
    noc = job_noc if isinstance(job_noc, str) and job_noc else None
    if not noc:
        return None
    try:
        row = _regulated(noc, None)
    except Exception:
        return None
    if not row:
        return None
    return (
        f"This occupation may require Canadian/Ontario licensing or certification "
        f"({row['regulator_name']}). {row.get('licensing_note') or ''}"
    ).strip()


# ------------------------------------------------------------ public entry
def enrich_accepted_adjacency_jobs(
    accepted: list[dict],
    user_rows: list[UserSkillRow],
    user_skill_ids: set[str],
    user_skill_names: set[str],
    user_skill_names_canon: set[str],
    *,
    target_noc: str | None,
    user_embeddings_matrix=None,
    exclude_job_ids: set[str] | frozenset[str] = frozenset(),
) -> list[AdjacentJob]:
    """Project `accept_candidates` output into AdjacentJob records.

    Reuses `build_skill_alignment` for attribution. Does NOT re-run
    scoring, boosts, hard gates, or any non-alignment matcher pass.
    Filters out any `job_id` in `exclude_job_ids` so the Sideways-move
    tier stays exclusive of Strong and Worth-a-try job IDs (CC-5).

    `target_noc` is used ONLY for the `why_adjacent` derivation
    (minor-group comparison). It is NEVER passed to the credential-
    warning lookup — that path is keyed strictly on each adjacent
    job's own NOC. See `_credential_warning_text` for the rationale.

    Returned list preserves the input order of `accepted`. Caller (the
    tier evidence builder) is responsible for any final cap or
    ordering against the rest of the tier set.
    """
    out: list[AdjacentJob] = []
    seen_job_ids: set[str] = set()
    for job in accepted:
        if not isinstance(job, dict):
            continue
        job_id = job.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            continue
        if job_id in exclude_job_ids:
            continue
        # Fix 3 (step-8 review): within-tier dedup. A duplicate accepted
        # entry with the same job_id must not produce two AdjacentJob
        # records.
        if job_id in seen_job_ids:
            continue
        seen_job_ids.add(job_id)
        title = job.get("title")
        if not isinstance(title, str) or not title.strip():
            # accept_candidates' contract guarantees a title; this is
            # defense-in-depth for forged inputs and ill-formed rows.
            continue

        job_skills = job.get("skills") or []
        if not isinstance(job_skills, list):
            job_skills = []

        alignments, _classifications = build_skill_alignment(
            job_skills,
            user_rows,
            user_skill_ids,
            user_skill_names,
            user_skill_names_canon,
            user_embeddings_matrix=user_embeddings_matrix,
            job_skill_embeddings=None,
        )

        matched_req_names = {a.job_requirement for a in alignments}
        important_gaps = _important_non_credential_gaps(
            job_skills, matched_req_names
        )

        transferable_pairs = tuple(
            TransferablePair(
                user_skill=a.user_skill,
                applies_to=a.job_requirement,
                stage=a.stage,
            )
            for a in alignments
        )

        job_noc = job.get("noc_code")
        facts = JobFacts(
            posted_date=job.get("posted_date") if isinstance(job.get("posted_date"), date) else None,
            posted_days_ago=_posted_days_ago(job.get("posted_date")),
            location=job.get("location") if isinstance(job.get("location"), str) else None,
            employment_type=job.get("employment_type") if isinstance(job.get("employment_type"), str) else None,
            salary_text=job.get("salary_text") if isinstance(job.get("salary_text"), str) else None,
        )

        out.append(AdjacentJob(
            job_id=job_id,
            title=title,
            employer=job.get("employer") if isinstance(job.get("employer"), str) else None,
            location=job.get("location") if isinstance(job.get("location"), str) else None,
            noc_code=job_noc if isinstance(job_noc, str) else None,
            url=_validate_url(job.get("url")),
            job_facts=facts,
            skill_alignment=tuple(alignments),
            transferable_pairs=transferable_pairs,
            important_gaps=important_gaps,
            credential_warning_text=_credential_warning_text(job_noc),
            why_adjacent=_why_adjacent(target_noc, job_noc),
            strength_claim_text="transferable_lane",
        ))
    return out


# =========================================================================
# Step 8 — tier evidence builder
# =========================================================================
_STRONG_CAP_DEFAULT = 3
_STRETCH_CAP_DEFAULT = 2
_EXPLORE_LATER_CAP_DEFAULT = 2  # 2026-06-17 scoring-v6
_ADJACENT_CAP_DEFAULT = 3

# scoring-v6 (2026-06-17): 30% floor below which postings are filtered
# out of all tier surfaces. Matches scoring below this threshold are
# more noise than signal for the user — surfacing them violates the
# "user always gets something meaningful" principle. Anything between
# 30% and the engine's "stretch" band (40%) lands in the new
# Explore-later tier instead of being hidden, so the principle holds.
_MATCH_VISIBILITY_FLOOR = 0.30

_VALID_TRAINING_FORMATS: frozenset[str] = frozenset({"online", "in-person", "hybrid"})


def _job_facts_from_match(result: MatchResult) -> JobFacts:
    """Project a MatchResult into the shared JobFacts shape.

    MatchResult.posted_date is already a date|None from the engine.
    Other fields come from the raw job row pulled through CP1's
    MatchResult extensions (employment_type, salary_text).
    """
    return JobFacts(
        posted_date=result.posted_date,
        posted_days_ago=_posted_days_ago(result.posted_date),
        location=result.location,
        employment_type=result.employment_type,
        salary_text=result.salary_text,
    )


def _split_required_missing(required_missing: list[str]) -> tuple[list[str], list[str]]:
    """Partition a required_missing list into (credential, non-credential)
    while preserving the engine's importance-order positions in both
    sublists. Used by both the Worth-a-try filter and the strength-claim
    classifier.
    """
    cred: list[str] = []
    non_cred: list[str] = []
    for name in required_missing or []:
        if is_credential_skill_name(name):
            cred.append(name)
        else:
            non_cred.append(name)
    return cred, non_cred


def _required_missing_or_none(result: MatchResult) -> list[str] | None:
    """Fail-closed reader for `score_explanation.required_missing`.

    Returns the list ONLY when score_explanation is a dict and
    required_missing is actually present as a list. Otherwise None —
    callers must treat None as "cannot classify; reject."

    Absence vs empty matters: an explicitly empty list means the
    matcher confirmed no required gaps; an absent key means the
    matcher did not report. Step-8 fix 1: tier filters fail closed
    on the latter.
    """
    if not isinstance(result.score_explanation, dict):
        return None
    rm = result.score_explanation.get("required_missing")
    if not isinstance(rm, list):
        return None
    return rm


MatchLabel = Literal["strong", "good", "stretch", "explore_later"]


def _classify_match_label(
    result: MatchResult,
    training_for_job: list[dict] | None,
) -> MatchLabel | None:
    """3-signal classifier (2026-06-17, scoring-v6 LOCKED).

    Replaces the v5 two-layer model (band-only `match_band` + a
    separate gap-presence filter in `_is_apply_today` / `_is_worth_a_try`).
    Reads three signals — score band, blocker count, learnable-gap
    count — and returns one of four labels (or None when the match
    should not be surfaced at all).

    Inputs:
      - score band:       result.match_band (from MATCH.band_* thresholds:
                          strong>=0.75, good>=0.60, stretch>=0.40, else low)
      - blocker count:    count of credential gaps in required_missing
                          (is_credential_skill_name == True)
      - learnable count:  count of non-credential gaps in required_missing

    Decision flow (top-down — first matching rule wins):
      0.  Filtered (returns None):
          - match_eligible is False
          - match_score < 0.30 (the visibility floor)
          - score_explanation.required_missing is absent (fail closed,
            same as v5 _is_apply_today / _is_worth_a_try)
          - credential-only gap profile AND no actionable training for
            any cred gap (preserves v5 "actionable nothing" guard:
            a job with no apply path AND no training path offers the
            user nothing; surfacing it violates the principle)

      1.  band == "low" (0.30 <= score < 0.40)        → explore_later
      2.  learnable_count >= 5                         → explore_later
      3.  blocker_count >= 2                           → explore_later
      4.  band == "stretch"                            → stretch
      5.  band in {strong,good} AND learnable in 3..4  → stretch
      6.  band in {strong,good} AND blocker_count == 1 → stretch
      7.  band == "strong" (no blocker, <=2 learnable) → strong
      8.  band == "good"   (no blocker, <=2 learnable) → good

    Trade-off note (recommendation (i), 2026-06-17): all credential
    gaps count as blockers regardless of training availability. The
    spec calls one of these an "achievable blocker" (demoting Strong
    → Stretch) and two-or-more "stacked blockers" (demoting to
    Explore-later). A future slice can introduce an "achievable"
    distinction (e.g., training_options present → achievable). For
    now the simpler rule lets us ship the classifier without adding
    a new heuristic.
    """
    if not result.match_eligible:
        return None
    if result.match_score < _MATCH_VISIBILITY_FLOOR:
        return None

    required_missing = _required_missing_or_none(result)
    if required_missing is None:
        return None  # fail closed (matches v5 predicate behavior)

    cred_gaps, non_cred_gaps = _split_required_missing(required_missing)

    # "Actionable nothing" filter (v5 carry-forward): a credential-only
    # gap profile with no actionable training maps to no path forward.
    # Surfacing such a posting violates the user-always-gets-something
    # principle (no apply path AND no training path).
    if cred_gaps and not non_cred_gaps:
        if not _all_credentials_have_training(cred_gaps, training_for_job):
            return None

    blocker_count = len(cred_gaps)
    learnable_count = len(non_cred_gaps)

    if result.match_band == "low":
        return "explore_later"
    if learnable_count >= 5:
        return "explore_later"
    if blocker_count >= 2:
        return "explore_later"
    if result.match_band == "stretch":
        return "stretch"
    if learnable_count >= 3:
        return "stretch"
    if blocker_count == 1:
        return "stretch"
    if result.match_band == "strong":
        return "strong"
    if result.match_band == "good":
        return "good"

    return None  # defensive — should be unreachable given band coverage


def _is_apply_today(
    result: MatchResult,
    training_for_job: list[dict] | None = None,
) -> bool:
    """Apply-today admission (post scoring-v6, 2026-06-17).

    Delegates to `_classify_match_label`. Apply-today now means the
    classifier returned "strong" or "good" — i.e., high or mid score
    band AND no blockers AND <=2 learnable gaps. Replaces the v5
    rule (`band in {strong,good} AND required_missing == []`) — the
    old rule demoted ANY match with a gap, even a single learnable
    one. Per Nazmul (2026-06-17): a real coach treats a 9/10 match
    with one learnable gap as "go apply, here's a heads-up about X",
    not "Worth a try."
    """
    return _classify_match_label(result, training_for_job) in ("strong", "good")


def _all_credentials_have_training(
    cred_gaps: list[str],
    training_for_job: list[dict] | None,
) -> bool:
    """Fix 2 + step-12 review High: every credential gap must have at
    least one ACTIONABLE training entry. A training entry counts as
    actionable when:
      - `for_skill` is a string (defensive — non-string values are
        skipped without raising, fixing the AttributeError surface
        the review caught);
      - the URL passes `_validate_url` (rejecting ftp://, malformed,
        and otherwise non-https values). Without this check, an
        invalid URL would later project to `url=None` in
        `_training_options_for_gap`, but the credential filter would
        already have admitted the job under the false belief that
        training was available — producing a misleading
        `stretch_with_training_bridge` strength claim.

    Returns True when `cred_gaps` is empty (vacuous) and when every
    credential gap is covered by at least one actionable entry.
    """
    if not cred_gaps:
        return True
    training_skills: set[str] = set()
    for t in (training_for_job or []):
        if not isinstance(t, dict):
            continue
        for_skill = t.get("for_skill")
        if not isinstance(for_skill, str):
            continue
        if _validate_url(t.get("url")) is None:
            continue
        training_skills.add(for_skill.lower())
    return all(c.lower() in training_skills for c in cred_gaps)


def _is_worth_a_try(
    result: MatchResult,
    training_for_job: list[dict] | None,
) -> bool:
    """Worth-a-try admission (post scoring-v6, 2026-06-17).

    Delegates to `_classify_match_label`. Worth-a-try now means the
    classifier returned "stretch" — i.e., stretch-band match, OR
    high/mid band with 3-4 learnable gaps, OR high/mid band with one
    credential blocker.

    Behavioral diff vs v5:
      - A strong-band match with required_missing==[] previously was
        Apply-today; now (still) Apply-today (label "strong"). No
        change.
      - A strong-band match with 1 learnable gap previously was
        Worth-a-try; now it's Apply-today ("strong" label). The
        classifier admits up to 2 learnable gaps in the top tier.
      - A strong-band match with 5+ learnable gaps previously was
        Worth-a-try; now it's Explore-later. Honest demotion: many
        gaps is many gaps, even with a high overall score.
      - A stretch-band match with required gaps previously was
        Worth-a-try; still is. No change.
      - A credential-only gap profile without training is still
        filtered out — the "actionable nothing" guard moved into
        the classifier (returns None).
    """
    return _classify_match_label(result, training_for_job) == "stretch"


def _is_explore_later(
    result: MatchResult,
    training_for_job: list[dict] | None = None,
) -> bool:
    """Explore-later admission (post scoring-v6, 2026-06-17 NEW tier).

    A match is Explore-later when the classifier returned
    "explore_later" — i.e., low-band match (score 0.30-0.39, above
    the visibility floor but below stretch), OR any band with 5+
    learnable gaps, OR any band with 2+ credential blockers.

    Previously the engine's "low" band (<0.40) was hidden entirely
    by the responder's `Eligible-only-low` branch — users never saw
    these matches. Under the user-always-gets-something principle,
    showing them as a clearly-labeled "Explore later" tier is more
    honest: the user gets a panorama of what the engine considered,
    framed appropriately, instead of an opaque "no match."
    """
    return _classify_match_label(result, training_for_job) == "explore_later"


def _strength_claim_for_strong(result: MatchResult) -> StrengthClaim:
    """Apply-today claim:
       band == "strong" → competitive_match
       band == "good"   → strongest_current
    """
    if result.match_band == "strong":
        return "competitive_match"
    return "strongest_current"


def _strength_claim_for_stretch(
    result: MatchResult,
    training_for_job: list[dict] | None,
) -> StrengthClaim:
    """Worth-a-try claim:
       any credential gap → stretch_with_training_bridge
       otherwise          → close_with_named_gap

    The filter `_is_worth_a_try` (post-Fix-2) admits a record only
    when EVERY credential gap has mapped training. So if cred_gaps
    is non-empty here, training is by construction available; the
    claim is `stretch_with_training_bridge`.
    """
    required_missing = _required_missing_or_none(result) or []
    cred_gaps, _ = _split_required_missing(required_missing)
    if cred_gaps:
        return "stretch_with_training_bridge"
    return "close_with_named_gap"


def _non_blocking_gaps_for_strong(
    result: MatchResult,
) -> tuple[NonBlockingGap, ...]:
    """Locked rule: `material=True` for AT MOST ONE entry — the
    highest-priority preferred miss when all required are satisfied.

    Apply-today candidates have required_missing == [] by construction,
    so every entry in `result.missing_skills` is a preferred miss.
    `missing_skills` is engine-emitted in importance order
    (job_skills ORDER BY importance_rank), so the first entry IS the
    highest-priority preferred miss.
    """
    misses = list(result.missing_skills or [])
    if not misses:
        return ()
    return tuple(
        NonBlockingGap(job_requirement=name, material=(i == 0))
        for i, name in enumerate(misses)
    )


def _training_options_for_gap(
    gap_name: str,
    training_for_job: list[dict] | None,
) -> tuple[TrainingOption, ...]:
    """Map a single gap to its training options from `training_for_job`.

    Match is by lowercased `for_skill` to lowercased gap_name. All
    matching entries are returned in input order.
    """
    if not training_for_job:
        return ()
    needle = gap_name.lower()
    out: list[TrainingOption] = []
    for t in training_for_job:
        if not isinstance(t, dict):
            continue
        if (t.get("for_skill") or "").lower() != needle:
            continue
        fmt_raw = t.get("format")
        fmt: Literal["online", "in-person", "hybrid"] | None = (
            fmt_raw if isinstance(fmt_raw, str) and fmt_raw in _VALID_TRAINING_FORMATS
            else None
        )  # type: ignore[assignment]
        duration = t.get("duration_text")
        out.append(TrainingOption(
            provider=t.get("provider") or "",
            title=t.get("title") or "",
            url=_validate_url(t.get("url")),
            format=fmt,
            duration_text=duration if isinstance(duration, str) else None,
        ))
    return tuple(out)


def _prioritized_gaps_for_stretch(
    result: MatchResult,
    training_for_job: list[dict] | None,
) -> tuple[PrioritizedGap, ...]:
    """Project required_missing into PrioritizedGap records. Order
    follows the engine's importance order. `priority` is 1-indexed.
    """
    required_missing = (result.score_explanation or {}).get(
        "required_missing", []
    )
    out: list[PrioritizedGap] = []
    for i, name in enumerate(required_missing or []):
        out.append(PrioritizedGap(
            job_requirement=name,
            category="required",
            priority=i + 1,
            blocker=is_credential_skill_name(name),
            training_options=_training_options_for_gap(name, training_for_job),
        ))
    return tuple(out)


def _project_to_strong(result: MatchResult) -> StrongMatch:
    return StrongMatch(
        job_id=result.job_id,
        title=result.title,
        employer=result.employer,
        location=result.location,
        noc_code=result.noc_code,
        url=_validate_url(result.url),
        job_facts=_job_facts_from_match(result),
        skill_alignment=tuple(result.skill_alignment),
        non_blocking_gaps=_non_blocking_gaps_for_strong(result),
        credential_warning_text=result.credential_warning,
        strength_claim_text=_strength_claim_for_strong(result),
    )


def _project_to_stretch(
    result: MatchResult,
    training_for_job: list[dict] | None,
) -> StretchMatch:
    return StretchMatch(
        job_id=result.job_id,
        title=result.title,
        employer=result.employer,
        location=result.location,
        noc_code=result.noc_code,
        url=_validate_url(result.url),
        job_facts=_job_facts_from_match(result),
        skill_alignment=tuple(result.skill_alignment),
        prioritized_gaps=_prioritized_gaps_for_stretch(result, training_for_job),
        credential_warning_text=result.credential_warning,
        strength_claim_text=_strength_claim_for_stretch(result, training_for_job),
    )


def _in_target_noc_family(
    result_noc: str | None, target_noc: str | None,
) -> bool:
    """Same-NOC-family gate for direct-tier admission (Bug B fix,
    2026-06-15, v5 RELOCKED with this gate).

    The v5 contract originally left Apply-today and Worth-a-try
    NOC-blind, so a user whose target had ZERO local postings could
    still see strong-skill matches from unrelated NOCs in the direct
    tiers. CP4's diagnose contract (which says NO_OPPORTUNITY_FOUND
    when target_posting_count == 0) then disagreed with the user-facing
    surface. The relock: when target_noc is resolved, only jobs sharing
    the 4-digit NOC prefix (Statistics Canada "minor group") are
    eligible for Apply-today / Worth-a-try. Out-of-family jobs that
    pass the adjacency strict-AND gate still surface in Sideways-move
    — that's where cross-NOC moves belong.

    Returns True when:
      - target_noc is None / empty / shorter than 4 chars (target
        unresolved → existing NOC-blind behaviour preserved); OR
      - result_noc is a string ≥ 4 chars AND shares the same first 4
        characters as target_noc.

    Returns False when target is resolved but the job's NOC is missing,
    malformed, or in a different minor group.
    """
    if not isinstance(target_noc, str) or len(target_noc) < 4:
        return True
    if not isinstance(result_noc, str) or len(result_noc) < 4:
        return False
    return result_noc[:4] == target_noc[:4]


def build_tiered_evidence(
    results: list[MatchResult],
    accepted_adjacent: list[dict],
    user_rows: list[UserSkillRow],
    user_skill_ids: set[str],
    user_skill_names: set[str],
    user_skill_names_canon: set[str],
    *,
    training_by_job: dict[str, list[dict]] | None = None,
    target_noc: str | None = None,
    user_embeddings_matrix=None,
    strong_cap: int = _STRONG_CAP_DEFAULT,
    stretch_cap: int = _STRETCH_CAP_DEFAULT,
    explore_later_cap: int = _EXPLORE_LATER_CAP_DEFAULT,
    adjacent_cap: int = _ADJACENT_CAP_DEFAULT,
) -> TieredEvidence:
    """Deterministically partition `results` + `accepted_adjacent` into
    the three coach-tiers package.

    Inputs:
      results            — MatchResult list, ALREADY sorted by the
                            caller in descending score order (the
                            existing engine entrypoints sort their
                            output). The builder preserves that order
                            within each direct tier.
      accepted_adjacent  — output of the existing `accept_candidates`
                            strict-AND gate, ALREADY in adjacency-rank
                            order. The builder preserves that order
                            within Sideways-move.

    Tier rules (locked v5 + step-8 spec):
      Apply today    — band in {"strong","good"} AND required_missing
                       == []  (no missing required credential by
                       construction since credentials are a subset of
                       required).
      Worth a try    — band in {"good","stretch"} AND at least one
                       required gap. If the ONLY required gaps are
                       credentials, at least one must have mapped
                       training in `training_by_job[job_id]`.
      Sideways move  — `accepted_adjacent` enriched via
                       `enrich_accepted_adjacency_jobs`, with job_ids
                       in Apply-today or Worth-a-try excluded.

    Caps + exclusivity:
      - Apply-today is capped at strong_cap (default 3).
      - Worth-a-try is filled from results NOT in Apply-today, capped
        at stretch_cap (default 2).
      - Sideways is filled from accepted_adjacent NOT in
        (Apply-today ∪ Worth-a-try), capped at adjacent_cap (default 3).

    The builder produces NO prompt, NO presentation, NO prose. Empty
    tiers are empty tuples — never filled with placeholders.
    """
    training_map = training_by_job or {}

    # --- Apply today ---
    # Fix 3: within-tier dedup. Duplicate MatchResult rows with the
    # same job_id must not produce two entries in the same tier.
    # Bug B fix (2026-06-15): same-NOC-family gate. Out-of-family jobs
    # never reach Apply-today / Worth-a-try when target_noc is resolved.
    apply_today: list[StrongMatch] = []
    apply_today_ids: set[str] = set()
    for r in results:
        if len(apply_today) >= strong_cap:
            break
        if r.job_id in apply_today_ids:
            continue
        if not _in_target_noc_family(r.noc_code, target_noc):
            continue
        # scoring-v6 (2026-06-17): pass training_for_job so the
        # classifier's actionable-nothing guard has full inputs (a
        # credential-only gap profile w/o training can't be Apply-today
        # under any rule, but the classifier reads training to decide
        # whether to filter the match entirely or demote to a lower tier).
        if not _is_apply_today(r, training_map.get(r.job_id)):
            continue
        apply_today.append(_project_to_strong(r))
        apply_today_ids.add(r.job_id)

    # --- Worth a try ---
    # Fix 3: same within-tier dedup. Also continues to exclude job_ids
    # already in Apply today (cross-tier exclusivity).
    # Bug B fix (2026-06-15): same-NOC-family gate (see Apply today).
    worth_a_try: list[StretchMatch] = []
    worth_a_try_ids: set[str] = set()
    for r in results:
        if r.job_id in apply_today_ids or r.job_id in worth_a_try_ids:
            continue
        if len(worth_a_try) >= stretch_cap:
            break
        if not _in_target_noc_family(r.noc_code, target_noc):
            continue
        training_for_job = training_map.get(r.job_id)
        if not _is_worth_a_try(r, training_for_job):
            continue
        worth_a_try.append(_project_to_stretch(r, training_for_job))
        worth_a_try_ids.add(r.job_id)

    # --- Explore later (NEW, scoring-v6 2026-06-17) ---
    # Picks up matches the classifier labeled "explore_later" — i.e.,
    # low-band (score 0.30-0.39, above visibility floor), or any band
    # with 5+ learnable gaps, or any band with 2+ credential blockers.
    # Excludes job_ids already in Apply-today or Worth-a-try.
    # Reuses _project_to_stretch — Explore-later items share the
    # StretchMatch shape because they carry the same prioritized_gaps
    # and credential_warning_text fields (only the heading differs).
    explore_later: list[StretchMatch] = []
    explore_later_ids: set[str] = set()
    for r in results:
        if r.job_id in apply_today_ids or r.job_id in worth_a_try_ids:
            continue
        if r.job_id in explore_later_ids:
            continue
        if len(explore_later) >= explore_later_cap:
            break
        if not _in_target_noc_family(r.noc_code, target_noc):
            continue
        training_for_job = training_map.get(r.job_id)
        if not _is_explore_later(r, training_for_job):
            continue
        explore_later.append(_project_to_stretch(r, training_for_job))
        explore_later_ids.add(r.job_id)

    # --- Sideways move ---
    excluded = apply_today_ids | worth_a_try_ids | explore_later_ids
    sideways = enrich_accepted_adjacency_jobs(
        accepted_adjacent,
        user_rows,
        user_skill_ids,
        user_skill_names,
        user_skill_names_canon,
        target_noc=target_noc,
        user_embeddings_matrix=user_embeddings_matrix,
        exclude_job_ids=excluded,
    )
    sideways = sideways[:adjacent_cap]

    return TieredEvidence(
        apply_today=tuple(apply_today),
        worth_a_try=tuple(worth_a_try),
        sideways_move=tuple(sideways),
        explore_later=tuple(explore_later),
    )
