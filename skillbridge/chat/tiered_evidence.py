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
    """The three-tier evidence package.

    Tier exclusivity is invariant: a job_id appears in at most one
    of (apply_today, worth_a_try, sideways_move). Enforced centrally
    in `build_tiered_evidence`. Empty tiers are empty tuples — never
    filled with placeholder entries.
    """
    apply_today: tuple[StrongMatch, ...]
    worth_a_try: tuple[StretchMatch, ...]
    sideways_move: tuple[AdjacentJob, ...]


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
_ADJACENT_CAP_DEFAULT = 3

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


def _is_apply_today(result: MatchResult) -> bool:
    """v5 lock:
        band in {"strong", "good"} AND required_missing == []

    Fix 1 (fail closed): a job whose MatchResult lacks an explicit
    `required_missing` list cannot enter Apply Today. We don't admit
    on missing-data assumption.

    required_missing == [] implies no missing required credential
    (credentials are a subset of required skills) — so the three
    locked conditions collapse into these checks.
    """
    if not result.match_eligible:
        return False
    if result.match_band not in ("strong", "good"):
        return False
    required_missing = _required_missing_or_none(result)
    if required_missing is None:
        return False
    return required_missing == []


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
    """v5 lock + step-8 spec + Fix 2 + CP3 step 2 (2026-06-15):
        band in {"strong", "good", "stretch"} AND at least one required
        gap; EVERY missing required credential must have at least one
        mapped training option — including mixed credential/non-
        credential cases. A single credential gap with no training
        makes the job non-actionable.

    The original v5 design excluded band="strong" because Apply today
    was the canonical home for strong-band records. But a strong-band
    record with a non-empty `required_missing` falls between the two
    rules — Apply today requires `required_missing == []`, Worth a try
    used to require `band != "strong"`. That left the record dropped
    from the tier surface and forced the legacy card render.
    Admitting strong-band records with a real gap here keeps the tier
    surface inclusive: the strong overall score doesn't change the
    fact that there's a specific gap to close first.

    Fix 1 (fail closed): missing required_missing list → reject.
    """
    if not result.match_eligible:
        return False
    if result.match_band not in ("strong", "good", "stretch"):
        return False
    required_missing = _required_missing_or_none(result)
    if required_missing is None or not required_missing:
        return False
    cred_gaps, _non_cred_gaps = _split_required_missing(required_missing)
    return _all_credentials_have_training(cred_gaps, training_for_job)


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
        if not _is_apply_today(r):
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

    # --- Sideways move ---
    excluded = apply_today_ids | worth_a_try_ids
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
    )
