"""CP4 — DevelopmentPlan computation.

LOCKED contract: counterfactual-evidence-based development
recommendation. Two hard constraints govern this module:

  Constraint A. No claim of the form "unlocks N jobs" is permitted
  unless an actual counterfactual rescoring on real postings produced
  N tier transitions to Apply today. Counts of gap-name occurrences
  in required_missing lists are NOT evidence of unlocking.

  Constraint B. Planning is not limited to gaps with verified
  training. The best development move is ranked first on impact;
  training availability is attached afterward and controls only what
  actionable resource the responder can offer.

Shadow-only in the first increment: this module produces a
DevelopmentPlan + ShadowTrace per invocation, the handler logs them
as sanitized telemetry, NEITHER is surfaced to the user.

First release:
  - secondary_recommendation is always None (per spec sign-off).
  - CP4 secondary fires only on explicit user development-intent;
    proactive triggering deferred to post-shadow review.
  - Unverified training providers are never surfaced; only verified
    options with validated URLs reach the plan output.

The module accepts the InventoryDiagnosis as input and produces the
DevelopmentPlan + ShadowTrace. It does NOT inspect market state or
re-run inventory checks — the diagnosis is the gate.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Literal

from skillbridge.chat.inventory_diagnosis import InventoryDiagnosis

if TYPE_CHECKING:
    from skillbridge.match.engine import MatchResult
    from skillbridge.session.staging import StagedProfile

log = logging.getLogger(__name__)


# --- Output dataclasses --------------------------------------------------

ResponderAdvisory = Literal[
    "apply_is_primary",
    "development_is_primary",
    "barrier_present_no_move_identified",
]

HonestReason = Literal[
    "verified_route_available",
    "no_verified_option",
]


@dataclass(frozen=True)
class TrainingOption:
    """A verified training option attached to a recommendation.

    Only verified options with non-null URLs reach the
    DevelopmentPlan. Unverified mappings are recorded in
    `unverified_mapping_exists` on TrainingAttachment but never
    surfaced as named providers.
    """
    provider: str
    title: str
    url: str
    type: str | None = None


@dataclass(frozen=True)
class TrainingAttachment:
    """Training resources for a recommendation. Attached AFTER ranking
    (Constraint B). User-facing fields contain only verified options
    with validated URLs."""
    verified_options: tuple[TrainingOption, ...]
    has_actionable_url: bool
    honest_reason: HonestReason
    # Internal-only telemetry; never surfaced as named providers.
    unverified_mapping_exists: bool


@dataclass(frozen=True)
class JobTierTransition:
    """Per-job counterfactual rescoring trace."""
    job_id: str
    current_band: str
    projected_band: str
    current_required_missing_count: int
    projected_required_missing_count: int


@dataclass(frozen=True)
class EvidenceProvenance:
    """Provenance for every numeric claim on a RecommendationCandidate."""
    per_job_evidence: tuple[JobTierTransition, ...]
    counterfactual_method: str  # "reuse_cached_engine_state" | "fresh_per_job_score"
    dataset_version: str | None
    engine_version: str | None


@dataclass(frozen=True)
class RecommendationCandidate:
    """A development move recommendation backed by counterfactual
    evidence. Every list field is sorted deterministically."""
    skill_canonical_name: str
    source_job_ids: tuple[str, ...]
    promoted_job_ids: tuple[str, ...]
    tier_improvement_job_ids: tuple[str, ...]
    blocker_removed_job_ids: tuple[str, ...]
    active_market_frequency: int
    min_importance_rank: int
    training_attachment: TrainingAttachment
    evidence_provenance: EvidenceProvenance


@dataclass(frozen=True)
class EvaluationSummary:
    candidate_gaps_collected_count: int
    candidate_gaps_rescored_count: int
    candidate_gaps_truncated_count: int
    total_rescorings_performed: int
    target_noc_value: str | None
    dataset_version: str | None


@dataclass(frozen=True)
class DevelopmentPlan:
    """The CP4 output. Shadow-only in the first increment."""
    primary_recommendation: RecommendationCandidate | None
    secondary_recommendation: None  # Locked to None in first shadow release.
    evaluation_summary: EvaluationSummary
    responder_advisory: ResponderAdvisory


@dataclass(frozen=True)
class CandidateRankRow:
    """One row of the deterministic ranking trace. Sanitized — carries
    only canonical names and integer scores; no titles, employers, or
    URLs."""
    skill_canonical_name: str
    promoted_count: int
    blocker_removed_count: int
    tier_improvement_count: int
    active_market_frequency: int
    min_importance_rank: int


@dataclass(frozen=True)
class ShadowTrace:
    """Deterministic shadow telemetry. Excludes wall-clock metrics
    (those are emitted as separate runtime telemetry, not part of the
    determinism contract)."""
    invocation_outcome: str  # diagnosis.outcome
    invocation_trigger: str  # diagnosis_primary | user_explicit_request | none
    candidates_considered: tuple[CandidateRankRow, ...]
    candidates_truncated_by_cap: tuple[str, ...]  # canonical names only
    ranking_trace: tuple[CandidateRankRow, ...]
    selected_primary_id: str | None  # canonical name
    selected_secondary_id: None  # locked to None
    dataset_version: str | None
    engine_version: str | None
    total_rescorings_performed: int


# --- Constants -----------------------------------------------------------

# Per-invocation hard cap on counterfactual rescorings (§4 of spec).
# Provisional value; refined during shadow evaluation.
HARD_CAP_TOTAL_RESCORINGS: int = 100


# --- Shadow integration helper -------------------------------------------

# Keyword triggers for user-explicit development intent. Conservative;
# only fires on phrasings that unambiguously ask what to improve. The
# proper classifier integration is deferred to post-shadow.
_DEVELOPMENT_INTENT_KEYWORDS: tuple[str, ...] = (
    "what should i learn",
    "what can i learn",
    "any training",
    "anything i could work on",
    "anything i can work on",
    "what could i improve",
    "what can i improve",
    "how can i improve",
    "what else can i do",
    "what else could i do",
    "any course",
    "any courses",
    "any certification",
)


def _detect_explicit_development_intent(user_message: str | None) -> bool:
    """Conservative substring scan. Returns True iff a development-
    intent phrase is unambiguous. The classifier-based detection is
    a post-shadow upgrade."""
    if not user_message:
        return False
    lower = user_message.lower()
    return any(kw in lower for kw in _DEVELOPMENT_INTENT_KEYWORDS)


def emit_shadow_trace(
    *,
    staged: "StagedProfile",
    user_message: str | None,
    truth_enough_to_match: bool,
    truth_usable_evidence_present: bool,
    engine_completed: bool,
    in_memory_matches: list["MatchResult"],
    skill_adjacent_results: list | None,
    snapshot_usable: bool,
    target_posting_count: int | None,
) -> None:
    """Shadow-only CP4 invocation. Builds the InventoryDiagnosis,
    computes the DevelopmentPlan + ShadowTrace when authorized, and
    logs sanitized telemetry. NEVER modifies user-visible state.

    All exceptions are caught and logged at warning level so the
    existing response flow cannot be affected.

    Sanitized telemetry: no titles, employers, URLs, gap text, or
    prose appear in any log line emitted from this function. Only
    canonical skill names, job IDs (already stable opaque identifiers),
    integer counts, enum values, and dataset/engine versions are
    surfaced.
    """
    try:
        from skillbridge.chat.inventory_diagnosis import diagnose
        diagnosis = diagnose(
            enough_to_match=truth_enough_to_match,
            usable_evidence_present=truth_usable_evidence_present,
            engine_completed=engine_completed,
            snapshot_usable=snapshot_usable,
            direct_match_results=in_memory_matches,
            skill_adjacent_results=skill_adjacent_results or [],
            target_posting_count=target_posting_count,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "cp4_shadow: diagnose raised %s; trace skipped",
            type(exc).__name__,
        )
        return

    # Sanitized diagnosis log: outcome + reason + pillar count only.
    sid = (staged.session_id[:8] if isinstance(staged.session_id, str) else "?")
    log.info(
        "cp4_shadow_diagnosis session=%s outcome=%s reason=%s pillars=%d",
        sid, diagnosis.outcome, diagnosis.reason_code,
        len(diagnosis.pillars_authorized),
    )

    # CP4 runs only on authorized outcomes.
    if diagnosis.outcome not in ("PREPARATION_GAP", "READY_TO_APPLY"):
        return

    explicit_request = _detect_explicit_development_intent(user_message)

    try:
        result = compute_development_plan(
            diagnosis=diagnosis,
            in_memory_matches=in_memory_matches,
            staged=staged,
            user_explicit_development_request=explicit_request,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "cp4_shadow: compute_development_plan raised %s; trace skipped",
            type(exc).__name__,
        )
        return

    if result is None:
        log.info(
            "cp4_shadow_plan session=%s outcome=%s skipped reason=trigger_unmet",
            sid, diagnosis.outcome,
        )
        return

    plan, trace = result
    primary_name = (
        plan.primary_recommendation.skill_canonical_name
        if plan.primary_recommendation is not None else "<null>"
    )
    promoted = (
        len(plan.primary_recommendation.promoted_job_ids)
        if plan.primary_recommendation is not None else 0
    )
    blocker = (
        len(plan.primary_recommendation.blocker_removed_job_ids)
        if plan.primary_recommendation is not None else 0
    )
    improved = (
        len(plan.primary_recommendation.tier_improvement_job_ids)
        if plan.primary_recommendation is not None else 0
    )
    has_training = (
        plan.primary_recommendation.training_attachment.has_actionable_url
        if plan.primary_recommendation is not None else False
    )
    log.info(
        "cp4_shadow_plan session=%s outcome=%s trigger=%s advisory=%s "
        "primary=%s promoted=%d blocker_removed=%d improved=%d "
        "has_verified_training=%s candidates=%d rescored=%d truncated=%d "
        "total_rescorings=%d",
        sid, diagnosis.outcome, trace.invocation_trigger,
        plan.responder_advisory, primary_name,
        promoted, blocker, improved, has_training,
        plan.evaluation_summary.candidate_gaps_collected_count,
        plan.evaluation_summary.candidate_gaps_rescored_count,
        plan.evaluation_summary.candidate_gaps_truncated_count,
        plan.evaluation_summary.total_rescorings_performed,
    )
    for rank_idx, row in enumerate(trace.ranking_trace[:10]):
        log.info(
            "cp4_shadow_rank session=%s rank=%d candidate=%s "
            "promoted=%d blocker=%d improved=%d market=%d importance=%d",
            sid, rank_idx, row.skill_canonical_name,
            row.promoted_count, row.blocker_removed_count,
            row.tier_improvement_count, row.active_market_frequency,
            row.min_importance_rank,
        )


# --- Public entry point --------------------------------------------------

def compute_development_plan(
    *,
    diagnosis: InventoryDiagnosis,
    in_memory_matches: list["MatchResult"],
    staged: "StagedProfile",
    user_explicit_development_request: bool,
    dataset_version: str | None = None,
    engine_version: str | None = None,
) -> tuple[DevelopmentPlan, ShadowTrace] | None:
    """Compute the CP4 plan if authorized; return (plan, trace) or None.

    Returns None when:
      - diagnosis.outcome is not PREPARATION_GAP or READY_TO_APPLY, OR
      - diagnosis.outcome is READY_TO_APPLY AND the user did not
        explicitly request development advice.

    On PREPARATION_GAP, primary_recommendation may still be None when
    no usable canonical gap survives §2 collection (e.g., the barrier
    is in credential_warning_text only, work-type conflict, sparse
    extraction, or the source posting has no required_missing entries
    after soft-trait filtering). responder_advisory is then
    barrier_present_no_move_identified.

    Function is deterministic given inputs.
    """
    # §1 invocation gate.
    if diagnosis.outcome == "PREPARATION_GAP":
        trigger = "diagnosis_primary"
        advisory: ResponderAdvisory = "development_is_primary"
    elif diagnosis.outcome == "READY_TO_APPLY":
        if not user_explicit_development_request:
            return None
        trigger = "user_explicit_request"
        advisory = "apply_is_primary"
    else:
        return None

    # §2 candidate-gap collection from supporting evidence.
    candidates = _collect_candidate_gaps(
        diagnosis=diagnosis, in_memory_matches=in_memory_matches,
    )

    # M3 fix (2026-06-15): populate importance ranks via DB BEFORE
    # truncation so the truncation step can honor the spec's
    # "favor high-importance candidates" rule. Failure here degrades
    # to default-rank truncation (alphabetical), which is fine as a
    # fallback but documented.
    candidates = _populate_min_importance_ranks(candidates)

    # §4 computational-limit truncation.
    candidates, truncated_names = _truncate_to_cap(
        candidates, cap=HARD_CAP_TOTAL_RESCORINGS,
    )

    # §3 counterfactual rescoring per candidate.
    rescored: list[_RescoredCandidate] = []
    total_rescorings = 0
    target_noc_value = (
        staged.target_noc if isinstance(staged.target_noc, str) else None
    )
    for cand in candidates:
        rescored_cand = _counterfactual_rescore(
            candidate_skill=cand.canonical_name,
            source_job_ids=cand.source_job_ids,
            in_memory_matches=in_memory_matches,
            staged=staged,
            target_noc_value=target_noc_value,
            min_importance_rank=cand.min_importance_rank,
        )
        if rescored_cand is not None:
            rescored.append(rescored_cand)
            total_rescorings += len(rescored_cand.tier_transitions)

    # §5 lexicographic ranking.
    ranking = _rank_candidates(rescored)

    # §5 secondary always None in first shadow release.
    primary = ranking[0] if ranking else None
    secondary = None

    # §7 training attachment AFTER ranking.
    if primary is not None:
        primary_with_training = _attach_training(
            primary, dataset_version=dataset_version,
            engine_version=engine_version,
        )
    else:
        primary_with_training = None

    # Override advisory when primary is None on PREPARATION_GAP.
    if primary_with_training is None and trigger == "diagnosis_primary":
        advisory = "barrier_present_no_move_identified"

    # Build deterministic plan + trace.
    plan = DevelopmentPlan(
        primary_recommendation=primary_with_training,
        secondary_recommendation=None,
        evaluation_summary=EvaluationSummary(
            candidate_gaps_collected_count=len(candidates) + len(truncated_names),
            candidate_gaps_rescored_count=len(rescored),
            candidate_gaps_truncated_count=len(truncated_names),
            total_rescorings_performed=total_rescorings,
            target_noc_value=target_noc_value,
            dataset_version=dataset_version,
        ),
        responder_advisory=advisory,
    )
    trace = ShadowTrace(
        invocation_outcome=diagnosis.outcome,
        invocation_trigger=trigger,
        candidates_considered=tuple(
            _rank_row(r) for r in ranking
        ),
        candidates_truncated_by_cap=tuple(sorted(truncated_names)),
        ranking_trace=tuple(_rank_row(r) for r in ranking),
        selected_primary_id=(
            primary_with_training.skill_canonical_name
            if primary_with_training is not None else None
        ),
        selected_secondary_id=None,
        dataset_version=dataset_version,
        engine_version=engine_version,
        total_rescorings_performed=total_rescorings,
    )
    return plan, trace


# --- §2 candidate-gap collection -----------------------------------------

@dataclass(frozen=True)
class _Candidate:
    """Internal candidate-gap representation before rescoring."""
    canonical_name: str
    source_job_ids: tuple[str, ...]
    min_importance_rank: int


def _collect_candidate_gaps(
    *,
    diagnosis: InventoryDiagnosis,
    in_memory_matches: list["MatchResult"],
) -> list[_Candidate]:
    """Walk supporting evidence; collect canonicalized, soft-trait-
    filtered candidate gaps. Each candidate carries the set of jobs
    where it appeared in required_missing."""
    from skillbridge.match.alignment import canonicalize_skill
    from skillbridge.match.engine import (
        _SOFT_TRAIT_SKILL_NAMES,
        _is_soft_trait_skill_name,
    )

    # Source MatchResult records:
    #   - PREPARATION_GAP: gap_record_job_ids from supporting_evidence
    #   - READY_TO_APPLY: every MatchResult with required_missing != []
    if diagnosis.outcome == "PREPARATION_GAP":
        source_ids = set(
            diagnosis.supporting_evidence.get("gap_record_job_ids") or ()
        )
        source_records = [r for r in in_memory_matches if r.job_id in source_ids]
    else:  # READY_TO_APPLY secondary
        source_records = [
            r for r in in_memory_matches
            if _has_required_missing(r)
        ]

    # gap_name -> (source_job_ids set, min importance rank)
    by_gap: dict[str, dict] = {}
    for record in source_records:
        rm = _required_missing(record)
        for raw_name in rm:
            if not isinstance(raw_name, str) or not raw_name.strip():
                continue
            if _is_soft_trait_skill_name(raw_name):
                continue
            canonical = canonicalize_skill(raw_name) or raw_name.strip().lower()
            if not canonical:
                continue
            if canonical in _SOFT_TRAIT_SKILL_NAMES:
                continue
            entry = by_gap.setdefault(
                canonical,
                {"job_ids": set(), "min_rank": 999},
            )
            entry["job_ids"].add(record.job_id)
            # Use the record's required_skills_count proxy for "min
            # importance rank" — true rank lives in extracted.job_skill
            # rows, fetched per job during rescoring. For now use a
            # constant ordering by job count as a stand-in until the
            # rescoring path returns importance.
            # (Importance rank is read from per-job skill rows in §3.)

    # M3 architecture note (2026-06-15): importance ranks are NOT
    # populated here. `_collect_candidate_gaps` must stay a pure
    # function so unit tests run without a DB. The orchestrator
    # (`compute_development_plan`) calls `_populate_min_importance_ranks`
    # on the returned candidate set BEFORE truncation runs.

    candidates = [
        _Candidate(
            canonical_name=name,
            source_job_ids=tuple(sorted(entry["job_ids"])),
            min_importance_rank=entry["min_rank"],
        )
        for name, entry in by_gap.items()
    ]
    # Stable order by canonical name (deterministic). Truncation order
    # is separate (importance-aware) — see _truncate_to_cap.
    candidates.sort(key=lambda c: c.canonical_name)
    return candidates


def _populate_min_importance_ranks(
    candidates: list[_Candidate],
) -> list[_Candidate]:
    """Bulk-fetch importance ranks for all candidate gap × source-job
    pairs and return a NEW candidate list with `min_importance_rank`
    populated (low rank = high importance).

    Architectural note: this DB-touching helper is invoked only from
    `compute_development_plan` (the orchestrator), NOT from
    `_collect_candidate_gaps`. Keeping candidate collection pure means
    unit tests run without a DB. Production CP4 calls this once per
    invocation; failure to populate degrades gracefully — candidates
    keep their default rank (999) and truncation falls back to
    canonical-name order.

    Single round-trip; bounded by the union of source job ids.
    """
    if not candidates:
        return candidates
    job_ids: set[str] = set()
    for c in candidates:
        job_ids.update(c.source_job_ids)
    if not job_ids:
        return candidates
    try:
        from skillbridge.db import sync_cursor
        from skillbridge.match.alignment import canonicalize_skill
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "cp4_importance_lookup import failed: %s", type(exc).__name__,
        )
        return candidates
    try:
        with sync_cursor() as cur:
            cur.execute(
                """
                SELECT job_id, skill_name, importance_rank
                  FROM extracted.job_skill
                 WHERE job_id = ANY(%s)
                """,
                (list(job_ids),),
            )
            rows = cur.fetchall()
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "cp4_importance_lookup query failed: %s", type(exc).__name__,
        )
        return candidates

    # Index per (job_id, canonical_skill_name) → importance_rank.
    rank_index: dict[tuple[str, str], int] = {}
    for r in rows:
        jid = r.get("job_id") if isinstance(r, dict) else r[0]
        sname = r.get("skill_name") if isinstance(r, dict) else r[1]
        rank = r.get("importance_rank") if isinstance(r, dict) else r[2]
        if not isinstance(sname, str) or not isinstance(rank, int):
            continue
        canonical = canonicalize_skill(sname) or sname.strip().lower()
        if not canonical:
            continue
        key = (str(jid), canonical)
        prev = rank_index.get(key)
        if prev is None or rank < prev:
            rank_index[key] = rank

    # Construct new frozen candidates with populated ranks.
    out: list[_Candidate] = []
    for c in candidates:
        ranks: list[int] = []
        for jid in c.source_job_ids:
            r = rank_index.get((str(jid), c.canonical_name))
            if isinstance(r, int):
                ranks.append(r)
        new_rank = min(ranks) if ranks else c.min_importance_rank
        out.append(_Candidate(
            canonical_name=c.canonical_name,
            source_job_ids=c.source_job_ids,
            min_importance_rank=new_rank,
        ))
    return out


# --- §4 computational-limit truncation -----------------------------------

def _truncate_to_cap(
    candidates: list[_Candidate], *, cap: int,
) -> tuple[list[_Candidate], list[str]]:
    """Importance-aware truncation. When `sum(|source_job_ids|)` exceeds
    `cap`, sort candidates ascending by `min_importance_rank` (low
    rank = high importance) and greedy-pack until the budget is full.
    Remaining candidates are truncated.

    M3 fix (2026-06-15): previously this iterated in canonical-name
    order, ignoring the spec's "high-importance candidates win"
    rule. With per-candidate importance now populated at collection
    time (see `_populate_min_importance_ranks`), truncation honors
    the spec.

    Determinism: ties on `min_importance_rank` break by
    `canonical_name` ascending. Same inputs always produce the same
    (kept, truncated) split.

    First shadow release: cap is HARD_CAP_TOTAL_RESCORINGS=100.
    """
    sorted_cands = sorted(
        candidates,
        key=lambda c: (c.min_importance_rank, c.canonical_name),
    )
    kept: list[_Candidate] = []
    truncated: list[str] = []
    running_total = 0
    for c in sorted_cands:
        cost = len(c.source_job_ids)
        if running_total + cost > cap:
            truncated.append(c.canonical_name)
            continue
        kept.append(c)
        running_total += cost
    # Preserve a deterministic public order on `kept`: alphabetical by
    # canonical name (matches §2's stable order).
    kept.sort(key=lambda c: c.canonical_name)
    return kept, truncated


# --- §3 counterfactual rescoring -----------------------------------------

@dataclass(frozen=True)
class _RescoredCandidate:
    canonical_name: str
    source_job_ids: tuple[str, ...]
    promoted_job_ids: tuple[str, ...]
    tier_improvement_job_ids: tuple[str, ...]
    blocker_removed_job_ids: tuple[str, ...]
    tier_transitions: tuple[JobTierTransition, ...]
    min_importance_rank: int
    active_market_frequency: int
    counterfactual_method: str
    dataset_version: str | None
    engine_version: str | None


def _counterfactual_rescore(
    *,
    candidate_skill: str,
    source_job_ids: tuple[str, ...],
    in_memory_matches: list["MatchResult"],
    staged: "StagedProfile",
    target_noc_value: str | None,
    min_importance_rank: int,
) -> _RescoredCandidate | None:
    """Per-candidate counterfactual rescoring on the bounded subset.

    For each source job: build synthetic user_skill_set = base + {gap},
    refetch the job + its skills, call _score_one_job, read the new
    band. Compare to the current band from in_memory_matches.

    Returns _RescoredCandidate aggregating the per-job transitions,
    or None if rescoring failed wholesale.
    """
    try:
        from skillbridge.match.engine import (
            _fetch_job_skill_embeddings,
            _fetch_job_skills,
            _maybe_embed_user_skill_rows,
            _score_one_job,
            is_credential_skill_name,
        )
        from skillbridge.match.alignment import (
            UserSkillRow,
            build_user_skill_rows,
            canonicalize_skill,
            derive_user_skill_sets,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "cp4_counterfactual import failed: %s", type(exc).__name__,
        )
        return None

    # Build synthetic user skill set = base + candidate
    base_user_rows = build_user_skill_rows(staged.skills)
    base_ids, base_names, base_canon = derive_user_skill_sets(base_user_rows)
    candidate_canonical = canonicalize_skill(candidate_skill) or candidate_skill.lower()

    # Skip if candidate is already in user skills (shouldn't happen but
    # defensive — it would yield zero impact anyway).
    if candidate_canonical in base_canon:
        return None

    # H1 fix (2026-06-15): bypass build_user_skill_rows' source filter
    # (which accepts only "resume"/"chat"). Construct UserSkillRow
    # directly so the counterfactual skill participates in scoring
    # identically to a real evidence-bound row.
    synth_row = UserSkillRow(
        skill_id=None,
        text=candidate_skill,
        name=candidate_skill.lower(),
        canon=candidate_canonical,
    )
    synth_rows = base_user_rows + [synth_row]
    synth_ids, synth_names, synth_canon = derive_user_skill_sets(synth_rows)

    # Build current bands index from in_memory_matches.
    current_by_id = {r.job_id: r for r in in_memory_matches}

    profile_dict = {
        "profile_id": staged.session_id,
        "preferred_location": staged.preferred_location,
        "target_role_text": staged.target_role_text,
        "target_noc": target_noc_value,
        "work_type_preference": staged.work_type_preference,
        "shift_preference": staged.shift_preference,
        "experience_text": staged.experience_text,
    }

    # M1 fix (2026-06-15): encode synthetic-skill embeddings so the
    # counterfactual scoring path uses the same semantic-match
    # infrastructure the baseline `in_memory_matches` used. Without
    # this, the projected band is computed under different rules
    # (lexical-only) than the current band (lexical + semantic),
    # invalidating the tier-transition comparison.
    synth_embeddings_matrix = _maybe_embed_user_skill_rows(synth_rows)

    transitions: list[JobTierTransition] = []
    promoted: list[str] = []
    improved: list[str] = []
    blocker_removed: list[str] = []
    actual_min_importance = 9999

    for job_id in source_job_ids:
        # Refetch the job row.
        job_row = _fetch_job_row(job_id)
        if job_row is None:
            continue
        job_skills = _fetch_job_skills(job_id)
        job_skill_embeddings = (
            _fetch_job_skill_embeddings(job_id)
            if synth_embeddings_matrix is not None else None
        )

        # Compute "current" baseline.
        current = current_by_id.get(job_id)
        if current is None:
            continue
        current_band = current.match_band
        current_rm = _required_missing(current)

        # Track min importance rank of the candidate gap across source jobs.
        for s in job_skills:
            name = s.get("skill_name") or ""
            if canonicalize_skill(name) == candidate_canonical:
                rank = s.get("importance_rank")
                if isinstance(rank, int) and rank < actual_min_importance:
                    actual_min_importance = rank
                break

        # Counterfactual score with parity semantic infrastructure.
        try:
            projected = _score_one_job(
                job_row, job_skills,
                synth_ids, synth_names, profile_dict,
                user_skill_names_canon=synth_canon,
                user_rows=synth_rows,
                user_embeddings_matrix=synth_embeddings_matrix,
                job_skill_embeddings=job_skill_embeddings,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "cp4_counterfactual: rescore failed job=%s reason=%s",
                job_id, type(exc).__name__,
            )
            continue
        if projected is None:
            continue

        projected_band = projected.match_band
        projected_rm = _required_missing(projected)

        transitions.append(JobTierTransition(
            job_id=job_id,
            current_band=current_band,
            projected_band=projected_band,
            current_required_missing_count=len(current_rm),
            projected_required_missing_count=len(projected_rm),
        ))

        # Aggregate: promoted to Apply today.
        if (
            projected.match_eligible
            and projected_band in ("strong", "good")
            and not projected_rm
            and not (
                current.match_eligible
                and current_band in ("strong", "good")
                and not current_rm
            )
        ):
            promoted.append(job_id)

        # Tier improvement (any band rise OR required-missing emptied).
        if (
            _band_rank(projected_band) > _band_rank(current_band)
            or (
                len(projected_rm) < len(current_rm)
                and len(projected_rm) == 0
            )
        ):
            improved.append(job_id)

        # Blocker removal: was a credential blocker, now removed,
        # AND the job is target-relevant (same NOC family).
        if _was_credential_blocker_removed(
            current=current,
            projected=projected,
            target_noc_value=target_noc_value,
            is_credential=is_credential_skill_name,
        ):
            blocker_removed.append(job_id)

    if not transitions:
        return None

    if actual_min_importance == 9999:
        actual_min_importance = 99

    # Active market frequency (independent query).
    active_market_frequency = _query_active_market_frequency(candidate_canonical)

    # Deterministic dataset/engine version.
    dataset_version = _get_dataset_version()
    engine_version = _get_engine_version()

    return _RescoredCandidate(
        canonical_name=candidate_canonical,
        source_job_ids=tuple(sorted(source_job_ids)),
        promoted_job_ids=tuple(sorted(promoted)),
        tier_improvement_job_ids=tuple(sorted(improved)),
        blocker_removed_job_ids=tuple(sorted(blocker_removed)),
        tier_transitions=tuple(transitions),
        min_importance_rank=actual_min_importance,
        active_market_frequency=active_market_frequency,
        counterfactual_method="fresh_per_job_score",
        dataset_version=dataset_version,
        engine_version=engine_version,
    )


# --- §5 lexicographic ranking --------------------------------------------

def _rank_candidates(
    rescored: list[_RescoredCandidate],
) -> list[_RescoredCandidate]:
    """Lexicographic precedence, locked from spec §5:

    1. len(promoted_job_ids)               desc
    2. len(blocker_removed_job_ids)        desc
    3. len(tier_improvement_job_ids)       desc
    4. active_market_frequency             desc
    5. min_importance_rank                 asc
    6. canonical_name                      asc (stable tie-break)

    No weighted sum. No magic constants. Training availability is NOT
    a ranking signal (Constraint B); it is attached AFTER ranking.
    """
    def key(c: _RescoredCandidate):
        return (
            -len(c.promoted_job_ids),
            -len(c.blocker_removed_job_ids),
            -len(c.tier_improvement_job_ids),
            -c.active_market_frequency,
            c.min_importance_rank,
            c.canonical_name,
        )
    return sorted(rescored, key=key)


# --- §7 training attachment AFTER ranking --------------------------------

def _attach_training(
    rescored: _RescoredCandidate, *, dataset_version: str | None,
    engine_version: str | None,
) -> RecommendationCandidate:
    """Query training registry for the recommendation's canonical
    name. Surface only verified options with non-null URLs. Record
    presence of unverified mappings for telemetry but never name
    them user-facing."""
    verified: list[TrainingOption] = []
    unverified_exists = False
    try:
        from skillbridge.training.registry import get_registry
        registry = get_registry()
        today = date.today()
        resources = registry.surface_resources(
            rescored.canonical_name, today=today, limit=5,
        )
        for r in resources:
            url = r.surface_url(today)
            if url is not None:
                verified.append(TrainingOption(
                    provider=str(r.provider),
                    title=f"{r.provider} — {rescored.canonical_name}",
                    url=str(url),
                    type=str(r.type) if getattr(r, "type", None) else None,
                ))
            else:
                unverified_exists = True
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "cp4_training_attachment failed: %s", type(exc).__name__,
        )

    has_actionable = bool(verified)
    honest_reason: HonestReason = (
        "verified_route_available" if has_actionable else "no_verified_option"
    )

    return RecommendationCandidate(
        skill_canonical_name=rescored.canonical_name,
        source_job_ids=rescored.source_job_ids,
        promoted_job_ids=rescored.promoted_job_ids,
        tier_improvement_job_ids=rescored.tier_improvement_job_ids,
        blocker_removed_job_ids=rescored.blocker_removed_job_ids,
        active_market_frequency=rescored.active_market_frequency,
        min_importance_rank=rescored.min_importance_rank,
        training_attachment=TrainingAttachment(
            verified_options=tuple(verified),
            has_actionable_url=has_actionable,
            honest_reason=honest_reason,
            unverified_mapping_exists=unverified_exists,
        ),
        evidence_provenance=EvidenceProvenance(
            per_job_evidence=rescored.tier_transitions,
            counterfactual_method=rescored.counterfactual_method,
            dataset_version=dataset_version or rescored.dataset_version,
            engine_version=engine_version or rescored.engine_version,
        ),
    )


# --- Helpers -------------------------------------------------------------

def _has_required_missing(result: "MatchResult") -> bool:
    return bool(_required_missing(result))


def _required_missing(result: "MatchResult") -> list[str]:
    if not isinstance(result.score_explanation, dict):
        return []
    rm = result.score_explanation.get("required_missing")
    return list(rm) if isinstance(rm, list) else []


_BAND_ORDER = {"low": 0, "stretch": 1, "good": 2, "strong": 3}


def _band_rank(band: str) -> int:
    return _BAND_ORDER.get(band, -1)


def _was_credential_blocker_removed(
    *, current, projected, target_noc_value: str | None, is_credential,
) -> bool:
    """A credential blocker is removed when the job currently has a
    credential gap and the projected scoring does not. Target-relevant
    only: the job's NOC must share at least the minor-group (4-digit)
    prefix with the user's target NOC. When target NOC is unknown,
    blocker removal counts iff the job has the credential gap and the
    candidate skill canonically equals the credential."""
    current_rm = _required_missing(current)
    projected_rm = _required_missing(projected)
    had_cred = any(is_credential(s) for s in current_rm)
    has_cred = any(is_credential(s) for s in projected_rm)
    if not (had_cred and not has_cred):
        return False
    if not target_noc_value:
        return True
    job_noc = getattr(current, "noc_code", None)
    if not isinstance(job_noc, str) or len(job_noc) < 4:
        return False
    return job_noc[:4] == target_noc_value[:4]


def _fetch_job_row(job_id: str) -> dict | None:
    """Refetch the v_current_job row by job_id. Single-row query;
    bounded by the per-candidate source_job_ids count."""
    try:
        from skillbridge.db import sync_cursor
        with sync_cursor() as cur:
            cur.execute(
                "SELECT * FROM core.v_current_job WHERE job_id = %s",
                (job_id,),
            )
            return cur.fetchone()
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "cp4_counterfactual: job-row fetch failed id=%s reason=%s",
            job_id, type(exc).__name__,
        )
        return None


def _query_active_market_frequency(canonical_name: str) -> int:
    """Independent count over current LOCAL postings (SSM-proper only)
    requiring a skill whose canonical form matches the candidate.

    Round 3 fix (2026-06-15): the previous SQL compared
    LOWER(s.skill_name) directly against the candidate's canonical
    name. Aliases (e.g. "MS Excel" lower-cases to "ms excel" but
    canonicalises to "excel") were undercounted, which could change
    the §5 ranking. The fix fetches local rows and applies the
    existing `canonicalize_skill` authority Python-side, matching
    the alias treatment the engine uses elsewhere.

    Bounded by the count of local job_skill rows on the candidate's
    skill family — typically <1000.
    """
    try:
        from skillbridge.db import sync_cursor
        from skillbridge.match.alignment import canonicalize_skill
        from skillbridge.match.region import is_ssm_region_job
        with sync_cursor() as cur:
            cur.execute(
                """
                SELECT j.job_id, j.region_code, j.location, s.skill_name
                  FROM core.v_current_job j
                  JOIN extracted.job_skill s ON s.job_id = j.job_id
                """,
            )
            rows = cur.fetchall()
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "cp4_active_market_frequency failed: %s", type(exc).__name__,
        )
        return 0

    matching_job_ids: set = set()
    for r in rows:
        sname = r.get("skill_name") if isinstance(r, dict) else r[3]
        if not isinstance(sname, str) or not sname.strip():
            continue
        if (canonicalize_skill(sname) or sname.strip().lower()) != canonical_name:
            continue
        if not is_ssm_region_job(r):
            continue
        jid = r.get("job_id") if isinstance(r, dict) else r[0]
        if jid is not None:
            matching_job_ids.add(jid)
    return len(matching_job_ids)


def _get_dataset_version() -> str | None:
    """Read the current dataset_version, if exposed. Falls back to
    None (deterministic trace will still hash on inputs)."""
    try:
        from skillbridge.chat.pipeline_snapshot import fetch_pipeline_snapshot
        snap = fetch_pipeline_snapshot()
        # PipelineSnapshot may not yet expose dataset_version directly;
        # use last_publish_at_text as a stable per-publish identifier
        # if dataset_version field is unavailable.
        return getattr(snap, "dataset_version", None) or getattr(
            snap, "last_publish_at_text", None,
        )
    except Exception:  # noqa: BLE001
        return None


def _get_engine_version() -> str | None:
    try:
        from skillbridge.versions import ENGINE_VERSION_JOB_MATCH
        return str(ENGINE_VERSION_JOB_MATCH)
    except Exception:  # noqa: BLE001
        return None


def _rank_row(c: _RescoredCandidate) -> CandidateRankRow:
    return CandidateRankRow(
        skill_canonical_name=c.canonical_name,
        promoted_count=len(c.promoted_job_ids),
        blocker_removed_count=len(c.blocker_removed_job_ids),
        tier_improvement_count=len(c.tier_improvement_job_ids),
        active_market_frequency=c.active_market_frequency,
        min_importance_rank=c.min_importance_rank,
    )
