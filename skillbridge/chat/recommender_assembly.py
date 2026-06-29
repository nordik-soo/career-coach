"""Slice 5 step 4 (2026-06-19) -- recommender evidence assembly.

Three build helpers, one per recommender mode. The handler calls the
matching helper based on the active `pending_recommender_offer` and
passes the returned RecommenderEvidence into ResponderV2Input. Each
helper is single-purpose:

  build_recommender_evidence_local_gap_coach (Layer B):
    Filters Layer B records by the CP4 primary recommendation's
    canonical name (caller passes it in) and attaches verified
    TrainingResources from the registry.

  build_recommender_evidence_target_noc_standard (Layer A):
    Runs the Layer A detector against the user's resolved skill IDs
    and the target NOC, caps at top-3 by importance.

  build_recommender_evidence_adjacent_noc_standard (Layer C):
    Runs the Layer C detector against the persisted last_adjacent_nocs
    (captured at the original match turn by handler step 3 SET).
    Caps each NOC's contribution at top-3 by importance.

All helpers are SIDE-EFFECT FREE: no DB writes, no staged mutation, no
network. The handler is the orchestrator; the helpers do focused
data shaping for the responder.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable

from skillbridge.chat.gap_evidence import (
    GapEvidence,
    RecommenderEvidence,
    TrainingResource,
    compute_adjacent_noc_standard_gaps,
    compute_local_posting_gaps,
    compute_target_noc_standard_gaps,
)
from skillbridge.match.aliases import canonicalize_skill
from skillbridge.training.models import Resource
from skillbridge.training.registry import TrainingRegistry

log = logging.getLogger(__name__)


# Layer A / Layer C cap per the locked design.
_TOP_K_PER_NOC: int = 3


# Slice 2 (re-introduced 2026-06-23 after revert of f277e60): Layer B
# target-NOC family filter. Matching engine's top-5 by skill overlap
# can include off-target NOCs (live verify: Communication Operator
# 14404 and admin assistant 13110 appeared when target was accounting
# clerk 14200). Layer B must anchor on target-NOC postings before CP4
# picks the primary gap; otherwise CP4 ranks across off-target
# postings and recommends gaps from the wrong occupation. Strict
# filter: exact NOC preferred, minor-group fallback, else empty
# (Layer B then enters the new slice 2 three-branch logic in handler).
def filter_matches_to_target_family(
    match_results: Iterable[Any],
    target_noc: str | None,
) -> list[Any]:
    """Filter MatchResults to target-NOC family for Layer B grounding.

    Returns:
        - All match_results unchanged when target_noc is None / not a
          5-digit numeric (no anchor available).
        - Subset whose `noc_code` equals target_noc exactly (preferred).
        - Else subset whose `noc_code` shares the first 4 digits with
          target_noc (minor-group fallback).
        - Empty list when neither yields a result. The caller's
          Layer B path then enters the empty-evidence branch.

    Args:
        match_results: any iterable of MatchResult-like objects with
            a `noc_code: str | None` attribute. Items without a valid
            noc_code are skipped in family fallback.
        target_noc: the user's resolved target NOC, 5-digit string.
    """
    materialized = list(match_results)
    if not isinstance(target_noc, str):
        return materialized
    code = target_noc.strip()
    if len(code) != 5 or not code.isdigit():
        return materialized

    exact = [
        m for m in materialized
        if isinstance(getattr(m, "noc_code", None), str)
        and getattr(m, "noc_code") == code
    ]
    if exact:
        return exact

    target_minor = code[:4]
    family = [
        m for m in materialized
        if isinstance(getattr(m, "noc_code", None), str)
        and getattr(m, "noc_code", "").startswith(target_minor)
        and len(getattr(m, "noc_code", "")) == 5
    ]
    return family


@dataclass(frozen=True, slots=True)
class _AdjacentNocShim:
    """Tiny adapter so compute_adjacent_noc_standard_gaps (which expects
    objects with a `noc_code` attribute) can consume the persisted
    last_adjacent_nocs strings. Local scope; not exported."""
    noc_code: str


# ---------------------------------------------------------------------------
# Layer B helper -- local_gap_coach
# ---------------------------------------------------------------------------
def build_recommender_evidence_local_gap_coach(
    *,
    match_results: Iterable[Any],
    primary_gap_name: str | None,
    registry: TrainingRegistry | None,
    today: date,
) -> RecommenderEvidence:
    """Build the local_gap_coach payload.

    Args:
        match_results: engine MatchResult iterable from the current
            turn (or a snapshot reconstruction; whatever the handler
            has on consume).
        primary_gap_name: the CP4 ranker's primary recommendation
            canonical name (caller computes via compute_primary_gap_name
            -- this helper does NOT call CP4 itself, keeping its
            inputs narrow). When None, no Layer B record will pass
            the canonical filter and `evidence` will be empty.
        registry: training registry instance. When None (e.g. registry
            failed to load at startup), no TrainingResources will be
            attached -- behavior degrades gracefully.
        today: current date for Resource.surface_url freshness check.

    Returns:
        RecommenderEvidence(mode="local_gap_coach", evidence=top-1,
        training=verified resources for that gap). evidence is empty
        when primary_gap_name is None or filtering yields no record.
    """
    all_gaps = compute_local_posting_gaps(match_results=match_results)

    if primary_gap_name is None or not primary_gap_name.strip():
        return RecommenderEvidence(
            mode="local_gap_coach",
            evidence=(),
            training=(),
        )

    target_canonical = canonicalize_skill(primary_gap_name)
    if not target_canonical:
        return RecommenderEvidence(
            mode="local_gap_coach",
            evidence=(),
            training=(),
        )

    # Filter Layer B records to those matching the CP4 primary's
    # canonical name. Per the locked design we use the SAME canonicalizer
    # authority the matcher uses -- never a lowercase-name compare.
    matching: list[GapEvidence] = [
        g for g in all_gaps
        if canonicalize_skill(g.skill_name) == target_canonical
    ]

    if not matching:
        return RecommenderEvidence(
            mode="local_gap_coach",
            evidence=(),
            training=(),
        )

    # Top-1 by design.
    top = matching[0]

    training = _build_training_resources(
        gap_name=primary_gap_name,
        gap_skill_id=top.skill_id,
        registry=registry,
        today=today,
    )

    return RecommenderEvidence(
        mode="local_gap_coach",
        evidence=(top,),
        training=training,
    )


def _build_training_resources(
    *,
    gap_name: str,
    gap_skill_id: str | None,
    registry: TrainingRegistry | None,
    today: date,
) -> tuple[TrainingResource, ...]:
    """Look up the registry gap by canonical name and emit
    TrainingResource records ONLY for resources with a non-None
    surface_url(today) -- mirroring the freshness gate documented on
    Resource.surface_url. referral_only resources are filtered out by
    construction (surface_url short-circuits on type=='referral_only')."""
    if registry is None or not gap_name.strip():
        return ()
    try:
        resources = registry.surface_resources(gap_name, today=today)
    except Exception:  # noqa: BLE001
        log.warning(
            "recommender_assembly registry.surface_resources_failed "
            "gap_name=%r", gap_name,
        )
        return ()

    out: list[TrainingResource] = []
    for res in resources:
        url = res.surface_url(today)
        if not url:
            continue
        out.append(TrainingResource(
            skill_id=gap_skill_id,
            skill_name=gap_name,
            provider=res.provider,
            type=res.type,
            url=url,
            summary=res.summary,
        ))
    return tuple(out)


# ---------------------------------------------------------------------------
# Layer A helper -- target_noc_standard
# ---------------------------------------------------------------------------
def build_recommender_evidence_target_noc_standard(
    *,
    user_skill_ids: Iterable[str],
    target_noc: str | None,
) -> RecommenderEvidence:
    """Build the target_noc_standard payload.

    Args:
        user_skill_ids: same canonical skill IDs the matcher uses
            (build_user_skill_rows -> derive_user_skill_sets). Passed
            through to the Layer A detector.
        target_noc: the user's resolved target NOC code (5 digits).
            When None / invalid, the detector returns empty gaps and
            this helper passes that through honestly.

    Returns:
        RecommenderEvidence(mode="target_noc_standard", evidence=top-3
        by importance, training=()). Training is always empty for this
        mode by design (occupation-standard development areas, not
        provider recommendations).
    """
    try:
        result = compute_target_noc_standard_gaps(
            user_skill_ids=user_skill_ids,
            target_noc=target_noc,
        )
    except Exception:  # noqa: BLE001
        log.warning(
            "recommender_assembly layer_a_failed target_noc=%r",
            target_noc,
        )
        return RecommenderEvidence(
            mode="target_noc_standard",
            evidence=(),
            training=(),
        )

    # Detector emits ORDER BY importance DESC NULLS LAST, skill_name.
    # Top-K cap is applied here at the wrapper-assembly layer per the
    # locked design.
    capped = result.gaps[:_TOP_K_PER_NOC]
    return RecommenderEvidence(
        mode="target_noc_standard",
        evidence=tuple(capped),
        training=(),
    )


# ---------------------------------------------------------------------------
# Layer C helper -- adjacent_noc_standard
# ---------------------------------------------------------------------------
def build_recommender_evidence_adjacent_noc_standard(
    *,
    user_skill_ids: Iterable[str],
    last_adjacent_nocs: Iterable[str],
) -> RecommenderEvidence:
    """Build the adjacent_noc_standard payload.

    Args:
        user_skill_ids: same canonical skill IDs the matcher uses.
        last_adjacent_nocs: the per-target persisted NOC codes captured
            by handler step 3 SET from CP5 tier_evidence.sideways_move.
            The chain-bound TTL is enforced staging-side; this helper
            trusts the input.

    Returns:
        RecommenderEvidence(mode="adjacent_noc_standard", evidence=concat
        of per-NOC top-K, training=()). Each NOC contributes at most
        _TOP_K_PER_NOC records, ordered by importance DESC NULLS LAST.
        Per-NOC ordering is preserved (first-seen NOC first).
    """
    shims: list[_AdjacentNocShim] = []
    for noc in last_adjacent_nocs:
        if isinstance(noc, str) and noc.strip():
            shims.append(_AdjacentNocShim(noc_code=noc.strip()))

    if not shims:
        return RecommenderEvidence(
            mode="adjacent_noc_standard",
            evidence=(),
            training=(),
        )

    try:
        slices = compute_adjacent_noc_standard_gaps(
            user_skill_ids=user_skill_ids,
            sideways_move=shims,
        )
    except Exception:  # noqa: BLE001
        log.warning(
            "recommender_assembly layer_c_failed nocs=%r",
            list(last_adjacent_nocs),
        )
        return RecommenderEvidence(
            mode="adjacent_noc_standard",
            evidence=(),
            training=(),
        )

    out: list[GapEvidence] = []
    for sl in slices:
        # Each slice's gaps are already importance-sorted at the SQL
        # level (Layer A/C share _LAYER_A_SQL); top-K applied here per
        # the locked design.
        out.extend(sl.gaps[:_TOP_K_PER_NOC])

    return RecommenderEvidence(
        mode="adjacent_noc_standard",
        evidence=tuple(out),
        training=(),
    )


# ---------------------------------------------------------------------------
# Slice 4 (2026-06-26) -- recommender-internal adjacency derivation
# ---------------------------------------------------------------------------
#
# Cap = 3 adjacent NOCs in the surface. Justification: keep the Layer C
# response conversational. A coach surfaces 1-3 related-role suggestions,
# not 8; the user has to be able to pick one to drill into via the
# existing follow-up close ("Want to dig into one of these in
# particular?"). This is independent of _TOP_K_PER_NOC (which controls
# per-NOC development-area count); they happen to be the same number (3)
# but for different reasons.
#
# Internal coupling acknowledgment: the helper below imports
# `_load_active_jobs_with_skills` (leading-underscore private function)
# and `build_user_skill_sets` from skillbridge/match/adjacent.py. This
# is intentional internal reuse -- we deliberately call the matching
# engine's read-only helpers instead of reimplementing them -- but it
# creates a coupling. If adjacent.py renames or changes those helpers,
# the recommender breaks. Slice 4 tests catch this immediately (ImportError
# or assertion failure). Accepted risk; documented here so the next
# person renaming adjacent.py knows the recommender depends on it.
_MAX_RECOMMENDER_ADJACENT_NOCS: int = 3


def _compute_adjacent_nocs_for_recommender(staged: Any) -> tuple[str, ...]:
    """Slice 4 (locked 2026-06-26, refined slice 4a): recommender-internal
    adjacency derivation with strict-first + soft-fallback.

    When Layer C is dispatched but staged.last_adjacent_nocs is empty
    (cold start -- no prior matching turn populated it), this helper
    invokes the matching engine's adjacency pipeline READ-ONLY to
    derive adjacent NOCs ephemerally. Result is NOT persisted to
    staged; the caller uses it only for this turn's Layer C evidence
    wrapper assembly.

    Strict-first / soft-fallback design (locked by live verify
    2026-06-26):

        retrieve_candidates    # broad SSM-filter
            ↓
        try accept_candidates  # CP5's strict gates (required coverage,
                               # transferable threshold)
            ↓
        if strict_accepted has candidates:
            candidates = strict_accepted
            soft_used = False
        else:
            candidates = retrieved   # SOFT FALLBACK: drop the strict
                                     # accept gate. Layer C is "career
                                     # paths worth exploring," not
                                     # "jobs you may fit" (CP5). The
                                     # retrieve gate (SSM region +
                                     # skill-hit OR minor-group) is
                                     # the right bound for exploration.
            soft_used = True
            ↓
        rank_adjacent          # always apply
            ↓
        drop_excluded(presented_job_ids)
            ↓
        extract distinct NOC codes, cap = _MAX_RECOMMENDER_ADJACENT_NOCS

    Why strict-first instead of always-soft: when strict acceptance
    DOES yield candidates, those NOCs have earned the surface (user
    has real coverage + transferable evidence). Skipping that signal
    would treat strong-fit users the same as weak-fit users. Live
    verify of an accounting-clerk profile vs. SSM admin-assistant
    postings hit the soft-fallback path because the strict gate's
    required-coverage threshold rejected all candidates (Jordan
    Miller has bookkeeping + payroll + Excel; admin assistant
    postings list reception + calendar + phone as required). The
    coach VOICE is still appropriate for the soft path -- the prompt
    uses exploratory framing ("if you wanted to move toward [NOC],
    that role leans on..."), not "you're well-positioned for X."

    Internal coupling acknowledgment: imports `_load_active_jobs_with_skills`
    (leading-underscore private function) and `build_user_skill_sets`
    from skillbridge/match/adjacent.py. Intentional internal reuse;
    if adjacent.py renames these, slice 4 tests catch it immediately.

    Args:
        staged: StagedProfile-like object. Reads only:
            staged.skills, staged.target_noc, staged.last_match_snapshot.

    Returns:
        Tuple of distinct NOC code strings in rank order. Capped at
        _MAX_RECOMMENDER_ADJACENT_NOCS (=3). Empty tuple if both
        strict and soft yield nothing.

    Defensive: any exception -> log + return (). Cold-start path
    must NEVER crash the recommender turn -- slice 2 follow-up
    empty-evidence guard then emits honest text.
    """
    try:
        from skillbridge.match.adjacent import (
            _load_active_jobs_with_skills,
            accept_candidates,
            build_user_skill_sets,
            drop_excluded,
            rank_adjacent,
            retrieve_candidates,
        )

        user_ids, user_names, user_canon = build_user_skill_sets(
            staged.skills,
        )
        all_jobs = _load_active_jobs_with_skills()

        # Slice 4 diagnostic (2026-06-29 live verify): when retrieved=0
        # in production we need to know WHY -- empty user-side skill
        # sets, empty job pool, or non-empty-but-non-matching canonical
        # gates. These three log lines provide enough signal to root-
        # cause without instrumenting matching engine internals.
        log.info(
            "recommender_internal_adjacency_pre_retrieve "
            "user_ids=%d user_names=%d user_canon=%d staged_skills=%d",
            len(user_ids), len(user_names), len(user_canon),
            len(staged.skills) if staged.skills else 0,
        )
        log.info(
            "recommender_internal_adjacency_jobs_loaded "
            "all_jobs=%d target_noc=%r",
            len(all_jobs) if all_jobs else 0,
            getattr(staged, "target_noc", None),
        )
        # Sample first 5 user skill names + first 5 names from the
        # first SSM-region job so canonicalization mismatch shows up
        # immediately (e.g. resume "QuickBooks Desktop" vs job
        # "Microsoft Office").
        user_name_sample = sorted(user_names)[:5] if user_names else []
        sample_job_skills: list[str] = []
        if all_jobs and isinstance(all_jobs, list):
            for j in all_jobs:
                if not isinstance(j, dict):
                    continue
                raw = j.get("skills") or []
                if not isinstance(raw, list):
                    continue
                for s in raw:
                    if isinstance(s, dict):
                        n = s.get("skill_name")
                        if isinstance(n, str) and n.strip():
                            sample_job_skills.append(n.strip())
                            if len(sample_job_skills) >= 5:
                                break
                if sample_job_skills:
                    break
        log.info(
            "recommender_internal_adjacency_skill_samples "
            "user_skill_sample=%s job_skill_sample=%s",
            user_name_sample, sample_job_skills,
        )

        retrieved = retrieve_candidates(
            staged,
            snapshot=None,
            all_jobs=all_jobs,
            user_ids=user_ids,
            user_names=user_names,
            user_canon=user_canon,
        )

        # Strict-first: try the matching engine's strict acceptance.
        strict_accepted, _drops = accept_candidates(
            retrieved, staged, user_ids, user_names, user_canon,
        )

        # Soft-fallback: if strict accept yielded nothing but retrieve
        # found plausible candidates, drop the strict gate. The
        # retrieve gate (SSM + skill-hit OR minor-group hit) is
        # already a meaningful bound for exploration purposes.
        if strict_accepted:
            candidates = strict_accepted
            soft_used = False
        else:
            candidates = list(retrieved)
            soft_used = True

        ranked = rank_adjacent(
            candidates, user_ids, user_names, user_canon,
        )

        # Exclude jobs already shown in this session (prevents
        # recommender re-surfacing the same role the matcher just
        # rendered as a tier card).
        presented: tuple[str, ...] = ()
        if isinstance(staged.last_match_snapshot, dict):
            raw_presented = (
                staged.last_match_snapshot.get("presented_job_ids") or ()
            )
            if isinstance(raw_presented, (list, tuple)):
                presented = tuple(
                    x for x in raw_presented if isinstance(x, str)
                )
        filtered = drop_excluded(ranked, presented)

        # Extract distinct NOC codes from the ranked-and-filtered
        # candidates, preserving rank order, capped at the
        # conversational surface.
        out: list[str] = []
        seen: set[str] = set()
        for job in filtered:
            if not isinstance(job, dict):
                continue
            noc = job.get("noc_code")
            if not isinstance(noc, str):
                continue
            noc_stripped = noc.strip()
            if not noc_stripped or noc_stripped in seen:
                continue
            seen.add(noc_stripped)
            out.append(noc_stripped)
            if len(out) >= _MAX_RECOMMENDER_ADJACENT_NOCS:
                break

        # Diagnostic log so future "empty" cases are visible without
        # adding instrumentation. Counts at each gate make the
        # strict-vs-soft path immediately clear in production logs.
        log.info(
            "recommender_internal_adjacency "
            "retrieved=%d strict_accepted=%d soft_used=%s "
            "returned_nocs=%s",
            len(retrieved),
            len(strict_accepted),
            soft_used,
            list(out),
        )
        return tuple(out)
    except Exception:  # noqa: BLE001
        log.exception(
            "recommender_internal_adjacency_pipeline_failed; "
            "returning empty",
        )
        return ()
