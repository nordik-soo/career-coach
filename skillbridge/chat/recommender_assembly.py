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
    RoleDrilldownEvidence,
    RoleDrilldownSkillRow,
    TrainingResource,
    _fetch_noc_skill_rows,
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


# ---------------------------------------------------------------------------
# Slice 5 (2026-06-29) -- role drilldown skill-comparison evidence
# ---------------------------------------------------------------------------
_MAX_DRILLDOWN_ROWS: int = 7  # top-7 OaSIS skills by importance (lock)
_MAX_YOUR_SKILL_NAMES_PER_ROW: int = 2  # cap per locked design

# Slice 8 (2026-06-30): cosine pre-pass threshold. Lowered from
# slice 7a's 0.25 to 0.15 because cosine is NO LONGER the gate;
# it's a SIGNAL the batched LLM judgment sees. The LLM rejects
# weak cosine candidates that don't actually transfer. At 0.15,
# more candidates surface for the LLM to consider, without
# spamming the payload (cap _MAX_COSINE_CANDIDATES_PER_ROW=3).
#
# Calibration history (slice 7a): max cosine = 0.390 against
# Jordan Miller resume vs NOC 13110. At 0.25, 3 rows match
# (mostly weak bridges like 'account reconciliation' -> 'Coordinating').
# At 0.15, more rows surface candidates but quality is the LLM's
# job to decide.
#
# When LLM is unavailable (LLM_ENABLED=False / call failure), this
# threshold RESUMES the slice-7a gate role: cosine-as-gate fallback.
_DRILLDOWN_SEMANTIC_THRESHOLD: float = 0.15

# Module-load startup log so operators can see active mode without
# reading code. Fires once per process at import time.
_STARTUP_LOG_EMITTED: bool = False


def _emit_drilldown_startup_log() -> None:
    """Emit one INFO line per process recording the active
    DRILLDOWN_SEMANTIC mode + threshold. Idempotent."""
    global _STARTUP_LOG_EMITTED
    if _STARTUP_LOG_EMITTED:
        return
    _STARTUP_LOG_EMITTED = True
    try:
        from config import DRILLDOWN_SEMANTIC_MODE
        mode = DRILLDOWN_SEMANTIC_MODE
    except Exception:  # noqa: BLE001
        mode = "off"
    threshold_str = (
        f"{_DRILLDOWN_SEMANTIC_THRESHOLD:.2f}" if mode != "off" else "N/A"
    )
    log.info(
        "drilldown_semantic_mode=%s threshold=%s",
        mode, threshold_str,
    )


_emit_drilldown_startup_log()


def _semantic_score_user_vs_oasis(
    *,
    user_skill_names: list[str],
    oasis_skill_name: str,
) -> list[tuple[str, float]] | None:
    """Slice 7a + Slice 8 hardening (2026-06-30): compute cosine
    similarity between every user skill and the OaSIS skill.

    Slice 8 lock: OaSIS side embeds skill_name ONLY (no description).
    The earlier slice-7a version embedded "name: description" but
    that was rejected at the slice 8 sign-off: descriptions inflate
    cosine scores via generic boilerplate and conflict with the
    "NOC skillset + user profile, not OaSIS description text" rule.
    Bare names produce honest concrete<->concrete or concrete<->
    abstract cosines, and the LLM judgment step (when mode=on) does
    the real reasoning over user profile.

    Returns [(user_name, score), ...] sorted by score DESC, or None
    on embedding service failure.

    Caller decides what to do with the scores based on mode:
      - mode=log: log them; ✓/✗ unchanged (debug aid only)
      - mode=on:  use as candidates for the batched LLM judgment
    """
    if not user_skill_names:
        return []
    try:
        from skillbridge.embed.service import get_embedder
    except Exception:  # noqa: BLE001
        log.warning(
            "drilldown_semantic_failed reason=embed_service_import_failed "
            "degrading=exact_only",
        )
        return None

    embedder = get_embedder()
    if embedder is None:
        log.warning(
            "drilldown_semantic_failed reason=embedder_unavailable "
            "degrading=exact_only",
        )
        return None

    try:
        user_vecs = embedder.encode_many(list(user_skill_names))
        oasis_vec = embedder.encode_one(oasis_skill_name)
    except Exception:  # noqa: BLE001
        log.warning(
            "drilldown_semantic_failed reason=encode_failed "
            "degrading=exact_only",
            exc_info=True,
        )
        return None

    # Vectors are L2-normalized -> cosine = dot product.
    import numpy as np
    scores = (user_vecs @ oasis_vec).tolist()

    pairs = list(zip(user_skill_names, scores))
    pairs.sort(key=lambda p: p[1], reverse=True)
    return pairs


# ---------------------------------------------------------------------------
# Slice 8 (2026-06-30) -- batched LLM judgment for drilldown
# ---------------------------------------------------------------------------
_MAX_COSINE_CANDIDATES_PER_ROW: int = 3
_MAX_USER_EVIDENCE_CHARS: int = 200  # ~150 plus buffer
_MAX_REASON_CHARS: int = 180  # ~120 plus buffer

# Tool schema for Anthropic tool_use forced output. Mirrors the
# pattern in recommender_intent.py.
_DRILLDOWN_TOOL_NAME: str = "submit_drilldown_judgment"
_DRILLDOWN_TOOL_SCHEMA: dict = {
    "name": _DRILLDOWN_TOOL_NAME,
    "description": (
        "Submit per-skill judgment of whether the user's profile "
        "demonstrates each OaSIS skill required for the adjacent "
        "NOC. Return one judgment per skill in the order received."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "judgments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "oasis_skill": {"type": "string"},
                        "matched": {"type": "boolean"},
                        "user_evidence": {"type": ["string", "null"]},
                        "reason": {"type": ["string", "null"]},
                    },
                    "required": [
                        "oasis_skill", "matched",
                        "user_evidence", "reason",
                    ],
                },
            },
        },
        "required": ["judgments"],
    },
}


def _judge_drilldown_with_llm(
    *,
    noc_code: str,
    role_title: str,
    noc_skillset: list[dict],
    user_profile: dict,
) -> dict[str, dict] | None:
    """Slice 8: batched LLM judgment for drilldown table.

    One Anthropic tool_use call with structured output. The LLM
    receives all OaSIS skills (with their match_signal + cosine
    candidates) plus the user's full profile, and returns one
    judgment per skill.

    Args:
        noc_code: 5-digit adjacent NOC code.
        role_title: OaSIS noc_title for the chosen adjacent role.
        noc_skillset: list of dicts, one per OaSIS skill:
            {
              "skill": "Writing",
              "match_signal": "exact" | "cosine" | "none",
              "cosine_candidates": [
                {"user_skill": "microsoft word", "score": 0.39},
                ...
              ]  (empty list when match_signal != "cosine")
            }
        user_profile: dict with keys skills / work_history /
            education / certifications.

    Returns:
        Dict mapping oasis_skill name -> judgment dict, where each
        judgment has keys (matched, user_evidence, reason). Indexed
        by skill name so the caller can join into RoleDrilldownSkillRow
        without ordering assumptions.

        Returns None on any failure (LLM disabled / call errored /
        invalid tool output / wrong structure). Caller falls back to
        cosine-only result.
    """
    import json as _json
    try:
        from skillbridge.llm import LLM_ENABLED, LLM_MODEL, LLM_FALLBACK_MODEL
    except Exception:  # noqa: BLE001
        log.warning(
            "drilldown_llm_judgment_failed reason=llm_module_import_failed "
            "degrading=cosine_only",
        )
        return None

    if not LLM_ENABLED:
        log.info(
            "drilldown_llm_disabled noc=%s -- using cosine-only fallback",
            noc_code,
        )
        return None

    try:
        import anthropic  # noqa: F401
        from skillbridge.llm import _client_get
    except Exception:  # noqa: BLE001
        log.warning(
            "drilldown_llm_judgment_failed reason=anthropic_client_unavailable "
            "degrading=cosine_only",
        )
        return None

    from skillbridge.chat.prompts import DRILLDOWN_JUDGMENT_PROMPT

    user_block = _json.dumps({
        "noc_code": noc_code,
        "role_title": role_title,
        "noc_skillset": noc_skillset,
        "user_profile": user_profile,
    }, ensure_ascii=False, indent=2)

    system_blocks = [{
        "type": "text",
        "text": DRILLDOWN_JUDGMENT_PROMPT,
        "cache_control": {"type": "ephemeral"},
    }]

    def _do_call(model: str):
        client = _client_get()
        return client.messages.create(
            model=model,
            max_tokens=1200,  # ~7 judgments x ~120 chars + JSON shell
            temperature=0,
            system=system_blocks,
            tools=[_DRILLDOWN_TOOL_SCHEMA],
            tool_choice={"type": "tool", "name": _DRILLDOWN_TOOL_NAME},
            messages=[{"role": "user", "content": user_block}],
        )

    try:
        import anthropic
        try:
            resp = _do_call(LLM_MODEL)
        except anthropic.APIStatusError as e:
            if (e.status_code in (429, 529, 503)
                    and LLM_MODEL != LLM_FALLBACK_MODEL):
                log.warning(
                    "drilldown llm overloaded on %s; falling back to %s",
                    LLM_MODEL, LLM_FALLBACK_MODEL,
                )
                resp = _do_call(LLM_FALLBACK_MODEL)
            else:
                raise
    except Exception:  # noqa: BLE001
        log.warning(
            "drilldown_llm_judgment_failed reason=llm_call_failed "
            "degrading=cosine_only", exc_info=True,
        )
        return None

    # Extract tool_use block; mirror recommender_intent.py pattern.
    tool_input = None
    for block in resp.content:
        block_type = getattr(block, "type", None)
        if block_type == "tool_use":
            tool_name = getattr(block, "name", None)
            if tool_name != _DRILLDOWN_TOOL_NAME:
                continue
            tool_input = getattr(block, "input", None)
            break

    if not isinstance(tool_input, dict):
        log.warning(
            "drilldown_llm_judgment_invalid_output reason=no_tool_use_block "
            "degrading=cosine_only",
        )
        return None

    judgments_raw = tool_input.get("judgments")
    if not isinstance(judgments_raw, list):
        log.warning(
            "drilldown_llm_judgment_invalid_output reason=judgments_not_list "
            "degrading=cosine_only",
        )
        return None

    # Map oasis_skill -> judgment, with defensive normalization.
    out: dict[str, dict] = {}
    for j in judgments_raw:
        if not isinstance(j, dict):
            continue
        skill = j.get("oasis_skill")
        if not isinstance(skill, str) or not skill.strip():
            continue
        matched_raw = j.get("matched")
        if not isinstance(matched_raw, bool):
            continue
        ev = j.get("user_evidence")
        rs = j.get("reason")
        if matched_raw:
            # Matched rows MUST have both evidence and reason.
            if not isinstance(ev, str) or not ev.strip():
                # Defensive: matched but no evidence -> drop to False.
                matched_raw = False
                ev, rs = None, None
            else:
                ev = ev.strip()[:_MAX_USER_EVIDENCE_CHARS]
                rs = (
                    rs.strip()[:_MAX_REASON_CHARS]
                    if isinstance(rs, str) and rs.strip()
                    else None
                )
        else:
            # matched=False rows must NOT carry evidence/reason
            ev, rs = None, None
        out[skill.strip()] = {
            "matched": matched_raw,
            "user_evidence": ev,
            "reason": rs,
        }

    if not out:
        log.warning(
            "drilldown_llm_judgment_invalid_output reason=empty_after_parse "
            "degrading=cosine_only",
        )
        return None

    log.info(
        "drilldown_llm_judgment_ok noc=%s judgments=%d matched=%d",
        noc_code, len(out),
        sum(1 for j in out.values() if j["matched"]),
    )
    return out


# ---------------------------------------------------------------------------
# Slice 9 (2026-06-30) -- Coach Training Guide (below-the-table action plan)
# ---------------------------------------------------------------------------
# Deterministic top-N gap selection + registry-backed evidence package +
# one Anthropic tool_use call for coach prose + deterministic renderer.
#
# Zero-gap case: deterministic encouragement template referencing the
# top-3 matched skills. No LLM call.
_MAX_PRIORITY_GAPS: int = 3
_MAX_MATCHED_IN_ENCOURAGEMENT: int = 3
_MAX_MATCHED_ROWS_TO_LLM: int = 7  # cap on strengths context
_COACH_CLOSING_QUESTION: str = "Want me to help you pick the first skill to work on?"

_COACH_GUIDE_TOOL_NAME: str = "emit_coach_training_guide"
_COACH_GUIDE_TOOL_SCHEMA: dict = {
    "name": _COACH_GUIDE_TOOL_NAME,
    "description": (
        "Emit the Coach Training Guide section for the drilldown. "
        "Write one opening sentence + one gap section per priority_gap "
        "in the order given. Do NOT reorder, invent providers/URLs, "
        "or add closing text -- the renderer appends it verbatim."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "opening_sentence": {"type": "string"},
            "gaps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "skill": {"type": "string"},
                        "why_it_matters": {"type": "string"},
                        "how_to_build": {"type": "string"},
                        "training_direction": {"type": "string"},
                    },
                    "required": [
                        "skill", "why_it_matters",
                        "how_to_build", "training_direction",
                    ],
                },
            },
        },
        "required": ["opening_sentence", "gaps"],
    },
}


def _sort_rows_by_importance(
    rows: Iterable["RoleDrilldownSkillRow"],
) -> list["RoleDrilldownSkillRow"]:
    """Slice 9: importance DESC, None LAST.

    Python 3 can't compare None to float directly, so the key tuple
    puts None-importance rows in the None-last bucket, then sorts
    numeric rows by negated importance so higher values come first.
    """
    return sorted(
        rows,
        key=lambda r: (
            r.importance is None,
            -(r.importance if r.importance is not None else 0.0),
        ),
    )


def _format_matched_skill_list(names: list[str]) -> str:
    """Slice 9: grammatical Oxford-comma joiner for 1/2/3 skill names.

    - 1 name: "Writing"
    - 2 names: "Writing and Coordinating"
    - 3+ names: "Writing, Coordinating, and Digital Literacy"
      (extra items truncated -- caller passes top-N already)
    """
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    head = ", ".join(names[:-1])
    return f"{head}, and {names[-1]}"


def _build_zero_gap_encouragement(
    *,
    role_title: str,
    matched_rows: list["RoleDrilldownSkillRow"],
) -> "CoachTrainingGuide":
    """Slice 9: deterministic encouragement when the drilldown has zero
    gap rows. No LLM call. Names the top-3 matched skills by importance
    and prompts the user to turn matches into resume/interview proof.
    """
    from skillbridge.chat.gap_evidence import CoachTrainingGuide

    top_matched = _sort_rows_by_importance(matched_rows)[
        :_MAX_MATCHED_IN_ENCOURAGEMENT
    ]
    names_list = _format_matched_skill_list(
        [r.oasis_skill_name for r in top_matched]
    )
    if names_list:
        opening = (
            f"You already show the main skill areas for {role_title}. "
            f"The next step is to turn that into proof: add one resume "
            f"bullet or interview example showing how you used "
            f"{names_list} in your work."
        )
    else:
        opening = (
            f"You show the main skill areas for {role_title}. The "
            f"next step is to turn that into proof: add one resume "
            f"bullet or interview example that names those strengths."
        )
    return CoachTrainingGuide(
        opening_sentence=opening,
        priority_gaps=(),
        closing_question=_COACH_CLOSING_QUESTION,
    )


def _build_coach_evidence_package(
    *,
    role_title: str,
    noc_code: str,
    matched_rows: list["RoleDrilldownSkillRow"],
    priority_gap_rows: list["RoleDrilldownSkillRow"],
    registry: "TrainingRegistry | None",
    today: date,
    user_skill_names: list[str],
    user_work_history: list,
    user_education: list,
    user_certifications: list,
) -> dict:
    """Slice 9: assemble the evidence package the LLM receives.

    Pre-filters everything so the LLM cannot invent a provider or URL:
      * priority_gaps carry the FIRST fresh registry resource (via
        Resource.surface_url(today)) if any -- LLM cites verbatim.
      * priority_gaps also carry registry_description + registry_category
        so the "why it matters" line is grounded.
      * matched_rows are capped at _MAX_MATCHED_ROWS_TO_LLM (7) and
        sorted by importance DESC so the LLM references the strongest
        first when writing opening_sentence.
    """
    matched_capped = _sort_rows_by_importance(matched_rows)[
        :_MAX_MATCHED_ROWS_TO_LLM
    ]
    matched_payload = [
        {
            "skill": r.oasis_skill_name,
            "importance": r.importance,
            "user_evidence": r.user_evidence,
        }
        for r in matched_capped
    ]

    gaps_payload = []
    for r in priority_gap_rows:
        gap_entry: dict = {
            "skill": r.oasis_skill_name,
            "importance": r.importance,
            "registry_hit": False,
            "registry_category": None,
            "registry_description": None,
            "training_resources": [],
        }
        if registry is not None:
            try:
                gap = registry.lookup(r.oasis_skill_name)
            except Exception:  # noqa: BLE001
                gap = None
                log.warning(
                    "coach_guide_registry_lookup_failed skill=%r",
                    r.oasis_skill_name,
                )
            if gap is not None:
                gap_entry["registry_hit"] = True
                gap_entry["registry_category"] = gap.category
                gap_entry["registry_description"] = gap.description
                # First-surfaced pattern: iterate registry.surface_resources
                # in YAML order and take the first Resource whose
                # surface_url is non-None (matches existing drilldown
                # gap-training lookup at recommender_assembly.py ~L1170).
                try:
                    resources = registry.surface_resources(
                        r.oasis_skill_name, today=today,
                    )
                except Exception:  # noqa: BLE001
                    resources = []
                    log.warning(
                        "coach_guide_surface_resources_failed skill=%r",
                        r.oasis_skill_name,
                    )
                for res in resources:
                    url = res.surface_url(today)
                    if url:
                        gap_entry["training_resources"].append({
                            "provider": res.provider,
                            "type": res.type,
                            "url": url,
                            "summary": res.summary,
                        })
                        break  # first-surfaced only
        gaps_payload.append(gap_entry)

    return {
        "target_role": role_title,
        "noc_code": noc_code,
        "matched_rows": matched_payload,
        "priority_gaps": gaps_payload,
        "user_profile": {
            "skills": list(user_skill_names),
            "work_history": user_work_history or [],
            "education": user_education or [],
            "certifications": user_certifications or [],
        },
    }


def _generate_coach_guide_with_llm(
    *,
    noc_code: str,
    evidence_package: dict,
) -> "CoachTrainingGuide | None":
    """Slice 9: one Anthropic tool_use call for the Coach Training Guide.

    Returns:
        CoachTrainingGuide when the LLM call succeeds and produces a
        parseable structured payload. None on any failure (LLM disabled
        / call errored / invalid tool output / gap count mismatch).

    On None, the caller falls through to the None coach_guide branch
    -- the table renders alone (no misleading half-guide).
    """
    import json as _json
    from skillbridge.chat.gap_evidence import (
        CoachGapGuide,
        CoachTrainingGuide,
    )

    try:
        from skillbridge.llm import LLM_ENABLED, LLM_MODEL, LLM_FALLBACK_MODEL
    except Exception:  # noqa: BLE001
        log.warning(
            "coach_guide_llm_failed reason=llm_module_import_failed "
            "degrading=no_guide",
        )
        return None

    if not LLM_ENABLED:
        log.info(
            "coach_guide_llm_disabled noc=%s -- guide will be omitted",
            noc_code,
        )
        return None

    try:
        import anthropic  # noqa: F401
        from skillbridge.llm import _client_get
    except Exception:  # noqa: BLE001
        log.warning(
            "coach_guide_llm_failed reason=anthropic_client_unavailable "
            "degrading=no_guide",
        )
        return None

    from skillbridge.chat.prompts import DRILLDOWN_COACH_GUIDE_PROMPT

    priority_gaps_input = evidence_package.get("priority_gaps") or []
    expected_gap_skills = [g["skill"] for g in priority_gaps_input]

    user_block = _json.dumps(
        {"evidence_package": evidence_package},
        ensure_ascii=False, indent=2,
    )

    system_blocks = [{
        "type": "text",
        "text": DRILLDOWN_COACH_GUIDE_PROMPT,
        "cache_control": {"type": "ephemeral"},
    }]

    def _do_call(model: str):
        client = _client_get()
        return client.messages.create(
            model=model,
            max_tokens=1400,
            temperature=0,
            system=system_blocks,
            tools=[_COACH_GUIDE_TOOL_SCHEMA],
            tool_choice={"type": "tool", "name": _COACH_GUIDE_TOOL_NAME},
            messages=[{"role": "user", "content": user_block}],
        )

    try:
        import anthropic
        try:
            resp = _do_call(LLM_MODEL)
        except anthropic.APIStatusError as e:
            if (e.status_code in (429, 529, 503)
                    and LLM_MODEL != LLM_FALLBACK_MODEL):
                log.warning(
                    "coach_guide llm overloaded on %s; falling back to %s",
                    LLM_MODEL, LLM_FALLBACK_MODEL,
                )
                resp = _do_call(LLM_FALLBACK_MODEL)
            else:
                raise
    except Exception:  # noqa: BLE001
        log.warning(
            "coach_guide_llm_failed reason=llm_call_failed "
            "degrading=no_guide", exc_info=True,
        )
        return None

    tool_input = None
    for block in resp.content:
        block_type = getattr(block, "type", None)
        if block_type == "tool_use":
            tool_name = getattr(block, "name", None)
            if tool_name != _COACH_GUIDE_TOOL_NAME:
                continue
            tool_input = getattr(block, "input", None)
            break

    if not isinstance(tool_input, dict):
        log.warning(
            "coach_guide_invalid_output reason=no_tool_use_block "
            "degrading=no_guide",
        )
        return None

    opening = tool_input.get("opening_sentence")
    gaps_raw = tool_input.get("gaps")
    if not isinstance(opening, str) or not opening.strip():
        log.warning(
            "coach_guide_invalid_output reason=missing_opening_sentence",
        )
        return None
    if not isinstance(gaps_raw, list):
        log.warning(
            "coach_guide_invalid_output reason=gaps_not_list",
        )
        return None

    # LLM must produce one entry per priority_gap in the same order.
    # Index by skill (verbatim match) and fill in provider/URL from
    # the evidence package -- the LLM did NOT receive those to write
    # into training_direction (assembly attaches them for rendering).
    expected_set = set(expected_gap_skills)
    gap_records: dict[str, CoachGapGuide] = {}
    for g in gaps_raw:
        if not isinstance(g, dict):
            continue
        skill = g.get("skill")
        if not isinstance(skill, str) or skill not in expected_set:
            log.warning(
                "coach_guide_invalid_output reason=unexpected_skill "
                "skill=%r", skill,
            )
            continue
        why = g.get("why_it_matters")
        how = g.get("how_to_build")
        td = g.get("training_direction")
        if not all(isinstance(x, str) and x.strip() for x in (why, how, td)):
            log.warning(
                "coach_guide_invalid_output reason=missing_prose_field "
                "skill=%r", skill,
            )
            continue
        # Attach provider + URL from the evidence package (assembly-
        # controlled truth, not LLM invention).
        provider: str | None = None
        url: str | None = None
        for gap_entry in priority_gaps_input:
            if gap_entry["skill"] == skill:
                resources = gap_entry.get("training_resources") or []
                if resources:
                    provider = resources[0].get("provider")
                    url = resources[0].get("url")
                break
        gap_records[skill] = CoachGapGuide(
            skill=skill,
            why_it_matters=why.strip(),
            how_to_build=how.strip(),
            training_direction=td.strip(),
            training_provider=provider,
            training_url=url,
        )

    # Preserve the input order (assembly-picked, importance DESC).
    ordered_gaps = tuple(
        gap_records[s] for s in expected_gap_skills if s in gap_records
    )
    if not ordered_gaps:
        log.warning(
            "coach_guide_invalid_output reason=no_valid_gaps_after_parse",
        )
        return None

    log.info(
        "coach_guide_ok noc=%s gaps_rendered=%d",
        evidence_package.get("noc_code"), len(ordered_gaps),
    )
    return CoachTrainingGuide(
        opening_sentence=opening.strip(),
        priority_gaps=ordered_gaps,
        closing_question=_COACH_CLOSING_QUESTION,
    )


def build_recommender_evidence_role_drilldown(
    *,
    noc_code: str,
    user_skill_ids: Iterable[str],
    user_skill_names: Iterable[str],
    user_skill_canon: Iterable[str],
    user_skill_name_to_canon: dict[str, str] | None,
    registry: TrainingRegistry | None,
    today: date,
    # Slice 8 hardening (2026-06-30): explicit kwargs replace the
    # module-level _DRILLDOWN_USER_CONTEXT shim. The LLM judgment
    # step uses these to reason about the user's full background
    # (not just bare skill names). Defaults to None so slice 5/7a
    # callers that don't pass them get empty lists in the LLM
    # payload (LLM judges from skills only).
    user_work_history: list | None = None,
    user_education: list | None = None,
    user_certifications: list | None = None,
    # Slice 9 (2026-06-30): opt-in Coach Training Guide generation.
    # False by default so slice 5/7a/8 callers + tests get unchanged
    # behavior. Handler flips this to True when
    # DRILLDOWN_COACH_GUIDE_MODE == "on".
    coach_guide_enabled: bool = False,
) -> RoleDrilldownEvidence | None:
    """Slice 5: build the side-by-side OaSIS-vs-resume comparison
    payload for an adjacent role the user picked from Layer C's
    surface.

    Pipeline:
      1. Fetch OaSIS skills for `noc_code` via _fetch_noc_skill_rows
         (sorted importance DESC by SQL).
      2. Cap to top _MAX_DRILLDOWN_ROWS (=7).
      3. Per row, run cascade match against the user's skill sets:
         skill_id -> canonical -> name (binary outcome).
      4. For matched rows: collect 0-2 user-side skill names
         (alphabetical, capped at _MAX_YOUR_SKILL_NAMES_PER_ROW).
      5. For unmatched rows: lookup TrainingRegistry by the OaSIS
         skill name to attach a provider+URL (registry hit) or leave
         None (renderer will emit "ask SCCC").

    Args:
        noc_code: 5-digit NOC code for the chosen adjacent role.
        user_skill_ids: canonical skill IDs from the user's profile.
        user_skill_names: lowercased raw skill names from the user.
        user_skill_canon: canonical-form skill names from the user.
        user_skill_name_to_canon: optional reverse map letting the
            helper recover the user's raw skill name from a canonical
            match, so the "Your Skill" cell shows the resume wording.
            None -> match still works, but Your Skill cell uses the
            canonical form.
        registry: training registry instance; None -> all unmatched
            rows have training=None (renderer falls back to SCCC).
        today: date for Resource.surface_url freshness gate.

    Returns:
        RoleDrilldownEvidence with role_title (from OaSIS noc_title)
        and rows tuple (up to 7).
        None when noc_code is invalid (non-5-digit) -- caller emits
        honest fallback.
        RoleDrilldownEvidence with empty rows tuple when OaSIS has
        no profile for this NOC -- caller emits honest fallback.
    """
    # Validate NOC code shape; mirrors _is_valid_noc_code discipline.
    if not isinstance(noc_code, str):
        return None
    code = noc_code.strip()
    if len(code) != 5 or not code.isdigit():
        return None

    try:
        raw_rows = _fetch_noc_skill_rows(code)
    except Exception:  # noqa: BLE001
        log.exception(
            "recommender_drilldown_oasis_fetch_failed noc=%s", code,
        )
        return RoleDrilldownEvidence(
            noc_code=code, role_title="", rows=(),
        )

    if not raw_rows:
        # OaSIS profile not loaded for this NOC. Caller emits
        # honest fallback (no table render).
        return RoleDrilldownEvidence(
            noc_code=code, role_title="", rows=(),
        )

    role_title = ""
    if isinstance(raw_rows[0], dict):
        first_title = raw_rows[0].get("noc_title")
        if isinstance(first_title, str):
            role_title = first_title.strip()

    # Convert user inputs to sets for O(1) matching. Lowercase the
    # name set (matching the canonicalize_skill convention).
    id_set = {x for x in user_skill_ids if isinstance(x, str) and x}
    name_set = {
        x.lower() for x in user_skill_names
        if isinstance(x, str) and x.strip()
    }
    canon_set = {
        x for x in user_skill_canon if isinstance(x, str) and x.strip()
    }
    # Build canon -> raw name lookup so matched rows can show the
    # user's resume wording in the "Your Skill" cell. When the
    # caller didn't pass the reverse map, fall back to canonical
    # form via canonicalize_skill round-trip.
    name_to_canon = user_skill_name_to_canon or {}

    # Slice 7a (2026-06-30) + Slice 8 (2026-06-30): read mode.
    # off -> exact-only cascade, no cosine, no LLM (legacy behavior).
    # log -> DEPRECATED post-slice-8, falls to off (no calibration
    #        needed once LLM gates ✓/✗; threshold loses meaning).
    # on  -> cosine pre-pass (signal) + batched LLM judgment (gate).
    try:
        from config import DRILLDOWN_SEMANTIC_MODE
        semantic_mode = DRILLDOWN_SEMANTIC_MODE
    except Exception:  # noqa: BLE001
        semantic_mode = "off"

    # Slice 8 hardening (2026-06-30): tri-state preserved.
    #   off → no semantic, no LLM
    #   log → cosine scored + Cartesian-logged, no LLM, no visible
    #         effect (same ✓/✗ as off; useful for debugging score
    #         distributions)
    #   on  → cosine candidates + one batched LLM judgment (LLM is
    #         the ✓/✗ gate)
    # No silent collapse of log → off. The tri-state user contract
    # holds.

    # Materialize user_skill_names as a stable list (semantic helper
    # consumes a list; existing exact-match cascade consumed sets).
    user_names_list = [
        x for x in user_skill_names if isinstance(x, str) and x.strip()
    ]

    # First pass: build prelim rows from exact cascade + (when mode=on)
    # cosine candidates. This produces:
    #   - row.matched from exact cascade (cosine still NOT a gate at
    #     this point)
    #   - row.your_skill_names from exact cascade only
    #   - cosine_candidates_by_row: for LLM payload
    #   - row_match_signal: 'exact' | 'cosine' | 'none' for LLM payload
    rows: list[RoleDrilldownSkillRow] = []
    row_match_signal: dict[str, str] = {}
    cosine_candidates_by_row: dict[str, list[dict]] = {}

    for raw in raw_rows[:_MAX_DRILLDOWN_ROWS]:
        if not isinstance(raw, dict):
            continue
        skill_id = raw.get("skill_id")
        skill_name = raw.get("skill_name")
        importance = raw.get("importance")
        # Slice 8 hardening (2026-06-30): description NO LONGER read
        # for cosine. Per the lock, "do not use OaSIS descriptions
        # as the matching anchor." The bare oasis_skill_name is
        # what gets embedded. (Description still flows from SQL but
        # is unused by this helper.)

        if not isinstance(skill_id, str) or not isinstance(skill_name, str):
            continue
        oasis_name_stripped = skill_name.strip()
        if not oasis_name_stripped:
            continue
        imp = float(importance) if isinstance(importance, (int, float)) else None

        # ----- Exact cascade: id -> canonical -> name -------------
        matched = False
        matched_user_names: list[str] = []
        if skill_id in id_set:
            matched = True
        oasis_canon = canonicalize_skill(oasis_name_stripped) or ""
        if not matched and oasis_canon and oasis_canon in canon_set:
            matched = True
        if not matched and oasis_name_stripped.lower() in name_set:
            matched = True

        if matched:
            row_match_signal[oasis_name_stripped] = "exact"
            for user_raw, user_canon in name_to_canon.items():
                if user_canon == oasis_canon:
                    matched_user_names.append(user_raw)
            if not matched_user_names:
                target_lower = oasis_name_stripped.lower()
                if target_lower in name_set:
                    matched_user_names.append(oasis_name_stripped)
            matched_user_names = sorted(set(matched_user_names))[
                :_MAX_YOUR_SKILL_NAMES_PER_ROW
            ]

        # ----- Slice 8 hardening (2026-06-30): cosine pre-pass -----
        # mode=on:  cosine produces candidates for the LLM payload.
        #           Cosine does NOT gate ✓/✗; LLM does.
        # mode=log: cosine computed + Cartesian-logged for debugging.
        #           NO LLM call. NO ✓/✗ effect (same visible result
        #           as off).
        # mode=off: cosine NOT computed; sem_pairs stays None.
        sem_pairs = None
        if semantic_mode != "off" and not matched and user_names_list:
            sem_pairs = _semantic_score_user_vs_oasis(
                user_skill_names=user_names_list,
                oasis_skill_name=oasis_name_stripped,
            )

        # Calibration log (slice 7a; preserved in slice 8 for log mode).
        if semantic_mode == "log" and sem_pairs:
            for u_name, score in sem_pairs:
                u_token = (
                    f"'{u_name}'"
                    if (" " in u_name or "\t" in u_name)
                    else u_name
                )
                log.info(
                    "drilldown_calibration noc=%s oasis_id=%s "
                    "oasis_name=%r user_name=%s score=%.3f",
                    code, skill_id, oasis_name_stripped,
                    u_token, score,
                )

        if not matched:
            # Build cosine_candidates for LLM payload (top-K above
            # threshold). Used by LLM as signal, not as the gate.
            candidates: list[dict] = []
            if sem_pairs:
                for u_name, score in sem_pairs[
                    :_MAX_COSINE_CANDIDATES_PER_ROW
                ]:
                    if score >= _DRILLDOWN_SEMANTIC_THRESHOLD:
                        candidates.append({
                            "user_skill": u_name,
                            "score": round(float(score), 3),
                        })
            cosine_candidates_by_row[oasis_name_stripped] = candidates
            row_match_signal[oasis_name_stripped] = (
                "cosine" if candidates else "none"
            )

        # ----- Training direction lookup (unchanged) ---------------
        training_provider: str | None = None
        training_url: str | None = None
        if not matched and registry is not None:
            try:
                resources = registry.surface_resources(
                    oasis_name_stripped, today=today,
                )
                for res in resources:
                    url = res.surface_url(today)
                    if url:
                        training_provider = res.provider
                        training_url = url
                        break
            except Exception:  # noqa: BLE001
                log.warning(
                    "recommender_drilldown_training_lookup_failed "
                    "skill=%r", oasis_name_stripped,
                )

        rows.append(RoleDrilldownSkillRow(
            oasis_skill_name=oasis_name_stripped,
            oasis_skill_id=skill_id,
            importance=imp,
            matched=matched,
            your_skill_names=tuple(matched_user_names),
            training_provider=training_provider,
            training_url=training_url,
            user_evidence=None,  # filled by LLM judgment below
            reason=None,
        ))

    # ----- Slice 8: batched LLM judgment over ALL rows --------------
    # Fires only in mode=on. Sees all 7 rows + their match_signal +
    # cosine_candidates + the user's full profile. Returns one
    # judgment per skill: matched (bool), user_evidence (string or
    # null), reason (string or null).
    #
    # The judgment can CONFIRM or REJECT a cosine candidate. It
    # cannot override an exact cascade match (those are locked at
    # match_signal=exact; LLM sees them but is instructed to pass
    # through).
    if semantic_mode == "on" and rows:
        # Build the noc_skillset payload (all rows, with match_signal
        # + cosine_candidates).
        noc_skillset = []
        for r in rows:
            signal = row_match_signal.get(r.oasis_skill_name, "none")
            candidates = (
                cosine_candidates_by_row.get(r.oasis_skill_name, [])
                if signal == "cosine" else []
            )
            noc_skillset.append({
                "skill": r.oasis_skill_name,
                "match_signal": signal,
                "cosine_candidates": candidates,
            })

        # Slice 8 hardening (2026-06-30): user_profile built from
        # explicit kwargs (replaces the slice-8 first-cut module-
        # level shim, which was unsafe under concurrent requests).
        user_profile = {
            "skills": list(user_names_list),
            "work_history": user_work_history or [],
            "education": user_education or [],
            "certifications": user_certifications or [],
        }

        judgments = _judge_drilldown_with_llm(
            noc_code=code,
            role_title=role_title,
            noc_skillset=noc_skillset,
            user_profile=user_profile,
        )

        if judgments is not None:
            # Merge LLM judgments into rows. Exact-cascade matches
            # are preserved by trusting the LLM's matched=true echo
            # (or, defensively, keeping matched=true even if LLM
            # said False for an exact row -- exact is locked).
            new_rows: list[RoleDrilldownSkillRow] = []
            for r in rows:
                j = judgments.get(r.oasis_skill_name)
                if j is None:
                    # LLM skipped this row -- keep cosine-only fallback.
                    new_rows.append(r)
                    continue
                signal = row_match_signal.get(r.oasis_skill_name, "none")
                # Exact cascade matches are locked (slice 8 spec §3
                # second paragraph). LLM can ADD evidence/reason but
                # cannot REJECT.
                final_matched = r.matched or bool(j["matched"])
                # If LLM returned evidence + reason, use them.
                # Otherwise (LLM said matched=False for an exact-row),
                # synthesize from the exact-cascade your_skill_names.
                ev = j.get("user_evidence")
                rs = j.get("reason")
                if final_matched and not ev and r.your_skill_names:
                    # Exact-cascade row with no LLM evidence: build
                    # a minimal evidence string from the matched
                    # user names so the cell isn't empty.
                    ev = ", ".join(r.your_skill_names)
                # If LLM said matched=False and we're NOT an exact row,
                # also clear training (it should ✗ "ask SCCC").
                training_provider = r.training_provider
                training_url = r.training_url
                if not final_matched and signal != "exact":
                    # Already set if registry hit; otherwise None.
                    pass

                new_rows.append(RoleDrilldownSkillRow(
                    oasis_skill_name=r.oasis_skill_name,
                    oasis_skill_id=r.oasis_skill_id,
                    importance=r.importance,
                    matched=final_matched,
                    your_skill_names=r.your_skill_names,
                    training_provider=training_provider,
                    training_url=training_url,
                    user_evidence=ev,
                    reason=rs,
                ))
            rows = new_rows
        else:
            # Slice 8 hardening (2026-06-30): LLM unavailable /
            # failed. CONSERVATIVE fallback (locked option 'a' at
            # sign-off): exact/canonical/name matches stay ✓; cosine-
            # only rows stay ✗. We do NOT promote weak cosine
            # candidates to ✓ without coach judgment -- 0.15 is too
            # weak to become user-visible truth.
            #
            # Rows already passed through the exact cascade above,
            # so r.matched=True iff exact/canonical/name hit. Cosine-
            # signal rows (match_signal=cosine, candidates >= 0.15)
            # stay at r.matched=False. Their training direction
            # falls to the existing gap-row path (registry hit or
            # "ask SCCC").
            log.info(
                "drilldown_llm_fallback noc=%s -- exact-only "
                "(no cosine->matched promotion)",
                code,
            )
            # rows stays as-is from the prelim pass. No change
            # needed -- exact matches already have matched=True;
            # cosine-only rows already have matched=False.

    # ----- Slice 9: Coach Training Guide -----------------------------
    coach_guide = None
    if not coach_guide_enabled and rows:
        # Observability lock: absence of any coach_guide_* log line
        # during rollout is ambiguous -- gate off vs code failure.
        # A single explicit line resolves that fast.
        log.info(
            "coach_guide_disabled_by_env noc=%s -- table renders alone",
            code,
        )
    if coach_guide_enabled and rows:
        matched_rows_list = [r for r in rows if r.matched]
        gap_rows_list = [r for r in rows if not r.matched]
        priority_gap_rows = _sort_rows_by_importance(gap_rows_list)[
            :_MAX_PRIORITY_GAPS
        ]

        if not priority_gap_rows:
            # Zero-gap branch: deterministic encouragement, no LLM.
            coach_guide = _build_zero_gap_encouragement(
                role_title=role_title,
                matched_rows=matched_rows_list,
            )
            log.info(
                "coach_guide_zero_gap noc=%s matched=%d",
                code, len(matched_rows_list),
            )
        else:
            # Gap>=1 branch: LLM coach pass.
            evidence_package = _build_coach_evidence_package(
                role_title=role_title,
                noc_code=code,
                matched_rows=matched_rows_list,
                priority_gap_rows=priority_gap_rows,
                registry=registry,
                today=today,
                user_skill_names=user_names_list,
                user_work_history=user_work_history or [],
                user_education=user_education or [],
                user_certifications=user_certifications or [],
            )
            coach_guide = _generate_coach_guide_with_llm(
                noc_code=code,
                evidence_package=evidence_package,
            )
            # coach_guide may be None if the LLM call failed; caller
            # renders the table alone in that case (no half-guide).

    return RoleDrilldownEvidence(
        noc_code=code,
        role_title=role_title,
        rows=tuple(rows),
        coach_guide=coach_guide,
    )


# Slice 8 hardening (2026-06-30): the transitional module-level
# _DRILLDOWN_USER_CONTEXT shim (with set_/clear_ helpers) was
# REMOVED. It was unsafe under concurrent requests -- one drilldown
# request could read another's context. The helper signature now
# accepts user_work_history / user_education / user_certifications
# as explicit kwargs (see build_recommender_evidence_role_drilldown
# signature above). Callers pass them per-request; no shared state.


# ---------------------------------------------------------------------------
# Slice 5: ordinal/name resolver for the adjacent_role_drilldown_select
# pending state. Maps the user's selection message to one of the NOCs
# in staged.last_recommender_adjacent_surface.
# ---------------------------------------------------------------------------
import re as _re

# Ordinal word -> 0-based index mapping. Matches whole tokens only
# (word-boundary) so "the third option" matches but "thirdsomething"
# doesn't.
#
# Deliberately exclude "one" / "two" / "three" as standalone words:
# they're filler in compound phrasings like "the second ONE", which
# would otherwise match BOTH "second" (index 1) AND "one" (index 0)
# -> ambiguous -> returns None instead of selecting index 1. Users
# who really want index 0 will say "first" or "1st" or "1".
_ORDINAL_PATTERNS: tuple[tuple[_re.Pattern[str], int], ...] = (
    (_re.compile(r"\b(?:first|1st|1)\b", _re.IGNORECASE), 0),
    (_re.compile(r"\b(?:second|2nd|2)\b", _re.IGNORECASE), 1),
    (_re.compile(r"\b(?:third|3rd|3)\b", _re.IGNORECASE), 2),
)


def resolve_drilldown_selection(
    user_message: str,
    surface: Iterable[dict],
) -> dict | None:
    """Slice 5: map a user message to one of the surfaced adjacent
    NOCs. Returns the matched surface entry dict, or None when the
    message doesn't unambiguously resolve.

    Match conditions (OR-ed):
      1. NOC code exact: "13110" -> matches surface entry with that
         noc_code
      2. Ordinal: "first", "second", "third", "1", "2", "3", "1st",
         "2nd", "3rd", "the first one", "option 2", etc. ->
         index into surface
      3. Title substring (case-insensitive): "sales manager" matches
         "Area sales manager"; "secretary" matches "Administrative
         secretary"

    Returns None when:
      - message is empty / non-str
      - no condition matches
      - multiple surface entries match ambiguously (e.g. "the
        assistant" when surface has two assistant titles)

    Caller (consume hook) treats None as ambiguous and re-prompts
    OR clears state and hands off based on consent classifier.
    """
    if not isinstance(user_message, str) or not user_message.strip():
        return None
    surface_list = [
        e for e in surface
        if isinstance(e, dict)
        and isinstance(e.get("noc_code"), str)
        and isinstance(e.get("title"), str)
    ]
    if not surface_list:
        return None
    msg = user_message.strip()

    # 1. NOC code exact match (highest priority).
    for entry in surface_list:
        if entry["noc_code"] in msg:
            return entry

    # 2. Ordinal match (single-position only; ambiguous if multiple
    # ordinals appear).
    ordinal_matches: list[int] = []
    for pattern, idx in _ORDINAL_PATTERNS:
        if pattern.search(msg):
            ordinal_matches.append(idx)
    if len(ordinal_matches) == 1:
        idx = ordinal_matches[0]
        if 0 <= idx < len(surface_list):
            return surface_list[idx]
    # If multiple ordinals match (e.g. "the first or second") OR an
    # ordinal points past the surface length, fall through to title
    # substring matching -- maybe the user disambiguated by name too.

    # 3. Title substring match (case-insensitive). The entry title
    # must appear as a substring of the user's message.
    #
    # Slice 5 hardening (2026-06-30): the OLD impl had a word-≥5-char
    # fallback that matched any title word in the message. Live verify
    # caught this firing too loosely: "construction site manager"
    # matched BOTH "Construction managers" (via "construction") AND
    # "Area sales manager" (via "manager") -> ambiguous -> None ->
    # consume hook cleared state -> drilldown lost. The fallback was
    # well-intentioned (paraphrase tolerance) but couldn't distinguish
    # a strong topical hit ("construction") from a noisy filler hit
    # ("manager"). The cleaner contract: require a real substring
    # match of the FULL title. If the user paraphrases (typed
    # "construction site manager" vs stored "Construction managers"),
    # the resolver returns None and the consume hook re-prompts with
    # explicit options.
    msg_lower = msg.lower()
    title_hits: list[dict] = []
    for entry in surface_list:
        title_lower = entry["title"].lower().strip()
        if not title_lower:
            continue
        if title_lower in msg_lower:
            title_hits.append(entry)

    if len(title_hits) == 1:
        return title_hits[0]
    # Multiple title hits OR none -> ambiguous.
    return None
