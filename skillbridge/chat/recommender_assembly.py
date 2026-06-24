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
