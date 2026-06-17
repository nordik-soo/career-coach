"""Sanitized responder view for AR-9.bug.2a (sub-step 3).

Materializes a closed-allowlist projection of the responder input with:
  - SanitizedURL items embedded on each projected entry (no parallel
    URL arrays);
  - Move-gated URL validation: only the URL occurrences the current
    decision's move can surface are validated;
  - Move-gated allowlists (prompt_urls, fallback_urls) derived from
    populated items;
  - Source-validation evidence retained as RejectedSourceURL entries
    keyed by structural occurrence path, for sub-step 5 telemetry.

This module does NOT touch responder.py. Projection helpers are
private duplicates of `_narration_skill_view` and
`_capped_score_explanation`; their parity with the responder originals
is verified by tests in `tests/test_url_views.py`. Sub-step 4 decides
whether to centralize the shared logic.

No consumer migration happens here. The view exists in isolation;
neither `compose_reply` nor `compose_response_v2` calls into this
module until sub-step 4 wires the call.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Mapping

from skillbridge.chat.url_policy import (
    Validated,
    Violation,
    validate,
)
from skillbridge.chat.tiered_evidence import (
    AdjacentJob,
    JobFacts,
    NonBlockingGap,
    PrioritizedGap,
    StretchMatch,
    StrongMatch,
    TieredEvidence,
    TrainingOption,
    TransferablePair,
)
from skillbridge.match.alignment import SkillAlignment

if TYPE_CHECKING:
    # Runtime-free imports: PEP 563 string annotations defer evaluation,
    # so the builders below take ResponderInput / ResponderV2Input
    # without creating a runtime import cycle with responder.py.
    from skillbridge.chat.responder import (
        ResponderInput,
        ResponderV2Input,
    )


# =========================================================================
# SanitizedURL — the only URL representation that flows through the view
# =========================================================================
@dataclass(frozen=True)
class SanitizedURL:
    """A URL that has passed every structural check in url_policy.validate.

    Construct via from_validated() in production. Direct construction is
    structurally available and is used by tests for fixture purposes.
    """

    raw: str
    canonical: str
    hash_sha256: str

    @classmethod
    def from_validated(cls, validated: Validated) -> "SanitizedURL":
        """Required production construction path.

        Captures the validated-source contract: raw, canonical, and hash
        all flow from one Validated object so the audit trail is
        consistent across components.
        """
        return cls(
            raw=validated.raw_token,
            canonical=validated.canonical,
            hash_sha256=validated.raw_token_hash,
        )


# =========================================================================
# RejectedSourceURL — evidence retained for sub-step 5 telemetry
# =========================================================================
@dataclass(frozen=True)
class RejectedSourceURL:
    """One source URL position that failed structural validation.

    The occurrence_path is a structural identifier matching the real
    responder input shape (e.g. 'results[3].url',
    'training_by_job[\"job-001\"][1].url',
    'adjacent_role_description_payload.job.url'). Per-response-composition
    dedup: at most one entry per occurrence_path per view construction.
    """

    occurrence_path: str
    violation: Violation


# =========================================================================
# Projected nested types under ScoreExplanationView
# =========================================================================
@dataclass(frozen=True)
class SkillBaseView:
    value: float | None
    mode: str | None
    required_match_ratio: float | None
    required_weight: float | None
    preferred_match_ratio: float | None
    preferred_weight: float | None


@dataclass(frozen=True)
class BoostsView:
    location: float | None
    recency: float | None
    target_role: float | None
    target_noc_match: float | None
    work_type_fit: float | None
    shift_fit: float | None


@dataclass(frozen=True)
class TitleMatchView:
    applied: bool | None
    raw_similarity: float | None


@dataclass(frozen=True)
class ScoreComponentsView:
    skill_base: SkillBaseView | None
    boosts: BoostsView | None
    title_match: TitleMatchView | None
    score_pre_caps: float | None
    score_post_caps: float | None


@dataclass(frozen=True)
class ScoreExplanationView:
    """Closed-allowlist projection of score_explanation.

    Fields code-anchored to match/engine.py:989-1073 (direct-title path)
    and match/engine.py:1272-1362 (main path) plus the cap-flag
    injections from _apply_hard_gates (match/engine.py:563-636).
    """

    matched_skills: tuple[str, ...]
    missing_skills: tuple[str, ...]
    required_matched: tuple[str, ...]
    required_missing: tuple[str, ...]
    preferred_matched: tuple[str, ...]
    preferred_missing: tuple[str, ...]
    required_match_strengths: tuple[float, ...]
    required_match_stages: tuple[str, ...]
    preferred_match_strengths: tuple[float, ...]
    preferred_match_stages: tuple[str, ...]
    required_match_strength_sum: float | None
    preferred_match_strength_sum: float | None
    skill_match_ratio: float | None
    required_match_ratio: float | None
    required_total: int | None
    preferred_match_ratio: float | None
    preferred_total: int | None
    title_match_similarity: float | None
    title_match_override: bool | None
    recency_days: int | None
    location_boosted: bool | None
    work_type_fit: str | None
    shift_fit: str | None
    credential_warning_present: bool | None
    credential_gap_skills: tuple[str, ...]
    work_type_user: str | None
    work_type_job: str | None
    # Cap flags: bool | None — None means absent from raw payload, so
    # sub-step 4's serializer can omit absent fields (preserving the
    # signal that no cap fired vs. an explicit False).
    band_capped_by_credential: bool | None
    band_capped_by_no_experience: bool | None
    band_capped_by_work_type_mismatch: bool | None
    caps_applied: tuple[str, ...]
    score_components: ScoreComponentsView | None


# =========================================================================
# Projected match-result types — separate prompt and fallback
# =========================================================================
@dataclass(frozen=True)
class PromptResultView:
    title: str
    employer: str | None
    url: SanitizedURL | None
    location: str | None
    match_band: str | None
    matched_skills: tuple[str, ...]
    missing_skills: tuple[str, ...]
    credential_warning: str | None
    score_explanation: ScoreExplanationView | None


@dataclass(frozen=True)
class FallbackResultView:
    job_id: str | None
    title: str | None
    employer: str | None
    url: SanitizedURL | None
    match_band: str | None
    credential_warning: str | None
    missing_skills: tuple[str, ...]


# =========================================================================
# Projected training type — full approved field union
# =========================================================================
@dataclass(frozen=True)
class TrainingView:
    provider: str | None
    title: str | None
    url: SanitizedURL | None
    for_skill: str | None
    duration_band: str | None
    resource_type: str | None
    reason: str | None
    type: str | None
    for_gap: str | None
    summary: str | None
    verified: bool | None


@dataclass(frozen=True)
class PromptJobTrainingGroup:
    """Bug.4 prompt-only ownership wrapper. Groups training resources
    by the owning job_id so the V2 LLM prompt carries per-job
    attribution structurally rather than as an instruction the model
    might ignore.

    Used only by the V2 present_matches prompt path. The fallback path
    iterates results directly (per-job attribution is already
    structural there), and non-present_matches prompt moves don't have
    the multi-job attribution problem (each narrates against a single
    role context).
    """
    job_id: str
    job_title: str | None
    resources: tuple[TrainingView, ...]


# =========================================================================
# Projected adjacent-recommendation types
# =========================================================================
@dataclass(frozen=True)
class PromptAdjacentRecommendationView:
    job_id: str | None
    title: str
    employer: str | None
    location: str | None
    evidence_summary: str | None
    why_adjacent: str | None
    matched_skills: tuple[str, ...]
    # No url field — bug.2a contract: zero URL surface on this projection.


@dataclass(frozen=True)
class PromptAdjacentRecommendationsContainerView:
    """Payload-level container for the recommend_adjacent_roles prompt.

    The current prompt at responder.py:1441-1444 emits both the
    recommendations list AND total_retrieved. Sub-step 4 cannot
    reproduce the prompt output if only the recommendation tuple is
    available, so total_retrieved is projected alongside.

    total_retrieved is `int | None`: None means the field was absent
    from the raw payload; sub-step 4's serializer decides whether to
    emit 0 (matching current `int(... or 0)` behavior) or omit.

    The four total_dropped_by_* fields on the raw payload are
    deliberately NOT projected — they are not surfaced in the current
    prompt (responder.py:1441-1444), so the dataclass-as-allowlist
    rule excludes them. Adding any of them is a sub-step 4 decision
    if a new consumer needs them.
    """

    recommendations: tuple[PromptAdjacentRecommendationView, ...]
    total_retrieved: int | None


@dataclass(frozen=True)
class FallbackAdjacentRecommendationView:
    title: str
    employer: str | None
    evidence_summary: str | None
    # No url field — bug.2a contract.


# =========================================================================
# Projected adjacent-role-description types
# =========================================================================
@dataclass(frozen=True)
class PromptAdjacentRoleView:
    job_id: str | None
    title: str | None
    employer: str | None
    location: str | None
    posted_date: str | None
    url: SanitizedURL | None
    evidence_summary: str | None
    matched_skills: tuple[str, ...]
    expired: bool
    # True iff the raw payload's `job` value was a Mapping (including
    # empty dict {}). False when missing, None, or non-dict. Drives the
    # serializer's "job: null" vs "job: {6 null fields}" emission per
    # the locked sub-step 4 contract (matches responder.py:1455's
    # `isinstance(job, dict)` check).
    job_is_mapping: bool


@dataclass(frozen=True)
class FallbackAdjacentRoleView:
    job_id: str | None
    title: str | None
    employer: str | None
    location: str | None
    evidence_summary: str | None
    matched_skills: tuple[str, ...]
    expired: bool
    has_validated_url: bool
    # True iff the raw payload's `job` value was a Mapping (including
    # empty dict {}). False when missing, None, or non-dict. Matches
    # the original responder.py:1787's `isinstance(job, dict)` check
    # so the fallback renderer can distinguish "no job dict" from
    # "job dict with only employer/location populated".
    job_is_mapping: bool
    # Bug.2b: render the validated adjacent-role posting URL inline
    # in the deterministic fallback. The url is the same SanitizedURL
    # that drives `has_validated_url`; sub-step 3's bug.2a lock
    # discarded it after deriving the boolean. Bug.2b's narrow scope
    # is exactly to surface that already-validated URL where the user
    # needs it. `None` when validation failed or no url was on the
    # raw payload.
    url: "SanitizedURL | None"


# =========================================================================
# Projected coach-tier types (AR-9.feat.coach-tiers step 9)
#
# These view-side dataclasses are the SOLE shape the responder and
# deterministic fallback see for tier evidence. They:
#   - Promote `Validated` URL slots → `SanitizedURL` at construction
#     time. The view never carries a raw `Validated` (or a raw `str`)
#     URL.
#   - Carry exactly the fields the coach prompt and the deterministic
#     fallback need. No raw `MatchResult`, no raw job dict, no internal
#     matcher field leaks here.
#   - Mirror `tiered_evidence.*` shapes so the projection is mechanical
#     and provenance-preserving.
# =========================================================================
_TIER_STRENGTH_CLAIM_TOKENS = (
    "competitive_match",
    "strongest_current",
    "close_with_named_gap",
    "stretch_with_training_bridge",
    "transferable_lane",
)


@dataclass(frozen=True)
class PromptJobFacts:
    """View-side mirror of `tiered_evidence.JobFacts`. Scalars only —
    no URL fields. `posted_days_ago` is derived upstream in
    tiered_evidence.
    """
    posted_date: date | None
    posted_days_ago: int | None
    location: str | None
    employment_type: str | None
    salary_text: str | None


@dataclass(frozen=True)
class PromptTrainingOption:
    """View-side mirror of `tiered_evidence.TrainingOption`. `url`
    has been promoted from `Validated` → `SanitizedURL` here. The
    fallback URL allowlist treats this URL as a renderable.
    """
    provider: str
    title: str
    url: SanitizedURL | None
    format: Literal["online", "in-person", "hybrid"] | None
    duration_text: str | None


@dataclass(frozen=True)
class PromptPrioritizedGap:
    """View-side mirror of `tiered_evidence.PrioritizedGap`."""
    job_requirement: str
    category: Literal["required", "preferred"]
    priority: int
    blocker: bool
    training_options: tuple[PromptTrainingOption, ...]


@dataclass(frozen=True)
class PromptNonBlockingGap:
    """View-side mirror of `tiered_evidence.NonBlockingGap`."""
    job_requirement: str
    material: bool


@dataclass(frozen=True)
class PromptTransferablePair:
    """View-side mirror of `tiered_evidence.TransferablePair`."""
    user_skill: str
    applies_to: str
    stage: Literal["exact", "fuzzy", "semantic"]


_TierStrengthClaim = Literal[
    "competitive_match",
    "strongest_current",
    "close_with_named_gap",
    "stretch_with_training_bridge",
    "transferable_lane",
]


@dataclass(frozen=True)
class PromptStrongMatch:
    """Apply-today tier projection. The responder reads only these
    fields; raw MatchResult and raw job dicts never reach this layer.
    """
    job_id: str
    title: str
    employer: str | None
    location: str | None
    noc_code: str | None
    url: SanitizedURL | None
    job_facts: PromptJobFacts
    skill_alignment: tuple[SkillAlignment, ...]
    non_blocking_gaps: tuple[PromptNonBlockingGap, ...]
    credential_warning_text: str | None
    strength_claim_text: _TierStrengthClaim


@dataclass(frozen=True)
class PromptStretchMatch:
    """Worth-a-try tier projection."""
    job_id: str
    title: str
    employer: str | None
    location: str | None
    noc_code: str | None
    url: SanitizedURL | None
    job_facts: PromptJobFacts
    skill_alignment: tuple[SkillAlignment, ...]
    prioritized_gaps: tuple[PromptPrioritizedGap, ...]
    credential_warning_text: str | None
    strength_claim_text: _TierStrengthClaim


@dataclass(frozen=True)
class PromptAdjacentJob:
    """Sideways-move tier projection."""
    job_id: str
    title: str
    employer: str | None
    location: str | None
    noc_code: str | None
    url: SanitizedURL | None
    job_facts: PromptJobFacts
    skill_alignment: tuple[SkillAlignment, ...]
    transferable_pairs: tuple[PromptTransferablePair, ...]
    important_gaps: tuple[str, ...]
    credential_warning_text: str | None
    why_adjacent: Literal["same_noc_minor_group", "skill_evidence"]
    strength_claim_text: _TierStrengthClaim


# =========================================================================
# SanitizedResponderView — the materialized projection
# =========================================================================
@dataclass(frozen=True)
class SanitizedResponderView:
    """Move-gated projection of the responder input.

    Only slots for the current decision's move are populated; everything
    else is empty / None. Allowlists (prompt_urls, fallback_urls) derive
    from populated items' canonical URLs.
    """

    # Present matches
    prompt_results: tuple[PromptResultView, ...]
    fallback_results: tuple[FallbackResultView, ...]
    # v1 path: legacy flat training (preserves first-6-across-
    # training_by_job.values() iteration, no orphan removal, no
    # result-order grouping). Only the v1 builder populates this slot.
    prompt_present_matches_training_flat: tuple[TrainingView, ...]
    # v2 path (bug.4): training grouped by owning job_id, result-ordered,
    # orphan-dropped (training_by_job entries for jobs not in results[:5]
    # are excluded), and capped at 6 total resources across groups.
    # Only the v2 builder populates this slot.
    prompt_present_matches_training_groups: tuple[PromptJobTrainingGroup, ...]
    fallback_present_matches_training_by_job: Mapping[
        str, tuple[TrainingView, ...]
    ]
    # Explain gap
    prompt_explain_gap_training_flat: tuple[TrainingView, ...]
    fallback_explain_gap_training_flat: tuple[TrainingView, ...]
    # Present near miss (prompt only — fallback has no URL surface)
    prompt_present_near_miss_training_flat: tuple[TrainingView, ...]
    # Explain remaining gaps (prompt only)
    prompt_explain_remaining_gaps_training_flat: tuple[TrainingView, ...]
    # Adjacent recommendation (0 URLs)
    # Prompt slot is a container carrying recommendations + total_retrieved
    # so sub-step 4 can reproduce responder.py:1441-1444 verbatim. None when
    # the payload is absent (any move other than recommend_adjacent_roles
    # with payload set).
    prompt_adjacent_recommendations: PromptAdjacentRecommendationsContainerView | None
    fallback_adjacent_recommendations: tuple[
        FallbackAdjacentRecommendationView, ...
    ]
    # Adjacent role description
    prompt_adjacent_role: PromptAdjacentRoleView | None
    fallback_adjacent_role: FallbackAdjacentRoleView | None
    # Source-validation evidence
    rejected_source_urls: tuple[RejectedSourceURL, ...]
    # Derived move-gated allowlists
    prompt_urls: frozenset[str]
    fallback_urls: frozenset[str]
    # AR-9.feat.coach-tiers step 9: tier projections for the
    # present_tiered_matches move. Default-empty so the existing five
    # builders construct correctly without modification (move gating:
    # tier data exists ONLY when the new
    # `build_sanitized_responder_view_for_tiered_matches` builder ran).
    prompt_tiered_apply_today: tuple[PromptStrongMatch, ...] = field(
        default_factory=tuple
    )
    prompt_tiered_worth_a_try: tuple[PromptStretchMatch, ...] = field(
        default_factory=tuple
    )
    prompt_tiered_sideways_move: tuple[PromptAdjacentJob, ...] = field(
        default_factory=tuple
    )

    def __post_init__(self) -> None:
        # Enforce that the Mapping field is structurally immutable.
        if not isinstance(
            self.fallback_present_matches_training_by_job, MappingProxyType
        ):
            raise TypeError(
                "fallback_present_matches_training_by_job must be "
                "wrapped in types.MappingProxyType"
            )


# =========================================================================
# Private projection helpers — parity-tested against responder.py
# =========================================================================
_NARRATION_SKILL_CAP = 3


def _is_credential_skill_name_for_view(name: str | None) -> bool:
    """Local reproduction of match.engine.is_credential_skill_name's
    contract for the narration cap's force-include rule.

    Imported at runtime from match.engine (not from responder.py) — the
    circular-import risk is with responder.py, not the engine module.
    The narration cap rule is: keep credentials beyond the top-N cap.
    """
    # Defer import to call site to avoid module-load-order issues.
    from skillbridge.match.engine import is_credential_skill_name
    return is_credential_skill_name(name)


def _project_narration_skills(
    skills: list[str] | tuple[str, ...] | None,
) -> tuple[str, ...]:
    """Private projection of responder._narration_skill_view.

    Behavior parity with responder._narration_skill_view, verified by
    tests/test_url_views.py::test_narration_skill_parity. Returns a
    tuple (not a list) so the projected view stays immutable.

    Sub-step 4 may decide to centralize this helper in a
    dependency-neutral module — until then, projection happens here
    and the parity test guards the equivalence.
    """
    if not skills:
        return ()
    seq = list(skills)
    top = list(seq[:_NARRATION_SKILL_CAP])
    seen = set(top)
    credentials_below_cap = [
        s for s in seq[_NARRATION_SKILL_CAP:]
        if _is_credential_skill_name_for_view(s) and s not in seen
    ]
    return tuple(top + credentials_below_cap)


def _project_narration_skills_with_indices(
    skills: list[str] | tuple[str, ...] | None,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Private projection of responder._narration_skill_view_with_indices.

    Returns kept names + their source indices so parallel arrays
    (required_match_strengths / required_match_stages) can be sliced
    to stay index-aligned with required_matched / preferred_matched.
    """
    if not skills:
        return ((), ())
    seq = list(skills)
    kept_indices = list(range(min(_NARRATION_SKILL_CAP, len(seq))))
    seen = set(seq[:_NARRATION_SKILL_CAP])
    for i in range(_NARRATION_SKILL_CAP, len(seq)):
        name = seq[i]
        if _is_credential_skill_name_for_view(name) and name not in seen:
            kept_indices.append(i)
            seen.add(name)
    kept_names = tuple(seq[i] for i in kept_indices)
    return kept_names, tuple(kept_indices)


def _project_cap_flag(raw_value: object) -> bool | None:
    """Strict bool/None projection of a band_capped_by_* flag.

    - True (the bool) -> True
    - False (the bool) -> False
    - None or any non-bool value -> None

    isinstance(value, bool) excludes int 1/0 because despite bool being
    a subclass of int, the inverse check is one-way: True is bool, but
    1 is not. This preserves the engine's absence-vs-explicit-value
    distinction without coercing arbitrary truthy values.
    """
    if isinstance(raw_value, bool):
        return raw_value
    return None


def _project_string_tuple(value: object) -> tuple[str, ...]:
    """Coerce a list-like field of strings into a tuple, silently
    dropping non-string entries. Used wherever the engine emits a list
    of skill names.
    """
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(v for v in value if isinstance(v, str))


def _project_float_tuple(value: object) -> tuple[float, ...]:
    """Coerce a list of numeric values into a tuple of floats, silently
    dropping non-numeric entries.
    """
    if not isinstance(value, (list, tuple)):
        return ()
    out: list[float] = []
    for v in value:
        if isinstance(v, bool):
            # bool is a subclass of int; explicitly skip booleans here
            continue
        if isinstance(v, (int, float)):
            out.append(float(v))
    return tuple(out)


def _project_float_or_none(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _project_int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _project_str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _project_bool_or_none(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _project_skill_base(raw: object) -> SkillBaseView | None:
    if not isinstance(raw, dict):
        return None
    return SkillBaseView(
        value=_project_float_or_none(raw.get("value")),
        mode=_project_str_or_none(raw.get("mode")),
        required_match_ratio=_project_float_or_none(
            raw.get("required_match_ratio")
        ),
        required_weight=_project_float_or_none(raw.get("required_weight")),
        preferred_match_ratio=_project_float_or_none(
            raw.get("preferred_match_ratio")
        ),
        preferred_weight=_project_float_or_none(raw.get("preferred_weight")),
    )


def _project_boosts(raw: object) -> BoostsView | None:
    if not isinstance(raw, dict):
        return None
    return BoostsView(
        location=_project_float_or_none(raw.get("location")),
        recency=_project_float_or_none(raw.get("recency")),
        target_role=_project_float_or_none(raw.get("target_role")),
        target_noc_match=_project_float_or_none(raw.get("target_noc_match")),
        work_type_fit=_project_float_or_none(raw.get("work_type_fit")),
        shift_fit=_project_float_or_none(raw.get("shift_fit")),
    )


def _project_title_match(raw: object) -> TitleMatchView | None:
    if not isinstance(raw, dict):
        return None
    return TitleMatchView(
        applied=_project_bool_or_none(raw.get("applied")),
        raw_similarity=_project_float_or_none(raw.get("raw_similarity")),
    )


def _project_score_components(raw: object) -> ScoreComponentsView | None:
    if not isinstance(raw, dict):
        return None
    return ScoreComponentsView(
        skill_base=_project_skill_base(raw.get("skill_base")),
        boosts=_project_boosts(raw.get("boosts")),
        title_match=_project_title_match(raw.get("title_match")),
        score_pre_caps=_project_float_or_none(raw.get("score_pre_caps")),
        score_post_caps=_project_float_or_none(raw.get("score_post_caps")),
    )


def _project_score_explanation(
    se: dict | None,
) -> ScoreExplanationView | None:
    """Private projection of responder._capped_score_explanation +
    type projection into ScoreExplanationView.

    Behavior parity for the cap rules verified by
    tests/test_url_views.py::test_score_explanation_parity.
    """
    if not se:
        return None

    # Cap the matched/missing/required_missing/preferred_missing lists
    # using the narration view (matches responder._capped_score_explanation
    # at responder.py:172-180).
    matched_skills = _project_narration_skills(se.get("matched_skills"))
    missing_skills = _project_narration_skills(se.get("missing_skills"))
    required_missing = _project_narration_skills(se.get("required_missing"))
    preferred_missing = _project_narration_skills(se.get("preferred_missing"))

    # Cap the required_matched / preferred_matched lists with kept-indices
    # so parallel arrays (strengths, stages) stay aligned (matches
    # responder._capped_score_explanation at lines 184-196).
    required_matched, req_kept_idx = _project_narration_skills_with_indices(
        se.get("required_matched"),
    )
    preferred_matched, pref_kept_idx = _project_narration_skills_with_indices(
        se.get("preferred_matched"),
    )

    def _slice_parallel(key: str, kept: tuple[int, ...]) -> tuple[float, ...]:
        raw = se.get(key)
        if not isinstance(raw, (list, tuple)):
            return ()
        return tuple(
            float(raw[i]) for i in kept
            if i < len(raw)
            and isinstance(raw[i], (int, float))
            and not isinstance(raw[i], bool)
        )

    def _slice_parallel_str(key: str, kept: tuple[int, ...]) -> tuple[str, ...]:
        raw = se.get(key)
        if not isinstance(raw, (list, tuple)):
            return ()
        return tuple(
            raw[i] for i in kept
            if i < len(raw) and isinstance(raw[i], str)
        )

    return ScoreExplanationView(
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        required_matched=required_matched,
        required_missing=required_missing,
        preferred_matched=preferred_matched,
        preferred_missing=preferred_missing,
        required_match_strengths=_slice_parallel(
            "required_match_strengths", req_kept_idx,
        ),
        required_match_stages=_slice_parallel_str(
            "required_match_stages", req_kept_idx,
        ),
        preferred_match_strengths=_slice_parallel(
            "preferred_match_strengths", pref_kept_idx,
        ),
        preferred_match_stages=_slice_parallel_str(
            "preferred_match_stages", pref_kept_idx,
        ),
        required_match_strength_sum=_project_float_or_none(
            se.get("required_match_strength_sum"),
        ),
        preferred_match_strength_sum=_project_float_or_none(
            se.get("preferred_match_strength_sum"),
        ),
        skill_match_ratio=_project_float_or_none(se.get("skill_match_ratio")),
        required_match_ratio=_project_float_or_none(
            se.get("required_match_ratio"),
        ),
        required_total=_project_int_or_none(se.get("required_total")),
        preferred_match_ratio=_project_float_or_none(
            se.get("preferred_match_ratio"),
        ),
        preferred_total=_project_int_or_none(se.get("preferred_total")),
        title_match_similarity=_project_float_or_none(
            se.get("title_match_similarity"),
        ),
        title_match_override=_project_bool_or_none(
            se.get("title_match_override"),
        ),
        recency_days=_project_int_or_none(se.get("recency_days")),
        location_boosted=_project_bool_or_none(se.get("location_boosted")),
        work_type_fit=_project_str_or_none(se.get("work_type_fit")),
        shift_fit=_project_str_or_none(se.get("shift_fit")),
        credential_warning_present=_project_bool_or_none(
            se.get("credential_warning_present"),
        ),
        credential_gap_skills=_project_string_tuple(
            se.get("credential_gap_skills"),
        ),
        work_type_user=_project_str_or_none(se.get("work_type_user")),
        work_type_job=_project_str_or_none(se.get("work_type_job")),
        band_capped_by_credential=_project_cap_flag(
            se.get("band_capped_by_credential"),
        ),
        band_capped_by_no_experience=_project_cap_flag(
            se.get("band_capped_by_no_experience"),
        ),
        band_capped_by_work_type_mismatch=_project_cap_flag(
            se.get("band_capped_by_work_type_mismatch"),
        ),
        caps_applied=_project_string_tuple(se.get("caps_applied")),
        score_components=_project_score_components(se.get("score_components")),
    )


# =========================================================================
# Move-gated URL enumeration + validation
# =========================================================================
def _enumerate_url_occurrence(
    raw_url: object,
    path: str,
    accumulator: list[tuple[str, str]],
) -> None:
    """Add (path, raw_url) to the validation set, applying Lock D.

    - None or empty string: absent, skipped (no Violation produced).
    - Non-empty string: enqueued for validate().
    - Any other type (int, list, dict, bool, etc.): suppressed (no
      Violation; hashing requires a string).
    """
    if raw_url is None or raw_url == "":
        return
    if not isinstance(raw_url, str):
        return
    accumulator.append((path, raw_url))


def _validate_occurrences(
    occurrences: list[tuple[str, str]],
) -> dict[str, Violation | Validated]:
    """Validate each occurrence once, return a dict keyed by path."""
    out: dict[str, Violation | Validated] = {}
    for path, raw in occurrences:
        out[path] = validate(raw)
    return out


def _sanitized(
    path: str, validations: dict[str, Violation | Validated],
) -> SanitizedURL | None:
    """Look up the cached validation result by path; return a
    SanitizedURL only when Validated.
    """
    v = validations.get(path)
    if isinstance(v, Validated):
        return SanitizedURL.from_validated(v)
    return None


def _build_rejected_list(
    validations: dict[str, Violation | Validated],
) -> tuple[RejectedSourceURL, ...]:
    """Build the per-composition rejected list from the validation cache."""
    return tuple(
        RejectedSourceURL(occurrence_path=path, violation=v)
        for path, v in validations.items() if isinstance(v, Violation)
    )


# =========================================================================
# Item-level projection helpers
# =========================================================================
def _project_prompt_result(
    r: dict, idx: int, validations: dict[str, Violation | Validated],
) -> PromptResultView:
    return PromptResultView(
        title=(r.get("title") if isinstance(r.get("title"), str) else "") or "",
        employer=_project_str_or_none(r.get("employer")),
        url=_sanitized(f"results[{idx}].url", validations),
        location=_project_str_or_none(r.get("location")),
        match_band=_project_str_or_none(r.get("match_band")),
        matched_skills=_project_narration_skills(r.get("matched_skills")),
        missing_skills=_project_narration_skills(r.get("missing_skills")),
        credential_warning=_project_str_or_none(r.get("credential_warning")),
        score_explanation=_project_score_explanation(r.get("score_explanation")),
    )


def _project_fallback_result(
    r: dict, idx: int, validations: dict[str, Violation | Validated],
) -> FallbackResultView:
    return FallbackResultView(
        job_id=_project_str_or_none(r.get("job_id")),
        title=_project_str_or_none(r.get("title")),
        employer=_project_str_or_none(r.get("employer")),
        url=_sanitized(f"results[{idx}].url", validations),
        match_band=_project_str_or_none(r.get("match_band")),
        credential_warning=_project_str_or_none(r.get("credential_warning")),
        missing_skills=_project_narration_skills(r.get("missing_skills")),
    )


def _project_training(
    t: dict, path: str, validations: dict[str, Violation | Validated],
) -> TrainingView:
    return TrainingView(
        provider=_project_str_or_none(t.get("provider")),
        title=_project_str_or_none(t.get("title")),
        url=_sanitized(path, validations),
        for_skill=_project_str_or_none(t.get("for_skill")),
        duration_band=_project_str_or_none(t.get("duration_band")),
        resource_type=_project_str_or_none(t.get("resource_type")),
        reason=_project_str_or_none(t.get("reason")),
        type=_project_str_or_none(t.get("type")),
        for_gap=_project_str_or_none(t.get("for_gap")),
        summary=_project_str_or_none(t.get("summary")),
        verified=_project_bool_or_none(t.get("verified")),
    )


def _project_prompt_adjacent_recommendation(
    r: dict,
) -> PromptAdjacentRecommendationView:
    return PromptAdjacentRecommendationView(
        job_id=_project_str_or_none(r.get("job_id")),
        title=(r.get("title") if isinstance(r.get("title"), str) else "") or "",
        employer=_project_str_or_none(r.get("employer")),
        location=_project_str_or_none(r.get("location")),
        evidence_summary=_project_str_or_none(r.get("evidence_summary")),
        why_adjacent=_project_str_or_none(r.get("why_adjacent")),
        matched_skills=_project_string_tuple(r.get("matched_skills")),
    )


def _project_fallback_adjacent_recommendation(
    r: dict,
) -> FallbackAdjacentRecommendationView:
    return FallbackAdjacentRecommendationView(
        title=(r.get("title") if isinstance(r.get("title"), str) else "") or "",
        employer=_project_str_or_none(r.get("employer")),
        evidence_summary=_project_str_or_none(r.get("evidence_summary")),
    )


def _project_prompt_adjacent_role(
    payload: dict, validations: dict[str, Violation | Validated],
) -> PromptAdjacentRoleView:
    """Always returns a view when payload is a dict. When `job` is
    absent or not a dict, job-derived fields are None but the
    payload-level fields (evidence_summary, matched_skills, expired)
    are preserved.

    Matches the v2 prompt at responder.py:1462-1467 where `job: null`
    is emitted when job is missing — the LLM still sees the
    evidence_summary, matched_skills, and especially expired:true.
    """
    job_raw = payload.get("job")
    job = job_raw if isinstance(job_raw, dict) else None

    if job is not None:
        posted = job.get("posted_date")
        if isinstance(posted, str):
            posted_str: str | None = posted
        elif posted is None:
            posted_str = None
        else:
            # date or datetime — coerce to string to match the v2 prompt's
            # job_safe['posted_date'] = str(pd) pattern at responder.py:1459.
            posted_str = str(posted)
        job_id = _project_str_or_none(job.get("job_id"))
        title = _project_str_or_none(job.get("title"))
        employer = _project_str_or_none(job.get("employer"))
        location = _project_str_or_none(job.get("location"))
        sanitized_url = _sanitized(
            "adjacent_role_description_payload.job.url", validations,
        )
    else:
        posted_str = None
        job_id = None
        title = None
        employer = None
        location = None
        sanitized_url = None

    return PromptAdjacentRoleView(
        job_id=job_id,
        title=title,
        employer=employer,
        location=location,
        posted_date=posted_str,
        url=sanitized_url,
        evidence_summary=_project_str_or_none(payload.get("evidence_summary")),
        matched_skills=_project_string_tuple(payload.get("matched_skills")),
        expired=bool(payload.get("expired")),
        job_is_mapping=isinstance(payload.get("job"), dict),
    )


def _project_fallback_adjacent_role(
    payload: dict, validations: dict[str, Violation | Validated],
) -> FallbackAdjacentRoleView:
    """Always returns a view when payload is a dict. When `job` is
    absent, job-derived fields are None and has_validated_url is False.
    Payload-level fields (evidence_summary, matched_skills, expired)
    are preserved so the fallback renderer can still branch on
    expired=True even without a job dict.
    """
    job_raw = payload.get("job")
    job = job_raw if isinstance(job_raw, dict) else None

    if job is not None:
        job_id = _project_str_or_none(job.get("job_id"))
        title = _project_str_or_none(job.get("title"))
        employer = _project_str_or_none(job.get("employer"))
        location = _project_str_or_none(job.get("location"))
        sanitized_url = _sanitized(
            "adjacent_role_description_payload.job.url", validations,
        )
    else:
        job_id = None
        title = None
        employer = None
        location = None
        sanitized_url = None

    return FallbackAdjacentRoleView(
        job_id=job_id,
        title=title,
        employer=employer,
        location=location,
        evidence_summary=_project_str_or_none(payload.get("evidence_summary")),
        matched_skills=_project_string_tuple(payload.get("matched_skills")),
        expired=bool(payload.get("expired")),
        has_validated_url=sanitized_url is not None,
        job_is_mapping=isinstance(payload.get("job"), dict),
        url=sanitized_url,
    )


# =========================================================================
# Allowlist derivation
# =========================================================================
def _collect_canonical_urls(*sources: object) -> frozenset[str]:
    """Union the .url.canonical across a sequence of projected items
    or single-item slots. Items with url=None contribute nothing.

    Supported source shapes:
      - None (skipped)
      - list/tuple of items with .url
      - list/tuple of PromptJobTrainingGroup (walks group.resources)
      - Mapping with list/tuple values of items with .url
      - Single item with .url
    """
    out: set[str] = set()
    for source in sources:
        if source is None:
            continue
        if isinstance(source, (list, tuple)):
            for item in source:
                # Bug.4: training groups carry their resources as a
                # nested tuple of TrainingView items. Walk into the
                # group to reach the .url on each resource.
                if isinstance(item, PromptJobTrainingGroup):
                    for resource in item.resources:
                        url = getattr(resource, "url", None)
                        if isinstance(url, SanitizedURL):
                            out.add(url.canonical)
                    continue
                # AR-9.feat.coach-tiers step 9: Worth-a-try records
                # carry training URLs nested in prioritized_gaps.
                # Walk them in addition to the job-URL on the item.
                if isinstance(item, PromptStretchMatch):
                    if isinstance(item.url, SanitizedURL):
                        out.add(item.url.canonical)
                    for gap in item.prioritized_gaps:
                        for opt in gap.training_options:
                            if isinstance(opt.url, SanitizedURL):
                                out.add(opt.url.canonical)
                    continue
                url = getattr(item, "url", None)
                if isinstance(url, SanitizedURL):
                    out.add(url.canonical)
        elif isinstance(source, Mapping):
            for value in source.values():
                if isinstance(value, (list, tuple)):
                    for item in value:
                        url = getattr(item, "url", None)
                        if isinstance(url, SanitizedURL):
                            out.add(url.canonical)
        else:
            url = getattr(source, "url", None)
            if isinstance(url, SanitizedURL):
                out.add(url.canonical)
    return frozenset(out)


# =========================================================================
# Empty-view helper — used for moves with no URL surface
# =========================================================================
def _empty_view(
    rejected: tuple[RejectedSourceURL, ...] = (),
) -> SanitizedResponderView:
    return SanitizedResponderView(
        prompt_results=(),
        fallback_results=(),
        prompt_present_matches_training_flat=(),
        prompt_present_matches_training_groups=(),
        fallback_present_matches_training_by_job=MappingProxyType({}),
        prompt_explain_gap_training_flat=(),
        fallback_explain_gap_training_flat=(),
        prompt_present_near_miss_training_flat=(),
        prompt_explain_remaining_gaps_training_flat=(),
        prompt_adjacent_recommendations=None,
        fallback_adjacent_recommendations=(),
        prompt_adjacent_role=None,
        fallback_adjacent_role=None,
        rejected_source_urls=rejected,
        prompt_urls=frozenset(),
        fallback_urls=frozenset(),
    )


# =========================================================================
# v1 builder
# =========================================================================
def build_sanitized_responder_view_v1(
    inp: "ResponderInput",
) -> SanitizedResponderView:
    """Build a SanitizedResponderView from a v1 ResponderInput.

    v1 only surfaces present_matches URLs (results + training). All
    adjacency / near-miss / remaining-gaps slots are empty because v1
    doesn't have those moves or payloads.

    Gated by `decision.show_matches`: when False, the view is empty
    (no validation, no rejections, no allowlist contributions).
    """
    decision = inp.decision
    if not getattr(decision, "show_matches", False):
        return _empty_view()

    return _build_present_matches_view(inp)


# =========================================================================
# v2 builder
# =========================================================================
def build_sanitized_responder_view_v2(
    inp: "ResponderV2Input",
) -> SanitizedResponderView:
    """Build a SanitizedResponderView from a v2 ResponderV2Input.

    Dispatches on `decision.final_move`:

      present_matches        -> results + training (per-job for fallback,
                                flat for prompt)
      explain_gap            -> training (flat for both surfaces)
      present_near_miss      -> training prompt only (fallback has no URL)
      explain_remaining_gaps -> training prompt only (fallback has no URL)
      recommend_adjacent_roles -> recommendations (0 URLs, projection only)
      describe_adjacent_role -> live job URL (single)
      (any other)            -> empty view
    """
    move = getattr(inp.decision, "final_move", None)

    if move == "present_matches":
        return _build_present_matches_view(inp, is_v2=True)
    if move == "explain_gap":
        return _build_explain_gap_view(inp)
    if move == "present_near_miss":
        return _build_present_near_miss_view(inp)
    if move == "explain_remaining_gaps":
        return _build_explain_remaining_gaps_view(inp)
    if move == "recommend_adjacent_roles":
        return _build_recommend_adjacent_roles_view(inp)
    if move == "describe_adjacent_role":
        return _build_describe_adjacent_role_view(inp)
    return _empty_view()


# =========================================================================
# Per-move construction (shared by v1/v2 where applicable)
# =========================================================================
def _flat_training_with_paths(
    training_by_job: Any,
) -> list[tuple[str, int, dict]]:
    """Return a flat ordered list of (job_id, index, training_dict) for
    every entry in training_by_job. Skips non-string job_ids and
    non-dict training entries silently.
    """
    out: list[tuple[str, int, dict]] = []
    if not isinstance(training_by_job, Mapping):
        return out
    for job_id, ts in training_by_job.items():
        if not isinstance(job_id, str):
            continue
        if not isinstance(ts, (list, tuple)):
            continue
        for i, t in enumerate(ts):
            if isinstance(t, dict):
                out.append((job_id, i, t))
    return out


def _build_present_matches_view(
    inp: Any, is_v2: bool = False,
) -> SanitizedResponderView:
    """Build the present_matches view.

    Sub-step 3 shared logic + bug.4 version-aware training shape:
      - v1 path (is_v2=False): populates
        `prompt_present_matches_training_flat` with the legacy first-6-
        across-training_by_job.values() iteration. No orphan removal,
        no result-order grouping. Leaves
        `prompt_present_matches_training_groups` as ().
      - v2 path (is_v2=True): populates
        `prompt_present_matches_training_groups` with PromptJobTrainingGroup
        items in result order, dropping orphan training (training_by_job
        keys whose job_id is not in results[:5]) and capping at 6 total
        resources across groups. Leaves
        `prompt_present_matches_training_flat` as ().

    Both paths share results enumeration + projection and the fallback
    per-job training projection. URL enumeration also forks by version
    so the per-version `prompt_urls` reflects only what that version's
    prompt actually surfaces.
    """
    results_raw = list(inp.results)[:5] if isinstance(inp.results, list) else []
    training_flat = _flat_training_with_paths(inp.training_by_job)

    occurrences: list[tuple[str, str]] = []
    seen_paths: set[str] = set()

    # Result URLs (shared)
    for i, r in enumerate(results_raw):
        if not isinstance(r, dict):
            continue
        path = f"results[{i}].url"
        if path not in seen_paths:
            seen_paths.add(path)
            _enumerate_url_occurrence(r.get("url"), path, occurrences)

    # Prompt training URL enumeration — version-specific.
    if is_v2:
        # v2: iterate results in order; for each, walk training_by_job[job_id]
        # until the 6-resource cap is hit. Orphans (training_by_job keys not
        # in results_raw) are naturally excluded.
        v2_resources_enumerated = 0
        for r in results_raw:
            if v2_resources_enumerated >= 6:
                break
            if not isinstance(r, dict):
                continue
            job_id = r.get("job_id")
            if not isinstance(job_id, str):
                continue
            ts = inp.training_by_job.get(job_id) if isinstance(
                inp.training_by_job, Mapping,
            ) else None
            if not isinstance(ts, (list, tuple)):
                continue
            for i, t in enumerate(list(ts)):
                if v2_resources_enumerated >= 6:
                    break
                if not isinstance(t, dict):
                    continue
                path = f"training_by_job[{job_id!r}][{i}].url"
                if path not in seen_paths:
                    seen_paths.add(path)
                    _enumerate_url_occurrence(t.get("url"), path, occurrences)
                v2_resources_enumerated += 1
    else:
        # v1: legacy flat first-6 across training_by_job.values() iteration.
        for job_id, idx, t in training_flat[:6]:
            path = f"training_by_job[{job_id!r}][{idx}].url"
            if path not in seen_paths:
                seen_paths.add(path)
                _enumerate_url_occurrence(t.get("url"), path, occurrences)

    # Fallback per-job training URLs (shared — first 2 per result)
    for r in results_raw:
        if not isinstance(r, dict):
            continue
        job_id = r.get("job_id")
        if not isinstance(job_id, str):
            continue
        ts = inp.training_by_job.get(job_id) if isinstance(
            inp.training_by_job, Mapping,
        ) else None
        if not isinstance(ts, (list, tuple)):
            continue
        for i, t in enumerate(list(ts)[:2]):
            if not isinstance(t, dict):
                continue
            path = f"training_by_job[{job_id!r}][{i}].url"
            if path not in seen_paths:
                seen_paths.add(path)
                _enumerate_url_occurrence(t.get("url"), path, occurrences)

    validations = _validate_occurrences(occurrences)

    # Projections (shared)
    prompt_results = tuple(
        _project_prompt_result(r, i, validations)
        for i, r in enumerate(results_raw) if isinstance(r, dict)
    )
    fallback_results = tuple(
        _project_fallback_result(r, i, validations)
        for i, r in enumerate(results_raw) if isinstance(r, dict)
    )

    # Version-specific prompt training projection
    prompt_training_flat: tuple[TrainingView, ...] = ()
    prompt_training_groups: tuple[PromptJobTrainingGroup, ...] = ()
    if is_v2:
        # Build PromptJobTrainingGroup items in result order, capping at
        # 6 total resources across groups. Drop empty groups.
        groups_acc: list[PromptJobTrainingGroup] = []
        total_resources = 0
        for r in results_raw:
            if total_resources >= 6:
                break
            if not isinstance(r, dict):
                continue
            job_id = r.get("job_id")
            if not isinstance(job_id, str):
                continue
            ts = inp.training_by_job.get(job_id) if isinstance(
                inp.training_by_job, Mapping,
            ) else None
            if not isinstance(ts, (list, tuple)):
                continue
            group_resources: list[TrainingView] = []
            for i, t in enumerate(list(ts)):
                if total_resources >= 6:
                    break
                if not isinstance(t, dict):
                    continue
                group_resources.append(_project_training(
                    t, f"training_by_job[{job_id!r}][{i}].url", validations,
                ))
                total_resources += 1
            if group_resources:
                title = r.get("title") if isinstance(r.get("title"), str) else None
                groups_acc.append(PromptJobTrainingGroup(
                    job_id=job_id,
                    job_title=title,
                    resources=tuple(group_resources),
                ))
        prompt_training_groups = tuple(groups_acc)
    else:
        # v1: legacy flat[:6]
        prompt_training_flat = tuple(
            _project_training(
                t, f"training_by_job[{job_id!r}][{idx}].url", validations,
            )
            for job_id, idx, t in training_flat[:6]
        )

    # Fallback projection (shared)
    fallback_training_by_job_raw: dict[str, tuple[TrainingView, ...]] = {}
    for r in results_raw:
        if not isinstance(r, dict):
            continue
        job_id = r.get("job_id")
        if not isinstance(job_id, str):
            continue
        ts = inp.training_by_job.get(job_id) if isinstance(
            inp.training_by_job, Mapping,
        ) else None
        if not isinstance(ts, (list, tuple)):
            continue
        projected: list[TrainingView] = []
        for i, t in enumerate(list(ts)[:2]):
            if not isinstance(t, dict):
                continue
            projected.append(_project_training(
                t, f"training_by_job[{job_id!r}][{i}].url", validations,
            ))
        if projected:
            fallback_training_by_job_raw[job_id] = tuple(projected)

    # Allowlists — derived from the version's actual prompt projection.
    prompt_urls = _collect_canonical_urls(
        prompt_results,
        prompt_training_flat if not is_v2 else prompt_training_groups,
    )
    fallback_urls = _collect_canonical_urls(
        fallback_results, fallback_training_by_job_raw,
    )

    return SanitizedResponderView(
        prompt_results=prompt_results,
        fallback_results=fallback_results,
        prompt_present_matches_training_flat=prompt_training_flat,
        prompt_present_matches_training_groups=prompt_training_groups,
        fallback_present_matches_training_by_job=MappingProxyType(
            fallback_training_by_job_raw,
        ),
        prompt_explain_gap_training_flat=(),
        fallback_explain_gap_training_flat=(),
        prompt_present_near_miss_training_flat=(),
        prompt_explain_remaining_gaps_training_flat=(),
        prompt_adjacent_recommendations=None,
        fallback_adjacent_recommendations=(),
        prompt_adjacent_role=None,
        fallback_adjacent_role=None,
        rejected_source_urls=_build_rejected_list(validations),
        prompt_urls=prompt_urls,
        fallback_urls=fallback_urls,
    )


def _build_training_move_view(
    inp: Any,
    prompt_slot: str,
    fallback_slot: str | None,
    fallback_cap: int | None,
) -> SanitizedResponderView:
    """Generic builder for moves whose URL surface is flat training only.

    explain_gap has prompt[:6] and fallback[:3]; present_near_miss and
    explain_remaining_gaps have prompt[:6] only (no fallback URL surface).
    """
    training_flat = _flat_training_with_paths(inp.training_by_job)

    occurrences: list[tuple[str, str]] = []
    seen_paths: set[str] = set()
    for job_id, idx, t in training_flat[:6]:
        path = f"training_by_job[{job_id!r}][{idx}].url"
        if path not in seen_paths:
            seen_paths.add(path)
            _enumerate_url_occurrence(t.get("url"), path, occurrences)

    validations = _validate_occurrences(occurrences)

    prompt_training_flat = tuple(
        _project_training(
            t, f"training_by_job[{job_id!r}][{idx}].url", validations,
        )
        for job_id, idx, t in training_flat[:6]
    )

    fallback_training_flat: tuple[TrainingView, ...] = ()
    if fallback_slot is not None and fallback_cap is not None:
        fallback_training_flat = tuple(
            _project_training(
                t, f"training_by_job[{job_id!r}][{idx}].url", validations,
            )
            for job_id, idx, t in training_flat[:fallback_cap]
        )

    prompt_urls = _collect_canonical_urls(prompt_training_flat)
    fallback_urls = _collect_canonical_urls(fallback_training_flat)

    kwargs: dict[str, Any] = {
        "prompt_results": (),
        "fallback_results": (),
        "prompt_present_matches_training_flat": (),
        "prompt_present_matches_training_groups": (),
        "fallback_present_matches_training_by_job": MappingProxyType({}),
        "prompt_explain_gap_training_flat": (),
        "fallback_explain_gap_training_flat": (),
        "prompt_present_near_miss_training_flat": (),
        "prompt_explain_remaining_gaps_training_flat": (),
        "prompt_adjacent_recommendations": None,
        "fallback_adjacent_recommendations": (),
        "prompt_adjacent_role": None,
        "fallback_adjacent_role": None,
        "rejected_source_urls": _build_rejected_list(validations),
        "prompt_urls": prompt_urls,
        "fallback_urls": fallback_urls,
    }
    kwargs[prompt_slot] = prompt_training_flat
    if fallback_slot is not None:
        kwargs[fallback_slot] = fallback_training_flat
    return SanitizedResponderView(**kwargs)


def _build_explain_gap_view(inp: Any) -> SanitizedResponderView:
    return _build_training_move_view(
        inp,
        prompt_slot="prompt_explain_gap_training_flat",
        fallback_slot="fallback_explain_gap_training_flat",
        fallback_cap=3,
    )


def _build_present_near_miss_view(inp: Any) -> SanitizedResponderView:
    return _build_training_move_view(
        inp,
        prompt_slot="prompt_present_near_miss_training_flat",
        fallback_slot=None,
        fallback_cap=None,
    )


def _build_explain_remaining_gaps_view(inp: Any) -> SanitizedResponderView:
    return _build_training_move_view(
        inp,
        prompt_slot="prompt_explain_remaining_gaps_training_flat",
        fallback_slot=None,
        fallback_cap=None,
    )


def _build_recommend_adjacent_roles_view(
    inp: Any,
) -> SanitizedResponderView:
    """Adjacent recommendations: 0 URLs in both prompt and fallback.

    Projection still happens (the recommendation list is surfaced), but
    no URLs are validated and the allowlists are empty.
    """
    payload = getattr(inp, "adjacent_recommendations_payload", None)
    recs_raw: list[dict] = []
    total_retrieved: int | None = None
    if isinstance(payload, dict):
        recs = payload.get("recommendations")
        if isinstance(recs, list):
            for r in recs:
                if isinstance(r, dict) and isinstance(r.get("title"), str) \
                        and r.get("title").strip():
                    recs_raw.append(r)
        # Project total_retrieved as int when present, None when absent.
        # Sub-step 4's serializer decides how to handle None (the current
        # responder code uses int(... or 0) -> 0 for missing values).
        tr_raw = payload.get("total_retrieved")
        if isinstance(tr_raw, bool):
            total_retrieved = None
        elif isinstance(tr_raw, int):
            total_retrieved = tr_raw

    prompt_recs = tuple(
        _project_prompt_adjacent_recommendation(r) for r in recs_raw
    )
    fallback_recs = tuple(
        _project_fallback_adjacent_recommendation(r) for r in recs_raw
    )
    # Truthy gate matches responder.py:1439 (prompt block emitted only
    # when `inp.adjacent_recommendations_payload` is truthy — `{}` is
    # falsy and would not produce a prompt block).
    prompt_container = PromptAdjacentRecommendationsContainerView(
        recommendations=prompt_recs,
        total_retrieved=total_retrieved,
    ) if isinstance(payload, dict) and payload else None

    return SanitizedResponderView(
        prompt_results=(),
        fallback_results=(),
        prompt_present_matches_training_flat=(),
        prompt_present_matches_training_groups=(),
        fallback_present_matches_training_by_job=MappingProxyType({}),
        prompt_explain_gap_training_flat=(),
        fallback_explain_gap_training_flat=(),
        prompt_present_near_miss_training_flat=(),
        prompt_explain_remaining_gaps_training_flat=(),
        prompt_adjacent_recommendations=prompt_container,
        fallback_adjacent_recommendations=fallback_recs,
        prompt_adjacent_role=None,
        fallback_adjacent_role=None,
        rejected_source_urls=(),
        prompt_urls=frozenset(),
        fallback_urls=frozenset(),
    )


def _build_describe_adjacent_role_view(
    inp: Any,
) -> SanitizedResponderView:
    payload = getattr(inp, "adjacent_role_description_payload", None)
    if not isinstance(payload, dict):
        return _empty_view()

    job = payload.get("job")
    occurrences: list[tuple[str, str]] = []
    if isinstance(job, dict):
        _enumerate_url_occurrence(
            job.get("url"),
            "adjacent_role_description_payload.job.url",
            occurrences,
        )

    validations = _validate_occurrences(occurrences)
    # Truthy gate matches responder.py:1451 (prompt block emitted only
    # when `inp.adjacent_role_description_payload` is truthy — `{}` is
    # falsy and would not produce a prompt block). The fallback view
    # still exists for `{}` so the deterministic fallback consumer can
    # read expired / has_validated_url and branch correctly.
    prompt_role = (
        _project_prompt_adjacent_role(payload, validations)
        if payload else None
    )
    fallback_role = _project_fallback_adjacent_role(payload, validations)

    prompt_urls = _collect_canonical_urls(prompt_role) if prompt_role else frozenset()
    # Bug.2b: the fallback renders the validated adjacent-role URL
    # inline. The allowlist must reflect that — otherwise downstream
    # consumers reading fallback_urls would see an empty set while
    # the fallback renderer surfaces a URL, an internal contradiction.
    fallback_urls = (
        _collect_canonical_urls(fallback_role) if fallback_role else frozenset()
    )

    return SanitizedResponderView(
        prompt_results=(),
        fallback_results=(),
        prompt_present_matches_training_flat=(),
        prompt_present_matches_training_groups=(),
        fallback_present_matches_training_by_job=MappingProxyType({}),
        prompt_explain_gap_training_flat=(),
        fallback_explain_gap_training_flat=(),
        prompt_present_near_miss_training_flat=(),
        prompt_explain_remaining_gaps_training_flat=(),
        prompt_adjacent_recommendations=None,
        fallback_adjacent_recommendations=(),
        prompt_adjacent_role=prompt_role,
        fallback_adjacent_role=fallback_role,
        rejected_source_urls=_build_rejected_list(validations),
        prompt_urls=prompt_urls,
        fallback_urls=fallback_urls,
    )


# =========================================================================
# Prompt JSON serializers — convert projected views back to the dict
# shape the current prompt builders emit. Sub-step 4 consumers call
# these instead of constructing the dict from raw inp.
#
# Parity contract (locked sub-step 4 revision 4):
#   - Byte-identical for valid URLs, non-URL content, and per the
#     locked path-specific absent-URL table.
#   - Projection normalization (deliberate) for: score_explanation
#     partial/empty dicts; V2 training unknown fields dropped;
#     V2 training None-field omission; adjacent recommendation
#     non-canonical filtering; adjacent role job dict 6-field allowlist
#     expansion; posted_date scalar coercion.
# =========================================================================
def serialize_score_components_for_prompt(
    sc: "ScoreComponentsView | None",
) -> dict | None:
    if sc is None:
        return None
    out: dict[str, object] = {}
    if sc.skill_base is not None:
        sb_out: dict[str, object] = {}
        for f in ("value", "mode", "required_match_ratio",
                  "required_weight", "preferred_match_ratio",
                  "preferred_weight"):
            v = getattr(sc.skill_base, f)
            if v is not None:
                sb_out[f] = v
        out["skill_base"] = sb_out
    if sc.boosts is not None:
        b_out: dict[str, object] = {}
        for f in ("location", "recency", "target_role",
                  "target_noc_match", "work_type_fit", "shift_fit"):
            v = getattr(sc.boosts, f)
            if v is not None:
                b_out[f] = v
        out["boosts"] = b_out
    if sc.title_match is not None:
        tm_out: dict[str, object] = {}
        for f in ("applied", "raw_similarity"):
            v = getattr(sc.title_match, f)
            if v is not None:
                tm_out[f] = v
        out["title_match"] = tm_out
    if sc.score_pre_caps is not None:
        out["score_pre_caps"] = sc.score_pre_caps
    if sc.score_post_caps is not None:
        out["score_post_caps"] = sc.score_post_caps
    return out


def serialize_score_explanation_for_prompt(
    se: "ScoreExplanationView | None",
) -> dict | None:
    """Locked sub-step 4 score-explanation projection normalization.

    Always-list fields are emitted as lists; scalar `T | None` fields are
    omitted when None; nested score_components serializes recursively.
    Partial input dicts gain empty-list fields for absent always-list
    fields (this is the locked normalization divergence). `{}` input
    produces `None` here (projection at view-build time); caller
    serializes that as JSON null.
    """
    if se is None:
        return None
    out: dict[str, object] = {}
    # Always-list fields
    out["matched_skills"] = list(se.matched_skills)
    out["missing_skills"] = list(se.missing_skills)
    out["required_matched"] = list(se.required_matched)
    out["required_missing"] = list(se.required_missing)
    out["preferred_matched"] = list(se.preferred_matched)
    out["preferred_missing"] = list(se.preferred_missing)
    out["required_match_strengths"] = list(se.required_match_strengths)
    out["required_match_stages"] = list(se.required_match_stages)
    out["preferred_match_strengths"] = list(se.preferred_match_strengths)
    out["preferred_match_stages"] = list(se.preferred_match_stages)
    out["credential_gap_skills"] = list(se.credential_gap_skills)
    out["caps_applied"] = list(se.caps_applied)
    # Optional scalar fields — omit when None
    for f in (
        "required_match_strength_sum", "preferred_match_strength_sum",
        "skill_match_ratio", "required_match_ratio", "required_total",
        "preferred_match_ratio", "preferred_total",
        "title_match_similarity", "title_match_override",
        "recency_days", "location_boosted", "work_type_fit", "shift_fit",
        "credential_warning_present", "work_type_user", "work_type_job",
        "band_capped_by_credential", "band_capped_by_no_experience",
        "band_capped_by_work_type_mismatch",
    ):
        v = getattr(se, f)
        if v is not None:
            out[f] = v
    if se.score_components is not None:
        out["score_components"] = serialize_score_components_for_prompt(
            se.score_components,
        )
    return out


def serialize_result_for_v1_prompt(r: PromptResultView) -> dict:
    """V1 prompt result JSON shape (responder.py:234-249).
    URL key always present (null when absent).
    score_explanation key always present (null when absent).
    """
    return {
        "title": r.title,
        "employer": r.employer,
        "url": r.url.raw if r.url is not None else None,
        "location": r.location,
        "match_band": r.match_band,
        "matched_skills": list(r.matched_skills),
        "missing_skills": list(r.missing_skills),
        "credential_warning": r.credential_warning,
        "score_explanation": serialize_score_explanation_for_prompt(
            r.score_explanation,
        ),
    }


def serialize_result_for_v2_prompt(r: PromptResultView) -> dict:
    """V2 prompt result JSON shape (responder.py:1380-1392).
    Same shape as V1 — url key always present, null when absent.
    """
    return serialize_result_for_v1_prompt(r)


def serialize_training_for_v1_prompt(t: TrainingView) -> dict:
    """V1 prompt training JSON shape — exactly 4 fields, all keys
    always present (responder.py:258-263). Url key emits null when
    absent.
    """
    return {
        "title": t.title,
        "provider": t.provider,
        "url": t.url.raw if t.url is not None else None,
        "for_skill": t.for_skill,
    }


def serialize_training_for_v2_prompt(t: TrainingView) -> dict:
    """V2 prompt training JSON shape — 11-field TrainingView allowlist.
    None fields are OMITTED from the output (deliberate divergence
    from V1 which emits nulls). Unknown raw fields were already
    dropped at projection.
    """
    out: dict[str, object] = {}
    if t.provider is not None: out["provider"] = t.provider
    if t.title is not None: out["title"] = t.title
    if t.url is not None: out["url"] = t.url.raw
    if t.for_skill is not None: out["for_skill"] = t.for_skill
    if t.duration_band is not None: out["duration_band"] = t.duration_band
    if t.resource_type is not None: out["resource_type"] = t.resource_type
    if t.reason is not None: out["reason"] = t.reason
    if t.type is not None: out["type"] = t.type
    if t.for_gap is not None: out["for_gap"] = t.for_gap
    if t.summary is not None: out["summary"] = t.summary
    if t.verified is not None: out["verified"] = t.verified
    return out


def serialize_adjacent_recommendations_for_prompt(
    container: "PromptAdjacentRecommendationsContainerView | None",
) -> dict | None:
    """ADJACENT_RECOMMENDATIONS prompt block (responder.py:1441-1444).

    Emits two fields: `recommendations` (list of 7-field allowlist dicts)
    and `total_retrieved` (int, defaulting to 0 when projection produced
    None per the current `int(... or 0)` falsy-default behavior).

    Returns None when container is None (no payload). The caller decides
    whether to emit the block at all.
    """
    if container is None:
        return None
    recs_out: list[dict] = []
    for rec in container.recommendations:
        recs_out.append({
            "job_id": rec.job_id,
            "title": rec.title,
            "employer": rec.employer,
            "location": rec.location,
            "evidence_summary": rec.evidence_summary,
            "why_adjacent": rec.why_adjacent,
            "matched_skills": list(rec.matched_skills),
        })
    return {
        "recommendations": recs_out,
        "total_retrieved": (
            container.total_retrieved
            if container.total_retrieved is not None else 0
        ),
    }


def serialize_prompt_adjacent_role(
    view: "PromptAdjacentRoleView | None",
) -> dict | None:
    """ADJACENT_ROLE_DESCRIPTION prompt block (responder.py:1462-1467).

    When view is None (payload absent or empty {} per truthy gate),
    returns None and caller skips the block.

    When view.job_is_mapping is True, emits the 6-field allowlist
    dict (including empty dict expanded to 6 nulls — deliberate
    normalization). When False, emits "job": null.

    URL renders as raw form; posted_date as projected string or None.
    """
    if view is None:
        return None
    if view.job_is_mapping:
        job = {
            "job_id": view.job_id,
            "title": view.title,
            "employer": view.employer,
            "location": view.location,
            "url": view.url.raw if view.url is not None else None,
            "posted_date": view.posted_date,
        }
    else:
        job = None
    return {
        "job": job,
        "evidence_summary": view.evidence_summary or "",
        "matched_skills": list(view.matched_skills),
        "expired": view.expired,
    }


# =========================================================================
# AR-9.feat.coach-tiers step 9 — present_tiered_matches view builder
# =========================================================================
def _project_validated_to_sanitized(v: Validated | None) -> SanitizedURL | None:
    """Single-spot Validated → SanitizedURL projection. The evidence
    layer (`chat.tiered_evidence`) deliberately stops at Validated so
    it doesn't depend on `url_views`; this view-side projector closes
    the loop without re-validating.
    """
    if isinstance(v, Validated):
        return SanitizedURL.from_validated(v)
    return None


def _project_job_facts(facts: JobFacts) -> PromptJobFacts:
    return PromptJobFacts(
        posted_date=facts.posted_date,
        posted_days_ago=facts.posted_days_ago,
        location=facts.location,
        employment_type=facts.employment_type,
        salary_text=facts.salary_text,
    )


def _project_training_option(t: TrainingOption) -> PromptTrainingOption:
    return PromptTrainingOption(
        provider=t.provider,
        title=t.title,
        url=_project_validated_to_sanitized(t.url),
        format=t.format,
        duration_text=t.duration_text,
    )


def _project_prioritized_gap(g: PrioritizedGap) -> PromptPrioritizedGap:
    return PromptPrioritizedGap(
        job_requirement=g.job_requirement,
        category=g.category,
        priority=g.priority,
        blocker=g.blocker,
        training_options=tuple(
            _project_training_option(t) for t in g.training_options
        ),
    )


def _project_non_blocking_gap(g: NonBlockingGap) -> PromptNonBlockingGap:
    return PromptNonBlockingGap(
        job_requirement=g.job_requirement,
        material=g.material,
    )


def _project_transferable_pair(p: TransferablePair) -> PromptTransferablePair:
    return PromptTransferablePair(
        user_skill=p.user_skill,
        applies_to=p.applies_to,
        stage=p.stage,
    )


def _project_strong_match(m: StrongMatch) -> PromptStrongMatch:
    return PromptStrongMatch(
        job_id=m.job_id,
        title=m.title,
        employer=m.employer,
        location=m.location,
        noc_code=m.noc_code,
        url=_project_validated_to_sanitized(m.url),
        job_facts=_project_job_facts(m.job_facts),
        skill_alignment=tuple(m.skill_alignment),
        non_blocking_gaps=tuple(
            _project_non_blocking_gap(g) for g in m.non_blocking_gaps
        ),
        credential_warning_text=m.credential_warning_text,
        strength_claim_text=m.strength_claim_text,
    )


def _project_stretch_match(m: StretchMatch) -> PromptStretchMatch:
    return PromptStretchMatch(
        job_id=m.job_id,
        title=m.title,
        employer=m.employer,
        location=m.location,
        noc_code=m.noc_code,
        url=_project_validated_to_sanitized(m.url),
        job_facts=_project_job_facts(m.job_facts),
        skill_alignment=tuple(m.skill_alignment),
        prioritized_gaps=tuple(
            _project_prioritized_gap(g) for g in m.prioritized_gaps
        ),
        credential_warning_text=m.credential_warning_text,
        strength_claim_text=m.strength_claim_text,
    )


def _project_adjacent_job(a: AdjacentJob) -> PromptAdjacentJob:
    return PromptAdjacentJob(
        job_id=a.job_id,
        title=a.title,
        employer=a.employer,
        location=a.location,
        noc_code=a.noc_code,
        url=_project_validated_to_sanitized(a.url),
        job_facts=_project_job_facts(a.job_facts),
        skill_alignment=tuple(a.skill_alignment),
        transferable_pairs=tuple(
            _project_transferable_pair(p) for p in a.transferable_pairs
        ),
        important_gaps=tuple(a.important_gaps),
        credential_warning_text=a.credential_warning_text,
        why_adjacent=a.why_adjacent,
        strength_claim_text=a.strength_claim_text,
    )


def build_sanitized_responder_view_for_tiered_matches(
    tier_evidence: TieredEvidence,
    *,
    rejected: tuple[RejectedSourceURL, ...] = (),
) -> SanitizedResponderView:
    """Move-gated builder for the new `present_tiered_matches` move.

    Projects `TieredEvidence` into the SanitizedResponderView with:
      - Every `Validated` URL promoted to `SanitizedURL` (the view
        boundary is the only place this projection happens).
      - Only the fields the coach prompt and deterministic fallback
        need on each tier record — raw `MatchResult`, raw job dicts,
        and unvalidated URLs are excluded by construction.
      - `prompt_urls`: union of every tier-record job URL plus every
        training-option URL across all three tiers (LLM allowlist).
      - `fallback_urls`: the SanitizedURL set the deterministic
        fallback is allowed to render. Mirrors `prompt_urls` for
        step 9 — step 10 may narrow it to exactly what the fallback
        body references.
      - All non-tier slots are empty (move gating: tier slots exist
        ONLY when this builder ran).

    Tier order and exclusivity follow `tier_evidence` directly — this
    builder does NOT re-order or re-filter.
    """
    apply_today = tuple(
        _project_strong_match(m) for m in tier_evidence.apply_today
    )
    worth_a_try = tuple(
        _project_stretch_match(m) for m in tier_evidence.worth_a_try
    )
    sideways = tuple(
        _project_adjacent_job(a) for a in tier_evidence.sideways_move
    )

    # prompt_urls covers every renderable URL on every tier item.
    # _collect_canonical_urls walks PromptStretchMatch into its
    # prioritized_gaps[].training_options[] automatically (see the
    # type-specific branch added in step 9).
    prompt_urls = _collect_canonical_urls(
        apply_today, worth_a_try, sideways,
    )
    # AR-9.feat.coach-tiers step 10: fallback_urls is recomputed from
    # exactly the URLs the deterministic fallback renderer would emit.
    # The renderer is the authority — if it ever caps training options
    # or skips a URL slot, fallback_urls follows automatically. Lazy
    # import: coach_tiers_fallback consumes view types from this
    # module, so we defer the import to the call site to avoid a
    # circular at module load.
    from skillbridge.chat.coach_tiers_fallback import (
        collect_fallback_render_urls,
    )
    # Construct a transient view-shaped object that the fallback can
    # read: it consumes only the prompt_tiered_* slots, so we only
    # need to thread those through. We re-use the actual builder
    # result for max parity.
    _probe_view = SanitizedResponderView(
        prompt_results=(),
        fallback_results=(),
        prompt_present_matches_training_flat=(),
        prompt_present_matches_training_groups=(),
        fallback_present_matches_training_by_job=MappingProxyType({}),
        prompt_explain_gap_training_flat=(),
        fallback_explain_gap_training_flat=(),
        prompt_present_near_miss_training_flat=(),
        prompt_explain_remaining_gaps_training_flat=(),
        prompt_adjacent_recommendations=None,
        fallback_adjacent_recommendations=(),
        prompt_adjacent_role=None,
        fallback_adjacent_role=None,
        rejected_source_urls=rejected,
        prompt_urls=prompt_urls,
        fallback_urls=frozenset(),
        prompt_tiered_apply_today=apply_today,
        prompt_tiered_worth_a_try=worth_a_try,
        prompt_tiered_sideways_move=sideways,
    )
    fallback_urls = collect_fallback_render_urls(_probe_view)

    return SanitizedResponderView(
        prompt_results=(),
        fallback_results=(),
        prompt_present_matches_training_flat=(),
        prompt_present_matches_training_groups=(),
        fallback_present_matches_training_by_job=MappingProxyType({}),
        prompt_explain_gap_training_flat=(),
        fallback_explain_gap_training_flat=(),
        prompt_present_near_miss_training_flat=(),
        prompt_explain_remaining_gaps_training_flat=(),
        prompt_adjacent_recommendations=None,
        fallback_adjacent_recommendations=(),
        prompt_adjacent_role=None,
        fallback_adjacent_role=None,
        rejected_source_urls=rejected,
        prompt_urls=prompt_urls,
        fallback_urls=fallback_urls,
        prompt_tiered_apply_today=apply_today,
        prompt_tiered_worth_a_try=worth_a_try,
        prompt_tiered_sideways_move=sideways,
    )
