"""CP4-pre-1 — Inventory-Diagnosis gate.

LOCKED contract: six mutually exclusive outcomes from a single
deterministic function. Drives which downstream pillar (CP4 primary,
CP4 secondary admissibility, sideways recommendation surface,
inventory coaching, intake clarification) is authorized to render
output. Diagnosis is the SOLE authority for pillar authorization.

Two load-bearing principles preserved by this module:
  1. Job existence is measured INDEPENDENTLY from user-job matching.
     A user with weak skills against a populous target NOC must not be
     diagnosed as a local-inventory absence.
  2. Diagnosis identifies the problem. CP4 determines whether a
     credible solution exists. PREPARATION_GAP fires whenever the
     target exists locally and the user has unmet requirements —
     regardless of whether training is mapped or the gap is closable.

Decision rules (top-to-bottom, first match wins) — six rules total:

  Pre-0. Engine must have completed successfully over the intended
         scope. Engine failure → MARKET_DATA_UNAVAILABLE.
  1. not enough_to_match or not usable_evidence_present  → UNDETERMINED
  2. not snapshot_usable                                  → MARKET_DATA_UNAVAILABLE
  3. ≥1 direct match with band ∈ {strong, good} AND
     required_missing == [] AND match_eligible            → READY_TO_APPLY
  4. target_posting_count is known AND > 0                → PREPARATION_GAP
  5. skill_adjacent_results non-empty                     → SKILL_ADJACENT_AVAILABLE
  6. otherwise                                            → NO_OPPORTUNITY_FOUND

NO_OPPORTUNITY_FOUND is named honestly. It claims that DIRECT MATCH
and SKILL ADJACENCY (the two capabilities currently implemented)
returned nothing. It does NOT claim the local market has nothing —
that stronger claim (`LOCAL_INVENTORY_GAP`) is reserved for after
CP5 ships and occupational adjacency is also evaluated.

target_posting_count is scoped to the user's exact resolved 5-digit
NOC. For vague or unresolved targets, the field is absent and
rule 4 is unreachable.

This module does not run the engine, query the registry, or compose
prose. It composes signals other modules already produce.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from skillbridge.match.engine import MatchResult


# --- Outcome enum --------------------------------------------------------

Outcome = Literal[
    "UNDETERMINED",
    "MARKET_DATA_UNAVAILABLE",
    "READY_TO_APPLY",
    "PREPARATION_GAP",
    "SKILL_ADJACENT_AVAILABLE",
    "NO_OPPORTUNITY_FOUND",
]


# --- Pillar enum ---------------------------------------------------------

Pillar = Literal[
    "intake_clarification",
    "inventory_coaching",
    "CP4_primary",
    "CP4_secondary_admissible",
    "adjacency_recommendation_surface",
]


# --- Diagnosis object ----------------------------------------------------

@dataclass(frozen=True)
class InventoryDiagnosis:
    """The locked contract's output. Six outcomes, deterministic from
    inputs. The `pillars_authorized` set is the SOLE authority for
    which downstream surface may render output."""
    outcome: Outcome
    pillars_authorized: frozenset[Pillar]
    reason_code: str
    # The supporting evidence used. CP4 reads this to collect candidate
    # gaps when invoked under PREPARATION_GAP. Other consumers may
    # inspect for telemetry. Keys are stable; values are minimal.
    supporting_evidence: dict


# --- Pillar authorization table ------------------------------------------

_PILLARS_BY_OUTCOME: dict[Outcome, frozenset[Pillar]] = {
    "UNDETERMINED": frozenset({"intake_clarification"}),
    "MARKET_DATA_UNAVAILABLE": frozenset({"inventory_coaching"}),
    "READY_TO_APPLY": frozenset({"CP4_secondary_admissible"}),
    "PREPARATION_GAP": frozenset({"CP4_primary"}),
    "SKILL_ADJACENT_AVAILABLE": frozenset({"adjacency_recommendation_surface"}),
    "NO_OPPORTUNITY_FOUND": frozenset({"inventory_coaching"}),
}


# --- Diagnose ------------------------------------------------------------

def diagnose(
    *,
    enough_to_match: bool,
    usable_evidence_present: bool,
    engine_completed: bool,
    snapshot_usable: bool,
    direct_match_results: list["MatchResult"],
    skill_adjacent_results: list,
    target_posting_count: int | None,
) -> InventoryDiagnosis:
    """Return the locked-contract diagnosis.

    Parameters mirror the locked input schema:
      - enough_to_match, usable_evidence_present: from truth-summary.
      - engine_completed: False iff the match engine raised or did not
        run over the intended scope. (Pre-rule 0.)
      - snapshot_usable: thin read-only derivation from pipeline
        snapshot — True iff a non-running publication exists and the
        snapshot query succeeded.
      - direct_match_results: MatchResult list from the engine. Scoped
        to the user's target NOC when resolved; broader otherwise.
      - skill_adjacent_results: output of the existing strict-AND
        adjacency gate.
      - target_posting_count: count of currently-live local postings
        in the user's exact 5-digit NOC. None when the target is vague
        or unresolved (rule 4 is then unreachable, falling through to
        rule 5/6).

    Returns InventoryDiagnosis. Function is total and deterministic.
    """
    # Pre-0: engine completion precondition.
    if not engine_completed:
        return _build(
            "MARKET_DATA_UNAVAILABLE",
            "market_data_unavailable_engine_failed",
            supporting_evidence={
                "direct_match_count": 0,
                "skill_adjacent_count": 0,
                "target_posting_count": target_posting_count,
            },
        )

    # Rule 1: insufficient user evidence.
    if not enough_to_match or not usable_evidence_present:
        return _build(
            "UNDETERMINED",
            "insufficient_user_evidence",
            supporting_evidence={
                "enough_to_match": enough_to_match,
                "usable_evidence_present": usable_evidence_present,
            },
        )

    # Rule 2: market data unusable.
    if not snapshot_usable:
        return _build(
            "MARKET_DATA_UNAVAILABLE",
            "market_data_unavailable_snapshot",
            supporting_evidence={
                "direct_match_count": len(direct_match_results),
                "target_posting_count": target_posting_count,
            },
        )

    # Rule 3: Apply-today match exists.
    # H3 fix (2026-06-15): require required_missing to be EXPLICITLY
    # the empty list. Absence (score_explanation missing or malformed)
    # must NOT be admitted as "no gaps" — that's the same fail-closed
    # principle the tier builder uses in _required_missing_or_none.
    ready_records = [
        r for r in direct_match_results
        if r.match_eligible
        and r.match_band in ("strong", "good")
        and _required_missing_or_none(r) == []
    ]
    if ready_records:
        return _build(
            "READY_TO_APPLY",
            "apply_today_target_match",
            supporting_evidence={
                "ready_job_ids": sorted(r.job_id for r in ready_records),
            },
        )

    # Rule 4: target postings exist, user has unmet requirements.
    # PREPARATION_GAP does NOT inspect whether the gap is closable —
    # that's CP4's question.
    if target_posting_count is not None and target_posting_count > 0:
        # Supporting evidence: MatchResult records over the target NOC
        # with EXPLICIT non-empty required_missing. Records with
        # absent/malformed score_explanation are excluded — CP4
        # cannot collect candidate gaps from None.
        gap_records = [
            r for r in direct_match_results
            if (rm := _required_missing_or_none(r)) is not None and rm
        ]
        return _build(
            "PREPARATION_GAP",
            "target_exists_user_underqualified",
            supporting_evidence={
                "target_posting_count": target_posting_count,
                "gap_record_job_ids": sorted(r.job_id for r in gap_records),
                "gap_record_count": len(gap_records),
            },
        )

    # Rule 5: skill-adjacent jobs exist.
    if skill_adjacent_results:
        return _build(
            "SKILL_ADJACENT_AVAILABLE",
            "skill_adjacency_accepted",
            supporting_evidence={
                "skill_adjacent_count": len(skill_adjacent_results),
            },
        )

    # Rule 6: nothing found by the capabilities currently implemented.
    # NOT a claim that the local market has nothing — that stronger
    # claim is reserved for after CP5 ships and occupational adjacency
    # is also checked.
    return _build(
        "NO_OPPORTUNITY_FOUND",
        "direct_and_skill_empty",
        supporting_evidence={
            "direct_match_count": len(direct_match_results),
            "skill_adjacent_count": 0,
            "target_posting_count": target_posting_count,
        },
    )


# --- Helpers -------------------------------------------------------------

def _build(
    outcome: Outcome, reason_code: str, *, supporting_evidence: dict,
) -> InventoryDiagnosis:
    return InventoryDiagnosis(
        outcome=outcome,
        pillars_authorized=_PILLARS_BY_OUTCOME[outcome],
        reason_code=reason_code,
        supporting_evidence=supporting_evidence,
    )


def _required_missing(result: "MatchResult") -> list[str]:
    """Read the required_missing list from a MatchResult's score
    explanation. Returns [] when absent or wrong shape.

    Note: this helper conflates absence with empty. For rules 3 and 4
    use `_required_missing_or_none` instead — those rules MUST
    distinguish "matcher confirmed no gaps" (empty list) from "matcher
    did not report" (absent). This helper is retained for telemetry
    paths where the distinction does not matter.
    """
    rm = _required_missing_or_none(result)
    return rm if rm is not None else []


def _required_missing_or_none(result: "MatchResult") -> list[str] | None:
    """Fail-closed reader, mirroring tiered_evidence._required_missing_or_none.

    Returns the list ONLY when score_explanation is a dict AND
    required_missing is present as a list. Otherwise None — callers
    must treat None as "cannot classify; reject."

    H3 fix (2026-06-15): rule 3 (READY_TO_APPLY) and rule 4
    (PREPARATION_GAP evidence collection) both fail closed on absence.
    Without this distinction, a malformed score_explanation produced a
    spurious READY_TO_APPLY because `not []` is True.
    """
    if not isinstance(result.score_explanation, dict):
        return None
    rm = result.score_explanation.get("required_missing")
    if not isinstance(rm, list):
        return None
    return rm
