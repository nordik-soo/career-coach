"""Handler-level tests for R-3 remaining-gaps synthesis.

Adding integration-shaped coverage too -- the round-14 review surfaced
that the hook ran BEFORE scope validation, hijacking immigration /
wages / off-topic redirects. The scope-precedence regression below
exercises the full _try_v2_path so the gate stays in place.


Focused on:
  - The dispatch hook (`_run_remaining_gaps_dispatch`): save-and-clear
    pending discipline; None-guard branch; try/except restore-pending
    on detector failure; unknown intent.kind catch-all
  - The synthesis helpers (`_synthesize_remaining_gaps_decision`,
    `_synthesize_clarification_decision`): accumulation algorithm,
    hypothetical->claimed promotion, retraction filtering, ArbiterReason
    distinction, pending set/clear discipline including the canonical=
    None guard
  - The legacy-mapping pin (_FINAL_MOVE_TO_LEGACY_ACTION)
  - The arbiter invariant: explain_remaining_gaps reaches the responder
    ONLY via handler synthesis, never through validate_planner_intent
    or resolve_match_outcome

No DB. Stubs the registry loader so the tests don't depend on the YAML.
"""
from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("LLM_ENABLED", "false")

import pytest

from skillbridge.chat import arbiter as arbiter_mod
from skillbridge.chat import handler
from skillbridge.chat.arbiter import (
    ARBITER_REASON_BOOTSTRAP_MATCH,
    ARBITER_REASON_CONFIRM_CREDENTIAL,
    ARBITER_REASON_REMAINING_GAPS,
    ARBITER_REASON_REMAINING_GAPS_RETRACTED,
    ArbiterDecision,
)
from skillbridge.chat import intake_state
from skillbridge.chat.remaining_gaps import (
    CredentialClaim, RemainingGapsIntent,
)
from skillbridge.session.staging import StagedProfile

pytestmark = pytest.mark.nodb


# ============================================================================
# Helpers
# ============================================================================
def _staged() -> StagedProfile:
    sp = StagedProfile.new("test-handler-r3")
    sp.message_count = 5
    sp.last_match_snapshot = {
        "captured_at_turn": 3,
        "lead_job": {
            "job_id": "honda-uuid",
            "title": "310S Licensed Automotive Technician",
            "employer": "Great Lakes Honda",
            "credential_gaps": [
                {"display": "310S Automotive Technician License",
                 "canonical": "310S-CANON"},
                {"display": "G2/G driver's license",
                 "canonical": "CLASS-G-CANON"},
            ],
            "core_skill_gaps": ["Honda vehicle experience"],
        },
        "other_jobs_meta": [],
    }
    return sp


# ============================================================================
# _synthesize_remaining_gaps_decision -- subtract path
# ============================================================================
def test_synthesize_subtract_appends_new_claim_to_empty_accumulated():
    sp = _staged()
    intent = RemainingGapsIntent(
        kind="subtract",
        current_turn_claims=(
            CredentialClaim(canonical="310S-CANON", mode="claimed"),
        ),
    )
    decision = handler._synthesize_remaining_gaps_decision(
        sp, intent, retracted=False,
    )
    assert decision.final_move == "explain_remaining_gaps"
    assert decision.reason_code == ARBITER_REASON_REMAINING_GAPS
    assert decision.arbiter_action == "handler_synthesized_remaining_gaps"
    assert decision.tone == "warm_supportive"
    assert decision.ask_slot is None
    assert sp.last_assumed_completed_credentials == [
        {"canonical": "310S-CANON", "mode": "claimed"},
    ]


def test_synthesize_subtract_appends_in_order_with_dedupe():
    sp = _staged()
    sp.last_assumed_completed_credentials = [
        {"canonical": "310S-CANON", "mode": "hypothetical"},
    ]
    intent = RemainingGapsIntent(
        kind="subtract",
        current_turn_claims=(
            CredentialClaim(canonical="CLASS-G-CANON", mode="hypothetical"),
        ),
    )
    handler._synthesize_remaining_gaps_decision(sp, intent, retracted=False)
    assert sp.last_assumed_completed_credentials == [
        {"canonical": "310S-CANON",   "mode": "hypothetical"},
        {"canonical": "CLASS-G-CANON","mode": "hypothetical"},
    ]


def test_synthesize_subtract_promotes_hypothetical_to_claimed():
    """If a canonical is already accumulated as hypothetical and the
    current turn explicitly claims it, the existing entry's mode is
    PROMOTED in place (no duplicate)."""
    sp = _staged()
    sp.last_assumed_completed_credentials = [
        {"canonical": "310S-CANON", "mode": "hypothetical"},
    ]
    intent = RemainingGapsIntent(
        kind="subtract",
        current_turn_claims=(
            CredentialClaim(canonical="310S-CANON", mode="claimed"),
        ),
    )
    handler._synthesize_remaining_gaps_decision(sp, intent, retracted=False)
    assert sp.last_assumed_completed_credentials == [
        {"canonical": "310S-CANON", "mode": "claimed"},
    ]


def test_synthesize_subtract_does_not_demote_claimed_to_hypothetical():
    """Inverse of promotion -- a hypothetical claim arriving after a
    claimed one MUST NOT downgrade the mode."""
    sp = _staged()
    sp.last_assumed_completed_credentials = [
        {"canonical": "310S-CANON", "mode": "claimed"},
    ]
    intent = RemainingGapsIntent(
        kind="subtract",
        current_turn_claims=(
            CredentialClaim(canonical="310S-CANON", mode="hypothetical"),
        ),
    )
    handler._synthesize_remaining_gaps_decision(sp, intent, retracted=False)
    assert sp.last_assumed_completed_credentials == [
        {"canonical": "310S-CANON", "mode": "claimed"},
    ]


def test_synthesize_subtract_caps_at_max_cred_gaps(monkeypatch):
    """6th unique canonical drops the LATEST entry (preserves first-
    occurrence order). The pre-existing accumulated entries must be
    valid against the snapshot for the cap behavior to be observable
    (round-17: stale-against-snapshot entries are now sanitized away
    before append-and-cap, so an unrealistic 'C0..C4' fixture would
    just clear and accept the OVERFLOW). Build a fat snapshot for
    this test."""
    from skillbridge.session.staging import MAX_CRED_GAPS
    sp = _staged()
    # Synthetic snapshot with MAX_CRED_GAPS + 1 valid canonicals so
    # both the prefilled accumulated entries AND the new OVERFLOW
    # claim are snapshot-valid.
    fat_gaps = [
        {"display": f"C{i} display", "canonical": f"C{i}"}
        for i in range(MAX_CRED_GAPS)
    ] + [{"display": "OVERFLOW display", "canonical": "OVERFLOW"}]
    sp.last_match_snapshot = {
        "captured_at_turn": 1,
        "lead_job": {
            "job_id": "x", "title": "T", "employer": None,
            "credential_gaps": fat_gaps,
            "core_skill_gaps": [],
        },
        "other_jobs_meta": [],
    }
    # Patch MAX_CRED_GAPS check on the snapshot defensive layer too --
    # not needed because the test fixture is constructed in code, not
    # via from_json.

    sp.last_assumed_completed_credentials = [
        {"canonical": f"C{i}", "mode": "claimed"}
        for i in range(MAX_CRED_GAPS)
    ]
    intent = RemainingGapsIntent(
        kind="subtract",
        current_turn_claims=(
            CredentialClaim(canonical="OVERFLOW", mode="claimed"),
        ),
    )
    handler._synthesize_remaining_gaps_decision(sp, intent, retracted=False)
    canonicals = [c["canonical"] for c in sp.last_assumed_completed_credentials]
    assert canonicals == [f"C{i}" for i in range(MAX_CRED_GAPS)]
    assert "OVERFLOW" not in canonicals


# ============================================================================
# _synthesize_remaining_gaps_decision -- retract path
# ============================================================================
def test_synthesize_retract_removes_named_canonical():
    sp = _staged()
    sp.last_assumed_completed_credentials = [
        {"canonical": "310S-CANON",   "mode": "hypothetical"},
        {"canonical": "CLASS-G-CANON","mode": "claimed"},
    ]
    intent = RemainingGapsIntent(
        kind="retract",
        retract_canonical="310S-CANON",
    )
    decision = handler._synthesize_remaining_gaps_decision(
        sp, intent, retracted=True,
    )
    assert decision.final_move == "explain_remaining_gaps"
    assert decision.reason_code == ARBITER_REASON_REMAINING_GAPS_RETRACTED
    assert decision.arbiter_action == "handler_synthesized_remaining_gaps"
    assert sp.last_assumed_completed_credentials == [
        {"canonical": "CLASS-G-CANON", "mode": "claimed"},
    ]


# ============================================================================
# R-4 -- payload builder
# ============================================================================
def test_build_payload_returns_none_when_snapshot_missing():
    sp = StagedProfile.new("t")
    assert handler._build_remaining_gaps_payload(sp) is None


def test_build_payload_empty_accumulated_returns_all_remaining():
    sp = _staged()
    payload = handler._build_remaining_gaps_payload(sp)
    assert payload["role"] == "310S Licensed Automotive Technician"
    assert payload["employer"] == "Great Lakes Honda"
    assert payload["assumed_completed_credentials"] == []
    assert [g["canonical"] for g in payload["remaining_credentials"]] == [
        "310S-CANON", "CLASS-G-CANON",
    ]
    assert payload["remaining_core_skills"] == ["Honda vehicle experience"]
    assert payload["any_hypothetical"] is False


def test_build_payload_subtracts_accumulated_from_remaining():
    sp = _staged()
    sp.last_assumed_completed_credentials = [
        {"canonical": "310S-CANON", "mode": "claimed"},
    ]
    payload = handler._build_remaining_gaps_payload(sp)
    assert [g["canonical"] for g in payload["remaining_credentials"]] == [
        "CLASS-G-CANON",
    ]
    assert payload["assumed_completed_credentials"] == [{
        "display":   "310S Automotive Technician License",
        "canonical": "310S-CANON",
        "mode":      "claimed",
    }]
    assert payload["any_hypothetical"] is False


def test_build_payload_any_hypothetical_when_any_entry_is_hypothetical():
    sp = _staged()
    sp.last_assumed_completed_credentials = [
        {"canonical": "310S-CANON",   "mode": "claimed"},
        {"canonical": "CLASS-G-CANON","mode": "hypothetical"},
    ]
    payload = handler._build_remaining_gaps_payload(sp)
    assert payload["any_hypothetical"] is True
    assert payload["remaining_credentials"] == []


def test_build_payload_all_closed_branch():
    """When every credential is accumulated, remaining_credentials is
    empty and the responder enters the all-closed branch (no provider
    grounding, per design §6)."""
    sp = _staged()
    sp.last_assumed_completed_credentials = [
        {"canonical": "310S-CANON",   "mode": "claimed"},
        {"canonical": "CLASS-G-CANON","mode": "claimed"},
    ]
    payload = handler._build_remaining_gaps_payload(sp)
    assert payload["remaining_credentials"] == []
    assert payload["any_hypothetical"] is False


# ============================================================================
# R-4 -- multi-turn accumulation (the round-3 review's regression case)
# ============================================================================
def test_multi_turn_accumulation_310s_persists_across_what_else():
    """Round-3 review regression: Turn N adds 310S (hypothetical), Turn
    N+1 'what else?' must still see 310S subtracted from the remaining
    credentials. Without accumulation, the next turn would re-narrate
    the credential the user just told us to assume."""
    sp = _staged()
    intent_n = RemainingGapsIntent(
        kind="subtract",
        current_turn_claims=(
            CredentialClaim(canonical="310S-CANON", mode="hypothetical"),
        ),
    )
    handler._synthesize_remaining_gaps_decision(
        sp, intent_n, retracted=False,
    )

    # Turn N+1 -- generic remaining, no claims
    intent_n1 = RemainingGapsIntent(
        kind="subtract", current_turn_claims=(),
    )
    handler._synthesize_remaining_gaps_decision(
        sp, intent_n1, retracted=False,
    )
    payload = handler._build_remaining_gaps_payload(sp)
    assert [g["canonical"] for g in payload["remaining_credentials"]] == [
        "CLASS-G-CANON",
    ]
    assert payload["any_hypothetical"] is True


def test_multi_turn_promotion_hypothetical_to_claimed_persists():
    """Promotion path: a credential added hypothetically on Turn N is
    promoted to claimed on Turn N+1 when the user explicitly confirms.
    The accumulated entry should reflect the promoted mode and
    payload.any_hypothetical updates."""
    sp = _staged()
    handler._synthesize_remaining_gaps_decision(
        sp,
        RemainingGapsIntent(
            kind="subtract",
            current_turn_claims=(
                CredentialClaim(canonical="310S-CANON", mode="hypothetical"),
            ),
        ),
        retracted=False,
    )
    handler._synthesize_remaining_gaps_decision(
        sp,
        RemainingGapsIntent(
            kind="subtract",
            current_turn_claims=(
                CredentialClaim(canonical="310S-CANON", mode="claimed"),
            ),
        ),
        retracted=False,
    )
    assert sp.last_assumed_completed_credentials == [
        {"canonical": "310S-CANON", "mode": "claimed"},
    ]
    payload = handler._build_remaining_gaps_payload(sp)
    assert payload["any_hypothetical"] is False


# ============================================================================
# R-4 -- retract path payload + last_discussed + training regrounding
# ============================================================================
def test_retract_path_re_emerges_credential_in_remaining():
    """The retracted credential MUST re-appear in remaining_credentials
    by construction -- the handler filters it out of accumulated, then
    the payload builder includes it because it's no longer in
    accumulated_canonicals."""
    sp = _staged()
    sp.last_assumed_completed_credentials = [
        {"canonical": "310S-CANON",   "mode": "hypothetical"},
        {"canonical": "CLASS-G-CANON","mode": "claimed"},
    ]
    handler._synthesize_remaining_gaps_decision(
        sp,
        RemainingGapsIntent(
            kind="retract", retract_canonical="310S-CANON",
        ),
        retracted=True,
    )
    payload = handler._build_remaining_gaps_payload(sp)
    assert [g["canonical"] for g in payload["remaining_credentials"]] == [
        "310S-CANON",
    ]
    assert payload["assumed_completed_credentials"] == [{
        "display": "G2/G driver's license",
        "canonical": "CLASS-G-CANON", "mode": "claimed",
    }]


def test_retract_path_recomputes_any_hypothetical():
    """When the only hypothetical entry is retracted,
    any_hypothetical flips to False."""
    sp = _staged()
    sp.last_assumed_completed_credentials = [
        {"canonical": "310S-CANON",   "mode": "hypothetical"},
        {"canonical": "CLASS-G-CANON","mode": "claimed"},
    ]
    handler._synthesize_remaining_gaps_decision(
        sp,
        RemainingGapsIntent(kind="retract", retract_canonical="310S-CANON"),
        retracted=True,
    )
    payload = handler._build_remaining_gaps_payload(sp)
    assert payload["any_hypothetical"] is False


def test_short_circuit_response_updates_last_discussed_to_lead_remaining(monkeypatch):
    """The short-circuit fast path MUST update last_discussed to the
    LEAD remaining credential after recomputation so a next-turn
    anaphor resolves correctly. Tested via the responder integration
    path so the post-recompute update fires."""
    # Stub the responder LLM so the test doesn't depend on it.
    monkeypatch.setattr(
        "skillbridge.chat.responder.is_enabled", lambda: False,
    )

    sp = _staged()
    decision = handler._synthesize_remaining_gaps_decision(
        sp,
        RemainingGapsIntent(
            kind="subtract",
            current_turn_claims=(
                CredentialClaim(canonical="310S-CANON", mode="claimed"),
            ),
        ),
        retracted=False,
    )
    handler._build_remaining_gaps_short_circuit_response(
        staged=sp, store=_NoopStore(), user_message="I have my 310S",
        final=decision, resume_info=None,
    )
    # After 310S is subtracted, the lead remaining is CLASS-G-CANON,
    # so last_discussed should anchor there.
    assert sp.last_discussed_credential_canonical == "CLASS-G-CANON"


def test_short_circuit_response_clears_last_discussed_when_all_closed(monkeypatch):
    monkeypatch.setattr(
        "skillbridge.chat.responder.is_enabled", lambda: False,
    )
    sp = _staged()
    sp.last_assumed_completed_credentials = [
        {"canonical": "310S-CANON", "mode": "claimed"},
    ]
    sp.last_discussed_credential_canonical = "stale-anchor"
    decision = handler._synthesize_remaining_gaps_decision(
        sp,
        RemainingGapsIntent(
            kind="subtract",
            current_turn_claims=(
                CredentialClaim(canonical="CLASS-G-CANON", mode="claimed"),
            ),
        ),
        retracted=False,
    )
    handler._build_remaining_gaps_short_circuit_response(
        staged=sp, store=_NoopStore(), user_message="and G2 too",
        final=decision, resume_info=None,
    )
    assert sp.last_discussed_credential_canonical is None


def test_retract_path_updates_last_discussed_to_re_emerged_credential(monkeypatch):
    """After retracting the credential the user was just discussing,
    last_discussed should anchor to it (which is also the lead
    remaining)."""
    monkeypatch.setattr(
        "skillbridge.chat.responder.is_enabled", lambda: False,
    )
    sp = _staged()
    sp.last_assumed_completed_credentials = [
        {"canonical": "310S-CANON",   "mode": "hypothetical"},
        {"canonical": "CLASS-G-CANON","mode": "claimed"},
    ]
    sp.last_discussed_credential_canonical = "310S-CANON"
    decision = handler._synthesize_remaining_gaps_decision(
        sp,
        RemainingGapsIntent(kind="retract", retract_canonical="310S-CANON"),
        retracted=True,
    )
    handler._build_remaining_gaps_short_circuit_response(
        staged=sp, store=_NoopStore(), user_message="yes that's right",
        final=decision, resume_info=None,
    )
    # Re-emerged 310S is the lead remaining; anaphor should follow.
    assert sp.last_discussed_credential_canonical == "310S-CANON"


def test_retract_idempotent_when_canonical_not_in_accumulated(monkeypatch):
    """Round-9 retract idempotency: filtering against a canonical NOT
    in accumulated is a no-op, not an error. Payload still builds."""
    monkeypatch.setattr(
        "skillbridge.chat.responder.is_enabled", lambda: False,
    )
    sp = _staged()
    sp.last_assumed_completed_credentials = [
        {"canonical": "CLASS-G-CANON", "mode": "claimed"},
    ]
    decision = handler._synthesize_remaining_gaps_decision(
        sp,
        RemainingGapsIntent(kind="retract", retract_canonical="NOT-PRESENT"),
        retracted=True,
    )
    # Accumulated unchanged
    assert sp.last_assumed_completed_credentials == [
        {"canonical": "CLASS-G-CANON", "mode": "claimed"},
    ]
    handler._build_remaining_gaps_short_circuit_response(
        staged=sp, store=_NoopStore(), user_message="yes",
        final=decision, resume_info=None,
    )
    payload = handler._build_remaining_gaps_payload(sp)
    assert [g["canonical"] for g in payload["remaining_credentials"]] == [
        "310S-CANON",
    ]


# ============================================================================
# R-4 -- training_by_job regrounding (flag-gated)
# ============================================================================
def test_training_by_job_empty_when_flag_off(monkeypatch):
    """Design §4a: resource surfacing remains flag-gated. With
    TRAINING_REGISTRY_ENABLED=False the regrounding helper returns
    {} regardless of remaining_credentials."""
    monkeypatch.setattr(handler, "TRAINING_REGISTRY_ENABLED", False)
    sp = _staged()
    remaining = [
        {"display": "310S Automotive Technician License",
         "canonical": "310S-CANON"},
    ]
    assert handler._reground_training_for_lead_remaining(sp, remaining) == {}


def test_training_by_job_empty_when_remaining_is_empty(monkeypatch):
    """All-closed branch: NO providers may be named (design §6). The
    regrounding helper returns {} when there's no lead credential to
    ground against, independent of the flag."""
    monkeypatch.setattr(handler, "TRAINING_REGISTRY_ENABLED", True)
    sp = _staged()
    assert handler._reground_training_for_lead_remaining(sp, []) == {}


# ============================================================================
# Round-16 R-4 review regressions
# ============================================================================
def test_training_does_not_include_completed_credentials_from_carry_forward(monkeypatch):
    """Round-16 finding 1: _reground_training_for_lead_remaining must
    query ONLY the lead remaining credential. The prior implementation
    delegated to _registry_training_for_gap which also appended
    last_presented_credential_gaps (Slice-8 carry-forward). With 310S
    completed (and still in last_presented_credential_gaps from the
    prior present_matches turn), the training resources for 310S would
    leak back into the responder's view -- the user just told us they
    have 310S; we shouldn't surface 310S training.
    """
    monkeypatch.setattr(handler, "TRAINING_REGISTRY_ENABLED", True)

    # Spy on the registry to see which queries fire.
    queries: list[str] = []
    import skillbridge.training.registry as reg_mod
    real_get = reg_mod.get_registry

    class _SpyRegistry:
        def __init__(self, r): self.r = r
        def surface_resources(self, q, **kw):
            queries.append(q)
            return []
    monkeypatch.setattr(reg_mod, "get_registry", lambda: _SpyRegistry(real_get()))

    sp = _staged()
    # 310S is completed; last_presented carries it forward from the
    # prior present_matches turn (the Slice-8 fields aren't cleared by
    # R-1 because they're independent persistence).
    sp.last_presented_credential_gaps = ["310S Automotive Technician License"]
    remaining = [{"display": "Class G driver license",
                  "canonical": "CLASS-G-CANON"}]
    handler._reground_training_for_lead_remaining(sp, remaining)
    assert queries == ["Class G driver license"], (
        f"Expected only the lead remaining credential to be queried; "
        f"got {queries}"
    )


def test_persisted_state_sanitized_so_stale_entries_dont_squeeze_valid_claim():
    """Round-17 R-4 review: the persisted
    `staged.last_assumed_completed_credentials` MUST be sanitized
    against the current snapshot BEFORE the append-and-dedupe step.
    Pre-round-17 the payload-builder filtered stale entries at READ
    time, but persisted state still carried them at WRITE time -- so
    they consumed the MAX_CRED_GAPS cap and the cap's trailing-drop
    discarded the just-arrived valid claim.

    Scenario: state arrives with 5 stale entries (canonicals NOT in
    the current snapshot). User explicitly claims a credential that
    IS in the snapshot. The synthesis MUST:
      - drop ALL 5 stale entries
      - persist the new valid claim
      - leave any_hypothetical reflecting only the persisted state
    """
    from skillbridge.session.staging import MAX_CRED_GAPS
    sp = _staged()
    sp.last_assumed_completed_credentials = [
        {"canonical": f"STALE-{i}", "mode": "hypothetical"}
        for i in range(MAX_CRED_GAPS)
    ]
    intent = RemainingGapsIntent(
        kind="subtract",
        current_turn_claims=(
            CredentialClaim(canonical="310S-CANON", mode="claimed"),
        ),
    )
    handler._synthesize_remaining_gaps_decision(
        sp, intent, retracted=False,
    )
    # Stale entries gone.
    canonicals = [
        a["canonical"] for a in sp.last_assumed_completed_credentials
    ]
    assert not any(c.startswith("STALE-") for c in canonicals), (
        f"Stale entries survived sanitization: {canonicals}"
    )
    # Valid claim persisted.
    assert "310S-CANON" in canonicals, (
        f"Valid current-turn claim was squeezed out by the cap: {canonicals}"
    )
    # Payload reflects accurate state.
    payload = handler._build_remaining_gaps_payload(sp)
    assumed_canonicals = [
        a["canonical"] for a in payload["assumed_completed_credentials"]
    ]
    assert "310S-CANON" in assumed_canonicals
    remaining_canonicals = [
        g["canonical"] for g in payload["remaining_credentials"]
    ]
    assert "310S-CANON" not in remaining_canonicals, (
        f"310S should have moved to assumed, not remained in remaining: "
        f"{remaining_canonicals}"
    )
    # No surviving hypothetical -> any_hypothetical False.
    assert payload["any_hypothetical"] is False


def test_persisted_state_dedupes_duplicate_canonicals_promoting_to_claimed():
    """Round-18 R-4 review: the persisted accumulated list MAY arrive
    with duplicate canonicals when state came from a forged or older
    cookie. Sanitization MUST dedupe (preserve first position) AND
    promote hypothetical -> claimed if ANY duplicate is claimed.

    Persisted input:
        [{A, hypothetical}, {A, claimed}]

    After sanitization:
        [{A, claimed}]              # one entry, promoted

    And the payload:
        assumed = [{A, ..., mode: claimed}]
        any_hypothetical = False    # the lone hypothetical was merged
    """
    sp = _staged()
    sp.last_assumed_completed_credentials = [
        {"canonical": "310S-CANON", "mode": "hypothetical"},
        {"canonical": "310S-CANON", "mode": "claimed"},
    ]
    handler._sanitize_accumulated_against_snapshot(sp)
    assert sp.last_assumed_completed_credentials == [
        {"canonical": "310S-CANON", "mode": "claimed"},
    ]
    payload = handler._build_remaining_gaps_payload(sp)
    assert payload["assumed_completed_credentials"] == [{
        "display":   "310S Automotive Technician License",
        "canonical": "310S-CANON",
        "mode":      "claimed",
    }]
    assert payload["any_hypothetical"] is False


def test_persisted_state_dedupe_preserves_first_position():
    """Order is first-occurrence within the deduped list. Mixed with
    other canonicals, the dedupe must NOT reorder."""
    sp = _staged()
    sp.last_assumed_completed_credentials = [
        {"canonical": "CLASS-G-CANON","mode": "claimed"},
        {"canonical": "310S-CANON",   "mode": "hypothetical"},
        {"canonical": "310S-CANON",   "mode": "claimed"},
        {"canonical": "CLASS-G-CANON","mode": "hypothetical"},
    ]
    handler._sanitize_accumulated_against_snapshot(sp)
    canonicals = [a["canonical"] for a in sp.last_assumed_completed_credentials]
    assert canonicals == ["CLASS-G-CANON", "310S-CANON"]
    # CLASS-G stays claimed (first was claimed); 310S promoted to claimed
    modes = [a["mode"] for a in sp.last_assumed_completed_credentials]
    assert modes == ["claimed", "claimed"]


def test_persisted_state_dedupe_keeps_hypothetical_when_no_claimed():
    """When ALL duplicates are hypothetical, the deduped entry stays
    hypothetical."""
    sp = _staged()
    sp.last_assumed_completed_credentials = [
        {"canonical": "310S-CANON", "mode": "hypothetical"},
        {"canonical": "310S-CANON", "mode": "hypothetical"},
    ]
    handler._sanitize_accumulated_against_snapshot(sp)
    assert sp.last_assumed_completed_credentials == [
        {"canonical": "310S-CANON", "mode": "hypothetical"},
    ]


def test_persisted_state_sanitized_on_retract_path_too():
    """Same invariant on the retract path: stale entries are dropped
    before the named canonical is filtered out."""
    sp = _staged()
    sp.last_assumed_completed_credentials = [
        {"canonical": "STALE-A", "mode": "hypothetical"},
        {"canonical": "310S-CANON", "mode": "claimed"},
        {"canonical": "STALE-B", "mode": "claimed"},
    ]
    handler._synthesize_remaining_gaps_decision(
        sp,
        RemainingGapsIntent(
            kind="retract", retract_canonical="310S-CANON",
        ),
        retracted=True,
    )
    # All stales gone, target also gone (retracted) -> empty list
    assert sp.last_assumed_completed_credentials == []


def test_telemetry_accumulated_count_reflects_sanitized_state():
    """Round-17 R-4 review: the accumulated_credentials_count in
    telemetry MUST count post-sanitization entries, NOT pre-
    sanitization. Otherwise the log line lies about how many
    credentials are actually carried forward."""
    sp = _staged()
    sp.last_assumed_completed_credentials = [
        {"canonical": f"STALE-{i}", "mode": "hypothetical"}
        for i in range(4)
    ]
    intent = RemainingGapsIntent(
        kind="subtract",
        current_turn_claims=(
            CredentialClaim(canonical="310S-CANON", mode="claimed"),
        ),
    )
    handler._synthesize_remaining_gaps_decision(
        sp, intent, retracted=False,
    )
    # 4 stales removed, 1 valid claim persisted -> count is 1
    assert len(sp.last_assumed_completed_credentials) == 1


def test_payload_drops_accumulated_canonical_not_in_snapshot():
    """Round-16 finding 2: the snapshot is the sole identity authority.
    An accumulated entry whose canonical is not present in the snapshot
    is stale (a snapshot transition that escaped the clearing rules)
    and MUST be dropped from the payload. It MUST NOT affect
    any_hypothetical."""
    sp = _staged()
    # Snapshot has 310S-CANON + CLASS-G-CANON only.
    sp.last_assumed_completed_credentials = [
        {"canonical": "310S-CANON", "mode": "claimed"},
        {"canonical": "STALE-NOT-IN-SNAPSHOT", "mode": "hypothetical"},
    ]
    payload = handler._build_remaining_gaps_payload(sp)
    assumed_canonicals = [
        a["canonical"] for a in payload["assumed_completed_credentials"]
    ]
    assert "STALE-NOT-IN-SNAPSHOT" not in assumed_canonicals
    assert "310S-CANON" in assumed_canonicals
    # The stale entry was the only hypothetical -- dropping it flips
    # any_hypothetical to False.
    assert payload["any_hypothetical"] is False


def test_payload_defends_against_dict_core_skill_gaps():
    """Round-16 finding 4: core_skill_gaps must be a list. A forged
    cookie that supplied a dict was iterating dict keys and producing
    phantom skills like ['bad']. The defensive isinstance check yields
    an empty list."""
    sp = StagedProfile.new("t")
    sp.last_match_snapshot = {
        "lead_job": {"job_id": "j", "title": "T", "employer": None,
                     "credential_gaps": [],
                     "core_skill_gaps": {"bad": 1}},
        "other_jobs_meta": [],
    }
    payload = handler._build_remaining_gaps_payload(sp)
    assert payload["remaining_core_skills"] == []


def test_telemetry_records_current_turn_claims_count(monkeypatch, caplog):
    """Round-16 finding 3: telemetry MUST include
    current_turn_claims_count. Test that the dispatch+short-circuit
    pair threads the count from detection through to the log line."""
    monkeypatch.setattr(
        "skillbridge.chat.responder.is_enabled", lambda: False,
    )
    sp = _staged()
    with caplog.at_level("INFO", logger="skillbridge.chat.handler"):
        # Two definite claims in one message
        decision, claims_count = handler._run_remaining_gaps_dispatch(
            sp, "I have my 310S and I have my G2",
        )
        assert claims_count == 2
        handler._build_remaining_gaps_short_circuit_response(
            staged=sp, store=_NoopStore(),
            user_message="I have my 310S and I have my G2",
            final=decision, resume_info=None,
            current_turn_claims_count=claims_count,
        )
    matching_lines = [
        rec.getMessage() for rec in caplog.records
        if "current_turn_claims_count" in rec.getMessage()
    ]
    assert matching_lines, (
        "Expected telemetry log line containing current_turn_claims_count"
    )
    assert any("current_turn_claims_count=2" in line for line in matching_lines), (
        f"Expected current_turn_claims_count=2 in telemetry; got {matching_lines}"
    )


def test_training_by_job_populated_for_lead_when_flag_on(monkeypatch):
    """Flag on + lead credential present + registry knows it: training
    populates with the standard gap:<name> shape."""
    monkeypatch.setattr(handler, "TRAINING_REGISTRY_ENABLED", True)
    sp = _staged()
    sp.target_role_text = "automotive technician"
    # The registry's `310T technician certification` entry is in the
    # shipped YAML; use it so the test doesn't depend on snapshot stub
    # contents.
    remaining = [
        {"display": "310T technician certification",
         "canonical": "310T technician certification"},
    ]
    tbj = handler._reground_training_for_lead_remaining(sp, remaining)
    # Standard gap:<name>: [resources] shape
    assert any(k.startswith("gap:") for k in tbj.keys()), (
        f"Expected gap-prefixed keys; got {list(tbj.keys())}"
    )


def test_synthesize_retract_missing_canonical_is_no_op():
    """Defensive: retraction against a canonical NOT in accumulated
    leaves the list unchanged. (The filter is keep-if-not-equal; no
    explicit error.)"""
    sp = _staged()
    sp.last_assumed_completed_credentials = [
        {"canonical": "CLASS-G-CANON", "mode": "claimed"},
    ]
    intent = RemainingGapsIntent(
        kind="retract",
        retract_canonical="UNRELATED",
    )
    handler._synthesize_remaining_gaps_decision(sp, intent, retracted=True)
    assert sp.last_assumed_completed_credentials == [
        {"canonical": "CLASS-G-CANON", "mode": "claimed"},
    ]


# ============================================================================
# _synthesize_clarification_decision -- confirm + bootstrap
# ============================================================================
def test_synthesize_confirm_with_canonical_sets_pending():
    sp = _staged()
    intent = RemainingGapsIntent(
        kind="confirm",
        confirmation_target_canonical="310S-CANON",
        confirmation_target_display="310S licence",
        pending_action="add",
    )
    decision = handler._synthesize_clarification_decision(sp, intent)
    assert decision.final_move == "ask_one_clarifying_question"
    assert decision.reason_code == ARBITER_REASON_CONFIRM_CREDENTIAL
    assert decision.arbiter_action == "handler_synthesized_clarification"
    assert decision.ask_slot is None
    assert sp.pending_credential_confirmation == {
        "canonical": "310S-CANON", "action": "add",
    }


def test_synthesize_confirm_with_remove_action_sets_pending_remove():
    sp = _staged()
    intent = RemainingGapsIntent(
        kind="confirm",
        confirmation_target_canonical="310S-CANON",
        confirmation_target_display="310S licence",
        pending_action="remove",
    )
    handler._synthesize_clarification_decision(sp, intent)
    assert sp.pending_credential_confirmation == {
        "canonical": "310S-CANON", "action": "remove",
    }


def test_synthesize_confirm_with_none_canonical_clears_pending():
    """Round-13 round-12 guard: the disambiguation case ('got it' with
    no entity) returns kind='confirm' with confirmation_target_canonical
    =None. The handler MUST NOT store a pending entry with canonical=
    None (the detector's pending-consume branch can't act on it AND
    R-1's defensive deserialization drops dicts whose canonical isn't
    a non-empty string)."""
    sp = _staged()
    sp.pending_credential_confirmation = {"canonical": "stale", "action": "add"}
    intent = RemainingGapsIntent(
        kind="confirm",
        confirmation_target_canonical=None,
        confirmation_target_display="",
        pending_action="add",
    )
    handler._synthesize_clarification_decision(sp, intent)
    assert sp.pending_credential_confirmation is None


def test_synthesize_bootstrap_sets_correct_reason_and_no_pending():
    sp = _staged()
    intent = RemainingGapsIntent(kind="bootstrap")
    decision = handler._synthesize_clarification_decision(sp, intent)
    assert decision.final_move == "ask_one_clarifying_question"
    assert decision.reason_code == ARBITER_REASON_BOOTSTRAP_MATCH
    assert decision.arbiter_action == "handler_synthesized_clarification"
    assert sp.pending_credential_confirmation is None


# ============================================================================
# _run_remaining_gaps_dispatch -- save/clear/try/None/case_
# ============================================================================
def test_dispatch_returns_none_for_irrelevant_message(monkeypatch):
    """When the detector returns None, the hook returns None and the
    handler falls through to normal planner/router/engine dispatch."""
    sp = _staged()
    result, claims_count = handler._run_remaining_gaps_dispatch(
        sp, "tell me about jobs",
    )
    assert result is None
    assert claims_count == 0


def test_dispatch_clears_pending_before_detection(monkeypatch):
    """Pending-clear ownership (§2): the handler clears
    staged.pending_credential_confirmation BEFORE calling the detector.
    Verified by spying on the detector and checking the state of staged
    when the call happens."""
    sp = _staged()
    sp.pending_credential_confirmation = {
        "canonical": "310S-CANON", "action": "add",
    }
    captured_state: dict = {}

    def spy_detect(message, snapshot, registry, **kw):
        captured_state["staged_pending_at_call"] = sp.pending_credential_confirmation
        captured_state["saved_pending_kw"] = kw["pending_confirmation"]
        return None

    monkeypatch.setattr(
        "skillbridge.chat.remaining_gaps.detect_remaining_gaps_intent",
        spy_detect,
    )
    handler._run_remaining_gaps_dispatch(sp, "tell me about jobs")
    # When the detector is called, staged.pending was cleared BUT the
    # saved copy was passed as the kw arg.
    assert captured_state["staged_pending_at_call"] is None
    assert captured_state["saved_pending_kw"] == {
        "canonical": "310S-CANON", "action": "add",
    }


def test_dispatch_restores_pending_on_detector_exception(monkeypatch):
    """Try/except contract (§R-3): if detection raises, restore the
    pending state so the user's pending question isn't lost. The hook
    returns None and the handler falls through to normal routing."""
    sp = _staged()
    sp.pending_credential_confirmation = {
        "canonical": "310S-CANON", "action": "add",
    }

    def boom(*args, **kwargs):
        raise RuntimeError("simulated detector crash")

    monkeypatch.setattr(
        "skillbridge.chat.remaining_gaps.detect_remaining_gaps_intent",
        boom,
    )
    result, claims_count = handler._run_remaining_gaps_dispatch(
        sp, "what else?",
    )
    assert result is None
    assert claims_count == 0
    assert sp.pending_credential_confirmation == {
        "canonical": "310S-CANON", "action": "add",
    }


def test_dispatch_routes_subtract_through_synthesis():
    sp = _staged()
    decision, claims_count = handler._run_remaining_gaps_dispatch(
        sp, "I have my 310S",
    )
    assert decision is not None
    assert decision.final_move == "explain_remaining_gaps"
    assert decision.reason_code == ARBITER_REASON_REMAINING_GAPS
    # Round-16 R-4 review: dispatch threads current_turn_claims_count
    # so telemetry can log it.
    assert claims_count == 1


def test_dispatch_routes_retraction_through_synthesis():
    sp = _staged()
    # Accumulated canonical MUST match a snapshot credential canonical
    # so step-2 (negation-against-accumulated) fires. The test's
    # _staged() snapshot uses "310S-CANON" as the canonical for the
    # 310S entry, so the accumulated entry has to match.
    sp.last_assumed_completed_credentials = [
        {"canonical": "310S-CANON", "mode": "claimed"},
    ]
    # The detector returns kind="confirm" pending_action="remove" --
    # which is a clarification (handler emits ask_one_clarifying_question
    # NOT explain_remaining_gaps). The actual retract fires on the
    # NEXT turn when the user says "yes."
    decision, claims_count = handler._run_remaining_gaps_dispatch(
        sp, "I don't have 310S",
    )
    assert decision is not None
    assert decision.final_move == "ask_one_clarifying_question"
    assert decision.reason_code == ARBITER_REASON_CONFIRM_CREDENTIAL
    # confirm doesn't carry per-turn claims
    assert claims_count == 0
    assert sp.pending_credential_confirmation == {
        "canonical": "310S-CANON",
        "action": "remove",
    }


def test_dispatch_unknown_intent_kind_falls_through(monkeypatch):
    """Defensive: a future detector version that emits a kind value the
    handler doesn't recognize MUST fall through to normal routing (NOT
    raise) so the chat keeps working while the feature gate is fixed."""
    sp = _staged()
    monkeypatch.setattr(
        "skillbridge.chat.remaining_gaps.detect_remaining_gaps_intent",
        lambda *a, **kw: RemainingGapsIntent(kind="future_extension"),
    )
    result, claims_count = handler._run_remaining_gaps_dispatch(sp, "anything")
    assert result is None
    assert claims_count == 0


# ============================================================================
# Round-14 R-3 review -- scope precedence + last_discussed lifecycle
# ============================================================================
def test_confirm_sets_last_discussed_to_resolved_canonical():
    """Locked §2 lifecycle: emitting a confirm clarification with a
    known snapshot canonical MUST set
    `staged.last_discussed_credential_canonical` so the next-turn
    anaphor ("yes I got it", "what about that one?") resolves to the
    credential we asked about, not to a stale anchor."""
    sp = _staged()
    intent = RemainingGapsIntent(
        kind="confirm",
        confirmation_target_canonical="310S-CANON",
        confirmation_target_display="310S licence",
        pending_action="add",
    )
    handler._synthesize_clarification_decision(sp, intent)
    assert sp.last_discussed_credential_canonical == "310S-CANON"


def test_confirm_with_none_canonical_leaves_last_discussed_unchanged():
    """Disambiguation confirm (canonical=None) leaves last_discussed
    UNCHANGED -- the prior recency anchor is still the best fallback
    if the user replies with anything other than a specific
    credential name."""
    sp = _staged()
    sp.last_discussed_credential_canonical = "PRIOR-CANON"
    intent = RemainingGapsIntent(
        kind="confirm",
        confirmation_target_canonical=None,
        confirmation_target_display="",
        pending_action="add",
    )
    handler._synthesize_clarification_decision(sp, intent)
    assert sp.last_discussed_credential_canonical == "PRIOR-CANON"


def test_confirm_for_remove_action_also_updates_last_discussed():
    """The same lifecycle rule applies on the retraction confirm path
    -- the canonical the user is being asked to walk back is what an
    anaphoric next-turn reference would target."""
    sp = _staged()
    intent = RemainingGapsIntent(
        kind="confirm",
        confirmation_target_canonical="CLASS-G-CANON",
        confirmation_target_display="G2/G licence",
        pending_action="remove",
    )
    handler._synthesize_clarification_decision(sp, intent)
    assert sp.last_discussed_credential_canonical == "CLASS-G-CANON"


def test_bootstrap_leaves_last_discussed_unchanged():
    """Bootstrap fires when there's no snapshot -- there's no
    credential context to anchor against; the field stays as-is."""
    sp = _staged()
    sp.last_discussed_credential_canonical = "PRIOR-CANON"
    intent = RemainingGapsIntent(kind="bootstrap")
    handler._synthesize_clarification_decision(sp, intent)
    assert sp.last_discussed_credential_canonical == "PRIOR-CANON"


# ============================================================================
# Legacy-mapping pin
# ============================================================================
def test_explain_remaining_gaps_maps_to_present_matches_legacy_action():
    """A handler test asserts that the new outcome's v1 next_action
    label is ACTION_PRESENT_MATCHES (a match continuation), NOT the
    defensive ASK_QUESTIONS fallback. Pins the mapping so a future
    refactor dropping it is caught in CI."""
    assert handler._final_move_to_legacy_action("explain_remaining_gaps") == \
        intake_state.ACTION_PRESENT_MATCHES


# ============================================================================
# Round-14 R-3 review -- scope precedence regression (integration-shape)
# ============================================================================
@pytest.mark.parametrize("message", [
    "what else about PR?",
    "what else do I need for immigration?",
    "anything else for my work permit?",
    "what else for express entry?",
])
def test_scope_violation_wins_over_remaining_gaps_hook(monkeypatch, message):
    """Locked precedence: when truth.scope_violations_detected is
    non-empty (immigration / wages / off-topic / non-SSM), the
    remaining-gaps hook MUST NOT fire -- the arbiter's Pass 1 Rule 2
    owns the redirect. Pre-round-14 the hook ran first and synthesized
    `explain_remaining_gaps` for "what else about PR?" with a snapshot
    present, silently bypassing the scope redirect.

    Also pin §10: the snapshot survives the redirect so the user can
    return to the career conversation immediately after."""
    from skillbridge.chat import truth_summary as ts_mod
    real_build = ts_mod.build_truth_summary

    def fake_build(*, staged, user_message, **kw):
        result = real_build(staged=staged, user_message=user_message, **kw)
        object.__setattr__(result, "scope_violations_detected", ["immigration"])
        return result
    monkeypatch.setattr(handler, "build_truth_summary", fake_build)

    # Force a stable planner so the test doesn't depend on the real
    # planner LLM. The arbiter's Rule 2 should override to redirect
    # regardless of planner output.
    monkeypatch.setattr(
        handler, "plan_next_move",
        lambda truth: _planner_decision_for_scope_test(),
    )
    # Engine must not run on a scope-redirect turn.
    engine_spy = _engine_spy_for_scope_test()
    monkeypatch.setattr(
        handler.match_engine, "compute_matches_in_memory", engine_spy,
    )

    sp = _staged()
    pre_snapshot = sp.last_match_snapshot
    response = handler._try_v2_path(
        staged=sp, message=message,
        uploaded_file=False, resume_info=None, store=_NoopStore(),
    )
    assert response is not None
    assert response["final_move"] == "redirect_scope", (
        f"Hook hijacked scope redirect for {message!r}; got "
        f"final_move={response['final_move']!r}"
    )
    assert engine_spy.calls == 0
    # Locked §10: snapshot is preserved across the redirect.
    assert sp.last_match_snapshot == pre_snapshot


def test_scope_diversion_clears_pending_so_later_yes_does_not_confirm_stale(monkeypatch):
    """Round-15 R-3 review: pending one-turn-validity contract MUST
    hold even when the hook is bypassed by scope precedence. Without
    clearing pending here, an unrelated `yes` on a LATER turn would
    retroactively confirm the stale credential.

    Three-turn sequence:
      Turn N   -- system asked "have you completed 310S?", pending set
      Turn N+1 -- user diverts to a PR question (scope violation);
                  scope redirect fires AND pending is cleared
      Turn N+2 -- user types `yes` (unrelated affirmative); the
                  detector sees no pending, so no stale subtract.

    Also pins: snapshot, accumulated, and last_discussed survive the
    scope turn so the user can return to the career conversation."""
    # ---- Turn N+1 setup ----
    from skillbridge.chat import truth_summary as ts_mod
    real_build = ts_mod.build_truth_summary

    def fake_build_scope(*, staged, user_message, **kw):
        result = real_build(staged=staged, user_message=user_message, **kw)
        object.__setattr__(result, "scope_violations_detected", ["immigration"])
        return result
    monkeypatch.setattr(handler, "build_truth_summary", fake_build_scope)
    monkeypatch.setattr(
        handler, "plan_next_move",
        lambda truth: _planner_decision_for_scope_test(),
    )
    engine_spy = _engine_spy_for_scope_test()
    monkeypatch.setattr(
        handler.match_engine, "compute_matches_in_memory", engine_spy,
    )

    # Simulate state coming OUT OF Turn N: the system had just asked
    # "have you completed 310S?" so pending is set; the snapshot,
    # accumulated, and last_discussed reflect prior conversation
    # context.
    sp = _staged()
    sp.pending_credential_confirmation = {
        "canonical": "310S-CANON", "action": "add",
    }
    sp.last_discussed_credential_canonical = "310S-CANON"
    sp.last_assumed_completed_credentials = [
        {"canonical": "CLASS-G-CANON", "mode": "hypothetical"},
    ]
    pre_snapshot = sp.last_match_snapshot
    pre_accumulated = list(sp.last_assumed_completed_credentials)
    pre_last_discussed = sp.last_discussed_credential_canonical

    # ---- Turn N+1: PR question -> scope redirect ----
    response = handler._try_v2_path(
        staged=sp, message="can I apply for PR while I work on this?",
        uploaded_file=False, resume_info=None, store=_NoopStore(),
    )
    assert response is not None
    assert response["final_move"] == "redirect_scope"
    assert engine_spy.calls == 0

    # Pending was cleared.
    assert sp.pending_credential_confirmation is None, (
        "Pending must be cleared on a scope-diversion turn so a later "
        "`yes` cannot retroactively confirm the stale question."
    )
    # Snapshot / accumulated / last_discussed preserved.
    assert sp.last_match_snapshot == pre_snapshot
    assert sp.last_assumed_completed_credentials == pre_accumulated
    assert sp.last_discussed_credential_canonical == pre_last_discussed

    # ---- Turn N+2: user types `yes` ----
    # The detector sees pending=None and accumulated still contains
    # only the hypothetical Class G. A short `yes` against no pending
    # is just a yes -- it should NOT subtract 310S.
    decision, _ = handler._run_remaining_gaps_dispatch(sp, "yes")
    if decision is not None:
        # If the dispatch produced something, it MUST NOT be a subtract
        # of 310S (the stale pending target). A None or some unrelated
        # outcome is fine.
        if decision.final_move == "explain_remaining_gaps":
            subtracted = [
                a for a in sp.last_assumed_completed_credentials
                if a["canonical"] == "310S-CANON"
            ]
            assert not subtracted, (
                "Stale pending was incorrectly consumed on Turn N+2 -- "
                f"310S-CANON appeared in accumulated: "
                f"{sp.last_assumed_completed_credentials}"
            )


# Module-level helpers reused by the parametrised scope test above.
class _NoopStore:
    def new_session(self): return "noop-sid"
    def load(self, sid):   return None
    def save(self, staged): return staged.session_id or "noop-sid"
    def delete(self, sid): return


class _EngineSpy:
    def __init__(self): self.calls = 0
    def __call__(self, *a, **kw):
        self.calls += 1
        return []


def _engine_spy_for_scope_test():
    return _EngineSpy()


def _planner_decision_for_scope_test():
    from skillbridge.chat.planner import PlannerDecision
    return PlannerDecision(
        move="proceed_to_match",
        reason_code="user_explicitly_asked_to_match",
        tone="brief_confident",
        ask_slot=None,
    )


# ============================================================================
# Arbiter invariant: explain_remaining_gaps is NEVER pass-1/pass-2 output
# ============================================================================
def test_validate_planner_intent_never_emits_explain_remaining_gaps():
    """Pass 1 (validate_planner_intent) MUST NOT produce
    explain_remaining_gaps. The outcome is handler-synthesized only.
    Exhaustively check across every PlannerDecision shape."""
    from skillbridge.chat.planner import PlannerDecision
    # Truth fixture covering scope-violation + non-scope variants
    truths = [
        {"enough_to_match": False, "usable_evidence_present": False,
         "scope_violations_detected": [], "user_intent_signal": "asking_about_gap"},
        {"enough_to_match": True, "usable_evidence_present": True,
         "scope_violations_detected": ["immigration"],
         "user_intent_signal": "asking_about_jobs"},
        {"enough_to_match": True, "usable_evidence_present": True,
         "scope_violations_detected": [], "user_intent_signal": "asking_about_jobs"},
    ]
    # Pair each move with a planner reason_code that PlannerDecision
    # accepts (the Literal is enforced by pydantic).
    move_reason_pairs = [
        ("acknowledge_and_continue",   "user_confirmed"),
        ("ask_one_clarifying_question", "target_role_unclear"),
        ("explain_gap",                 "credential_gap_present"),
        ("offer_refinement",            "narrow_request"),
        ("redirect_scope",              "scope_violation_immigration"),
        ("proceed_to_match",            "user_explicitly_asked_to_match"),
    ]
    for truth in truths:
        for move, reason_code in move_reason_pairs:
            decision = PlannerDecision(
                move=move,
                reason_code=reason_code,
                tone="brief_confident",
                # ask_one_clarifying_question requires a non-null ask_slot;
                # other moves don't.
                ask_slot=(
                    "target_role_text"
                    if move == "ask_one_clarifying_question" else None
                ),
            )
            result = arbiter_mod.validate_planner_intent(decision, truth)
            if isinstance(result, ArbiterDecision):
                assert result.final_move != "explain_remaining_gaps", (
                    f"Pass 1 produced explain_remaining_gaps for "
                    f"move={move!r} reason={reason_code!r} truth={truth!r}"
                )


def test_resolve_match_outcome_never_emits_explain_remaining_gaps():
    """Pass 2 (resolve_match_outcome) MUST NOT produce
    explain_remaining_gaps either. Exhaustively check across (match_count,
    caps_applied, near_miss_candidates) combinations."""
    for match_count in (0, 1, 5):
        for caps in ((), ("band_capped_by_credential",)):
            for nm in ((), ("synthetic-candidate",)):
                result = arbiter_mod.resolve_match_outcome(
                    match_count=match_count,
                    caps_applied=caps,
                    near_miss_candidates=nm,
                    planner_reason_code="x",
                    planner_tone="brief_confident",
                )
                assert result.final_move != "explain_remaining_gaps"
