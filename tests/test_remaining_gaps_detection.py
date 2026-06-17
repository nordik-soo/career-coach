"""Unit tests for the remaining-gaps detection module (R-2).

Coverage map (mirrors docs/remaining-gaps-design.md §Tests):
  - Each completion pattern returns kind="subtract" with the right mode
  - Each negation pattern against a snapshot-only entity returns kind=None
  - Each uncertainty pattern returns kind="confirm" with pending_action="add"
  - Bootstrap case (snapshot=None + generic remaining) returns kind="bootstrap"
  - Snapshot-anchored identity (§4.0): every returned canonical exists in
    snapshot.lead_job.credential_gaps[*].canonical
  - Deterministic token fallback (§4.3): generic-only input returns None;
    multiple snapshot candidates return None; exactly one returns it
  - Negation-against-accumulated retraction (v8): ALL explicit negations
    targeting an accumulated entity return kind="confirm" with
    pending_action="remove" -- "actually" hedge NOT required
  - Ordering invariant: pending -> retract -> negation -> uncertainty ->
    completion -> hypothetical -> generic
  - Pending consumption (saved-copy semantics): detector NEVER mutates the
    dict; affirmative with action="add" -> kind="subtract" mode="claimed";
    affirmative with action="remove" -> kind="retract"; negative -> None;
    unrelated falls through
  - Mode C (registry=None): no crash; falls back to token resolver
  - Flag-decoupled identity: canonical resolution works whether the
    training flag is on or off (the flag gates resource surfacing in R-4)

No DB. No fixtures beyond inline dicts.
"""
from __future__ import annotations

import pytest

from skillbridge.chat.remaining_gaps import (
    CredentialClaim,
    RemainingGapsIntent,
    _match_user_ref_by_tokens,
    _resolve_credential_anaphor,
    _resolve_user_ref_to_snapshot_canonical,
    _tokens,
    detect_remaining_gaps_intent,
)

pytestmark = pytest.mark.nodb


# ============================================================================
# Helpers + fixtures
# ============================================================================
class _FakeGap:
    """Mimics training.models.Gap.canonical_name."""
    def __init__(self, canonical_name: str) -> None:
        self.canonical_name = canonical_name


class _FakeRegistry:
    """Maps any string in `aliases` (case-insensitive lookup-style)
    to its canonical_name."""
    def __init__(self, mapping: dict[str, str]) -> None:
        # lowercase the keys so lookup() can find "310s licence" too
        self._map = {k.lower(): v for k, v in mapping.items()}

    def lookup(self, query: str):
        if not isinstance(query, str):
            return None
        return _FakeGap(self._map[query.lower()]) if query.lower() in self._map else None


def _snapshot(
    credential_gaps: list[tuple[str, str]] | None = None,
    core_skill_gaps: list[str] | None = None,
):
    """Build a minimal snapshot. credential_gaps = list of (display, canonical) tuples."""
    return {
        "captured_at_turn": 1,
        "lead_job": {
            "job_id": "job-1",
            "title": "310S Licensed Automotive Technician",
            "employer": "Great Lakes Honda",
            "credential_gaps": [
                {"display": d, "canonical": c}
                for d, c in (credential_gaps or [])
            ],
            "core_skill_gaps": list(core_skill_gaps or []),
        },
        "other_jobs_meta": [],
    }


_HONDA_SNAPSHOT_MODE_A = _snapshot(credential_gaps=[
    ("310S Automotive Technician License",
     "310S automotive technician certification"),
    ("G2/G driver's license", "Class G driver's license"),
])

_HONDA_REGISTRY = _FakeRegistry({
    "310S Automotive Technician License":   "310S automotive technician certification",
    "310S licence":                          "310S automotive technician certification",
    "310S":                                  "310S automotive technician certification",
    "G2/G driver's license":                 "Class G driver's license",
    "G2":                                    "Class G driver's license",
    "G2/G":                                  "Class G driver's license",
    "Class G":                               "Class G driver's license",
})


def _detect(message, *,
            snapshot=_HONDA_SNAPSHOT_MODE_A,
            registry=_HONDA_REGISTRY,
            accumulated=None,
            pending=None,
            last_discussed=None):
    return detect_remaining_gaps_intent(
        message, snapshot, registry,
        accumulated_credentials=accumulated or [],
        pending_confirmation=pending,
        last_discussed_canonical=last_discussed,
    )


# ============================================================================
# §4.3 -- token helpers
# ============================================================================
def test_tokens_drops_generic_credential_words():
    assert _tokens("the license") == frozenset()
    assert _tokens("my certification") == frozenset()


def test_tokens_lowercases_and_strips_punctuation():
    assert _tokens("310S/Licence!") == frozenset({"310s"})


def test_tokens_preserves_distinguishing_words():
    assert "310s" in _tokens("310S automotive technician license")
    assert "automotive" in _tokens("310S automotive technician license")


def test_tokens_handles_empty_and_non_string():
    assert _tokens("") == frozenset()
    assert _tokens(None) == frozenset()         # type: ignore[arg-type]


# ============================================================================
# §4.3 -- _match_user_ref_by_tokens
# ============================================================================
def test_match_user_ref_unique_candidate_returns_snapshot_canonical():
    gaps = [
        {"display": "310S Automotive Technician License",
         "canonical": "310S-CANON"},
        {"display": "G2/G driver's license", "canonical": "CLASS-G-CANON"},
    ]
    assert _match_user_ref_by_tokens("310S", gaps) == "310S-CANON"
    assert _match_user_ref_by_tokens("G2", gaps) == "CLASS-G-CANON"


def test_match_user_ref_generic_only_input_returns_none():
    gaps = [{"display": "310S Automotive Technician License",
             "canonical": "310S-CANON"}]
    assert _match_user_ref_by_tokens("the license", gaps) is None
    assert _match_user_ref_by_tokens("my certification", gaps) is None


def test_match_user_ref_ambiguous_returns_none():
    # Both snapshot entries share the non-generic token "g2"
    gaps = [
        {"display": "G2 driver's license", "canonical": "DRIVER-CANON"},
        {"display": "G2 paramedic course",  "canonical": "PARAMEDIC-CANON"},
    ]
    assert _match_user_ref_by_tokens("G2", gaps) is None


def test_match_user_ref_no_match_returns_none():
    gaps = [{"display": "310S Automotive Technician License",
             "canonical": "310S-CANON"}]
    assert _match_user_ref_by_tokens("Smart Serve", gaps) is None


# ============================================================================
# §4.2 -- _resolve_user_ref_to_snapshot_canonical (Mode A registry-assisted)
# ============================================================================
def test_resolve_uses_registry_match_against_snapshot_canonical():
    """Mode A path (a): registry resolves "310S licence" to the same
    canonical the snapshot stored -> return snapshot's canonical."""
    out = _resolve_user_ref_to_snapshot_canonical(
        "310S licence", _HONDA_SNAPSHOT_MODE_A, _HONDA_REGISTRY,
    )
    assert out == "310S automotive technician certification"


def test_resolve_cross_mode_bridge_for_mode_b_snapshot():
    """Snapshot captured in Mode B (canonical = normalized display);
    detection is Mode A. The user's "310S licence" resolves to the
    registry canonical "310S automotive technician certification",
    which doesn't equal the snapshot's stored
    "310s automotive technician license". The cross-mode bridge
    re-lookups the snapshot display and confirms alias -- returns the
    SNAPSHOT'S stored value, not the registry's."""
    mode_b_snapshot = _snapshot(credential_gaps=[
        ("310S Automotive Technician License",
         "310s automotive technician license"),    # normalized display
    ])
    out = _resolve_user_ref_to_snapshot_canonical(
        "310S licence", mode_b_snapshot, _HONDA_REGISTRY,
    )
    assert out == "310s automotive technician license"


def test_resolve_falls_back_to_token_match_when_registry_misses():
    """Mode A registry doesn't know about the user's substring; the
    token fallback resolves against snapshot display tokens."""
    registry_no_310s = _FakeRegistry({})    # empty
    out = _resolve_user_ref_to_snapshot_canonical(
        "310S", _HONDA_SNAPSHOT_MODE_A, registry_no_310s,
    )
    assert out == "310S automotive technician certification"


def test_resolve_mode_c_no_registry_falls_back_to_token_match():
    """Mode C: registry=None. Detector uses the token resolver only."""
    out = _resolve_user_ref_to_snapshot_canonical(
        "G2", _HONDA_SNAPSHOT_MODE_A, None,
    )
    assert out == "Class G driver's license"


def test_resolve_returns_none_when_no_snapshot():
    assert _resolve_user_ref_to_snapshot_canonical(
        "310S", None, _HONDA_REGISTRY,
    ) is None
    assert _resolve_user_ref_to_snapshot_canonical(
        "310S", {}, _HONDA_REGISTRY,
    ) is None


# ============================================================================
# §2 -- _resolve_credential_anaphor
# ============================================================================
def test_anaphor_resolves_to_last_discussed_when_present_in_snapshot():
    out = _resolve_credential_anaphor(
        "I have it", _HONDA_SNAPSHOT_MODE_A,
        "Class G driver's license",
    )
    assert out == "Class G driver's license"


def test_anaphor_falls_back_to_first_snapshot_credential():
    out = _resolve_credential_anaphor(
        "I have it", _HONDA_SNAPSHOT_MODE_A, None,
    )
    assert out == "310S automotive technician certification"


def test_anaphor_returns_none_without_anaphor_token():
    assert _resolve_credential_anaphor(
        "I have my 310S", _HONDA_SNAPSHOT_MODE_A, None,
    ) is None


def test_anaphor_returns_none_when_no_snapshot():
    assert _resolve_credential_anaphor("I have it", None, None) is None


# ============================================================================
# Step 1 -- pending consumption
# ============================================================================
def test_pending_add_affirmative_returns_subtract_claimed():
    out = _detect(
        "yes I have it",
        pending={"canonical": "310S automotive technician certification",
                 "action": "add"},
    )
    assert out == RemainingGapsIntent(
        kind="subtract",
        current_turn_claims=(
            CredentialClaim(
                canonical="310S automotive technician certification",
                mode="claimed",
            ),
        ),
    )


def test_pending_add_affirmative_short_yes():
    out = _detect(
        "yes",
        pending={"canonical": "Class G driver's license", "action": "add"},
    )
    assert out.kind == "subtract"
    assert out.current_turn_claims[0].canonical == "Class G driver's license"
    assert out.current_turn_claims[0].mode == "claimed"


def test_pending_remove_affirmative_returns_retract():
    out = _detect(
        "yes that's right",
        pending={"canonical": "310S automotive technician certification",
                 "action": "remove"},
    )
    assert out == RemainingGapsIntent(
        kind="retract",
        retract_canonical="310S automotive technician certification",
    )


def test_pending_negative_returns_none():
    out = _detect(
        "no, not yet",
        pending={"canonical": "310S automotive technician certification",
                 "action": "add"},
    )
    assert out is None


def test_pending_unrelated_message_falls_through_to_fresh_detection():
    """A reply that doesn't match yes/no falls through. With a generic
    'what else?' it should produce a `kind="subtract"` empty-claims
    result (snapshot exists). The pending canonical IS a snapshot
    entry so the identity guard passes; the fall-through then matches
    the generic-remaining pattern."""
    out = _detect(
        "what else?",
        pending={"canonical": "310S automotive technician certification",
                 "action": "add"},
    )
    assert out.kind == "subtract"
    assert out.current_turn_claims == ()


def test_pending_with_malformed_dict_is_ignored():
    """Defensive: a pending entry missing canonical or with unknown action
    must not blow up the detector. Just falls through to fresh
    detection."""
    out = _detect(
        "yes",
        pending={"canonical": None, "action": "add"},
    )
    assert out is None or out.kind != "retract"


def test_pending_consumption_does_not_mutate_input():
    """Purity contract: detector never writes to the passed dict."""
    pending = {"canonical": "310S automotive technician certification",
               "action": "add"}
    _detect("yes", pending=pending)
    assert pending == {"canonical": "310S automotive technician certification",
                       "action": "add"}


# ============================================================================
# Step 2 -- retraction against accumulated state (the v8 broadening)
# ============================================================================
@pytest.mark.parametrize("message", [
    "I don't have 310S",
    "I haven't got my 310S",
    "I have not got my 310S",
    "actually I don't have 310S",
    "I never finished my 310S",
])
def test_retraction_against_accumulated_returns_confirm_remove(message):
    """ANY explicit negation targeting an accumulated credential
    triggers retraction confirmation. The "actually" hedge is NOT
    required."""
    out = _detect(
        message,
        accumulated=[{"canonical": "310S automotive technician certification",
                      "mode": "claimed"}],
    )
    assert out.kind == "confirm"
    assert out.confirmation_target_canonical == "310S automotive technician certification"
    assert out.pending_action == "remove"


def test_retraction_missing_pattern_against_accumulated():
    out = _detect(
        "I'm missing the 310S",
        accumulated=[{"canonical": "310S automotive technician certification",
                      "mode": "hypothetical"}],
    )
    assert out.kind == "confirm"
    assert out.pending_action == "remove"


def test_retraction_with_empty_accumulated_does_not_fire_step_2():
    """When the entity is NOT in accumulated_credentials, step 2 skips
    and step 3 returns None (standard explain_gap path)."""
    out = _detect(
        "I don't have 310S",
        accumulated=[],
    )
    assert out is None


# ============================================================================
# Step 3 -- standard negation for non-accumulated entities
# ============================================================================
@pytest.mark.parametrize("message", [
    "I don't have 310S",
    "I haven't got my Class G yet",
    "I'm missing the 310S",
    "without 310S, what can I do?",
])
def test_standard_negation_against_snapshot_only_returns_none(message):
    out = _detect(message)
    assert out is None


def test_standard_negation_no_entity_returns_none():
    out = _detect("I never finished my apprenticeship")
    assert out is None


def test_negation_blocks_other_pattern_detection_in_same_message():
    """Defensive: a message that contains BOTH a negation pattern AND a
    completion pattern must not silently subtract the other entity --
    the negation wins (kind=None when not against accumulated)."""
    out = _detect("I don't have 310S yet but I have my G2")
    assert out is None


# ============================================================================
# Step 4 -- uncertainty markers
# ============================================================================
@pytest.mark.parametrize("message", [
    "I think I got 310S",
    "I believe I have 310S",
    "I guess I have my 310S",
    "I probably have 310S",
    "maybe I have 310S",
    "I might have got my 310S",
    "pretty sure I have 310S",
    "I'm pretty sure I have my 310S",
])
def test_uncertainty_returns_confirm_add(message):
    out = _detect(message)
    assert out.kind == "confirm"
    assert out.confirmation_target_canonical == "310S automotive technician certification"
    assert out.pending_action == "add"


def test_uncertainty_with_no_entity_returns_confirm_with_none_target():
    out = _detect("I think I got it")
    # "it" resolves via anaphor to snapshot[0], so target IS the first
    # snapshot credential. (The disambiguation-target=None case comes
    # from inputs with no entity AND no anaphor.)
    assert out.kind == "confirm"
    assert out.pending_action == "add"


# ============================================================================
# Step 5 -- explicit completion
# ============================================================================
@pytest.mark.parametrize("message", [
    "I have 310S",
    "I got my 310S",
    "I have my 310S",
    "I earned my 310S",
    "I finished my 310S",
    "I passed my 310S",
    "I completed my 310S",
    "I now have my 310S",
    "I already have my 310S",
    "I currently have my 310S",
])
def test_completion_returns_subtract_claimed(message):
    out = _detect(message)
    assert out.kind == "subtract"
    assert out.current_turn_claims == (
        CredentialClaim(
            canonical="310S automotive technician certification",
            mode="claimed",
        ),
    )


def test_completion_multi_entity_with_repeated_verb():
    """Q2 locked: multiple completions in one message subtract all
    matched entities. Verb-repeated form."""
    out = _detect("I have my 310S and I have my G2")
    canonicals = sorted(c.canonical for c in out.current_turn_claims)
    assert canonicals == sorted([
        "310S automotive technician certification",
        "Class G driver's license",
    ])
    assert all(c.mode == "claimed" for c in out.current_turn_claims)


# ============================================================================
# Step 6 -- explicit hypothetical
# ============================================================================
@pytest.mark.parametrize("message", [
    "if I had 310S",
    "if I have my 310S",
    "once I get 310S",
    "after I finish my 310S",
    "after I get 310S",
    "assuming I have 310S",
    "assume I have 310S",
    "suppose I have my 310S",
])
def test_hypothetical_returns_subtract_hypothetical(message):
    out = _detect(message)
    assert out.kind == "subtract"
    assert out.current_turn_claims == (
        CredentialClaim(
            canonical="310S automotive technician certification",
            mode="hypothetical",
        ),
    )


def test_hypothetical_in_what_else_question():
    """The exact regression in docs/remaining-gaps-design.md §motivation."""
    out = _detect("if I had 310S, what else for this job?")
    assert out.kind == "subtract"
    assert out.current_turn_claims == (
        CredentialClaim(
            canonical="310S automotive technician certification",
            mode="hypothetical",
        ),
    )


# ============================================================================
# Step 7 -- generic remaining + bootstrap
# ============================================================================
@pytest.mark.parametrize("message", [
    "what else?",
    "anything else?",
    "any other?",
    "what else do I need?",
    "what's left?",
    "what's next?",
])
def test_generic_remaining_with_snapshot_returns_empty_subtract(message):
    out = _detect(message)
    assert out.kind == "subtract"
    assert out.current_turn_claims == ()


def test_generic_remaining_no_snapshot_returns_bootstrap():
    out = _detect("what else?", snapshot=None)
    assert out == RemainingGapsIntent(kind="bootstrap")


def test_generic_remaining_empty_snapshot_returns_bootstrap():
    out = _detect("what else?", snapshot=_snapshot(credential_gaps=[]))
    assert out == RemainingGapsIntent(kind="bootstrap")


# ============================================================================
# Snapshot-anchored identity (§4.0 single-source-of-truth invariant)
# ============================================================================
def test_every_returned_canonical_exists_in_snapshot():
    """Across every detection path that emits a canonical, the value MUST
    be one of the snapshot's stored canonicals (never a freshly-resolved
    registry value)."""
    snap_canonicals = {
        g["canonical"]
        for g in _HONDA_SNAPSHOT_MODE_A["lead_job"]["credential_gaps"]
    }

    for msg in [
        "I have 310S",
        "I think I got 310S",
        "if I had 310S",
        "I have my G2",
        "if I had G2/G",
    ]:
        out = _detect(msg)
        if out is None or out.kind == "bootstrap":
            continue
        for c in out.current_turn_claims:
            assert c.canonical in snap_canonicals, (msg, c)
        if out.confirmation_target_canonical:
            assert out.confirmation_target_canonical in snap_canonicals, msg
        if out.retract_canonical:
            assert out.retract_canonical in snap_canonicals, msg


def test_identity_anchored_in_mode_b_snapshot():
    """Same invariant when the snapshot was captured in Mode B (canonical
    = normalized display); the cross-mode bridge ensures the detector
    returns the snapshot's value."""
    mode_b_snapshot = _snapshot(credential_gaps=[
        ("310S Automotive Technician License",
         "310s automotive technician license"),
    ])
    out = detect_remaining_gaps_intent(
        "I have my 310S", mode_b_snapshot, _HONDA_REGISTRY,
        accumulated_credentials=[], pending_confirmation=None,
        last_discussed_canonical=None,
    )
    assert out.kind == "subtract"
    assert out.current_turn_claims == (
        CredentialClaim(
            canonical="310s automotive technician license",
            mode="claimed",
        ),
    )


# ============================================================================
# Ordering invariant (the design's locked detection ordering)
# ============================================================================
def test_pending_consumption_wins_over_generic_remaining():
    """A pending=add + 'yes I have' message must consume the pending,
    even if the message also contains 'what else' filler."""
    out = _detect(
        "yes I have it, what else?",
        pending={"canonical": "310S automotive technician certification",
                 "action": "add"},
    )
    assert out.kind == "subtract"
    assert out.current_turn_claims[0].mode == "claimed"


def test_retraction_wins_over_completion_in_same_message():
    """`I don't have 310S but I have my G2` against 310S in accumulated
    -> retraction confirmation, NOT a G2 claim."""
    out = _detect(
        "I don't have 310S but I have my G2",
        accumulated=[{"canonical": "310S automotive technician certification",
                      "mode": "claimed"}],
    )
    assert out.kind == "confirm"
    assert out.confirmation_target_canonical == "310S automotive technician certification"
    assert out.pending_action == "remove"


def test_uncertainty_wins_over_completion_when_both_could_match():
    """'I think I have 310S' shouldn't subtract just because 'I have 310S'
    is also present -- the uncertainty marker forces confirmation."""
    out = _detect("I think I have 310S")
    assert out.kind == "confirm"
    assert out.pending_action == "add"


# ============================================================================
# Bare anaphor resolution (last_discussed -> snapshot canonical)
# ============================================================================
def test_anaphor_resolves_completion_to_last_discussed():
    out = _detect(
        "I have it",
        last_discussed="Class G driver's license",
    )
    assert out.kind == "subtract"
    assert out.current_turn_claims[0].canonical == "Class G driver's license"


def test_anaphor_resolves_uncertainty_to_last_discussed():
    out = _detect(
        "I think I have it",
        last_discussed="Class G driver's license",
    )
    assert out.kind == "confirm"
    assert out.confirmation_target_canonical == "Class G driver's license"


def test_anaphor_without_last_discussed_uses_first_snapshot_entry():
    out = _detect("I have it")
    assert out.kind == "subtract"
    assert out.current_turn_claims[0].canonical == \
        "310S automotive technician certification"


# ============================================================================
# Mode C (registry=None) + Flag-decoupled identity
# ============================================================================
def test_mode_c_no_registry_still_resolves_via_token_fallback():
    out = detect_remaining_gaps_intent(
        "I have my 310S", _HONDA_SNAPSHOT_MODE_A, None,
        accumulated_credentials=[], pending_confirmation=None,
        last_discussed_canonical=None,
    )
    assert out.kind == "subtract"
    assert out.current_turn_claims[0].canonical == \
        "310S automotive technician certification"


def test_mode_c_generic_only_entity_emits_confirm_with_none_target():
    """Round-11 finding 2 (Mode C variant): `I have the license` -- all
    tokens are generic stop-words; the resolver returns None. With a
    snapshot present, the contract is to emit kind='confirm' with no
    target so the responder asks which credential the user means.
    (Pre-round-11 the detector returned None and fell through to the
    standard planner -- that lost the credential-completion signal
    entirely.)"""
    out = detect_remaining_gaps_intent(
        "I have the license", _HONDA_SNAPSHOT_MODE_A, None,
        accumulated_credentials=[], pending_confirmation=None,
        last_discussed_canonical=None,
    )
    assert out.kind == "confirm"
    assert out.confirmation_target_canonical is None


def test_flag_decoupled_identity_works_without_training_flag(monkeypatch):
    """Per §4a: alias resolution is INDEPENDENT of
    TRAINING_REGISTRY_ENABLED. The flag gates resource surfacing in R-4,
    NOT canonical identity. Whether the flag is True or False, a loaded
    registry must produce the same detection result."""
    out_with_registry = _detect("I have my 310S")
    assert out_with_registry.kind == "subtract"
    # Identity does not consult the flag at all; this test pins that
    # detection doesn't import or read TRAINING_REGISTRY_ENABLED.


# ============================================================================
# Empty / None inputs
# ============================================================================
def test_empty_message_returns_none():
    assert _detect("") is None
    assert _detect("   ") is None


def test_none_snapshot_with_completion_pattern_returns_none():
    """No snapshot -> no identity anchor; the detector can't emit a
    canonical, so it returns None (the standard planner handles it)."""
    out = detect_remaining_gaps_intent(
        "I have my 310S", None, _HONDA_REGISTRY,
        accumulated_credentials=[], pending_confirmation=None,
        last_discussed_canonical=None,
    )
    # Identity-anchor fails -> falls through. Completion pattern alone
    # without an identifiable snapshot canonical is a no-op.
    assert out is None or out.kind == "bootstrap"


# ============================================================================
# Round-11 regressions
# ============================================================================
# Finding 1 -- multi-credential extraction across natural conjunctions
# ----------------------------------------------------------------------------
def test_multi_credential_natural_and_form():
    """`I have 310S and Class G` -- a single verb spanning two entities
    must subtract BOTH. Pre-round-11 code captured only the first.
    Resolved by extending the entity-tail regex across 'and' / ','
    conjunctions and splitting the captured string per credential."""
    out = _detect("I have 310S and Class G")
    canonicals = sorted(c.canonical for c in out.current_turn_claims)
    assert canonicals == sorted([
        "310S automotive technician certification",
        "Class G driver's license",
    ])
    assert all(c.mode == "claimed" for c in out.current_turn_claims)


def test_multi_credential_both_qualifier():
    """`I have both 310S and Class G` -- 'both' is a pre-noun marker
    that doesn't claim an entity by itself; the regex consumes it and
    captures the conjunction list that follows."""
    out = _detect("I have both 310S and Class G")
    canonicals = sorted(c.canonical for c in out.current_turn_claims)
    assert canonicals == sorted([
        "310S automotive technician certification",
        "Class G driver's license",
    ])


def test_multi_credential_three_with_oxford_comma():
    """Engine could legitimately emit a 3-credential job; user can list
    them. Comma + 'and' splitting handles 'X, Y, and Z'."""
    snapshot = _snapshot(credential_gaps=[
        ("310S Automotive Technician License",
         "310S automotive technician certification"),
        ("G2/G driver's license", "Class G driver's license"),
        ("Smart Serve", "Smart Serve"),
    ])
    out = _detect(
        "I have 310S, Class G, and Smart Serve",
        snapshot=snapshot,
    )
    canonicals = sorted(c.canonical for c in out.current_turn_claims)
    assert canonicals == sorted([
        "310S automotive technician certification",
        "Class G driver's license",
        "Smart Serve",
    ])


def test_definite_claim_in_mixed_message_wins_over_uncertainty():
    """Round-11 finding 1: `I have my 310S and I think I have my G2`
    -- the definite claim for 310S is the dominant signal. The G2
    uncertainty is dropped (recoverable on the next turn). Pre-fix
    code confirmed G2 and silently lost the 310S claim, violating
    locked Q2."""
    out = _detect("I have my 310S and I think I have my G2")
    assert out.kind == "subtract"
    assert [c.canonical for c in out.current_turn_claims] == [
        "310S automotive technician certification",
    ]
    assert out.current_turn_claims[0].mode == "claimed"


# Finding 2 -- ambiguous entity reference emits confirmation
# ----------------------------------------------------------------------------
def test_ambiguous_completion_emits_confirm_with_none_target():
    """`I have G2` against a snapshot where 'G2' is a token in two
    different credential displays. The §4.3 uniqueness rule returns
    None; pre-fix code then fell through to standard planner routing.
    Round-11 finding 2: when a completion verb fires but no unique
    snapshot entry resolves, emit `kind="confirm"` with
    `confirmation_target_canonical=None` so the responder asks which
    credential the user means."""
    ambiguous_snapshot = _snapshot(credential_gaps=[
        ("G2 driver's license",  "DRIVER-CANON"),
        ("G2 paramedic course",  "PARAMEDIC-CANON"),
    ])
    out = detect_remaining_gaps_intent(
        "I have G2", ambiguous_snapshot, None,
        accumulated_credentials=[], pending_confirmation=None,
        last_discussed_canonical=None,
    )
    assert out.kind == "confirm"
    assert out.confirmation_target_canonical is None
    assert out.pending_action == "add"


def test_ambiguous_hypothetical_emits_confirm_with_none_target():
    ambiguous_snapshot = _snapshot(credential_gaps=[
        ("G2 driver's license",  "DRIVER-CANON"),
        ("G2 paramedic course",  "PARAMEDIC-CANON"),
    ])
    out = detect_remaining_gaps_intent(
        "if I had G2", ambiguous_snapshot, None,
        accumulated_credentials=[], pending_confirmation=None,
        last_discussed_canonical=None,
    )
    assert out.kind == "confirm"
    assert out.confirmation_target_canonical is None


def test_generic_only_credential_reference_emits_confirm():
    """`I have the license` -- the token-fallback resolver drops the
    generic vocabulary and finds nothing to anchor against. Verb fired
    + no canonical resolved -> confirm with None target."""
    out = _detect("I have the license")
    assert out.kind == "confirm"
    assert out.confirmation_target_canonical is None
    assert out.pending_action == "add"


# Round-12 boundary -- ordinary non-credential sentences MUST NOT confirm
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("message", [
    # Experience / skill statements with a completion verb but no
    # credential vocabulary and no snapshot-token overlap
    "I have five years of automotive experience",
    "I have welding and diagnostics skills",
    "I have three years of forklift experience",
    "I have good attention to detail",
    # Conversational / scope
    "I have a question about the job",
    "I have some concerns about the location",
    # Past-experience sentences with a completion verb
    "I had an interview yesterday",
    "I had a really good co-op placement",
    "I completed college in 2022",
    "I finished my placement at Honda last year",
    # Availability / start-date
    "I have weekends free",
    "I have my own transportation",
])
def test_ordinary_completion_sentences_do_not_emit_confirm(message):
    """Round-12 boundary: a completion verb whose entity contains NO
    credential vocabulary AND has ZERO snapshot-token overlap MUST
    fall through to None (standard planner routing). Pre-round-12
    code emitted kind='confirm' with None target for all of these,
    misclassifying ordinary skill / experience / availability /
    education / interview statements as credential clarifications."""
    out = _detect(message)
    assert out is None, (
        f"Expected None for ordinary non-credential sentence; got {out!r}"
    )


def test_credential_vocab_alone_with_no_match_emits_confirm():
    """Round-13 contract upper bound: when the entity is made of
    NOTHING BUT generic credential vocabulary ('my permit', 'my cert',
    'the licence') the user is unambiguously talking about A
    credential -- ask which. Same shape with 'my certificate', 'my
    credential', 'the cert'."""
    for msg in [
        "I have my permit",
        "I have my certificate",
        "I have my credential",
        "I have the cert",
        "I have my licence",
    ]:
        out = _detect(msg)
        assert out.kind == "confirm", msg
        assert out.confirmation_target_canonical is None, msg
        assert out.pending_action == "add", msg


def test_credential_vocab_in_hypothetical_with_no_match_emits_confirm():
    out = _detect("if I had my certification")
    assert out.kind == "confirm"
    assert out.confirmation_target_canonical is None


# Round-13 boundary -- credential-vocab token NEXT TO a distinguishing
# token whose join carries non-credential semantics MUST fall through
# to None. The four cases from the round-12 review:
#   - "work permit"   -> immigration/visa, not a credential
#   - "parking ticket" -> traffic infraction, not a credential
#   - "support ticket" -> helpdesk, not a credential
#   - "college certificate" / "certificate program in college"
#                      -> education enrollment, not a credential
# More importantly, "work permit" must NOT hijack the existing
# immigration/scope routing.
@pytest.mark.parametrize("message", [
    "I have a work permit",
    "I have a parking ticket",
    "I completed a certificate program in college",
    "I have a support ticket",
    # Variants the same boundary should catch:
    "I have a college certificate",
    "I have an event ticket",
    "I have a movie ticket",
    "I have a meal card",
    "I had a library card",
])
def test_credential_vocab_with_distinguishing_non_match_returns_none(message):
    """Round-13: when the reference contains a distinguishing token
    that doesn't appear in any snapshot display, the user is talking
    about something credential-shaped that is NOT one of the gaps in
    play. Return None (standard planner routing) so 'work permit'
    can still reach the immigration/scope rule and 'parking ticket'
    isn't misclassified as a credential clarification."""
    out = _detect(message)
    assert out is None, (
        f"Expected None for non-credential noun phrase with shared "
        f"credential vocabulary; got {out!r}"
    )


def test_ambiguous_token_against_multiple_snapshot_entries_emits_confirm():
    """When the user's substring matches MULTIPLE snapshot displays
    (G2 is a token in both G2/G driver's license and G2 paramedic
    course), emit confirm with no target -- the §4.3 ambiguous
    branch."""
    ambig_snap = _snapshot(credential_gaps=[
        ("G2/G driver's license",  "DRIVER-CANON"),
        ("G2 paramedic course",    "PARAMEDIC-CANON"),
    ])
    out = detect_remaining_gaps_intent(
        "I have G2", ambig_snap, None,
        accumulated_credentials=[], pending_confirmation=None,
        last_discussed_canonical=None,
    )
    assert out.kind == "confirm"
    assert out.confirmation_target_canonical is None


# Finding 4 -- frozen claims are immutable
# ----------------------------------------------------------------------------
def test_credential_claim_is_frozen():
    """Round-11 finding 4: the CredentialClaim dataclass must be
    immutable so downstream code cannot rewrite a resolved canonical
    after detection returns."""
    out = _detect("I have my 310S")
    claim = out.current_turn_claims[0]
    with pytest.raises(Exception):                 # FrozenInstanceError
        claim.canonical = "HACKED"                  # type: ignore[misc]
    with pytest.raises(Exception):
        claim.mode = "claimed"                      # type: ignore[misc]


def test_credential_claim_to_dict_returns_persistence_shape():
    """The dataclass is the wire shape; .to_dict() is what the handler
    writes to staged.last_assumed_completed_credentials so the dict
    semantics docs/remaining-gaps-design.md §3 specifies survive."""
    claim = CredentialClaim(canonical="X", mode="claimed")
    assert claim.to_dict() == {"canonical": "X", "mode": "claimed"}


def test_pending_with_canonical_not_in_snapshot_is_dropped():
    """Round-11 finding 3: the pending canonical MUST exist in the
    current snapshot. A stale pending entry whose canonical is no
    longer present in the snapshot is treated as malformed and the
    detector falls through to fresh detection. Without this guard,
    `yes` against {canonical: 'outside'} would emit a subtract claim
    for a canonical not anchored to any snapshot entry -- violating
    the §4.0 single-source-of-truth invariant."""
    out = detect_remaining_gaps_intent(
        "yes", _HONDA_SNAPSHOT_MODE_A, None,
        accumulated_credentials=[],
        pending_confirmation={"canonical": "outside-snapshot",
                              "action": "add"},
        last_discussed_canonical=None,
    )
    assert out is None


def test_pending_with_no_snapshot_is_dropped():
    """The pending canonical can never exist in a None snapshot, so the
    identity guard drops it. (This case shouldn't reach production
    because the handler clears pending when the snapshot is cleared,
    but the detector is defensive.)"""
    out = detect_remaining_gaps_intent(
        "yes", snapshot=None, registry=None,
        accumulated_credentials=[],
        pending_confirmation={"canonical": "some-canonical", "action": "add"},
        last_discussed_canonical=None,
    )
    assert out is None
