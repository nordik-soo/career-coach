"""AR-9.feat.coach-tiers CP2 step 4 — handler tier-evidence wiring.

Unit tests for `_build_tier_evidence_for_handler` and
`_tier_evidence_has_any_records` — the two helpers `_try_v2_path` uses
to gate the `present_tiered_matches` dispatch.

Pins:
  - Tier evidence is built BEFORE the handler sets
    `tiered_evidence_available` (the helper returns evidence; the
    caller decides the flag from it);
  - The helper degrades gracefully when adjacency is not enabled
    (cookie-mode, flag-off) — Sideways tier is empty, Strong/Stretch
    still surface;
  - The helper degrades gracefully when adjacency raises a DB error —
    Sideways empty, direct tiers still surface;
  - Non-MatchResult inputs (legacy test mocks that pass raw dicts)
    short-circuit to empty evidence so the legacy `present_matches`
    dispatch is preserved;
  - The `_FINAL_MOVE_TO_LEGACY_ACTION` dict maps
    `present_tiered_matches` → ACTION_PRESENT_MATCHES (signed-off pin
    from step 2; re-verified here);
  - The session-snapshot capture extension fires for
    `present_tiered_matches` (static audit on the source);
  - The adjacency soft-offer suppression is automatic because
    `_maybe_append_soft_offer` only triggers on `final_move ==
    "present_matches"` (NOT `present_tiered_matches`).
"""
from __future__ import annotations

import inspect

import pytest

from skillbridge.chat import handler, intake_state
from skillbridge.chat.handler import (
    _FINAL_MOVE_TO_LEGACY_ACTION,
    _build_tier_evidence_for_handler,
    _tier_evidence_has_any_records,
)
from skillbridge.chat.tiered_evidence import TieredEvidence
from skillbridge.session.staging import StagedProfile, StagedSkill

pytestmark = pytest.mark.nodb


# =========================================================================
# Fixtures
# =========================================================================
def _staged(*skills: str) -> StagedProfile:
    sp = StagedProfile.new("test-tier-wiring")
    sp.skills = [
        StagedSkill(skill_name=s, source="chat", confidence=0.8)
        for s in skills
    ]
    sp.target_role_text = "accountant"
    sp.target_noc = "13102"
    return sp


# =========================================================================
# _tier_evidence_has_any_records
# =========================================================================
def test_has_any_records_returns_false_for_empty_evidence():
    ev = TieredEvidence(apply_today=(), worth_a_try=(), sideways_move=())
    assert _tier_evidence_has_any_records(ev) is False


def test_has_any_records_returns_true_with_apply_today_records():
    ev = TieredEvidence(
        apply_today=("strong-stub",),  # type: ignore[arg-type]
        worth_a_try=(), sideways_move=(),
    )
    assert _tier_evidence_has_any_records(ev) is True


def test_has_any_records_returns_true_with_only_sideways():
    """Sideways-only is still a populated tier; the helper signals
    True so the arbiter re-dispatches to present_tiered_matches."""
    ev = TieredEvidence(
        apply_today=(), worth_a_try=(),
        sideways_move=("adj-stub",),  # type: ignore[arg-type]
    )
    assert _tier_evidence_has_any_records(ev) is True


# =========================================================================
# _build_tier_evidence_for_handler — defensive cases
# =========================================================================
def test_build_returns_empty_evidence_for_dict_results():
    """Legacy test fixtures (e.g. test_chat_transcripts) mock the
    engine to return raw dicts. The helper must return empty evidence
    in that case so the legacy present_matches dispatch is preserved."""
    fake_dict_results = [
        {"job_id": "j1", "title": "Warehouse Associate",
         "match_band": "strong", "match_eligible": True},
    ]
    sp = _staged("forklift")
    ev = _build_tier_evidence_for_handler(
        results=fake_dict_results,  # type: ignore[arg-type]
        training_by_job={},
        staged=sp,
    )
    assert ev.apply_today == ()
    assert ev.worth_a_try == ()
    assert ev.sideways_move == ()


def test_build_returns_empty_evidence_for_empty_results():
    sp = _staged("forklift")
    ev = _build_tier_evidence_for_handler(
        results=[], training_by_job={}, staged=sp,
    )
    assert isinstance(ev, TieredEvidence)
    assert ev.apply_today == ()
    assert ev.worth_a_try == ()
    assert ev.sideways_move == ()


def test_build_skips_adjacency_when_not_enabled(monkeypatch):
    """When `_adjacency_enabled()` returns False (cookie-mode or
    flag-off), the helper still runs build_tiered_evidence but with
    no accepted_adjacent — Sideways tier ends up empty."""
    from skillbridge.match import adjacent as adj_mod
    monkeypatch.setattr(adj_mod, "_adjacency_enabled", lambda: False)

    sp = _staged("forklift")
    ev = _build_tier_evidence_for_handler(
        results=[], training_by_job={}, staged=sp,
    )
    # Adjacent must be empty (no DB call happened).
    assert ev.sideways_move == ()


def test_build_degrades_when_adjacency_raises(monkeypatch):
    """When adjacency is enabled but the DB query raises, the helper
    must catch the exception and emit empty Sideways — the direct
    tiers still surface (here both empty because results=[])."""
    from skillbridge.match import adjacent as adj_mod
    monkeypatch.setattr(adj_mod, "_adjacency_enabled", lambda: True)

    def _explode(*a, **kw):
        raise RuntimeError("simulated DB failure")

    monkeypatch.setattr(adj_mod, "_load_active_jobs_with_skills", _explode)

    sp = _staged("forklift")
    ev = _build_tier_evidence_for_handler(
        results=[], training_by_job={}, staged=sp,
    )
    assert ev.sideways_move == ()


# =========================================================================
# Final-move → legacy action mapping (signed-off pin re-verified here)
# =========================================================================
def test_final_move_mapping_present_tiered_matches_maps_to_present_matches_action():
    assert (
        _FINAL_MOVE_TO_LEGACY_ACTION["present_tiered_matches"]
        == intake_state.ACTION_PRESENT_MATCHES
    )


def test_final_move_mapping_present_matches_unchanged():
    """Sanity: the legacy mapping for `present_matches` is unchanged."""
    assert (
        _FINAL_MOVE_TO_LEGACY_ACTION["present_matches"]
        == intake_state.ACTION_PRESENT_MATCHES
    )


# =========================================================================
# Source-level audits — the dispatch + snapshot-lifecycle wiring is
# in `_try_v2_path` (signed-off pin checks).
# =========================================================================
def _v2_src() -> str:
    return inspect.getsource(handler._try_v2_path)


def test_dispatch_calls_build_tier_evidence_before_setting_flag():
    """Signed-off pin: tier evidence is built BEFORE
    tiered_evidence_available is set. Static audit: in the source,
    the `_build_tier_evidence_for_handler` call precedes the
    `tiered_evidence_available=True,` keyword argument (the trailing
    comma narrows the match to the kwarg site; prose mentions in
    comments don't include the comma)."""
    src = _v2_src()
    build_idx = src.find("_build_tier_evidence_for_handler(")
    flag_idx = src.find("tiered_evidence_available=True,")
    assert build_idx != -1
    assert flag_idx != -1, (
        "kwarg site `tiered_evidence_available=True,` not found"
    )
    assert build_idx < flag_idx, (
        "tier evidence must be built before the flag is set"
    )


def test_dispatch_uses_in_memory_matches_not_results():
    """The tier builder consumes `list[MatchResult]`. The dispatch
    must pass `in_memory_matches`, not the dict-projected `results`."""
    src = _v2_src()
    # Find the _build_tier_evidence_for_handler call site and verify
    # it passes in_memory_matches.
    call_idx = src.find("_build_tier_evidence_for_handler(")
    call_block = src[call_idx:call_idx + 400]
    assert "results=in_memory_matches" in call_block


def test_dispatch_resolves_again_with_tiered_flag():
    """When tier evidence has records, dispatch re-calls
    resolve_match_outcome with tiered_evidence_available=True."""
    src = _v2_src()
    # The second resolve_match_outcome must carry the flag.
    flag_idx = src.find("tiered_evidence_available=True")
    assert flag_idx != -1
    # Look backward — `resolve_match_outcome` should appear before it.
    pre = src[max(0, flag_idx - 600):flag_idx]
    assert "resolve_match_outcome(" in pre


def test_snapshot_capture_extends_to_present_tiered_matches():
    """Signed-off pin: present_tiered_matches receives the same
    session-snapshot capture and last_adjacent_snapshot lifecycle as
    present_matches. CP2 step 6.1 refined the lifecycle: when the
    Sideways tier is populated, the snapshot is STAMPED with those
    records (for ordinal follow-ups); otherwise it is cleared. Both
    cases live inside the same branch."""
    src = _v2_src()
    capture_idx = src.find(
        'if final.final_move in {"present_matches", "present_tiered_matches"}'
    )
    assert capture_idx != -1, (
        "Snapshot-capture branch must include present_tiered_matches"
    )
    block = src[capture_idx:capture_idx + 2400]
    assert "_capture_presented_context" in block
    assert "_capture_match_snapshot" in block
    # CP2 step 6.1: the clear is now the `else` arm of a stamp/clear
    # conditional, not unconditional. Both sides of the conditional
    # must remain inside this branch — the stamp path covers the
    # Sideways-only case; the clear path covers Strong/Stretch-only.
    assert "_build_adjacent_snapshot_from_sideways" in block, (
        "Sideways stamp path missing — ordinal follow-ups on "
        "Sideways-only output would have no snapshot to bind against."
    )
    assert "staged.last_adjacent_snapshot = None" in block, (
        "Clear path missing — direct-match-only response must "
        "invalidate any prior adjacent snapshot."
    )


def test_soft_offer_not_triggered_for_tiered_matches():
    """The adjacency soft-offer logic in `_maybe_append_soft_offer`
    only triggers for `move == 'present_matches'` (or
    `present_no_match`). It must NOT trigger for the new move."""
    src = inspect.getsource(handler._maybe_append_soft_offer)
    # The function predicates `emit = True` only on the legacy moves.
    assert 'if move == "present_matches":' in src
    # And it does NOT have a special case for present_tiered_matches.
    assert "present_tiered_matches" not in src


# =========================================================================
# Step-4 review High — _build_v2_response forces empty cards for the
# tiered move. Locked design: ChatGPT-style prose-only, no cards.
# =========================================================================
def _v2_response_fixture(*, final_move: str) -> dict:
    """Drive _build_v2_response with the same shape the dispatch uses."""
    return handler._build_v2_response(
        staged=_staged("forklift"),
        new_session_id="sess-1",
        reply="prose body",
        final_move=final_move,
        ask_slot=None,
        resume_info=None,
        results=[
            {"job_id": "j1", "title": "Warehouse Associate",
             "employer": "Acme", "match_band": "strong",
             "matched_skills": ["forklift"], "missing_skills": []},
        ],
        training_by_job={"j1": [{"provider": "P", "title": "T",
                                  "url": "https://x.example/t"}]},
        next_skill=("WHMIS", 3),
    )


def test_tiered_response_forces_empty_recommended_jobs():
    """High finding: the tiered move's API response must NOT carry
    `recommended_jobs` records. Locked: prose-only, no cards."""
    resp = _v2_response_fixture(final_move="present_tiered_matches")
    assert resp["recommended_jobs"] == []


def test_tiered_response_suppresses_next_skill_fields():
    """Companion: the next-skill suggestion fields (used by the
    legacy card UI to surface unlock hints) are nulled on the
    tiered surface."""
    resp = _v2_response_fixture(final_move="present_tiered_matches")
    assert resp["next_skill_suggestion"] is None
    assert resp["next_skill_jobs_unlocked"] == 0


def test_legacy_present_matches_still_populates_recommended_jobs():
    """Regression guard: the legacy present_matches surface must
    keep its card payload unchanged."""
    resp = _v2_response_fixture(final_move="present_matches")
    assert resp["recommended_jobs"] != []
    assert resp["recommended_jobs"][0]["job_id"] == "j1"


def test_legacy_present_matches_still_carries_next_skill():
    """Regression guard: legacy path still surfaces the next-skill
    suggestion when the unlock count is positive."""
    resp = _v2_response_fixture(final_move="present_matches")
    assert resp["next_skill_suggestion"] == "WHMIS"
    assert resp["next_skill_jobs_unlocked"] == 3


def test_tiered_response_carries_final_move_field():
    """Step-4 / step-2 contract preserved: the v2 response carries
    `final_move` so analytics/transcript-tests can identify the new
    surface."""
    resp = _v2_response_fixture(final_move="present_tiered_matches")
    assert resp["final_move"] == "present_tiered_matches"
    assert resp["next_action"] == intake_state.ACTION_PRESENT_MATCHES


# =========================================================================
# Step-4 review Medium — _build_tier_evidence_for_handler gates the
# board load on `has_usable_skill_evidence(staged)`. The DB query is
# skipped when adjacency cannot produce results regardless.
# =========================================================================
def test_build_skips_board_load_when_evidence_floor_fails(monkeypatch):
    """`has_usable_skill_evidence(staged)` is the strict-AND gate's
    evidence floor. When it returns False, the strict gate would drop
    every candidate; loading the active-job board first is wasted DB
    cost. The helper must check the floor BEFORE loading the board."""
    from skillbridge.match import adjacent as adj_mod
    monkeypatch.setattr(adj_mod, "_adjacency_enabled", lambda: True)
    monkeypatch.setattr(
        adj_mod, "has_usable_skill_evidence", lambda staged: False,
    )

    load_calls = []

    def _spy_load(*a, **kw):
        load_calls.append("loaded")
        return []

    monkeypatch.setattr(adj_mod, "_load_active_jobs_with_skills", _spy_load)

    sp = _staged("forklift")
    _build_tier_evidence_for_handler(
        results=[], training_by_job={}, staged=sp,
    )
    assert load_calls == [], (
        "active-job board must not be loaded when the evidence floor fails"
    )


def test_build_loads_board_when_evidence_floor_passes(monkeypatch):
    """The negative control: when the floor passes, the board IS
    loaded."""
    from skillbridge.match import adjacent as adj_mod
    monkeypatch.setattr(adj_mod, "_adjacency_enabled", lambda: True)
    monkeypatch.setattr(
        adj_mod, "has_usable_skill_evidence", lambda staged: True,
    )

    load_calls = []

    def _spy_load(*a, **kw):
        load_calls.append("loaded")
        return []

    monkeypatch.setattr(adj_mod, "_load_active_jobs_with_skills", _spy_load)

    sp = _staged("forklift")
    _build_tier_evidence_for_handler(
        results=[], training_by_job={}, staged=sp,
    )
    assert load_calls == ["loaded"]


def test_build_skips_board_load_when_adjacency_disabled(monkeypatch):
    """When `_adjacency_enabled()` is False (cookie-mode / flag-off),
    the board is not loaded regardless of evidence floor."""
    from skillbridge.match import adjacent as adj_mod
    monkeypatch.setattr(adj_mod, "_adjacency_enabled", lambda: False)

    load_calls = []

    def _spy_load(*a, **kw):
        load_calls.append("loaded")
        return []

    monkeypatch.setattr(adj_mod, "_load_active_jobs_with_skills", _spy_load)

    sp = _staged("forklift")
    _build_tier_evidence_for_handler(
        results=[], training_by_job={}, staged=sp,
    )
    assert load_calls == []


def test_responder_v2_input_includes_tier_evidence():
    """The ResponderV2Input construction at the v2 dispatch site
    threads tier_evidence (and the placeholder for pipeline_snapshot)
    through to compose_response_v2."""
    src = _v2_src()
    # The final compose_response_v2 call must include tier_evidence
    # as a keyword.
    call_idx = src.rfind("compose_response_v2(ResponderV2Input(")
    assert call_idx != -1
    call_block = src[call_idx:call_idx + 1200]
    assert "tier_evidence=tier_evidence" in call_block


# =========================================================================
# CP2 step 6.1 — Sideways-only path + adjacent-snapshot retention.
# =========================================================================
def test_dispatch_has_sibling_branch_for_present_no_match():
    """CP2 step 6.1: `_try_v2_path` must build tier evidence on the
    `present_no_match` branch too, so a Sideways-only surface is
    reachable when the engine returns no Strong/Stretch but the user
    has transferable skills.

    Source audit asserts the elif clause exists and calls the same
    tier-evidence builder + re-resolves with the flag."""
    src = _v2_src()
    elif_idx = src.find('elif final.final_move == "present_no_match":')
    assert elif_idx != -1, (
        "missing elif branch for present_no_match — Sideways-only "
        "path would be unreachable from the handler side."
    )
    block = src[elif_idx:elif_idx + 1800]
    assert "_build_tier_evidence_for_handler(" in block, (
        "elif branch must call the tier-evidence builder"
    )
    assert "tiered_evidence_available=True" in block, (
        "elif branch must re-resolve the arbiter with the flag set"
    )
    assert "_tier_evidence_has_any_records(" in block, (
        "elif branch must guard the re-dispatch on actual records"
    )


def test_arbiter_check_precedes_no_match_fallback():
    """CP2 step 6.1 — arbiter ordering: `tiered_evidence_available`
    must be checked AHEAD of the bare `match_count == 0 → no_match`
    return. Otherwise the Sideways-only surface stays unreachable
    even when the handler supplies the flag."""
    from skillbridge.chat import arbiter as arb_mod
    src = inspect.getsource(arb_mod.resolve_match_outcome)
    flag_idx = src.find("if tiered_evidence_available:")
    no_match_idx = src.find('final_move="present_no_match"')
    assert flag_idx != -1, (
        "tiered_evidence_available branch not found in arbiter"
    )
    assert no_match_idx != -1
    assert flag_idx < no_match_idx, (
        "tiered_evidence_available must be checked BEFORE the bare "
        "match_count == 0 present_no_match return; otherwise the "
        "Sideways-only path is unreachable."
    )


def _fake_adjacent_record(
    *,
    job_id: str,
    title: str,
    transferable_pairs: tuple,
    important_gaps: tuple = (),
    why_adjacent: str = "skill_evidence",
):
    """Duck-typed AdjacentJob stand-in for the snapshot builder.
    The builder accesses .job_id, .title, .transferable_pairs,
    .important_gaps, .why_adjacent — using a SimpleNamespace avoids
    the full dataclass constructor's JobFacts/skill_alignment burden.
    """
    from types import SimpleNamespace
    return SimpleNamespace(
        job_id=job_id,
        title=title,
        transferable_pairs=transferable_pairs,
        important_gaps=important_gaps,
        why_adjacent=why_adjacent,
    )


def _fake_pair(user_skill: str, applies_to: str):
    from types import SimpleNamespace
    return SimpleNamespace(
        user_skill=user_skill, applies_to=applies_to, stage="exact",
    )


def test_build_adjacent_snapshot_from_sideways_shape():
    """The stamp helper produces the exact shape the ordinal
    resolver consumes: `{created_message_count, items}` with each
    item carrying job_id / title / evidence_summary / why_adjacent /
    matched_skills."""
    sp = _staged("forklift", "shipping")
    sp.message_count = 7

    adj1 = _fake_adjacent_record(
        job_id="job-a",
        title="Dispatch Coordinator",
        transferable_pairs=(
            _fake_pair("forklift", "material handling"),
            _fake_pair("shipping", "shipment routing"),
        ),
        important_gaps=("dispatch software",),
        why_adjacent="same_noc_minor_group",
    )
    adj2 = _fake_adjacent_record(
        job_id="job-b",
        title="Warehouse Lead",
        transferable_pairs=(_fake_pair("shipping", "loading bay coord"),),
        important_gaps=(),
        why_adjacent="skill_evidence",
    )

    snap = handler._build_adjacent_snapshot_from_sideways(sp, (adj1, adj2))

    assert snap["created_message_count"] == 7
    items = snap["items"]
    assert len(items) == 2

    a = items[0]
    assert a["job_id"] == "job-a"
    assert a["title"] == "Dispatch Coordinator"
    assert a["why_adjacent"] == "same_noc_minor_group"
    # 2 transferable_pairs, 1 important_gap → "2 of 3 required skills"
    assert a["evidence_summary"] == "2 of 3 required skills"
    assert a["matched_skills"] == ["material handling", "shipment routing"]

    b = items[1]
    assert b["job_id"] == "job-b"
    assert b["title"] == "Warehouse Lead"
    assert b["evidence_summary"] == "1 of 1 required skills"
    assert b["matched_skills"] == ["loading bay coord"]


def test_ordinal_followup_resolves_against_sideways_snapshot():
    """Regression test: a Sideways-only present_tiered_matches surface
    populates `last_adjacent_snapshot` so the next-turn ordinal
    resolver (`resolve_adjacent_followup`) can bind "tell me about
    the second one" to the second Sideways record.

    Without the CP2 step 6.1 stamp, the snapshot was unconditionally
    cleared and ordinal follow-up against the freshly surfaced
    Sideways tier had nothing to resolve against."""
    from skillbridge.chat.adjacent_followup import resolve_adjacent_followup

    sp = _staged("forklift", "shipping")
    sp.message_count = 4

    sideways = (
        _fake_adjacent_record(
            job_id="job-1", title="Operations Assistant",
            transferable_pairs=(_fake_pair("forklift", "warehouse ops"),),
        ),
        _fake_adjacent_record(
            job_id="job-2", title="Logistics Coordinator",
            transferable_pairs=(_fake_pair("shipping", "freight handling"),),
        ),
        _fake_adjacent_record(
            job_id="job-3", title="Dispatch Trainee",
            transferable_pairs=(_fake_pair("forklift", "yard movement"),),
        ),
    )
    sp.last_adjacent_snapshot = handler._build_adjacent_snapshot_from_sideways(
        sp, sideways,
    )

    # Next turn: message_count = 5. TTL is created + 1.
    item = resolve_adjacent_followup(
        "tell me about the second one",
        sp.last_adjacent_snapshot,
        current_message_count=5,
    )
    assert item is not None, (
        "ordinal resolver returned None — Sideways snapshot stamp "
        "did not round-trip to the resolver"
    )
    assert item["job_id"] == "job-2"
    assert item["title"] == "Logistics Coordinator"
    # And the resolver should ALSO carry the data the describe-payload
    # renderer reads (evidence_summary + matched_skills).
    assert item["evidence_summary"] == "1 of 1 required skills"
    assert item["matched_skills"] == ["freight handling"]


def test_snapshot_clears_when_sideways_empty_on_tiered_matches():
    """The else arm of the snapshot conditional: when present_tiered_matches
    fires but the Sideways tier is empty (Strong/Stretch only), the
    `last_adjacent_snapshot` clear is preserved — a direct-match-only
    response invalidates any standing adjacent recommendations.

    Verified via source audit on the conditional structure."""
    src = _v2_src()
    capture_idx = src.find(
        'if final.final_move in {"present_matches", "present_tiered_matches"}'
    )
    block = src[capture_idx:capture_idx + 2400]
    # The conditional must read sideways_move presence AND fall through
    # to the bare clear when absent.
    assert "tier_evidence.sideways_move" in block, (
        "stamp condition must guard on sideways_move presence"
    )
    assert "else:" in block, (
        "missing else arm — Strong/Stretch-only surfaces would "
        "leak a stale adjacent snapshot forward."
    )
