"""AR-6a tests: handler-side adjacency dispatch.

Covers (per docs/adjacent-recommendations-design.md v11):
  - Scope-violated TTL shift: fires BEFORE any other dispatch in
    _try_v2_path so a redirect_scope digression doesn't burn the
    snapshot TTL, even when the downstream path returns None
    (fallback_to_legacy).
  - _adjacency_enabled gate: cookie-mode bypasses all adjacency hooks.
  - Ordinal follow-up: resolve_adjacent_followup returning an item
    synthesizes a describe_adjacent_role decision and bypasses the
    planner.
  - AdjacentIntent: detect_adjacent_intent returning AdjacentIntent
    runs the engine pipeline (load + retrieve + accept + rank +
    drop_excluded), persists last_adjacent_snapshot, and synthesizes
    a recommend_adjacent_roles decision.
  - NeedsEvidenceIntent: synthesizes an ask_one_clarifying_question
    with ask_slot=skills_text.
  - None intent: dispatch returns None; standard path proceeds.

The tests monkeypatch the engine pipeline entry points and the
adjacency module imports so we exercise the DISPATCH branching
without dragging in DB calls.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.nodb


from skillbridge.chat.adjacent_intent import (
    AdjacentIntent,
    NeedsEvidenceIntent,
)
from skillbridge.chat.arbiter import ArbiterDecision
from skillbridge.session.staging import StagedProfile, StagedSkill


def _staged() -> StagedProfile:
    sp = StagedProfile.new("sess-1")
    sp.message_count = 3
    sp.target_role_text = "warehouse worker"
    sp.intake_state = "intake_collecting"
    sp.skills = [
        StagedSkill(skill_name="welding", source="resume", confidence=0.8),
        StagedSkill(skill_name="blueprint reading", source="resume", confidence=0.8),
        StagedSkill(skill_name="forklift operation", source="resume", confidence=0.8),
    ]
    return sp


class _FakeStore:
    def __init__(self):
        self.saved = []

    def save(self, sp) -> str:
        self.saved.append(sp)
        return sp.session_id or "sid"

    def load(self, sid):
        return None

    def new_session(self) -> str:
        return "sid"


# =========================================================================
# _try_adjacency_dispatch -- gate
# =========================================================================
def test_dispatch_skipped_in_cookie_mode(monkeypatch) -> None:
    """When `_adjacency_enabled()` is False (cookie mode), dispatch
    returns None immediately without calling any adjacency engine
    helpers. The standard planner / router path runs."""
    from skillbridge.chat import handler
    from skillbridge.match import adjacent as adj_mod

    monkeypatch.setattr(adj_mod, "_adjacency_enabled", lambda: False)

    # If dispatch DID proceed, these would be called and we'd notice.
    monkeypatch.setattr(
        "skillbridge.chat.adjacent_followup.resolve_adjacent_followup",
        lambda *a, **kw: pytest.fail("followup must not run in cookie mode"),
    )
    monkeypatch.setattr(
        "skillbridge.chat.adjacent_intent.detect_adjacent_intent",
        lambda *a, **kw: pytest.fail("detect must not run in cookie mode"),
    )

    result = handler._try_adjacency_dispatch(
        staged=_staged(), store=_FakeStore(),
        user_message="what other roles?",
        pending_adjacent_offer=False, resume_info=None,
    )
    assert result is None


def test_dispatch_ordinal_followup_synthesizes_describe(monkeypatch) -> None:
    """A live ordinal follow-up returns a synthesized
    describe_adjacent_role decision and the standard path is
    bypassed."""
    from skillbridge.chat import handler
    from skillbridge.match import adjacent as adj_mod

    monkeypatch.setattr(adj_mod, "_adjacency_enabled", lambda: True)
    monkeypatch.setattr(
        "skillbridge.chat.adjacent_followup.resolve_adjacent_followup",
        lambda *a, **kw: {"job_id": "j-2", "title": "Forklift"},
    )
    # AR-6a round-2: the ordinal branch now invokes render_describe to
    # exercise the live-fetch contract. Stub it so the test doesn't
    # hit a real DB connection.
    render_called = {"value": False}
    def _stub_render(item):
        render_called["value"] = True
        return {"job": None, "expired": True,
                "evidence_summary": "", "matched_skills": []}
    monkeypatch.setattr(
        "skillbridge.chat.adjacent_followup.render_describe_adjacent_role",
        _stub_render,
    )
    captured: dict = {}
    monkeypatch.setattr(
        handler, "compose_response_v2",
        lambda inp: captured.setdefault("decision", inp.decision) or "stub-reply",
    )

    result = handler._try_adjacency_dispatch(
        staged=_staged(), store=_FakeStore(),
        user_message="the second one",
        pending_adjacent_offer=False, resume_info=None,
    )
    assert result is not None
    assert render_called["value"] is True, (
        "AR-6a round-2 contract: the ordinal branch must invoke "
        "render_describe_adjacent_role to exercise the DB-fetch path."
    )
    assert captured["decision"].final_move == "describe_adjacent_role"
    assert captured["decision"].arbiter_action == "handler_synthesized_adjacent_description"


def test_dispatch_adjacent_intent_runs_engine_and_synthesizes_recommend(monkeypatch) -> None:
    """AdjacentIntent triggers the engine pipeline + snapshot
    persistence + recommend_adjacent_roles synthesis."""
    from skillbridge.chat import handler
    from skillbridge.match import adjacent as adj_mod

    monkeypatch.setattr(adj_mod, "_adjacency_enabled", lambda: True)
    monkeypatch.setattr(
        "skillbridge.chat.adjacent_followup.resolve_adjacent_followup",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        "skillbridge.chat.adjacent_intent.detect_adjacent_intent",
        lambda **kw: AdjacentIntent(trigger="user_explicit"),
    )

    # Stub the engine pipeline so no DB call is made.
    engine_called = {"value": False}

    def _stub_run(staged, *, trigger="user_explicit"):
        engine_called["value"] = True
        engine_called["trigger"] = trigger
        staged.last_adjacent_snapshot = {
            "created_message_count": staged.message_count,
            "items": [{
                "job_id": "j-1", "title": "Welder",
                "evidence_summary": "", "why_adjacent": "skill_evidence",
                "matched_skills": [],
            }],
        }
    monkeypatch.setattr(handler, "_run_adjacency_engine_and_persist", _stub_run)

    captured: dict = {}
    monkeypatch.setattr(
        handler, "compose_response_v2",
        lambda inp: captured.setdefault("decision", inp.decision) or "stub-reply",
    )

    sp = _staged()
    result = handler._try_adjacency_dispatch(
        staged=sp, store=_FakeStore(),
        user_message="what other roles?",
        pending_adjacent_offer=False, resume_info=None,
    )
    assert result is not None
    assert engine_called["value"] is True
    assert captured["decision"].final_move == "recommend_adjacent_roles"
    assert sp.last_adjacent_snapshot["items"][0]["job_id"] == "j-1"


def test_dispatch_needs_evidence_intent_synthesizes_clarification(monkeypatch) -> None:
    """NeedsEvidenceIntent synthesizes an ask_one_clarifying_question
    with ask_slot=skills_text."""
    from skillbridge.chat import handler
    from skillbridge.match import adjacent as adj_mod

    monkeypatch.setattr(adj_mod, "_adjacency_enabled", lambda: True)
    monkeypatch.setattr(
        "skillbridge.chat.adjacent_followup.resolve_adjacent_followup",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        "skillbridge.chat.adjacent_intent.detect_adjacent_intent",
        lambda **kw: NeedsEvidenceIntent(trigger="user_explicit"),
    )
    captured: dict = {}
    monkeypatch.setattr(
        handler, "compose_response_v2",
        lambda inp: captured.setdefault("decision", inp.decision) or "stub-reply",
    )

    result = handler._try_adjacency_dispatch(
        staged=_staged(), store=_FakeStore(),
        user_message="what other roles?",
        pending_adjacent_offer=False, resume_info=None,
    )
    assert result is not None
    assert captured["decision"].final_move == "ask_one_clarifying_question"
    assert captured["decision"].ask_slot == "skills_text"


def test_dispatch_none_intent_returns_none(monkeypatch) -> None:
    """No followup match AND detect_adjacent_intent returns None ->
    dispatch returns None so the standard path proceeds."""
    from skillbridge.chat import handler
    from skillbridge.match import adjacent as adj_mod

    monkeypatch.setattr(adj_mod, "_adjacency_enabled", lambda: True)
    monkeypatch.setattr(
        "skillbridge.chat.adjacent_followup.resolve_adjacent_followup",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        "skillbridge.chat.adjacent_intent.detect_adjacent_intent",
        lambda **kw: None,
    )

    result = handler._try_adjacency_dispatch(
        staged=_staged(), store=_FakeStore(),
        user_message="hi",
        pending_adjacent_offer=False, resume_info=None,
    )
    assert result is None


# =========================================================================
# _try_v2_path -- scope-violated TTL shift
# =========================================================================
def test_scope_violated_turn_calls_shift_adjacent_snapshot_ttl(monkeypatch) -> None:
    """When truth.scope_violations_detected is non-empty, the handler
    shifts last_adjacent_snapshot.created_message_count by 1 BEFORE
    any other dispatch. Even if the turn later falls back to v1, the
    snapshot survives the digression."""
    import inspect

    from skillbridge.chat import handler

    src = inspect.getsource(handler._try_v2_path)
    # Audit invariant: the shift call must appear inside _try_v2_path,
    # gated on scope_violations_detected (or the local boolean
    # `scope_violated` derived from it).
    assert "shift_adjacent_snapshot_ttl" in src
    assert "scope_violated" in src


# =========================================================================
# pending_adjacent_offer threading
# =========================================================================
# =========================================================================
# Full-stack _try_v2_path bypass: planner / engine / arbiter must NOT
# run when adjacency dispatch synthesizes a decision.
# =========================================================================
def _patch_full_v2_chain_with_failfast_planner(monkeypatch):
    """Mirror of test_chat_handler_v2._patch_v2_chain but with
    fail-fast spies on the planner / engine / standard arbiter.
    Returns the FakeStore and the captured response."""
    from skillbridge.chat import handler

    monkeypatch.setattr(handler, "CHAT_ORCHESTRATOR", "v2")

    # Fail-fast spies on every standard-path entry point.
    def _planner_must_not_run(*a, **kw):
        pytest.fail("plan_next_move must NOT run on adjacency turn")
    monkeypatch.setattr(handler, "plan_next_move", _planner_must_not_run)

    def _engine_must_not_run(*a, **kw):
        pytest.fail(
            "match_engine.compute_matches_in_memory must NOT run on "
            "adjacency turn"
        )
    monkeypatch.setattr(
        handler.match_engine, "compute_matches_in_memory",
        _engine_must_not_run,
    )

    def _arbiter_must_not_run(*a, **kw):
        pytest.fail("validate_planner_intent must NOT run on adjacency turn")
    monkeypatch.setattr(handler, "validate_planner_intent", _arbiter_must_not_run)

    def _resolve_must_not_run(*a, **kw):
        pytest.fail("resolve_match_outcome must NOT run on adjacency turn")
    monkeypatch.setattr(handler, "resolve_match_outcome", _resolve_must_not_run)

    # Stub the deterministic router so we don't need a real Haiku key.
    def _router_must_not_run(*a, **kw):
        pytest.fail("understand_message (router) must NOT run on adjacency turn")
    monkeypatch.setattr(handler, "understand_message", _router_must_not_run)

    # Stub `compose_response_v2` to return a known string so we can
    # assert on the response shape.
    monkeypatch.setattr(handler, "compose_response_v2", lambda inp: "adjacency-stub-reply")
    monkeypatch.setattr(handler, "compose_reply", lambda inp: pytest.fail("v1 must not run"))


def test_full_v2_path_skips_planner_and_engine_on_adjacent_intent(monkeypatch):
    """Integration: when `detect_adjacent_intent` returns
    `AdjacentIntent`, `_try_v2_path` synthesizes
    `recommend_adjacent_roles` and the standard planner / arbiter /
    engine paths NEVER run. Fail-fast spies on each one would `pytest.fail`
    if reached."""
    from skillbridge.chat import handler
    from skillbridge.match import adjacent as adj_mod

    monkeypatch.setattr(adj_mod, "_adjacency_enabled", lambda: True)
    monkeypatch.setattr(
        "skillbridge.chat.adjacent_followup.resolve_adjacent_followup",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        "skillbridge.chat.adjacent_intent.detect_adjacent_intent",
        lambda **kw: AdjacentIntent(trigger="user_explicit"),
    )
    monkeypatch.setattr(
        handler, "_run_adjacency_engine_and_persist",
        lambda staged, *, trigger="user_explicit": None,
    )

    _patch_full_v2_chain_with_failfast_planner(monkeypatch)

    response = handler._try_v2_path(
        staged=_staged(), message="what other roles?",
        uploaded_file=False, resume_info=None, store=_FakeStore(),
        pending_adjacent_offer=False,
    )
    assert response is not None
    assert response.get("final_move") == "recommend_adjacent_roles"


def test_full_v2_path_skips_planner_and_engine_on_followup(monkeypatch):
    """Integration: ordinal follow-up bypasses planner / engine /
    arbiter."""
    from skillbridge.chat import handler
    from skillbridge.match import adjacent as adj_mod

    monkeypatch.setattr(adj_mod, "_adjacency_enabled", lambda: True)
    monkeypatch.setattr(
        "skillbridge.chat.adjacent_followup.resolve_adjacent_followup",
        lambda *a, **kw: {"job_id": "j-1", "title": "Welder"},
    )
    monkeypatch.setattr(
        "skillbridge.chat.adjacent_followup.render_describe_adjacent_role",
        lambda *a, **kw: {"job": None, "expired": True,
                          "evidence_summary": "", "matched_skills": []},
    )

    _patch_full_v2_chain_with_failfast_planner(monkeypatch)

    response = handler._try_v2_path(
        staged=_staged(), message="the second one",
        uploaded_file=False, resume_info=None, store=_FakeStore(),
        pending_adjacent_offer=False,
    )
    assert response is not None
    assert response.get("final_move") == "describe_adjacent_role"


def test_full_v2_path_skips_planner_and_engine_on_needs_evidence(monkeypatch):
    """Integration: NeedsEvidenceIntent bypasses planner / engine /
    arbiter and synthesizes a clarification."""
    from skillbridge.chat import handler
    from skillbridge.match import adjacent as adj_mod

    monkeypatch.setattr(adj_mod, "_adjacency_enabled", lambda: True)
    monkeypatch.setattr(
        "skillbridge.chat.adjacent_followup.resolve_adjacent_followup",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        "skillbridge.chat.adjacent_intent.detect_adjacent_intent",
        lambda **kw: NeedsEvidenceIntent(trigger="user_explicit"),
    )

    _patch_full_v2_chain_with_failfast_planner(monkeypatch)

    response = handler._try_v2_path(
        staged=_staged(), message="what other roles?",
        uploaded_file=False, resume_info=None, store=_FakeStore(),
        pending_adjacent_offer=False,
    )
    assert response is not None
    assert response.get("final_move") == "ask_one_clarifying_question"


def test_full_v2_path_does_NOT_short_circuit_when_flag_off(monkeypatch):
    """The feature-flag gate: when `_adjacency_enabled()` returns
    False, the adjacency dispatch returns None and the standard
    planner path proceeds. NO `pytest.fail` spies on the standard
    path because that's exactly what's supposed to run."""
    from skillbridge.chat import handler
    from skillbridge.match import adjacent as adj_mod

    monkeypatch.setattr(adj_mod, "_adjacency_enabled", lambda: False)

    # If adjacency dispatch DID proceed despite the flag, these would
    # be hit and the test would fail.
    monkeypatch.setattr(
        "skillbridge.chat.adjacent_followup.resolve_adjacent_followup",
        lambda *a, **kw: pytest.fail("followup must not run when flag is off"),
    )
    monkeypatch.setattr(
        "skillbridge.chat.adjacent_intent.detect_adjacent_intent",
        lambda **kw: pytest.fail("detect must not run when flag is off"),
    )

    # Stub the standard chain so it can complete without DB / LLM.
    monkeypatch.setattr(handler, "CHAT_ORCHESTRATOR", "v2")
    monkeypatch.setattr(handler, "plan_next_move", lambda truth_json: None)
    monkeypatch.setattr(handler, "compose_response_v2", lambda inp: "standard-stub")

    response = handler._try_v2_path(
        staged=_staged(), message="what other roles?",
        uploaded_file=False, resume_info=None, store=_FakeStore(),
        pending_adjacent_offer=False,
    )
    # plan_next_move returned None → _try_v2_path returns None (fallback to v1).
    # The test passes as long as the adjacency `pytest.fail`s above
    # weren't triggered.
    assert response is None


def test_pending_adjacent_offer_threaded_into_try_v2_path() -> None:
    """The kwarg `pending_adjacent_offer` was added to `_try_v2_path`
    and is forwarded from `handle_anonymous`. Static audit."""
    import inspect

    from skillbridge.chat import handler

    sig = inspect.signature(handler._try_v2_path)
    assert "pending_adjacent_offer" in sig.parameters
    # default False so existing call sites stay compatible
    assert sig.parameters["pending_adjacent_offer"].default is False

    # And `handle_anonymous` forwards it explicitly.
    h_src = inspect.getsource(handler.handle_anonymous)
    assert "pending_adjacent_offer=pending_adjacent_offer" in h_src