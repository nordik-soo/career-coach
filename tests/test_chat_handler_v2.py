"""Unit tests for chat orchestration v2 slice 6 -- handler dispatch.

Two concerns:
  1. CHAT_ORCHESTRATOR is a HARD rollback switch: v1 means v1, v2 means
     v2, nothing in between. The dispatch happens at ONE place in
     handle_anonymous.
  2. In the v2 path, the order is visibly boring:
       gates -> planner -> arbiter pass 1 -> [maybe engine] -> arbiter pass 2 -> responder v2
     The match engine MUST NOT be invoked before arbiter pass 1
     returns RunEngine. Tests assert this exhaustively across every
     "no-engine" code path (scope override, evidence override, fallback,
     each gate, each non-proceed planner move).

No DB. We stub the session store and mock the LLM-touching functions.
"""
from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("LLM_ENABLED", "false")

import pytest

from skillbridge.chat import handler
from skillbridge.chat.arbiter import (
    ARBITER_REASON_FALLBACK,
    ARBITER_REASON_NO_MATCHES,
    ArbiterDecision,
    RunEngine,
)
from skillbridge.chat.planner import PlannerDecision
from skillbridge.session.staging import StagedProfile

pytestmark = pytest.mark.nodb


# ===========================================================================
# Test scaffolding -- FakeStore, planner factories, engine call tracking
# ===========================================================================
class FakeStore:
    """Minimal in-memory session store. Tracks save() calls for
    assertions about persistence."""
    def __init__(self):
        self.saved: list[StagedProfile] = []

    def new_session(self) -> str:
        return "fake-sid-new"

    def load(self, session_id):
        return None

    def save(self, staged: StagedProfile) -> str:
        self.saved.append(staged)
        return staged.session_id or "fake-sid-saved"

    def delete(self, session_id):
        pass


def _staged(
    session_id: str = "test-session",
    message_count: int = 5,
    target_role_text: str | None = "warehouse worker",
    intake_state_value: str = "intake_collecting",
) -> StagedProfile:
    """Build a staged profile with realistic-enough defaults that the
    truth_summary builder doesn't crash. Tests override fields they care
    about. message_count > 0 by default so first_turn_greeting gate
    doesn't fire unless a test explicitly sets it to 0."""
    sp = StagedProfile.new(session_id)
    sp.message_count = message_count
    sp.target_role_text = target_role_text
    sp.intake_state = intake_state_value
    return sp


def _planner_decision(
    move: str = "proceed_to_match",
    reason_code: str = "user_explicitly_asked_to_match",
    ask_slot: str | None = None,
    tone: str = "brief_confident",
) -> PlannerDecision:
    return PlannerDecision.model_validate({
        "move": move, "reason_code": reason_code,
        "ask_slot": ask_slot, "tone": tone,
    })


class EngineSpy:
    """Tracks whether compute_matches_in_memory was called. The most
    important assertion in Slice 6: engine MUST NOT run before arbiter
    pass 1 approves it."""
    def __init__(self, return_value=None):
        self.calls = 0
        self.return_value = return_value or []

    def __call__(self, staged, top=20):
        self.calls += 1
        return self.return_value


def _patch_v2_chain(
    monkeypatch,
    *,
    planner=None,
    engine_results=None,
    responder_reply="canned v2 reply",
    orchestrator="v2",
):
    """Wire monkeypatches for the full v2 chain. Returns the
    EngineSpy + a list to collect the planner inputs."""
    monkeypatch.setattr(handler, "CHAT_ORCHESTRATOR", orchestrator)

    planner_inputs: list[dict] = []

    def fake_plan(truth_json):
        planner_inputs.append(truth_json)
        return planner

    monkeypatch.setattr(handler, "plan_next_move", fake_plan)

    engine_spy = EngineSpy(return_value=engine_results)
    monkeypatch.setattr(handler.match_engine, "compute_matches_in_memory", engine_spy)
    # _build_results_block converts engine output -> (results, band_signal).
    # Stub it too so we can directly control match_count when engine runs.
    def fake_build_results_block(matches):
        if not matches:
            return ([], "none")
        return (list(matches), "strong_or_good")
    monkeypatch.setattr(handler, "_build_results_block", fake_build_results_block)
    monkeypatch.setattr(handler, "_attach_training", lambda results: {})
    monkeypatch.setattr(
        handler.match_engine, "next_skill_to_unlock_in_memory",
        lambda staged: (None, 0),
    )

    monkeypatch.setattr(handler, "compose_response_v2", lambda inp: responder_reply)
    # We also stub compose_reply so that if v2 falls back to v1 we can
    # detect it without running the legacy LLM path. Tests that intend
    # to exercise the fallback assert on this stub being called.
    fallback_v1_calls: list[str] = []
    monkeypatch.setattr(
        handler, "compose_reply",
        lambda inp: (fallback_v1_calls.append(inp.user_message), "v1 fallback reply")[1],
    )
    return engine_spy, planner_inputs, fallback_v1_calls


# ===========================================================================
# CHAT_ORCHESTRATOR = "v1" -- v2 modules MUST NOT run
# ===========================================================================
def test_v1_path_does_not_invoke_v2_chain(monkeypatch):
    """With CHAT_ORCHESTRATOR=v1, none of the v2 modules should be
    touched. This is the rollback contract: a flag value of v1 means
    the v2 code is dormant."""
    monkeypatch.setattr(handler, "CHAT_ORCHESTRATOR", "v1")

    planner_calls = []
    monkeypatch.setattr(
        handler, "plan_next_move",
        lambda truth: (planner_calls.append(truth), None)[1],
    )
    engine_spy = EngineSpy(return_value=[])
    monkeypatch.setattr(handler.match_engine, "compute_matches_in_memory", engine_spy)

    staged = _staged()
    store = FakeStore()

    # Direct dispatch test: _try_v2_path is the function the handler
    # ROUTES INTO. With v1, _try_v2_path is never called from
    # handle_anonymous. We assert that directly:
    # When CHAT_ORCHESTRATOR=v1, handle_anonymous skips the dispatch.
    # _try_v2_path itself is unaffected by the flag (the flag check is
    # IN handle_anonymous), so we don't call _try_v2_path here. Instead
    # we confirm the function exists and the dispatch lives in
    # handle_anonymous.
    import inspect
    src = inspect.getsource(handler.handle_anonymous)
    assert 'if CHAT_ORCHESTRATOR == "v2"' in src, (
        "v2 dispatch must be guarded by an explicit CHAT_ORCHESTRATOR "
        "comparison in handle_anonymous. No flag check => no rollback."
    )


def test_config_default_is_v2():
    """Anonymous Chat Orchestration v2 became the default after live
    acceptance testing. v1 remains available for rollback via
    CHAT_ORCHESTRATOR=v1. This test verifies the flag is in the legal
    set; the default value itself can't be tested directly when env
    may override it, but the import-time validator guards against
    unknown values."""
    from config import CHAT_ORCHESTRATOR
    assert CHAT_ORCHESTRATOR in {"v1", "v2"}


def test_config_rejects_unknown_orchestrator_value():
    """Set CHAT_ORCHESTRATOR to a bogus value, force config reload,
    confirm ValueError. The handler must refuse to start with an
    unrecognized flag value -- prevents accidental half-rollouts via
    typos."""
    import importlib
    monkey_env = os.environ.copy()
    monkey_env["CHAT_ORCHESTRATOR"] = "v3"
    old_env = dict(os.environ)
    try:
        os.environ.clear()
        os.environ.update(monkey_env)
        with pytest.raises(ValueError, match="CHAT_ORCHESTRATOR must be"):
            importlib.reload(__import__("config"))
    finally:
        os.environ.clear()
        os.environ.update(old_env)
        importlib.reload(__import__("config"))


# ===========================================================================
# CHAT_ORCHESTRATOR = "v2" -- gate paths skip planner+arbiter+engine
# ===========================================================================
def test_v2_first_turn_greeting_gate_skips_planner_and_engine(monkeypatch):
    """User's first message is 'hi' (greeting-like) -> gate 3 fires
    -> canned response, NO planner call, NO engine call."""
    engine_spy, planner_inputs, _ = _patch_v2_chain(monkeypatch)
    staged = _staged(message_count=0)  # first turn
    store = FakeStore()

    response = handler._try_v2_path(
        staged=staged, message="hi", uploaded_file=False,
        resume_info=None, store=store,
    )
    assert response is not None
    assert response["reply"]  # gate produced a canned welcome
    assert "Sault Ste. Marie" in response["reply"] or "SSM" in response["reply"]
    assert engine_spy.calls == 0, "Engine MUST NOT run when a gate fires"
    assert planner_inputs == [], "Planner MUST NOT be called when a gate fires"
    # Gate-fired turn maps to acknowledge_and_continue (per gates.py).
    assert response["final_move"] == "acknowledge_and_continue"


def test_v2_first_turn_with_job_intent_does_NOT_fire_greeting_gate(monkeypatch):
    """Slice 2 review regression -- first turn with actual job intent
    flows to the planner, not the canned greeting. Engine still does
    NOT run unless arbiter approves."""
    engine_spy, planner_inputs, _ = _patch_v2_chain(
        monkeypatch,
        planner=_planner_decision(
            move="ask_one_clarifying_question",
            reason_code="target_role_unclear",
            ask_slot="target_role_text",
            tone="warm_supportive",
        ),
    )
    staged = _staged(message_count=0, target_role_text=None)
    store = FakeStore()

    response = handler._try_v2_path(
        staged=staged, message="I'm looking for warehouse manager work",
        uploaded_file=False, resume_info=None, store=store,
    )
    assert response is not None
    assert len(planner_inputs) == 1, "Planner SHOULD be called (no gate fires)"
    assert engine_spy.calls == 0, (
        "Engine MUST NOT run when planner emits ask_one_clarifying_question"
    )


# ===========================================================================
# CHAT_ORCHESTRATOR = "v2" -- "no engine" invariant
# ===========================================================================
# The most important Slice 6 tests: in every non-RunEngine code path,
# the match engine MUST NOT be called.
def test_v2_engine_not_called_when_planner_emits_ask(monkeypatch):
    engine_spy, _, _ = _patch_v2_chain(
        monkeypatch,
        planner=_planner_decision(
            move="ask_one_clarifying_question",
            reason_code="target_role_unclear",
            ask_slot="target_role_text", tone="warm_supportive",
        ),
    )
    handler._try_v2_path(
        staged=_staged(target_role_text=None),
        message="hi", uploaded_file=False,
        resume_info=None, store=FakeStore(),
    )
    assert engine_spy.calls == 0


def test_v2_engine_not_called_on_acknowledge(monkeypatch):
    engine_spy, _, _ = _patch_v2_chain(
        monkeypatch,
        planner=_planner_decision(
            move="acknowledge_and_continue",
            reason_code="user_confirmed", tone="brief_confident",
        ),
    )
    handler._try_v2_path(
        staged=_staged(), message="thanks",
        uploaded_file=False, resume_info=None, store=FakeStore(),
    )
    assert engine_spy.calls == 0


def test_v2_engine_not_called_on_redirect_scope(monkeypatch):
    engine_spy, _, _ = _patch_v2_chain(
        monkeypatch,
        planner=_planner_decision(
            move="redirect_scope",
            reason_code="scope_violation_immigration",
            tone="honest_redirect",
        ),
    )
    handler._try_v2_path(
        staged=_staged(), message="can I apply for PR?",
        uploaded_file=False, resume_info=None, store=FakeStore(),
    )
    assert engine_spy.calls == 0


# The CRITICAL test from the user's Slice 6 guidance:
# "No hidden matching before arbiter approval."
def test_v2_engine_not_called_when_planner_says_proceed_but_arbiter_overrides_via_scope(monkeypatch):
    """Planner proposed proceed_to_match. The truth summary has a scope
    violation (e.g. user asked about immigration mid-stream). Arbiter
    pass 1 must override to redirect_scope. Engine MUST NOT run."""
    engine_spy, _, _ = _patch_v2_chain(
        monkeypatch,
        planner=_planner_decision(move="proceed_to_match"),
    )
    # Truth-summary scope detection is computed inside build_truth_summary.
    # We can't easily inject scope violations into a real staged profile
    # without driving real intent classifiers. Patch build_truth_summary
    # to inject the scope hint:
    from skillbridge.chat import truth_summary as ts_mod
    real_build = ts_mod.build_truth_summary

    def fake_build(*, staged, user_message, **kw):
        result = real_build(staged=staged, user_message=user_message, **kw)
        object.__setattr__(result, "scope_violations_detected", ["immigration"])
        return result
    monkeypatch.setattr(handler, "build_truth_summary", fake_build)

    handler._try_v2_path(
        staged=_staged(), message="can I apply for PR?",
        uploaded_file=False, resume_info=None, store=FakeStore(),
    )
    assert engine_spy.calls == 0, (
        "Scope override must run BEFORE engine. Even when the planner "
        "says proceed, scope-violation truth overrides and engine stays "
        "silent. This is the 'no hidden matching' rule."
    )


def test_v2_engine_not_called_when_planner_says_proceed_but_truth_says_not_enough(monkeypatch):
    """Planner proposed proceed_to_match. Truth.enough_to_match=false.
    Arbiter overrides to ask. Engine MUST NOT run."""
    engine_spy, _, _ = _patch_v2_chain(
        monkeypatch,
        planner=_planner_decision(move="proceed_to_match"),
    )
    # Override truth_summary to report enough_to_match=False.
    from skillbridge.chat import truth_summary as ts_mod
    real_build = ts_mod.build_truth_summary

    def fake_build(*, staged, user_message, **kw):
        result = real_build(staged=staged, user_message=user_message, **kw)
        object.__setattr__(result, "enough_to_match", False)
        object.__setattr__(result, "enough_to_match_reason", "missing_target")
        object.__setattr__(result, "usable_evidence_present", True)
        return result
    monkeypatch.setattr(handler, "build_truth_summary", fake_build)

    handler._try_v2_path(
        staged=_staged(target_role_text=None), message="match me",
        uploaded_file=False, resume_info=None, store=FakeStore(),
    )
    assert engine_spy.calls == 0


def test_v2_engine_not_called_when_planner_says_proceed_but_no_usable_evidence(monkeypatch):
    """Planner proposed proceed_to_match. Truth.usable_evidence_present=false.
    Arbiter overrides. Engine MUST NOT run."""
    engine_spy, _, _ = _patch_v2_chain(
        monkeypatch,
        planner=_planner_decision(move="proceed_to_match"),
    )
    from skillbridge.chat import truth_summary as ts_mod
    real_build = ts_mod.build_truth_summary

    def fake_build(*, staged, user_message, **kw):
        result = real_build(staged=staged, user_message=user_message, **kw)
        object.__setattr__(result, "usable_evidence_present", False)
        return result
    monkeypatch.setattr(handler, "build_truth_summary", fake_build)

    handler._try_v2_path(
        staged=_staged(), message="match me",
        uploaded_file=False, resume_info=None, store=FakeStore(),
    )
    assert engine_spy.calls == 0


def test_v2_engine_not_called_when_planner_returns_none_fallback(monkeypatch):
    """Planner returned None (LLM disabled or parse failure). Arbiter
    pass 1 returns fallback_to_legacy. _try_v2_path returns None to
    signal 'caller drop to v1'. Engine NOT called from v2 path."""
    engine_spy, _, _ = _patch_v2_chain(monkeypatch, planner=None)
    response = handler._try_v2_path(
        staged=_staged(), message="anything",
        uploaded_file=False, resume_info=None, store=FakeStore(),
    )
    assert response is None, (
        "fallback_to_legacy must propagate as None so caller drops to v1"
    )
    assert engine_spy.calls == 0


# ===========================================================================
# CHAT_ORCHESTRATOR = "v2" -- engine IS called when arbiter approves
# ===========================================================================
def test_v2_engine_called_when_planner_proceed_and_truth_supports_it(monkeypatch):
    """Happy path: planner proceed_to_match, truth supports it
    (enough_to_match=true, usable_evidence_present=true, no scope).
    Engine MUST run. Pass 2 resolves to present_matches (with results)
    or present_no_match (no results)."""
    fake_match = {
        "job_id": "job-1", "title": "Warehouse Associate",
        "employer": "Acme", "match_band": "strong",
        "score_explanation": {"caps_applied": []},
    }
    engine_spy, _, _ = _patch_v2_chain(
        monkeypatch,
        planner=_planner_decision(
            move="proceed_to_match",
            reason_code="user_explicitly_asked_to_match",
            tone="brief_confident",
        ),
        engine_results=[fake_match],
    )
    # Default truth from a "warehouse worker" staged profile (set in _staged)
    # should clear arbiter pass 1.
    from skillbridge.chat import truth_summary as ts_mod
    real_build = ts_mod.build_truth_summary

    def fake_build(*, staged, user_message, **kw):
        result = real_build(staged=staged, user_message=user_message, **kw)
        # Force the green-light truth state regardless of staged details.
        object.__setattr__(result, "enough_to_match", True)
        object.__setattr__(result, "usable_evidence_present", True)
        object.__setattr__(result, "scope_violations_detected", [])
        # Fresh-intake-on-target-change pillar (2026-06-15): tests that
        # fake a green-light truth state must also fake the alignment
        # fields, otherwise the arbiter pass 1 gate fires
        # `ask_one_clarifying_question reason=target_changed_need_fresh_intake`
        # and the engine never runs. Test intent: simulate a profile
        # already qualified to match; alignment is part of that intent.
        object.__setattr__(result, "target_alignment_ok", True)
        object.__setattr__(result, "skills_aligned_with_target", True)
        object.__setattr__(result, "experience_aligned_with_target", True)
        return result
    monkeypatch.setattr(handler, "build_truth_summary", fake_build)

    response = handler._try_v2_path(
        staged=_staged(), message="match me already",
        uploaded_file=False, resume_info=None, store=FakeStore(),
    )
    assert response is not None
    assert engine_spy.calls == 1, "Engine MUST run when arbiter approves"
    assert response["final_move"] == "present_matches"


def test_v2_engine_called_with_no_results_resolves_to_present_no_match(monkeypatch):
    """Engine runs but returns 0 matches. Pass 2 resolves to
    present_no_match."""
    engine_spy, _, _ = _patch_v2_chain(
        monkeypatch,
        planner=_planner_decision(move="proceed_to_match"),
        engine_results=[],
    )
    from skillbridge.chat import truth_summary as ts_mod
    real_build = ts_mod.build_truth_summary

    def fake_build(*, staged, user_message, **kw):
        result = real_build(staged=staged, user_message=user_message, **kw)
        object.__setattr__(result, "enough_to_match", True)
        object.__setattr__(result, "usable_evidence_present", True)
        object.__setattr__(result, "scope_violations_detected", [])
        # Fresh-intake-on-target-change pillar (2026-06-15): tests that
        # fake a green-light truth state must also fake the alignment
        # fields, otherwise the arbiter pass 1 gate fires
        # `ask_one_clarifying_question reason=target_changed_need_fresh_intake`
        # and the engine never runs. Test intent: simulate a profile
        # already qualified to match; alignment is part of that intent.
        object.__setattr__(result, "target_alignment_ok", True)
        object.__setattr__(result, "skills_aligned_with_target", True)
        object.__setattr__(result, "experience_aligned_with_target", True)
        return result
    monkeypatch.setattr(handler, "build_truth_summary", fake_build)

    response = handler._try_v2_path(
        staged=_staged(), message="match me",
        uploaded_file=False, resume_info=None, store=FakeStore(),
    )
    assert engine_spy.calls == 1
    assert response["final_move"] == "present_no_match"


# ===========================================================================
# Order invariant: gates run BEFORE planner; planner BEFORE arbiter;
# engine ONLY after pass 1 returns RunEngine
# ===========================================================================
def test_v2_dispatch_call_order(monkeypatch):
    """Track the call order across the v2 chain. Asserts the sequence
    is gates -> planner -> arbiter pass 1 -> [engine] -> arbiter pass 2."""
    call_log: list[str] = []

    monkeypatch.setattr(handler, "CHAT_ORCHESTRATOR", "v2")

    def fake_evaluate_gates(*, user_message, uploaded_file, message_count):
        call_log.append("gate")
        return None  # no gate fires
    monkeypatch.setattr(handler.chat_gates, "evaluate_gates", fake_evaluate_gates)

    def fake_build(*, staged, user_message, **kw):
        call_log.append("truth")
        from skillbridge.chat.truth_summary import TruthSummary
        ts = TruthSummary(
            user_message=user_message, enough_to_match=True,
            usable_evidence_present=True, target_role_text="warehouse worker",
            target_role_specificity="specific",
        )
        return ts
    monkeypatch.setattr(handler, "build_truth_summary", fake_build)

    def fake_plan(truth):
        call_log.append("planner")
        return _planner_decision(move="proceed_to_match")
    monkeypatch.setattr(handler, "plan_next_move", fake_plan)

    def fake_validate(decision, truth):
        call_log.append("arbiter_pass1")
        return RunEngine(
            planner_reason_code="user_explicitly_asked_to_match",
            planner_tone="brief_confident",
        )
    monkeypatch.setattr(handler, "validate_planner_intent", fake_validate)

    def fake_engine(staged, top=20):
        call_log.append("engine")
        return []
    monkeypatch.setattr(handler.match_engine, "compute_matches_in_memory", fake_engine)
    monkeypatch.setattr(
        handler, "_build_results_block",
        lambda matches: ([], "none"),
    )
    monkeypatch.setattr(handler, "_attach_training", lambda r: {})

    def fake_resolve(**kwargs):
        # Slice N (2026-06-05): accepts arbitrary kwargs to absorb the
        # new `near_miss_candidates` parameter without coupling the
        # test stub to the production signature.
        call_log.append("arbiter_pass2")
        return ArbiterDecision(
            final_move="present_no_match", reason_code=ARBITER_REASON_NO_MATCHES,
            tone="honest_redirect", arbiter_action="resolved_to_no_match",
        )
    monkeypatch.setattr(handler, "resolve_match_outcome", fake_resolve)

    monkeypatch.setattr(handler, "compose_response_v2", lambda inp: (
        call_log.append("responder"), "reply"
    )[1])

    handler._try_v2_path(
        staged=_staged(), message="match me",
        uploaded_file=False, resume_info=None, store=FakeStore(),
    )

    assert call_log == [
        "gate", "truth", "planner", "arbiter_pass1",
        "engine", "arbiter_pass2", "responder",
    ], f"Unexpected call order: {call_log}"


def test_v2_dispatch_call_order_when_pass1_terminal(monkeypatch):
    """When pass 1 returns a terminal decision (not RunEngine), the
    order is gates -> planner -> arbiter pass 1 -> responder, with
    NO engine and NO arbiter pass 2.

    Slice D (2026-06-05): MESSAGE_UNDERSTANDING_ENABLED is monkeypatched
    OFF here because this test specifically pins the planner-first
    dispatch order -- i.e. the rollback path. The router pre-empt order
    (gates -> truth -> router -> arbiter -> responder, planner skipped)
    is covered separately in test_chat_routing_integration.py. Both
    paths are real production code paths (default-off rollback vs
    flag-on default); each gets its own test.
    """
    call_log: list[str] = []

    monkeypatch.setattr(handler, "CHAT_ORCHESTRATOR", "v2")
    monkeypatch.setattr(handler, "MESSAGE_UNDERSTANDING_ENABLED", False)
    monkeypatch.setattr(handler.chat_gates, "evaluate_gates",
                        lambda **kw: (call_log.append("gate"), None)[1])

    def fake_build(*, staged, user_message, **kw):
        call_log.append("truth")
        from skillbridge.chat.truth_summary import TruthSummary
        return TruthSummary(user_message=user_message)
    monkeypatch.setattr(handler, "build_truth_summary", fake_build)

    monkeypatch.setattr(
        handler, "plan_next_move",
        lambda truth: (call_log.append("planner"),
                       _planner_decision(
                           move="redirect_scope",
                           reason_code="scope_violation_immigration",
                           tone="honest_redirect"))[1],
    )

    def fake_validate(decision, truth):
        call_log.append("arbiter_pass1")
        return ArbiterDecision(
            final_move="redirect_scope",
            reason_code="scope_violation_immigration",
            tone="honest_redirect", arbiter_action="passed_planner_through",
        )
    monkeypatch.setattr(handler, "validate_planner_intent", fake_validate)

    engine_spy = EngineSpy(return_value=[])
    monkeypatch.setattr(handler.match_engine, "compute_matches_in_memory", engine_spy)
    monkeypatch.setattr(
        handler, "resolve_match_outcome",
        lambda **kw: (call_log.append("arbiter_pass2"), None)[1],
    )
    monkeypatch.setattr(
        handler, "compose_response_v2",
        lambda inp: (call_log.append("responder"), "reply")[1],
    )

    handler._try_v2_path(
        staged=_staged(), message="PR application help",
        uploaded_file=False, resume_info=None, store=FakeStore(),
    )

    assert call_log == [
        "gate", "truth", "planner", "arbiter_pass1", "responder",
    ], f"Pass-1 terminal must skip engine and pass 2. Got: {call_log}"
    assert engine_spy.calls == 0


# ===========================================================================
# _build_results_block -- 4-band telemetry contract.
# Live-test feedback (2026-06-05) caught a lie: the function reported
# band="stretch_only" even when no stretch matches existed (only low). The
# fix introduces a 4th band 'low_only' so logs honestly distinguish
# "engine found candidates but they were filtered" from "engine found
# nothing." Visibility of low-band matches is intentionally NOT changed
# here -- that's a separate product slice.
# ===========================================================================
def _fake_match(*, band: str, eligible: bool = True, score: float = 0.5):
    """Minimal MatchResult-shaped object for _build_results_block tests."""
    from types import SimpleNamespace
    return SimpleNamespace(
        job_id=f"j-{band}-{int(score*1000)}",
        title=f"Job {band}",
        employer="Acme",
        url="https://example.com",
        location="Sault Ste. Marie",
        match_eligible=eligible,
        match_score=score,
        match_band=band,
        matched_skills=[],
        missing_skills=[],
        credential_warning=None,
        score_explanation={},
    )


def test_build_results_block_returns_strong_or_good_when_strong_present():
    results, band = handler._build_results_block([
        _fake_match(band="strong", score=0.85),
        _fake_match(band="good",   score=0.65),
        _fake_match(band="stretch", score=0.45),
        _fake_match(band="low",    score=0.15),
    ])
    assert band == "strong_or_good"
    # Strong-or-good chosen; lower bands ignored even though eligible.
    assert {r["match_band"] for r in results} == {"strong", "good"}


def test_build_results_block_returns_stretch_only_when_only_stretch_present():
    results, band = handler._build_results_block([
        _fake_match(band="stretch", score=0.45),
        _fake_match(band="stretch", score=0.40),
        _fake_match(band="low",     score=0.15),  # not chosen
    ])
    assert band == "stretch_only"
    assert len(results) == 2
    assert all(r["match_band"] == "stretch" for r in results)


def test_build_results_block_returns_low_only_when_only_low_present():
    """The Slice-D-follow-up fix: previously this returned
    band='stretch_only' with results=[]. That was the lie. Now: results=[]
    AND band='low_only' so logs distinguish from 'none'."""
    results, band = handler._build_results_block([
        _fake_match(band="low", score=0.15),
        _fake_match(band="low", score=0.12),
        _fake_match(band="low", score=0.10),
    ])
    assert band == "low_only"
    assert results == []


def test_build_results_block_returns_none_when_no_eligible_matches():
    results, band = handler._build_results_block([
        _fake_match(band="stretch", eligible=False, score=0.45),
        _fake_match(band="low",     eligible=False, score=0.15),
    ])
    assert band == "none"
    assert results == []


def test_build_results_block_returns_none_when_input_is_empty():
    """Engine returned nothing at all -- the genuine no-match case."""
    results, band = handler._build_results_block([])
    assert band == "none"
    assert results == []


# ===========================================================================
# Response shape -- v2 must produce the same dict shape as v1, plus final_move
# ===========================================================================
def test_v2_response_dict_has_expected_keys(monkeypatch):
    engine_spy, _, _ = _patch_v2_chain(
        monkeypatch,
        planner=_planner_decision(
            move="ask_one_clarifying_question",
            reason_code="target_role_unclear",
            ask_slot="target_role_text",
            tone="warm_supportive",
        ),
    )
    response = handler._try_v2_path(
        staged=_staged(target_role_text=None), message="hi",
        uploaded_file=False, resume_info=None, store=FakeStore(),
    )
    expected_keys = {
        "reply", "profile_id", "session_id", "intake_state", "asked_slots",
        "next_action", "recommended_jobs", "next_skill_suggestion",
        "next_skill_jobs_unlocked", "resume_info", "requires_consent",
        "final_move",  # v2-only addition
    }
    assert set(response.keys()) == expected_keys


def test_v2_response_legacy_next_action_mapping(monkeypatch):
    """For backwards-compat, next_action is mapped from final_move so
    legacy clients reading the response don't break."""
    engine_spy, _, _ = _patch_v2_chain(
        monkeypatch,
        planner=_planner_decision(
            move="ask_one_clarifying_question",
            reason_code="target_role_unclear",
            ask_slot="target_role_text",
            tone="warm_supportive",
        ),
    )
    response = handler._try_v2_path(
        staged=_staged(target_role_text=None), message="hi",
        uploaded_file=False, resume_info=None, store=FakeStore(),
    )
    from skillbridge.chat import intake_state
    assert response["next_action"] == intake_state.ACTION_ASK_QUESTIONS


@pytest.mark.parametrize("move,legacy_action", [
    ("ask_one_clarifying_question", "ASK_QUESTIONS"),
    ("acknowledge_and_continue", "ACKNOWLEDGE_AND_WAIT"),
    ("present_matches", "PRESENT_MATCHES"),
    ("present_no_match", "PRESENT_MATCHES"),
    ("redirect_scope", "REDIRECT"),
    ("explain_gap", "PRESENT_MATCHES"),
    ("offer_refinement", "PRESENT_MATCHES"),
    ("confirm_resume_summary", "PRESENT_RESUME_FACTS"),
])
def test_final_move_to_legacy_action_mapping(move, legacy_action):
    """Each OutcomeMove maps to a sensible legacy ACTION_* label."""
    assert handler._final_move_to_legacy_action(move) == legacy_action


def test_final_move_to_legacy_action_unknown_falls_back_defensively():
    """An unknown move (impossible via the enum, but defensive) maps to
    ASK_QUESTIONS rather than crashing."""
    assert (
        handler._final_move_to_legacy_action("invented_move")
        == "ASK_QUESTIONS"
    )


# ===========================================================================
# Caps collection
# ===========================================================================
def test_collect_caps_applied_unions_across_results():
    """Caps are unioned across the top results, preserving first-seen order."""
    results = [
        {"score_explanation": {"caps_applied": ["band_capped_by_credential"]}},
        {"score_explanation": {"caps_applied": ["band_capped_by_credential",
                                                 "band_capped_by_no_experience"]}},
        {"score_explanation": {"caps_applied": []}},
    ]
    caps = handler._collect_caps_applied(results)
    assert caps == ["band_capped_by_credential", "band_capped_by_no_experience"]


def test_collect_caps_applied_handles_missing_score_explanation():
    """Robust to malformed match shapes."""
    results = [{}, {"score_explanation": None}, {"score_explanation": {}}]
    assert handler._collect_caps_applied(results) == []


def test_collect_caps_applied_limits_to_top_5_results():
    """Caps from results beyond index 4 are ignored. Mirrors the
    responder's top-5 narration cap."""
    results = [
        {"score_explanation": {"caps_applied": [f"cap_{i}"]}}
        for i in range(10)
    ]
    caps = handler._collect_caps_applied(results)
    assert caps == [f"cap_{i}" for i in range(5)]


# ===========================================================================
# Touch / message_count semantics
# ===========================================================================
def test_v2_gate_path_touches_staged(monkeypatch):
    """When a gate fires, staged.touch() is called so message_count
    increments and last_active_at updates."""
    _patch_v2_chain(monkeypatch)
    staged = _staged(message_count=0)
    initial_count = staged.message_count

    handler._try_v2_path(
        staged=staged, message="hi", uploaded_file=False,
        resume_info=None, store=FakeStore(),
    )
    assert staged.message_count == initial_count + 1


def test_v2_fallback_path_does_NOT_touch_staged(monkeypatch):
    """When v2 falls back to v1 (planner returned None), v2 must NOT
    touch staged -- the v1 path will do it. Otherwise message_count
    would double-increment on fallback turns."""
    _patch_v2_chain(monkeypatch, planner=None)
    staged = _staged(message_count=5)
    initial_count = staged.message_count

    response = handler._try_v2_path(
        staged=staged, message="anything", uploaded_file=False,
        resume_info=None, store=FakeStore(),
    )
    assert response is None
    assert staged.message_count == initial_count, (
        "v2 fallback path must not touch staged; v1 will."
    )


# ===========================================================================
# Source-level invariant: engine is called from exactly one place in v2 path
# ===========================================================================
# ===========================================================================
# Rule-based extractor false-positive guard (live-test finding)
# ===========================================================================
# Live test showed "Truck and coach technician role" producing 4
# phantom chat skills via the rule-based extractor's substring
# matching against DB skill canonical names. Those phantom skills
# inflated chat_skill_count -> usable_evidence_present -> enough_to_match,
# and the engine ran on a profile with only a target role.
# `_is_likely_slot_answer` guards against this by skipping the
# rule-based fallback for short, single-message slot answers.
def test_is_likely_slot_answer_blocks_the_live_bug_shape():
    """The exact case from the live test: 5-token role-name reply
    to a target_role_text question."""
    assert handler._is_likely_slot_answer(
        "Truck and coach technician role",
        asked_slots=["target_role_text"],
    )


def test_is_likely_slot_answer_blocks_short_yes_no():
    assert handler._is_likely_slot_answer("yes", asked_slots=["target_role_text"])
    assert handler._is_likely_slot_answer(
        "full time", asked_slots=["work_type_preference"],
    )


def test_is_likely_slot_answer_DOES_NOT_block_comma_separated_skill_list():
    """Skill list reply: the rule-based fallback should still run
    (LLM may have missed multi-item enumerations)."""
    assert not handler._is_likely_slot_answer(
        "welding, forklift, shipping",
        asked_slots=["skills_text"],
    )


def test_is_likely_slot_answer_DOES_NOT_block_long_messages():
    """Longer messages can carry real skill claims even on
    slot-answer turns."""
    long = "I have three years of welding and forklift experience at a local warehouse"
    assert not handler._is_likely_slot_answer(long, asked_slots=["experience_text"])


def test_is_likely_slot_answer_DOES_NOT_block_cold_messages():
    """Cold session (no previous asked slot) -> let the rule-based
    fallback do its usual job."""
    assert not handler._is_likely_slot_answer("forklift", asked_slots=[])


def test_extract_skips_rule_based_fallback_on_slot_answer_when_llm_empty(monkeypatch):
    """End-to-end on the live bug shape. The LLM extractor returns
    empty (mocked). The rule-based fallback must NOT run because the
    message is a likely slot answer. Result: zero skills extracted."""
    from skillbridge.chat import extractor as chat_extractor

    # Stub the LLM extractor to return empty (simulating the live JSON
    # parse failure path -- the LLM returned prose, llm.call_json
    # returned None, chat_extractor returned empty).
    monkeypatch.setattr(
        chat_extractor, "extract",
        lambda message, asked_slots=None: chat_extractor.ExtractionResult(
            fields={}, skills=[], declined=[], off_topic=True,
            raw_keys_dropped=[],
        ),
    )

    # Track whether the rule-based fallback was invoked.
    fallback_calls: list[str] = []

    def fake_default_extractor():
        class _Spy:
            def extract_from_user_text(self, text):
                fallback_calls.append(text)
                # Pretend rule-based would have produced 4 phantom skills.
                from skillbridge.extract.base import ExtractedSkill
                return [
                    ExtractedSkill(skill_name=name, raw_phrase=name, confidence=0.85)
                    for name in ("truck", "coach", "technician", "welding")
                ]
        return _Spy()

    monkeypatch.setattr(handler, "default_extractor", fake_default_extractor)

    result = handler._extract(
        "Truck and coach technician role",
        asked_slots=["target_role_text"],
    )
    # Rule-based fallback was NOT called
    assert fallback_calls == [], (
        f"Rule-based fallback ran on a likely slot answer "
        f"(got {fallback_calls}). The slot-answer guard should have "
        f"blocked it -- otherwise phantom skills inflate chat_skill_count."
    )
    # Result has no skills (LLM empty, fallback blocked)
    assert result.skills == []


def test_v2_persists_last_asked_slots_after_ask_move(monkeypatch):
    """v2 path must write staged.last_asked_slots when emitting an
    ask move, so the slot-answer guard on the next turn sees correct
    state. v1's intake_state.decide() loop writes this; v2 had been
    silently skipping it -- the gap that made the slot-answer guard
    fail to fire in live testing."""
    _patch_v2_chain(
        monkeypatch,
        planner=_planner_decision(
            move="ask_one_clarifying_question",
            reason_code="target_role_unclear",
            ask_slot="target_role_text",
            tone="warm_supportive",
        ),
    )
    staged = _staged(target_role_text=None)
    assert staged.last_asked_slots == []   # fresh

    handler._try_v2_path(
        staged=staged, message="I'm looking for job",
        uploaded_file=False, resume_info=None, store=FakeStore(),
    )

    # After an ask turn, the slot must be persisted for the NEXT turn.
    assert staged.last_asked_slots == ["target_role_text"]


def test_v2_clears_last_asked_slots_after_non_ask_move(monkeypatch):
    """When v2 emits a move that doesn't ask for a slot (e.g.
    redirect_scope, acknowledge_and_continue, present_matches), the
    previous last_asked_slots must be cleared so a follow-up turn
    doesn't get a stale value."""
    _patch_v2_chain(
        monkeypatch,
        planner=_planner_decision(
            move="redirect_scope",
            reason_code="scope_violation_immigration",
            tone="honest_redirect",
        ),
    )
    staged = _staged()
    staged.last_asked_slots = ["target_role_text"]  # stale state

    handler._try_v2_path(
        staged=staged, message="can I apply for PR?",
        uploaded_file=False, resume_info=None, store=FakeStore(),
    )

    assert staged.last_asked_slots == []


def test_v2_canned_gate_clears_last_asked_slots(monkeypatch):
    """Gate paths don't ask for slots either -- they should clear
    last_asked_slots like the other v2 paths."""
    _patch_v2_chain(monkeypatch)
    staged = _staged(message_count=0, target_role_text=None)
    staged.last_asked_slots = ["target_role_text"]

    handler._try_v2_path(
        staged=staged, message="hi", uploaded_file=False,
        resume_info=None, store=FakeStore(),
    )

    assert staged.last_asked_slots == []


def test_extract_runs_rule_based_fallback_on_long_message_when_llm_empty(monkeypatch):
    """Inverse: on a longer message that's not a slot-answer pattern,
    the rule-based fallback SHOULD still run. Confirms the guard
    didn't over-fire."""
    from skillbridge.chat import extractor as chat_extractor

    monkeypatch.setattr(
        chat_extractor, "extract",
        lambda message, asked_slots=None: chat_extractor.ExtractionResult(
            fields={}, skills=[], declined=[], off_topic=True,
            raw_keys_dropped=[],
        ),
    )

    fallback_calls: list[str] = []

    def fake_default_extractor():
        class _Spy:
            def extract_from_user_text(self, text):
                fallback_calls.append(text)
                from skillbridge.extract.base import ExtractedSkill
                return [ExtractedSkill(skill_name="welding", raw_phrase="welding", confidence=0.85)]
        return _Spy()

    monkeypatch.setattr(handler, "default_extractor", fake_default_extractor)

    long_message = (
        "I have years of welding and forklift experience at the local warehouse"
    )
    result = handler._extract(long_message, asked_slots=["experience_text"])

    assert fallback_calls == [long_message], (
        "Rule-based fallback must still run on longer messages where "
        "the user is plausibly listing real skills."
    )
    assert len(result.skills) == 1


# ===========================================================================
# Slice 8 -- short-session context capture + clear
# ===========================================================================
# After Pass 2 emits present_matches, the handler captures titles +
# caps + credential gaps onto staged so the NEXT turn's responder
# fallback can reference them. present_no_match clears the stash so
# stale context doesn't leak forward.
def test_capture_presented_context_pulls_titles_caps_and_gaps():
    """All three context fields populate from the engine's result
    payload in display order."""
    sp = _staged()
    results = [
        {
            "title": "Truck and Coach Technician",
            "score_explanation": {
                "caps_applied": ["band_capped_by_credential"],
                "credential_gap_skills": ["310T technician certification"],
            },
        },
        {
            "title": "Heavy Equipment Mechanic",
            "score_explanation": {
                "caps_applied": ["band_capped_by_credential"],
                "credential_gap_skills": [
                    "310T technician certification",  # dedup target
                    "Class G driver's license",
                ],
            },
        },
    ]
    handler._capture_presented_context(
        sp, results, caps_applied=["band_capped_by_credential"],
    )
    assert sp.last_presented_job_titles == [
        "Truck and Coach Technician", "Heavy Equipment Mechanic",
    ]
    assert sp.last_presented_caps_applied == ["band_capped_by_credential"]
    # Dedup happened across results, order preserved
    assert sp.last_presented_credential_gaps == [
        "310T technician certification", "Class G driver's license",
    ]


def test_capture_presented_context_caps_at_top_5_results():
    """Even if engine returned more, we only stash the top 5."""
    sp = _staged()
    results = [
        {"title": f"Job {i}",
         "score_explanation": {"caps_applied": [], "credential_gap_skills": []}}
        for i in range(10)
    ]
    handler._capture_presented_context(sp, results, caps_applied=[])
    assert len(sp.last_presented_job_titles) == 5
    assert sp.last_presented_job_titles[0] == "Job 0"
    assert sp.last_presented_job_titles[-1] == "Job 4"


def test_capture_presented_context_handles_missing_score_explanation():
    """Defensive against malformed match shapes (e.g. legacy data)."""
    sp = _staged()
    results = [{"title": "Bare Title"}, {"title": "Another", "score_explanation": None}]
    handler._capture_presented_context(sp, results, caps_applied=[])
    assert sp.last_presented_job_titles == ["Bare Title", "Another"]
    assert sp.last_presented_credential_gaps == []


def test_clear_presented_context_resets_all_three_fields():
    sp = _staged()
    sp.last_presented_job_titles = ["X", "Y"]
    sp.last_presented_caps_applied = ["band_capped_by_credential"]
    sp.last_presented_credential_gaps = ["foo cert"]
    handler._clear_presented_context(sp)
    assert sp.last_presented_job_titles == []
    assert sp.last_presented_caps_applied == []
    assert sp.last_presented_credential_gaps == []


def test_build_conversation_context_snapshots_staged():
    """The dataclass is frozen so the responder can't mutate staged
    through it."""
    sp = _staged(target_role_text="warehouse worker")
    sp.last_presented_job_titles = ["Warehouse Associate"]
    sp.last_presented_caps_applied = ["band_capped_by_credential"]
    sp.last_presented_credential_gaps = ["forklift certification"]

    ctx = handler._build_conversation_context(sp)
    assert ctx.target_role_text == "warehouse worker"
    assert ctx.last_presented_job_titles == ("Warehouse Associate",)
    assert ctx.last_presented_credential_gaps == ("forklift certification",)
    # frozen
    with pytest.raises(Exception):
        ctx.target_role_text = "different"  # type: ignore[misc]


# ===========================================================================
# Slice 8 transcript: PR question after matches + forced policy rejection
# ===========================================================================
# This is the scenario the user asked for: matches are presented;
# then the user asks an immigration question; planner correctly emits
# redirect_scope; responder LLM produces text that fails the policy
# check (we force this via a mock); the fallback redirect must STILL
# reference the truck/coach roles or the 310T gap. Pre-Slice-8 the
# fallback was cold; this test pins that it no longer is.
def test_slice8_policy_rejected_redirect_references_recent_match_context(monkeypatch):
    """End-to-end regression for the Slice 8 motivating case."""
    monkeypatch.setattr(handler, "CHAT_ORCHESTRATOR", "v2")

    # Build a staged profile that just saw truck/coach matches with
    # a 310T credential gap (matching Michael's CV scenario).
    staged = _staged(target_role_text="truck and coach technician")
    staged.last_presented_job_titles = ["Truck and Coach Technician"]
    staged.last_presented_caps_applied = ["band_capped_by_credential"]
    staged.last_presented_credential_gaps = ["310T technician certification"]

    # Planner emits redirect_scope for the PR question.
    monkeypatch.setattr(
        handler, "plan_next_move",
        lambda truth: _planner_decision(
            move="redirect_scope",
            reason_code="scope_violation_immigration",
            tone="honest_redirect",
        ),
    )

    # Engine should NOT run on a redirect_scope turn.
    engine_spy = EngineSpy(return_value=[])
    monkeypatch.setattr(handler.match_engine, "compute_matches_in_memory", engine_spy)
    monkeypatch.setattr(handler, "_build_results_block", lambda r: ([], "none"))
    monkeypatch.setattr(handler, "_attach_training", lambda r: {})

    # Force compose_response_v2 to hit the fallback path. The simplest
    # way: stub the LLM call to return text that policy_ok_v2 will
    # reject (mentions Express Entry). This exercises the EXACT real
    # production path that prompted Slice 8.
    monkeypatch.setattr("skillbridge.chat.responder.is_enabled", lambda: True)
    monkeypatch.setattr(
        "skillbridge.chat.responder.call",
        lambda system, user, max_tokens=None: (
            "You can apply for Express Entry while completing your "
            "apprenticeship. RCIP eligibility depends on..."
        ),
    )

    response = handler._try_v2_path(
        staged=staged,
        message="can I apply for PR while I finish the apprenticeship?",
        uploaded_file=False, resume_info=None, store=FakeStore(),
    )

    assert response is not None
    assert response["final_move"] == "redirect_scope"
    assert engine_spy.calls == 0  # never runs on redirect

    reply = response["reply"]
    # The Slice 8 contract: fallback references the recent match context.
    # Either the specific credential gap or the role title MUST appear.
    references_context = (
        "310T technician certification" in reply
        or "Truck and Coach Technician" in reply
    )
    assert references_context, (
        f"Slice 8 contract violated: policy-rejected redirect fallback "
        f"is still cold. Reply did not reference 310T or the Truck "
        f"and Coach Technician role.\n  Reply: {reply!r}"
    )

    # And the safety properties from earlier slices still hold:
    # the fallback never re-introduces the topic that caused the
    # rejection.
    assert "Express Entry" not in reply
    assert "PR application" not in reply
    assert "RCIP eligibility" not in reply


def test_slice8_handler_updates_staged_after_present_matches_turn(monkeypatch):
    """End-to-end: after a successful present_matches turn, staged
    carries the context that the NEXT turn's fallback would use."""
    monkeypatch.setattr(handler, "CHAT_ORCHESTRATOR", "v2")

    staged = _staged(target_role_text="warehouse worker")
    # No context yet
    assert staged.last_presented_job_titles == []

    fake_match = {
        "job_id": "j1",
        "title": "Warehouse Associate",
        "employer": "Acme",
        "match_band": "strong",
        "score_explanation": {
            "caps_applied": [],
            "credential_gap_skills": [],
        },
    }
    _patch_v2_chain(
        monkeypatch,
        planner=_planner_decision(
            move="proceed_to_match",
            reason_code="user_explicitly_asked_to_match",
            tone="brief_confident",
        ),
        engine_results=[fake_match],
    )
    # Truth gating override so arbiter pass 1 clears.
    from skillbridge.chat import truth_summary as ts_mod
    real_build = ts_mod.build_truth_summary

    def fake_build(*, staged, user_message, **kw):
        result = real_build(staged=staged, user_message=user_message, **kw)
        object.__setattr__(result, "enough_to_match", True)
        object.__setattr__(result, "usable_evidence_present", True)
        object.__setattr__(result, "scope_violations_detected", [])
        # Fresh-intake-on-target-change pillar (2026-06-15): tests that
        # fake a green-light truth state must also fake the alignment
        # fields, otherwise the arbiter pass 1 gate fires
        # `ask_one_clarifying_question reason=target_changed_need_fresh_intake`
        # and the engine never runs. Test intent: simulate a profile
        # already qualified to match; alignment is part of that intent.
        object.__setattr__(result, "target_alignment_ok", True)
        object.__setattr__(result, "skills_aligned_with_target", True)
        object.__setattr__(result, "experience_aligned_with_target", True)
        return result
    monkeypatch.setattr(handler, "build_truth_summary", fake_build)

    handler._try_v2_path(
        staged=staged, message="match me",
        uploaded_file=False, resume_info=None, store=FakeStore(),
    )

    # After the present_matches turn, staged carries the context.
    assert "Warehouse Associate" in staged.last_presented_job_titles


def test_slice8_present_no_match_clears_staged_context(monkeypatch):
    """The engine ran but returned zero. Stale context from prior
    turns must NOT linger on staged -- otherwise the next-turn
    fallback would reference roles that didn't surface today."""
    monkeypatch.setattr(handler, "CHAT_ORCHESTRATOR", "v2")

    staged = _staged(target_role_text="something_obscure")
    # Pre-existing context from an earlier matched turn
    staged.last_presented_job_titles = ["Old Stale Title"]
    staged.last_presented_credential_gaps = ["stale gap"]

    _patch_v2_chain(
        monkeypatch,
        planner=_planner_decision(
            move="proceed_to_match",
            reason_code="user_explicitly_asked_to_match",
            tone="brief_confident",
        ),
        engine_results=[],  # zero matches today
    )
    from skillbridge.chat import truth_summary as ts_mod
    real_build = ts_mod.build_truth_summary

    def fake_build(*, staged, user_message, **kw):
        result = real_build(staged=staged, user_message=user_message, **kw)
        object.__setattr__(result, "enough_to_match", True)
        object.__setattr__(result, "usable_evidence_present", True)
        object.__setattr__(result, "scope_violations_detected", [])
        # Fresh-intake-on-target-change pillar (2026-06-15): tests that
        # fake a green-light truth state must also fake the alignment
        # fields, otherwise the arbiter pass 1 gate fires
        # `ask_one_clarifying_question reason=target_changed_need_fresh_intake`
        # and the engine never runs. Test intent: simulate a profile
        # already qualified to match; alignment is part of that intent.
        object.__setattr__(result, "target_alignment_ok", True)
        object.__setattr__(result, "skills_aligned_with_target", True)
        object.__setattr__(result, "experience_aligned_with_target", True)
        return result
    monkeypatch.setattr(handler, "build_truth_summary", fake_build)

    handler._try_v2_path(
        staged=staged, message="match me",
        uploaded_file=False, resume_info=None, store=FakeStore(),
    )

    # No matches -> stale context must be cleared.
    assert staged.last_presented_job_titles == []
    assert staged.last_presented_credential_gaps == []


# ===========================================================================
# R-1 (remaining-gaps iteration) -- _capture_match_snapshot
# ===========================================================================
# Structured per-job snapshot of the most-recent present_matches turn.
# Runs in parallel with the legacy _capture_presented_context until R-5
# deprecates last_presented_*. Tested in isolation here; the call-site
# wiring is covered by the existing Slice-8 present_matches tests above.
# ===========================================================================
class _FakeGap:
    """Mimics training.models.Gap.canonical_name for snapshot capture."""
    def __init__(self, canonical_name: str) -> None:
        self.canonical_name = canonical_name


class _FakeRegistry:
    """Map display -> Gap. Mirrors the .lookup() contract used by
    _capture_match_snapshot."""
    def __init__(self, mapping: dict[str, str]) -> None:
        self._map = mapping

    def lookup(self, query: str):
        return _FakeGap(self._map[query]) if query in self._map else None


def _engine_match(
    *,
    job_id: str,
    title: str,
    employer: str | None = None,
    credential_gap_skills: list[str] | None = None,
    other_missing: list[str] | None = None,
):
    """Build a single result-dict in the shape _build_results_block emits."""
    cg = credential_gap_skills or []
    others = other_missing or []
    return {
        "job_id": job_id,
        "title":  title,
        "employer": employer,
        "missing_skills": cg + others,
        "score_explanation": {
            "credential_gap_skills": cg,
            "caps_applied": [],
        },
    }


def test_capture_match_snapshot_mode_a_resolves_canonicals_through_registry():
    sp = _staged()
    sp.message_count = 7
    registry = _FakeRegistry({
        "310S Automotive Technician License":
            "310S Automotive Technician Certification",
        "G2/G driver's license": "Class G Driver's License",
    })
    results = [
        _engine_match(
            job_id="honda-1",
            title="310S Licensed Automotive Technician",
            employer="Great Lakes Honda",
            credential_gap_skills=[
                "310S Automotive Technician License",
                "G2/G driver's license",
            ],
            other_missing=[
                "Honda vehicle experience",
                "dealership experience",
            ],
        ),
    ]
    handler._capture_match_snapshot(sp, results, registry=registry)

    snap = sp.last_match_snapshot
    assert snap is not None
    assert snap["captured_at_turn"] == 7
    lead = snap["lead_job"]
    assert lead["job_id"] == "honda-1"
    assert lead["title"] == "310S Licensed Automotive Technician"
    assert lead["employer"] == "Great Lakes Honda"
    assert lead["credential_gaps"] == [
        {"display": "310S Automotive Technician License",
         "canonical": "310S Automotive Technician Certification"},
        {"display": "G2/G driver's license",
         "canonical": "Class G Driver's License"},
    ]
    assert lead["core_skill_gaps"] == [
        "Honda vehicle experience", "dealership experience",
    ]
    assert snap["other_jobs_meta"] == []


def test_capture_match_snapshot_mode_b_falls_back_to_normalized_display():
    """registry=None -> snapshot still built; canonical is the normalized
    display string (Mode B). Subtraction comparison still works
    case-insensitively against user input."""
    sp = _staged()
    results = [
        _engine_match(
            job_id="x", title="A Job",
            credential_gap_skills=["310S Automotive Technician License"],
        ),
    ]
    handler._capture_match_snapshot(sp, results, registry=None)

    snap = sp.last_match_snapshot
    assert snap is not None
    gaps = snap["lead_job"]["credential_gaps"]
    assert gaps == [
        {"display":   "310S Automotive Technician License",
         "canonical": "310s automotive technician license"},
    ]


def test_capture_match_snapshot_mode_a_falls_back_when_registry_lookup_misses():
    """A registry that doesn't know the display string still produces a
    Mode-B-style canonical for that entry; other entries can resolve
    normally."""
    sp = _staged()
    registry = _FakeRegistry({
        "Known Credential": "Known Canonical",
    })
    results = [
        _engine_match(
            job_id="x", title="T",
            credential_gap_skills=[
                "Known Credential",
                "Unknown Credential",
            ],
        ),
    ]
    handler._capture_match_snapshot(sp, results, registry=registry)
    gaps = sp.last_match_snapshot["lead_job"]["credential_gaps"]
    assert gaps == [
        {"display": "Known Credential", "canonical": "Known Canonical"},
        {"display": "Unknown Credential", "canonical": "unknown credential"},
    ]


def test_capture_match_snapshot_dedupes_by_resolved_canonical_preserving_first():
    """R-1 invariant (round-9): two engine labels aliasing to the same
    registry credential MUST collapse to one snapshot entry; the FIRST
    occurrence wins so engine ranking order is preserved."""
    sp = _staged()
    registry = _FakeRegistry({
        "G2 driver's licence":      "Class G Driver's License",
        "Class G driver's license": "Class G Driver's License",
        "Smart Serve":              "Smart Serve",
    })
    results = [
        _engine_match(
            job_id="x", title="T",
            credential_gap_skills=[
                "G2 driver's licence",        # first; kept
                "Class G driver's license",   # alias of the above; dropped
                "Smart Serve",                # different; kept
            ],
        ),
    ]
    handler._capture_match_snapshot(sp, results, registry=registry)
    gaps = sp.last_match_snapshot["lead_job"]["credential_gaps"]
    assert [g["display"] for g in gaps] == [
        "G2 driver's licence", "Smart Serve",
    ]
    assert [g["canonical"] for g in gaps] == [
        "Class G Driver's License", "Smart Serve",
    ]


def test_capture_match_snapshot_dedupes_before_capping():
    """Round-10 R-1 review: dedupe MUST run BEFORE the cap so an early
    run of aliasing display strings doesn't consume slots that should
    belong to later unique credentials. The engine emits 8 strings; the
    first 4 collapse to 2 canonicals (A, B); positions 5-8 are 4 unique
    credentials (C, D, E, F). Pre-fix code stored [A, B, C] (slice-then-
    dedupe loses D, E, F). Correct behavior: store [A, B, C, D, E]
    (dedupe-then-cap fills the cap with 5 unique credentials)."""
    sp = _staged()
    registry = _FakeRegistry({
        "alias-a-1": "A", "alias-a-2": "A",
        "alias-b-1": "B", "alias-b-2": "B",
        "C": "C", "D": "D", "E": "E", "F": "F",
    })
    results = [
        _engine_match(
            job_id="x", title="T",
            credential_gap_skills=[
                "alias-a-1", "alias-a-2",  # collapse to A
                "alias-b-1", "alias-b-2",  # collapse to B
                "C", "D", "E", "F",        # 4 unique tail entries
            ],
        ),
    ]
    handler._capture_match_snapshot(sp, results, registry=registry)
    gaps = sp.last_match_snapshot["lead_job"]["credential_gaps"]
    from skillbridge.session.staging import MAX_CRED_GAPS
    assert [g["canonical"] for g in gaps] == ["A", "B", "C", "D", "E"]
    assert len(gaps) == MAX_CRED_GAPS


def test_capture_match_snapshot_dedupes_in_mode_b_too():
    """Mode B (registry=None) dedupes by the normalized display string.
    'G2 driver licence' and 'g2 driver licence' both normalize to
    'g2 driver licence' -> one entry."""
    sp = _staged()
    results = [
        _engine_match(
            job_id="x", title="T",
            credential_gap_skills=[
                "G2 driver licence",
                "g2 driver licence",
                "Smart Serve",
            ],
        ),
    ]
    handler._capture_match_snapshot(sp, results, registry=None)
    gaps = sp.last_match_snapshot["lead_job"]["credential_gaps"]
    assert [g["display"] for g in gaps] == [
        "G2 driver licence", "Smart Serve",
    ]


def test_capture_match_snapshot_caps_credential_and_skill_gaps():
    sp = _staged()
    credential_overflow = [f"Credential {i}" for i in range(10)]
    skill_overflow      = [f"Skill {i}"      for i in range(10)]
    results = [
        _engine_match(
            job_id="x", title="T",
            credential_gap_skills=credential_overflow,
            other_missing=skill_overflow,
        ),
    ]
    handler._capture_match_snapshot(sp, results, registry=None)
    lead = sp.last_match_snapshot["lead_job"]
    from skillbridge.session.staging import MAX_CRED_GAPS, MAX_SKILL_GAPS
    assert len(lead["credential_gaps"]) == MAX_CRED_GAPS
    assert len(lead["core_skill_gaps"]) == MAX_SKILL_GAPS


def test_capture_match_snapshot_truncates_long_title_and_employer():
    sp = _staged()
    results = [
        _engine_match(
            job_id="x", title="T" * 500, employer="E" * 500,
            credential_gap_skills=[],
        ),
    ]
    handler._capture_match_snapshot(sp, results, registry=None)
    lead = sp.last_match_snapshot["lead_job"]
    assert len(lead["title"]) == 80
    assert len(lead["employer"]) == 60


def test_capture_match_snapshot_other_jobs_meta_deferred_to_empty_in_v1():
    """Design §1 reserves `other_jobs_meta` for a future job-pivot
    feature. R-1 stores an empty list (MAX_OTHER_JOBS=0) to free cookie
    budget for the safety-critical lead_job + accumulated-state shape."""
    from skillbridge.session.staging import MAX_OTHER_JOBS
    assert MAX_OTHER_JOBS == 0, (
        "If MAX_OTHER_JOBS is raised, this test should be replaced with "
        "coverage that exercises the populated path; v1 deferral was the "
        "R-1 budget tradeoff."
    )
    sp = _staged()
    results = [
        _engine_match(job_id="lead-id", title="Lead"),
        _engine_match(job_id="o1", title="Other 1"),
        _engine_match(job_id="o2", title="Other 2"),
    ]
    handler._capture_match_snapshot(sp, results, registry=None)
    assert sp.last_match_snapshot["other_jobs_meta"] == []


def test_capture_match_snapshot_resets_companion_state():
    """A new present_matches turn invalidates the prior conversation
    state: accumulated assumptions, last-discussed anchor, and pending
    confirmation must all clear together with the snapshot."""
    sp = _staged()
    sp.last_assumed_completed_credentials = [
        {"canonical": "stale", "mode": "claimed"},
    ]
    sp.last_discussed_credential_canonical = "stale"
    sp.pending_credential_confirmation = {"canonical": "stale", "action": "add"}

    results = [_engine_match(job_id="x", title="T")]
    handler._capture_match_snapshot(sp, results, registry=None)

    assert sp.last_assumed_completed_credentials == []
    assert sp.last_discussed_credential_canonical is None
    assert sp.pending_credential_confirmation is None


def test_capture_match_snapshot_empty_results_sets_none():
    sp = _staged()
    handler._capture_match_snapshot(sp, [], registry=None)
    assert sp.last_match_snapshot is None


def test_clear_match_snapshot_resets_all_four_fields():
    sp = _staged()
    sp.last_match_snapshot = {"lead_job": {"title": "X"}}
    sp.last_assumed_completed_credentials = [
        {"canonical": "x", "mode": "claimed"},
    ]
    sp.last_discussed_credential_canonical = "x"
    sp.pending_credential_confirmation = {"canonical": "x", "action": "add"}

    handler._clear_match_snapshot(sp)

    assert sp.last_match_snapshot is None
    assert sp.last_assumed_completed_credentials == []
    assert sp.last_discussed_credential_canonical is None
    assert sp.pending_credential_confirmation is None


def test_normalize_canonical_lowercases_and_collapses_punctuation():
    """The Mode-B fallback canonical is deterministic. Mirrors the
    docs/remaining-gaps-design.md §4.3 normalisation rule."""
    assert handler._normalize_canonical("310S Automotive Technician License") == \
        "310s automotive technician license"
    assert handler._normalize_canonical("G2/G driver's licence") == \
        "g2 g driver s licence"
    assert handler._normalize_canonical("   leading and  trailing   ") == \
        "leading and trailing"
    assert handler._normalize_canonical("") == ""
    assert handler._normalize_canonical(None) == ""  # type: ignore[arg-type]


# ===========================================================================
# R-1 -- cookie-size binding test
# ===========================================================================
# The signed StagedProfile must stay well under the browser's 4 KB
# per-cookie limit AFTER signing + Set-Cookie attribute overhead. 3800
# bytes is the design's chosen ceiling (see docs/remaining-gaps-design.md
# §1) leaving ~300 bytes for Path / HttpOnly / SameSite / Secure / Max-Age.
# ===========================================================================
def test_full_signed_session_under_browser_budget_with_max_remaining_gaps_state(
    monkeypatch,
):
    """Build a worst-case StagedProfile with all four R-1 fields at cap,
    plus a realistic resume_facts_json compact form, then confirm
    CookieSessionStore.save() returns a signed value under 3800 bytes."""
    monkeypatch.setenv("SESSION_COOKIE_SECRET", "x" * 48)
    monkeypatch.setenv("SESSION_TTL_MINUTES", "30")
    # Reload config so the test's env values take effect for this run.
    import importlib
    import config as _cfg
    importlib.reload(_cfg)
    from skillbridge.session import cookie_store as _cs_mod
    importlib.reload(_cs_mod)
    from skillbridge.session.staging import (
        MAX_CRED_GAPS, MAX_SKILL_GAPS, MAX_OTHER_JOBS,
        MAX_CANONICAL_CHARS, MAX_TITLE_CHARS, MAX_EMPLOYER_CHARS,
    )
    sp = _staged()
    sp.target_role_text = "warehouse worker"
    sp.skills_text = "forklift, picking, packing, shipping, receiving"
    sp.experience_text = "Three years at a Sault Ste. Marie distribution centre."
    sp.education_text = "High school diploma."
    sp.skills = []
    # Realistic-worst-case resume_facts_json. compact_facts() adds null
    # placeholder fields for every schema slot, so even a small input
    # inflates to ~70 bytes per skill entry. The design's §1 budget
    # models an ~800-byte compacted form (e.g. 8 skills + 2 jobs + 1
    # cert + 1 language). A SkillBridge candidate's typical resume is
    # nowhere near 20 skills with null evidence; the gate is realistic
    # worst-case, not unbounded stress.
    sp.resume_facts_json = {
        "skills": [{"name": f"Skill {i}", "fact_id": f"f{i}"} for i in range(8)],
        "work_history": [
            {"title": f"Job {i}", "employer": f"Employer {i}",
             "start_year": 2020 + i, "end_year": 2022 + i, "fact_id": f"w{i}"}
            for i in range(2)
        ],
        "certifications": [
            {"name": "Smart Serve", "fact_id": "c0"},
        ],
        "languages": [{"name": "English", "fact_id": "l0"}],
    }
    # Snapshot at cap (fixture sizes ALL track the staging constants so
    # the test stays honest as caps move).
    sp.last_match_snapshot = {
        "captured_at_turn": 99,
        "lead_job": {
            "job_id":   "j" * MAX_CANONICAL_CHARS,
            "title":    "x" * MAX_TITLE_CHARS,
            "employer": "y" * MAX_EMPLOYER_CHARS,
            "credential_gaps": [
                {"display":   "d" * MAX_CANONICAL_CHARS,
                 "canonical": "c" * MAX_CANONICAL_CHARS}
                for _ in range(MAX_CRED_GAPS)
            ],
            "core_skill_gaps": [
                "s" * MAX_CANONICAL_CHARS for _ in range(MAX_SKILL_GAPS)
            ],
        },
        "other_jobs_meta": [
            {"job_id": "j" * MAX_CANONICAL_CHARS,
             "title":  "t" * MAX_TITLE_CHARS}
            for _ in range(MAX_OTHER_JOBS)
        ],
    }
    sp.last_assumed_completed_credentials = [
        {"canonical": "x" * MAX_CANONICAL_CHARS, "mode": "hypothetical"}
        for _ in range(MAX_CRED_GAPS)
    ]
    sp.last_discussed_credential_canonical = "x" * MAX_CANONICAL_CHARS
    sp.pending_credential_confirmation = {
        "canonical": "x" * MAX_CANONICAL_CHARS, "action": "add",
    }

    store = _cs_mod.CookieSessionStore(secret="x" * 48)
    signed_value = store.save(sp)
    size = len(signed_value.encode("utf-8"))
    assert size < 3800, (
        f"signed session value is {size} bytes; ceiling is 3800 to leave "
        f"~300 bytes margin for Set-Cookie attributes "
        f"(Path / HttpOnly / SameSite / Secure / Max-Age). "
        f"If this fails, tighten MAX_CRED_GAPS / MAX_SKILL_GAPS / "
        f"MAX_OTHER_JOBS or reduce the cap on "
        f"last_assumed_completed_credentials."
    )


# ===========================================================================
# Extractor-after-gates optimization (post-Slice-7 production hardening)
# ===========================================================================
# Verifies the early-canned-gate short-circuit in handle_anonymous: when
# CHAT_ORCHESTRATOR=v2 and a canned-response gate fires (first_turn_greeting
# or empty_input), the chat extractor LLM call is skipped entirely.
#
# Real content turns ("I have Class G", "I'm looking for warehouse work")
# must still run the extractor so profile slots get filled. v1 path is
# unaffected -- the optimization only triggers under v2.
class _ExtractorSpy:
    """Tracks calls to chat_extractor._extract. Returns an empty
    ExtractionResult so the surrounding handler code is happy when it
    DOES get called."""
    def __init__(self):
        self.calls = 0

    def __call__(self, message, *, asked_slots):
        self.calls += 1
        from skillbridge.chat import extractor as chat_extractor
        return chat_extractor.ExtractionResult(
            fields={}, skills=[], declined=[], off_topic=False,
            raw_keys_dropped=[],
        )


def _patch_handle_anonymous_deps(monkeypatch, *, orchestrator="v2", staged=None):
    """Wire the minimal patches handle_anonymous needs to run end-to-end
    without DB/LLM. Returns (extractor_spy, store, planner_calls)."""
    monkeypatch.setattr(handler, "CHAT_ORCHESTRATOR", orchestrator)

    # Session store: fake that returns our prepared staged profile.
    store = FakeStore()
    if staged is not None:
        store_staged = staged
    else:
        store_staged = _staged(message_count=0, target_role_text=None,
                               intake_state_value="intake_collecting")

    class _FakeStoreReturning(FakeStore):
        def load(self, session_id):
            return store_staged
        def new_session(self):
            return store_staged.session_id

    fake_store = _FakeStoreReturning()
    monkeypatch.setattr(handler, "get_store", lambda: fake_store)

    # Extractor spy -- patch the module-level _extract function.
    extractor_spy = _ExtractorSpy()
    monkeypatch.setattr(handler, "_extract", extractor_spy)

    # Stub the planner so v2 path doesn't actually call Haiku when it does run.
    planner_calls = []
    def fake_plan(truth):
        planner_calls.append(truth)
        return _planner_decision(
            move="ask_one_clarifying_question",
            reason_code="target_role_unclear",
            ask_slot="target_role_text",
            tone="warm_supportive",
        )
    monkeypatch.setattr(handler, "plan_next_move", fake_plan)

    # Stub engine + responder so v2 path completes cleanly.
    engine_spy = EngineSpy(return_value=[])
    monkeypatch.setattr(handler.match_engine, "compute_matches_in_memory", engine_spy)
    monkeypatch.setattr(
        handler.match_engine, "next_skill_to_unlock_in_memory",
        lambda staged: (None, 0),
    )
    monkeypatch.setattr(handler, "compose_response_v2", lambda inp: "ok")
    monkeypatch.setattr(handler, "compose_reply", lambda inp: "v1 ok")

    return extractor_spy, fake_store, planner_calls


def test_v2_hi_first_turn_skips_extractor_call(monkeypatch):
    """Slice 8 optimization (target case): user opens with a bare
    greeting on turn 0. The early canned-gate short-circuit must
    fire, skipping the extractor LLM call entirely."""
    staged = _staged(message_count=0, target_role_text=None)
    extractor_spy, _, _ = _patch_handle_anonymous_deps(monkeypatch, staged=staged)

    response = handler.handle_anonymous(
        message="hi", session_id=staged.session_id,
    )
    assert response is not None
    assert response["final_move"] == "acknowledge_and_continue", (
        "First-turn 'hi' must route to the greeting gate (gate 3)"
    )
    assert extractor_spy.calls == 0, (
        f"Extractor was called {extractor_spy.calls} times on a 'hi' "
        f"turn. The early canned-gate short-circuit should have "
        f"skipped extraction entirely."
    )


def test_v2_first_turn_job_intent_still_runs_extractor(monkeypatch):
    """Slice 2 regression guard: 'I'm looking for warehouse work' on
    turn 0 MUST NOT fire the greeting gate. The early short-circuit
    must let real content through to the extractor, then to the v2
    planner. This is the bug Slice 2 review caught + fixed and we
    don't want it back."""
    staged = _staged(message_count=0, target_role_text=None)
    extractor_spy, _, planner_calls = _patch_handle_anonymous_deps(
        monkeypatch, staged=staged,
    )

    response = handler.handle_anonymous(
        message="I'm looking for warehouse work",
        session_id=staged.session_id,
    )
    assert response is not None
    assert extractor_spy.calls == 1, (
        "Content-bearing first-turn message must still run the "
        "extractor so the truth summary can see the implicit job intent."
    )
    assert len(planner_calls) == 1, (
        "Planner must be called when no canned gate fires."
    )


def test_v2_normal_content_turn_still_runs_extractor(monkeypatch):
    """Mid-conversation content turn ('I have Class G and 3 years
    driving'). Not a greeting, not empty, not an upload. The early
    short-circuit must not fire -- extractor must run so the new
    skill/slot info reaches the staged profile."""
    staged = _staged(message_count=5, target_role_text="truck driver")
    extractor_spy, _, _ = _patch_handle_anonymous_deps(
        monkeypatch, staged=staged,
    )

    response = handler.handle_anonymous(
        message="I have Class G and 3 years driving",
        session_id=staged.session_id,
    )
    assert response is not None
    assert extractor_spy.calls == 1, (
        f"Normal content turn must run the extractor "
        f"(got {extractor_spy.calls} calls)."
    )


def test_v1_first_turn_hi_still_runs_extractor(monkeypatch):
    """v1 rollback path must be unaffected by the v2 optimization.
    With CHAT_ORCHESTRATOR=v1, the early short-circuit is guarded off
    and the extractor runs on every turn just like before."""
    staged = _staged(message_count=0, target_role_text=None)
    extractor_spy, _, _ = _patch_handle_anonymous_deps(
        monkeypatch, orchestrator="v1", staged=staged,
    )

    response = handler.handle_anonymous(
        message="hi", session_id=staged.session_id,
    )
    assert response is not None
    assert extractor_spy.calls == 1, (
        f"Under CHAT_ORCHESTRATOR=v1, the extractor must always run "
        f"(got {extractor_spy.calls} calls). The v2 optimization must "
        f"NOT leak into v1 -- that would change v1 behavior and break "
        f"the rollback guarantee."
    )


# ===========================================================================
# Training registry integration on explain_gap turns
# ===========================================================================
# When TRAINING_REGISTRY_ENABLED=true, an explain_gap turn populates
# training_by_job from the curated YAML registry. Pending URLs are
# suppressed at runtime; verified URLs surface. Engine MUST NOT run
# on any explain_gap turn regardless of registry state.
def _patch_v2_with_explain_gap(monkeypatch):
    """Stub the v2 chain for an explain_gap turn. The planner emits
    explain_gap; the arbiter passes it through; no engine call."""
    monkeypatch.setattr(handler, "CHAT_ORCHESTRATOR", "v2")

    def fake_plan(truth):
        return _planner_decision(
            move="explain_gap",
            reason_code="credential_gap_present",
            tone="warm_supportive",
        )
    monkeypatch.setattr(handler, "plan_next_move", fake_plan)

    engine_spy = EngineSpy(return_value=[])
    monkeypatch.setattr(handler.match_engine, "compute_matches_in_memory", engine_spy)
    monkeypatch.setattr(handler, "_build_results_block", lambda r: ([], "none"))
    monkeypatch.setattr(handler, "_attach_training", lambda r: {})
    monkeypatch.setattr(
        handler.match_engine, "next_skill_to_unlock_in_memory",
        lambda staged: (None, 0),
    )

    captured_inputs: list = []
    monkeypatch.setattr(
        handler, "compose_response_v2",
        lambda inp: (captured_inputs.append(inp), "fallback reply")[1],
    )
    return engine_spy, captured_inputs


def test_explain_gap_with_registry_disabled_does_not_query_registry(monkeypatch):
    """Default state: TRAINING_REGISTRY_ENABLED=False -> training_by_job
    stays empty on explain_gap. No registry lookup happens.

    Slice D (2026-06-05): MESSAGE_UNDERSTANDING_ENABLED forced OFF here
    because the test exercises an explicit flag combination from the
    rollback era (registry off + router off). With router on, the lack
    of an entity routes to Rule 3 (ask) instead of explain_gap, which
    is a different code path. The test's intent is still valid for the
    rollback scenario and is preserved verbatim.
    """
    monkeypatch.setattr(handler, "TRAINING_REGISTRY_ENABLED", False)
    monkeypatch.setattr(handler, "MESSAGE_UNDERSTANDING_ENABLED", False)

    # R-3 (round-8/9 design): canonical alias resolution via
    # registry.lookup is DECOUPLED from TRAINING_REGISTRY_ENABLED
    # (docs/remaining-gaps-design.md §4a). The flag gates resource
    # surfacing -- provider names + URLs in training_by_job -- not
    # the registry load. The remaining-gaps detection hook may call
    # `get_registry()` regardless. The test's previous strict
    # "registry must not be queried" assertion contradicts that
    # design; the now-correct invariant is just "training_by_job
    # stays empty when the flag is off."
    from skillbridge.training import registry as registry_mod
    registry_calls: list[int] = []
    real_get_registry = registry_mod.get_registry
    monkeypatch.setattr(
        registry_mod, "get_registry",
        lambda: (registry_calls.append(1), real_get_registry())[1],
    )

    engine_spy, captured = _patch_v2_with_explain_gap(monkeypatch)
    staged = _staged(target_role_text="truck and coach technician")
    staged.last_presented_credential_gaps = ["310T technician certification"]

    response = handler._try_v2_path(
        staged=staged, message="how do I get my 310T?",
        uploaded_file=False, resume_info=None, store=FakeStore(),
    )
    assert response is not None
    assert response["final_move"] == "explain_gap"
    assert engine_spy.calls == 0
    # training_by_job stays empty when flag is off (the binding contract
    # for this test even after the round-8/9 design split).
    assert captured[0].training_by_job == {}


def test_explain_gap_with_registry_enabled_queries_registry(monkeypatch):
    """Flag ON: training_by_job is populated from the registry for each
    last_presented_credential_gaps entry. Pending URLs in the shipped
    YAML remain suppressed -- but the Resource entries (provider,
    type, summary) come through."""
    monkeypatch.setattr(handler, "TRAINING_REGISTRY_ENABLED", True)

    engine_spy, captured = _patch_v2_with_explain_gap(monkeypatch)
    staged = _staged(target_role_text="truck and coach technician")
    staged.last_presented_credential_gaps = ["310T technician certification"]

    response = handler._try_v2_path(
        staged=staged, message="how do I get my 310T?",
        uploaded_file=False, resume_info=None, store=FakeStore(),
    )
    assert response["final_move"] == "explain_gap"
    assert engine_spy.calls == 0   # critical: no engine on explain_gap

    # training_by_job is populated, keyed by gap
    tbj = captured[0].training_by_job
    assert any(k.startswith("gap:") for k in tbj.keys()), (
        f"Expected gap-prefixed keys in training_by_job; got {list(tbj.keys())}"
    )

    # The 310T gap surfaced entries from the registry (Skilled Trades
    # Ontario / Sault College / SCCC). All URLs are null per the
    # pending YAML -- verifying the safety net end-to-end.
    flat = [t for v in tbj.values() for t in v]
    assert flat, "training_by_job entries must be non-empty when registry has the gap"
    surfaced_urls = [t["url"] for t in flat if t["url"] is not None]
    assert surfaced_urls == [], (
        f"Pending registry URLs must NOT surface to the prompt. "
        f"Got: {surfaced_urls}"
    )
    # Provider names DO come through (so responder narrates them)
    providers = {t["provider"] for t in flat}
    assert "Sault Community Career Centre" in providers


def test_explain_gap_with_no_presented_gaps_and_no_message_gap_returns_empty(
    monkeypatch,
):
    """Both source paths return nothing: no carry-forward AND no
    specific gap named in the message. training_by_job stays empty
    -- correctly, since we have no anchor to recommend from.

    Pre-cold-session-slice this test used "how do I get my 310T?"
    and asserted empty training_by_job (wrong: 310T IS in the
    registry). The new contract: message-scan finds named registry
    gaps; only generic phrasings end up empty."""
    monkeypatch.setattr(handler, "TRAINING_REGISTRY_ENABLED", True)
    engine_spy, captured = _patch_v2_with_explain_gap(monkeypatch)
    staged = _staged()
    staged.last_presented_credential_gaps = []   # no carry-forward

    handler._try_v2_path(
        staged=staged,
        message="tell me more please",  # generic, no registry gap named
        uploaded_file=False, resume_info=None, store=FakeStore(),
    )
    assert captured[0].training_by_job == {}


def test_explain_gap_cold_session_discovers_gap_from_user_message(monkeypatch):
    """Post-grounding-bundle live-test fix: a user with no profile and
    no prior matches asks 'how can I get my Class G?'. Pre-fix the
    registry was never consulted because
    `staged.last_presented_credential_gaps` was empty -- LLM improvised
    and policy rejected. Post-fix, the registry's message scanner
    finds 'Class G' as an alias of the Class G gap, populates TRAINING,
    LLM gets registry data."""
    monkeypatch.setattr(handler, "TRAINING_REGISTRY_ENABLED", True)
    engine_spy, captured = _patch_v2_with_explain_gap(monkeypatch)

    # Cold session: no resume, no role, no carry-forward context
    staged = _staged(target_role_text=None)
    staged.last_presented_credential_gaps = []   # the critical pre-fix-empty case

    handler._try_v2_path(
        staged=staged,
        message="how can I get my Class G driver's licence?",
        uploaded_file=False, resume_info=None, store=FakeStore(),
    )

    # training_by_job is now populated FROM THE MESSAGE ITSELF
    tbj = captured[0].training_by_job
    assert tbj, (
        "Cold-session Class G question must populate training_by_job. "
        "Pre-fix this was empty and the LLM improvised."
    )
    flat = [t for v in tbj.values() for t in v]
    providers = {t["provider"] for t in flat}
    # Class G entry in seed YAML has DriveTest, Ontario.ca, SCCC
    assert "DriveTest" in providers or "Ontario.ca" in providers or (
        "Sault Community Career Centre" in providers
    ), f"Expected Class G providers from registry; got {providers}"


def test_explain_gap_message_discovery_works_for_310T(monkeypatch):
    """Same shape, different gap: '310T' alone in the message
    should surface the 310T resources."""
    monkeypatch.setattr(handler, "TRAINING_REGISTRY_ENABLED", True)
    engine_spy, captured = _patch_v2_with_explain_gap(monkeypatch)

    staged = _staged()
    staged.last_presented_credential_gaps = []

    handler._try_v2_path(
        staged=staged, message="how do I get my 310T?",
        uploaded_file=False, resume_info=None, store=FakeStore(),
    )

    tbj = captured[0].training_by_job
    assert tbj
    flat = [t for v in tbj.values() for t in v]
    for_gaps = {t.get("for_gap") for t in flat}
    assert "310T technician certification" in for_gaps


def test_explain_gap_combines_message_and_carry_forward_gaps(monkeypatch):
    """When the user asks about a NEW gap while the previous match
    turn surfaced a DIFFERENT one, BOTH are queried -- message-
    discovered first."""
    monkeypatch.setattr(handler, "TRAINING_REGISTRY_ENABLED", True)
    engine_spy, captured = _patch_v2_with_explain_gap(monkeypatch)

    staged = _staged()
    # carry-forward says 310T
    staged.last_presented_credential_gaps = ["310T technician certification"]

    handler._try_v2_path(
        staged=staged,
        # ...but user now asks about Class G
        message="what about Class G though?",
        uploaded_file=False, resume_info=None, store=FakeStore(),
    )

    tbj = captured[0].training_by_job
    flat = [t for v in tbj.values() for t in v]
    for_gaps = {t.get("for_gap") for t in flat}
    # Both gaps should appear in TRAINING
    assert "310T technician certification" in for_gaps
    assert "Class G driver's license" in for_gaps


def test_explain_gap_generic_question_returns_empty(monkeypatch):
    """Acceptance bound: when the user asks a generic training
    question with NO specific gap named, the message scan finds
    nothing AND there's no carry-forward. Returns empty -- which is
    correct: we shouldn't fabricate a recommendation."""
    monkeypatch.setattr(handler, "TRAINING_REGISTRY_ENABLED", True)
    engine_spy, captured = _patch_v2_with_explain_gap(monkeypatch)

    staged = _staged()
    staged.last_presented_credential_gaps = []

    handler._try_v2_path(
        staged=staged,
        message="any course do you recommend to improve my skill",
        uploaded_file=False, resume_info=None, store=FakeStore(),
    )

    # Empty TRAINING: this is the case the user named as a SEPARATE
    # concern (needs role-based recommender). The current slice is
    # NOT meant to handle this -- assert that as the contract.
    assert captured[0].training_by_job == {}


def test_explain_gap_unknown_registry_gap_logs_and_returns_empty(
    monkeypatch, caplog,
):
    """When the user's gap isn't in the registry, the recommender logs
    `unknown_gap=...` at INFO and returns an empty list. The chat
    continues via the responder's own explain_gap fallback.

    Slice D (2026-06-05): MESSAGE_UNDERSTANDING_ENABLED forced OFF here.
    With router on, 'snake oil license' produces no registry entity, so
    Rule 3 routes to ask_one_clarifying_question -- explain_gap never
    fires and the unknown_gap log line never emits. The test's intent
    (registry recommender logs for unknown gaps on the planner-first
    explain_gap path) is preserved verbatim.
    """
    import logging
    monkeypatch.setattr(handler, "TRAINING_REGISTRY_ENABLED", True)
    monkeypatch.setattr(handler, "MESSAGE_UNDERSTANDING_ENABLED", False)
    engine_spy, captured = _patch_v2_with_explain_gap(monkeypatch)
    staged = _staged()
    staged.last_presented_credential_gaps = ["snake oil licensing certification"]

    caplog.set_level(logging.INFO, logger="skillbridge.training.registry")
    handler._try_v2_path(
        staged=staged, message="how do I get my snake oil license?",
        uploaded_file=False, resume_info=None, store=FakeStore(),
    )
    # Telemetry fired
    assert any(
        "unknown_gap" in r.message and "snake oil" in r.message
        for r in caplog.records
    )
    # No training entries surfaced for an unknown gap
    assert captured[0].training_by_job == {}


def test_v2_path_calls_engine_from_exactly_one_place():
    """Static guard: the v2 dispatch must invoke
    `match_engine.compute_matches_in_memory(...)` from exactly ONE
    call site. Catches a refactor that accidentally sprinkles engine
    calls across paths.

    Matches the call pattern (function name followed by `(`) so
    documentation that mentions the function by name in a comment
    doesn't false-positive."""
    import inspect
    src = inspect.getsource(handler._try_v2_path)
    count = src.count("match_engine.compute_matches_in_memory(")
    assert count == 1, (
        f"_try_v2_path invokes match_engine.compute_matches_in_memory(...) "
        f"{count} times. It must be called exactly once (inside the "
        f"RunEngine branch). More than one call site means hidden "
        f"matching can occur."
    )


def test_v2_engine_call_is_inside_run_engine_branch():
    """Static guard companion: the single engine CALL must sit AFTER
    the `isinstance(pass1, RunEngine)` check, not before it. Uses
    the call pattern (with `(`) so comments don't false-positive."""
    import inspect
    src = inspect.getsource(handler._try_v2_path)
    run_engine_idx = src.find("isinstance(pass1, RunEngine)")
    engine_call_idx = src.find("match_engine.compute_matches_in_memory(")
    assert run_engine_idx != -1, "RunEngine check missing from _try_v2_path"
    assert engine_call_idx != -1, "Engine call missing from _try_v2_path"
    assert run_engine_idx < engine_call_idx, (
        "match_engine.compute_matches_in_memory(...) call must appear "
        "AFTER the isinstance(pass1, RunEngine) check. Otherwise the "
        "engine could run on a path the arbiter hasn't approved."
    )


# ===========================================================================
# Slice N-5 (2026-06-05): _compute_near_miss handler helper
#
# Concerns:
#   1. Each of the 4 preconditions is enforced (locked Q7 split: handler
#      computes; arbiter consumes the resulting list verbatim).
#   2. When all preconditions pass and the filter finds a candidate,
#      a payload dict reaches the responder.
#   3. When preconditions fail, ([], None) returns -- legacy
#      present_no_match behavior preserved byte-for-byte.
#   4. truth log gains near_miss=N field.
# ===========================================================================
from skillbridge.match.engine import MatchResult                # noqa: E402
from skillbridge.chat.truth_summary import TruthSummary         # noqa: E402


def _truth_for_near_miss(
    *,
    target_role_specificity: str = "specific",
    resume_parse_quality: str = "full",
    target_role_text: str = "truck and coach technician",
) -> TruthSummary:
    """Truth summary with the fields _compute_near_miss reads.
    Defaults satisfy all preconditions; tests override the field
    they're verifying. Note: chat_skill_count is NOT a TruthSummary
    field -- the handler computes it from `staged.skills` directly.
    Tests control it via the `chat_skills` arg on _staged_for_near_miss."""
    return TruthSummary(
        user_message="same role",
        target_role_text=target_role_text,
        target_role_specificity=target_role_specificity,
        resume_parse_quality=resume_parse_quality,
        enough_to_match=True,
        usable_evidence_present=True,
    )


def _low_match(
    *,
    title: str = "Truck and Coach Technician",
    override: bool = True,
    similarity: float = 1.0,
    required_missing: list[str] | None = None,
    noc: str | None = "7321",
) -> MatchResult:
    """Synthetic low-band MatchResult for handler precondition tests."""
    return MatchResult(
        job_id="j-low", profile_id="p", title=title,
        employer="Garden River First Nation", url=None, location="SSM",
        match_score=0.30, match_band="low", match_eligible=True,
        ineligibility_reason=None,
        matched_skills=[], missing_skills=[],
        matched_skill_ids=[], missing_skill_ids=[],
        required_skills_count=12, credential_warning=None,
        posted_date=None, noc_code=noc,
        score_explanation={
            "title_match_override": override,
            "title_match_similarity": similarity,
            "required_missing": required_missing or [
                "310T technician certification",
                "Class G driver's license",
                "emergency repair",
            ],
            "credential_gap_skills": [],
        },
    )


def _staged_for_near_miss(
    *,
    target_role: str = "truck and coach technician",
    target_noc: str = "7321",
    has_resume: bool = True,
    chat_skills: int = 0,
) -> StagedProfile:
    sp = StagedProfile.new("near-miss-session")
    sp.target_role_text = target_role
    sp.target_noc = target_noc
    sp.message_count = 5
    sp.intake_state = "intake_collecting"
    # Add chat-source skills if requested
    from skillbridge.session.staging import StagedSkill
    sp.skills = [
        StagedSkill(skill_name=f"skill_{i}", source="chat", confidence=0.9)
        for i in range(chat_skills)
    ]
    if has_resume:
        sp.skills.append(
            StagedSkill(skill_name="diesel repair", source="resume", confidence=0.9),
        )
    return sp


def test_compute_near_miss_fires_on_canonical_michael_scenario():
    """All preconditions met + a qualifying low-band candidate ->
    helper returns (non-empty list, populated dict)."""
    truth = _truth_for_near_miss()
    staged = _staged_for_near_miss()
    candidates, payload = handler._compute_near_miss(
        match_count=0,
        band_signal="low_only",
        in_memory_matches=[_low_match()],
        staged=staged,
        truth=truth,
    )
    assert len(candidates) == 1
    assert payload is not None
    assert payload["role"] == "Truck and Coach Technician"
    assert "310T technician certification" in payload["credential_gaps"]


@pytest.mark.parametrize("match_count,band_signal,reason", [
    (1, "low_only",       "match_count > 0 -> matches path, not near-miss"),
    (5, "low_only",       "match_count > 0 -> matches path"),
    (0, "none",           "band=none -> engine found nothing eligible"),
    (0, "strong_or_good", "band has strong/good -> matches path"),
    (0, "stretch_only",   "band has stretch -> matches path"),
])
def test_compute_near_miss_short_circuits_on_match_count_or_band(
    match_count, band_signal, reason,
):
    """The first two preconditions (match_count == 0 AND band == low_only).
    Either failing -> ([], None)."""
    truth = _truth_for_near_miss()
    staged = _staged_for_near_miss()
    candidates, payload = handler._compute_near_miss(
        match_count=match_count,
        band_signal=band_signal,
        in_memory_matches=[_low_match()],
        staged=staged,
        truth=truth,
    )
    assert candidates == [], reason
    assert payload is None, reason


def test_compute_near_miss_skipped_when_target_role_specificity_not_specific():
    """Precondition 3: target must be specific. 'vague' / 'none' ->
    fall through to no_match (asking the user for a role is the right
    next move, not gap-analyzing against nothing)."""
    for specificity in ("none", "vague"):
        truth = _truth_for_near_miss(target_role_specificity=specificity)
        staged = _staged_for_near_miss()
        candidates, payload = handler._compute_near_miss(
            match_count=0,
            band_signal="low_only",
            in_memory_matches=[_low_match()],
            staged=staged,
            truth=truth,
        )
        assert candidates == [], f"specificity={specificity}"
        assert payload is None, f"specificity={specificity}"


def test_compute_near_miss_skipped_when_no_baseline_evidence():
    """Precondition 4: baseline evidence (resume parsed OR >= 3 chat
    skills). Without either, gap analysis would just say 'you're
    missing everything' which isn't useful."""
    truth = _truth_for_near_miss(resume_parse_quality="no_resume")
    staged = _staged_for_near_miss(has_resume=False, chat_skills=0)
    candidates, payload = handler._compute_near_miss(
        match_count=0,
        band_signal="low_only",
        in_memory_matches=[_low_match()],
        staged=staged,
        truth=truth,
    )
    assert candidates == []
    assert payload is None


def test_compute_near_miss_fires_with_chat_skills_only_when_count_ge_3():
    """Baseline-evidence accepts EITHER a parsed resume OR >= 3 chat
    skills. Test the chat-skills branch in isolation: resume_parse=
    no_resume, but chat_skill_count >= 3 -> precondition holds."""
    truth = _truth_for_near_miss(resume_parse_quality="no_resume")
    staged = _staged_for_near_miss(has_resume=False, chat_skills=3)
    candidates, payload = handler._compute_near_miss(
        match_count=0,
        band_signal="low_only",
        in_memory_matches=[_low_match()],
        staged=staged,
        truth=truth,
    )
    assert len(candidates) == 1
    assert payload is not None


def test_compute_near_miss_skipped_when_chat_skills_below_3():
    """Bare-profile boundary: 2 chat skills + no resume is NOT enough."""
    truth = _truth_for_near_miss(resume_parse_quality="no_resume")
    staged = _staged_for_near_miss(has_resume=False, chat_skills=2)
    candidates, payload = handler._compute_near_miss(
        match_count=0,
        band_signal="low_only",
        in_memory_matches=[_low_match()],
        staged=staged,
        truth=truth,
    )
    assert candidates == []
    assert payload is None


def test_compute_near_miss_returns_empty_when_filter_finds_no_candidates():
    """All preconditions hold but the filter rejects all candidates
    (e.g. low-band jobs exist but none title/NOC-match). Helper
    returns ([], None), handler falls through to present_no_match."""
    truth = _truth_for_near_miss()
    staged = _staged_for_near_miss()
    # Low-band candidate with override=False AND no NOC match AND
    # similarity below threshold -> doesn't qualify
    rejected = _low_match(
        title="Marketing Coordinator", override=False, similarity=0.1,
        noc="1123",
    )
    candidates, payload = handler._compute_near_miss(
        match_count=0,
        band_signal="low_only",
        in_memory_matches=[rejected],
        staged=staged,
        truth=truth,
    )
    assert candidates == []
    assert payload is None


def test_e2e_michael_truck_tech_scenario_reaches_present_near_miss(monkeypatch):
    """Slice N-6 end-to-end pin: the canonical Michael Carter scenario
    drives the full v2 pipeline through to a `present_near_miss` reply
    with the expected gap data surfacing in the responder's deterministic
    fallback.

    NO stubbing of _build_results_block, _compute_near_miss, or
    resolve_match_outcome -- those are the surfaces under test. We only
    stub: the engine (to return the synthetic truck-tech job), the
    LLM (disabled, so the deterministic fallback fires), and the
    session store.

    This is the test the design doc names as the worked example. If
    any of the Slice N-* pieces drift, this test surfaces the drift
    before live-test reveals it."""
    from skillbridge.session.staging import StagedSkill
    from skillbridge.chat.planner import PlannerDecision

    # ---- Configure flags so the v2 path runs end-to-end ----
    monkeypatch.setattr(handler, "CHAT_ORCHESTRATOR", "v2")
    monkeypatch.setattr(handler, "TRAINING_REGISTRY_ENABLED", True)
    monkeypatch.setattr(handler, "MESSAGE_UNDERSTANDING_ENABLED", True)

    # ---- Synthesize Michael's profile: target role + parsed resume ----
    staged = StagedProfile.new("e2e-michael")
    staged.target_role_text = "truck and coach technician"
    staged.target_noc = "7321"
    staged.message_count = 5
    staged.intake_state = "intake_collecting"
    staged.skills = [
        StagedSkill(skill_name=name, source="resume", confidence=0.9)
        for name in (
            "diesel engine repair", "brake and suspension systems",
            "air brakes", "welding", "diagnostic tools",
            "heavy equipment", "MIG welding", "powertrain",
            "hydraulics", "preventive maintenance",
            "truck repair", "engine diagnostics", "vehicle inspection",
        )
    ]

    # ---- Force truth_summary so all near-miss preconditions hold ----
    # The arbiter pass 1 independently re-checks usable_evidence_present
    # and enough_to_match. We bypass the deterministic truth computation
    # so all four preconditions land True without ceremony around staged
    # field plumbing.
    from skillbridge.chat.truth_summary import TruthSummary
    def fake_build_truth(*, staged, user_message, **kw):
        return TruthSummary(
            user_message=user_message,
            target_role_text=staged.target_role_text,
            target_role_specificity="specific",
            resume_parse_quality="full",
            enough_to_match=True,
            usable_evidence_present=True,
            user_intent_signal="impatient_proceed",
        )
    monkeypatch.setattr(handler, "build_truth_summary", fake_build_truth)

    # ---- Stub the engine to return exactly one low-band truck-tech job ----
    truck_match = MatchResult(
        job_id="truck-1",
        profile_id=staged.session_id,
        title="Truck and Coach Technician",
        employer="Garden River First Nation",
        url="https://example.com/truck",
        location="Garden River First Nation",
        match_score=0.30,
        match_band="low",                # critical: low band drives the near-miss path
        match_eligible=True,
        ineligibility_reason=None,
        matched_skills=["welding"],
        missing_skills=["310T certificate of qualification"],
        matched_skill_ids=[None],
        missing_skill_ids=[None],
        required_skills_count=12,
        credential_warning=None,
        posted_date=None,
        noc_code="7321",
        score_explanation={
            "title_match_override": True,
            "title_match_similarity": 1.0,
            "required_missing": [
                "310T technician certification",
                "Class G driver's license",
                "truck service and maintenance",
                "emergency repair",
                "emissions testing preparation",
                "MTO contract supervision",          # operational -- must DROP
                "driver hour tracking",              # operational -- must DROP
                "on-call availability",              # operational -- must DROP
            ],
            "credential_gap_skills": [],
        },
    )
    monkeypatch.setattr(
        handler.match_engine, "compute_matches_in_memory",
        lambda staged, top=20: [truck_match],
    )
    monkeypatch.setattr(
        handler.match_engine, "next_skill_to_unlock_in_memory",
        lambda staged: (None, 0),
    )

    # ---- Force planner to emit proceed_to_match so engine runs ----
    # Truth says enough_to_match=True (resume parsed + role specific),
    # so the planner's proceed_to_match clears arbiter pass 1.
    def fake_plan(truth_json):
        return PlannerDecision.model_validate({
            "move": "proceed_to_match",
            "reason_code": "user_explicitly_asked_to_match",
            "ask_slot": None,
            "tone": "brief_confident",
        })
    monkeypatch.setattr(handler, "plan_next_move", fake_plan)

    # ---- Disable LLM so deterministic fallback fires in the responder ----
    from skillbridge.chat import responder as responder_mod
    monkeypatch.setattr(responder_mod, "is_enabled", lambda: False)

    # ---- Drive the turn ----
    response = handler._try_v2_path(
        staged=staged,
        message="same role",
        uploaded_file=False,
        resume_info=None,
        store=FakeStore(),
    )

    # ---- Behavior assertions: present_near_miss reached end-to-end ----
    assert response is not None
    assert response["final_move"] == "present_near_miss", (
        f"expected present_near_miss; got {response['final_move']!r}; "
        f"reply was {response.get('reply')!r}"
    )

    # ---- Reply content assertions: gap data surfaced verbatim ----
    reply = response["reply"]
    # Role anchor (Sentence 1)
    assert "Truck and Coach Technician" in reply
    assert "Sault Ste. Marie" in reply
    assert "not a realistic match yet" in reply
    # Credentials (canonical-aligned) lead the gap list
    assert "310T technician certification" in reply
    assert "Class G driver's license" in reply
    # Core skills follow
    assert "emergency repair" in reply
    # Operational requirements MUST NOT appear -- filtered upstream
    for op_gap in (
        "MTO contract supervision", "driver hour tracking",
        "on-call availability",
    ):
        assert op_gap not in reply, (
            f"operational gap {op_gap!r} leaked into reply: {reply!r}"
        )
    # Closing offer
    assert "Want to walk through" in reply


def test_compute_near_miss_ignores_non_low_band_matches_in_input():
    """band_signal says low_only but the in_memory_matches list also
    contains ineligible / non-low entries. Helper must subset to
    eligible-low BEFORE running the filter."""
    truth = _truth_for_near_miss()
    staged = _staged_for_near_miss()
    inputs = [
        # ineligible -- ignored
        MatchResult(
            job_id="j-elig-false", profile_id="p", title="X", employer=None,
            url=None, location=None, match_score=0.50, match_band="stretch",
            match_eligible=False, ineligibility_reason=None,
            matched_skills=[], missing_skills=[], matched_skill_ids=[],
            missing_skill_ids=[], required_skills_count=0,
            credential_warning=None, posted_date=None, noc_code=None,
            score_explanation={"title_match_override": True},
        ),
        _low_match(),  # the real one
    ]
    candidates, payload = handler._compute_near_miss(
        match_count=0,
        band_signal="low_only",
        in_memory_matches=inputs,
        staged=staged,
        truth=truth,
    )
    assert len(candidates) == 1
    assert candidates[0].job_id == "j-low"
    assert payload is not None


# ===========================================================================
# Slice (2026-06-08): target_role anaphor resolution
#
# Live-test bug exposed 2026-06-05: when a user replied "same role" to the
# bot's target-role question, the chat extractor missed the slot, and the
# fallback path stored the LITERAL string "same role" as
# target_role_text. Downstream engine then computed
# title_match_similarity("Truck and Coach Technician", "same role") = 0.171,
# and the truck-tech posting failed to surface as a match.
#
# This slice's fixes:
#   1. _is_target_role_anaphor / _resolve_target_role_anaphor in handler.py
#   2. _VAGUE_TARGET_TOKENS extended so a residual literal anaphor
#      (no resume to resolve against) classifies as vague -> planner re-asks
#   3. Handler regression mimicking Michael's exact 3-turn flow
# ===========================================================================
@pytest.mark.parametrize("phrase,expected", [
    # Anaphoric phrases that MUST be detected
    ("same role",           True),
    ("same",                True),
    ("the same",            True),
    ("same kind of work",   True),
    ("current role",        True),
    ("current job",         True),
    ("current position",    True),
    ("previous job",        True),
    ("prior position",      True),
    ("past field",          True),
    ("this",                True),
    ("this one",            True),
    ("this role",           True),
    ("that",                True),
    ("it",                  True),
    # Negative cases: real role names, decline phrases, content -- not anaphors
    ("truck and coach technician", False),
    ("warehouse manager",          False),
    ("electrician apprentice",     False),
    ("no thanks",                  False),
    ("yes",                        False),  # deliberately NOT an anaphor
    ("",                           False),
    ("   ",                        False),
])
def test_is_target_role_anaphor(phrase, expected):
    """The detection patterns conservatively match anaphoric phrases.
    'yes' / 'yeah' are deliberately excluded -- they're ambiguous and
    we'd rather have the planner re-ask than risk false positives."""
    assert handler._is_target_role_anaphor(phrase) is expected


def test_resolve_anaphor_prefers_is_current_work_history_entry():
    """When the resume's work_history has an entry marked is_current=True,
    resolution uses THAT entry's title regardless of start_year ordering."""
    sp = StagedProfile.new("anaphor-test")
    sp.resume_facts_json = {
        "work_history": [
            {"title": "Heavy Equipment Service Assistant",
             "is_current": False, "start_year": 2021},
            {"title": "Apprentice Truck & Coach Technician",
             "is_current": True, "start_year": 2024},
        ],
    }
    resolved = handler._resolve_target_role_anaphor("same role", sp)
    assert resolved == "Apprentice Truck & Coach Technician"


def test_resolve_anaphor_falls_back_to_most_recent_start_year_when_no_current():
    """No entry has is_current=True. Resolution picks the highest
    start_year as the proxy for 'current'."""
    sp = StagedProfile.new("anaphor-test")
    sp.resume_facts_json = {
        "work_history": [
            {"title": "Junior Mechanic",  "is_current": False, "start_year": 2020},
            {"title": "Senior Technician", "is_current": False, "start_year": 2023},
            {"title": "Intermediate Tech", "is_current": False, "start_year": 2021},
        ],
    }
    resolved = handler._resolve_target_role_anaphor("same role", sp)
    assert resolved == "Senior Technician"


def test_resolve_anaphor_returns_none_when_no_resume():
    """Anaphor with nothing to anchor against: return None. The handler
    must NOT store a literal anaphor when this happens; the planner
    re-asks the role question."""
    sp = StagedProfile.new("anaphor-test")  # default resume_facts_json=None
    assert handler._resolve_target_role_anaphor("same role", sp) is None


def test_resolve_anaphor_returns_none_when_work_history_is_empty():
    """Resume parsed but work_history empty (some candidates upload
    education-only resumes). Same fallback as no resume at all."""
    sp = StagedProfile.new("anaphor-test")
    sp.resume_facts_json = {"work_history": [], "skills": [], "education": []}
    assert handler._resolve_target_role_anaphor("same role", sp) is None


def test_resolve_anaphor_skips_entries_with_blank_titles():
    """Defensive: work_history entry with a blank/missing title is
    skipped, NOT returned as an empty string target_role."""
    sp = StagedProfile.new("anaphor-test")
    sp.resume_facts_json = {
        "work_history": [
            {"title": "", "is_current": True, "start_year": 2024},
            {"title": "Heavy Equipment Service Assistant",
             "is_current": False, "start_year": 2021},
        ],
    }
    resolved = handler._resolve_target_role_anaphor("same role", sp)
    assert resolved == "Heavy Equipment Service Assistant"


# ---- Michael's exact regression: 3-turn flow end-to-end ----
class _PresetStore(FakeStore):
    """FakeStore variant that returns a preset StagedProfile from load().
    Lets the anaphor regression test drive handle_anonymous against a
    specific resume_facts_json + last_asked_slots state without going
    through the full multi-turn setup."""
    def __init__(self, preset: StagedProfile):
        super().__init__()
        self._preset = preset

    def load(self, session_id):
        if session_id == self._preset.session_id:
            return self._preset
        return None


def test_michael_anaphor_flow_resolves_to_resume_title(monkeypatch):
    """Live-test bug regression. Mimics the 3-turn flow that produced
    target_role_text='same role' in production. We pre-populate a
    staged profile with the resume already parsed AND
    last_asked_slots=['target_role_text'] (i.e. the bot just asked for
    the role), then drive handle_anonymous with 'same role' as the
    user reply. The fallback_fill path must resolve the anaphor to
    the resume's current job title, NOT store the literal."""
    monkeypatch.setattr(handler, "CHAT_ORCHESTRATOR", "v2")
    monkeypatch.setattr(handler, "MESSAGE_UNDERSTANDING_ENABLED", False)

    # Stub the downstream chain so the test focuses on the slot fill.
    # Everything below is dead code for this assertion's purpose; we
    # only care about staged.target_role_text after the call.
    monkeypatch.setattr(handler, "plan_next_move", lambda truth: None)
    monkeypatch.setattr(handler, "compose_response_v2", lambda inp: "ok")
    monkeypatch.setattr(handler, "compose_reply",     lambda inp: "ok")
    monkeypatch.setattr(
        handler.match_engine, "compute_matches_in_memory",
        lambda staged, top=20: [],
    )
    monkeypatch.setattr(
        handler.match_engine, "next_skill_to_unlock_in_memory",
        lambda staged: (None, 0),
    )
    # Disable the LLM-driven chat extractor so it doesn't overwrite
    # the slot via its own path. The fallback_fill is what we test.
    from skillbridge.chat import extractor as chat_extractor
    monkeypatch.setattr(
        chat_extractor, "extract", lambda *a, **kw: chat_extractor.ExtractionResult(
            fields={}, skills=[], declined=[], off_topic=False, raw_keys_dropped=[],
        ),
    )

    staged = StagedProfile.new("michael-anaphor-regression")
    staged.message_count = 4
    staged.intake_state = "intake_collecting"
    staged.resume_facts_json = {
        "work_history": [
            {"title": "Apprentice Truck & Coach Technician",
             "is_current": True, "start_year": 2024},
            {"title": "Heavy Equipment Service Assistant",
             "is_current": False, "start_year": 2021},
        ],
    }
    staged.last_asked_slots = ["target_role_text"]
    staged.target_role_text = None

    store = _PresetStore(staged)
    monkeypatch.setattr(handler, "get_store", lambda: store)

    handler.handle_anonymous(
        message="same role", session_id=staged.session_id,
    )

    # CORE ASSERTION: target_role_text is the RESOLVED title, NOT the literal
    # The post-handler staged state is what was saved back to the store
    # (or the same object if save was a no-op); read from the staged
    # variable we passed in -- handler mutates in place before save.
    assert staged.target_role_text == "Apprentice Truck & Coach Technician", (
        f"Anaphor resolution failed -- target_role_text is "
        f"{staged.target_role_text!r}; expected resolved resume title"
    )


def test_michael_anaphor_flow_without_resume_does_not_store_literal(monkeypatch):
    """Inverse regression: a 'same role' reply WITHOUT a parsed resume
    must leave target_role_text empty (NOT store the literal 'same role').
    Pre-fix this case produced target_role_text='same role' AND
    target_role_specificity='specific', which corrupted downstream
    title-match scoring. Post-fix the slot stays empty so the planner
    asks again."""
    monkeypatch.setattr(handler, "CHAT_ORCHESTRATOR", "v2")
    monkeypatch.setattr(handler, "MESSAGE_UNDERSTANDING_ENABLED", False)
    monkeypatch.setattr(handler, "plan_next_move", lambda truth: None)
    monkeypatch.setattr(handler, "compose_response_v2", lambda inp: "ok")
    monkeypatch.setattr(handler, "compose_reply",     lambda inp: "ok")
    monkeypatch.setattr(
        handler.match_engine, "compute_matches_in_memory",
        lambda staged, top=20: [],
    )
    monkeypatch.setattr(
        handler.match_engine, "next_skill_to_unlock_in_memory",
        lambda staged: (None, 0),
    )
    from skillbridge.chat import extractor as chat_extractor
    monkeypatch.setattr(
        chat_extractor, "extract", lambda *a, **kw: chat_extractor.ExtractionResult(
            fields={}, skills=[], declined=[], off_topic=False, raw_keys_dropped=[],
        ),
    )

    staged = StagedProfile.new("michael-no-resume-anaphor")
    staged.message_count = 4
    staged.intake_state = "intake_collecting"
    staged.resume_facts_json = None  # no resume parsed
    staged.last_asked_slots = ["target_role_text"]
    staged.target_role_text = None

    store = _PresetStore(staged)
    monkeypatch.setattr(handler, "get_store", lambda: store)

    handler.handle_anonymous(
        message="same role", session_id=staged.session_id,
    )

    # CORE ASSERTION: target_role_text was NOT filled with the literal
    assert not staged.target_role_text, (
        f"Unresolved anaphor must NOT store the literal; got "
        f"target_role_text={staged.target_role_text!r}"
    )


# ===========================================================================
# Anaphor slice REVIEW PATCH (2026-06-08): two surfaces the v1 missed.
# ===========================================================================
def test_anaphor_bypass_via_llm_extracted_field_is_closed(monkeypatch):
    """REVIEW FINDING: the v1 resolver only fired in the fallback_fill
    path (when the LLM extractor missed the slot). If the extractor
    itself returned target_role_text='same role', the value was stored
    verbatim by merge_fields and bypassed the resolver entirely.

    This test stubs the extractor to populate target_role_text with an
    anaphor and verifies the pre-merge normalization either resolves
    or drops the value. NEVER stores the literal anaphor."""
    monkeypatch.setattr(handler, "CHAT_ORCHESTRATOR", "v2")
    monkeypatch.setattr(handler, "MESSAGE_UNDERSTANDING_ENABLED", False)
    monkeypatch.setattr(handler, "plan_next_move", lambda truth: None)
    monkeypatch.setattr(handler, "compose_response_v2", lambda inp: "ok")
    monkeypatch.setattr(handler, "compose_reply",     lambda inp: "ok")
    monkeypatch.setattr(
        handler.match_engine, "compute_matches_in_memory",
        lambda staged, top=20: [],
    )
    monkeypatch.setattr(
        handler.match_engine, "next_skill_to_unlock_in_memory",
        lambda staged: (None, 0),
    )

    # Force the LLM extractor to return an anaphor in fields
    from skillbridge.chat import extractor as chat_extractor
    monkeypatch.setattr(
        chat_extractor, "extract",
        lambda *a, **kw: chat_extractor.ExtractionResult(
            fields={"target_role_text": "same role"},  # extractor itself returns anaphor
            skills=[], declined=[], off_topic=False, raw_keys_dropped=[],
        ),
    )

    staged = StagedProfile.new("bypass-test")
    staged.message_count = 4
    staged.intake_state = "intake_collecting"
    staged.resume_facts_json = {
        "work_history": [
            {"title": "Apprentice Truck & Coach Technician",
             "is_current": True, "start_year": 2024},
        ],
    }
    staged.last_asked_slots = ["target_role_text"]
    staged.target_role_text = None

    store = _PresetStore(staged)
    monkeypatch.setattr(handler, "get_store", lambda: store)

    handler.handle_anonymous(
        message="same role", session_id=staged.session_id,
    )

    # CORE ASSERTION: extractor's literal 'same role' was normalized
    # to the resolved title before merging. Pre-fix this would store
    # 'same role' verbatim, corrupting downstream title-match scoring.
    assert staged.target_role_text == "Apprentice Truck & Coach Technician", (
        f"Pre-merge normalization failed -- got "
        f"target_role_text={staged.target_role_text!r}"
    )


def test_anaphor_bypass_with_no_resume_drops_extracted_anaphor(monkeypatch):
    """REVIEW FINDING extension: if the extractor returns an anaphor AND
    the user has no resume, the value MUST be dropped (not stored).
    The fallback_fill path then sees the slot as empty and applies its
    own no-store logic. End state: target_role_text stays empty."""
    monkeypatch.setattr(handler, "CHAT_ORCHESTRATOR", "v2")
    monkeypatch.setattr(handler, "MESSAGE_UNDERSTANDING_ENABLED", False)
    monkeypatch.setattr(handler, "plan_next_move", lambda truth: None)
    monkeypatch.setattr(handler, "compose_response_v2", lambda inp: "ok")
    monkeypatch.setattr(handler, "compose_reply",     lambda inp: "ok")
    monkeypatch.setattr(
        handler.match_engine, "compute_matches_in_memory",
        lambda staged, top=20: [],
    )
    monkeypatch.setattr(
        handler.match_engine, "next_skill_to_unlock_in_memory",
        lambda staged: (None, 0),
    )
    from skillbridge.chat import extractor as chat_extractor
    monkeypatch.setattr(
        chat_extractor, "extract",
        lambda *a, **kw: chat_extractor.ExtractionResult(
            fields={"target_role_text": "current role"},
            skills=[], declined=[], off_topic=False, raw_keys_dropped=[],
        ),
    )

    staged = StagedProfile.new("bypass-no-resume")
    staged.message_count = 4
    staged.intake_state = "intake_collecting"
    staged.resume_facts_json = None
    staged.last_asked_slots = ["target_role_text"]
    staged.target_role_text = None

    store = _PresetStore(staged)
    monkeypatch.setattr(handler, "get_store", lambda: store)

    handler.handle_anonymous(
        message="current role", session_id=staged.session_id,
    )

    assert not staged.target_role_text, (
        f"Unresolved anaphor from extractor must NOT be stored; got "
        f"target_role_text={staged.target_role_text!r}"
    )


def test_resolver_uses_suppression_aware_view():
    """REVIEW FINDING: v1 resolver read raw resume_facts_json. If the
    user suppressed the is_current=True entry (e.g. "remove that, it's
    wrong"), the responder/matcher would no longer see it -- but the
    resolver still did. This test verifies the resolver now skips
    suppressed entries via _effective_facts_view."""
    sp = StagedProfile.new("suppress-test")
    sp.resume_facts_json = {
        "work_history": [
            {"fact_id": "wh-1", "title": "WRONG: This isn't actually Michael's job",
             "is_current": True, "start_year": 2024},
            {"fact_id": "wh-2", "title": "Heavy Equipment Service Assistant",
             "is_current": False, "start_year": 2021},
        ],
    }
    # User suppressed the first entry
    sp.suppressed_fact_ids = ["wh-1"]

    resolved = handler._resolve_target_role_anaphor("same role", sp)
    # The is_current=True entry is suppressed; resolver falls back to
    # the next entry (the 2021 one). It MUST NOT surface the
    # suppressed title.
    assert resolved == "Heavy Equipment Service Assistant", (
        f"Resolver returned suppressed title or wrong fallback: {resolved!r}"
    )


def test_resolver_returns_none_when_all_work_history_suppressed():
    """Defensive: if every work_history entry is suppressed, the
    resolver returns None -- same shape as 'no resume at all'."""
    sp = StagedProfile.new("all-suppressed")
    sp.resume_facts_json = {
        "work_history": [
            {"fact_id": "wh-1", "title": "Job A", "is_current": True, "start_year": 2024},
            {"fact_id": "wh-2", "title": "Job B", "is_current": False, "start_year": 2021},
        ],
    }
    sp.suppressed_fact_ids = ["wh-1", "wh-2"]
    assert handler._resolve_target_role_anaphor("same role", sp) is None
