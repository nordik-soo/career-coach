"""Handler-level integration tests for the deterministic router.

The locked architectural promise from docs/message-understanding-design.md:

    Planner MUST NOT run for high-confidence scope_violation or
    training-with-entity cases.

Unit tests in test_chat_routing.py verify the router in isolation.
THESE tests verify the *handler* honors that contract when the
MESSAGE_UNDERSTANDING_ENABLED flag is ON. We monkey-patch
plan_next_move to track every call and assert it is NEVER called on
the HIGH cases, and IS called on ambiguous/low cases.

The flag-OFF behavior is already covered: every pre-existing handler
test passes unchanged, which proves the v2.1 wiring is a true no-op
when the flag is off.
"""
from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("LLM_ENABLED", "false")

import pytest

from skillbridge.chat import handler
from skillbridge.chat.arbiter import ArbiterDecision
from skillbridge.chat.planner import PlannerDecision
from skillbridge.session.staging import StagedProfile


pytestmark = pytest.mark.nodb


# ===========================================================================
# Local scaffolding -- kept minimal to make the test file self-contained.
# Mirrors test_chat_handler_v2.py's helpers but only what the router
# integration needs.
# ===========================================================================
class _FakeStore:
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
    *,
    message_count: int = 5,
    target_role_text: str | None = "warehouse worker",
) -> StagedProfile:
    sp = StagedProfile.new("test-session")
    sp.message_count = message_count
    sp.target_role_text = target_role_text
    sp.intake_state = "intake_collecting"
    return sp


def _wire_v2_handler(
    monkeypatch,
    *,
    flag_on: bool,
    training_registry_on: bool = True,
):
    """Common scaffolding: enable v2, set the new flag, stub the LLM
    pieces of the chain. Returns three things the tests assert on:

      planner_calls  -- list[dict] populated each time the planner LLM
                        would have been called. For HIGH cases this MUST
                        stay empty.
      engine_spy     -- callable + .calls counter. Independent of routing;
                        the engine is only relevant to Rule 4 tests.
      responder_inputs -- list of ResponderV2Input objects fed to compose.
    """
    monkeypatch.setattr(handler, "CHAT_ORCHESTRATOR", "v2")
    monkeypatch.setattr(handler, "MESSAGE_UNDERSTANDING_ENABLED", flag_on)
    monkeypatch.setattr(
        handler, "TRAINING_REGISTRY_ENABLED", training_registry_on,
    )

    planner_calls: list[dict] = []

    def fake_plan(truth_json):
        planner_calls.append(truth_json)
        # Return a safe default so if the planner IS called the rest of
        # the chain doesn't crash -- but the assertion is that it
        # WASN'T called for these inputs.
        return PlannerDecision.model_validate({
            "move": "ask_one_clarifying_question",
            "reason_code": "target_role_unclear",
            "ask_slot": "target_role_text",
            "tone": "warm_supportive",
        })

    monkeypatch.setattr(handler, "plan_next_move", fake_plan)

    engine_calls = {"n": 0, "last_staged": None}

    def fake_engine(staged, top=20):
        engine_calls["n"] += 1
        engine_calls["last_staged"] = staged
        return []

    monkeypatch.setattr(
        handler.match_engine, "compute_matches_in_memory", fake_engine,
    )

    def fake_build_results_block(matches):
        return (list(matches), "none" if not matches else "strong_or_good")

    monkeypatch.setattr(handler, "_build_results_block", fake_build_results_block)
    monkeypatch.setattr(handler, "_attach_training", lambda results: {})
    monkeypatch.setattr(
        handler.match_engine, "next_skill_to_unlock_in_memory",
        lambda staged: (None, 0),
    )

    responder_inputs: list[Any] = []

    def fake_compose_v2(inp):
        responder_inputs.append(inp)
        return "canned router-test reply"

    monkeypatch.setattr(handler, "compose_response_v2", fake_compose_v2)
    monkeypatch.setattr(
        handler, "compose_reply",
        lambda inp: "v1 fallback reply (should not appear in these tests)",
    )

    return planner_calls, engine_calls, responder_inputs


# ===========================================================================
# Locked architectural promise:
# HIGH-confidence cases -> planner LLM NEVER called.
# ===========================================================================
def test_flag_on_pr_scope_question_skips_planner_lll(monkeypatch):
    """Cold-session 'Can I apply for PR while looking?' MUST resolve
    to redirect_scope via the router -- planner not consulted."""
    planner_calls, engine_calls, responder_inputs = _wire_v2_handler(
        monkeypatch, flag_on=True,
    )
    staged = _staged()

    response = handler._try_v2_path(
        staged=staged,
        message="Can I apply for PR while looking for work?",
        uploaded_file=False,
        resume_info=None,
        store=_FakeStore(),
    )
    assert response is not None
    assert planner_calls == [], (
        "Planner LLM MUST NOT be called for HIGH-confidence scope "
        "violations. The router's job is to enforce that contract."
    )
    assert engine_calls["n"] == 0  # no engine on redirect_scope
    assert response["final_move"] == "redirect_scope"


def test_flag_on_class_g_training_question_skips_planner_llm(monkeypatch):
    """Cold-session 'how can I get my Class G driver's licence' is the
    canonical training+entity case. Planner skipped, explain_gap emitted."""
    planner_calls, engine_calls, _ = _wire_v2_handler(
        monkeypatch, flag_on=True,
    )

    response = handler._try_v2_path(
        staged=_staged(),
        message="how can I get my Class G driver's licence",
        uploaded_file=False,
        resume_info=None,
        store=_FakeStore(),
    )
    assert response is not None
    assert planner_calls == [], (
        "Planner LLM MUST NOT be called for training+entity HIGH cases."
    )
    assert response["final_move"] == "explain_gap"


def test_flag_on_excel_course_skips_planner_llm(monkeypatch):
    """Cold-session 'online Excel course' -- training+entity HIGH.
    Same architectural guarantee."""
    planner_calls, _, _ = _wire_v2_handler(monkeypatch, flag_on=True)

    response = handler._try_v2_path(
        staged=_staged(),
        message="online Excel course",
        uploaded_file=False,
        resume_info=None,
        store=_FakeStore(),
    )
    assert response is not None
    assert planner_calls == []
    assert response["final_move"] == "explain_gap"


def test_flag_on_training_no_entity_skips_planner_llm(monkeypatch):
    """Rule 3: training intent without specific credential. Router asks
    a clarifying question deterministically; planner not consulted."""
    planner_calls, _, responder_inputs = _wire_v2_handler(
        monkeypatch, flag_on=True,
    )

    response = handler._try_v2_path(
        staged=_staged(),
        message="what training do I need",
        uploaded_file=False,
        resume_info=None,
        store=_FakeStore(),
    )
    assert response is not None
    assert planner_calls == []
    assert response["final_move"] == "ask_one_clarifying_question"


# ===========================================================================
# MEDIUM / LOW cases -> planner IS called (as today).
# Proves the router is not over-eager.
# ===========================================================================
def test_flag_on_low_signal_message_still_consults_planner(monkeypatch):
    """'hello there' -- no router rule fires. The planner is consulted
    exactly as today. This is the test that proves the router doesn't
    over-route."""
    planner_calls, _, _ = _wire_v2_handler(monkeypatch, flag_on=True)

    handler._try_v2_path(
        staged=_staged(message_count=2),
        message="thanks for your help",
        uploaded_file=False,
        resume_info=None,
        store=_FakeStore(),
    )
    assert len(planner_calls) == 1, (
        "Planner MUST be called on low-signal messages. The router "
        "only pre-empts HIGH-confidence cases."
    )


def test_flag_on_conversational_yes_consults_planner(monkeypatch):
    """Multi-turn context: 'yes' after the bot asked something -- the
    planner has last_assistant_move; the router does not. Planner MUST
    be called."""
    planner_calls, _, _ = _wire_v2_handler(monkeypatch, flag_on=True)

    handler._try_v2_path(
        staged=_staged(message_count=3),
        message="yes that sounds right",
        uploaded_file=False,
        resume_info=None,
        store=_FakeStore(),
    )
    assert len(planner_calls) == 1


# ===========================================================================
# Flag OFF -- the new wiring is a NO-OP: planner runs on every message,
# exactly the pre-v2.1 behavior. (The full suite passing with flag off
# is the broad guarantee; this test pins the specific case.)
# ===========================================================================
def test_flag_off_pr_question_still_consults_planner(monkeypatch):
    """When the flag is OFF, even a HIGH-confidence PR question goes
    through the planner. The existing arbiter scope-override safety
    net catches it downstream (until Slice D removes that rule).
    This pins the rollback contract."""
    planner_calls, _, _ = _wire_v2_handler(monkeypatch, flag_on=False)

    handler._try_v2_path(
        staged=_staged(),
        message="Can I apply for PR while looking for work?",
        uploaded_file=False,
        resume_info=None,
        store=_FakeStore(),
    )
    assert len(planner_calls) == 1, (
        "With MESSAGE_UNDERSTANDING_ENABLED off, the planner MUST be "
        "called as before. No silent v2.1 behavior."
    )


# ===========================================================================
# Defensive: a router exception falls through to the planner cleanly.
# ===========================================================================
def test_flag_on_router_exception_falls_through_to_planner(monkeypatch):
    """If understand_message or route_from_understanding raise for any
    reason, the handler MUST NOT crash. It logs WARNING and proceeds
    with the planner-first path. This is the safety net for the
    rollout."""
    planner_calls, _, _ = _wire_v2_handler(monkeypatch, flag_on=True)

    def boom(*args, **kwargs):
        raise RuntimeError("synthetic router failure")

    monkeypatch.setattr(handler, "understand_message", boom)

    response = handler._try_v2_path(
        staged=_staged(),
        message="any message will do",
        uploaded_file=False,
        resume_info=None,
        store=_FakeStore(),
    )
    # Planner was called as fallback -- handler did not crash.
    assert response is not None
    assert len(planner_calls) == 1
