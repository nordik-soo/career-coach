"""Unit tests for chat orchestration v2 slice 2 -- the three deterministic gates.

Pure-function coverage of every branch in `skillbridge/chat/gates.py`:

  - Each individual gate's positive + negative cases
  - The dispatcher's precedence (empty -> resume -> first-turn)
  - Compound turns (multiple gate triggers at once -> first-match-wins)
  - The "no fourth gate" invariant -- imported gate functions match the
    design doc's 3-gate count; if a contributor adds a fourth, this
    test fails as a structural smell signal.

No DB, no LLM. The `nodb` marker keeps the conftest TRUNCATE off.
"""
from __future__ import annotations

import os

os.environ.setdefault("LLM_ENABLED", "false")

import pytest

from skillbridge.chat.gates import (
    GateDecision,
    _is_empty_input_gate,
    _is_first_turn_greeting_gate,
    _is_resume_upload_gate,
    evaluate_gates,
)

pytestmark = pytest.mark.nodb


# ---------------------------------------------------------------------------
# Gate 1: empty input
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("user_message", ["", "   ", "\n", "\t  \n  "])
def test_empty_input_gate_fires_on_whitespace_only(user_message):
    d = _is_empty_input_gate(user_message=user_message, uploaded_file=False)
    assert d is not None
    assert d.gate_name == "empty_input"
    assert d.final_move == "ask_one_clarifying_question"
    assert d.canned_response is not None
    assert d.canned_response.strip()
    assert d.ask_slot is None


def test_empty_input_gate_does_not_fire_when_message_has_content():
    """Even a single non-whitespace character routes to planner."""
    d = _is_empty_input_gate(user_message="hi", uploaded_file=False)
    assert d is None


def test_empty_input_gate_skips_when_file_uploaded():
    """An upload with no caption is NOT empty input -- it routes to
    the resume-upload gate. The whole point of the gate ordering is
    that an upload always wins over text-emptiness."""
    d = _is_empty_input_gate(user_message="", uploaded_file=True)
    assert d is None


# ---------------------------------------------------------------------------
# Gate 2: resume upload
# ---------------------------------------------------------------------------
def test_resume_upload_gate_fires_on_any_upload():
    d = _is_resume_upload_gate(uploaded_file=True)
    assert d is not None
    assert d.gate_name == "resume_upload"
    assert d.final_move == "confirm_resume_summary"
    # canned_response is None -- the responder needs to narrate the
    # actual parsed facts, which the gate doesn't have access to.
    assert d.canned_response is None


def test_resume_upload_gate_does_not_fire_without_upload():
    d = _is_resume_upload_gate(uploaded_file=False)
    assert d is None


# ---------------------------------------------------------------------------
# Gate 3: first-turn greeting (content-aware)
# ---------------------------------------------------------------------------
def test_first_turn_greeting_fires_on_bare_greeting_at_count_zero():
    d = _is_first_turn_greeting_gate(user_message="hi", message_count=0)
    assert d is not None
    assert d.gate_name == "first_turn_greeting"
    assert d.final_move == "acknowledge_and_continue"
    assert d.canned_response is not None
    # Greeting mentions SSM scope -- locking the SSM-only product framing
    # so a future greeting edit can't drop it without breaking the test.
    assert "Sault Ste. Marie" in d.canned_response or "SSM" in d.canned_response


@pytest.mark.parametrize("count", [1, 2, 5, 50])
def test_first_turn_greeting_does_not_fire_after_first_turn(count):
    """Even on a textbook 'hi', any count > 0 means the gate skips --
    isolates the count check from the content check."""
    d = _is_first_turn_greeting_gate(user_message="hi", message_count=count)
    assert d is None


# Slice 2 review fix: content-aware greeting whitelist.
# These are the openers a real user might type that ARE greetings.
@pytest.mark.parametrize("greeting", [
    "hi",
    "Hi",
    "HI",
    "hi!",
    "hi.",
    "  hi  ",
    "hello",
    "hello!",
    "hey",
    "hey there",
    "Hi there",
    "good morning",
    "Good Morning!",
    "good afternoon",
    "good evening",
    "morning",
    "howdy",
    "yo",
    "how are you",
    "how's it going",
    "help",
    "help me",
    "can you help",
    "can you help me",
    "Can you help me?",
    "i need help",
    "please help",
])
def test_first_turn_greeting_fires_on_whitelist_phrases(greeting):
    """Whitelist match after normalization. Case, trailing punctuation,
    and surrounding whitespace are all tolerated. Anything substantive
    beyond the bare phrase falls through (see negative cases below)."""
    d = _is_first_turn_greeting_gate(user_message=greeting, message_count=0)
    assert d is not None, f"Expected greeting gate to fire on {greeting!r}"
    assert d.gate_name == "first_turn_greeting"


# These are the openers a real user might type that ARE NOT greetings --
# they carry job intent and must route to the planner. This is the
# specific regression the reviewer caught in Slice 2: the prior
# implementation fired the greeting gate on any non-empty first-turn
# message, which would short-circuit "I'm looking for warehouse work"
# into the canned 'What kind of work are you looking for?' reply.
@pytest.mark.parametrize("opener", [
    "software developer",
    "I'm looking for a warehouse job",
    "I need full-time admin work",
    "truck and coach technician",
    "warehouse manager",
    "electrical journeyman",
    "looking for retail",
    "hi I'm looking for warehouse work",  # greeting + intent -- planner decides
    "hello can you find me a job",         # ditto
    "any jobs near me",
    "what jobs are there in SSM",
    "?",
    "what",
])
def test_first_turn_greeting_does_not_fire_on_job_intent(opener):
    """Slice 2 review-fix regression test. The greeting gate must NOT
    fire when the first-turn message carries routable content -- even
    if it starts with 'hi'. The planner is responsible for these turns."""
    d = _is_first_turn_greeting_gate(user_message=opener, message_count=0)
    assert d is None, (
        f"Expected greeting gate to PASS THROUGH on {opener!r} "
        f"(planner handles job intent on first turn)"
    )


# ---------------------------------------------------------------------------
# Dispatcher precedence (empty -> resume -> first-turn)
# ---------------------------------------------------------------------------
def test_dispatcher_returns_none_when_no_gate_fires():
    """Standard chat turn: non-empty message, no upload, not first turn.
    Routes through to the planner."""
    d = evaluate_gates(
        user_message="show me jobs",
        uploaded_file=False,
        message_count=3,
    )
    assert d is None


def test_dispatcher_empty_input_fires_over_first_turn():
    """An empty message on the user's very first turn IS empty input,
    not a greeting. Gate 1 wins over gate 3."""
    d = evaluate_gates(
        user_message="",
        uploaded_file=False,
        message_count=0,
    )
    assert d is not None
    assert d.gate_name == "empty_input"


def test_dispatcher_resume_upload_fires_over_first_turn():
    """The crucial compound case from §4 of the design doc: a user
    opens the chat and immediately uploads a resume. The greeting
    gate would route to a generic welcome; the resume gate routes to
    confirm_resume_summary. Resume gate must win."""
    d = evaluate_gates(
        user_message="here's my cv",
        uploaded_file=True,
        message_count=0,
    )
    assert d is not None
    assert d.gate_name == "resume_upload"


def test_dispatcher_resume_upload_fires_over_empty_input():
    """Upload with no caption: the file IS the content. Resume gate
    wins, not empty-input gate."""
    d = evaluate_gates(
        user_message="",
        uploaded_file=True,
        message_count=2,
    )
    assert d is not None
    assert d.gate_name == "resume_upload"


def test_dispatcher_first_turn_greeting_fires_when_alone():
    """First turn with a bare greeting and no upload -> greeting wins
    by elimination (gates 1 + 2 don't fire, gate 3 whitelist-matches)."""
    d = evaluate_gates(
        user_message="hi",
        uploaded_file=False,
        message_count=0,
    )
    assert d is not None
    assert d.gate_name == "first_turn_greeting"


def test_dispatcher_first_turn_with_job_intent_passes_through_to_planner():
    """Slice 2 review-fix regression test at the dispatcher level.

    Previously, ANY non-empty first-turn message fired the greeting
    gate. That meant 'I'm looking for warehouse manager work' would
    return the canned 'What kind of work are you looking for?' reply
    -- a regression the user (rightly) flagged.

    With the content-aware whitelist on gate 3, this message now
    falls through to the planner, where the truth-summary + LLM call
    can extract 'warehouse manager' as the target role and route to
    proceed_to_match or ask_one_clarifying_question as appropriate."""
    d = evaluate_gates(
        user_message="I'm looking for warehouse manager work",
        uploaded_file=False,
        message_count=0,
    )
    assert d is None


# ---------------------------------------------------------------------------
# Compound: empty input + upload + first turn (the kitchen-sink case)
# ---------------------------------------------------------------------------
def test_dispatcher_resume_wins_in_full_compound_case():
    """Empty text + upload + first turn -- all three gates would
    individually fire if standalone. Per the ordering rule, gate 1
    (empty) checks first but short-circuits on uploaded_file=True,
    then gate 2 (resume upload) fires. Gate 3 never runs."""
    d = evaluate_gates(
        user_message="",
        uploaded_file=True,
        message_count=0,
    )
    assert d is not None
    assert d.gate_name == "resume_upload"


# ---------------------------------------------------------------------------
# Structural invariant: three gates, no more
# ---------------------------------------------------------------------------
def test_only_three_gate_functions_exported():
    """Discipline test from design doc §4: 'That's it. Three. No more.'
    If a contributor adds a fourth gate function, this test catches it
    as a structural smell that warrants a design conversation -- adding
    gates undermines the planner-default principle that orchestration
    v2 is built on.

    The test counts gate functions in the module by naming convention
    (`_is_*_gate`). Adjust the expected count only after the design
    doc has been updated to justify the new gate."""
    from skillbridge.chat import gates as gates_module
    gate_funcs = [
        name for name in dir(gates_module)
        if name.startswith("_is_") and name.endswith("_gate")
        and callable(getattr(gates_module, name))
    ]
    assert len(gate_funcs) == 3, (
        f"Expected exactly 3 gate functions; found {len(gate_funcs)}: "
        f"{gate_funcs}. If you intentionally added a gate, update the "
        f"design doc and bump this count -- but read §4 first to make "
        f"sure you're not rebuilding the state machine."
    )


# ---------------------------------------------------------------------------
# Shape contract: GateDecision fields
# ---------------------------------------------------------------------------
def test_gate_decision_has_expected_fields():
    """Pin the GateDecision dataclass shape. Downstream consumers
    (Slice 6 handler integration) depend on these fields."""
    d = GateDecision(
        final_move="confirm_resume_summary",
        gate_name="resume_upload",
    )
    assert d.final_move == "confirm_resume_summary"
    assert d.gate_name == "resume_upload"
    assert d.canned_response is None
    assert d.ask_slot is None
    assert d.tone == "warm_supportive"
