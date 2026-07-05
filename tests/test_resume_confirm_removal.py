"""Resume-confirm gate removal (2026-06-29) -- pin the new behavior:

  - _present_resume_facts_fallback NEVER ends with a confirmation
    question ("does that look right?", "anything I missed?", etc.).
  - When target_role_text is missing/empty: closes with
    "What kind of work are you looking for right now?"
  - When target_role_text is set: closes with no question; user
    drives the next turn.

The LLM-driven path is governed by the confirm_resume_summary section
in RECOMMENDER_RESPONDER_PROMPT / responder prompt. The deterministic
fallback is the focus of these tests -- they pin both the negative
("never these phrases") and positive ("conditional close") behavior.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from skillbridge.chat.responder import _present_resume_facts_fallback

pytestmark = pytest.mark.nodb


@dataclass
class _FakeInp:
    """ResponderInput stub. Only the fields _present_resume_facts_fallback
    reads are present."""
    resume_facts: dict[str, Any] | None
    target_role_text: str | None = None


FORBIDDEN_CONFIRMATION_PHRASES = (
    "does that look right",
    "did i get that right",
    "anything i missed",
    "got wrong",
    "is that all correct",
    "is that correct",
    "did i miss",
)


def _assert_no_confirmation_phrases(text: str) -> None:
    """Assert the resume-parsed summary does NOT contain any of the
    legacy confirmation prompts. This is the locked behavior for
    2026-06-29: the system never asks the user to validate parsed
    facts."""
    lowered = text.lower()
    for phrase in FORBIDDEN_CONFIRMATION_PHRASES:
        assert phrase not in lowered, (
            f"forbidden confirmation phrase {phrase!r} appeared in "
            f"resume-parsed summary: {text!r}"
        )


def test_fallback_drops_confirmation_question_when_target_missing():
    """No target_role_text -> conditional question is the next-step
    coaching prompt 'what kind of work...', NOT 'does that look right'."""
    facts = {
        "work_history": [
            {"title": "Bookkeeper", "employer": "Algoma Family Services",
             "start_year": 2021, "is_current": True},
        ],
        "education": [
            {"credential": "Accounting Diploma", "institution": "Sault College"},
        ],
        "skills": [{"name": "QuickBooks"}, {"name": "Excel"}],
    }
    inp = _FakeInp(resume_facts=facts, target_role_text=None)
    text = _present_resume_facts_fallback(inp)
    _assert_no_confirmation_phrases(text)
    assert "What kind of work are you looking for" in text


def test_fallback_drops_confirmation_question_when_target_blank():
    """Empty/whitespace target_role_text treated as missing -> ask
    the coaching question."""
    facts = {
        "work_history": [
            {"title": "Bookkeeper", "employer": "Algoma",
             "start_year": 2021, "is_current": True},
        ],
        "skills": [{"name": "Excel"}],
    }
    inp = _FakeInp(resume_facts=facts, target_role_text="   ")
    text = _present_resume_facts_fallback(inp)
    _assert_no_confirmation_phrases(text)
    assert "What kind of work" in text


def test_fallback_no_question_when_target_already_set():
    """When the user already named a target before uploading, the
    summary closes with no question. User drives the next turn."""
    facts = {
        "work_history": [
            {"title": "Bookkeeper", "employer": "Algoma",
             "start_year": 2021, "is_current": True},
        ],
        "education": [
            {"credential": "Accounting Diploma", "institution": "Sault College"},
        ],
        "skills": [{"name": "QuickBooks"}],
    }
    inp = _FakeInp(resume_facts=facts, target_role_text="accounting clerk")
    text = _present_resume_facts_fallback(inp)
    _assert_no_confirmation_phrases(text)
    # No "what kind of work" question either.
    assert "What kind of work" not in text
    # Brief ack still has the facts.
    assert "Bookkeeper" in text
    assert "Algoma" in text


def test_fallback_summary_still_has_facts():
    """The summary itself is unchanged: title + employer + year,
    credential + institution, top skills."""
    facts = {
        "work_history": [
            {"title": "Accounts Clerk", "employer": "Sault Steel",
             "start_year": 2018, "end_year": 2021},
        ],
        "education": [
            {"credential": "Accounting Diploma",
             "institution": "Sault College"},
        ],
        "skills": [
            {"name": "QuickBooks"},
            {"name": "Excel"},
            {"name": "payroll"},
        ],
    }
    inp = _FakeInp(resume_facts=facts, target_role_text=None)
    text = _present_resume_facts_fallback(inp)
    assert "Accounts Clerk" in text
    assert "Sault Steel" in text
    assert "2018-2021" in text or "2018" in text
    assert "Accounting Diploma" in text
    assert "Sault College" in text
    assert "QuickBooks" in text
    assert "Excel" in text


def test_fallback_empty_facts_falls_back_to_neutral_prompt():
    """When parsed facts are empty (the LLM gave us nothing useful),
    the fallback bails to a neutral 'tell me about your background'
    prompt. This path is unchanged by the confirm-gate removal."""
    inp = _FakeInp(resume_facts={}, target_role_text=None)
    text = _present_resume_facts_fallback(inp)
    assert "couldn't pull much" in text
    _assert_no_confirmation_phrases(text)


def test_fallback_current_job_renders_year_range_correctly():
    """is_current=True yields '2021-present'."""
    facts = {
        "work_history": [
            {"title": "Bookkeeper", "employer": "Algoma",
             "start_year": 2021, "is_current": True},
        ],
        "skills": [{"name": "Excel"}],
    }
    inp = _FakeInp(resume_facts=facts, target_role_text=None)
    text = _present_resume_facts_fallback(inp)
    assert "2021-present" in text


# ===========================================================================
# Prompt-level lock: the LLM prompt section instructs against
# confirmation questions.
# ===========================================================================
def test_prompt_confirm_resume_summary_forbids_confirmation_questions():
    """The 'confirm_resume_summary' section of the responder prompt
    must explicitly instruct the LLM NOT to ask the user to validate
    parsed facts. Without this guard the LLM will default to the
    common 'does that look right?' pattern."""
    from skillbridge.chat.prompts import RECOMMENDER_RESPONDER_PROMPT
    # The locked design instructs against confirmation questions in
    # the prompt body for the present-resume-facts move. We don't
    # source this from RECOMMENDER_RESPONDER_PROMPT because that's
    # the recommender-mode prompt; the present-resume-facts prompt
    # body lives in the main responder prompt. Read it directly.
    import skillbridge.chat.prompts as prompts_mod
    # Grab the module text for a string-level assertion.
    import inspect
    src = inspect.getsource(prompts_mod)
    assert "confirm_resume_summary" in src
    # The 2026-06-29 update instructs against confirmation questions.
    assert "do NOT ask the user to confirm" in src or \
        "Do NOT ask the user to confirm" in src or \
        "do NOT ask the\n    user to confirm" in src.replace("\r", "")


def test_prompt_confirm_resume_summary_has_conditional_close():
    """The locked design specifies a conditional close based on
    TARGET_ROLE being set or missing."""
    import skillbridge.chat.prompts as prompts_mod
    import inspect
    src = inspect.getsource(prompts_mod)
    # The conditional close references the target being missing.
    src_normalized = " ".join(src.split())
    assert "If TARGET_ROLE is missing" in src_normalized or \
        "TARGET_ROLE is missing/empty" in src_normalized
    # And the "what kind of work" branch.
    assert "What kind of work are you looking for" in src_normalized


# ===========================================================================
# CRITICAL: assert the ACTIVE prompt (NEXT_ACTION_RESPONDER_PROMPT) is fixed.
#
# Code review 2026-06-29 caught that the resume-confirm fix was first applied
# to OUTCOME_RESPONDER_PROMPT (a SECONDARY surface) -- the ACTIVE path for
# resume-upload turns is compose_reply(...) which uses
# NEXT_ACTION_RESPONDER_PROMPT. The earlier tests passed because they used
# inspect.getsource(prompts_mod) which spans ALL prompts in the module, so
# OUTCOME_RESPONDER_PROMPT's correct text masked NEXT_ACTION_RESPONDER_PROMPT
# still having the old confirmation instruction.
#
# These tests check NEXT_ACTION_RESPONDER_PROMPT SPECIFICALLY, so an old/new
# divergence between the two prompt surfaces would fail the test
# immediately.
# ===========================================================================
def test_active_prompt_next_action_responder_drops_confirmation_instruction():
    """The ACTIVE prompt used by compose_reply for ACTION_PRESENT_RESUME_FACTS
    must NOT contain the old instruction to ask a confirmation question.

    The old instruction was:
        ask one short question like "does that look right?" or
        "anything I missed or got wrong?"
    """
    from skillbridge.chat.prompts import NEXT_ACTION_RESPONDER_PROMPT
    # The instruction text itself is gone.
    assert "ask one short question like" not in NEXT_ACTION_RESPONDER_PROMPT
    # The phrasing "matching comes after the user confirms" (which presumed
    # a confirmation turn) is also gone -- there is no longer a "confirms"
    # step the matching waits on.
    # (We do NOT assert quoted-forbidden phrases like "does that look
    # right?" are absent, because the new guard quotes them in a negative
    # instruction: NEVER end with "does that look right?". That literal
    # string still appears in the prompt body, but inside a "DON'T" list.)


def test_active_prompt_next_action_responder_has_resume_confirm_gate_removal_note():
    """The 2026-06-29 fix is anchored in a dated note in the prompt body
    so future readers can find why the confirmation instruction was
    removed."""
    from skillbridge.chat.prompts import NEXT_ACTION_RESPONDER_PROMPT
    assert "resume-confirm gate removal" in NEXT_ACTION_RESPONDER_PROMPT


def test_active_prompt_next_action_responder_has_conditional_close():
    """The ACTIVE prompt has the target-conditional close: ask 'what kind
    of work?' when TARGET_ROLE is missing, no question when set."""
    from skillbridge.chat.prompts import NEXT_ACTION_RESPONDER_PROMPT
    body = NEXT_ACTION_RESPONDER_PROMPT
    # Normalize across line wraps.
    normalized = " ".join(body.split())
    assert "TARGET_ROLE is missing" in normalized
    assert "What kind of work are you looking for" in normalized
    # When target is set: no question (the "no question" intent is
    # expressed via "no question" or "stop").
    assert "no question" in normalized.lower()


def test_active_prompt_next_action_responder_explicitly_forbids_old_phrases():
    """The ACTIVE prompt has a NEVER-end-with list of the legacy
    confirmation phrases so the LLM has explicit pressure against
    defaulting to them from training."""
    from skillbridge.chat.prompts import NEXT_ACTION_RESPONDER_PROMPT
    normalized = " ".join(NEXT_ACTION_RESPONDER_PROMPT.split())
    # The explicit NEVER list mentions at least these well-known
    # phrases so we know the LLM has been told NOT to use them.
    assert "does that look right" in normalized
    assert "anything I missed" in normalized
    # And the framing is negative ("NEVER end with").
    assert "NEVER end with" in NEXT_ACTION_RESPONDER_PROMPT


# ===========================================================================
# TARGET_ROLE prompt/user-block contract (2026-07-05 fix):
#
# The ACTIVE prompt used by compose_reply for ACTION_PRESENT_RESUME_FACTS
# is NEXT_ACTION_RESPONDER_PROMPT. Its conditional close depends on
# whether TARGET_ROLE appears in the user block sent alongside the prompt.
#
# Before the 2026-07-05 fix, _build_user_block() did not serialize
# TARGET_ROLE into the block. The LLM was told "if TARGET_ROLE is set,
# no question" but never received the field to check -- so it always
# fell into the "TARGET_ROLE missing/empty -> ask what kind of work"
# branch even for users who had already stated their target 1-2 turns
# earlier, and made the resume-review turn feel like a state reset.
#
# The existing tests in this file cover the deterministic fallback and
# the prompt text; they do NOT verify the user block content. These
# tests close that gap.
# ===========================================================================


def _make_resume_review_decision():
    """Build the Decision object that handler.py:6048 constructs on a
    resume-upload turn: STATE_RESUME_REVIEW + ACTION_PRESENT_RESUME_FACTS
    with show_matches=False (no engine results to render on this turn)."""
    from skillbridge.chat import intake_state
    return intake_state.Decision(
        next_state=intake_state.STATE_RESUME_REVIEW,
        action=intake_state.ACTION_PRESENT_RESUME_FACTS,
        ask_slots=[],
        show_matches=False,
    )


def _make_responder_input(target_role_text=None):
    """Build a minimal ResponderInput mirroring the resume-review call
    site at handler.py:6056-6066. Only fields _build_user_block actually
    reads on this path are non-default."""
    from skillbridge.chat.responder import ResponderInput
    return ResponderInput(
        user_message="",
        decision=_make_resume_review_decision(),
        results=[],
        training_by_job={},
        next_skill=(None, 0),
        band_signal="none",
        requires_consent=True,
        target_role_text=target_role_text,
        resume_facts={
            "work_history": [
                {"title": "Registered Nurse", "employer": "SAH",
                 "start_year": 2020, "is_current": True},
            ],
            "skills": [{"name": "patient assessment"}],
        },
    )


def test_build_user_block_emits_target_role_when_set_on_resume_review():
    """The 2026-07-05 fix: when the resume-upload path calls compose_reply
    with target_role_text set, _build_user_block MUST include a
    'TARGET_ROLE: <text>' line in the block sent to the LLM.

    Without this, the LLM's conditional close in NEXT_ACTION_RESPONDER_PROMPT
    ('if TARGET_ROLE is set -> no question') never fires because the
    field it checks is never in the block."""
    from skillbridge.chat.responder import (
        _build_user_block,
        build_sanitized_responder_view_v1,
    )
    inp = _make_responder_input(target_role_text="acute care nursing")
    view = build_sanitized_responder_view_v1(inp)
    block = _build_user_block(inp, view)
    assert "TARGET_ROLE: acute care nursing" in block, (
        "_build_user_block did not emit TARGET_ROLE for a resume-review "
        "turn with target set. This is the exact prompt/block wire gap "
        f"that caused the 2026-07-05 bug. Block was:\n{block}"
    )


def test_build_user_block_omits_target_role_when_missing_on_resume_review():
    """Mirror: when target_role_text is None, no TARGET_ROLE line is
    emitted. This preserves the prompt's 'missing/empty -> ask what
    kind of work' branch and prevents anyone from later 'fixing' the
    emit to unconditionally include the label with an empty value
    (which would suppress the coaching question incorrectly)."""
    from skillbridge.chat.responder import (
        _build_user_block,
        build_sanitized_responder_view_v1,
    )
    inp = _make_responder_input(target_role_text=None)
    view = build_sanitized_responder_view_v1(inp)
    block = _build_user_block(inp, view)
    assert "TARGET_ROLE" not in block, (
        "_build_user_block emitted TARGET_ROLE with no target set; the "
        "prompt's 'missing/empty -> ask what kind of work' branch relies "
        f"on the label being absent. Block was:\n{block}"
    )


def test_prompt_block_contract_target_role_referenced_by_prompt_is_emitted_by_block():
    """Narrow contract lock: as long as NEXT_ACTION_RESPONDER_PROMPT
    (the ACTIVE prompt for resume-review turns) references TARGET_ROLE
    in its conditional close, _build_user_block MUST emit TARGET_ROLE
    when the input has it set.

    This is deliberately scoped to the ONE prompt/block pair that this
    bug lived in, not a generic prompt-parser framework. The invariant
    is: prompt-says-check-X must imply block-emits-X. If the prompt is
    ever rewritten to no longer reference TARGET_ROLE, the first
    assertion here fails loudly and a future editor can remove or
    refit this test."""
    from skillbridge.chat.prompts import NEXT_ACTION_RESPONDER_PROMPT
    from skillbridge.chat.responder import (
        _build_user_block,
        build_sanitized_responder_view_v1,
    )
    # Prompt side: as long as TARGET_ROLE is referenced in the active
    # prompt, the block-side invariant applies.
    prompt_normalized = " ".join(NEXT_ACTION_RESPONDER_PROMPT.split())
    assert "TARGET_ROLE" in prompt_normalized, (
        "NEXT_ACTION_RESPONDER_PROMPT no longer references TARGET_ROLE. "
        "If the conditional close was intentionally removed, this "
        "contract test should be removed or updated. If it was removed "
        "by accident, the prompt/block invariant is broken."
    )
    # Block side: since the prompt references TARGET_ROLE, the block
    # sent to the LLM must contain it when the input has it set.
    inp = _make_responder_input(target_role_text="acute care nursing")
    view = build_sanitized_responder_view_v1(inp)
    block = _build_user_block(inp, view)
    assert "TARGET_ROLE: acute care nursing" in block, (
        "The active prompt references TARGET_ROLE (verified above), but "
        "_build_user_block did not emit it. This is the exact prompt/"
        "block wire gap that caused the resume-upload responder to "
        "ask 'What kind of work are you looking for?' after the user "
        f"had already stated their target. Block was:\n{block}"
    )
