"""Structural pins for the EVIDENCE_BOUND_EXTRACTOR_PROMPT after the
target-role-extraction guidance added 2026-06-22 (post-live-test).

These tests don't invoke the LLM -- they assert the prompt contains
the rules and examples that drive the new behavior. Live verification
of actual extraction quality requires LLM calls and lives in the
manual verify steps documented in the project memo.
"""
from __future__ import annotations

import pytest

from skillbridge.chat.prompts import EVIDENCE_BOUND_EXTRACTOR_PROMPT

pytestmark = pytest.mark.nodb


def test_extractor_prompt_has_target_role_extraction_section():
    """The new section must exist by name so the LLM reads it as a
    distinct rule before the EVIDENCE RULES general exclusion."""
    assert "TARGET ROLE EXTRACTION" in EVIDENCE_BOUND_EXTRACTOR_PROMPT


def test_extractor_prompt_section_precedes_general_question_exclusion():
    """The new section must appear BEFORE the general 'asking a
    question' exclusion in the EVIDENCE RULES so the LLM reads the
    target-role override first."""
    body = EVIDENCE_BOUND_EXTRACTOR_PROMPT
    target_idx = body.index("TARGET ROLE EXTRACTION")
    evidence_idx = body.index("EVIDENCE RULES")
    assert target_idx < evidence_idx


@pytest.mark.parametrize("pattern_phrase", [
    "show me accounting clerk jobs",
    "find me admin work",
    "any nursing roles",
    "looking for retail openings",
    "interested in trades",
])
def test_extractor_prompt_lists_command_request_target_examples(pattern_phrase):
    """The prompt must include concrete examples of command/request
    shapes that name a target work area. These are the patterns the
    LLM was previously missing (see Bug 4 in the peer-engine memo)."""
    assert pattern_phrase in EVIDENCE_BOUND_EXTRACTOR_PROMPT


@pytest.mark.parametrize("pattern_phrase", [
    "are there welding jobs in Sault?",
    "what construction roles are open?",
])
def test_extractor_prompt_lists_question_naming_target_examples(pattern_phrase):
    """Questions that name a target work area must still extract
    target_role_text. The prompt has explicit examples to teach this."""
    assert pattern_phrase in EVIDENCE_BOUND_EXTRACTOR_PROMPT


@pytest.mark.parametrize("omit_phrase", [
    "show me jobs",
    "show me a job",
    "any good jobs",
    "what's hiring",
])
def test_extractor_prompt_lists_no_target_omit_examples(omit_phrase):
    """Messages without a named work area must NOT extract target.
    The prompt has explicit omit examples so the LLM doesn't
    over-extract."""
    assert omit_phrase in EVIDENCE_BOUND_EXTRACTOR_PROMPT


@pytest.mark.parametrize("omit_phrase", [
    "what should I improve?",
    "what training should I take?",
    "what gaps do I have?",
])
def test_extractor_prompt_lists_improvement_omit_examples(omit_phrase):
    """Learning / improvement / gap questions must NOT extract
    target_role_text -- they are intent questions, not target
    declarations. Without these omit examples, the LLM might
    over-extract."""
    assert omit_phrase in EVIDENCE_BOUND_EXTRACTOR_PROMPT


def test_extractor_prompt_question_exclusion_calls_out_target_override():
    """The EVIDENCE RULES section's question-exclusion clause must
    explicitly note the target-role override so the LLM doesn't read
    the two rules as contradictory."""
    body = EVIDENCE_BOUND_EXTRACTOR_PROMPT
    # The question exclusion line must reference the new section.
    assert (
        "TARGET ROLE EXTRACTION" in body
        and ("EXCEPT" in body or "exception" in body.lower()
             or "EXCEPT when" in body)
    )
