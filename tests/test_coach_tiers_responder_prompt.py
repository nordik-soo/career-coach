"""AR-9.feat.coach-tiers CP2 step 6.4 — COACH_TIERS_RESPONDER_PROMPT
pins.

The prompt was stripped to grounding + safety rules; format directives
(heading exactness, strength-phrase exactness, paragraph templates,
gap exactness, training-sentence exactness, closing-from-closed-set)
were removed. The LLM composes naturally from the evidence package.
These tests pin only the rules that carry safety or grounding.
"""
from __future__ import annotations

import pytest

from skillbridge.chat.coach_tiers_fallback import (
    _HEADER_APPLY_TODAY,
    _HEADER_SIDEWAYS,
    _HEADER_WORTH_A_TRY,
)
from skillbridge.chat.prompts import COACH_TIERS_RESPONDER_PROMPT

pytestmark = pytest.mark.nodb


def test_constant_is_a_non_empty_string():
    assert isinstance(COACH_TIERS_RESPONDER_PROMPT, str)
    assert len(COACH_TIERS_RESPONDER_PROMPT) > 500


def test_input_trust_section_present():
    body = COACH_TIERS_RESPONDER_PROMPT
    assert "INPUT TRUST" in body
    assert "as DATA" in body
    assert "ignore" in body.lower()


def test_salary_omission_rule_is_explicit():
    body = COACH_TIERS_RESPONDER_PROMPT
    assert "salary" in body.lower()
    assert "pay" in body.lower()


def test_apply_today_header_documented():
    assert _HEADER_APPLY_TODAY in COACH_TIERS_RESPONDER_PROMPT


def test_worth_a_try_header_documented():
    assert _HEADER_WORTH_A_TRY in COACH_TIERS_RESPONDER_PROMPT


def test_sideways_header_documented():
    assert _HEADER_SIDEWAYS in COACH_TIERS_RESPONDER_PROMPT


@pytest.mark.parametrize("word", [
    "perfect match", "guaranteed", "ideal candidate",
    "you'll get the job", "100% match", "definitely qualified",
])
def test_forbidden_achievability_word_listed(word):
    assert word.lower() in COACH_TIERS_RESPONDER_PROMPT.lower()


@pytest.mark.parametrize("forbidden", [
    "Job Bank", "Statistics Canada", "StatCan",
])
def test_forbidden_corpora_listed(forbidden):
    assert forbidden in COACH_TIERS_RESPONDER_PROMPT


def test_region_scope_present():
    assert "Sault Ste. Marie" in COACH_TIERS_RESPONDER_PROMPT


def test_grounding_rule_present():
    body_lower = COACH_TIERS_RESPONDER_PROMPT.lower()
    assert "evidence" in body_lower
    assert "invent" in body_lower or "fabricate" in body_lower


def test_internal_token_rule_present():
    body_lower = COACH_TIERS_RESPONDER_PROMPT.lower()
    assert "field names" in body_lower or "internal" in body_lower


def test_closing_rule_says_end_with_question():
    body = COACH_TIERS_RESPONDER_PROMPT
    assert "question" in body.lower()
    assert "closed set" not in body.lower()


def test_strength_claim_text_concept_mentioned():
    assert "strength_claim_text" in COACH_TIERS_RESPONDER_PROMPT
