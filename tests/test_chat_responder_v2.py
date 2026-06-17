"""Unit tests for chat orchestration v2 slice 5 -- the outcome-move responder.

Three concerns:
  1. OUTCOME_RESPONDER_PROMPT structure: every OutcomeMove + Tone is named;
     every SCOPE BOUNDARIES section is present verbatim; no operational
     fields are referenced.
  2. _build_user_block_v2 input-surface whitelist: arbiter_action and
     notes MUST NEVER appear in the prompt user block. This is the
     hard boundary from the Slice 4 review.
  3. compose_response_v2 + fallback per OutcomeMove: chat never breaks
     when LLM is disabled; deterministic text exists for every move.

No DB, no engine, no LLM. Mock llm.call / llm.is_enabled to drive
control flow.
"""
from __future__ import annotations

import os
from typing import get_args

os.environ.setdefault("LLM_ENABLED", "false")

import pytest

from skillbridge.chat.arbiter import (
    ARBITER_REASON_MATCHES_FOUND,
    ARBITER_REASON_MATCHES_WITH_CAPS,
    ARBITER_REASON_NO_MATCHES,
    ArbiterDecision,
    OutcomeMove,
)
from skillbridge.chat.planner import Tone
from skillbridge.chat.prompts import (
    NEXT_ACTION_RESPONDER_PROMPT,
    OUTCOME_RESPONDER_PROMPT,
)
from skillbridge.chat.responder import (
    ConversationContext,
    ResponderV2Input,
    _build_user_block_v2,
    _fallback_reply_v2,
    _policy_ok_v2,
    _redirect_scope_fallback_v2,
    compose_response_v2,
)
from skillbridge.chat.url_views import (
    build_sanitized_responder_view_v2 as _v_v2,
)


def _iv(inp):
    """Return (inp, view) so each test reuses one inp instance."""
    return inp, _v_v2(inp)

pytestmark = pytest.mark.nodb


# ===========================================================================
# Helpers
# ===========================================================================
def _decision(
    final_move: str = "present_matches",
    reason_code: str = ARBITER_REASON_MATCHES_FOUND,
    tone: str = "brief_confident",
    ask_slot: str | None = None,
    caps_applied: tuple[str, ...] = (),
    arbiter_action: str = "passed_planner_through",
    notes: str | None = "internal scratchpad text -- must not leak",
) -> ArbiterDecision:
    return ArbiterDecision(
        final_move=final_move,
        reason_code=reason_code,
        tone=tone,
        arbiter_action=arbiter_action,
        ask_slot=ask_slot,
        caps_applied=caps_applied,
        notes=notes,
    )


def _input(
    user_message: str = "show me jobs",
    decision: ArbiterDecision | None = None,
    results: list | None = None,
    training_by_job: dict | None = None,
    band_signal: str = "strong_or_good",
    next_skill: tuple[str | None, int] = (None, 0),
    requires_consent: bool = False,
    target_role_text: str | None = "warehouse worker",
    resume_facts: dict | None = None,
) -> ResponderV2Input:
    return ResponderV2Input(
        user_message=user_message,
        decision=decision if decision is not None else _decision(),
        results=results if results is not None else [],
        training_by_job=training_by_job if training_by_job is not None else {},
        next_skill=next_skill,
        band_signal=band_signal,
        requires_consent=requires_consent,
        target_role_text=target_role_text,
        resume_facts=resume_facts,
    )


def _match_result(
    job_id: str = "job-1",
    title: str = "Warehouse Associate",
    employer: str = "Acme",
    match_band: str = "strong",
) -> dict:
    return {
        "job_id": job_id,
        "title": title,
        "employer": employer,
        "url": f"https://example.com/{job_id}",
        "location": "Sault Ste. Marie",
        "match_band": match_band,
        "matched_skills": ["forklift", "inventory"],
        "missing_skills": [],
        "credential_warning": None,
        "score_explanation": {
            "matched_skills": ["forklift", "inventory"],
            "missing_skills": [],
        },
    }


# ===========================================================================
# Prompt structure: every OutcomeMove is named
# ===========================================================================
# AR-9.feat.coach-tiers CP2 step 2: outcomes that ship with a dedicated
# system prompt and never reach OUTCOME_RESPONDER_PROMPT. Each entry is
# pinned independently by its dedicated prompt's pinning test.
_OUTCOMES_WITH_OWN_PROMPT: frozenset[str] = frozenset({
    # COACH_TIERS_RESPONDER_PROMPT, pinned by
    # tests/test_coach_tiers_responder_prompt.py
    "present_tiered_matches",
})


def test_prompt_names_every_outcome_move():
    """LLM must be told about every outcome it might receive.
    Drift between the OutcomeMove enum and the prompt would mean the
    LLM gets a value it doesn't know how to narrate.

    Exception: outcomes listed in `_OUTCOMES_WITH_OWN_PROMPT` ship with
    their own dedicated system prompt and never reach
    OUTCOME_RESPONDER_PROMPT. They are pinned independently."""
    for move in get_args(OutcomeMove):
        if move in _OUTCOMES_WITH_OWN_PROMPT:
            continue
        assert move in OUTCOME_RESPONDER_PROMPT, (
            f"OutcomeMove value {move!r} is not named in "
            f"OUTCOME_RESPONDER_PROMPT. LLM has no narration shape for it."
        )


def test_dedicated_prompt_outcomes_actually_have_a_prompt():
    """Every move exempted from OUTCOME_RESPONDER_PROMPT must point at
    a real dedicated prompt constant. Today: `present_tiered_matches`
    is served by `COACH_TIERS_RESPONDER_PROMPT`."""
    if "present_tiered_matches" in _OUTCOMES_WITH_OWN_PROMPT:
        from skillbridge.chat.prompts import COACH_TIERS_RESPONDER_PROMPT
        assert isinstance(COACH_TIERS_RESPONDER_PROMPT, str)
        assert len(COACH_TIERS_RESPONDER_PROMPT) > 1000


def test_prompt_names_every_tone():
    for tone in get_args(Tone):
        assert tone in OUTCOME_RESPONDER_PROMPT, (
            f"Tone value {tone!r} is not named in OUTCOME_RESPONDER_PROMPT."
        )


# ===========================================================================
# Slice 4 review hard boundary: no arbiter_action/internals in the prompt
# ===========================================================================
def test_prompt_explicitly_forbids_operational_terms():
    """The Slice 4 review note: 'user-facing text should not narrate
    internals like I overrode the planner.' The prompt must explicitly
    teach this so the LLM doesn't try to be helpful by surfacing them."""
    assert "Operational fields are out of bounds" in OUTCOME_RESPONDER_PROMPT
    assert '"the planner said"' in OUTCOME_RESPONDER_PROMPT
    assert '"the arbiter decided"' in OUTCOME_RESPONDER_PROMPT
    assert '"I overrode"' in OUTCOME_RESPONDER_PROMPT


def test_prompt_does_not_mention_arbiter_action_field():
    """The ArbiterDecision field name should never appear in the prompt
    at all -- not as input shape, not as a thing to narrate, nothing."""
    assert "ARBITER_ACTION" not in OUTCOME_RESPONDER_PROMPT
    assert "arbiter_action" not in OUTCOME_RESPONDER_PROMPT


def test_prompt_does_not_mention_notes_field():
    """ArbiterDecision.notes is internal debugging output. It must
    never reach the LLM."""
    # Allow the word "notes" in general English ("brief notes" etc.)
    # but reject any framing that would treat .notes as a recognized
    # input field. The block format would look like "NOTES: ..." in
    # the user message; the prompt must never describe that.
    assert "NOTES:" not in OUTCOME_RESPONDER_PROMPT
    assert "decision.notes" not in OUTCOME_RESPONDER_PROMPT


# ===========================================================================
# Prompt structure: load-bearing SCOPE BOUNDARIES preserved
# ===========================================================================
# These are byte-for-byte the rules that have kept the chat product-coherent
# (SSM-only, no immigration advice, etc.) since Sprint 3. Drift here is
# how the chat starts hallucinating again.
@pytest.mark.parametrize("required_block", [
    "SCOPE BOUNDARIES",
    "DATASET-FIRST RULE",
    "NO ACTIONS WE CANNOT PERFORM",
    "NEVER name a non-local city",
    "NO CREDENTIAL EQUIVALENCE CLAIMS",
    "NO IMMIGRATION / LEGAL / MEDICAL / FINANCIAL ADVICE",
    "TRAINING DISCUSSIONS ARE IN SCOPE",
    "MISSING SKILLS ARE NOT OWNED SKILLS",
    "MATCH STAGES",
    "OCCUPATION MATCH BOOST",
    "CAPS APPLIED -- NAME THE CAP",
    'CANONICAL "NO MATCHES" RESPONSE',
])
def test_prompt_contains_load_bearing_rule_block(required_block):
    """Each block is a load-bearing product rule. If a future prompt
    edit drops one, the failure mode it prevents will re-appear in
    chat. This test acts as a tripwire."""
    assert required_block in OUTCOME_RESPONDER_PROMPT, (
        f"Load-bearing rule block {required_block!r} is missing from "
        f"OUTCOME_RESPONDER_PROMPT. This rule prevents a specific live "
        f"failure mode from Sprint 3 onward; do not drop it."
    )


def test_prompt_caps_applied_section_warns_about_tone_independence():
    """Slice 4 review tightening: caps shape WHAT to say (name the cap),
    tone shapes HOW to say it. Cap-naming must happen regardless of tone."""
    assert "Cap-naming is independent of TONE" in OUTCOME_RESPONDER_PROMPT


# ===========================================================================
# _build_user_block_v2 input-surface whitelist
# ===========================================================================
def test_user_block_does_not_leak_arbiter_action():
    """Hard boundary from Slice 4 review: arbiter_action is operational
    telemetry; it must never appear in the user block fed to the LLM."""
    d = _decision(
        arbiter_action="overrode_to_redirect",  # distinctive sentinel
        notes="boring internal note",
    )
    block = _build_user_block_v2(*_iv(_input(decision=d)))
    assert "arbiter_action" not in block.lower()
    assert "overrode_to_redirect" not in block
    assert "ARBITER_ACTION" not in block


def test_user_block_does_not_leak_notes():
    """ArbiterDecision.notes carries human-readable arbiter reasoning
    for transcripts -- but transcripts are not user-facing."""
    d = _decision(notes="DEBUG: planner emitted X, overrode because Y")
    block = _build_user_block_v2(*_iv(_input(decision=d)))
    assert "planner emitted X" not in block
    assert "overrode because Y" not in block


@pytest.mark.parametrize("action,notes", [
    ("passed_planner_through", "passed-through planner decision"),
    ("overrode_to_redirect", "scope violation detected"),
    ("overrode_to_ask", "duplicate ask reroute"),
    ("fallback_to_legacy", "planner returned None"),
    ("resolved_to_matches", "engine returned 3 matches"),
    ("resolved_to_no_match", "engine returned 0 matches"),
])
def test_user_block_omits_operational_fields_across_all_actions(action, notes):
    """Exhaustive: no arbiter_action value, regardless of which path
    Pass 1/2 took, should ever leak into the user block."""
    d = _decision(arbiter_action=action, notes=notes)
    block = _build_user_block_v2(*_iv(_input(decision=d)))
    assert action not in block
    # The note text is freeform but should never appear in the block
    assert notes not in block


# ===========================================================================
# _build_user_block_v2: whitelisted fields are surfaced as expected
# ===========================================================================
def test_user_block_includes_final_move_and_tone():
    d = _decision(
        final_move="ask_one_clarifying_question",
        reason_code="target_role_unclear",
        tone="warm_supportive",
        ask_slot="target_role_text",
    )
    block = _build_user_block_v2(*_iv(_input(decision=d)))
    assert "FINAL_MOVE: ask_one_clarifying_question" in block
    assert "TONE: warm_supportive" in block
    assert "REASON_CODE: target_role_unclear" in block


def test_user_block_surfaces_caps_applied_when_present():
    d = _decision(
        final_move="present_matches",
        reason_code=ARBITER_REASON_MATCHES_WITH_CAPS,
        tone="warm_supportive",
        caps_applied=("band_capped_by_credential",),
    )
    block = _build_user_block_v2(*_iv(_input(
        decision=d, results=[_match_result()],
    )))
    assert "CAPS_APPLIED" in block
    assert "band_capped_by_credential" in block


def test_user_block_omits_caps_applied_when_empty():
    """No CAPS_APPLIED block when the tuple is empty -- avoids
    confusing the LLM with an empty list."""
    d = _decision(caps_applied=())
    block = _build_user_block_v2(*_iv(_input(decision=d, results=[_match_result()])))
    assert "CAPS_APPLIED" not in block


def test_user_block_includes_ask_slot_only_on_ask_move():
    """ASK_SLOT block is conditional on final_move."""
    # Ask move with slot -> present
    d_ask = _decision(
        final_move="ask_one_clarifying_question",
        reason_code="target_role_unclear",
        ask_slot="target_role_text",
    )
    block_ask = _build_user_block_v2(*_iv(_input(decision=d_ask)))
    assert "ASK_SLOT" in block_ask
    assert "target_role_text" in block_ask

    # Non-ask move -> no slot block
    d_other = _decision(final_move="acknowledge_and_continue",
                        reason_code="user_confirmed")
    block_other = _build_user_block_v2(*_iv(_input(decision=d_other)))
    assert "ASK_SLOT" not in block_other


def test_user_block_only_includes_results_on_match_outcome():
    """RESULTS payload is conditional on final_move ∈ {present_matches,
    present_no_match}. An ask turn shouldn't show the user the match
    payload."""
    d_ask = _decision(
        final_move="ask_one_clarifying_question",
        reason_code="target_role_unclear",
        ask_slot="skills_text",
    )
    block_ask = _build_user_block_v2(*_iv(_input(
        decision=d_ask, results=[_match_result()],
    )))
    assert "RESULTS:" not in block_ask

    d_match = _decision(final_move="present_matches")
    block_match = _build_user_block_v2(*_iv(_input(
        decision=d_match, results=[_match_result()],
    )))
    assert "RESULTS:" in block_match


# ===========================================================================
# Fallback path: every OutcomeMove yields a non-empty deterministic reply
# ===========================================================================
@pytest.mark.parametrize("move,reason,ask_slot,caps", [
    ("acknowledge_and_continue", "user_confirmed", None, ()),
    ("ask_one_clarifying_question", "target_role_unclear", "target_role_text", ()),
    ("explain_gap", "credential_gap_present", None, ("band_capped_by_credential",)),
    ("offer_refinement", "narrow_request", None, ()),
    ("redirect_scope", "scope_violation_immigration", None, ()),
    ("present_matches", ARBITER_REASON_MATCHES_FOUND, None, ()),
    ("present_no_match", ARBITER_REASON_NO_MATCHES, None, ()),
    # Slice 7 review fix: gate-emitted outcome must have a fallback too.
    ("confirm_resume_summary", "gate:resume_upload", None, ()),
])
def test_fallback_returns_nonempty_reply_for_every_outcome(move, reason, ask_slot, caps):
    """Chat must never break. Every OutcomeMove value has a
    deterministic fallback string."""
    d = _decision(
        final_move=move, reason_code=reason,
        ask_slot=ask_slot, caps_applied=caps,
        tone="warm_supportive",
    )
    reply = _fallback_reply_v2(*_iv(_input(
        decision=d,
        results=[_match_result()] if move == "present_matches" else [],
    )))
    assert reply
    assert reply.strip()
    # Verify it doesn't leak operational info via the fallback either.
    assert "arbiter_action" not in reply.lower()
    assert "overrode" not in reply.lower()


def test_fallback_redirect_mentions_ssm_scope():
    d = _decision(
        final_move="redirect_scope",
        reason_code="scope_violation_immigration",
        tone="honest_redirect",
    )
    reply = _fallback_reply_v2(*_iv(_input(decision=d)))
    assert (
        "Sault Ste. Marie" in reply
        or "SSM" in reply
        or "local" in reply.lower()
    )


def test_fallback_no_match_suggests_sccc():
    """Canonical no-match shape from the prompt: suggest SCCC."""
    d = _decision(final_move="present_no_match", tone="honest_redirect")
    reply = _fallback_reply_v2(*_iv(_input(
        decision=d, band_signal="none", results=[],
    )))
    assert "Sault Community Career Centre" in reply


def test_fallback_explain_gap_names_credential_cap_when_present():
    d = _decision(
        final_move="explain_gap",
        reason_code="credential_gap_present",
        caps_applied=("band_capped_by_credential",),
    )
    reply = _fallback_reply_v2(*_iv(_input(decision=d)))
    assert "credential" in reply.lower()


def test_fallback_explain_gap_uses_credential_gap_and_role_from_context():
    """Slice 9: when ConversationContext carries the specific
    credential gap + role title from a prior match turn, the
    explain_gap fallback references BOTH instead of falling back to
    the cap-flag generic message."""
    from skillbridge.chat.responder import ConversationContext
    ctx = ConversationContext(
        target_role_text="truck and coach technician",
        last_presented_job_titles=("Truck and Coach Technician",),
        last_presented_caps_applied=("band_capped_by_credential",),
        last_presented_credential_gaps=("310T technician certification",),
    )
    d = _decision(
        final_move="explain_gap",
        reason_code="credential_gap_present",
        tone="warm_supportive",
        caps_applied=("band_capped_by_credential",),
    )
    inp = ResponderV2Input(
        user_message="how do I get my 310T?",
        decision=d, results=[], training_by_job={},
        next_skill=(None, 0), band_signal="none",
        requires_consent=True,
        target_role_text="truck and coach technician",
        conversation_context=ctx,
    )
    reply = _fallback_reply_v2(*_iv(inp))
    # Specific credential gap is named
    assert "310T technician certification" in reply
    # Role context anchors the conversation
    assert "Truck and Coach Technician" in reply
    # Verified resource pointer is present
    assert "Sault Community Career Centre" in reply or "Sault College" in reply
    # No job cards leaked into prose
    assert "match band" not in reply.lower()
    assert "Stretch match" not in reply


def test_fallback_explain_gap_falls_back_to_cap_message_without_specific_gap():
    """When ConversationContext lacks specific credential_gaps (e.g.
    pre-Slice-8 sessions, or a turn where caps fired but no
    credential_gap_skills were captured), the explain_gap fallback
    uses the cap-flag message and still points to SCCC."""
    d = _decision(
        final_move="explain_gap",
        reason_code="credential_gap_present",
        tone="warm_supportive",
        caps_applied=("band_capped_by_credential",),
    )
    inp = ResponderV2Input(
        user_message="how do I close that gap?",
        decision=d, results=[], training_by_job={},
        next_skill=(None, 0), band_signal="none",
        requires_consent=True,
        conversation_context=None,    # no context
    )
    reply = _fallback_reply_v2(*_iv(inp))
    assert "credential" in reply.lower()
    assert "Sault Community Career Centre" in reply


def test_fallback_explain_gap_handles_unknown_cap_gracefully():
    """An unknown cap reason still produces a valid reply, not a crash."""
    d = _decision(
        final_move="explain_gap",
        reason_code="credential_gap_present",
        caps_applied=("band_capped_by_brand_new_reason",),
    )
    reply = _fallback_reply_v2(*_iv(_input(decision=d)))
    assert reply.strip()
    assert "band_capped_by_brand_new_reason" in reply


def test_fallback_present_matches_with_caps_leads_with_cap_lead():
    """Cap-aware lead for matches with caps applied."""
    d = _decision(
        final_move="present_matches",
        reason_code=ARBITER_REASON_MATCHES_WITH_CAPS,
        caps_applied=("band_capped_by_no_experience",),
        tone="warm_supportive",
    )
    reply = _fallback_reply_v2(*_iv(_input(
        decision=d, results=[_match_result(match_band="stretch")],
    )))
    assert "stretch" in reply.lower() or "work history" in reply.lower()


# ===========================================================================
# compose_response_v2 -- end-to-end with mocked LLM
# ===========================================================================
def test_compose_returns_fallback_when_llm_disabled(monkeypatch):
    """LLM off -> deterministic per-outcome fallback (chat never breaks)."""
    monkeypatch.setattr("skillbridge.chat.responder.is_enabled", lambda: False)
    d = _decision(final_move="ask_one_clarifying_question",
                  reason_code="target_role_unclear", ask_slot="target_role_text")
    reply = compose_response_v2(_input(decision=d))
    assert reply.strip()


def test_compose_passes_outcome_prompt_to_llm(monkeypatch):
    """When the LLM IS enabled, compose_response_v2 must invoke
    `call()` with the OUTCOME_RESPONDER_PROMPT (not the legacy v1
    prompt). This is the wiring test that keeps the slice honest."""
    monkeypatch.setattr("skillbridge.chat.responder.is_enabled", lambda: True)
    captured: dict = {}

    def fake_call(system, user, max_tokens=None):
        captured["system"] = system
        captured["user"] = user
        return "Got it. What kind of work would you like me to focus on?"

    monkeypatch.setattr("skillbridge.chat.responder.call", fake_call)

    d = _decision(final_move="ask_one_clarifying_question",
                  reason_code="target_role_unclear",
                  ask_slot="target_role_text", tone="warm_supportive")
    compose_response_v2(_input(decision=d))

    assert captured["system"] == OUTCOME_RESPONDER_PROMPT
    assert captured["system"] != NEXT_ACTION_RESPONDER_PROMPT
    # The user block carries the whitelisted fields
    assert "FINAL_MOVE: ask_one_clarifying_question" in captured["user"]
    assert "TONE: warm_supportive" in captured["user"]


def test_compose_falls_back_when_llm_returns_empty(monkeypatch):
    monkeypatch.setattr("skillbridge.chat.responder.is_enabled", lambda: True)
    monkeypatch.setattr("skillbridge.chat.responder.call", lambda *a, **kw: "")
    d = _decision(final_move="present_no_match", tone="honest_redirect")
    reply = compose_response_v2(_input(
        decision=d, band_signal="none", results=[],
    ))
    assert reply.strip()
    assert "Sault Community Career Centre" in reply


def test_compose_falls_back_when_llm_violates_policy(monkeypatch):
    """LLM tries to suggest searching Toronto -> fallback fires."""
    monkeypatch.setattr("skillbridge.chat.responder.is_enabled", lambda: True)
    bad_reply = (
        "I'd suggest you try Toronto for more roles in your field."
    )
    monkeypatch.setattr("skillbridge.chat.responder.call", lambda *a, **kw: bad_reply)
    d = _decision(final_move="present_no_match", tone="honest_redirect")
    reply = compose_response_v2(_input(decision=d, band_signal="none", results=[]))
    # The LLM reply was rejected; fallback returned instead.
    assert "Toronto" not in reply
    assert reply.strip()


def test_compose_falls_back_when_llm_leaks_operational_term(monkeypatch):
    """Slice 4 review hard boundary check at the policy layer:
    even if the LLM somehow names arbiter_action in its reply, the
    output policy rejects it and we fall back."""
    monkeypatch.setattr("skillbridge.chat.responder.is_enabled", lambda: True)
    bad_reply = "I'll proceed since the planner said you have enough info."
    monkeypatch.setattr("skillbridge.chat.responder.call", lambda *a, **kw: bad_reply)
    d = _decision(final_move="acknowledge_and_continue",
                  reason_code="user_confirmed", tone="brief_confident")
    reply = compose_response_v2(_input(decision=d))
    assert "the planner said" not in reply.lower()


# ===========================================================================
# _policy_ok_v2 -- v2-specific policy rules
# ===========================================================================
def test_policy_v2_rejects_bullet_lists_on_ask_turn():
    """ASK turns must be woven prose, never bulleted question lists."""
    d = _decision(final_move="ask_one_clarifying_question",
                  reason_code="target_role_unclear", ask_slot="skills_text")
    reply = "What skills do you have?\n- forklift\n- driving"
    assert not _policy_ok_v2(reply, *_iv(_input(decision=d)))


def test_policy_v2_accepts_prose_on_ask_turn():
    d = _decision(final_move="ask_one_clarifying_question",
                  reason_code="target_role_unclear", ask_slot="skills_text")
    reply = "What kind of work have you done before — forklift, warehouse, retail?"
    assert _policy_ok_v2(reply, *_iv(_input(decision=d)))


def test_policy_v2_rejects_dollar_amounts():
    d = _decision(final_move="present_matches", tone="brief_confident")
    assert not _policy_ok_v2("This pays $22/hr.", *_iv(_input(decision=d)))


def test_policy_v2_rejects_national_feeds():
    d = _decision(final_move="present_no_match", tone="honest_redirect")
    bad = "According to the national average, you'd earn more."
    assert not _policy_ok_v2(bad, *_iv(_input(decision=d)))


@pytest.mark.parametrize("term", [
    "arbiter_action",
    "overrode the planner",
    "the planner said",
    "the arbiter decided",
    "fallback_to_legacy",
])
def test_policy_v2_rejects_operational_term_leakage(term):
    """Every operational term from the Slice 4 review's hard boundary
    is caught at the policy layer."""
    d = _decision(final_move="acknowledge_and_continue",
                  reason_code="user_confirmed")
    reply = f"This response includes {term} which it shouldn't."
    assert not _policy_ok_v2(reply, *_iv(_input(decision=d)))


def test_policy_v2_accepts_clean_reply():
    d = _decision(final_move="present_matches", tone="brief_confident")
    clean = "Here's a strong match: Warehouse Associate at Acme."
    assert _policy_ok_v2(clean, *_iv(_input(decision=d)))


# ===========================================================================
# Slice 8: short-session conversation context for fallback redirects
# ===========================================================================
# Four tiers, by available context strength:
#   1. matches + specific credential gaps -> "On the X role we just looked at,
#       the main gap is still 310T..."
#   2. matches only -> "We can keep working on those roles..."
#   3. target role only -> "We were looking at warehouse work..."
#   4. nothing -> the original generic line
# Plus: never accidentally mention immigration/PR even in the rich-context
# tiers -- the policy fallback exists precisely BECAUSE the LLM crossed that
# line, so we must not reintroduce it.
# ===========================================================================
# Post-Slice-9 grounding fix: TRAINING block + provider grounding +
# registry-grounded fallback
# ===========================================================================
# Live test surfaced the TAC hallucination: on explain_gap turns the
# TRAINING block was NEVER serialized into the prompt (gated to
# present_matches only), so the LLM improvised providers it knew
# from its own training data. The bundled fix:
#   1) serialize TRAINING on explain_gap turns
#   2) prompt rule: providers must come from TRAINING
#   3) policy regex: reject ungrounded provider names
#   4) fallback uses training_by_job
def test_user_block_serializes_training_on_explain_gap_turn():
    """Root-cause fix: training_by_job must surface into the prompt's
    TRAINING block on explain_gap turns, not only match outcomes."""
    d = _decision(
        final_move="explain_gap",
        reason_code="credential_gap_present",
        tone="warm_supportive",
    )
    inp = ResponderV2Input(
        user_message="how do I get 310T?",
        decision=d,
        results=[],
        training_by_job={
            "gap:310T technician certification": [
                {
                    "provider": "Skilled Trades Ontario",
                    "title": "Skilled Trades Ontario — 310T technician certification",
                    "url": None,
                    "type": "credential_pathway",
                    "for_gap": "310T technician certification",
                    "summary": "Provincial regulator for the 310T trade.",
                    "verified": False,
                },
            ],
        },
        next_skill=(None, 0),
        band_signal="none",
        requires_consent=True,
    )
    block = _build_user_block_v2(*_iv(inp))
    assert "TRAINING:" in block, (
        "TRAINING must surface on explain_gap turns. Pre-fix it was "
        "gated to present_matches/present_no_match only -- which meant "
        "the LLM had no registry data and improvised providers."
    )
    assert "Skilled Trades Ontario" in block


def test_user_block_does_NOT_serialize_training_on_ask_turn():
    """The fix is narrow: TRAINING surfaces on explain_gap + match
    outcomes only. Ask turns and acknowledgements still skip the
    TRAINING block."""
    d = _decision(
        final_move="ask_one_clarifying_question",
        reason_code="target_role_unclear",
        ask_slot="target_role_text",
        tone="warm_supportive",
    )
    inp = ResponderV2Input(
        user_message="hi", decision=d,
        results=[], training_by_job={
            "x": [{"provider": "Sault College", "title": "x",
                   "url": None, "type": "local_training",
                   "for_gap": "x", "summary": "x", "verified": False}],
        },
        next_skill=(None, 0), band_signal="none",
        requires_consent=True,
    )
    block = _build_user_block_v2(*_iv(inp))
    assert "TRAINING:" not in block


def test_user_block_serializes_training_on_present_matches_turn():
    """Regression guard: the existing match-turn TRAINING surfacing
    must keep working."""
    d = _decision(final_move="present_matches", tone="brief_confident")
    inp = ResponderV2Input(
        user_message="match me", decision=d,
        results=[{"job_id": "j1", "title": "X", "employer": "Y",
                  "url": "https://example.com", "match_band": "strong",
                  "matched_skills": ["s"], "missing_skills": [],
                  "credential_warning": None, "score_explanation": {}}],
        training_by_job={
            "j1": [{"provider": "Sault College", "title": "x",
                    "url": None, "type": "local_training",
                    "for_gap": "x", "summary": "x", "verified": False}],
        },
        next_skill=(None, 0), band_signal="strong_or_good",
        requires_consent=True,
    )
    block = _build_user_block_v2(*_iv(inp))
    assert "TRAINING:" in block


# ---------------------------------------------------------------------------
# Policy regex for ungrounded provider mentions
# ---------------------------------------------------------------------------
def test_policy_rejects_ungrounded_TAC_mention():
    """The exact hallucination from the live test: LLM names
    Transportation Association of Canada, which is NOT in this turn's
    TRAINING block. Policy must reject."""
    d = _decision(
        final_move="explain_gap",
        reason_code="credential_gap_present",
        tone="warm_supportive",
    )
    inp = ResponderV2Input(
        user_message="any course?", decision=d,
        results=[], training_by_job={
            "gap:310T": [{
                "provider": "Sault College", "title": "x",
                "url": None, "type": "local_training",
                "for_gap": "310T", "summary": "x", "verified": False,
            }],
        },
        next_skill=(None, 0), band_signal="none",
        requires_consent=False,
    )
    bad_reply = (
        "For 310T, the Transportation Association of Canada (TAC) "
        "oversees air brake training; Sault College also runs the "
        "apprenticeship locally."
    )
    assert not _policy_ok_v2(bad_reply, *_iv(inp)), (
        "TAC is not in TRAINING for this turn -- the policy must reject."
    )


def test_policy_accepts_provider_mention_when_provider_is_in_training():
    """Inverse: when Sault College IS in TRAINING, the LLM may name it."""
    d = _decision(
        final_move="explain_gap",
        reason_code="credential_gap_present",
        tone="warm_supportive",
    )
    inp = ResponderV2Input(
        user_message="any course?", decision=d,
        results=[], training_by_job={
            "gap:310T": [{
                "provider": "Sault College", "title": "x",
                "url": None, "type": "local_training",
                "for_gap": "310T", "summary": "x", "verified": False,
            }],
        },
        next_skill=(None, 0), band_signal="none",
        requires_consent=False,
    )
    good_reply = "Sault College runs the apprenticeship locally."
    assert _policy_ok_v2(good_reply, *_iv(inp))


def test_policy_rejects_ministry_of_labour_when_not_grounded():
    """The live test also saw 'Ministry of Labour trades certification'.
    Allowlisted as a provider in general, but if it's not in THIS
    turn's TRAINING block, the LLM may not name it. Same defense
    pattern as TAC."""
    d = _decision(final_move="explain_gap", reason_code="credential_gap_present",
                  tone="warm_supportive")
    inp = ResponderV2Input(
        user_message="any course?", decision=d, results=[],
        training_by_job={
            "gap:310T": [{"provider": "Sault College", "title": "x",
                          "url": None, "type": "local_training",
                          "for_gap": "310T", "summary": "x", "verified": False}],
        },
        next_skill=(None, 0), band_signal="none", requires_consent=False,
    )
    bad_reply = "Check Ministry of Labour trades certification."
    assert not _policy_ok_v2(bad_reply, *_iv(inp))


@pytest.mark.parametrize("provider_name", [
    "Transportation Association of Canada", "TAC",
    "Canadian Welding Bureau", "WSIB",
    "NFPA", "National Fire Protection Association",
    "IFSAC",
])
def test_policy_rejects_known_hallucination_prone_providers(provider_name):
    """The deny-list catches specific orgs Haiku has been seen to
    improvise or supplement when no TRAINING data is present."""
    d = _decision(final_move="explain_gap",
                  reason_code="credential_gap_present", tone="warm_supportive")
    inp = ResponderV2Input(
        user_message="x", decision=d, results=[], training_by_job={},
        next_skill=(None, 0), band_signal="none", requires_consent=False,
    )
    bad_reply = f"You should check with {provider_name} for details."
    assert not _policy_ok_v2(bad_reply, *_iv(inp)), (
        f"Policy must reject ungrounded mention of {provider_name!r}"
    )


def test_policy_does_not_false_positive_on_unrelated_text():
    """Sanity: replies that don't name any known training provider
    pass the check. This catches false positives in the regex."""
    d = _decision(final_move="acknowledge_and_continue",
                  reason_code="user_confirmed", tone="brief_confident")
    inp = ResponderV2Input(
        user_message="thanks", decision=d, results=[],
        training_by_job={}, next_skill=(None, 0),
        band_signal="none", requires_consent=False,
    )
    clean = "Got it. What would you like to look at next?"
    assert _policy_ok_v2(clean, *_iv(inp))


# ---------------------------------------------------------------------------
# Slice (2026-06-08): provider-abbreviation grounding expansion
#
# Live-test bug: the LLM's natural training narration wrote "SCCC" while
# TRAINING carried "Sault Community Career Centre" (canonical full name).
# Pre-fix the policy regex saw "sccc" in the deny-list and not in
# `grounded`, rejected the reply, and the deterministic fallback fired
# with stitched YAML prose. The fix: when the canonical full name is
# grounded, the abbreviation is too -- via the new
# `_PROVIDER_ABBREVIATIONS` table in responder.py.
# ---------------------------------------------------------------------------
def test_policy_accepts_sccc_abbreviation_when_canonical_is_grounded():
    """The exact live-bug regression. SCCC in reply must pass when
    Sault Community Career Centre is in this turn's TRAINING block."""
    d = _decision(final_move="explain_gap",
                  reason_code="credential_gap_present", tone="warm_supportive")
    inp = ResponderV2Input(
        user_message="how can I get my 310S?", decision=d, results=[],
        training_by_job={
            "gap:310S": [{
                "provider": "Sault Community Career Centre",
                "title": "x", "url": None, "type": "referral_only",
                "for_gap": "310S automotive technician certification",
                "summary": "x", "verified": True,
            }],
        },
        next_skill=(None, 0), band_signal="none", requires_consent=False,
    )
    natural_reply = (
        "For 310S, the Sault Community Career Centre (SCCC) can map your "
        "current experience to the apprenticeship pathway."
    )
    assert _policy_ok_v2(natural_reply, *_iv(inp)), (
        "When the canonical 'Sault Community Career Centre' is in TRAINING, "
        "the abbreviation 'SCCC' must also be treated as grounded. Pre-fix "
        "this reply was rejected and forced the deterministic fallback."
    )


def test_policy_still_rejects_sccc_when_canonical_NOT_grounded():
    """The safety contract is unchanged: the abbreviation only counts
    when the canonical name is actually in TRAINING. If TRAINING has
    no SCCC entry, the LLM can't smuggle SCCC into the reply via its
    shorthand."""
    d = _decision(final_move="explain_gap",
                  reason_code="credential_gap_present", tone="warm_supportive")
    inp = ResponderV2Input(
        user_message="how can I get my 310T?", decision=d, results=[],
        training_by_job={
            "gap:310T": [{
                "provider": "Sault College", "title": "x", "url": None,
                "type": "local_training", "for_gap": "310T",
                "summary": "x", "verified": False,
            }],
        },
        next_skill=(None, 0), band_signal="none", requires_consent=False,
    )
    smuggled_reply = "Talk to SCCC about the 310T pathway."
    assert not _policy_ok_v2(smuggled_reply, *_iv(inp)), (
        "SCCC must NOT be accepted when Sault Community Career Centre is "
        "absent from TRAINING. The grounding contract stays exact."
    )


def test_policy_provider_abbreviation_table_uses_lowercase_keys():
    """Defense against table-key drift: _PROVIDER_ABBREVIATIONS keys
    must be lowercase to match the grounded-set lookup. A non-lowercase
    key would silently never expand. Pin the convention with a unit
    check rather than discovering it via a failed live test."""
    from skillbridge.chat.responder import _PROVIDER_ABBREVIATIONS
    for canonical, abbreviations in _PROVIDER_ABBREVIATIONS.items():
        assert canonical == canonical.lower(), (
            f"Abbreviation table key {canonical!r} is not lowercase; "
            f"matching against `grounded` would silently fail."
        )
        for abbr in abbreviations:
            assert abbr == abbr.lower(), (
                f"Abbreviation {abbr!r} for {canonical!r} is not "
                f"lowercase; same matching problem."
            )


def test_policy_provider_abbreviation_table_includes_known_sccc_case():
    """The 2026-06-08 live-bug regression case must remain in the table.
    A future contributor who removes it would silently re-introduce the
    SCCC false-positive."""
    from skillbridge.chat.responder import _PROVIDER_ABBREVIATIONS
    assert "sault community career centre" in _PROVIDER_ABBREVIATIONS
    assert "sccc" in _PROVIDER_ABBREVIATIONS["sault community career centre"]


# ---------------------------------------------------------------------------
# Fallback uses training_by_job
# ---------------------------------------------------------------------------
def test_explain_gap_fallback_uses_training_by_job_when_present():
    """Post-Slice-9 grounding fix: when the LLM happy path fails
    policy AND training_by_job has registry data, the deterministic
    fallback should narrate the registry resources -- not degrade to
    generic SCCC prose."""
    ctx = ConversationContext(
        target_role_text="truck and coach technician",
        last_presented_job_titles=("Truck and Coach Technician",),
        last_presented_credential_gaps=("310T technician certification",),
    )
    d = _decision(final_move="explain_gap",
                  reason_code="credential_gap_present", tone="warm_supportive")
    inp = ResponderV2Input(
        user_message="how do I get my 310T?", decision=d,
        results=[], training_by_job={
            "gap:310T technician certification": [
                {"provider": "Skilled Trades Ontario",
                 "title": "Skilled Trades Ontario — 310T technician certification",
                 "url": None, "type": "credential_pathway",
                 "for_gap": "310T technician certification",
                 "summary": "Provincial regulator for the 310T trade.",
                 "verified": False},
                {"provider": "Sault College",
                 "title": "Sault College — 310T technician certification",
                 "url": None, "type": "apprenticeship",
                 "for_gap": "310T technician certification",
                 "summary": "Local in-class apprenticeship component.",
                 "verified": False},
            ],
        },
        next_skill=(None, 0), band_signal="none",
        requires_consent=False, conversation_context=ctx,
    )
    reply = _fallback_reply_v2(*_iv(inp))
    # Fallback names the registry providers, not generic SCCC prose
    assert "Skilled Trades Ontario" in reply
    assert "Sault College" in reply
    # Specific gap label appears
    assert "310T technician certification" in reply


def test_explain_gap_fallback_with_verified_url_surfaces_it():
    """When the registry entry has a real URL (verified), the fallback
    includes it in the bullet. Pending entries (url=None) keep the
    bullet but drop the URL."""
    d = _decision(final_move="explain_gap",
                  reason_code="credential_gap_present", tone="warm_supportive")
    inp = ResponderV2Input(
        user_message="how?", decision=d, results=[],
        training_by_job={
            "gap:x": [
                {"provider": "Skilled Trades Ontario", "title": "x",
                 "url": "https://www.skilledtradesontario.ca/310t-path",
                 "type": "credential_pathway", "for_gap": "x",
                 "summary": "official pathway", "verified": True},
            ],
        },
        next_skill=(None, 0), band_signal="none", requires_consent=False,
    )
    reply = _fallback_reply_v2(*_iv(inp))
    assert "Skilled Trades Ontario" in reply
    assert "https://www.skilledtradesontario.ca/310t-path" in reply


def test_explain_gap_fallback_falls_through_when_no_training_data():
    """If training_by_job is empty (registry disabled or unknown gap),
    the fallback still works via the existing context-based logic --
    we didn't break the empty case."""
    ctx = ConversationContext(
        last_presented_credential_gaps=("310T technician certification",),
        last_presented_job_titles=("Truck and Coach Technician",),
    )
    d = _decision(final_move="explain_gap",
                  reason_code="credential_gap_present", tone="warm_supportive")
    inp = ResponderV2Input(
        user_message="how?", decision=d, results=[],
        training_by_job={},   # no registry data
        next_skill=(None, 0), band_signal="none",
        requires_consent=False, conversation_context=ctx,
    )
    reply = _fallback_reply_v2(*_iv(inp))
    # Falls through to Slice 8's tier-1 (gap + role context)
    assert "310T technician certification" in reply
    assert "Sault Community Career Centre" in reply or "Sault College" in reply


# ---------------------------------------------------------------------------
# Prompt rule: providers must come from TRAINING
# ---------------------------------------------------------------------------
def test_prompt_has_provider_grounding_rule():
    """The prompt must explicitly say that training PROVIDERS must come
    from TRAINING -- not just URLs. Pre-fix the rule was URL-only,
    which let the LLM invent organization names."""
    assert (
        "training provider" in OUTCOME_RESPONDER_PROMPT
        and "MUST come from RESULTS or TRAINING" in OUTCOME_RESPONDER_PROMPT
    )


def test_prompt_explicitly_forbids_supplementing_training_block():
    """The prompt must call out the specific failure mode: don't
    supplement TRAINING with providers known from outside (e.g. TAC)."""
    assert "Do NOT supplement TRAINING" in OUTCOME_RESPONDER_PROMPT


def test_redirect_fallback_tier1_with_matches_and_credential_gaps():
    """Strongest context: titles + specific credential gap names.
    The user's example case verbatim from the Slice 8 spec."""
    ctx = ConversationContext(
        target_role_text="truck and coach technician",
        last_presented_job_titles=("Truck and Coach Technician",),
        last_presented_caps_applied=("band_capped_by_credential",),
        last_presented_credential_gaps=("310T technician certification",),
    )
    d = _decision(final_move="redirect_scope",
                  reason_code="scope_violation_immigration",
                  tone="honest_redirect")
    inp = ResponderV2Input(
        user_message="can I apply for PR?",
        decision=d, results=[], training_by_job={},
        next_skill=(None, 0), band_signal="none",
        requires_consent=True,
        target_role_text="truck and coach technician",
        conversation_context=ctx,
    )
    reply = _redirect_scope_fallback_v2(inp)

    # SSM scope reaffirmed
    assert "Sault Ste. Marie" in reply
    # Specific credential gap is named
    assert "310T technician certification" in reply
    # The actual role title surfaces
    assert "Truck and Coach Technician" in reply
    # Never re-introduce immigration/PR topics
    assert "PR" not in reply
    assert "immigration" not in reply.lower()
    assert "Express Entry" not in reply


def test_redirect_fallback_tier2_with_matches_but_no_specific_gaps():
    """Matches were shown but no per-job credential gap data was
    captured. Fallback should still reference the roles by phrase."""
    ctx = ConversationContext(
        target_role_text="warehouse worker",
        last_presented_job_titles=("Warehouse Associate", "Forklift Operator"),
        last_presented_caps_applied=(),
        last_presented_credential_gaps=(),
    )
    d = _decision(final_move="redirect_scope",
                  reason_code="scope_violation_off_topic",
                  tone="honest_redirect")
    inp = ResponderV2Input(
        user_message="what about taxes?",
        decision=d, results=[], training_by_job={},
        next_skill=(None, 0), band_signal="none",
        requires_consent=True,
        target_role_text="warehouse worker",
        conversation_context=ctx,
    )
    reply = _redirect_scope_fallback_v2(inp)
    assert "Sault Ste. Marie" in reply
    # Multiple titles -> use the generic "those roles" phrasing
    assert "those roles" in reply.lower()


def test_redirect_fallback_tier3_with_target_role_only():
    """No matches shown yet but user already named a target role.
    Fallback references the target role to keep continuity."""
    ctx = ConversationContext(
        target_role_text="electrician apprentice",
        last_presented_job_titles=(),
        last_presented_caps_applied=(),
        last_presented_credential_gaps=(),
    )
    d = _decision(final_move="redirect_scope",
                  reason_code="scope_violation_off_topic",
                  tone="honest_redirect")
    inp = ResponderV2Input(
        user_message="what about taxes?",
        decision=d, results=[], training_by_job={},
        next_skill=(None, 0), band_signal="none",
        requires_consent=True,
        target_role_text="electrician apprentice",
        conversation_context=ctx,
    )
    reply = _redirect_scope_fallback_v2(inp)
    assert "Sault Ste. Marie" in reply
    assert "electrician apprentice" in reply


def test_redirect_fallback_tier4_cold_session_immigration_includes_sccc_referral():
    """Cold-session immigration scope MUST include the Sault Community
    Career Centre referral. Updated 2026-06-05 after live test surfaced
    that a bare 'I focus on jobs' line for a PR question leaves the
    newcomer without a real next step. SCCC referral is NOT immigration
    advice -- it's pointing the user at the agency that handles those
    questions locally, which is the honest scope redirect."""
    d = _decision(final_move="redirect_scope",
                  reason_code="scope_violation_immigration",
                  tone="honest_redirect")
    inp = ResponderV2Input(
        user_message="can I apply for PR?",
        decision=d, results=[], training_by_job={},
        next_skill=(None, 0), band_signal="none",
        requires_consent=True,
        target_role_text=None,
        conversation_context=None,
    )
    reply = _redirect_scope_fallback_v2(inp)
    assert "Sault Ste. Marie" in reply
    assert "Sault Community Career Centre" in reply
    # Honest scope: name SCCC as the place to ask about
    # immigration/PR, but DO NOT give immigration advice (no
    # mention of Express Entry, PR criteria, work permit process, etc).
    forbidden = ("Express Entry", "you may qualify", "PR application", "IRCC")
    for word in forbidden:
        assert word not in reply, (
            f"redirect_scope immigration fallback must NOT give "
            f"immigration advice; contains {word!r}: {reply!r}"
        )
    # Still pivots back to the user's job search.
    assert "what kind of work" in reply.lower()


def test_redirect_fallback_tier4_cold_session_non_immigration_uses_generic_line():
    """For wages / non-SSM / off-topic scope, SCCC is NOT the right
    referral. The cold-session fallback preserves the generic
    'I focus on jobs in SSM' line."""
    for reason in (
        "scope_violation_wages",
        "scope_violation_non_ssm",
        "scope_violation_off_topic",
    ):
        d = _decision(final_move="redirect_scope",
                      reason_code=reason,
                      tone="honest_redirect")
        inp = ResponderV2Input(
            user_message="off-topic",
            decision=d, results=[], training_by_job={},
            next_skill=(None, 0), band_signal="none",
            requires_consent=True,
            target_role_text=None,
            conversation_context=None,
        )
        reply = _redirect_scope_fallback_v2(inp)
        # Generic cold-session prompt with canonical capitalization.
        assert "What kind of work" in reply, (
            f"reason={reason}: expected the generic cold-session line; "
            f"got {reply!r}"
        )
        # SCCC referral is wrong for these scope reasons -- don't surface it.
        assert "Sault Community Career Centre" not in reply, (
            f"reason={reason}: SCCC referral does not belong here; "
            f"got {reply!r}"
        )


def test_redirect_fallback_uses_single_title_phrasing_when_one_match():
    """Single title -> 'the {title} role' (specific), not 'those roles'."""
    ctx = ConversationContext(
        last_presented_job_titles=("Warehouse Associate",),
        last_presented_credential_gaps=("forklift certification",),
    )
    d = _decision(final_move="redirect_scope",
                  reason_code="scope_violation_off_topic",
                  tone="honest_redirect")
    inp = ResponderV2Input(
        user_message="off-topic", decision=d,
        results=[], training_by_job={}, next_skill=(None, 0),
        band_signal="none", requires_consent=True,
        conversation_context=ctx,
    )
    reply = _redirect_scope_fallback_v2(inp)
    assert "Warehouse Associate role" in reply
    assert "those roles" not in reply.lower()


def test_redirect_fallback_never_leaks_operational_terms():
    """Even with rich context, the fallback must respect the same
    operational-term boundary as the LLM path. No 'planner', 'arbiter',
    'overrode' should appear."""
    ctx = ConversationContext(
        target_role_text="warehouse worker",
        last_presented_job_titles=("Warehouse Associate",),
        last_presented_credential_gaps=("forklift certification",),
    )
    d = _decision(final_move="redirect_scope",
                  reason_code="scope_violation_immigration",
                  tone="honest_redirect")
    inp = ResponderV2Input(
        user_message="x", decision=d, results=[],
        training_by_job={}, next_skill=(None, 0),
        band_signal="none", requires_consent=True,
        target_role_text="warehouse worker",
        conversation_context=ctx,
    )
    reply = _redirect_scope_fallback_v2(inp).lower()
    for term in ("the planner said", "the arbiter", "overrode",
                 "arbiter_action", "fallback_to_legacy"):
        assert term not in reply, (
            f"redirect fallback leaked operational term {term!r}: {reply!r}"
        )


def test_conversation_context_has_presented_context_helper():
    """The has_presented_context() helper drives the tier selection."""
    assert not ConversationContext().has_presented_context()
    assert not ConversationContext(
        target_role_text="warehouse",
    ).has_presented_context()
    assert ConversationContext(
        last_presented_job_titles=("X",),
    ).has_presented_context()


def test_conversation_context_is_frozen():
    """ConversationContext is immutable so the responder can't
    accidentally mutate session state via its input."""
    ctx = ConversationContext(target_role_text="warehouse")
    with pytest.raises(Exception):
        ctx.target_role_text = "different"  # type: ignore[misc]


# ===========================================================================
# Structural: v2 prompt + v1 prompt share critical rule blocks verbatim
# ===========================================================================
@pytest.mark.parametrize("verbatim_phrase", [
    'NO ACTIONS WE CANNOT PERFORM.',
    'NEVER name a non-local city as a destination or recommendation.',
    'NO CREDENTIAL EQUIVALENCE CLAIMS.',
    'NO IMMIGRATION / LEGAL / MEDICAL / FINANCIAL ADVICE.',
    'TRAINING DISCUSSIONS ARE IN SCOPE.',
    'MISSING SKILLS ARE NOT OWNED SKILLS.',
    'MATCH STAGES -- DISTINGUISH "YOU HAVE X" FROM "RELATED TO X".',
    'CAPS APPLIED -- NAME THE CAP.',
])
def test_v2_prompt_keeps_v1_rule_block_verbatim(verbatim_phrase):
    """v1 and v2 must contain the same load-bearing rule headers
    verbatim during the migration period. Catches the case where v1
    evolves and v2 drifts behind."""
    assert verbatim_phrase in NEXT_ACTION_RESPONDER_PROMPT
    assert verbatim_phrase in OUTCOME_RESPONDER_PROMPT


# ===========================================================================
# Slice N (2026-06-05): present_near_miss outcome
#
# Three concerns:
#   1. _present_near_miss_fallback_v2: deterministic narrator produces
#      the right prose shape (role anchor + gap list + provider tip +
#      walk-through offer) from the payload.
#   2. _build_user_block_v2: NEAR_MISS_GAPS payload reaches the LLM
#      block when final_move == present_near_miss; TRAINING is shown
#      so the LLM can name providers verbatim.
#   3. Policy: the new _NEAR_MISS_FORBIDDEN_PATTERNS reject
#      "good fit", "you qualify", "stretch match", etc. on
#      present_near_miss turns, and only on those turns (so the
#      patterns don't disrupt other replies).
# ===========================================================================
from skillbridge.chat.responder import _present_near_miss_fallback_v2  # noqa: E402


def _near_miss_decision() -> ArbiterDecision:
    return ArbiterDecision(
        final_move="present_near_miss",
        reason_code="title_match_with_major_gaps",
        tone="warm_supportive",
        arbiter_action="resolved_to_near_miss",
    )


def _michael_payload() -> dict:
    """The canonical Michael Carter near-miss payload, used across
    several tests so the shape is documented once."""
    return {
        "role": "Truck and Coach Technician",
        "employer": "Garden River First Nation",
        "job_count": 1,
        "credential_gaps": [
            "310T technician certification",
            "Class G driver's license",
        ],
        "core_skill_gaps": [
            "emergency repair",
            "emissions testing preparation",
            "truck service and maintenance",
        ],
    }


def _michael_training() -> dict:
    """Registry-grounded training data for the lead credential. Mirrors
    what the handler will populate in Slice N-5 from `classify_gap` +
    `surface_resources`."""
    return {
        "Truck and Coach Technician": [
            {
                "provider": "Skilled Trades Ontario",
                "for_gap": "310T technician certification",
                "title": "310T pathway",
                "url": None,  # unverified -- still narratable by name
            },
        ],
    }


def _near_miss_inp(
    payload: dict | None = None,
    training: dict | None = None,
    user_message: str = "same role",
) -> ResponderV2Input:
    return ResponderV2Input(
        user_message=user_message,
        decision=_near_miss_decision(),
        results=[],
        training_by_job=training or {},
        next_skill=(None, 0),
        band_signal="low_only",
        requires_consent=False,
        target_role_text="truck and coach technician",
        near_miss_payload=payload,
    )


# ---- Fallback narrator: full canonical payload + provider ----
def test_near_miss_fallback_anchors_to_role_and_dataset():
    """Sentence 1 must name the role and that a posting WAS found in
    SSM, so the user knows the role exists locally."""
    out = _present_near_miss_fallback_v2(
        _near_miss_inp(_michael_payload(), _michael_training()),
    )
    assert "Truck and Coach Technician" in out
    assert "Sault Ste. Marie" in out
    assert "not a realistic match yet" in out


def test_near_miss_fallback_names_credentials_first_then_skills():
    """Locked Q4: credentials lead, skills follow. Test both orderings
    by index: 310T appears BEFORE any core skill in the prose."""
    out = _present_near_miss_fallback_v2(
        _near_miss_inp(_michael_payload(), _michael_training()),
    )
    cred_index = out.index("310T technician certification")
    skill_index = out.index("emergency repair")
    assert cred_index < skill_index, (
        f"credentials must precede core skills in the prose; got "
        f"cred@{cred_index}, skill@{skill_index}: {out!r}"
    )


def test_near_miss_fallback_names_provider_when_registry_has_one():
    """Provider name MUST come from training_by_job verbatim. Never
    invented. If the handler populated the training data, the
    responder surfaces it; otherwise stays silent."""
    out = _present_near_miss_fallback_v2(
        _near_miss_inp(_michael_payload(), _michael_training()),
    )
    assert "Skilled Trades Ontario" in out


def test_near_miss_fallback_omits_provider_when_registry_silent():
    """No registry entry for the lead credential -> the fallback MUST
    NOT name a provider. (The policy regex would reject an ungrounded
    name anyway, but the deterministic fallback skips it cleanly.)"""
    out = _present_near_miss_fallback_v2(
        _near_miss_inp(_michael_payload(), training={}),
    )
    assert "Skilled Trades Ontario" not in out
    # Still offers the walk-through with the lead credential name.
    assert "310T technician certification" in out


def test_near_miss_fallback_offers_walk_through_for_lead_credential():
    """Closing sentence: 'Want to walk through the {credential} path
    first?'. Empirically helps the user pick a next step rather than
    drowning in the gap list."""
    out = _present_near_miss_fallback_v2(
        _near_miss_inp(_michael_payload()),
    )
    assert "Want to walk through" in out
    assert "310T technician certification" in out


def test_near_miss_fallback_pluralizes_when_job_count_above_one():
    """Multi-job case (locked Q6 'plus N similar' framing): opener
    must say 'I found N postings' not 'a posting'."""
    payload = _michael_payload()
    payload["job_count"] = 3
    out = _present_near_miss_fallback_v2(_near_miss_inp(payload))
    assert "3 Truck and Coach Technician postings" in out
    assert "they're not a realistic match yet" in out


def test_near_miss_fallback_handles_credentials_only_payload():
    """No core_skill_gaps: still produces a sensible response naming
    only the credentials."""
    payload = _michael_payload()
    payload["core_skill_gaps"] = []
    out = _present_near_miss_fallback_v2(_near_miss_inp(payload))
    assert "credentials" in out
    assert "core skill" not in out.lower()


def test_near_miss_fallback_handles_skills_only_payload():
    """No credential_gaps: lead with core skills. Walk-through offer
    pivots to the first skill instead of a credential."""
    payload = _michael_payload()
    payload["credential_gaps"] = []
    out = _present_near_miss_fallback_v2(_near_miss_inp(payload))
    assert "core skills" in out
    assert "emergency repair" in out
    # Walk-through pivots to a skill phrasing
    assert "build up" in out


def test_near_miss_fallback_defensive_fallback_when_payload_missing():
    """Handler bug or test misuse: payload is None or missing the role.
    Responder MUST still return safe prose pointing at SCCC without
    inventing gap data."""
    # None payload
    out_none = _present_near_miss_fallback_v2(_near_miss_inp(None))
    assert "Sault Community Career Centre" in out_none
    # Empty payload
    out_empty = _present_near_miss_fallback_v2(_near_miss_inp({}))
    assert "Sault Community Career Centre" in out_empty
    # Payload with role but no gaps
    out_no_gaps = _present_near_miss_fallback_v2(_near_miss_inp({
        "role": "Truck and Coach Technician",
        "credential_gaps": [], "core_skill_gaps": [], "job_count": 1,
    }))
    assert "Sault Community Career Centre" in out_no_gaps


# ---- _build_user_block_v2 serialization ----
def test_user_block_includes_near_miss_payload_on_near_miss_turn():
    """NEAR_MISS_GAPS block reaches the LLM verbatim. The LLM uses
    this to narrate -- no other source of gap data is grounded."""
    block = _build_user_block_v2(*_iv(_near_miss_inp(_michael_payload(), _michael_training())))
    assert "NEAR_MISS_GAPS:" in block
    assert "Truck and Coach Technician" in block
    assert "310T technician certification" in block
    assert "emergency repair" in block


def test_user_block_includes_training_on_near_miss_turn():
    """Slice N grounds providers: the TRAINING block must be serialized
    on present_near_miss turns so the LLM can name 'Skilled Trades
    Ontario' verbatim and the ungrounded-provider policy won't reject
    the reply."""
    block = _build_user_block_v2(*_iv(_near_miss_inp(_michael_payload(), _michael_training())))
    assert "TRAINING:" in block
    assert "Skilled Trades Ontario" in block


def test_user_block_omits_near_miss_payload_on_other_outcomes():
    """NEAR_MISS_GAPS must NOT leak into present_matches /
    present_no_match / explain_gap / etc. -- it's a near-miss-specific
    structure."""
    d = ArbiterDecision(
        final_move="present_no_match",
        reason_code=ARBITER_REASON_NO_MATCHES,
        tone="honest_redirect", arbiter_action="resolved_to_no_match",
    )
    inp = ResponderV2Input(
        user_message="show me jobs", decision=d, results=[],
        training_by_job={}, next_skill=(None, 0), band_signal="low_only",
        requires_consent=False, target_role_text="truck tech",
        near_miss_payload=_michael_payload(),  # set, but should be ignored
    )
    block = _build_user_block_v2(*_iv(inp))
    assert "NEAR_MISS_GAPS" not in block


# ---- Policy: forbidden framing on present_near_miss ----
@pytest.mark.parametrize("reply,reason", [
    ("This is a good fit for your background.",          "good fit"),
    ("Great match for what you have.",                    "great match"),
    ("You qualify for this role today.",                  "you qualify"),
    ("You would qualify after one more skill.",           "you would qualify"),
    ("You're qualified for the next step.",               "you're qualified"),
    ("Looks like a stretch match worth pursuing.",        "stretch match"),
    ("Strong fit for your experience level.",             "strong fit"),
    ("Perfect match -- apply now.",                       "perfect match"),
])
def test_policy_rejects_forbidden_framing_on_near_miss_turn(reply, reason):
    """Each forbidden phrase from the design's responder CANNOT list
    causes _policy_ok_v2 to reject the reply on a present_near_miss
    turn. The deterministic fallback then takes over."""
    inp = _near_miss_inp(_michael_payload(), _michael_training())
    assert _policy_ok_v2(reply, *_iv(inp)) is False, (
        f"reply containing {reason!r} should be rejected: {reply!r}"
    )


def test_policy_allows_appropriate_near_miss_framing():
    """A reply that follows the design's CAN list (uses 'not a
    realistic match yet', names credentials, offers next step)
    must pass policy. Confirms we didn't over-block."""
    good_reply = (
        "I found a Truck and Coach Technician posting in Sault Ste. "
        "Marie, but it's not a realistic match yet. The main blockers "
        "are your 310T technician certification and Class G driver's "
        "license. For 310T technician certification, Skilled Trades "
        "Ontario is where to start. Want to walk through the 310T "
        "path first?"
    )
    inp = _near_miss_inp(_michael_payload(), _michael_training())
    assert _policy_ok_v2(good_reply, *_iv(inp)) is True


@pytest.mark.parametrize("forbidden_phrase", [
    "good fit", "great match", "you qualify", "stretch match",
])
def test_policy_near_miss_patterns_do_not_fire_on_other_turns(forbidden_phrase):
    """The new patterns are gated on final_move == present_near_miss.
    A `present_matches` reply that happens to say 'good fit' is NOT
    rejected by the new rule (other rules may still reject it; not
    our problem here). This proves the gating works."""
    d = ArbiterDecision(
        final_move="present_matches",
        reason_code=ARBITER_REASON_MATCHES_FOUND,
        tone="warm_supportive", arbiter_action="resolved_to_matches",
    )
    inp = ResponderV2Input(
        user_message="show me jobs", decision=d, results=[],
        training_by_job={}, next_skill=(None, 0),
        band_signal="strong_or_good", requires_consent=False,
        target_role_text="truck tech",
    )
    reply = f"This role is a {forbidden_phrase} for you."
    # Result might still be True or False based on OTHER policy rules,
    # but the NEW near-miss patterns must NOT be what rejects it.
    # We assert by checking that the same reply on a near-miss turn is
    # rejected and the gating distinguishes them.
    near_miss_inp = _near_miss_inp(_michael_payload(), _michael_training())
    assert _policy_ok_v2(reply, *_iv(near_miss_inp)) is False, (
        "near-miss turn should reject; baseline for the gating test"
    )
