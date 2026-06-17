"""R-5 responder tests: deterministic clarification renderer,
REMAINING_GAPS user-block serialization, explain_remaining_gaps
fallback, policy rules, and the early-return that skips the LLM when
a clarification payload is set.

No DB. No LLM. Stubs `is_enabled` to False for any test that exercises
the fallback path; tests that exercise the early-return short-circuit
keep LLM enabled but never actually call out (the early-return
fires first).
"""
from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("LLM_ENABLED", "false")

import pytest

from skillbridge.chat import responder
from skillbridge.chat.arbiter import (
    ARBITER_REASON_REMAINING_GAPS,
    ARBITER_REASON_REMAINING_GAPS_RETRACTED,
    ArbiterDecision,
)
from skillbridge.chat.responder import (
    ResponderV2Input,
    _build_user_block_v2,
    _policy_ok_v2,
    _present_remaining_gaps_fallback_v2,
    _render_clarification,
    compose_response_v2,
)
from skillbridge.chat.url_views import (
    build_sanitized_responder_view_v2 as _v_v2,
)


def _iv(inp):
    """Return (inp, view) so each test reuses one inp instance."""
    return inp, _v_v2(inp)

pytestmark = pytest.mark.nodb


# ============================================================================
# Helpers
# ============================================================================
def _decision_explain_remaining_gaps(retracted: bool = False):
    return ArbiterDecision(
        final_move="explain_remaining_gaps",
        reason_code=(
            ARBITER_REASON_REMAINING_GAPS_RETRACTED if retracted
            else ARBITER_REASON_REMAINING_GAPS
        ),
        tone="warm_supportive",
        arbiter_action="handler_synthesized_remaining_gaps",
        ask_slot=None,
    )


def _decision_clarification(reason: str):
    return ArbiterDecision(
        final_move="ask_one_clarifying_question",
        reason_code=reason,
        tone="warm_supportive",
        arbiter_action="handler_synthesized_clarification",
        ask_slot=None,
    )


def _input(decision, **kw) -> ResponderV2Input:
    defaults = dict(
        user_message="anything",
        decision=decision,
        results=[],
        training_by_job={},
        next_skill=(None, 0),
        band_signal="none",
        requires_consent=True,
        target_role_text=None,
        resume_facts=None,
        conversation_context=None,
        near_miss_payload=None,
        remaining_gaps_payload=None,
        clarification_payload=None,
    )
    defaults.update(kw)
    return ResponderV2Input(**defaults)


# ============================================================================
# Clarification renderer (deterministic templates)
# ============================================================================
def test_render_clarification_add_with_known_canonical():
    payload = {
        "kind": "credential_completion_confirmation",
        "credential_canonical": "310S automotive technician certification",
        "credential_display":   "310S licence",
        # Round-20 identity contract: trusted_displays MUST contain
        # the display for it to render.
        "trusted_displays":     ["310S licence"],
        "action":               "add",
    }
    out = _render_clarification(payload)
    assert "310S licence" in out
    assert "have you completed" in out.lower()
    assert "still working" in out.lower()


def test_render_clarification_remove_with_known_canonical():
    payload = {
        "kind": "credential_completion_confirmation",
        "credential_canonical": "Class G driver's license",
        "credential_display":   "G2/G driver's licence",
        "trusted_displays":     ["G2/G driver's licence"],
        "action":               "remove",
    }
    out = _render_clarification(payload)
    assert "G2/G driver's licence" in out
    assert "don't have" in out.lower()
    assert "recalculate" in out.lower()


def test_render_clarification_no_canonical_asks_which_credential():
    payload = {
        "kind": "credential_completion_confirmation",
        "credential_canonical": None,
        "credential_display":   "",
        "action":               "add",
    }
    out = _render_clarification(payload)
    assert "which credential" in out.lower()


def test_render_clarification_bootstrap():
    payload = {"kind": "bootstrap_match_request"}
    out = _render_clarification(payload)
    assert "haven't shown" in out.lower()
    assert "target field" in out.lower()


def test_render_clarification_falls_back_on_unknown_kind():
    out = _render_clarification({"kind": "unknown_future_extension"})
    assert "tell me a bit more" in out.lower()


def test_render_clarification_no_provider_names_or_urls():
    """Clarification templates are trusted-by-construction: no
    provider names, no URLs, no scope-violation hooks."""
    for p in [
        {"kind": "bootstrap_match_request"},
        {"kind": "credential_completion_confirmation",
         "credential_canonical": "310S-canon", "credential_display": "310S",
         "action": "add"},
        {"kind": "credential_completion_confirmation",
         "credential_canonical": None, "credential_display": "",
         "action": "add"},
    ]:
        out = _render_clarification(p).lower()
        for forbidden in ("http://", "https://", "sault community",
                          "drivetest", "you qualify", "good fit",
                          "express entry", "$"):
            assert forbidden not in out, (p, forbidden)


# ============================================================================
# compose_response_v2 early-return -- LLM skipped + policy skipped
# ============================================================================
def test_clarification_payload_short_circuits_compose_response_v2(monkeypatch):
    """The early-return MUST fire BEFORE the LLM call AND BEFORE the
    policy sweep. Verified by patching `responder.call` to fail the
    test if invoked."""
    monkeypatch.setattr(responder, "is_enabled", lambda: True)
    monkeypatch.setattr(
        responder, "call",
        lambda *a, **kw: pytest.fail("LLM must not be called on clarification"),
    )
    monkeypatch.setattr(
        responder, "_policy_ok_v2",
        lambda *a, **kw: pytest.fail("Policy must not run on clarification"),
    )
    inp = _input(
        _decision_clarification("confirm_credential_completion"),
        clarification_payload={
            "kind": "credential_completion_confirmation",
            "credential_canonical": "X-CANON",
            "credential_display":   "X licence",
            "trusted_displays":     ["X licence"],
            "action":               "add",
        },
    )
    out = compose_response_v2(inp)
    assert "X licence" in out


def test_compose_response_v2_normal_flow_when_clarification_payload_none(monkeypatch):
    """Sanity: without clarification_payload the LLM path runs (or
    fallback if disabled). Pin that the early-return isn't accidentally
    short-circuiting the normal flow."""
    monkeypatch.setattr(responder, "is_enabled", lambda: False)
    inp = _input(
        _decision_clarification("confirm_credential_completion"),
        # clarification_payload left as None
    )
    # With LLM disabled, falls into _fallback_reply_v2 -> generic ask line.
    out = compose_response_v2(inp)
    assert isinstance(out, str) and out


# ============================================================================
# REMAINING_GAPS user-block serialization
# ============================================================================
def test_user_block_includes_remaining_gaps_for_explain_remaining_gaps_turn():
    payload = {
        "role": "310S Licensed Automotive Technician",
        "employer": "Great Lakes Honda",
        "assumed_completed_credentials": [
            {"display": "310S Automotive Technician License",
             "canonical": "310S-CANON", "mode": "hypothetical"},
        ],
        "remaining_credentials": [
            {"display": "G2/G driver's license",
             "canonical": "CLASS-G-CANON"},
        ],
        "remaining_core_skills": ["Honda vehicle experience"],
        "any_hypothetical": True,
    }
    inp = _input(
        _decision_explain_remaining_gaps(),
        remaining_gaps_payload=payload,
    )
    block = _build_user_block_v2(*_iv(inp))
    assert "REMAINING_GAPS:" in block
    assert "Great Lakes Honda" in block
    assert "G2/G driver's license" in block
    assert "Honda vehicle experience" in block
    assert "\"any_hypothetical\": true" in block


def test_user_block_omits_remaining_gaps_on_unrelated_final_move():
    """REMAINING_GAPS is keyed strictly on final_move ==
    explain_remaining_gaps; even with a payload set, an unrelated move
    must NOT serialize it."""
    payload = {"role": "X", "remaining_credentials": []}
    inp = _input(
        ArbiterDecision(
            final_move="explain_gap", reason_code="credential_gap_present",
            tone="warm_supportive", arbiter_action="passed_planner_through",
        ),
        remaining_gaps_payload=payload,
    )
    block = _build_user_block_v2(*_iv(inp))
    assert "REMAINING_GAPS:" not in block


# ============================================================================
# Round-19 R-5 review regressions
# ============================================================================
def test_user_block_includes_training_on_explain_remaining_gaps_turn():
    """Round-19 finding 1: the TRAINING gate in `_build_user_block_v2`
    omitted `explain_remaining_gaps`, so the LLM never saw the registry
    resources the handler regrounded for the lead remaining credential.
    Either it improvised a provider (policy-rejected) or named none.
    Add to the gate."""
    payload = {
        "role": "X", "remaining_credentials": [
            {"display": "G2/G driver license", "canonical": "G"},
        ],
        "remaining_core_skills": [], "any_hypothetical": False,
        "assumed_completed_credentials": [],
    }
    training = {
        "gap:G2/G driver license": [
            {"provider": "DriveTest", "summary": "road test"},
        ],
    }
    inp = _input(
        _decision_explain_remaining_gaps(),
        remaining_gaps_payload=payload,
        training_by_job=training,
    )
    block = _build_user_block_v2(*_iv(inp))
    assert "REMAINING_GAPS:" in block
    assert "TRAINING:" in block
    assert "DriveTest" in block


def test_clarification_renderer_rejects_unverified_display_containing_url():
    """Round-19 finding 2 + round-21 identity contract: a display
    that doesn't appear in trusted_displays is REJECTED, regardless
    of whether it contains a URL. The renderer falls back to the
    no-target template (or the canonical when canonical IS trusted)."""
    payload = {
        "kind": "credential_completion_confirmation",
        "credential_canonical": "X-CANON",
        "credential_display":   "310S licence https://evil.ca",
        "trusted_displays":     [],   # nothing trusted -- safest fallback
        "action":               "add",
    }
    out = _render_clarification(payload)
    assert "https://" not in out
    assert "evil.ca" not in out
    # Neither display nor canonical were trusted -> no-target template
    assert "which credential" in out.lower()


def test_clarification_renderer_strips_provider_names_from_display():
    """A snapshot display containing a known provider name (which is a
    snapshot-shape bug -- credential displays are gap names, not
    provider names) must be stripped before rendering."""
    payload = {
        "kind": "credential_completion_confirmation",
        "credential_canonical": "X-CANON",
        "credential_display":   "G2 DriveTest Sault College ticket",
        "action":               "add",
    }
    out = _render_clarification(payload).lower()
    assert "drivetest" not in out
    assert "sault college" not in out


def test_clarification_renderer_strips_scope_violations_from_display():
    """Scope-violation hooks ('express entry', 'job bank') must be
    stripped from the display so a forged display can't smuggle them
    past the clarification's policy bypass."""
    payload = {
        "kind": "credential_completion_confirmation",
        "credential_canonical": "X-CANON",
        "credential_display":   "Express Entry permit",
        "action":               "add",
    }
    out = _render_clarification(payload).lower()
    assert "express entry" not in out


def test_clarification_renderer_strips_dollar_amounts_from_display():
    payload = {
        "kind": "credential_completion_confirmation",
        "credential_canonical": "X-CANON",
        "credential_display":   "310S $35/hr licence",
        "action":               "add",
    }
    out = _render_clarification(payload)
    assert "$" not in out
    assert "/hr" not in out


def test_clarification_renderer_falls_back_when_display_sanitizes_to_empty():
    """A display made entirely of denied content (URLs, providers)
    sanitizes to empty; the renderer falls back to the no-target
    template rather than emitting a bare 'have you completed your , or
    are you still working' sentence."""
    payload = {
        "kind": "credential_completion_confirmation",
        "credential_canonical": "http://evil.ca",   # also sanitizes to empty
        "credential_display":   "https://drivetest.ca",
        "action":               "add",
    }
    out = _render_clarification(payload)
    assert "which credential" in out.lower()


# ============================================================================
# Round-21 R-5 review -- identity-based verification
# ============================================================================
# Syntactic validation (round 20) controls CHARACTERS, not meaning.
# Anything matching the allow-list could still be hostile prose:
#   "310S evil.technology"          -- domain-like extra prose
#   "310S 192.168.1.1"               -- IP address
#   "310S Evil Training Academy"     -- unknown provider name
#   "310S Ministry of Labour course" -- credible-sounding fake
#   "310S 35 dollars an hour"        -- salary as prose
# The reviewer's correct contract: only bypass policy when the
# displayed value is verified against a trusted snapshot/registry
# entry. The renderer now requires `trusted_displays` (the snapshot's
# credential_gaps[*].display) on the payload. Any candidate display
# not in this set falls back to the safe no-target template.
@pytest.mark.parametrize("hostile_prose", [
    "310S evil.technology",
    "310S 192.168.1.1",
    "310S Evil Training Academy",
    "310S Ministry of Labour course",
    "310S 35 dollars an hour",
    "310S Some Random Phrase",
    "310S OPS course",
    "310S taught by Mr X at College",
])
def test_renderer_rejects_unverified_prose_even_in_allow_charset(hostile_prose):
    """Round-21: a display made of allow-listed characters but NOT
    matching any snapshot display MUST be rejected. Pure syntactic
    validation can't catch semantic injection like 'Evil Training
    Academy' or '35 dollars an hour'."""
    payload = {
        "kind": "credential_completion_confirmation",
        "credential_canonical": "310S Automotive Technician License",
        "credential_display":   hostile_prose,
        # Trusted set contains ONLY the snapshot's legitimate display.
        "trusted_displays":     ["310S Automotive Technician License"],
        "action":               "add",
    }
    out = _render_clarification(payload)
    # The hostile prose MUST NOT appear in the output
    assert hostile_prose not in out, (
        f"Unverified prose leaked through identity check: {hostile_prose!r}"
    )
    # Output uses the canonical (which IS in trusted_displays) as the
    # safe fallback for the verified display.
    assert "310S Automotive Technician License" in out


def test_renderer_uses_display_when_it_matches_trusted_snapshot_entry():
    """Identity contract -- the display is rendered when it matches
    any trusted_displays entry."""
    payload = {
        "kind": "credential_completion_confirmation",
        "credential_canonical": "310S canon",
        "credential_display":   "310S Automotive Technician License",
        "trusted_displays":     [
            "310S Automotive Technician License",
            "G2/G driver's license",
        ],
        "action":               "add",
    }
    out = _render_clarification(payload)
    assert "310S Automotive Technician License" in out


def test_renderer_falls_back_to_canonical_when_only_canonical_is_trusted():
    """When display ISN'T in trusted_displays but the canonical IS,
    the renderer uses the canonical. This handles cases where the
    handler couldn't find a matching snapshot display but the canonical
    itself appears in the snapshot's credential_gaps[*].display."""
    payload = {
        "kind": "credential_completion_confirmation",
        # Canonical IS in trusted_displays
        "credential_canonical": "Class G driver's license",
        # Display is hostile prose
        "credential_display":   "Class G Evil Training Academy",
        "trusted_displays":     ["Class G driver's license"],
        "action":               "add",
    }
    out = _render_clarification(payload)
    assert "Evil Training Academy" not in out
    assert "Class G driver's license" in out


def test_renderer_falls_back_when_neither_display_nor_canonical_is_trusted():
    """When neither display nor canonical appears in trusted_displays,
    the renderer falls back to the no-target template -- the safest
    shape when nothing can be verified."""
    payload = {
        "kind": "credential_completion_confirmation",
        "credential_canonical": "stale-canon-not-in-snapshot",
        "credential_display":   "stale-display-not-in-snapshot",
        "trusted_displays":     ["310S Automotive Technician License"],
        "action":               "add",
    }
    out = _render_clarification(payload)
    assert "stale" not in out.lower()
    assert "which credential" in out.lower()


def test_renderer_falls_back_when_trusted_displays_is_missing():
    """A payload missing trusted_displays entirely is treated as
    untrusted -- no display can be verified, so the no-target
    template fires. This is the strictest interpretation of the
    identity contract."""
    payload = {
        "kind": "credential_completion_confirmation",
        "credential_canonical": "310S Automotive Technician License",
        "credential_display":   "310S Automotive Technician License",
        # trusted_displays MISSING
        "action":               "add",
    }
    out = _render_clarification(payload)
    assert "which credential" in out.lower()


def test_renderer_falls_back_when_trusted_displays_is_empty():
    """Same behavior with an empty list -- nothing is verifiable."""
    payload = {
        "kind": "credential_completion_confirmation",
        "credential_canonical": "310S Automotive Technician License",
        "credential_display":   "310S Automotive Technician License",
        "trusted_displays":     [],
        "action":               "add",
    }
    out = _render_clarification(payload)
    assert "which credential" in out.lower()


def test_handler_attaches_trusted_displays_from_current_snapshot():
    """End-to-end: the handler's _build_clarification_payload populates
    trusted_displays from the current snapshot. The renderer then
    correctly accepts any of those displays."""
    from skillbridge.chat import handler
    from skillbridge.chat.arbiter import ARBITER_REASON_CONFIRM_CREDENTIAL
    from skillbridge.session.staging import StagedProfile
    sp = StagedProfile.new("t")
    sp.last_match_snapshot = {
        "lead_job": {"job_id": "j", "title": "T", "employer": None,
                     "credential_gaps": [
                         {"display": "310S Automotive Technician License",
                          "canonical": "310S-CANON"},
                         {"display": "G2/G driver's license",
                          "canonical": "CLASS-G-CANON"},
                     ],
                     "core_skill_gaps": []},
        "other_jobs_meta": [],
    }
    sp.pending_credential_confirmation = {
        "canonical": "310S-CANON", "action": "add",
    }
    payload = handler._build_clarification_payload(
        sp, ARBITER_REASON_CONFIRM_CREDENTIAL,
    )
    assert payload["trusted_displays"] == [
        "310S Automotive Technician License",
        "G2/G driver's license",
    ]
    # The renderer accepts the display because it's in trusted_displays.
    out = _render_clarification(payload)
    assert "310S Automotive Technician License" in out


# ============================================================================
# Round-20 R-5 review -- strict allow-list display validation
# ============================================================================
@pytest.mark.parametrize("hostile_display", [
    # URI schemes (the reviewer's reproductions)
    "evil.ca/path",                    # bare domain
    "ftp://evil.ca",                   # non-HTTP scheme
    "javascript:alert(1)",              # XSS scheme
    "mailto:test@evil.ca",              # email scheme
    "data:text/html,<script>",          # data URI
    "vbscript:msgbox(1)",
    "file:///etc/passwd",
    "http://drivetest.ca/book",
    "https://Example.com/path",
    "www.evil.ca/path",                 # plain WWW
    # Markdown / HTML
    "[click](http://x)",
    "<a href=evil>X</a>",
    "<script>alert(1)</script>",
    "<img src=x>",
    "X [link]",
    "X {token}",
    "X | pipe",
    "X \\ backslash",
    "X ` backtick",
    # Email syntax
    "test@evil.ca",
    "@reply",
    # Unknown providers (deny-list miss): these contain a domain,
    # which the allow-list rejects regardless of the provider name.
    "PreyTech.ca",
    "SomeProvider.co",
    # Known providers without a TLD: still rejected by the explicit
    # provider deny pattern.
    "DriveTest licence",
    "Sault College certificate",
    "Sault Community Career Centre course",
    "WSIB ticket",
    "Skilled Trades Ontario program",
    # Salary / hourly in all forms
    "$ 35 per hour",
    "$35/hour",
    "$35/hr",
    "35/hour",
    "35 per hour role",
    "$50K per year",
    # Scope hooks
    "Express Entry permit",
    "Job Bank reference",
    "IRCC application",
    "WES credential",
    # Control characters
    "ABC\x00licence",
    "ABC\nlicence",
    "ABC\tlicence",
    # Length overflow (>80 chars after collapse)
    "x" * 81,
])
def test_sanitizer_rejects_hostile_display(hostile_display):
    """Round-20 R-5 review: strict allow-list validation. Any URI
    scheme, domain-like token, markup delimiter, email syntax, salary
    language, scope hook, known provider name, control character, or
    character outside the credential-display allow-set REJECTS the
    whole display. The renderer falls back to the no-target
    'Which credential?' template -- safer than stripping fragments."""
    from skillbridge.chat.responder import _sanitize_credential_display
    assert _sanitize_credential_display(hostile_display) == "", (
        f"Hostile display passed validation: {hostile_display!r}"
    )


@pytest.mark.parametrize("legitimate_display", [
    # Real credential names from data/training_registry.yaml
    "310S Automotive Technician License",
    "Class G driver's license",
    "G2/G driver's license",
    "WHMIS 2015 Certificate",
    "Standard First Aid and CPR Level C",
    "Personal Support Worker certification",
    "QuickBooks and basic accounting",
    "Microsoft Excel",
    # Shorter user-typed forms
    "310S",
    "G2",
    "310S licence",
    "Class G",
])
def test_sanitizer_accepts_legitimate_credential_displays(legitimate_display):
    """The strict validation MUST NOT over-reject real credential
    displays. These are the exact strings the snapshot stores and the
    renderer needs to interpolate."""
    from skillbridge.chat.responder import _sanitize_credential_display
    out = _sanitize_credential_display(legitimate_display)
    assert out == legitimate_display, (
        f"Legitimate display was rejected or modified: "
        f"{legitimate_display!r} -> {out!r}"
    )


@pytest.mark.parametrize("hostile_display", [
    "evil.ca/path",
    "ftp://evil.ca",
    "javascript:alert(1)",
    "mailto:test@evil.ca",
    "<a href=evil>X</a>",
    "[click](http://x)",
    "PreyTech.ca",
    "$ 35 per hour",
    "Express Entry permit",
])
def test_no_hostile_content_leaks_through_renderer(hostile_display):
    """End-to-end: a hostile display reaches `_render_clarification`
    via a forged cookie / malformed snapshot. The hostile content
    MUST NOT appear in the rendered output. The renderer may either
    fall back to the canonical (when canonical is safe) or to the
    no-target template (when canonical is also hostile / None) -- the
    invariant is that the response contains zero unsafe substitutions."""
    payload = {
        "kind": "credential_completion_confirmation",
        "credential_canonical": "X-CANON",      # safe canonical
        "credential_display":   hostile_display,
        "action":               "add",
    }
    out = _render_clarification(payload)
    # The hostile content must not appear verbatim anywhere
    for marker in (
        "evil.ca", "javascript:", "mailto:", "ftp://", "http://", "https://",
        "<", ">", "[click]", "PreyTech", "$", "/hour", "per hour",
        "Express Entry",
    ):
        if marker.lower() in hostile_display.lower():
            assert marker.lower() not in out.lower(), (
                f"Hostile marker {marker!r} leaked through renderer for "
                f"input {hostile_display!r}: {out!r}"
            )


@pytest.mark.parametrize("hostile_display", [
    "evil.ca/path",
    "ftp://evil.ca",
    "<a href=evil>X</a>",
])
def test_renderer_falls_back_to_no_target_when_no_safe_id_available(hostile_display):
    """When the display is hostile AND the canonical is None, the
    renderer falls back to the no-target 'Which credential?' template."""
    payload = {
        "kind": "credential_completion_confirmation",
        "credential_canonical": None,
        "credential_display":   hostile_display,
        "action":               "add",
    }
    out = _render_clarification(payload)
    assert "which credential" in out.lower()


def test_renderer_falls_back_when_canonical_is_also_hostile():
    """Both display AND canonical are hostile -- the renderer must
    still emit the safe no-target template, not some partial leak."""
    payload = {
        "kind": "credential_completion_confirmation",
        "credential_canonical": "http://canonical-evil.ca",
        "credential_display":   "javascript:alert(1)",
        "action":               "add",
    }
    out = _render_clarification(payload)
    assert "which credential" in out.lower()
    for marker in ("javascript", "canonical-evil", "http://", "://"):
        assert marker not in out.lower()


def test_grounded_provider_clause_honors_lead_display():
    """Round-19 finding 3: `_grounded_provider_clause` must only
    surface a provider when its gap key matches the lead_display.
    Pre-fix the helper returned the first provider from ANY list."""
    from skillbridge.chat.responder import _grounded_provider_clause
    training = {
        "gap:G2/G driver license": [
            {"provider": "DriveTest", "summary": "x"},
        ],
        "gap:310S certification": [
            {"provider": "Skilled Trades Ontario", "summary": "x"},
        ],
    }
    # Lead is G2/G -> DriveTest
    assert "DriveTest" in _grounded_provider_clause(
        training, "G2/G driver license",
    )
    # Lead is 310S -> Skilled Trades Ontario (NOT DriveTest)
    out = _grounded_provider_clause(training, "310S certification")
    assert "Skilled Trades Ontario" in out
    assert "DriveTest" not in out


def test_grounded_provider_clause_empty_when_lead_has_no_training_entry():
    """When the lead credential has no matching training entry, the
    helper MUST return "" rather than surfacing some other gap's
    provider."""
    from skillbridge.chat.responder import _grounded_provider_clause
    training = {
        "gap:Other credential": [
            {"provider": "DriveTest", "summary": "x"},
        ],
    }
    assert _grounded_provider_clause(training, "310S certification") == ""


# ============================================================================
# _present_remaining_gaps_fallback_v2 -- deterministic narration
# ============================================================================
def test_fallback_uses_conditional_tense_when_any_hypothetical_true():
    payload = {
        "role": "310S Licensed Automotive Technician",
        "employer": "Great Lakes Honda",
        "assumed_completed_credentials": [
            {"display": "310S licence", "canonical": "X", "mode": "hypothetical"},
        ],
        "remaining_credentials": [
            {"display": "G2/G driver's licence", "canonical": "Y"},
        ],
        "remaining_core_skills": ["Honda experience"],
        "any_hypothetical": True,
    }
    inp = _input(
        _decision_explain_remaining_gaps(),
        remaining_gaps_payload=payload,
    )
    out = _present_remaining_gaps_fallback_v2(inp)
    assert "If you've got" in out
    # No past-tense framing
    assert "With your 310S done" not in out


def test_fallback_uses_past_tense_when_any_hypothetical_false():
    payload = {
        "role": "310S Licensed Automotive Technician",
        "assumed_completed_credentials": [
            {"display": "310S licence", "canonical": "X", "mode": "claimed"},
        ],
        "remaining_credentials": [
            {"display": "G2/G driver's licence", "canonical": "Y"},
        ],
        "remaining_core_skills": [],
        "any_hypothetical": False,
    }
    inp = _input(
        _decision_explain_remaining_gaps(),
        remaining_gaps_payload=payload,
    )
    out = _present_remaining_gaps_fallback_v2(inp)
    assert "With" in out
    assert "If you've got" not in out


def test_fallback_does_not_name_provider_without_training_block():
    """Design §6 + §9: providers are named ONLY from TRAINING. With
    training_by_job={} the fallback must NOT mention any provider."""
    payload = {
        "role": "X", "remaining_credentials": [
            {"display": "G2/G driver's licence", "canonical": "G"},
        ],
        "remaining_core_skills": [], "any_hypothetical": False,
        "assumed_completed_credentials": [],
    }
    inp = _input(
        _decision_explain_remaining_gaps(),
        remaining_gaps_payload=payload,
        training_by_job={},
    )
    out = _present_remaining_gaps_fallback_v2(inp).lower()
    for provider in ("drivetest", "ontario.ca", "sault community",
                     "sault college", "skilled trades ontario"):
        assert provider not in out


def test_fallback_names_provider_only_when_training_block_supplies_it():
    payload = {
        "role": "X", "remaining_credentials": [
            {"display": "G2/G driver's licence", "canonical": "G"},
        ],
        "remaining_core_skills": [], "any_hypothetical": False,
        "assumed_completed_credentials": [],
    }
    training = {
        "gap:G2/G driver's licence": [
            {"provider": "DriveTest", "summary": "road test"},
        ],
    }
    inp = _input(
        _decision_explain_remaining_gaps(),
        remaining_gaps_payload=payload,
        training_by_job=training,
    )
    out = _present_remaining_gaps_fallback_v2(inp)
    assert "DriveTest" in out


def test_fallback_all_closed_does_not_name_any_provider():
    """Design §6: when remaining_credentials is empty, NO provider may
    be named even if training_by_job has entries from a prior turn."""
    payload = {
        "role": "X", "remaining_credentials": [],
        "remaining_core_skills": ["Honda experience"],
        "any_hypothetical": False,
        "assumed_completed_credentials": [
            {"display": "310S", "canonical": "X", "mode": "claimed"},
            {"display": "G2/G", "canonical": "Y", "mode": "claimed"},
        ],
    }
    training_leak = {
        "gap:G2/G driver's licence": [
            {"provider": "DriveTest", "summary": "x"},
        ],
    }
    inp = _input(
        _decision_explain_remaining_gaps(),
        remaining_gaps_payload=payload,
        training_by_job=training_leak,
    )
    out = _present_remaining_gaps_fallback_v2(inp).lower()
    assert "drivetest" not in out


# ============================================================================
# Policy: forbidden framing + speculation
# ============================================================================
@pytest.mark.parametrize("offending", [
    "You're a good fit for this role",
    "Looks like a good match for you",
    "You qualify for the role",
    "You're qualified for it",
    "Stretch match!",
])
def test_policy_forbids_match_framing_on_explain_remaining_gaps(offending):
    inp = _input(_decision_explain_remaining_gaps())
    assert _policy_ok_v2(offending, *_iv(inp)) is False


@pytest.mark.parametrize("speculation", [
    "Those usually come on the job.",
    "Honda dealership experience typically comes with experience.",
    "These are best learned through hands-on time at a shop.",
    "It usually takes a course to get there.",
    "That comes with time.",
    "You'll pick that up on the job.",
    "Diagnostic skills typically come on the job.",
])
def test_policy_forbids_speculation_on_explain_remaining_gaps(speculation):
    inp = _input(_decision_explain_remaining_gaps())
    assert _policy_ok_v2(speculation, *_iv(inp)) is False


def test_policy_allows_grounded_naming_of_remaining_gaps():
    """A reply that just names the remaining gaps without speculation
    or match framing must pass policy."""
    grounded = (
        "If you've got the 310S in hand, the next required item is your "
        "G2/G driver's licence. DriveTest can point you at the next step. "
        "Want to check what local shops are hiring while you work toward it?"
    )
    inp = _input(
        _decision_explain_remaining_gaps(),
        training_by_job={
            "gap:G2/G driver's licence": [
                {"provider": "DriveTest", "summary": "road test"},
            ],
        },
    )
    assert _policy_ok_v2(grounded, *_iv(inp)) is True


def test_policy_still_runs_existing_rules_on_explain_remaining_gaps():
    """Pre-R-5 rules (out-of-region, immigration-tier, credential
    equivalence, providers, scope) MUST still apply on the new turn
    type. Spot-check with a $ amount."""
    inp = _input(_decision_explain_remaining_gaps())
    assert _policy_ok_v2("The 310S typically pays $35/hour around here.", *_iv(inp)) is False
