"""AR-9.bug.2a sub-step 5: URL grounding enforcement in policy gates.

Tests cover:
  - URL_NOT_IN_TURN_ALLOWLIST: LLM emits URL not in view.prompt_urls -> rejected
  - Each structural code reachable through policy integration:
    URL_UNSUPPORTED_SCHEME, URL_CREDENTIALS_PRESENT, URL_DISALLOWED_PORT,
    URL_MALFORMED, URL_OVER_LIMIT.
    (URL_CONTROL_CHARS is NOT reachable from policy integration: the
    extraction right-boundary scan breaks at any control char or
    whitespace, so a control char in the reply truncates the URL token
    BEFORE it reaches validate(). URL_CONTROL_CHARS is verified at the
    primitives level in test_url_policy_primitives.py.)
  - Allowed URL: LLM emits URL in prompt_urls -> passes the URL check
  - Multiple URLs in reply: all-valid passes; first violation rejects
  - Case-folded canonical match (HTTPS://host vs https://host)
  - No URLs in reply: URL check is a no-op (existing rules still apply)
  - V1 _policy_ok grounding tests (parity with V2)
  - Sub-step 5 amendment: clickable non-HTTPS schemes (javascript:,
    data:, mailto:, etc.) are extracted and rejected
  - Markdown autolinks `<https://...>` extract the bare URL correctly

Telemetry surface (locked safe_telemetry_fields contract) is exercised
indirectly: the policy gate returns False when a violation occurs.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.nodb


from skillbridge.chat.arbiter import ArbiterDecision
from skillbridge.chat.intake_state import ACTION_PRESENT_MATCHES, Decision
from skillbridge.chat.responder import (
    ResponderInput,
    ResponderV2Input,
    _policy_ok,
    _policy_ok_v2,
)
from skillbridge.chat.url_policy import (
    Validated,
    Violation,
    ViolationCode,
    check_url_membership,
    safe_telemetry_fields,
    validate,
)
from skillbridge.chat.url_views import (
    build_sanitized_responder_view_v1,
    build_sanitized_responder_view_v2,
)
from tests._view_fixtures import view_with_prompt_urls


# =========================================================================
# Helpers
# =========================================================================
def _decision(move: str = "present_matches") -> ArbiterDecision:
    return ArbiterDecision(
        final_move=move,
        reason_code="x",
        tone="brief_confident",
        arbiter_action="passed_planner_through",
        ask_slot=None,
        caps_applied=(),
    )


def _input(move: str = "present_matches") -> ResponderV2Input:
    return ResponderV2Input(
        user_message="hi",
        decision=_decision(move),
        results=[],
        training_by_job={},
        next_skill=(None, 0),
        band_signal="strong_or_good",
        requires_consent=False,
        target_role_text="warehouse worker",
        resume_facts=None,
        conversation_context=None,
    )


# =========================================================================
# URL_NOT_IN_TURN_ALLOWLIST: LLM emits URL outside the move-gated allowlist
# =========================================================================
def test_policy_ok_v2_rejects_url_not_in_allowlist():
    """Reply with a structurally-valid URL that isn't in view.prompt_urls
    must be rejected. This is the central bug.2a behavior — the LLM
    cannot invent URLs.
    """
    view = view_with_prompt_urls({"https://example.com/jobs/123"})
    reply = "Try this opening: https://example.com/jobs/999"
    assert _policy_ok_v2(reply, _input(), view) is False


def test_policy_ok_v2_accepts_url_in_allowlist():
    """A URL that canonicalizes to a member of view.prompt_urls
    survives the URL check (other rules may still reject for other
    reasons; this test uses a benign body).
    """
    view = view_with_prompt_urls({"https://example.com/jobs/123"})
    reply = "Here's the role: https://example.com/jobs/123"
    assert _policy_ok_v2(reply, _input(), view) is True


def test_policy_ok_v2_empty_allowlist_rejects_any_url():
    """When view.prompt_urls is empty (e.g., a move with no URL
    surface), any URL in the reply triggers rejection.
    """
    view = view_with_prompt_urls(set())
    reply = "Check https://example.com"
    assert _policy_ok_v2(reply, _input(), view) is False


# =========================================================================
# Structural violations triggered from LLM output
# =========================================================================
def test_policy_ok_v2_rejects_url_unsupported_scheme():
    """LLM emits ftp:// -> URL_UNSUPPORTED_SCHEME -> rejected."""
    view = view_with_prompt_urls(set())
    reply = "Get the file at ftp://example.com/file.zip"
    assert _policy_ok_v2(reply, _input(), view) is False


def test_policy_ok_v2_rejects_javascript_scheme():
    view = view_with_prompt_urls(set())
    reply = "Click javascript://alert(1)"
    assert _policy_ok_v2(reply, _input(), view) is False


def test_policy_ok_v2_rejects_url_credentials_present():
    view = view_with_prompt_urls(set())
    reply = "Visit https://user:pass@example.com/path"
    assert _policy_ok_v2(reply, _input(), view) is False


def test_policy_ok_v2_rejects_url_disallowed_port():
    view = view_with_prompt_urls(set())
    reply = "Visit https://example.com:8080/path"
    assert _policy_ok_v2(reply, _input(), view) is False


def test_policy_ok_v2_rejects_ipv6_bracketed_authority():
    """IPv6 bracketed authority -> URL_MALFORMED."""
    view = view_with_prompt_urls(set())
    reply = "Try https://[::1]/path"
    assert _policy_ok_v2(reply, _input(), view) is False


def test_policy_ok_v2_rejects_over_limit_url():
    """A URL longer than MAX_URL_LENGTH (512) bytes -> URL_OVER_LIMIT."""
    view = view_with_prompt_urls(set())
    long_url = "https://example.com/" + ("a" * 600)
    reply = f"See {long_url}"
    assert _policy_ok_v2(reply, _input(), view) is False


# =========================================================================
# Canonical matching
# =========================================================================
def test_policy_ok_v2_case_folded_host_matches():
    """LLM emits HTTPS://EXAMPLE.com/x (uppercase). The canonical form
    is https://example.com/x, which is in the allowlist.
    """
    view = view_with_prompt_urls({"https://example.com/x"})
    reply = "See HTTPS://EXAMPLE.com/x"
    assert _policy_ok_v2(reply, _input(), view) is True


def test_policy_ok_v2_distinct_port_distinct_match():
    """`:443` and absent port are DISTINCT canonical forms per the
    locked port policy. Allowlist with port-absent does NOT match a
    reply with :443.
    """
    view = view_with_prompt_urls({"https://example.com/x"})
    reply = "See https://example.com:443/x"
    assert _policy_ok_v2(reply, _input(), view) is False

    view2 = view_with_prompt_urls({"https://example.com:443/x"})
    reply2 = "See https://example.com:443/x"
    assert _policy_ok_v2(reply2, _input(), view2) is True


def test_policy_ok_v2_distinct_trailing_slash():
    """`/jobs/123` and `/jobs/123/` are DISTINCT canonical forms."""
    view = view_with_prompt_urls({"https://example.com/jobs/123"})
    reply = "See https://example.com/jobs/123/"
    assert _policy_ok_v2(reply, _input(), view) is False


# =========================================================================
# Multiple URLs
# =========================================================================
def test_policy_ok_v2_all_urls_in_allowlist_pass():
    view = view_with_prompt_urls({
        "https://example.com/a", "https://example.com/b",
    })
    reply = "First https://example.com/a then https://example.com/b"
    assert _policy_ok_v2(reply, _input(), view) is True


def test_policy_ok_v2_first_violation_rejects():
    """When multiple URLs are present and any one violates, the gate
    returns False (we stop at the first violation, which is enough to
    reject the whole reply).
    """
    view = view_with_prompt_urls({"https://example.com/a"})
    reply = "First https://example.com/a then ftp://bad.com/x"
    assert _policy_ok_v2(reply, _input(), view) is False


# =========================================================================
# No URLs in reply
# =========================================================================
def test_policy_ok_v2_url_check_is_noop_when_no_urls():
    """A reply with no URL-shaped tokens passes the URL check and
    falls through to existing rules. Use a benign body that passes
    everything else.
    """
    view = view_with_prompt_urls(set())
    reply = "Got it. What kind of work would you like me to focus on?"
    assert _policy_ok_v2(reply, _input(), view) is True


# =========================================================================
# Prose discrimination — URL grounding doesn't false-positive on prose
# =========================================================================
def test_policy_ok_v2_does_not_extract_ordinary_prose_colon():
    """`Note:` / `ratio 16:9` / `12:00 PM` are not URLs — no extraction,
    no membership check.
    """
    view = view_with_prompt_urls(set())
    for reply in [
        "Note: this is the locked phrase.",
        "The aspect ratio is 16:9.",
        "Come back at 12:00 PM.",
        "See section 4: example.",
    ]:
        assert _policy_ok_v2(reply, _input(), view) is True, reply


# =========================================================================
# Surrounding punctuation
# =========================================================================
def test_policy_ok_v2_trailing_period_stripped_before_check():
    """`https://example.com/x.` is extracted as `https://example.com/x`
    (trailing period stripped). If the latter is in the allowlist,
    the reply passes.
    """
    view = view_with_prompt_urls({"https://example.com/x"})
    reply = "Visit https://example.com/x."
    assert _policy_ok_v2(reply, _input(), view) is True


def test_policy_ok_v2_parenthesized_url_stripped_before_check():
    view = view_with_prompt_urls({"https://example.com/x"})
    reply = "(see https://example.com/x)"
    assert _policy_ok_v2(reply, _input(), view) is True


# =========================================================================
# Telemetry shape verification (independent of log emission)
# =========================================================================
def test_telemetry_fields_for_not_in_allowlist_violation():
    """A URL_NOT_IN_TURN_ALLOWLIST violation produces the locked
    5-field telemetry shape. Verifies indirectly via check_url_membership.
    """
    result = check_url_membership(
        "https://example.com/x", frozenset({"https://other.com/y"}),
    )
    assert isinstance(result, Violation)
    fields = safe_telemetry_fields(result, move="present_matches")
    assert set(fields.keys()) == {
        "violation_code", "move", "scheme", "host", "url_hash",
    }
    assert fields["violation_code"] == "URL_NOT_IN_TURN_ALLOWLIST"
    assert fields["scheme"] == "https"
    assert fields["host"] == "example.com"
    # url_hash is SHA-256 lowercase hex of raw token
    import hashlib
    assert fields["url_hash"] == hashlib.sha256(
        b"https://example.com/x"
    ).hexdigest()


def test_telemetry_fields_never_contain_path_or_query():
    """No path/query/fragment/credentials/raw URL appears in
    telemetry under any key.
    """
    sensitive = "https://example.com/jobs?token=secret123&user=alice"
    result = check_url_membership(sensitive, frozenset())
    assert isinstance(result, Violation)
    fields = safe_telemetry_fields(result, move="present_matches")
    for key, value in fields.items():
        if key in ("violation_code", "move", "url_hash"):
            continue
        if value is None:
            continue
        assert "secret123" not in value
        assert "alice" not in value
        assert "/jobs" not in value
        assert "?" not in value


# =========================================================================
# Integration test: full v2 view from realistic inp
# =========================================================================
def test_policy_ok_v2_with_realistic_view_from_present_matches():
    """End-to-end: build a real view from a present_matches input,
    confirm a result URL in the view's allowlist is accepted; a
    sibling URL not in the view is rejected.
    """
    inp = ResponderV2Input(
        user_message="show me jobs",
        decision=_decision("present_matches"),
        results=[
            {"title": "T", "employer": "E",
             "url": "https://example.com/jobs/123",
             "job_id": "j1", "match_band": "good",
             "matched_skills": [], "missing_skills": []},
        ],
        training_by_job={},
        next_skill=(None, 0),
        band_signal="strong_or_good",
        requires_consent=False,
        target_role_text="warehouse worker",
        resume_facts=None,
        conversation_context=None,
    )
    view = build_sanitized_responder_view_v2(inp)
    assert "https://example.com/jobs/123" in view.prompt_urls
    # Allowlisted URL passes
    good_reply = "Here it is: https://example.com/jobs/123"
    assert _policy_ok_v2(good_reply, inp, view) is True
    # Sibling URL not in allowlist rejected
    bad_reply = "Try https://example.com/jobs/999 instead"
    assert _policy_ok_v2(bad_reply, inp, view) is False


# =========================================================================
# URL check runs BEFORE other checks (fail-fast)
# =========================================================================
def test_policy_ok_v2_url_check_runs_before_other_rules():
    """When a reply contains both a URL violation AND a regex
    violation (e.g., out-of-region), URL grounding fires first.
    Verified by checking that an out-of-region offer with a bad URL
    is rejected (both would reject; this test mainly documents the
    order).
    """
    view = view_with_prompt_urls(set())
    reply = "Try Toronto. Look at https://toronto.com/jobs"
    assert _policy_ok_v2(reply, _input(), view) is False


# =========================================================================
# Sub-step 5 amendment: dangerous clickable non-HTTPS schemes
# =========================================================================
@pytest.mark.parametrize("reply,scheme", [
    ("Click javascript:alert(1) to test", "javascript"),
    ("Open data:text/html,<html>x</html> here", "data"),
    ("Email mailto:user@example.com please", "mailto"),
    ("Run vbscript:msgbox(1) now", "vbscript"),
    ("Open file:///etc/passwd directly", "file"),
    ("Call tel:+15555551234 today", "tel"),
    ("Text sms:+15555551234 now", "sms"),
    # Case insensitivity
    ("Click JavaScript:foo to test", "javascript"),
    ("Email MAILTO:user@example.com", "mailto"),
])
def test_policy_ok_v2_rejects_dangerous_scheme_only_uris(reply, scheme):
    """Clickable non-HTTPS schemes without `://` are now extracted and
    rejected as URL_UNSUPPORTED_SCHEME. The sub-step 2 framing of
    'scheme-only URIs are out of scope' was amended in sub-step 5 to
    prevent these from bypassing enforcement in chat / markdown UIs.
    """
    view = view_with_prompt_urls(set())
    assert _policy_ok_v2(reply, _input(), view) is False


# =========================================================================
# Sub-step 5 amendment: Markdown autolink extraction
# =========================================================================
def test_policy_ok_v2_markdown_autolink_with_allowed_url_passes():
    """Markdown autolink `<https://approved-job-url>` — the trailing
    `>` is an unbalanced bracket; the extraction's bracket-balance
    rule (now including `<>`) strips it, leaving the bare URL.
    """
    view = view_with_prompt_urls({"https://example.com/jobs/123"})
    reply = "Try this role: <https://example.com/jobs/123>"
    assert _policy_ok_v2(reply, _input(), view) is True


def test_policy_ok_v2_markdown_autolink_with_dangerous_scheme_rejected():
    """A markdown autolink wrapping a dangerous scheme still gets
    extracted and rejected.
    """
    view = view_with_prompt_urls(set())
    reply = "<javascript:alert(1)>"
    assert _policy_ok_v2(reply, _input(), view) is False


def test_policy_ok_v2_markdown_link_with_allowed_url_passes():
    """Markdown link `[text](https://allowed)` — the existing
    balanced-paren rule strips the trailing `)`.
    """
    view = view_with_prompt_urls({"https://example.com/jobs/123"})
    reply = "[Click here](https://example.com/jobs/123)"
    assert _policy_ok_v2(reply, _input(), view) is True


# =========================================================================
# V1 _policy_ok grounding parity tests
# =========================================================================
def _v1_decision() -> Decision:
    return Decision(
        next_state="present_matches",
        action=ACTION_PRESENT_MATCHES,
        ask_slots=[],
        show_matches=True,
    )


def _v1_input(results=None, training_by_job=None) -> ResponderInput:
    return ResponderInput(
        user_message="show me jobs",
        decision=_v1_decision(),
        results=results or [],
        training_by_job=training_by_job or {},
        next_skill=(None, 0),
        band_signal="strong_or_good",
        requires_consent=False,
        target_role_text="warehouse worker",
        resume_facts=None,
    )


def test_policy_ok_v1_rejects_url_not_in_allowlist():
    """V1 _policy_ok rejects URL not in view.prompt_urls — same
    contract as V2.
    """
    view = view_with_prompt_urls({"https://example.com/jobs/123"})
    reply = "Try this opening: https://example.com/jobs/999"
    assert _policy_ok(reply, _v1_input(), view) is False


def test_policy_ok_v1_accepts_url_in_allowlist():
    view = view_with_prompt_urls({"https://example.com/jobs/123"})
    reply = "Here's the role: https://example.com/jobs/123"
    assert _policy_ok(reply, _v1_input(), view) is True


def test_policy_ok_v1_rejects_dangerous_scheme():
    view = view_with_prompt_urls(set())
    reply = "Click javascript:alert(1) now"
    assert _policy_ok(reply, _v1_input(), view) is False


def test_policy_ok_v1_rejects_url_unsupported_scheme():
    view = view_with_prompt_urls(set())
    reply = "Visit ftp://example.com/file.zip"
    assert _policy_ok(reply, _v1_input(), view) is False


def test_policy_ok_v1_rejects_url_credentials_present():
    view = view_with_prompt_urls(set())
    reply = "See https://user:pass@example.com/path"
    assert _policy_ok(reply, _v1_input(), view) is False


def test_policy_ok_v1_with_realistic_view():
    """End-to-end: build a real v1 view; sibling URL not in allowlist
    rejected; in-allowlist URL passes.
    """
    inp = _v1_input(
        results=[{
            "title": "T", "employer": "E",
            "url": "https://example.com/jobs/123",
            "job_id": "j1", "match_band": "good",
            "matched_skills": [], "missing_skills": [],
        }],
    )
    view = build_sanitized_responder_view_v1(inp)
    assert "https://example.com/jobs/123" in view.prompt_urls
    assert _policy_ok(
        "Here it is: https://example.com/jobs/123", inp, view,
    ) is True
    assert _policy_ok(
        "Try https://example.com/jobs/999 instead", inp, view,
    ) is False


def test_policy_ok_v1_markdown_autolink_handled():
    view = view_with_prompt_urls({"https://example.com/jobs/123"})
    reply = "Try this: <https://example.com/jobs/123>"
    assert _policy_ok(reply, _v1_input(), view) is True


def test_policy_ok_v1_no_urls_passes():
    view = view_with_prompt_urls(set())
    reply = "Tell me a bit more about what you're looking for."
    assert _policy_ok(reply, _v1_input(), view) is True
