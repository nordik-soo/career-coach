"""AR-9.bug.2b: render the validated adjacent-role posting URL inline.

Narrow scope: surface the already-validated SanitizedURL from the
view's FallbackAdjacentRoleView in `_describe_adjacent_role_fallback_v2`
so the user doesn't need a second round-trip to see it.

No new architecture. One field added to FallbackAdjacentRoleView
(url: SanitizedURL | None), populated with the same sanitized value
that already drives `has_validated_url`. Renderer reads `.url.raw`.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.nodb


from skillbridge.chat.arbiter import ArbiterDecision
from skillbridge.chat.responder import (
    ResponderV2Input,
    _describe_adjacent_role_fallback_v2,
)
from skillbridge.chat.url_views import (
    FallbackAdjacentRoleView,
    SanitizedURL,
    build_sanitized_responder_view_v2,
)


# =========================================================================
# Helpers
# =========================================================================
def _decision(move: str = "describe_adjacent_role") -> ArbiterDecision:
    return ArbiterDecision(
        final_move=move,
        reason_code="x",
        tone="brief_confident",
        arbiter_action="handler_synthesized_describe_adjacent_role",
        ask_slot=None,
        caps_applied=(),
    )


def _input(payload) -> ResponderV2Input:
    return ResponderV2Input(
        user_message="tell me about that role",
        decision=_decision(),
        results=[],
        training_by_job={},
        next_skill=(None, 0),
        band_signal="none",
        requires_consent=False,
        target_role_text="welder",
        resume_facts=None,
        conversation_context=None,
        adjacent_role_description_payload=payload,
    )


# =========================================================================
# Field-level: FallbackAdjacentRoleView.url is populated
# =========================================================================
def test_fallback_adjacent_role_view_exposes_validated_url():
    """The view's url field carries the SanitizedURL when the raw
    payload had a valid URL.
    """
    inp = _input({
        "job": {"title": "Welder", "employer": "ACME",
                "location": "Sault Ste. Marie",
                "url": "https://example.com/jobs/123"},
        "evidence_summary": "3 of 5",
        "matched_skills": ["welding"],
        "expired": False,
    })
    view = build_sanitized_responder_view_v2(inp)
    assert view.fallback_adjacent_role is not None
    assert view.fallback_adjacent_role.url is not None
    assert isinstance(view.fallback_adjacent_role.url, SanitizedURL)
    assert view.fallback_adjacent_role.url.raw == "https://example.com/jobs/123"
    # has_validated_url stays as a defensive cross-check; both fields
    # agree.
    assert view.fallback_adjacent_role.has_validated_url is True


def test_fallback_adjacent_role_view_url_none_when_payload_missing_url():
    inp = _input({
        "job": {"title": "Welder", "employer": "ACME",
                "location": "Sault Ste. Marie"},  # no url
        "evidence_summary": "3 of 5",
        "matched_skills": [],
        "expired": False,
    })
    view = build_sanitized_responder_view_v2(inp)
    assert view.fallback_adjacent_role.url is None
    assert view.fallback_adjacent_role.has_validated_url is False


def test_fallback_adjacent_role_view_url_none_when_payload_url_invalid():
    """A URL that fails structural validation (e.g., ftp://) is
    stripped at projection. url=None, has_validated_url=False.
    """
    inp = _input({
        "job": {"title": "T", "url": "ftp://bad.example.com"},
        "evidence_summary": "",
        "matched_skills": [],
        "expired": False,
    })
    view = build_sanitized_responder_view_v2(inp)
    assert view.fallback_adjacent_role.url is None
    assert view.fallback_adjacent_role.has_validated_url is False


# =========================================================================
# Render: the URL appears inline in the fallback reply
# =========================================================================
def test_describe_fallback_renders_url_inline_when_present():
    inp = _input({
        "job": {"title": "Welder", "employer": "ACME",
                "location": "Sault Ste. Marie",
                "url": "https://example.com/jobs/123"},
        "evidence_summary": "3 of 5",
        "matched_skills": ["welding"],
        "expired": False,
    })
    view = build_sanitized_responder_view_v2(inp)
    reply = _describe_adjacent_role_fallback_v2(inp, view)
    assert "https://example.com/jobs/123" in reply
    # The "Want the posting URL?" round-trip wording is gone when URL present
    assert "Want the posting URL?" not in reply
    # The next-step offer is still there
    assert "Want me to look at the path to apply?" in reply


def test_describe_fallback_renders_path_to_apply_when_url_missing():
    """When url is None (absent or stripped), the fallback offers
    the path-to-apply next step but does NOT mention or invent a URL.
    """
    inp = _input({
        "job": {"title": "Welder", "employer": None,
                "location": "Sault Ste. Marie"},
        "evidence_summary": "ev",
        "matched_skills": [],
        "expired": False,
    })
    view = build_sanitized_responder_view_v2(inp)
    reply = _describe_adjacent_role_fallback_v2(inp, view)
    assert "Want me to look at the path to apply?" in reply
    assert "https://" not in reply
    assert "URL" not in reply


def test_describe_fallback_does_not_render_url_when_invalid():
    """Invalid source URL (e.g., ftp://) is stripped at projection.
    The fallback behaves as if no URL was provided.
    """
    inp = _input({
        "job": {"title": "T", "url": "ftp://bad.invalid"},
        "evidence_summary": "ev",
        "matched_skills": [],
        "expired": False,
    })
    view = build_sanitized_responder_view_v2(inp)
    reply = _describe_adjacent_role_fallback_v2(inp, view)
    assert "ftp" not in reply
    assert "bad.invalid" not in reply
    assert "Want me to look at the path to apply?" in reply


def test_describe_fallback_url_uses_raw_form_not_canonical():
    """Rendering uses SanitizedURL.raw (matches the locked bug.2a
    rendering rule: .raw for output, .canonical for matching).
    """
    inp = _input({
        "job": {"title": "T", "employer": "E",
                "url": "https://EXAMPLE.com/Path"},  # original case
        "evidence_summary": "",
        "matched_skills": [],
        "expired": False,
    })
    view = build_sanitized_responder_view_v2(inp)
    reply = _describe_adjacent_role_fallback_v2(inp, view)
    # Raw preserves original host case
    assert "https://EXAMPLE.com/Path" in reply


# =========================================================================
# Existing behavior preserved: expired and missing-job paths unchanged
# =========================================================================
def test_describe_fallback_expired_unchanged():
    inp = _input({
        "job": {"title": "T", "url": "https://example.com/x"},
        "evidence_summary": "",
        "matched_skills": [],
        "expired": True,
    })
    view = build_sanitized_responder_view_v2(inp)
    reply = _describe_adjacent_role_fallback_v2(inp, view)
    assert "no longer on the board" in reply
    # URL not shown on the expired path
    assert "https://example.com/x" not in reply


def test_describe_fallback_no_job_mapping_unchanged():
    inp = _input({
        "job": None,
        "evidence_summary": "",
        "matched_skills": [],
        "expired": False,
    })
    view = build_sanitized_responder_view_v2(inp)
    reply = _describe_adjacent_role_fallback_v2(inp, view)
    assert "no longer on the board" in reply


# =========================================================================
# Bug.2b allowlist consistency: fallback_urls must reflect what the
# fallback renders. Bug.2a left this hardcoded to frozenset(); bug.2b
# corrects it.
# =========================================================================
def test_fallback_urls_includes_describe_adjacent_role_url():
    """The canonical URL the fallback renders must appear in
    view.fallback_urls. Otherwise downstream consumers would see an
    empty allowlist while the fallback surfaces the URL — an internal
    contradiction.
    """
    inp = _input({
        "job": {"title": "T", "url": "https://example.com/jobs/123"},
        "evidence_summary": "",
        "matched_skills": [],
        "expired": False,
    })
    view = build_sanitized_responder_view_v2(inp)
    assert view.fallback_urls == frozenset({"https://example.com/jobs/123"})
    # And prompt_urls still contains it (sub-step 3 invariant unchanged)
    assert view.prompt_urls == frozenset({"https://example.com/jobs/123"})


def test_fallback_urls_empty_when_no_validated_url():
    """No URL on the payload -> nothing to add to fallback_urls."""
    inp = _input({
        "job": {"title": "T"},  # no url
        "evidence_summary": "",
        "matched_skills": [],
        "expired": False,
    })
    view = build_sanitized_responder_view_v2(inp)
    assert view.fallback_urls == frozenset()


def test_fallback_urls_empty_when_url_invalid():
    """Invalid source URL is stripped at projection -> not in fallback_urls."""
    inp = _input({
        "job": {"title": "T", "url": "ftp://bad.invalid"},
        "evidence_summary": "",
        "matched_skills": [],
        "expired": False,
    })
    view = build_sanitized_responder_view_v2(inp)
    assert view.fallback_urls == frozenset()
