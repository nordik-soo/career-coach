"""AR-9.bug.2a sub-step 2: isolated url_policy.py primitives.

Unit coverage for the six structural violation codes, extraction
boundaries (generic scheme grammar + prose discrimination + punctuation
rules), SHA-256 hashing over the extracted token, character-for-character
canonicalization, the lexical port policy (rejecting :0443 et al.),
IPv6/non-ASCII host rejection, the locked safe_scheme / safe_host
invariants enforced by __post_init__, the discriminated Violation |
Validated return type, and the safe_telemetry_fields shape.

No responder dependencies. No view construction. No consumer migration.
Those land in sub-steps 3+.
"""
from __future__ import annotations

import dataclasses
import hashlib

import pytest

pytestmark = pytest.mark.nodb


from skillbridge.chat.url_policy import (
    MAX_URL_LENGTH,
    RawCandidate,
    Validated,
    Violation,
    ViolationCode,
    check_url_membership,
    extract_url_candidates,
    hash_raw_token,
    safe_telemetry_fields,
    validate,
)


# =========================================================================
# hash_raw_token
# =========================================================================
def test_hash_is_sha256_lowercase_hex_of_utf8_bytes():
    token = "https://example.com/path"
    expected = hashlib.sha256(token.encode("utf-8")).hexdigest()
    assert hash_raw_token(token) == expected
    assert expected == expected.lower()
    assert len(expected) == 64


def test_hash_is_deterministic_for_same_input():
    token = "https://example.com/jobs/123"
    assert hash_raw_token(token) == hash_raw_token(token)


def test_hash_differs_for_case_different_inputs():
    a = "https://example.com/PATH"
    b = "https://example.com/path"
    assert hash_raw_token(a) != hash_raw_token(b)


def test_hash_uses_extracted_token_not_further_trimmed():
    """If a caller passes a token with trailing prose punctuation,
    hash hashes the input verbatim. The hash never silently re-strips."""
    with_trail = "https://example.com/x."
    stripped = "https://example.com/x"
    assert hash_raw_token(with_trail) != hash_raw_token(stripped)


def test_hash_does_not_truncate_oversized_input():
    """600-char input hashes the full string, not a prefix."""
    long_token = "https://example.com/" + ("a" * 600)
    assert hash_raw_token(long_token) == hashlib.sha256(
        long_token.encode("utf-8")
    ).hexdigest()


# =========================================================================
# Extraction: scheme breadth (RFC 3986 generic grammar + :// required)
# =========================================================================
@pytest.mark.parametrize("text,expected_token", [
    ("Check out https://example.com/path for info",
     "https://example.com/path"),
    ("Try http://example.com/login",
     "http://example.com/login"),
    ("Download ftp://example.com/file.zip now",
     "ftp://example.com/file.zip"),
    ("Don't click javascript://alert(1)",
     "javascript://alert(1)"),
    ("Open chrome-extension://abc123/page",
     "chrome-extension://abc123/page"),
    ("Use httpx://example.com/path here",
     "httpx://example.com/path"),
])
def test_extracts_generic_scheme_tokens(text, expected_token):
    candidates = list(extract_url_candidates(text))
    assert len(candidates) == 1
    assert candidates[0].extracted_token == expected_token


@pytest.mark.parametrize("text,expected_scheme", [
    ("The data:text/html,<html>foo URI", "data"),
    ("Email mailto:user@example.com please", "mailto"),
    ("Run javascript:void(0) here", "javascript"),
    ("Call tel:+15555551234 today", "tel"),
    ("Use vbscript:msgbox(1) here", "vbscript"),
    ("Open file:///etc/passwd now", "file"),
    ("Text sms:+15555551234 today", "sms"),
])
def test_extracts_dangerous_scheme_only_uris(text, expected_scheme):
    """Sub-step 5 amendment: clickable non-HTTPS schemes without ://
    are now extracted and routed to URL_UNSUPPORTED_SCHEME at
    validation. Treating them as out-of-scope would let them bypass
    URL grounding in chat / markdown UIs.
    """
    candidates = list(extract_url_candidates(text))
    assert len(candidates) == 1, text
    # Validation produces URL_UNSUPPORTED_SCHEME with the matched scheme.
    result = validate(candidates[0].extracted_token)
    assert isinstance(result, Violation)
    assert result.code is ViolationCode.URL_UNSUPPORTED_SCHEME
    assert result.safe_scheme == expected_scheme


# =========================================================================
# Extraction: prose discrimination
# =========================================================================
@pytest.mark.parametrize("text", [
    "Note: see attached file for details",
    "See section 4: example reference",
    "ratio 16:9 aspect",
    "time 12:00 PM",
    "John: hello there",
])
def test_does_not_extract_ordinary_prose_colons(text):
    """`Note:`, `4:`, `16:9`, `12:00`, `John:` have a colon but no
    `://`, so they must not produce URL candidates."""
    assert list(extract_url_candidates(text)) == []


# =========================================================================
# Extraction: punctuation rules
# =========================================================================
@pytest.mark.parametrize("text,expected", [
    # Unconditional trim (one trailing prose char each)
    ("Visit https://example.com/x.",   "https://example.com/x"),
    ("Visit https://example.com/x,",   "https://example.com/x"),
    ("Visit https://example.com/x;",   "https://example.com/x"),
    ("Visit https://example.com/x!",   "https://example.com/x"),
    ("Visit https://example.com/x?",   "https://example.com/x"),
    # Iterative trim (multiple trailing prose chars)
    ("End https://example.com/x!?,;",   "https://example.com/x"),
    ("End https://example.com/x.....",  "https://example.com/x"),
])
def test_trailing_prose_punctuation_stripped(text, expected):
    candidates = list(extract_url_candidates(text))
    assert len(candidates) == 1
    assert candidates[0].extracted_token == expected


def test_unbalanced_closing_paren_stripped():
    """A `)` after the URL with no matching `(` inside is prose."""
    text = "(see https://example.com/jobs/123)"
    candidates = list(extract_url_candidates(text))
    assert len(candidates) == 1
    assert candidates[0].extracted_token == "https://example.com/jobs/123"


def test_balanced_parens_in_path_preserved():
    """Wikipedia-style paths like /Foo_(bar) must NOT be stripped."""
    text = "Visit https://en.wikipedia.org/wiki/Foo_(bar) today"
    candidates = list(extract_url_candidates(text))
    assert len(candidates) == 1
    assert candidates[0].extracted_token == "https://en.wikipedia.org/wiki/Foo_(bar)"


def test_balanced_parens_then_trailing_prose_punctuation_stripped():
    """Balanced parens inside path; trailing `.` is prose; strip only the `.`."""
    text = "Visit https://en.wikipedia.org/wiki/Foo_(bar)."
    candidates = list(extract_url_candidates(text))
    assert len(candidates) == 1
    assert candidates[0].extracted_token == "https://en.wikipedia.org/wiki/Foo_(bar)"


def test_unbalanced_bracket_strip_iterative():
    """Multiple unbalanced closers strip one at a time."""
    text = "(((see https://example.com/x)))"
    candidates = list(extract_url_candidates(text))
    assert len(candidates) == 1
    assert candidates[0].extracted_token == "https://example.com/x"


def test_query_commas_inside_token_preserved():
    """Trailing prose-punctuation rule operates only on the right edge.
    Commas inside the query string are part of the URL."""
    text = "See https://example.com/x?q=a,b for the results."
    candidates = list(extract_url_candidates(text))
    assert len(candidates) == 1
    assert candidates[0].extracted_token == "https://example.com/x?q=a,b"


# =========================================================================
# Extraction: left boundary semantics
# =========================================================================
def test_prefixhttps_extracts_as_single_candidate():
    """`prefixhttps://...` extracts a single candidate with scheme
    `prefixhttps`. Validation marks it URL_UNSUPPORTED_SCHEME with
    safe_scheme=`prefixhttps`, safe_host=None. The lookbehind cannot
    infer that `https` was intended inside the prefix."""
    text = "prefixhttps://example.com/path"
    candidates = list(extract_url_candidates(text))
    assert len(candidates) == 1
    assert candidates[0].extracted_token == "prefixhttps://example.com/path"

    result = validate(candidates[0].extracted_token)
    assert isinstance(result, Violation)
    assert result.code is ViolationCode.URL_UNSUPPORTED_SCHEME
    assert result.safe_scheme == "prefixhttps"
    assert result.safe_host is None


def test_wordhttps_extracts_as_single_candidate():
    """Leftmost match wins; we do NOT also produce a shadow candidate
    starting at the `h` of `https`."""
    text = "wordhttps://example.com/path"
    candidates = list(extract_url_candidates(text))
    assert len(candidates) == 1
    assert candidates[0].extracted_token == "wordhttps://example.com/path"

    result = validate(candidates[0].extracted_token)
    assert isinstance(result, Violation)
    assert result.code is ViolationCode.URL_UNSUPPORTED_SCHEME
    assert result.safe_scheme == "wordhttps"


def test_xhttps_extracts_at_word_start():
    text = "xhttps://example.com/path"
    candidates = list(extract_url_candidates(text))
    assert len(candidates) == 1
    assert candidates[0].extracted_token == "xhttps://example.com/path"


def test_scheme_must_start_with_letter():
    """RFC 3986 schemes start with a letter. A digit-led scheme
    pattern like `12://...` produces no candidate."""
    assert list(extract_url_candidates("12://example.com")) == []


def test_non_word_prefix_lets_letter_scheme_extract_from_next_position():
    """Non-word characters like `+` and `.` don't satisfy `\\w`, so the
    lookbehind passes at the position where the actual letter begins.
    This is the desired behavior: an LLM emitting `+invalid://x` should
    still surface `invalid://x` for URL_UNSUPPORTED_SCHEME classification."""
    for text, expected_token in [
        ("+invalid://x", "invalid://x"),
        (".bad://y",     "bad://y"),
    ]:
        candidates = list(extract_url_candidates(text))
        assert len(candidates) == 1
        assert candidates[0].extracted_token == expected_token
        result = validate(candidates[0].extracted_token)
        assert isinstance(result, Violation)
        assert result.code is ViolationCode.URL_UNSUPPORTED_SCHEME


def test_leading_space_starts_https_extraction():
    text = " https://example.com/path"
    candidates = list(extract_url_candidates(text))
    assert len(candidates) == 1
    assert candidates[0].extracted_token == "https://example.com/path"


# =========================================================================
# Extraction: span correctness
# =========================================================================
def test_span_covers_extracted_token_after_strip():
    text = "Visit https://example.com/x. for info"
    candidates = list(extract_url_candidates(text))
    assert len(candidates) == 1
    rc = candidates[0]
    assert rc.extracted_token == "https://example.com/x"
    assert text[rc.span_start:rc.span_end] == rc.extracted_token


def test_multiple_candidates_in_one_text():
    text = "Try https://example.com/a and also https://example.com/b please"
    candidates = list(extract_url_candidates(text))
    assert [c.extracted_token for c in candidates] == [
        "https://example.com/a",
        "https://example.com/b",
    ]


# =========================================================================
# Validation: each violation code is independently producible
# =========================================================================
def test_url_over_limit_violation():
    long_token = "https://example.com/" + ("a" * 600)
    result = validate(long_token)
    assert isinstance(result, Violation)
    assert result.code is ViolationCode.URL_OVER_LIMIT
    assert result.safe_scheme is None
    assert result.safe_host is None
    # Token retained verbatim; hash is of the full input.
    assert result.raw_token == long_token
    assert result.raw_token_hash == hashlib.sha256(
        long_token.encode("utf-8")
    ).hexdigest()


def test_url_control_chars_violation_for_internal_tab():
    token = "https://example.com/\tpath"
    result = validate(token)
    assert isinstance(result, Violation)
    assert result.code is ViolationCode.URL_CONTROL_CHARS


def test_url_control_chars_violation_for_internal_space():
    token = "https://example.com /path"
    result = validate(token)
    assert isinstance(result, Violation)
    assert result.code is ViolationCode.URL_CONTROL_CHARS


def test_url_control_chars_for_null_byte():
    token = "https://example.com/\x00x"
    result = validate(token)
    assert isinstance(result, Violation)
    assert result.code is ViolationCode.URL_CONTROL_CHARS


def test_url_malformed_no_scheme_syntax():
    """Garbage with no scheme prefix at all -> URL_MALFORMED, safe_scheme None."""
    result = validate("garbage_no_scheme")
    assert isinstance(result, Violation)
    assert result.code is ViolationCode.URL_MALFORMED
    assert result.safe_scheme is None


def test_url_malformed_empty_input():
    result = validate("")
    assert isinstance(result, Violation)
    assert result.code is ViolationCode.URL_MALFORMED
    assert result.safe_scheme is None


@pytest.mark.parametrize("scheme", [
    "http", "ftp", "javascript", "chrome-extension",
    "httpx", "wss", "ws", "file",
])
def test_url_unsupported_scheme_for_non_https(scheme):
    token = f"{scheme}://example.com/path"
    result = validate(token)
    assert isinstance(result, Violation)
    assert result.code is ViolationCode.URL_UNSUPPORTED_SCHEME
    assert result.safe_scheme == scheme.lower()
    assert result.safe_host is None  # locked invariant


def test_url_credentials_present():
    token = "https://user:pass@example.com/path"
    result = validate(token)
    assert isinstance(result, Violation)
    assert result.code is ViolationCode.URL_CREDENTIALS_PRESENT
    assert result.safe_scheme == "https"
    assert result.safe_host is None


def test_url_credentials_present_user_only():
    token = "https://user@example.com/path"
    result = validate(token)
    assert isinstance(result, Violation)
    assert result.code is ViolationCode.URL_CREDENTIALS_PRESENT


def test_url_disallowed_port_emits_safe_host():
    """Host validation passed before port check fired."""
    result = validate("https://example.com:80/path")
    assert isinstance(result, Violation)
    assert result.code is ViolationCode.URL_DISALLOWED_PORT
    assert result.safe_scheme == "https"
    assert result.safe_host == "example.com"


def test_validated_basic_url():
    result = validate("https://example.com/path")
    assert isinstance(result, Validated)
    assert result.canonical == "https://example.com/path"
    assert result.scheme == "https"
    assert result.host == "example.com"


# =========================================================================
# Validation: execution order (first failure wins)
# =========================================================================
def test_over_limit_wins_over_control_chars():
    """600 chars + an internal tab -> OVER_LIMIT, not CONTROL_CHARS."""
    token = "https://example.com/\t" + ("a" * 600)
    result = validate(token)
    assert isinstance(result, Violation)
    assert result.code is ViolationCode.URL_OVER_LIMIT


def test_control_chars_wins_over_credentials():
    """Embedded tab + @ -> CONTROL_CHARS, not CREDENTIALS_PRESENT."""
    token = "https://user@\texample.com/path"
    result = validate(token)
    assert isinstance(result, Violation)
    assert result.code is ViolationCode.URL_CONTROL_CHARS


def test_control_chars_wins_over_unsupported_scheme():
    """ftp scheme + embedded tab -> CONTROL_CHARS first."""
    token = "ftp://example.com\t/path"
    result = validate(token)
    assert isinstance(result, Violation)
    assert result.code is ViolationCode.URL_CONTROL_CHARS


def test_unsupported_scheme_short_circuits_before_credentials():
    """A non-https URL with credentials -> UNSUPPORTED_SCHEME, not CREDS."""
    token = "ftp://user:pass@example.com/file"
    result = validate(token)
    assert isinstance(result, Violation)
    assert result.code is ViolationCode.URL_UNSUPPORTED_SCHEME
    assert result.safe_scheme == "ftp"
    assert result.safe_host is None


def test_credentials_short_circuits_before_port():
    """user@host:80 -> CREDENTIALS_PRESENT, not DISALLOWED_PORT."""
    token = "https://user@example.com:80/path"
    result = validate(token)
    assert isinstance(result, Violation)
    assert result.code is ViolationCode.URL_CREDENTIALS_PRESENT


def test_host_validation_short_circuits_before_port():
    """Invalid host (consecutive dots) -> MALFORMED before port check."""
    token = "https://example..com:80/path"
    result = validate(token)
    assert isinstance(result, Violation)
    assert result.code is ViolationCode.URL_MALFORMED


# =========================================================================
# Port policy (lexical)
# =========================================================================
@pytest.mark.parametrize("port_text", ["0443", "00443", "0", "65536", "99999", "abc", "44a3", ""])
def test_port_malformed_lexical_forms(port_text):
    token = f"https://example.com:{port_text}/path"
    result = validate(token)
    assert isinstance(result, Violation)
    assert result.code is ViolationCode.URL_MALFORMED, (
        f"port {port_text!r} expected URL_MALFORMED, got {result.code}"
    )


@pytest.mark.parametrize("port_text", ["80", "8080", "1", "65535", "8443", "22"])
def test_port_disallowed_for_valid_non_443(port_text):
    token = f"https://example.com:{port_text}/path"
    result = validate(token)
    assert isinstance(result, Violation)
    assert result.code is ViolationCode.URL_DISALLOWED_PORT, (
        f"port {port_text!r} expected URL_DISALLOWED_PORT, got {result.code}"
    )


def test_explicit_443_validates():
    result = validate("https://example.com:443/path")
    assert isinstance(result, Validated)
    assert result.canonical == "https://example.com:443/path"


def test_absent_port_validates():
    result = validate("https://example.com/path")
    assert isinstance(result, Validated)
    assert result.canonical == "https://example.com/path"


def test_absent_port_and_explicit_443_distinct_canonical():
    a = validate("https://example.com/path")
    b = validate("https://example.com:443/path")
    assert isinstance(a, Validated) and isinstance(b, Validated)
    assert a.canonical != b.canonical


# =========================================================================
# IPv6 authorities (out of bug.2a scope; rejected as MALFORMED)
# =========================================================================
@pytest.mark.parametrize("token", [
    "https://[::1]/path",
    "https://[::1]:443/path",
    "https://[2001:db8::1]/path",
    "https://[2001:db8::1]:80/path",
])
def test_ipv6_bracketed_authority_rejected(token):
    result = validate(token)
    assert isinstance(result, Violation)
    assert result.code is ViolationCode.URL_MALFORMED


# =========================================================================
# Non-ASCII host (IDN out of bug.2a scope; rejected as MALFORMED)
# =========================================================================
def test_non_ascii_host_rejected():
    result = validate("https://exämple.com/path")
    assert isinstance(result, Violation)
    assert result.code is ViolationCode.URL_MALFORMED


def test_non_ascii_path_preserved():
    """Non-ASCII codepoints in the PATH are OK; preserved
    character-for-character in canonical."""
    result = validate("https://example.com/päth")
    assert isinstance(result, Validated)
    assert result.canonical == "https://example.com/päth"


# =========================================================================
# DNS host structural rules (RFC 1035 + RFC 1123)
# =========================================================================
@pytest.mark.parametrize("token", [
    "https://-example.com/path",      # leading hyphen on first label
    "https://example-.com/path",      # trailing hyphen on first label
    "https://example.com-/path",      # trailing hyphen on last label
    "https://-/path",                 # single-char hyphen-only label
    "https://foo.-bar.com/path",      # leading hyphen on middle label
    "https://foo.bar-.com/path",      # trailing hyphen on middle label
    "https://-foo-.com/path",         # both leading and trailing hyphen
])
def test_host_label_hyphen_boundaries_rejected(token):
    """Per RFC 1123, labels cannot start or end with a hyphen.
    Internal hyphens remain allowed."""
    result = validate(token)
    assert isinstance(result, Violation)
    assert result.code is ViolationCode.URL_MALFORMED


def test_host_label_exceeding_63_chars_rejected():
    """RFC 1035: each DNS label is at most 63 octets."""
    long_label = "a" * 64
    token = f"https://{long_label}.com/path"
    result = validate(token)
    assert isinstance(result, Violation)
    assert result.code is ViolationCode.URL_MALFORMED


def test_host_label_exactly_63_chars_accepted():
    """Boundary: 63 chars is at the limit, accepted."""
    label = "a" * 63
    token = f"https://{label}.com/path"
    result = validate(token)
    assert isinstance(result, Validated)
    assert result.host == f"{label}.com"


def test_hostname_exceeding_253_chars_rejected():
    """RFC 1035 hostname total cap: > 253 chars rejected."""
    # 4 labels x 63 chars + 3 dots = 255 chars, > 253
    host = ".".join(["a" * 63] * 4)
    assert len(host) == 255
    token = f"https://{host}/path"
    result = validate(token)
    assert isinstance(result, Violation)
    assert result.code is ViolationCode.URL_MALFORMED


def test_hostname_at_253_chars_accepted():
    """Boundary: exactly 253 chars is at the limit, accepted."""
    host = ".".join(["a" * 63, "a" * 63, "a" * 63, "a" * 61])
    assert len(host) == 253
    token = f"https://{host}/path"
    result = validate(token)
    assert isinstance(result, Validated)


def test_internal_hyphen_in_label_accepted():
    """RFC 1123 allows internal hyphens; only leading/trailing forbidden."""
    result = validate("https://a-b-c.example.com/path")
    assert isinstance(result, Validated)


def test_digit_led_label_accepted():
    """RFC 1123 relaxed RFC 952's letter-start rule; digit-led labels
    like 3com.com are valid."""
    result = validate("https://3com.com/path")
    assert isinstance(result, Validated)


def test_multi_label_subdomain_validated():
    """A typical subdomain hostname with multiple labels passes."""
    result = validate("https://api.v2.example.com/path")
    assert isinstance(result, Validated)
    assert result.host == "api.v2.example.com"


# =========================================================================
# Stray colons / malformed authority structure
# =========================================================================
@pytest.mark.parametrize("token", [
    "https://example.com:443:80/path",
    "https://example..com/path",
    "https://.example.com/path",
    "https://example.com./path",
    "https:///path",   # empty netloc
    "https://example_underscore.com/path",  # underscore not DNS-legal
])
def test_authority_structure_malformed(token):
    result = validate(token)
    assert isinstance(result, Violation)
    assert result.code is ViolationCode.URL_MALFORMED


# =========================================================================
# safe_scheme: derived independently from urlsplit
# =========================================================================
def test_safe_scheme_independent_of_parse_failure():
    """A broken authority that may still preserve the scheme prefix
    yields URL_MALFORMED with safe_scheme=`https`."""
    token = "https://[broken"
    result = validate(token)
    assert isinstance(result, Violation)
    assert result.code is ViolationCode.URL_MALFORMED
    assert result.safe_scheme == "https"


def test_safe_scheme_case_folded_in_validation():
    """`HTTPS://` is accepted as scheme `https` after case-folding."""
    result = validate("HTTPS://example.com/path")
    assert isinstance(result, Validated)
    assert result.canonical == "https://example.com/path"


def test_safe_scheme_populated_for_unsupported():
    result = validate("httpx://example.com/path")
    assert isinstance(result, Violation)
    assert result.code is ViolationCode.URL_UNSUPPORTED_SCHEME
    assert result.safe_scheme == "httpx"


def test_safe_scheme_none_for_garbage_input():
    result = validate("just_a_string_no_scheme")
    assert isinstance(result, Violation)
    assert result.code is ViolationCode.URL_MALFORMED
    assert result.safe_scheme is None


# =========================================================================
# Canonicalization
# =========================================================================
def test_canonicalization_case_folds_scheme_and_host():
    result = validate("HTTPS://EXAMPLE.com/Path")
    assert isinstance(result, Validated)
    assert result.canonical == "https://example.com/Path"
    # Path case preserved exactly.
    assert "/Path" in result.canonical


def test_canonicalization_preserves_trailing_slash_distinction():
    a = validate("https://example.com/jobs/123")
    b = validate("https://example.com/jobs/123/")
    assert isinstance(a, Validated) and isinstance(b, Validated)
    assert a.canonical != b.canonical


def test_canonicalization_preserves_query_exact():
    result = validate("https://example.com/x?q=a,b&r=c")
    assert isinstance(result, Validated)
    assert result.canonical == "https://example.com/x?q=a,b&r=c"


def test_canonicalization_preserves_fragment_exact():
    result = validate("https://example.com/x#section-1")
    assert isinstance(result, Validated)
    assert result.canonical == "https://example.com/x#section-1"


def test_canonicalization_preserves_percent_encoding_exactly():
    """%C3%A4 and ä are different characters as far as the matching key
    is concerned. No percent-decoding, no normalization."""
    a = validate("https://example.com/päth")
    b = validate("https://example.com/p%C3%A4th")
    assert isinstance(a, Validated) and isinstance(b, Validated)
    assert a.canonical != b.canonical


def test_canonicalization_preserves_empty_path():
    """A URL with no path stays without one. Trailing-slash distinction
    applies here too."""
    result = validate("https://example.com")
    assert isinstance(result, Validated)
    assert result.canonical == "https://example.com"


# =========================================================================
# safe_host invariants (via valid execution)
# =========================================================================
def test_safe_host_populated_for_disallowed_port_only():
    """Across all six codes, safe_host should be populated only on
    URL_DISALLOWED_PORT (and on the URL_MALFORMED variant where host
    validation passed but port malformed)."""
    cases_none = [
        ("https://example.com/path" + "a" * 600, ViolationCode.URL_OVER_LIMIT),
        ("https://example.com/\tx", ViolationCode.URL_CONTROL_CHARS),
        ("ftp://example.com/x", ViolationCode.URL_UNSUPPORTED_SCHEME),
        ("https://user@example.com/x", ViolationCode.URL_CREDENTIALS_PRESENT),
        ("https://[::1]/x", ViolationCode.URL_MALFORMED),
    ]
    for token, expected_code in cases_none:
        result = validate(token)
        assert isinstance(result, Violation)
        assert result.code is expected_code, (token, result.code)
        assert result.safe_host is None, (token, result.safe_host)

    disallowed = validate("https://example.com:80/x")
    assert isinstance(disallowed, Violation)
    assert disallowed.code is ViolationCode.URL_DISALLOWED_PORT
    assert disallowed.safe_host == "example.com"


def test_safe_host_populated_when_port_malformed_after_host_passed():
    """URL_MALFORMED at the port step retains safe_host because host
    validation already passed."""
    result = validate("https://example.com:0443/x")
    assert isinstance(result, Violation)
    assert result.code is ViolationCode.URL_MALFORMED
    assert result.safe_host == "example.com"


# =========================================================================
# __post_init__ invariants on Violation (defense in depth)
# =========================================================================
def test_violation_rejects_safe_host_for_unsupported_scheme():
    with pytest.raises(ValueError, match="safe_host must be None"):
        Violation(
            code=ViolationCode.URL_UNSUPPORTED_SCHEME,
            raw_token="ftp://x",
            raw_token_hash=hash_raw_token("ftp://x"),
            safe_scheme="ftp",
            safe_host="x",
        )


def test_violation_rejects_safe_scheme_for_over_limit():
    with pytest.raises(ValueError, match="safe_scheme must be None"):
        Violation(
            code=ViolationCode.URL_OVER_LIMIT,
            raw_token="x",
            raw_token_hash=hash_raw_token("x"),
            safe_scheme="https",
            safe_host=None,
        )


def test_violation_rejects_safe_host_for_over_limit():
    with pytest.raises(ValueError, match="safe_host must be None"):
        Violation(
            code=ViolationCode.URL_OVER_LIMIT,
            raw_token="x",
            raw_token_hash=hash_raw_token("x"),
            safe_scheme=None,
            safe_host="x",
        )


def test_violation_rejects_safe_scheme_for_control_chars():
    with pytest.raises(ValueError, match="safe_scheme must be None"):
        Violation(
            code=ViolationCode.URL_CONTROL_CHARS,
            raw_token="x",
            raw_token_hash=hash_raw_token("x"),
            safe_scheme="https",
            safe_host=None,
        )


def test_violation_rejects_null_safe_scheme_for_unsupported_scheme():
    with pytest.raises(ValueError, match="safe_scheme must be populated"):
        Violation(
            code=ViolationCode.URL_UNSUPPORTED_SCHEME,
            raw_token="x",
            raw_token_hash=hash_raw_token("x"),
            safe_scheme=None,
            safe_host=None,
        )


def test_violation_rejects_safe_host_for_credentials_present():
    with pytest.raises(ValueError, match="safe_host must be None"):
        Violation(
            code=ViolationCode.URL_CREDENTIALS_PRESENT,
            raw_token="x",
            raw_token_hash=hash_raw_token("x"),
            safe_scheme="https",
            safe_host="example.com",
        )


def test_violation_requires_safe_host_for_disallowed_port():
    with pytest.raises(ValueError, match="safe_host must be populated"):
        Violation(
            code=ViolationCode.URL_DISALLOWED_PORT,
            raw_token="x",
            raw_token_hash=hash_raw_token("x"),
            safe_scheme="https",
            safe_host=None,
        )


def test_violation_requires_safe_scheme_for_disallowed_port():
    with pytest.raises(ValueError, match="safe_scheme must be populated"):
        Violation(
            code=ViolationCode.URL_DISALLOWED_PORT,
            raw_token="x",
            raw_token_hash=hash_raw_token("x"),
            safe_scheme=None,
            safe_host="x",
        )


# =========================================================================
# Discriminated union + frozen dataclass invariants
# =========================================================================
def test_validate_returns_violation_or_validated():
    """Every input produces exactly one of the two types."""
    for token in ["https://example.com/x", "garbage", "ftp://x", "https://user@x/y"]:
        result = validate(token)
        assert isinstance(result, (Violation, Validated))


def test_violation_is_frozen():
    v = Violation(
        code=ViolationCode.URL_MALFORMED,
        raw_token="x",
        raw_token_hash=hash_raw_token("x"),
        safe_scheme=None,
        safe_host=None,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        v.code = ViolationCode.URL_OVER_LIMIT  # type: ignore[misc]


def test_validated_is_frozen():
    result = validate("https://example.com/x")
    assert isinstance(result, Validated)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.canonical = "https://other.example.com/x"  # type: ignore[misc]


def test_raw_candidate_is_frozen():
    rc = RawCandidate(extracted_token="https://x.com", span_start=0, span_end=13)
    with pytest.raises(dataclasses.FrozenInstanceError):
        rc.extracted_token = "https://y.com"  # type: ignore[misc]


# =========================================================================
# safe_telemetry_fields
# =========================================================================
def test_safe_telemetry_fields_shape_exact():
    """The mapping has exactly these five keys, no extras, no missing."""
    v = Violation(
        code=ViolationCode.URL_DISALLOWED_PORT,
        raw_token="https://example.com:80/x",
        raw_token_hash=hash_raw_token("https://example.com:80/x"),
        safe_scheme="https",
        safe_host="example.com",
    )
    fields = safe_telemetry_fields(v, move="present_matches")
    assert set(fields.keys()) == {"violation_code", "move", "scheme", "host", "url_hash"}


def test_safe_telemetry_fields_values():
    v = Violation(
        code=ViolationCode.URL_DISALLOWED_PORT,
        raw_token="https://example.com:80/x",
        raw_token_hash=hash_raw_token("https://example.com:80/x"),
        safe_scheme="https",
        safe_host="example.com",
    )
    fields = safe_telemetry_fields(v, move="present_matches")
    assert fields["violation_code"] == "URL_DISALLOWED_PORT"
    assert fields["move"] == "present_matches"
    assert fields["scheme"] == "https"
    assert fields["host"] == "example.com"
    assert fields["url_hash"] == hash_raw_token("https://example.com:80/x")


def test_safe_telemetry_fields_safe_host_none_for_unsupported_scheme():
    v = Violation(
        code=ViolationCode.URL_UNSUPPORTED_SCHEME,
        raw_token="ftp://example.com/x",
        raw_token_hash=hash_raw_token("ftp://example.com/x"),
        safe_scheme="ftp",
        safe_host=None,
    )
    fields = safe_telemetry_fields(v, move="explain_gap")
    assert fields["host"] is None
    assert fields["scheme"] == "ftp"


def test_safe_telemetry_fields_no_path_or_query_leak():
    """A long path with sensitive-looking query params must not appear
    in telemetry under any key."""
    sensitive = "https://example.com/jobs?token=secret123&user=alice"
    v = validate(sensitive + "&other=" + ("x" * 600))
    assert isinstance(v, Violation)
    assert v.code is ViolationCode.URL_OVER_LIMIT
    fields = safe_telemetry_fields(v, move="present_matches")
    # The hash is the only commitment to content; nothing else may
    # contain any substring of the raw URL beyond scheme.
    for key, value in fields.items():
        if key in ("violation_code", "move", "url_hash"):
            continue
        if value is None:
            continue
        assert "secret123" not in value
        assert "alice" not in value
        assert "/jobs" not in value
        assert "?" not in value
        assert "&" not in value


def test_safe_telemetry_fields_move_param_is_keyword_only():
    """The `move` argument must be keyword-only to prevent positional
    misuse alongside the violation arg."""
    v = Violation(
        code=ViolationCode.URL_MALFORMED,
        raw_token="x",
        raw_token_hash=hash_raw_token("x"),
        safe_scheme=None,
        safe_host=None,
    )
    with pytest.raises(TypeError):
        safe_telemetry_fields(v, "present_matches")  # type: ignore[misc]


# =========================================================================
# Hash flows through validate -> Violation -> safe_telemetry_fields
# =========================================================================
def test_hash_consistent_across_pipeline():
    """The url_hash in safe_telemetry_fields matches hash_raw_token of
    the same input — same SHA-256 lowercase hex, same bytes."""
    token = "ftp://example.com/file"
    result = validate(token)
    assert isinstance(result, Violation)
    fields = safe_telemetry_fields(result, move="present_matches")
    assert fields["url_hash"] == hash_raw_token(token)
    assert fields["url_hash"] == result.raw_token_hash


# =========================================================================
# MAX_URL_LENGTH constant
# =========================================================================
def test_max_url_length_is_512():
    assert MAX_URL_LENGTH == 512


def test_exactly_at_limit_is_valid():
    """A token whose UTF-8 byte length equals MAX_URL_LENGTH passes the
    length check (the check is strictly >, not >=)."""
    prefix = "https://example.com/"
    pad = MAX_URL_LENGTH - len(prefix.encode("utf-8"))
    token = prefix + ("a" * pad)
    assert len(token.encode("utf-8")) == MAX_URL_LENGTH
    result = validate(token)
    assert isinstance(result, Validated)


def test_one_byte_over_limit_is_violation():
    prefix = "https://example.com/"
    pad = MAX_URL_LENGTH - len(prefix.encode("utf-8")) + 1
    token = prefix + ("a" * pad)
    assert len(token.encode("utf-8")) == MAX_URL_LENGTH + 1
    result = validate(token)
    assert isinstance(result, Violation)
    assert result.code is ViolationCode.URL_OVER_LIMIT


# =========================================================================
# Sub-step 3 additions: URL_NOT_IN_TURN_ALLOWLIST + check_url_membership
# =========================================================================
def test_url_not_in_turn_allowlist_violation_invariants():
    """URL_NOT_IN_TURN_ALLOWLIST requires safe_scheme='https' and
    safe_host populated (structural validation must have passed)."""
    # Valid construction
    v = Violation(
        code=ViolationCode.URL_NOT_IN_TURN_ALLOWLIST,
        raw_token="https://example.com/x",
        raw_token_hash=hash_raw_token("https://example.com/x"),
        safe_scheme="https",
        safe_host="example.com",
    )
    assert v.code is ViolationCode.URL_NOT_IN_TURN_ALLOWLIST


def test_url_not_in_turn_allowlist_rejects_non_https_safe_scheme():
    with pytest.raises(ValueError, match="safe_scheme must be 'https'"):
        Violation(
            code=ViolationCode.URL_NOT_IN_TURN_ALLOWLIST,
            raw_token="x", raw_token_hash=hash_raw_token("x"),
            safe_scheme="ftp",  # not https
            safe_host="example.com",
        )


def test_url_not_in_turn_allowlist_rejects_none_safe_scheme():
    with pytest.raises(ValueError, match="safe_scheme must be 'https'"):
        Violation(
            code=ViolationCode.URL_NOT_IN_TURN_ALLOWLIST,
            raw_token="x", raw_token_hash=hash_raw_token("x"),
            safe_scheme=None,
            safe_host="example.com",
        )


def test_url_not_in_turn_allowlist_rejects_none_safe_host():
    with pytest.raises(ValueError, match="safe_host must be populated"):
        Violation(
            code=ViolationCode.URL_NOT_IN_TURN_ALLOWLIST,
            raw_token="x", raw_token_hash=hash_raw_token("x"),
            safe_scheme="https",
            safe_host=None,
        )


def test_check_url_membership_passes_through_violations_unchanged():
    """validate() returning Violation -> wrapper returns the same."""
    result = check_url_membership("ftp://example.com/x", frozenset())
    assert isinstance(result, Violation)
    assert result.code is ViolationCode.URL_UNSUPPORTED_SCHEME


def test_check_url_membership_returns_validated_when_canonical_in_allowlist():
    canonical = "https://example.com/x"
    result = check_url_membership(canonical, frozenset({canonical}))
    assert isinstance(result, Validated)
    assert result.canonical == canonical


def test_check_url_membership_returns_not_in_turn_allowlist_when_missing():
    """validate() passes structurally but canonical is not in allowlist
    -> URL_NOT_IN_TURN_ALLOWLIST with safe_scheme='https' and safe_host
    populated.
    """
    result = check_url_membership(
        "https://example.com/x",
        frozenset({"https://other.com/y"}),
    )
    assert isinstance(result, Violation)
    assert result.code is ViolationCode.URL_NOT_IN_TURN_ALLOWLIST
    assert result.safe_scheme == "https"
    assert result.safe_host == "example.com"
    assert result.raw_token == "https://example.com/x"
    assert result.raw_token_hash == hash_raw_token("https://example.com/x")


def test_check_url_membership_empty_allowlist_rejects_all():
    """An empty allowlist rejects every otherwise-valid URL."""
    result = check_url_membership(
        "https://example.com/jobs/123",
        frozenset(),
    )
    assert isinstance(result, Violation)
    assert result.code is ViolationCode.URL_NOT_IN_TURN_ALLOWLIST


def test_check_url_membership_canonical_distinct_from_raw():
    """A raw URL whose CASE differs from the allowlist canonical still
    matches because the allowlist contains the canonical form and
    validate() canonicalizes case-folded scheme/host.
    """
    raw_with_upper_host = "https://EXAMPLE.com/x"
    canonical = "https://example.com/x"
    result = check_url_membership(
        raw_with_upper_host, frozenset({canonical}),
    )
    assert isinstance(result, Validated)
    assert result.canonical == canonical


def test_check_url_membership_telemetry_field_shape_for_not_in_allowlist():
    """Telemetry produced from a URL_NOT_IN_TURN_ALLOWLIST violation
    has the standard five-field shape and emits the actual scheme +
    host (since validation passed).
    """
    result = check_url_membership(
        "https://example.com/x",
        frozenset({"https://other.com/y"}),
    )
    assert isinstance(result, Violation)
    fields = safe_telemetry_fields(result, move="present_matches")
    assert set(fields.keys()) == {
        "violation_code", "move", "scheme", "host", "url_hash",
    }
    assert fields["violation_code"] == "URL_NOT_IN_TURN_ALLOWLIST"
    assert fields["scheme"] == "https"
    assert fields["host"] == "example.com"
