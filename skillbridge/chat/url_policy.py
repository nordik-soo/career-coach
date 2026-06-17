"""URL grounding primitives for AR-9.bug.2a.

Structural URL validation, canonicalization, extraction, and telemetry.
No responder dependencies. No I/O. Sub-step 2 of the bug.2a slice;
consumers (SanitizedResponderView, policy gates) are introduced in
later sub-steps.

Locked contract:
  - Six structural violation codes (URL_NOT_IN_TURN_ALLOWLIST is
    consumer-side and added in sub-step 3, not here)
  - HTTPS scheme only
  - 512-byte length cap, no truncation
  - Lexical port matching: only absent or ":443" exact; ":0443" et al.
    are URL_MALFORMED
  - Bracketed IPv6 authorities -> URL_MALFORMED (out of scope)
  - Non-ASCII hosts -> URL_MALFORMED (IDN out of scope)
  - Canonicalization preserves path/query/fragment character-for-character
  - safe_telemetry_fields accepts Violation only; emits exactly
    {violation_code, move, scheme, host, url_hash}
  - safe_host = None for URL_UNSUPPORTED_SCHEME is a top-level invariant
    enforced by __post_init__, not a derived ordering consequence
"""
from __future__ import annotations

import enum
import hashlib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final
from urllib.parse import urlsplit


MAX_URL_LENGTH: Final = 512


class ViolationCode(enum.Enum):
    URL_OVER_LIMIT = "URL_OVER_LIMIT"
    URL_CONTROL_CHARS = "URL_CONTROL_CHARS"
    URL_MALFORMED = "URL_MALFORMED"
    URL_UNSUPPORTED_SCHEME = "URL_UNSUPPORTED_SCHEME"
    URL_CREDENTIALS_PRESENT = "URL_CREDENTIALS_PRESENT"
    URL_DISALLOWED_PORT = "URL_DISALLOWED_PORT"
    # Added in sub-step 3 (consumer-side check). Raised when the URL
    # passed every structural check but its canonical form is not in
    # the per-turn allowlist surfaced by the SanitizedResponderView.
    URL_NOT_IN_TURN_ALLOWLIST = "URL_NOT_IN_TURN_ALLOWLIST"


@dataclass(frozen=True)
class RawCandidate:
    extracted_token: str
    span_start: int
    span_end: int


@dataclass(frozen=True)
class Violation:
    code: ViolationCode
    raw_token: str
    raw_token_hash: str
    safe_scheme: str | None
    safe_host: str | None

    def __post_init__(self) -> None:
        code = self.code
        if code in (ViolationCode.URL_OVER_LIMIT, ViolationCode.URL_CONTROL_CHARS):
            if self.safe_scheme is not None:
                raise ValueError(
                    f"safe_scheme must be None for {code.value}"
                )
            if self.safe_host is not None:
                raise ValueError(
                    f"safe_host must be None for {code.value}"
                )
        if code is ViolationCode.URL_UNSUPPORTED_SCHEME:
            if self.safe_scheme is None:
                raise ValueError(
                    "safe_scheme must be populated for URL_UNSUPPORTED_SCHEME"
                )
            if self.safe_host is not None:
                raise ValueError(
                    "safe_host must be None for URL_UNSUPPORTED_SCHEME"
                )
        if code is ViolationCode.URL_CREDENTIALS_PRESENT:
            if self.safe_scheme is None:
                raise ValueError(
                    "safe_scheme must be populated for URL_CREDENTIALS_PRESENT"
                )
            if self.safe_host is not None:
                raise ValueError(
                    "safe_host must be None for URL_CREDENTIALS_PRESENT"
                )
        if code is ViolationCode.URL_DISALLOWED_PORT:
            if self.safe_scheme is None:
                raise ValueError(
                    "safe_scheme must be populated for URL_DISALLOWED_PORT"
                )
            if self.safe_host is None:
                raise ValueError(
                    "safe_host must be populated for URL_DISALLOWED_PORT"
                )
        if code is ViolationCode.URL_NOT_IN_TURN_ALLOWLIST:
            if self.safe_scheme != "https":
                raise ValueError(
                    "safe_scheme must be 'https' for URL_NOT_IN_TURN_ALLOWLIST "
                    "(structural validation must have passed)"
                )
            if self.safe_host is None:
                raise ValueError(
                    "safe_host must be populated for URL_NOT_IN_TURN_ALLOWLIST"
                )


@dataclass(frozen=True)
class Validated:
    canonical: str
    raw_token: str
    raw_token_hash: str
    scheme: str
    host: str


_SCHEME_PATTERN: Final = re.compile(
    r"(?<!\w)(?:"
    # Schemes with authority: any RFC 3986 scheme followed by "://"
    r"[A-Za-z][A-Za-z0-9+\-.]*://"
    r"|"
    # Sub-step 5 amendment: clickable non-HTTPS schemes that DON'T use
    # "://" but ARE clickable in chat / markdown UIs. Treating these
    # as out-of-scope (locked sub-step 2 framing) would let them bypass
    # URL grounding. Each will validate to URL_UNSUPPORTED_SCHEME via
    # the existing scheme-check path.
    r"(?:javascript|data|mailto|vbscript|file|tel|sms):"
    r")",
    re.IGNORECASE,
)

_SCHEME_PREFIX: Final = re.compile(
    r"^([A-Za-z][A-Za-z0-9+\-.]*):"
)

_CONTROL_CHAR_PATTERN: Final = re.compile(r"[\x00-\x1f\x7f\s]")

_PROSE_TRAIL_CHARS: Final = frozenset(".,;!?")

_BRACKET_PAIRS: Final = {")": "(", "]": "[", "}": "{", ">": "<"}

_HOST_CHAR_PATTERN: Final = re.compile(r"^[A-Za-z0-9\-.]+$")


def hash_raw_token(extracted_token: str) -> str:
    return hashlib.sha256(extracted_token.encode("utf-8")).hexdigest()


def _strip_trailing_punctuation(token: str) -> str:
    while True:
        if not token:
            return token
        last = token[-1]
        if last in _PROSE_TRAIL_CHARS:
            token = token[:-1]
            continue
        if last in _BRACKET_PAIRS:
            opener = _BRACKET_PAIRS[last]
            if token.count(last) > token.count(opener):
                token = token[:-1]
                continue
        return token


def extract_url_candidates(text: str) -> Iterable[RawCandidate]:
    for match in _SCHEME_PATTERN.finditer(text):
        start = match.start()
        end = match.end()
        while end < len(text):
            ch = text[end]
            if ch.isspace() or ord(ch) < 0x20 or ord(ch) == 0x7f:
                break
            end += 1
        raw = text[start:end]
        stripped = _strip_trailing_punctuation(raw)
        if not stripped:
            continue
        yield RawCandidate(
            extracted_token=stripped,
            span_start=start,
            span_end=start + len(stripped),
        )


def _extract_safe_scheme(token: str) -> str | None:
    m = _SCHEME_PREFIX.match(token)
    if not m:
        return None
    return m.group(1).lower()


def _validate_host(host: str) -> bool:
    """Locked DNS host rules (RFC 1035 + RFC 1123):

      - Non-empty
      - ASCII only
      - Total length <= 253 chars (dotted form)
      - Only DNS-legal characters: letters, digits, '-', '.'
      - No leading or trailing dot
      - No consecutive dots
      - Each dot-separated label is 1-63 chars
      - Each label does not start or end with a hyphen
        (internal hyphens are allowed; digit-led labels are allowed
        per RFC 1123)
    """
    if not host:
        return False
    if not host.isascii():
        return False
    if len(host) > 253:
        return False
    if not _HOST_CHAR_PATTERN.match(host):
        return False
    if host.startswith(".") or host.endswith("."):
        return False
    if ".." in host:
        return False
    for label in host.split("."):
        if not label:
            return False
        if len(label) > 63:
            return False
        if label[0] == "-" or label[-1] == "-":
            return False
    return True


def _validate_lexical_port(port_text: str) -> ViolationCode | None:
    """Locked port policy on the lexical substring after the host colon.

    Returns:
      None on canonical ":443"
      URL_DISALLOWED_PORT on canonical decimal in 1..65535 that is not 443
      URL_MALFORMED otherwise
    """
    if not port_text:
        return ViolationCode.URL_MALFORMED
    if not (port_text.isascii() and port_text.isdigit()):
        return ViolationCode.URL_MALFORMED
    if len(port_text) > 1 and port_text[0] == "0":
        return ViolationCode.URL_MALFORMED
    value = int(port_text)
    if not (1 <= value <= 65535):
        return ViolationCode.URL_MALFORMED
    if value == 443:
        return None
    return ViolationCode.URL_DISALLOWED_PORT


def validate(extracted_token: str) -> Violation | Validated:
    """Apply the locked ten-step validation.

    Execution order, with first failure short-circuiting:
      1. URL_OVER_LIMIT  (UTF-8 byte length > MAX_URL_LENGTH)
      2. URL_CONTROL_CHARS  (any C0, DEL, or whitespace code point)
      3. Independent scheme regex (populates safe_scheme for steps >= 4)
      4. URL_UNSUPPORTED_SCHEME (scheme present but != "https"); a
         missing scheme produces URL_MALFORMED, not UNSUPPORTED_SCHEME
      5. urlsplit; raise -> URL_MALFORMED
      6. Empty netloc -> URL_MALFORMED
      7. URL_CREDENTIALS_PRESENT (@ in netloc)
      8. Host validation: bracketed IPv6 rejected, non-empty, ASCII-only,
         DNS-legal chars, no leading/trailing/consecutive dots, no stray
         colons -> URL_MALFORMED on any failure
      9. Lexical port policy -> URL_MALFORMED or URL_DISALLOWED_PORT
      10. Canonicalize -> Validated
    """
    raw_hash = hash_raw_token(extracted_token)

    if len(extracted_token.encode("utf-8")) > MAX_URL_LENGTH:
        return Violation(
            code=ViolationCode.URL_OVER_LIMIT,
            raw_token=extracted_token,
            raw_token_hash=raw_hash,
            safe_scheme=None,
            safe_host=None,
        )

    if _CONTROL_CHAR_PATTERN.search(extracted_token):
        return Violation(
            code=ViolationCode.URL_CONTROL_CHARS,
            raw_token=extracted_token,
            raw_token_hash=raw_hash,
            safe_scheme=None,
            safe_host=None,
        )

    safe_scheme = _extract_safe_scheme(extracted_token)

    if safe_scheme is None:
        return Violation(
            code=ViolationCode.URL_MALFORMED,
            raw_token=extracted_token,
            raw_token_hash=raw_hash,
            safe_scheme=None,
            safe_host=None,
        )

    if safe_scheme != "https":
        return Violation(
            code=ViolationCode.URL_UNSUPPORTED_SCHEME,
            raw_token=extracted_token,
            raw_token_hash=raw_hash,
            safe_scheme=safe_scheme,
            safe_host=None,
        )

    try:
        parsed = urlsplit(extracted_token)
    except ValueError:
        return Violation(
            code=ViolationCode.URL_MALFORMED,
            raw_token=extracted_token,
            raw_token_hash=raw_hash,
            safe_scheme=safe_scheme,
            safe_host=None,
        )

    netloc = parsed.netloc

    if not netloc:
        return Violation(
            code=ViolationCode.URL_MALFORMED,
            raw_token=extracted_token,
            raw_token_hash=raw_hash,
            safe_scheme=safe_scheme,
            safe_host=None,
        )

    if "@" in netloc:
        return Violation(
            code=ViolationCode.URL_CREDENTIALS_PRESENT,
            raw_token=extracted_token,
            raw_token_hash=raw_hash,
            safe_scheme=safe_scheme,
            safe_host=None,
        )

    if "[" in netloc or "]" in netloc:
        return Violation(
            code=ViolationCode.URL_MALFORMED,
            raw_token=extracted_token,
            raw_token_hash=raw_hash,
            safe_scheme=safe_scheme,
            safe_host=None,
        )

    if ":" in netloc:
        host_text, _, port_text = netloc.rpartition(":")
        port_present = True
    else:
        host_text = netloc
        port_text = ""
        port_present = False

    if ":" in host_text:
        return Violation(
            code=ViolationCode.URL_MALFORMED,
            raw_token=extracted_token,
            raw_token_hash=raw_hash,
            safe_scheme=safe_scheme,
            safe_host=None,
        )

    if not _validate_host(host_text):
        return Violation(
            code=ViolationCode.URL_MALFORMED,
            raw_token=extracted_token,
            raw_token_hash=raw_hash,
            safe_scheme=safe_scheme,
            safe_host=None,
        )

    safe_host = host_text.lower()

    canonical_port = ""
    if port_present:
        port_outcome = _validate_lexical_port(port_text)
        if port_outcome is ViolationCode.URL_MALFORMED:
            return Violation(
                code=ViolationCode.URL_MALFORMED,
                raw_token=extracted_token,
                raw_token_hash=raw_hash,
                safe_scheme=safe_scheme,
                safe_host=safe_host,
            )
        if port_outcome is ViolationCode.URL_DISALLOWED_PORT:
            return Violation(
                code=ViolationCode.URL_DISALLOWED_PORT,
                raw_token=extracted_token,
                raw_token_hash=raw_hash,
                safe_scheme=safe_scheme,
                safe_host=safe_host,
            )
        canonical_port = ":443"

    # Lift the post-authority remainder of the URL directly from the
    # extracted token to preserve path/query/fragment (including the
    # presence-or-absence of trailing "?" or "#") character-for-character.
    # urlsplit collapses empty query/fragment to '', which would lose
    # the distinction between "https://x/?" and "https://x/".
    scheme_text = extracted_token.split("://", 1)[0]
    authority_end_in_token = len(scheme_text) + 3 + len(netloc)
    rest = extracted_token[authority_end_in_token:]

    canonical = f"https://{safe_host}{canonical_port}{rest}"

    return Validated(
        canonical=canonical,
        raw_token=extracted_token,
        raw_token_hash=raw_hash,
        scheme="https",
        host=safe_host,
    )


def safe_telemetry_fields(
    violation: Violation,
    *,
    move: str,
) -> Mapping[str, str | None]:
    """Telemetry record for one URL violation.

    Accepts Violation only. Valid URLs generate no violation telemetry.
    Returns exactly five keys: violation_code, move, scheme, host,
    url_hash. No path, query, fragment, port, user_info, raw_url, or
    canonical form ever appears.
    """
    return {
        "violation_code": violation.code.value,
        "move": move,
        "scheme": violation.safe_scheme,
        "host": violation.safe_host,
        "url_hash": violation.raw_token_hash,
    }


def check_url_membership(
    raw_token: str,
    allowlist: frozenset[str],
) -> Violation | Validated:
    """Apply structural validation, then check canonical membership.

    Three outcomes encoded in the existing Violation | Validated union:
      - validate() returned Violation -> return it unchanged
      - validate() returned Validated and canonical IS in allowlist
        -> return the Validated unchanged
      - validate() returned Validated but canonical is NOT in allowlist
        -> return Violation(URL_NOT_IN_TURN_ALLOWLIST, ...)

    The third outcome has safe_scheme='https' and safe_host populated
    because structural validation passed through host validation. The
    __post_init__ invariants on Violation enforce both.
    """
    result = validate(raw_token)
    if isinstance(result, Violation):
        return result
    if result.canonical in allowlist:
        return result
    return Violation(
        code=ViolationCode.URL_NOT_IN_TURN_ALLOWLIST,
        raw_token=result.raw_token,
        raw_token_hash=result.raw_token_hash,
        safe_scheme="https",
        safe_host=result.host,
    )
