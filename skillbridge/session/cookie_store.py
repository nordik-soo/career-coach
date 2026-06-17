"""Cookie-style session store (no infrastructure dependency).

The "session_id" returned to the client IS the signed payload. Each save()
rotates the value. Useful for local dev or if a Redis dependency is blocked.

For production with multiple workers, this is fine — there is no shared
state to coordinate. The trade-offs vs. Redis:
- ~4KB payload limit (skills list can get long)
- cannot revoke a session server-side until TTL expires
- payload is opaque to the server unless decoded
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

from itsdangerous import BadSignature, SignatureExpired, TimestampSigner

from config import SESSION_COOKIE_SECRET, SESSION_TTL_MINUTES
from skillbridge.session.staging import StagedProfile

log = logging.getLogger(__name__)


class CookieSessionStore:
    def __init__(self, secret: str | None = None) -> None:
        secret = secret or SESSION_COOKIE_SECRET
        if not secret or secret.startswith("PLACEHOLDER"):
            raise RuntimeError(
                "SESSION_COOKIE_SECRET is missing or placeholder. "
                "Generate one: python -c \"import secrets; print(secrets.token_urlsafe(48))\""
            )
        self._signer = TimestampSigner(secret, salt="sb-session-v1")
        self._max_age = SESSION_TTL_MINUTES * 60

    # ------------------------------------------------- API surface
    def new_session(self) -> str:
        """Return a signed envelope wrapping an empty staged profile.

        We use a UUID as the *internal* session_id field so logs can reference
        it. The session_id returned to the client is the signed token.
        """
        staged = StagedProfile.new(uuid.uuid4().hex)
        return self._sign(staged)

    def load(self, session_id: str | None) -> StagedProfile | None:
        if not session_id:
            return None
        try:
            raw = self._signer.unsign(session_id, max_age=self._max_age)
        except SignatureExpired:
            return None
        except BadSignature:
            return None
        try:
            return StagedProfile.from_json(raw)
        except Exception as e:
            log.warning("Failed to parse cookie session: %s", e)
            return None

    def save(self, staged: StagedProfile) -> str:
        return self._sign(staged)

    def delete(self, session_id: str) -> None:
        # Cookie-mode sessions are revoked by the client dropping the cookie.
        # There is nothing server-side to clear.
        return

    def ping(self) -> None:
        # No external dependency; readiness is implicit once the signer is
        # initialized (which happens in __init__ with a non-placeholder secret).
        return

    # ------------------------------------------------- internals
    def _sign(self, staged: StagedProfile) -> str:
        # redact_for_cookie=True strips resume_text and resume_facts_json
        # from the signed payload. Cookie has a ~4KB cap; raw text and
        # rich parsed facts can easily exceed that. Production should use
        # Redis if resume upload is enabled — see docs/resume-design.md §3.
        payload = staged.to_json(redact_for_cookie=True).encode("utf-8")
        return self._signer.sign(payload).decode("utf-8")
