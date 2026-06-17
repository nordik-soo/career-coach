"""Opaque session tokens for authenticated profile access.

Issued at consent grant time. The raw token is returned to the client ONCE
and never logged. Only the SHA-256 hash is stored in
profile.user_profile.session_token_hash.

Usage:
    token = mint_token()                       # raw token (returned once)
    h     = hash_token(token)                  # stored in Postgres

    # FastAPI dependency:
    @router.get(...)
    async def endpoint(profile_id: str = Depends(require_profile)):
        ...
"""
from __future__ import annotations

import hashlib
import logging
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import Header, HTTPException, Request

from config import ADMIN_API_KEY, REQUIRE_ADMIN_AUTH
from skillbridge.db import fetch_one

log = logging.getLogger(__name__)


ADMIN_TOKEN_PREFIX = "admin:"


def mint_token(nbytes: int = 32) -> str:
    """Generate a URL-safe random token. ~43 chars at 32 bytes."""
    return secrets.token_urlsafe(nbytes)


def hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def token_expires_at(hours: int) -> datetime:
    return datetime.now(timezone.utc) + timedelta(hours=hours)


# --------------------------------------------------------- FastAPI dependency
async def require_profile(
    request: Request,
    authorization: str | None = Header(default=None),
) -> str:
    """Resolve Bearer token -> profile_id. Raises 401 if invalid/expired."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Missing bearer token")
    raw = authorization.split(" ", 1)[1].strip()
    if not raw:
        raise HTTPException(401, "Empty bearer token")
    h = hash_token(raw)
    row = await fetch_one(
        """
        SELECT profile_id, session_token_expires_at
          FROM profile.user_profile
         WHERE session_token_hash = %s
           AND deleted_at IS NULL
        """,
        (h,),
    )
    if not row:
        raise HTTPException(401, "Invalid token")
    expires = row.get("session_token_expires_at")
    if expires is not None and expires < datetime.now(timezone.utc):
        raise HTTPException(401, "Token expired")
    pid = str(row["profile_id"])
    request.state.profile_id = pid
    return pid


async def optional_profile(
    request: Request,
    authorization: str | None = Header(default=None),
) -> str | None:
    """Same as require_profile but returns None instead of raising.

    Used by routes that allow both authenticated (post-consent) and
    anonymous (pre-consent) access — chat is the main example.

    Admin tokens (Bearer admin:<key>) are never resolved as a profile —
    this returns None for them so admin tokens cannot accidentally grant
    profile-scoped access.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    raw = authorization.split(" ", 1)[1].strip()
    if raw.startswith(ADMIN_TOKEN_PREFIX):
        return None
    try:
        return await require_profile(request, authorization)
    except HTTPException:
        return None


# ---------------------------------------------------- Admin dependency
def require_admin(authorization: str | None = Header(default=None)) -> None:
    """Verify bearer token matches ADMIN_API_KEY.

    Expected header form: `Authorization: Bearer admin:<key>`. The admin
    prefix prevents profile session tokens from being mistaken for admin
    tokens, even if they happened to collide with ADMIN_API_KEY by chance.

    If REQUIRE_ADMIN_AUTH=false (dev only), this is a no-op so engineers
    can hit admin endpoints without provisioning a key.

    Returns None; raises HTTPException on failure. Use as a route-level
    dependency: @router.get(..., dependencies=[Depends(require_admin)])
    """
    if not REQUIRE_ADMIN_AUTH:
        return
    if not ADMIN_API_KEY or ADMIN_API_KEY.startswith("PLACEHOLDER"):
        # Misconfiguration: admin auth is required but no key is set.
        # Surface as 503 so it's distinguishable from a bad token.
        raise HTTPException(503, "Admin auth not configured")
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Missing admin bearer token")
    raw = authorization.split(" ", 1)[1].strip()
    if not raw.startswith(ADMIN_TOKEN_PREFIX):
        raise HTTPException(401, "Admin token must be prefixed with 'admin:'")
    candidate = raw[len(ADMIN_TOKEN_PREFIX):]
    if not secrets.compare_digest(candidate, ADMIN_API_KEY):
        raise HTTPException(401, "Invalid admin token")
