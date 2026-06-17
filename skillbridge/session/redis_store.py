"""Redis-backed session store.

Stores StagedProfile as a JSON string keyed by a UUID. Sliding-expiration
TTL is applied on every save() — the user gets a fresh 30 minutes whenever
they interact.
"""
from __future__ import annotations

import logging
import uuid

import redis

from config import REDIS_URL, SESSION_TTL_MINUTES
from skillbridge.session.staging import StagedProfile

log = logging.getLogger(__name__)

_KEY_PREFIX = "sb:session:"


class RedisSessionStore:
    def __init__(self, url: str | None = None) -> None:
        self._client = redis.from_url(url or REDIS_URL, decode_responses=True)

    # ------------------------------------------------- API surface
    def new_session(self) -> str:
        return uuid.uuid4().hex

    def load(self, session_id: str | None) -> StagedProfile | None:
        if not session_id:
            return None
        raw = self._client.get(_KEY_PREFIX + session_id)
        if raw is None:
            return None
        try:
            return StagedProfile.from_json(raw)
        except Exception as e:
            log.warning("Failed to parse staged session %s: %s", session_id, e)
            return None

    def save(self, staged: StagedProfile) -> str:
        ttl = SESSION_TTL_MINUTES * 60
        self._client.setex(_KEY_PREFIX + staged.session_id, ttl, staged.to_json())
        return staged.session_id

    def delete(self, session_id: str) -> None:
        if not session_id:
            return
        try:
            self._client.delete(_KEY_PREFIX + session_id)
        except Exception:
            pass

    # ------------------------------------------------- diagnostics
    def ping(self) -> None:
        self._client.ping()
