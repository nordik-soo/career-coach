"""Session store package.

Backends:
- RedisSessionStore  (default; recommended for any multi-worker deploy)
- CookieSessionStore (signed payload, no infra; fine for local dev or
  if a Redis dependency is blocked)

Both implement a small, uniform interface so the chat layer doesn't care
which one is configured.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from config import SESSION_STORE
from skillbridge.session.staging import StagedProfile, StagedSkill

log = logging.getLogger(__name__)


class SessionStore(ABC):
    """Uniform interface over Redis and signed-cookie storage.

    For Redis: session_id is a UUID and persists across save() calls.
    For Cookie: session_id IS the signed payload — save() rotates it.

    The chat route always returns the value from save() back to the client,
    which echoes it on the next request. This works for both backends.
    """

    @abstractmethod
    def new_session(self) -> str: ...

    @abstractmethod
    def load(self, session_id: str | None) -> StagedProfile | None: ...

    @abstractmethod
    def save(self, staged: StagedProfile) -> str: ...

    @abstractmethod
    def delete(self, session_id: str) -> None: ...


# ----------------------------------------------------------- Factory
_store: SessionStore | None = None


def get_store() -> SessionStore:
    """Return the configured session store. Lazy-init on first use."""
    global _store
    if _store is not None:
        return _store
    if SESSION_STORE == "cookie":
        from skillbridge.session.cookie_store import CookieSessionStore
        _store = CookieSessionStore()
        log.info("Session store: signed cookie (no Redis)")
    else:
        from skillbridge.session.redis_store import RedisSessionStore
        _store = RedisSessionStore()
        log.info("Session store: Redis")
    return _store


async def init_store() -> None:
    """Called at app startup so we fail fast if Redis is unreachable."""
    store = get_store()
    # Best-effort connectivity check; Redis store implements ping().
    ping = getattr(store, "ping", None)
    if callable(ping):
        ping()


__all__ = ["SessionStore", "StagedProfile", "StagedSkill", "get_store", "init_store"]
