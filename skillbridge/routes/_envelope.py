"""Shared response envelope + meta helpers used by every route."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import Request

from skillbridge.db import fetch_one
from skillbridge.versions import ENGINE_VERSION_JOB_MATCH


def make_request_id() -> str:
    return f"req_{uuid.uuid4().hex[:16]}"


async def current_dataset_version() -> str | None:
    row = await fetch_one(
        "SELECT dataset_version FROM pipeline.dataset_state WHERE pointer_key = 'current_jobs'"
    )
    return row["dataset_version"] if row else None


async def last_publish_at() -> datetime | None:
    row = await fetch_one(
        "SELECT published_at FROM pipeline.dataset_state WHERE pointer_key = 'current_jobs'"
    )
    return row["published_at"] if row else None


async def build_meta(request: Request, *, engine_version: str | None = None) -> dict[str, Any]:
    return {
        "request_id": getattr(request.state, "request_id", make_request_id()),
        "dataset_version": await current_dataset_version(),
        "engine_version": engine_version or ENGINE_VERSION_JOB_MATCH,
        "data_as_of": await last_publish_at() or datetime.now(timezone.utc),
    }


async def envelope(request: Request, data: Any, *, engine_version: str | None = None,
                   warnings: list[str] | None = None) -> dict[str, Any]:
    meta = await build_meta(request, engine_version=engine_version)
    if warnings:
        meta["warnings"] = warnings
    return {"data": data, "meta": meta}
