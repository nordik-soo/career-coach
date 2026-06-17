"""Admin endpoints — gated by ADMIN_API_KEY.

Clients send: Authorization: Bearer admin:<key>

In production, REQUIRE_ADMIN_AUTH=true (the default). Set false only in
local dev when you don't want to provision a key.
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, Request

from skillbridge.auth import require_admin
from skillbridge.db import fetch_one
from skillbridge.routes._envelope import envelope

router = APIRouter(prefix="/v1/admin", tags=["admin"])


@router.get("/data-status", dependencies=[Depends(require_admin)])
async def data_status(request: Request):
    row = await fetch_one("SELECT * FROM pipeline.v_data_status")
    return await envelope(request, row or {})


@router.post("/pipeline/refresh", dependencies=[Depends(require_admin)])
async def refresh_pipeline(request: Request):
    """Kick a one-shot pipeline run in the background."""
    from skillbridge.pipeline.orchestrator import run_all

    asyncio.create_task(asyncio.to_thread(run_all))
    return await envelope(request, {"status": "started"})
