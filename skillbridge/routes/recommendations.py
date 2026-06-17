"""Training recommendation endpoint — token-authenticated /me path."""
from __future__ import annotations

import asyncio
from dataclasses import asdict

from fastapi import APIRouter, Depends, Query, Request

from skillbridge.auth import require_profile
from skillbridge.match.recommend import suggest_for_profile
from skillbridge.routes._envelope import envelope
from skillbridge.versions import ENGINE_VERSION_TRAINING_REC

router = APIRouter(tags=["recommendations"])


@router.get("/v1/profiles/me/training-recommendations")
async def training_recommendations(
    request: Request,
    profile_id: str = Depends(require_profile),
    job_id: str | None = Query(None, description="Restrict to one job's missing skills"),
):
    suggestions = await asyncio.to_thread(suggest_for_profile, profile_id, job_id)
    return await envelope(
        request,
        [asdict(s) for s in suggestions],
        engine_version=ENGINE_VERSION_TRAINING_REC,
    )
