"""Recommendation feedback: saved / applied / not_relevant.

Token-authenticated. profile_id is derived from the bearer token; the
client body carries job_id, action, and an optional note. Auth runs at
route level so missing token returns 401 before body validation.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from skillbridge.auth import require_profile
from skillbridge.db import fetch_one
from skillbridge.models import FeedbackIn
from skillbridge.routes._envelope import envelope

router = APIRouter(tags=["feedback"])


@router.post("/v1/recommendations/feedback", dependencies=[Depends(require_profile)])
async def post_feedback(request: Request, body: FeedbackIn):
    profile_id = request.state.profile_id
    row = await fetch_one(
        """
        INSERT INTO interaction.recommendation_feedback
            (profile_id, job_id, match_id, action, note)
        VALUES (%s,%s,%s,%s,%s)
        RETURNING feedback_id, created_at
        """,
        (
            profile_id, str(body.job_id),
            str(body.match_id) if body.match_id else None,
            body.action, body.note,
        ),
    )
    if not row:
        raise HTTPException(500, "Failed to record feedback")
    return await envelope(request, row)
