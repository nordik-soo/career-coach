"""Consent grant endpoint.

POST /v1/consent { session_id, consent_purposes }
  -> 200 { profile_id, session_token, expires_at, consent_purposes, recommended_jobs }

The raw session_token is returned ONCE. Store only the SHA-256 hash in
Postgres (handled inside handler.grant_consent). All subsequent API calls
must send `Authorization: Bearer <token>`.

If consent_purposes is missing the required 'profile_storage' entry, we
return 400 — there is nothing to consent to without it.
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from skillbridge.chat import handler as chat_handler
from skillbridge.db import fetch_all
from skillbridge.routes._envelope import envelope
from skillbridge.versions import ENGINE_VERSION_JOB_MATCH

log = logging.getLogger(__name__)

router = APIRouter(tags=["consent"])


class ConsentGrantBody(BaseModel):
    session_id: str = Field(..., description="Staged session id from anonymous chat")
    consent_purposes: list[str] = Field(
        ..., description="At minimum must include 'profile_storage'"
    )


@router.post("/v1/consent")
async def grant_consent(request: Request, body: ConsentGrantBody):
    try:
        result = await asyncio.to_thread(
            chat_handler.grant_consent, body.session_id, body.consent_purposes
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    # Return initial recommendations so the UI can transition directly from
    # consent into the authenticated experience without a second round trip.
    recommended = await fetch_all(
        """
        SELECT m.match_id, m.job_id, j.title, j.employer, j.url, j.location,
               m.match_score, m.match_band, m.matched_skills_count,
               m.required_skills_count, m.missing_skills_count,
               m.credential_warning
          FROM analytics.job_match m
          JOIN core.job_posting j ON j.job_id = m.job_id
         WHERE m.profile_id = %s
           AND m.match_eligible = TRUE
           AND m.match_band IN ('strong', 'good')
         ORDER BY m.match_score DESC
         LIMIT 5
        """,
        (result["profile_id"],),
    )

    return await envelope(
        request,
        {**result, "recommended_jobs": recommended},
        engine_version=ENGINE_VERSION_JOB_MATCH,
    )
