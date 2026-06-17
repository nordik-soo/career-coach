"""Job match endpoints — all token-authenticated /me paths.

The client never sends a profile_id. The server derives it from the
bearer token via require_profile. Auth runs at route level on POST so
missing token returns 401 before body validation.
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from skillbridge.auth import require_profile
from skillbridge.db import fetch_all, fetch_one
from skillbridge.match import engine as match_engine
from skillbridge.models import MatchRequest
from skillbridge.routes._envelope import envelope
from skillbridge.versions import ENGINE_VERSION_JOB_MATCH

router = APIRouter(tags=["matches"])


@router.post("/v1/matches/jobs", dependencies=[Depends(require_profile)])
async def compute_matches(request: Request, body: MatchRequest):
    """Recompute matches for the authenticated profile against the current dataset."""
    pid = request.state.profile_id
    results = await asyncio.to_thread(match_engine.compute_matches, pid)
    out = [
        {
            "job_id": r.job_id,
            "title": r.title,
            "employer": r.employer,
            "url": r.url,
            "location": r.location,
            "match_score": r.match_score,
            "match_band": r.match_band,
            "match_eligible": r.match_eligible,
            "matched_skills_count": len(r.matched_skills),
            "required_skills_count": r.required_skills_count,
            "missing_skills_count": len(r.missing_skills),
            "credential_warning": r.credential_warning,
        }
        for r in results[: body.top]
    ]
    return await envelope(request, out, engine_version=ENGINE_VERSION_JOB_MATCH)


@router.get("/v1/profiles/me/job-matches")
async def list_matches(
    request: Request,
    profile_id: str = Depends(require_profile),
    limit: int = Query(20, ge=1, le=100),
):
    rows = await fetch_all(
        """
        SELECT m.match_id, m.job_id, j.title, j.employer, j.url, j.location,
               m.match_score, m.match_band, m.match_eligible,
               m.matched_skills_count, m.required_skills_count,
               m.missing_skills_count, m.credential_warning, m.computed_at
          FROM analytics.job_match m
          JOIN core.job_posting j ON j.job_id = m.job_id
         WHERE m.profile_id = %s
           AND m.match_eligible = TRUE
         ORDER BY m.match_score DESC
         LIMIT %s
        """,
        (profile_id, limit),
    )
    return await envelope(request, rows, engine_version=ENGINE_VERSION_JOB_MATCH)


@router.get("/v1/profiles/me/job-matches/{match_id}")
async def get_match(request: Request, match_id: str,
                    profile_id: str = Depends(require_profile)):
    match = await fetch_one(
        """
        SELECT m.*, j.title, j.employer, j.url, j.description, j.location
          FROM analytics.job_match m
          JOIN core.job_posting j ON j.job_id = m.job_id
         WHERE m.match_id = %s AND m.profile_id = %s
        """,
        (match_id, profile_id),
    )
    if not match:
        raise HTTPException(404, "Match not found")
    skills = await fetch_all(
        "SELECT skill_name, skill_id, status FROM analytics.job_match_skill WHERE match_id = %s",
        (match_id,),
    )
    matched = [s for s in skills if s["status"] == "matched"]
    missing = [s for s in skills if s["status"] == "missing"]
    return await envelope(
        request,
        {**match, "matched_skills": matched, "missing_skills": missing},
        engine_version=ENGINE_VERSION_JOB_MATCH,
    )


@router.get("/v1/profiles/me/skill-gaps/{job_id}")
async def skill_gap(request: Request, job_id: str,
                    profile_id: str = Depends(require_profile)):
    row = await fetch_one(
        """
        SELECT m.match_id, m.match_score, m.match_band, m.required_skills_count,
               m.missing_skills_count, m.credential_warning, j.title
          FROM analytics.job_match m
          JOIN core.job_posting j ON j.job_id = m.job_id
         WHERE m.profile_id = %s AND m.job_id = %s
        """,
        (profile_id, job_id),
    )
    if not row:
        raise HTTPException(
            404,
            "No match between this profile and job — compute matches first",
        )
    skills = await fetch_all(
        "SELECT skill_name, skill_id, status FROM analytics.job_match_skill WHERE match_id = %s",
        (row["match_id"],),
    )
    return await envelope(
        request,
        {
            **row,
            "matched_skills": [s["skill_name"] for s in skills if s["status"] == "matched"],
            "missing_skills": [s["skill_name"] for s in skills if s["status"] == "missing"],
        },
        engine_version=ENGINE_VERSION_JOB_MATCH,
    )
