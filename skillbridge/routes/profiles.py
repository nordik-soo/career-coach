"""Profile lifecycle — token-authenticated `/me` routes.

PR 3 changes from earlier:
- POST /v1/profiles removed entirely. Profiles are only created via the
  chat → consent flow (POST /v1/consent).
- All routes use /me, never accept a path or body profile_id.
- profile_id is derived from the bearer token (require_profile dependency).

This eliminates the entire class of "anyone with a UUID can read any
profile" vulnerabilities — the client never tells the server who it is.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from config import CONSENT_PURPOSES, CONSENT_VERSION
from skillbridge.auth import require_profile
from skillbridge.db import acquire, fetch_all, fetch_one
from skillbridge.models import ProfilePatch
from skillbridge.routes._envelope import envelope

router = APIRouter(prefix="/v1/profiles", tags=["profiles"])


# ---------------------------------------------------------------- GET /me
@router.get("/me")
async def get_profile(request: Request, profile_id: str = Depends(require_profile)):
    row = await fetch_one(
        "SELECT * FROM profile.user_profile WHERE profile_id = %s AND deleted_at IS NULL",
        (profile_id,),
    )
    if not row:
        raise HTTPException(404, "Profile not found")
    return await envelope(request, row)


# -------------------------------------------------------------- PATCH /me
@router.patch("/me", dependencies=[Depends(require_profile)])
async def patch_profile(request: Request, body: ProfilePatch):
    """Update profile fields. Auth runs at route level so missing token
    returns 401 before the body is validated."""
    profile_id = request.state.profile_id
    cols: list[str] = []
    params: list = []
    fields = body.model_dump(exclude_unset=True, exclude_none=False)
    for k, v in fields.items():
        if k == "consent_purposes":
            valid = [p for p in (v or []) if p in CONSENT_PURPOSES]
            cols.append("consent_purposes = %s")
            params.append(valid)
            cols.append("consent_version = %s")
            params.append(CONSENT_VERSION if valid else None)
            cols.append(
                "consent_granted_at = CASE WHEN array_length(%s::text[], 1) > 0 "
                "THEN NOW() ELSE consent_granted_at END"
            )
            params.append(valid)
            continue
        cols.append(f"{k} = %s")
        params.append(v)
    if not cols:
        raise HTTPException(400, "No fields to update")
    cols.append("updated_at = NOW()")
    params.append(profile_id)
    sql = (
        f"UPDATE profile.user_profile SET {', '.join(cols)} "
        f"WHERE profile_id = %s AND deleted_at IS NULL RETURNING *"
    )
    row = await fetch_one(sql, tuple(params))
    if not row:
        raise HTTPException(404, "Profile not found")
    return await envelope(request, row)


# ------------------------------------------------------------- DELETE /me
@router.delete("/me")
async def delete_profile(request: Request, profile_id: str = Depends(require_profile)):
    """Hard-delete every profile-linked row; soft-delete the profile shell with all PII cleared.

    Tables hard-deleted (in FK-safe order):
        interaction.recommendation_feedback
        interaction.chat_event
        analytics.training_recommendation
        analytics.job_match          (cascades to analytics.job_match_skill)
        profile.user_skill

    profile.user_profile row is retained as an audit shell. Every PII, consent,
    and token field is cleared; only profile_id, created_at, deleted_at remain
    meaningful.

    Atomic: all writes run in a single transaction. Any failure rolls back.
    """
    async with acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SET LOCAL statement_timeout = '30s'")
            await cur.execute(
                "SELECT 1 FROM profile.user_profile "
                "WHERE profile_id = %s AND deleted_at IS NULL",
                (profile_id,),
            )
            if await cur.fetchone() is None:
                raise HTTPException(404, "Profile not found")

            await cur.execute(
                "DELETE FROM interaction.recommendation_feedback WHERE profile_id = %s",
                (profile_id,),
            )
            await cur.execute(
                "DELETE FROM interaction.chat_event WHERE profile_id = %s",
                (profile_id,),
            )
            await cur.execute(
                "DELETE FROM analytics.training_recommendation WHERE profile_id = %s",
                (profile_id,),
            )
            await cur.execute(
                "DELETE FROM analytics.job_match WHERE profile_id = %s",
                (profile_id,),
            )
            await cur.execute(
                "DELETE FROM profile.user_skill WHERE profile_id = %s",
                (profile_id,),
            )
            await cur.execute(
                """
                UPDATE profile.user_profile SET
                    session_id               = NULL,
                    preferred_location       = NULL,
                    target_role_text         = NULL,
                    education_text           = NULL,
                    experience_text          = NULL,
                    skills_text              = NULL,
                    work_type_preference     = NULL,
                    language_preferences     = '{}',
                    salary_expectation_text  = NULL,
                    shift_preference         = NULL,
                    transportation_text      = NULL,
                    availability_text        = NULL,
                    resume_text              = NULL,
                    resume_filename          = NULL,
                    resume_parsed_at         = NULL,
                    resume_facts_json        = NULL,
                    consent_version          = NULL,
                    consent_purposes         = '{}',
                    consent_granted_at       = NULL,
                    session_token_hash       = NULL,
                    session_token_expires_at = NULL,
                    deleted_at               = NOW(),
                    updated_at               = NOW()
                 WHERE profile_id = %s
                """,
                (profile_id,),
            )
    return await envelope(request, {"deleted": True, "profile_id": profile_id})


# ------------------------------------------------------------- GET /me/skills
@router.get("/me/skills")
async def list_user_skills(request: Request, profile_id: str = Depends(require_profile)):
    rows = await fetch_all(
        "SELECT skill_id, skill_name, source, confidence FROM profile.user_skill "
        "WHERE profile_id = %s",
        (profile_id,),
    )
    return await envelope(request, rows)
