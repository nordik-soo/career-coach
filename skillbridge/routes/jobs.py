"""Public job dashboard endpoints.

Per design §3, the dashboard is a public utility — no personalization, no
scores, no missing-skill info. Just current SSM jobs and search/filter.
"""
from __future__ import annotations

from fastapi import APIRouter, Query, Request, HTTPException

from skillbridge.db import fetch_all, fetch_one
from skillbridge.routes._envelope import envelope

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])


@router.get("")
async def list_jobs(
    request: Request,
    q: str | None = Query(None, description="title/employer/description search"),
    source: str | None = None,
    employer: str | None = None,
    employment_type: str | None = None,
    remote: bool | None = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """List current SSM jobs (active + within freshness window)."""
    where = ["TRUE"]
    params: list = []
    if q:
        where.append("(title ILIKE %s OR employer ILIKE %s OR description ILIKE %s)")
        pat = f"%{q}%"
        params.extend([pat, pat, pat])
    if source:
        where.append("source = %s")
        params.append(source)
    if employer:
        where.append("employer ILIKE %s")
        params.append(f"%{employer}%")
    if employment_type:
        where.append("employment_type = %s")
        params.append(employment_type)
    if remote is not None:
        where.append("remote_flag = %s")
        params.append(remote)
    sql = f"""
    SELECT job_id, source, title, employer, location, posted_date, closing_date,
           salary_text, employment_type, remote_flag, url, noc_code, is_active
      FROM core.v_current_job
     WHERE {' AND '.join(where)}
     ORDER BY posted_date DESC NULLS LAST, ingested_at DESC
     LIMIT %s OFFSET %s
    """
    params.extend([limit, offset])
    rows = await fetch_all(sql, tuple(params))
    return await envelope(request, rows)


@router.get("/{job_id}")
async def get_job(request: Request, job_id: str):
    job = await fetch_one(
        """
        SELECT j.*, e.name AS employer_name
          FROM core.job_posting j
          LEFT JOIN core.employer e ON e.employer_id = j.employer_id
         WHERE j.job_id = %s
        """,
        (job_id,),
    )
    if not job:
        raise HTTPException(404, "Job not found")
    skills = await fetch_all(
        """
        SELECT skill_id, skill_name, confidence, importance_rank, skill_type
          FROM extracted.job_skill
         WHERE job_id = %s
         ORDER BY importance_rank NULLS LAST
        """,
        (job_id,),
    )
    return await envelope(request, {**job, "skills": skills})
