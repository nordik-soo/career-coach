"""Training resources catalog."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from skillbridge.db import fetch_all, fetch_one
from skillbridge.routes._envelope import envelope

router = APIRouter(prefix="/v1/training-resources", tags=["training"])


@router.get("")
async def list_resources(
    request: Request,
    q: str | None = None,
    provider: str | None = None,
    delivery_mode: str | None = None,
    resource_type: str | None = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    where = ["is_active = TRUE"]
    params: list = []
    if q:
        where.append("(title ILIKE %s OR description ILIKE %s)")
        pat = f"%{q}%"
        params.extend([pat, pat])
    if provider:
        where.append("provider ILIKE %s")
        params.append(f"%{provider}%")
    if delivery_mode:
        where.append("delivery_mode = %s")
        params.append(delivery_mode)
    if resource_type:
        where.append("resource_type = %s")
        params.append(resource_type)
    sql = f"""
    SELECT resource_id, provider, title, description, url, location,
           delivery_mode, duration_text, duration_band, resource_type, cost_text
      FROM core.training_resource
     WHERE {' AND '.join(where)}
     ORDER BY provider, title
     LIMIT %s OFFSET %s
    """
    params.extend([limit, offset])
    rows = await fetch_all(sql, tuple(params))
    return await envelope(request, rows)


@router.get("/{resource_id}")
async def get_resource(request: Request, resource_id: str):
    row = await fetch_one(
        "SELECT * FROM core.training_resource WHERE resource_id = %s",
        (resource_id,),
    )
    if not row:
        raise HTTPException(404, "Resource not found")
    skills = await fetch_all(
        "SELECT skill_id, skill_name, confidence FROM extracted.training_skill WHERE resource_id = %s",
        (resource_id,),
    )
    return await envelope(request, {**row, "skills": skills})
