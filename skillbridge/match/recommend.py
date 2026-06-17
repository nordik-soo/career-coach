"""Training recommendation engine.

For each missing skill, find local resources that teach or support it.

Guardrails (from §11 of the design):
- short skill gaps prefer short resources
- don't suggest multi-year programs for basic skills unless the target job
  requires it
- cap at 3 suggestions per skill
- if no direct local resource, return an honest "no local match" sentinel
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from skillbridge.db import sync_cursor
from skillbridge.versions import ENGINE_VERSION_TRAINING_REC

log = logging.getLogger(__name__)


@dataclass
class TrainingSuggestion:
    resource_id: str | None
    provider: str
    title: str
    url: str | None
    duration_band: str | None
    resource_type: str | None
    reason: str
    score: float


# Basic vs. advanced skill heuristic. If a missing skill is in BASIC_SKILLS,
# we down-rank multi-year programs ("long" duration_band).
BASIC_SKILLS = {
    "microsoft excel", "microsoft word", "microsoft office", "computer skills",
    "typing", "data entry", "english", "esl", "phone communication",
    "customer service", "cash handling", "scheduling", "time management",
}


_NO_LOCAL_MATCH = TrainingSuggestion(
    resource_id=None,
    provider="Sault Community Career Centre",
    title="Speak with a career counsellor",
    url=None,
    duration_band=None,
    resource_type="counselling",
    reason=("No local program directly teaches this skill in the current dataset. "
            "SCCC counsellors can help find the right next step."),
    score=0.0,
)


def suggest_for_skill(skill_name: str, *, target_job_is_advanced: bool = False,
                      limit: int = 3) -> list[TrainingSuggestion]:
    """Find up to `limit` local resources matching this missing skill."""
    if not skill_name:
        return []
    is_basic = skill_name.strip().lower() in BASIC_SKILLS

    sql = """
    SELECT r.resource_id, r.provider, r.title, r.url, r.duration_band,
           r.resource_type, r.delivery_mode, r.location,
           MAX(ts.confidence) AS skill_confidence,
           similarity(ts.skill_name, %s) AS name_sim
      FROM core.training_resource r
      JOIN extracted.training_skill ts ON ts.resource_id = r.resource_id
     WHERE r.is_active = TRUE
       AND (ts.skill_name ILIKE %s OR similarity(ts.skill_name, %s) > 0.5)
     GROUP BY r.resource_id, ts.skill_name
     ORDER BY name_sim DESC, skill_confidence DESC NULLS LAST
     LIMIT 12
    """
    pattern = f"%{skill_name}%"
    with sync_cursor() as cur:
        cur.execute(sql, (skill_name, pattern, skill_name))
        rows = list(cur.fetchall())

    if not rows:
        return [_NO_LOCAL_MATCH]

    out: list[TrainingSuggestion] = []
    seen: set[str] = set()
    for r in rows:
        rid = str(r["resource_id"])
        if rid in seen:
            continue
        seen.add(rid)
        score = float(r.get("name_sim") or 0)
        # Apply guardrails:
        if is_basic and not target_job_is_advanced and r.get("duration_band") == "long":
            # Down-rank a multi-year program for a basic skill.
            score *= 0.5
        suggestion = TrainingSuggestion(
            resource_id=rid,
            provider=r["provider"],
            title=r["title"],
            url=r.get("url"),
            duration_band=r.get("duration_band"),
            resource_type=r.get("resource_type"),
            reason=f"Teaches/supports the missing skill: {skill_name}",
            score=round(score, 3),
        )
        out.append(suggestion)
        if len(out) >= limit:
            break

    if not out:
        return [_NO_LOCAL_MATCH]
    return out


def suggest_for_profile(profile_id: str, job_id: str | None = None) -> list[TrainingSuggestion]:
    """Return training suggestions for all missing skills associated with
    this profile (optionally constrained to a single job)."""
    sql = """
    SELECT DISTINCT skill_name
      FROM analytics.job_match m
      JOIN analytics.job_match_skill s ON s.match_id = m.match_id
     WHERE m.profile_id = %s
       AND s.status = 'missing'
    """
    params: tuple = (profile_id,)
    if job_id:
        sql += " AND m.job_id = %s"
        params = (profile_id, job_id)
    with sync_cursor() as cur:
        cur.execute(sql, params)
        skills = [r["skill_name"] for r in cur.fetchall()]
    suggestions: list[TrainingSuggestion] = []
    for sk in skills:
        suggestions.extend(suggest_for_skill(sk))
    return _persist_recommendations(profile_id, job_id, skills, suggestions)


def _persist_recommendations(profile_id: str, job_id: str | None,
                             skill_names: list[str],
                             suggestions: list[TrainingSuggestion]) -> list[TrainingSuggestion]:
    with sync_cursor() as cur:
        cur.execute(
            "DELETE FROM analytics.training_recommendation "
            "WHERE profile_id = %s AND (%s::uuid IS NULL OR job_id = %s::uuid)",
            (profile_id, job_id, job_id),
        )
        # Pair suggestion order to skills only loosely — many suggestions per skill
        # via suggest_for_skill; we re-derive per-skill via the reason string.
        for s in suggestions:
            if s.resource_id is None:
                continue
            cur.execute(
                """
                INSERT INTO analytics.training_recommendation
                    (profile_id, job_id, resource_id, skill_name, reason, score, engine_version)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    profile_id, job_id, s.resource_id,
                    s.reason.split(":", 1)[-1].strip() if ":" in s.reason else s.reason,
                    s.reason, s.score, ENGINE_VERSION_TRAINING_REC,
                ),
            )
    return suggestions
