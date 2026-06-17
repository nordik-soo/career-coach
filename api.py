"""SkillBridge SSM FastAPI app.

Run:
    uvicorn api:app --reload --port 8000
"""
from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import API_HOST, API_PORT, CORS_ALLOW_ORIGINS, LOG_LEVEL
from skillbridge.db import close_pool, init_pool
from skillbridge.routes import (
    admin,
    chat,
    consent,
    feedback,
    jobs,
    matches,
    profiles,
    recommendations,
    training,
)
from skillbridge.session import init_store

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await init_pool()
        log.info("API ready on %s:%d", API_HOST, API_PORT)
    except Exception as e:
        log.error("Database init failed: %s", e)
    try:
        await init_store()
    except Exception as e:
        log.error("Session store init failed: %s", e)
    yield
    await close_pool()


app = FastAPI(
    title="SkillBridge SSM API",
    description="Newcomer-centred skill-matching API for Sault Ste. Marie.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request.state.request_id = f"req_{uuid.uuid4().hex[:16]}"
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


# -------------------------------------------------------- Health (no DB)
@app.get("/health/live", tags=["meta"])
async def health_live():
    """Liveness probe — never hits DB. Used by k8s/uvicorn supervisors."""
    return {"status": "ok"}


@app.get("/health/ready", tags=["meta"])
async def health_ready():
    """Readiness — DB + session store. 503 if either is unreachable.

    Load balancers should pull this pod out of rotation when either
    dependency is down. The cookie session store has no external
    dependency, so it only fails when SESSION_COOKIE_SECRET is missing.
    """
    from skillbridge.db import fetch_one
    from skillbridge.session import get_store

    checks: dict[str, bool] = {"db": False, "session_store": False}
    details: dict[str, str] = {}

    try:
        await fetch_one("SELECT 1 AS ok")
        checks["db"] = True
    except Exception as e:
        details["db"] = str(e)[:200]

    try:
        store = get_store()
        ping = getattr(store, "ping", None)
        if callable(ping):
            ping()
        checks["session_store"] = True
    except Exception as e:
        details["session_store"] = str(e)[:200]

    ok = all(checks.values())
    return JSONResponse(
        status_code=200 if ok else 503,
        content={
            "status": "ready" if ok else "not_ready",
            "checks": checks,
            **({"details": details} if details else {}),
        },
    )


@app.get("/v1/meta/version", tags=["meta"])
async def version_info():
    from skillbridge.versions import (
        CHAT_PROMPT_VERSION,
        CONFIDENCE_RULE_VERSION,
        ENGINE_VERSION_JOB_MATCH,
        ENGINE_VERSION_TRAINING_REC,
        EXTRACTOR_VERSION_LLM,
        EXTRACTOR_VERSION_RULE,
    )
    return {
        "engine_versions": {
            "job_match": ENGINE_VERSION_JOB_MATCH,
            "training_rec": ENGINE_VERSION_TRAINING_REC,
        },
        "extractor_versions": {
            "rule": EXTRACTOR_VERSION_RULE,
            "llm": EXTRACTOR_VERSION_LLM,
        },
        "chat_prompt_version": CHAT_PROMPT_VERSION,
        "confidence_rule_version": CONFIDENCE_RULE_VERSION,
    }


# -------------------------------------------------------- Routers
app.include_router(jobs.router)
app.include_router(training.router)
app.include_router(profiles.router)
app.include_router(consent.router)
app.include_router(chat.router)
app.include_router(matches.router)
app.include_router(recommendations.router)
app.include_router(feedback.router)
app.include_router(admin.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host=API_HOST, port=API_PORT, reload=True)
