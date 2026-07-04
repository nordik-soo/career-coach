"""Central configuration for SkillBridge SSM API.

All values are env-driven so deployments differ only by .env contents.
Placeholders in .env.example are clearly labelled PLACEHOLDER_* — set them
when you have the real partner URL or API key.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


def _bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _list(name: str, default: str = "") -> list[str]:
    raw = os.getenv(name, default)
    return [v.strip() for v in raw.split(",") if v.strip()]


# --------------------------------------------------------------------- Service
APP_ENV = os.getenv("APP_ENV", "dev")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = _int("API_PORT", 8000)
REQUIRE_AUTH = _bool("REQUIRE_AUTH", False)
CORS_ALLOW_ORIGINS = _list("CORS_ALLOW_ORIGINS", "*")


# -------------------------------------------------------------------- Database
@dataclass(frozen=True)
class DBConfig:
    host: str = os.getenv("PGHOST", "localhost")
    port: int = _int("PGPORT", 5432)
    database: str = os.getenv("PGDATABASE", "skillbridge")
    user: str = os.getenv("PGUSER", "skillbridge")
    password: str = os.getenv("PGPASSWORD", "")

    @property
    def dsn(self) -> str:
        return (
            f"host={self.host} port={self.port} dbname={self.database} "
            f"user={self.user} password={self.password}"
        )


DB = DBConfig()


# ----------------------------------------------------------- Session / auth
SESSION_STORE = os.getenv("SESSION_STORE", "redis").strip().lower()
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
SESSION_TTL_MINUTES = _int("SESSION_TTL_MINUTES", 30)
SESSION_COOKIE_SECRET = os.getenv("SESSION_COOKIE_SECRET", "")
PROFILE_TOKEN_TTL_HOURS = _int("PROFILE_TOKEN_TTL_HOURS", 720)

# Admin gate for /v1/admin/* endpoints.
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")
REQUIRE_ADMIN_AUTH = _bool("REQUIRE_ADMIN_AUTH", True)


# ------------------------------------------------------------------- Anthropic
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")
LLM_FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL", "claude-sonnet-4-6")
LLM_MAX_TOKENS = _int("LLM_MAX_TOKENS", 600)
LLM_TIMEOUT_SECONDS = _int("LLM_TIMEOUT_SECONDS", 30)
LLM_ENABLED = _bool("LLM_ENABLED", True) and bool(ANTHROPIC_API_KEY) and not ANTHROPIC_API_KEY.startswith("PLACEHOLDER")


# ---------------------------------------------------------- Chat orchestration
# Hard rollback switch for the v2 chat pipeline (gates -> planner ->
# arbiter -> responder v2). When set to "v2" (current default after
# anonymous-chat acceptance testing), the handler routes turns through
# the new pipeline; when set to "v1", the legacy state-machine path
# runs unchanged for rollback. There is NO mixed half-path: a single
# dispatch at the top of handle_anonymous picks one end-to-end. Inside
# v2, the arbiter may explicitly fall back to the v1 path when the
# planner returns None (graceful degradation), but that fallback is
# observable and tested -- not "half v1 + half v2".
# See docs/chat-orchestration-v2-design.md sections 6.3 and 10.
CHAT_ORCHESTRATOR = os.getenv("CHAT_ORCHESTRATOR", "v2").strip().lower()
if CHAT_ORCHESTRATOR not in {"v1", "v2"}:
    raise ValueError(
        f"CHAT_ORCHESTRATOR must be 'v1' or 'v2'; got "
        f"{CHAT_ORCHESTRATOR!r}. No mixed half-path is supported."
    )


# ---------------------------------------------------------- Training registry
# Gates whether handle_anonymous consults the curated YAML training
# registry (data/training_registry.yaml) on explain_gap turns to
# populate the TRAINING block. Default OFF until the lead engineer
# has verified at least the 310T URL pathway. When OFF, the existing
# DB-backed `_attach_training` path runs unchanged (which today is
# effectively a no-op because the DB has 0 rows).
#
# When ON:
#   - explain_gap turns query the registry for last_presented_credential_gaps
#   - Resources with null/stale verified_at have their URL suppressed
#     at runtime (the safety net carries the load -- the registry itself
#     is structurally safe; chat never sees an unverified URL)
#   - Unknown gaps log INFO 'unknown_gap=...' for backlog telemetry
TRAINING_REGISTRY_ENABLED = _bool("TRAINING_REGISTRY_ENABLED", False)


# ------------------------------------------------- Drilldown semantic mode
# Slice 7a (2026-06-30): tri-state rollout switch for the role
# drilldown's 4th match cascade rung (semantic embedding +
# cosine similarity). Bridges the OaSIS abstract-competency vs
# resume concrete-task vocabulary mismatch surfaced in slice 5
# live verify.
#
# Values:
#   off  (default) -- exact-only cascade (id -> canonical -> name).
#                     No semantic computation. No log. Production-safe.
#   log            -- compute semantic scores + emit Cartesian
#                     calibration log per drilldown. Does NOT affect
#                     the visible ✓/✗ status. Use for calibration.
#   on             -- semantic affects ✓/✗ at the hardcoded threshold
#                     in recommender_assembly._DRILLDOWN_SEMANTIC_THRESHOLD.
#                     Use only after calibration locks the threshold.
#
# Bad value (e.g. "banana", "1", "", whitespace) -> off (defensive).
# Case-insensitive; whitespace-tolerant.
#
# Rollout sequence: deploy with off -> flip to log locally + inspect
# calibration log -> pick threshold from observed distribution +
# hardcode in recommender_assembly -> flip to on locally to verify
# table behavior -> deploy with on (or stay off until ready).
_DRILLDOWN_SEMANTIC_RAW = os.environ.get(
    "DRILLDOWN_SEMANTIC", "off",
).strip().lower()
_DRILLDOWN_SEMANTIC_VALID: set[str] = {"off", "log", "on"}
DRILLDOWN_SEMANTIC_MODE: str = (
    _DRILLDOWN_SEMANTIC_RAW
    if _DRILLDOWN_SEMANTIC_RAW in _DRILLDOWN_SEMANTIC_VALID
    else "off"
)


# ------------------------------------------------- Coach Training Guide gate
# Slice 9 (2026-06-30): DRILLDOWN_COACH_GUIDE binary env var that gates
# whether the drilldown assembly generates a Coach Training Guide section
# beneath the skill-comparison table.
#
# Values (case-insensitive, whitespace-tolerant, bad -> off defensively):
#   off  -- coach guide NOT generated. Table renders alone.
#           Production-safe default during rollout.
#   on   -- coach guide generated:
#             * Zero gaps: deterministic encouragement template
#               referencing the top-3 matched skills. No LLM call.
#             * >=1 gap: one Anthropic tool_use call per drilldown,
#               ~$0.01 per turn, ~2-4s added latency.
#
# Rollout: deploy off -> flip to on locally + live verify -> deploy on.
_DRILLDOWN_COACH_GUIDE_RAW = os.environ.get(
    "DRILLDOWN_COACH_GUIDE", "off",
).strip().lower()
_DRILLDOWN_COACH_GUIDE_VALID: set[str] = {"off", "on"}
DRILLDOWN_COACH_GUIDE_MODE: str = (
    _DRILLDOWN_COACH_GUIDE_RAW
    if _DRILLDOWN_COACH_GUIDE_RAW in _DRILLDOWN_COACH_GUIDE_VALID
    else "off"
)


# ------------------------------------------------- Deterministic message routing
# Gates whether the handler runs the deterministic `route_from_understanding`
# layer (chat orchestration v2.1) before the planner LLM call. When OFF
# (default during rollout), the planner-first path runs unchanged and the
# arbiter overrides (planner-overreach catches) carry the load. When ON,
# `understand_message` classifies the user message; if the result is
# HIGH-confidence (scope_violation, training_request with/without entity,
# or job_search with truth_summary ready), a PlannerDecision is synthesized
# deterministically and the planner LLM call is SKIPPED entirely.
#
# Design: docs/message-understanding-design.md
# Rollback: flip to False; the planner-first path is preserved exactly.
# Flip-to-true criterion: live regression tests in Slice C pass.
MESSAGE_UNDERSTANDING_ENABLED = _bool("MESSAGE_UNDERSTANDING_ENABLED", False)


# -------------------------------------------------------- Adjacent recommendations
#
# AR-6a (adjacent-recommendations design v12 amendment): the adjacency
# dispatch helpers and engine pipeline are wired into _try_v2_path,
# but the responder payload threading (adjacent_recommendations_payload,
# adjacent_role_description_payload) doesn't land until AR-6c. Until
# then, an active dispatch would fall through to the generic
# "no match" responder line, which is misleading to Redis users.
#
# Activation contract: the flag stays OFF in production until AR-6c
# is signed off. Acceptance tests flip it ON via env var or
# monkeypatch. See docs/adjacent-recommendations-design.md v11
# §"Activation deferral" and v12 §"Redis-mode activation gate".
#
# Rollback: flip to False; the pre-AR-1 user experience is preserved
# (no soft offer, no intent dispatch, no engine call).
ADJACENCY_ACTIVATION_ENABLED = _bool("ADJACENCY_ACTIVATION_ENABLED", False)


# ---------------------------------------------------------------------- Region
REGION_NAME = os.getenv("REGION_NAME", "Sault Ste. Marie")
# SSM-only product: no national region codes. National feeds (Job Bank,
# StatCan, Census) are NOT used as data sources — see BREAKING.md.
REGION_FSAS = _list("REGION_FSAS", "P6A,P6B,P6C,P0R,P0S,P5A")
LOCAL_CITIES = {c.lower() for c in _list("LOCAL_CITIES", "Sault Ste. Marie")}

# SSM coordinate bounding box for geo-tagged ingest sources (v1: AWIC jobs).
# Defaults are approximate; refinable via env vars. Coordinates are in
# WGS84 (lng, lat) matching AWIC's GeoJSON CRS urn:ogc:def:crs:OGC:1.3:CRS84.
SSM_BBOX_LAT_MIN = float(os.getenv("SSM_BBOX_LAT_MIN", "46.4"))
SSM_BBOX_LAT_MAX = float(os.getenv("SSM_BBOX_LAT_MAX", "46.6"))
SSM_BBOX_LNG_MIN = float(os.getenv("SSM_BBOX_LNG_MIN", "-84.5"))
SSM_BBOX_LNG_MAX = float(os.getenv("SSM_BBOX_LNG_MAX", "-84.2"))


# -------------------------------------------------------------------- Pipeline
PIPELINE_DAILY_HOUR_ET = _int("PIPELINE_DAILY_HOUR_ET", 6)
STALE_SWEEP_NOT_SEEN_DAYS = _int("STALE_SWEEP_NOT_SEEN_DAYS", 14)
# The 30-day "current job" freshness window is a product invariant, enforced
# in sql/schema.sql via core.v_current_job. Changing it requires a schema
# migration + engine_version bump, not a config flip.


# ------------------------------------------------------------------ Job sources
@dataclass(frozen=True)
class SourceConfig:
    name: str
    enabled: bool
    url: str
    api_key: str = ""
    extra: dict[str, str] = field(default_factory=dict)


JOB_SOURCES: list[SourceConfig] = [
    SourceConfig(
        name="sccc",
        # SCCC is a primary source — registered, not toggled. The
        # connector points at WordPress's public REST API by default and
        # always runs. Override only if SCCC publishes a new endpoint.
        enabled=_bool("SCCC_ENABLED", True),
        url=os.getenv("SCCC_FEED_URL", ""),
        api_key=os.getenv("SCCC_API_KEY", ""),
        extra={"format": os.getenv("SCCC_FEED_FORMAT", "wp_rest")},
    ),
    SourceConfig(
        name="awic_jobs",
        # AWIC jobs are a primary source — registered, not toggled.
        # The connector points at AWIC's public GeoJSON REST endpoint
        # by default and always runs. SEPARATE flag from AWIC labour-
        # market reports (AWIC_ENABLED) on purpose: jobs and reports
        # do not share a feature flag.
        #
        # Provenance policy: AWIC is a local aggregator. Some postings'
        # properties.url values point to third-party sources including
        # Job Bank. SkillBridge ingests AWIC's curated metadata layer
        # ONLY, not the third-party sources. Source is stamped as
        # 'awic_jobs'; the third-party URL is stored as the apply-URL.
        # See sql/schema.sql (core.approved_job_source description for
        # awic_jobs) and BREAKING.md for the durable provenance rule.
        enabled=_bool("AWIC_JOBS_ENABLED", True),
        url=os.getenv("AWIC_JOBS_FEED_URL", ""),
        extra={"format": os.getenv("AWIC_JOBS_FEED_FORMAT", "geojson")},
    ),
    SourceConfig(
        name="welcome_ssm",
        enabled=_bool("WELCOME_SSM_ENABLED", False),
        url=os.getenv("WELCOME_SSM_FEED_URL", ""),
    ),
    SourceConfig(
        name="city_ssm",
        enabled=_bool("CITY_SSM_ENABLED", False),
        url=os.getenv("CITY_SSM_CAREERS_URL", ""),
    ),
    SourceConfig(
        name="partner_csv",
        enabled=_bool("PARTNER_CSV_ENABLED", True),
        url=os.getenv("PARTNER_CSV_DIR", "./data/partner_uploads"),
    ),
]


# SSM employer career-page sources. Each entry has its own *_ENABLED + *_URL.
# Two are reference parsers (sault_area_hospital, city_of_ssm_hr); the rest
# are stubs awaiting selector verification against live pages.
EMPLOYER_SOURCES: list[SourceConfig] = [
    SourceConfig(name="sault_area_hospital",
                 enabled=_bool("SAULT_AREA_HOSPITAL_ENABLED", False),
                 url=os.getenv("SAULT_AREA_HOSPITAL_URL", "")),
    SourceConfig(name="city_of_ssm_hr",
                 enabled=_bool("CITY_OF_SSM_HR_ENABLED", False),
                 url=os.getenv("CITY_OF_SSM_HR_URL", "")),
    SourceConfig(name="algoma_steel",
                 enabled=_bool("ALGOMA_STEEL_ENABLED", False),
                 url=os.getenv("ALGOMA_STEEL_URL", "")),
    SourceConfig(name="sault_college_careers",
                 enabled=_bool("SAULT_COLLEGE_CAREERS_ENABLED", False),
                 url=os.getenv("SAULT_COLLEGE_CAREERS_URL", "")),
    SourceConfig(name="algoma_u_careers",
                 enabled=_bool("ALGOMA_U_CAREERS_ENABLED", False),
                 url=os.getenv("ALGOMA_U_CAREERS_URL", "")),
    SourceConfig(name="puc",
                 enabled=_bool("PUC_ENABLED", False),
                 url=os.getenv("PUC_URL", "")),
    SourceConfig(name="group_health_centre",
                 enabled=_bool("GROUP_HEALTH_CENTRE_ENABLED", False),
                 url=os.getenv("GROUP_HEALTH_CENTRE_URL", "")),
    SourceConfig(name="ymca_ssm",
                 enabled=_bool("YMCA_SSM_ENABLED", False),
                 url=os.getenv("YMCA_SSM_URL", "")),
    SourceConfig(name="cas_algoma",
                 enabled=_bool("CAS_ALGOMA_ENABLED", False),
                 url=os.getenv("CAS_ALGOMA_URL", "")),
    SourceConfig(name="adsab",
                 enabled=_bool("ADSAB_ENABLED", False),
                 url=os.getenv("ADSAB_URL", "")),
    SourceConfig(name="school_board",
                 enabled=_bool("SCHOOL_BOARD_ENABLED", False),
                 url=os.getenv("SCHOOL_BOARD_URL", "")),
]


TRAINING_SOURCES: list[SourceConfig] = [
    SourceConfig(
        name="sault_college",
        enabled=_bool("SAULT_COLLEGE_ENABLED", False),
        url=os.getenv("SAULT_COLLEGE_PROGRAMS_URL", ""),
        extra={"continuing_ed": os.getenv("SAULT_COLLEGE_CONTINUING_ED_URL", "")},
    ),
    SourceConfig(
        name="algoma_u",
        enabled=_bool("ALGOMA_U_ENABLED", False),
        url=os.getenv("ALGOMA_U_PROGRAMS_URL", ""),
    ),
    SourceConfig(
        name="northland",
        enabled=_bool("NORTHLAND_ENABLED", False),
        url=os.getenv("NORTHLAND_RESOURCES_URL", ""),
    ),
    SourceConfig(
        name="sccc_services",
        enabled=_bool("SCCC_SERVICES_ENABLED", False),
        url=os.getenv("SCCC_SERVICES_URL", ""),
    ),
]


# ---------------------------------------------------- Supplementary sources
# PR 6A: registry-only. Connectors load data into knowledge.* / core.* tables
# but don't yet feed the matching engine. RCIP scoring + chat badge is PR 6B.

AWIC_ENABLED = _bool("AWIC_ENABLED", False)
AWIC_REPORTS_URL = os.getenv("AWIC_REPORTS_URL", "")
AWIC_DATA_FEED_URL = os.getenv("AWIC_DATA_FEED_URL", "")

RCIP_ENABLED = _bool("RCIP_ENABLED", False)
RCIP_EMPLOYER_LIST_URL = os.getenv("RCIP_EMPLOYER_LIST_URL", "")
RCIP_EMPLOYER_LIST_CSV = os.getenv("RCIP_EMPLOYER_LIST_CSV", "./data/rcip_employers.csv")

SSM_LIP_ENABLED = _bool("SSM_LIP_ENABLED", False)
SSM_LIP_SERVICES_URL = os.getenv("SSM_LIP_SERVICES_URL", "")

SAULT_CHAMBER_ENABLED = _bool("SAULT_CHAMBER_ENABLED", False)
SAULT_CHAMBER_DIRECTORY_URL = os.getenv("SAULT_CHAMBER_DIRECTORY_URL", "")

CITY_SSM_OPEN_DATA_ENABLED = _bool("CITY_SSM_OPEN_DATA_ENABLED", False)
CITY_SSM_OPEN_DATA_URL = os.getenv("CITY_SSM_OPEN_DATA_URL", "")


# ------------------------------------------------------------- Reference paths
NOC_CSV_PATH = Path(os.getenv("NOC_CSV_PATH", "./data/noc_2021.csv"))
OASIS_SKILL_CSV = Path(os.getenv("OASIS_SKILL_CSV", "./data/oasis_skills.csv"))
NOC_SKILL_CSV = Path(os.getenv("NOC_SKILL_CSV", "./data/noc_skill_mapping.csv"))
REGULATED_OCCUPATIONS_CSV = Path(os.getenv("REGULATED_OCCUPATIONS_CSV", "./data/regulated_occupations_on.csv"))

# Matching v2 step 1: OaSIS + SCT occupation-title lexicon.
# Files downloaded manually from open.canada.ca -- see docs/oasis-download.md.
# Loaders skip gracefully when files are absent so the rest of the pipeline
# still runs.
OASIS_VERSION = os.getenv("OASIS_VERSION", "2025_v1.0")
OASIS_EXAMPLE_TITLES_EN_CSV = Path(os.getenv(
    "OASIS_EXAMPLE_TITLES_EN_CSV",
    "./data/example-titles_oasis_2025_v1.0.csv",
))
OASIS_EXAMPLE_TITLES_FR_CSV = Path(os.getenv(
    "OASIS_EXAMPLE_TITLES_FR_CSV",
    "./data/exemples-dappellation-demploi_sipec_2025_v1.0.csv",
))
OASIS_LEAD_STATEMENT_EN_CSV = Path(os.getenv(
    "OASIS_LEAD_STATEMENT_EN_CSV",
    "./data/lead-statement_oasis_2025_v1.0.csv",
))
OASIS_LEAD_STATEMENT_FR_CSV = Path(os.getenv(
    "OASIS_LEAD_STATEMENT_FR_CSV",
    "./data/enonce-principal_sipec_2025_v1.0.csv",
))
SCT_ALTERNATIVE_TITLES_CSV = Path(os.getenv(
    "SCT_ALTERNATIVE_TITLES_CSV",
    "./data/alternatives-titles-skills-and-competencies-taxonomy-2023-version-1.0-en-fr.csv",
))


# ----------------------------------------------------------- Skill extraction
EXTRACTION_BATCH_SIZE = _int("EXTRACTION_BATCH_SIZE", 20)
EXTRACTION_MAX_CONCURRENCY = _int("EXTRACTION_MAX_CONCURRENCY", 4)
SKILL_FUZZY_THRESHOLD = _float("SKILL_FUZZY_THRESHOLD", 0.75)


# Matching v2 step 5: semantic re-ranker. Soft dependency on
# sentence-transformers; engine falls back to lexical-only when absent.
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_MODEL_VERSION = os.getenv("EMBEDDING_MODEL_VERSION", "all-MiniLM-L6-v2-v1")
# Cosine similarity threshold below which we DO NOT treat as a semantic
# match (engine treats as no-match -> strength 0.0). Above this, the
# semantic stage contributes a strength capped at SEMANTIC_STRENGTH_CAP.
SEMANTIC_COSINE_THRESHOLD = _float("SEMANTIC_COSINE_THRESHOLD", 0.70)
# Strength assigned when semantic match fires. Capped well below stage-1
# (1.0) and below token-overlap fuzzy (0.85) so semantic NEVER overrides
# a lexical match via the engine's max-wins rule. Semantic only "wins"
# the strength assignment when no lexical match exists.
SEMANTIC_STRENGTH_CAP = _float("SEMANTIC_STRENGTH_CAP", 0.75)
MIN_EXTRACTION_CONFIDENCE = _float("MIN_EXTRACTION_CONFIDENCE", 0.60)


# ------------------------------------------------------------- Match engine v1
@dataclass(frozen=True)
class MatchEngineConfig:
    # Sprint 5 activation: v1.2.0 extraction produces 15-28 skills per JD
    # (up from 5-8 on v1.1.0). Top-5 was the right cutoff for the sparser
    # old extraction; with richer output, important skills (including
    # credentials) get pushed below position 5 by importance_rank and
    # disappear from the matching set. Bumped to 12 so the matcher sees
    # the meaningful core of each JD. Credentials get a separate carve-out
    # in _filter_eligible_skills so they survive regardless of rank.
    top_n_required_skills: int = _int("MATCH_TOP_N_REQUIRED_SKILLS", 12)
    min_required_skills_for_eligibility: int = _int("MATCH_MIN_REQUIRED_SKILLS_FOR_ELIGIBILITY", 3)
    recency_boost_days: int = _int("MATCH_RECENCY_BOOST_DAYS", 30)
    location_boost_local_csd: float = _float("MATCH_LOCATION_BOOST_LOCAL_CSD", 0.10)
    band_strong: float = _float("MATCH_BAND_STRONG", 0.75)
    band_good: float = _float("MATCH_BAND_GOOD", 0.60)
    band_stretch: float = _float("MATCH_BAND_STRETCH", 0.40)


MATCH = MatchEngineConfig()


# ------------------------------------------------------------- Consent purposes
CONSENT_PURPOSES = (
    "profile_storage",
    "job_recommendation",
    "training_recommendation",
    "follow_up_contact",
)
CONSENT_VERSION = "v1"
