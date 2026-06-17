-- ============================================================================
-- SkillBridge SSM MVP - Database Schema
-- PostgreSQL 14+, pgvector + pg_trgm + pgcrypto required.
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS unaccent;
-- pgvector is REQUIRED for matching v2 step 5 semantic re-ranker but
-- treated as a SOFT dependency at the schema layer: the migration
-- continues without it, the embedding table is skipped, and the
-- engine falls back to lexical-only matching. Same graceful-degradation
-- pattern as sentence-transformers being missing.
-- To enable semantic matching: install pgvector and re-run --schema.
DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS vector;
EXCEPTION
    WHEN feature_not_supported OR undefined_file THEN
        RAISE NOTICE 'pgvector extension not available -- semantic matching will be disabled. Install pgvector and re-run --schema to enable.';
END$$;

CREATE SCHEMA IF NOT EXISTS reference;
CREATE SCHEMA IF NOT EXISTS core;
CREATE SCHEMA IF NOT EXISTS extracted;
CREATE SCHEMA IF NOT EXISTS profile;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS interaction;
CREATE SCHEMA IF NOT EXISTS pipeline;
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS knowledge;

-- ============================ REFERENCE =====================================
CREATE TABLE IF NOT EXISTS reference.skill (
    skill_id        VARCHAR(40) PRIMARY KEY,
    skill_name      TEXT NOT NULL,
    aliases         TEXT[] NOT NULL DEFAULT '{}',
    source_taxonomy VARCHAR(30) NOT NULL DEFAULT 'OaSIS',
    category        VARCHAR(50),
    description     TEXT
);
CREATE INDEX IF NOT EXISTS ix_ref_skill_name_trgm
    ON reference.skill USING gin (skill_name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS ix_ref_skill_aliases
    ON reference.skill USING gin (aliases);

CREATE TABLE IF NOT EXISTS reference.occupation (
    noc_code        VARCHAR(5) PRIMARY KEY,
    title           TEXT NOT NULL,
    teer            SMALLINT,
    description     TEXT,
    parent_code     VARCHAR(5)
);
CREATE INDEX IF NOT EXISTS ix_ref_occ_title_trgm
    ON reference.occupation USING gin (title gin_trgm_ops);

CREATE TABLE IF NOT EXISTS reference.noc_skill (
    noc_code        VARCHAR(5) REFERENCES reference.occupation(noc_code),
    skill_id        VARCHAR(40) REFERENCES reference.skill(skill_id),
    importance      NUMERIC(3,1),
    level           NUMERIC(3,1),
    PRIMARY KEY (noc_code, skill_id)
);

-- ============================ CORE ==========================================

-- Ontario regulated occupations (e.g. RN, electrician, accountant).
-- Triggers a credential warning in match results.
CREATE TABLE IF NOT EXISTS core.regulated_occupation (
    noc_code           VARCHAR(5) PRIMARY KEY,
    occupation_title   TEXT NOT NULL,
    regulator_name     TEXT NOT NULL,
    regulator_url      TEXT,
    licensing_note     TEXT
);

-- Employer registry. RCIP-designated and Welcome-to-SSM partners get a boost.
CREATE TABLE IF NOT EXISTS core.employer (
    employer_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name               TEXT NOT NULL,
    name_normalized    TEXT NOT NULL,
    industry           TEXT,
    primary_location   TEXT,
    UNIQUE (name_normalized)
);
CREATE INDEX IF NOT EXISTS ix_employer_name_trgm
    ON core.employer USING gin (name gin_trgm_ops);

CREATE TABLE IF NOT EXISTS core.employer_designation (
    employer_id        UUID REFERENCES core.employer(employer_id) ON DELETE CASCADE,
    designation_type   VARCHAR(40) NOT NULL,    -- rcip | welcome_ssm_partner | sccc_partner
    granted_at         DATE,
    expires_at         DATE,
    source             TEXT,
    PRIMARY KEY (employer_id, designation_type)
);

-- =============== APPROVED JOB SOURCES (SSM-only product rule) ===============
-- HARD INVARIANT: no row may exist in core.job_posting unless its `source`
-- is in this table. Federal feeds (Job Bank, StatCan, Census) are NOT
-- listed and CANNOT be added without a conscious schema migration. See
-- BREAKING.md and tests/test_source_purity.py for the durable guard.
CREATE TABLE IF NOT EXISTS core.approved_job_source (
    source       VARCHAR(40) PRIMARY KEY,
    description  TEXT NOT NULL,
    scope        VARCHAR(20) NOT NULL,    -- 'partner' | 'employer' | 'bridge'
    added_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

INSERT INTO core.approved_job_source (source, description, scope) VALUES
    ('sccc',                  'Sault Community Career Centre job board (Tier 1, intended primary)', 'partner'),
    ('welcome_ssm',           'Welcome to SSM Careers (Tier 1, partner)',                            'partner'),
    ('city_ssm',              'City of SSM feed (partner-sanctioned, when available)',               'partner'),
    ('partner_csv',           'Partner CSV upload bridge (unknown / fallback partner)',              'bridge'),
    ('sault_area_hospital',   'Sault Area Hospital career page',                                     'employer'),
    ('city_of_ssm_hr',        'City of Sault Ste. Marie HR career page',                             'employer'),
    ('algoma_steel',          'Algoma Steel career page',                                            'employer'),
    ('sault_college_careers', 'Sault College HR career page (distinct from training programs)',      'employer'),
    ('algoma_u_careers',      'Algoma University HR career page (distinct from training programs)', 'employer'),
    ('puc',                   'PUC Services career page',                                            'employer'),
    ('group_health_centre',   'Group Health Centre career page',                                     'employer'),
    ('ymca_ssm',              'YMCA of Sault Ste. Marie career page',                                'employer'),
    ('cas_algoma',            'Children''s Aid Society of Algoma career page',                       'employer'),
    ('adsab',                 'Algoma District Services Administration Board career page',          'employer'),
    ('school_board',          'Local school board(s) career page (multiple boards aggregated)',     'employer')
ON CONFLICT (source) DO NOTHING;


-- Job postings (de-duplicated, normalized).
-- INVARIANT: source must be in core.approved_job_source.
CREATE TABLE IF NOT EXISTS core.job_posting (
    job_id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source             VARCHAR(40) NOT NULL REFERENCES core.approved_job_source(source),
    source_job_id      VARCHAR(120) NOT NULL,
    title              TEXT NOT NULL,
    employer           TEXT,
    employer_id        UUID REFERENCES core.employer(employer_id),
    location           TEXT,
    region_code        VARCHAR(10),
    description        TEXT,
    url                TEXT,
    posted_date        DATE,
    closing_date       DATE,
    salary_text        TEXT,
    salary_low         NUMERIC,
    salary_high        NUMERIC,
    employment_type    VARCHAR(40),
    remote_flag        BOOLEAN,
    noc_code           VARCHAR(5),
    is_active          BOOLEAN NOT NULL DEFAULT TRUE,
    last_seen_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ingested_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (source, source_job_id)
);
CREATE INDEX IF NOT EXISTS ix_job_active ON core.job_posting(is_active);
CREATE INDEX IF NOT EXISTS ix_job_posted_date ON core.job_posting(posted_date DESC);
CREATE INDEX IF NOT EXISTS ix_job_title_trgm
    ON core.job_posting USING gin (title gin_trgm_ops);
CREATE INDEX IF NOT EXISTS ix_job_noc ON core.job_posting(noc_code);

-- Skills extracted from a job posting's text.
CREATE TABLE IF NOT EXISTS extracted.job_skill (
    job_id             UUID REFERENCES core.job_posting(job_id) ON DELETE CASCADE,
    skill_id           VARCHAR(40) REFERENCES reference.skill(skill_id),
    skill_name         TEXT NOT NULL,
    raw_phrase         TEXT,
    confidence         NUMERIC(3,2),
    importance_rank    SMALLINT,
    extractor_version  VARCHAR(40) NOT NULL,
    extracted_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (job_id, skill_name)
);
CREATE INDEX IF NOT EXISTS ix_job_skill_skillid ON extracted.job_skill(skill_id);

-- Local training & skill-improvement resources.
CREATE TABLE IF NOT EXISTS core.training_resource (
    resource_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    provider           TEXT NOT NULL,
    title              TEXT NOT NULL,
    description        TEXT,
    url                TEXT,
    location           TEXT,
    delivery_mode      VARCHAR(30),       -- in_person | online | hybrid
    cost_text          TEXT,
    duration_text      TEXT,
    duration_band      VARCHAR(10),       -- short | medium | long
    resource_type      VARCHAR(30),       -- workshop | course | program | counselling | online
    is_active          BOOLEAN NOT NULL DEFAULT TRUE,
    ingested_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (provider, title)
);

CREATE TABLE IF NOT EXISTS extracted.training_skill (
    resource_id        UUID REFERENCES core.training_resource(resource_id) ON DELETE CASCADE,
    skill_id           VARCHAR(40) REFERENCES reference.skill(skill_id),
    skill_name         TEXT NOT NULL,
    raw_phrase         TEXT,
    confidence         NUMERIC(3,2),
    extractor_version  VARCHAR(40) NOT NULL,
    extracted_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (resource_id, skill_name)
);
CREATE INDEX IF NOT EXISTS ix_train_skill_skillid ON extracted.training_skill(skill_id);

-- ============================ PROFILE =======================================
CREATE TABLE IF NOT EXISTS profile.user_profile (
    profile_id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id              VARCHAR(256),      -- staged session id from session store (deleted after consent)
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    preferred_location      TEXT,
    target_role_text        TEXT,
    education_text          TEXT,
    experience_text         TEXT,
    skills_text             TEXT,
    work_type_preference    TEXT,
    language_preferences    TEXT[],
    -- PR 10 intake fields (asked during guided chat, persisted on consent)
    salary_expectation_text TEXT,                  -- raw user phrase; not used in scoring yet
    shift_preference        TEXT,                  -- days | evenings | nights | weekends | any | flexible
    transportation_text     TEXT,                  -- "own car", "no car / bus only", etc.
    availability_text       TEXT,                  -- "available immediately", "after May 30", etc.
    -- Sprint 1 resume fields (set on resume upload, persisted on consent).
    -- File bytes are NEVER stored; only extracted text + structured facts.
    -- See docs/resume-design.md §3 for the storage policy.
    resume_text             TEXT,                  -- extracted plain text; null until upload + consent
    resume_filename         TEXT,                  -- original filename for display only
    resume_parsed_at        TIMESTAMPTZ,           -- when the extractor ran (for re-parse detection)
    resume_facts_json       JSONB,                 -- canonical structured parse (see docs/resume-design.md §2)
    consent_version         VARCHAR(20),
    consent_purposes        TEXT[] NOT NULL DEFAULT '{}',
    consent_granted_at      TIMESTAMPTZ,
    -- Opaque session token: raw token returned to client once at consent
    -- grant; only the SHA-256 hash is stored here.
    session_token_hash         CHAR(64),
    session_token_expires_at   TIMESTAMPTZ,
    deleted_at              TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS ix_profile_session ON profile.user_profile(session_id);
CREATE INDEX IF NOT EXISTS ix_profile_token_hash ON profile.user_profile(session_token_hash) WHERE session_token_hash IS NOT NULL;

CREATE TABLE IF NOT EXISTS profile.user_skill (
    profile_id         UUID REFERENCES profile.user_profile(profile_id) ON DELETE CASCADE,
    skill_id           VARCHAR(40) REFERENCES reference.skill(skill_id),
    skill_name         TEXT NOT NULL,
    raw_phrase         TEXT,
    source             VARCHAR(20),       -- chat | form | resume | manual_update
    confidence         NUMERIC(3,2),
    added_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (profile_id, skill_name)
);

-- ============================ ANALYTICS =====================================
CREATE TABLE IF NOT EXISTS analytics.job_match (
    match_id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id              UUID NOT NULL REFERENCES profile.user_profile(profile_id) ON DELETE CASCADE,
    job_id                  UUID NOT NULL REFERENCES core.job_posting(job_id) ON DELETE CASCADE,
    match_score             NUMERIC(4,3) NOT NULL,
    match_band              VARCHAR(10) NOT NULL,        -- strong | good | stretch | low
    match_eligible          BOOLEAN NOT NULL DEFAULT TRUE,
    ineligibility_reason    TEXT,
    matched_skills_count    INTEGER NOT NULL DEFAULT 0,
    required_skills_count   INTEGER NOT NULL DEFAULT 0,
    missing_skills_count    INTEGER NOT NULL DEFAULT 0,
    credential_warning      TEXT,
    computed_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    dataset_version         VARCHAR(40),
    engine_version          VARCHAR(20) NOT NULL,
    UNIQUE (profile_id, job_id)
);
CREATE INDEX IF NOT EXISTS ix_match_profile_score
    ON analytics.job_match(profile_id, match_score DESC);

CREATE TABLE IF NOT EXISTS analytics.job_match_skill (
    match_id     UUID REFERENCES analytics.job_match(match_id) ON DELETE CASCADE,
    skill_id     VARCHAR(40) REFERENCES reference.skill(skill_id),
    skill_name   TEXT NOT NULL,
    status       VARCHAR(10) NOT NULL,    -- matched | missing
    PRIMARY KEY (match_id, skill_name, status)
);

CREATE TABLE IF NOT EXISTS analytics.training_recommendation (
    recommendation_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id          UUID NOT NULL REFERENCES profile.user_profile(profile_id) ON DELETE CASCADE,
    job_id              UUID REFERENCES core.job_posting(job_id) ON DELETE SET NULL,
    resource_id         UUID NOT NULL REFERENCES core.training_resource(resource_id) ON DELETE CASCADE,
    skill_name          TEXT NOT NULL,
    reason              TEXT,
    score               NUMERIC(4,3),
    computed_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    engine_version      VARCHAR(20) NOT NULL
);

-- ============================ INTERACTION ===================================
CREATE TABLE IF NOT EXISTS interaction.chat_event (
    event_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id     UUID REFERENCES profile.user_profile(profile_id) ON DELETE CASCADE,
    session_id     VARCHAR(64),
    role           VARCHAR(20) NOT NULL,      -- user | assistant | system
    message_text   TEXT NOT NULL,
    metadata       JSONB,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_chat_profile_time
    ON interaction.chat_event(profile_id, created_at DESC);

CREATE TABLE IF NOT EXISTS interaction.recommendation_feedback (
    feedback_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id     UUID NOT NULL REFERENCES profile.user_profile(profile_id) ON DELETE CASCADE,
    job_id         UUID NOT NULL REFERENCES core.job_posting(job_id) ON DELETE CASCADE,
    match_id       UUID REFERENCES analytics.job_match(match_id) ON DELETE SET NULL,
    action         VARCHAR(20) NOT NULL,       -- saved | applied | not_relevant
    note           TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================ PIPELINE ======================================
CREATE TABLE IF NOT EXISTS pipeline.run (
    run_id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    started_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at      TIMESTAMPTZ,
    status           VARCHAR(20) NOT NULL DEFAULT 'running', -- running | success | failed | partial
    dataset_version  VARCHAR(40),
    metrics          JSONB
);

CREATE TABLE IF NOT EXISTS pipeline.step (
    step_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id      UUID NOT NULL REFERENCES pipeline.run(run_id) ON DELETE CASCADE,
    name        VARCHAR(80) NOT NULL,
    status      VARCHAR(20) NOT NULL,
    started_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    metrics     JSONB,
    error       TEXT
);

CREATE TABLE IF NOT EXISTS pipeline.dataset_state (
    pointer_key      VARCHAR(40) PRIMARY KEY,    -- 'current_jobs', 'current_training', etc.
    dataset_version  VARCHAR(40) NOT NULL,
    published_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    run_id           UUID REFERENCES pipeline.run(run_id)
);

-- Bridge mechanism audit. Each CSV dropped into PARTNER_CSV_DIR gets a row.
-- Filename convention: '<partner_name>_YYYY-MM-DD.csv' — the prefix
-- determines which approved_job_source the rows are written under.
-- file_sha256 dedupes accidental re-drops of the same file.
CREATE TABLE IF NOT EXISTS pipeline.partner_upload (
    upload_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    partner_name     VARCHAR(40) NOT NULL,
    filename         TEXT NOT NULL,
    file_sha256      CHAR(64) NOT NULL,
    uploaded_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processed_at     TIMESTAMPTZ,
    row_count        INTEGER,
    upserted_count   INTEGER,
    error            TEXT,
    UNIQUE (file_sha256)
);
CREATE INDEX IF NOT EXISTS ix_partner_upload_partner ON pipeline.partner_upload(partner_name);

-- ============================ KNOWLEDGE =====================================
-- AWIC reports, FedNor docs, sector studies. Stored as opaque documents in
-- PR 6A; indicator extraction is a future PR.
CREATE TABLE IF NOT EXISTS knowledge.document (
    document_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source          VARCHAR(80) NOT NULL,    -- 'awic' | 'fednor' | etc
    title           TEXT NOT NULL,
    url             TEXT NOT NULL,
    document_type   VARCHAR(40),             -- 'report' | 'dashboard' | 'pdf' | etc
    published_date  DATE,
    raw_text        TEXT,
    metadata        JSONB,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (source, url)
);
CREATE INDEX IF NOT EXISTS ix_doc_source ON knowledge.document(source);
CREATE INDEX IF NOT EXISTS ix_doc_published ON knowledge.document(published_date DESC NULLS LAST);

-- Indicators extracted from documents (e.g. "Algoma vacancy rate Q2 2025 = 4.1%").
-- Empty in PR 6A — extraction logic is a future PR. Schema is in place so
-- nothing breaks when we add the extractor.
CREATE TABLE IF NOT EXISTS knowledge.document_indicator (
    indicator_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id     UUID NOT NULL REFERENCES knowledge.document(document_id) ON DELETE CASCADE,
    name            TEXT NOT NULL,           -- e.g. 'vacancy_rate', 'population_55_plus'
    value_numeric   NUMERIC,
    value_text      TEXT,
    region_code     VARCHAR(10),
    period          VARCHAR(40),             -- 'Q2-2025', '2024-annual', etc
    extracted_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_indicator_doc ON knowledge.document_indicator(document_id);
CREATE INDEX IF NOT EXISTS ix_indicator_name ON knowledge.document_indicator(name);


-- ============================ SERVICE PROVIDERS =============================
-- Settlement / employment / training / counselling organizations. Populated
-- by the SSM LIP connector. Used by chat to surface "talk to [provider]"
-- fallbacks when no local match exists.
CREATE TABLE IF NOT EXISTS core.service_provider (
    provider_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            TEXT NOT NULL,
    name_normalized TEXT NOT NULL,
    service_type    TEXT,            -- settlement | employment | training | counselling | other
    description     TEXT,
    url             TEXT,
    phone           TEXT,
    email           TEXT,
    address         TEXT,
    source          VARCHAR(80),
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (name_normalized, source)
);
CREATE INDEX IF NOT EXISTS ix_service_type ON core.service_provider(service_type);


-- ============================ RAW ===========================================
-- Raw audit copy of every ingested job posting. Lets us re-extract skills
-- without re-scraping.
CREATE TABLE IF NOT EXISTS raw.job_posting (
    raw_id          BIGSERIAL PRIMARY KEY,
    source          VARCHAR(40) NOT NULL,
    source_job_id   VARCHAR(120) NOT NULL,
    payload         JSONB NOT NULL,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_raw_job_source ON raw.job_posting(source, source_job_id);

-- ============================ VIEWS =========================================
-- Convenience view: jobs that are "current" per the product freshness rule.
--
-- PRODUCT INVARIANT (do not change without a migration + engine_version bump):
--   A job is "current" if AND ONLY IF:
--     - is_active = TRUE             (not deactivated by the stale-sweep
--                                     or per-source presence sweep)
--     - last_seen_at >= today - 2d   (the source still reports it within
--                                     the daily-refresh + 1d grace window)
--     - closing_date IS NULL OR closing_date >= today
--
-- Eligibility (2026-06-08 architecture change): switched from
-- `posted_date >= today - 30d` to `last_seen_at >= today - 2d`.
--
-- Rationale: SCCC's WordPress source preserves the original `posted_date`
-- on long-lived listings, so a real, currently-listed job could age past
-- the 30-day window and silently drop out of matching even though the
-- source was still serving it. `last_seen_at` is refreshed every time
-- the per-source presence sweep confirms the job is still in the feed,
-- so eligibility now tracks "source still says this exists" rather than
-- "the date stamped on the posting at first sight."
--
-- posted_date stays in the row for DISPLAY ("posted 34 days ago") and
-- recency boost in the scoring engine. It is NOT used for eligibility.
--
-- The 2-day grace handles a daily refresh that occasionally slips a day:
-- one missed refresh keeps yesterday's jobs visible. Two missed refreshes
-- means something is broken upstream and a stale view is the correct
-- signal.
--
-- This invariant is the user-facing promise. It is NOT configurable via
-- env — changing it changes what the product means. If the rule needs
-- to change, write a migration that creates v_current_job_vN and bump
-- the dataset_version / engine_version published in the API envelope.
CREATE OR REPLACE VIEW core.v_current_job AS
SELECT *
  FROM core.job_posting
 WHERE is_active = TRUE
   AND last_seen_at >= NOW() - INTERVAL '2 days'
   AND (closing_date IS NULL OR closing_date >= CURRENT_DATE);

-- Data-status view used by /v1/admin/data-status.
CREATE OR REPLACE VIEW pipeline.v_data_status AS
SELECT
    (SELECT COUNT(*) FROM core.job_posting WHERE is_active)            AS active_jobs,
    (SELECT COUNT(*) FROM core.v_current_job)                          AS current_jobs,
    (SELECT COUNT(*) FROM core.job_posting)                            AS total_jobs,
    (SELECT COUNT(*) FROM extracted.job_skill)                         AS extracted_job_skills,
    (SELECT COUNT(*) FROM core.training_resource WHERE is_active)      AS active_training_resources,
    (SELECT COUNT(*) FROM extracted.training_skill)                    AS extracted_training_skills,
    (SELECT COUNT(*) FROM profile.user_profile WHERE deleted_at IS NULL) AS active_profiles,
    (SELECT COUNT(*) FROM core.employer)                               AS employers,
    (SELECT COUNT(*) FROM core.employer_designation)                   AS employer_designations,
    (SELECT COUNT(*) FROM core.service_provider WHERE is_active)       AS active_service_providers,
    (SELECT COUNT(*) FROM knowledge.document)                          AS knowledge_documents,
    (SELECT MAX(published_at) FROM pipeline.dataset_state)             AS last_publish_at,
    (SELECT dataset_version FROM pipeline.dataset_state
        WHERE pointer_key = 'current_jobs')                            AS current_dataset_version;

-- ============================ PR 10 MIGRATION ===============================
-- Idempotent ALTERs for databases initialized before PR 10 added the four
-- intake fields. New databases get them via the CREATE TABLE above; these
-- ALTERs are a no-op on those. CREATE TABLE IF NOT EXISTS won't update
-- column lists on an existing table, so we need this block.
ALTER TABLE profile.user_profile ADD COLUMN IF NOT EXISTS salary_expectation_text TEXT;
ALTER TABLE profile.user_profile ADD COLUMN IF NOT EXISTS shift_preference        TEXT;
ALTER TABLE profile.user_profile ADD COLUMN IF NOT EXISTS transportation_text     TEXT;
ALTER TABLE profile.user_profile ADD COLUMN IF NOT EXISTS availability_text       TEXT;

-- ============================ SPRINT 1 MIGRATION (resume) ====================
-- Idempotent ALTERs for the resume upload feature. The four columns hold
-- extracted text and structured facts only -- file bytes are never stored.
-- See docs/resume-design.md §3 for the storage policy and §4 for how
-- corrections layer on top of resume_facts_json without mutating it.
ALTER TABLE profile.user_profile ADD COLUMN IF NOT EXISTS resume_text       TEXT;
ALTER TABLE profile.user_profile ADD COLUMN IF NOT EXISTS resume_filename   TEXT;
ALTER TABLE profile.user_profile ADD COLUMN IF NOT EXISTS resume_parsed_at  TIMESTAMPTZ;
ALTER TABLE profile.user_profile ADD COLUMN IF NOT EXISTS resume_facts_json JSONB;

-- ============================ SPRINT 5 MIGRATION (required/preferred) ========
-- The JD-side LLM extractor now labels each skill 'required' or 'preferred'
-- so the matcher can weight them differently (Sprint 5 step 3). Legacy rows
-- keep skill_type=NULL; the engine treats NULL as 'required' for backwards
-- compatibility, preserving v1.1 scoring until a row is re-extracted.
-- CHECK constraint is intentionally permissive (NULL allowed) so older
-- inserts that pre-date this column do not fail.
ALTER TABLE extracted.job_skill
    ADD COLUMN IF NOT EXISTS skill_type VARCHAR(10);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conname = 'job_skill_skill_type_check'
    ) THEN
        ALTER TABLE extracted.job_skill
            ADD CONSTRAINT job_skill_skill_type_check
            CHECK (skill_type IS NULL OR skill_type IN ('required', 'preferred'));
    END IF;
END$$;

-- ============================ MATCHING v2 STEP 1 (OaSIS occupation layer) ====
-- Extends reference.occupation with French canonical title + lead statement
-- (OaSIS data is bilingual EN/FR). Adds a separate synonym table for the
-- multi-title-per-NOC lexicon (OaSIS example titles + SCT alternative
-- titles). See docs/matching-v2-design.md §3 for design intent.
ALTER TABLE reference.occupation ADD COLUMN IF NOT EXISTS title_fr            TEXT;
ALTER TABLE reference.occupation ADD COLUMN IF NOT EXISTS lead_statement_en   TEXT;
ALTER TABLE reference.occupation ADD COLUMN IF NOT EXISTS lead_statement_fr   TEXT;
ALTER TABLE reference.occupation ADD COLUMN IF NOT EXISTS oasis_version       VARCHAR(20);

CREATE TABLE IF NOT EXISTS reference.occupation_title_synonym (
    noc_code   VARCHAR(5) NOT NULL REFERENCES reference.occupation(noc_code) ON DELETE CASCADE,
    title      TEXT       NOT NULL,
    lang       VARCHAR(2) NOT NULL,
    source     VARCHAR(30) NOT NULL,
    PRIMARY KEY (noc_code, title, lang, source)
);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conname = 'occupation_title_synonym_lang_check'
    ) THEN
        ALTER TABLE reference.occupation_title_synonym
            ADD CONSTRAINT occupation_title_synonym_lang_check
            CHECK (lang IN ('en', 'fr'));
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conname = 'occupation_title_synonym_source_check'
    ) THEN
        ALTER TABLE reference.occupation_title_synonym
            ADD CONSTRAINT occupation_title_synonym_source_check
            CHECK (source IN ('oasis_example', 'sct_alternative'));
    END IF;
END$$;

CREATE INDEX IF NOT EXISTS ix_occ_title_synonym_title_trgm
    ON reference.occupation_title_synonym USING gin (title gin_trgm_ops);
CREATE INDEX IF NOT EXISTS ix_occ_title_synonym_noc
    ON reference.occupation_title_synonym (noc_code);

-- ============================ MATCHING v2 STEP 5 (semantic re-ranker) ========
-- Stores sentence-transformer embeddings for JD-side skill phrases. The
-- semantic re-ranker (engine._semantic_match_strength) computes cosine
-- similarity between user-skill embeddings (transient, computed at chat
-- time) and these pre-computed JD-side embeddings.
--
-- Design notes (see docs/matching-v2-design.md §4):
--   * model_version is part of the PK so we can ship a model upgrade
--     without dropping old vectors -- both can coexist during rollout.
--   * 384 dims matches all-MiniLM-L6-v2 (the default). If a future
--     model has a different dimensionality, add a new row with the new
--     model_version; the engine will pick the active version via config.
--   * No FK from (job_id, skill_name) to extracted.job_skill because
--     the skills table uses (job_id, skill_name) as its own PK and the
--     same composite is used here; an FK would force cascade ordering
--     that conflicts with the orchestrator's DELETE-then-INSERT extract
--     pattern. Embeddings are auto-pruned by the orchestrator when a
--     job's skills change.
-- The embedding table needs the `vector` type from pgvector. Wrap the
-- CREATE in the same EXCEPTION block so a missing extension logs a
-- NOTICE and skips the table -- the engine queries this table and
-- handles "table missing" by falling back to lexical-only matching.
DO $$
BEGIN
    CREATE TABLE IF NOT EXISTS extracted.job_skill_embedding (
        job_id          UUID NOT NULL,
        skill_name      TEXT NOT NULL,
        model_version   VARCHAR(60) NOT NULL,
        embedding       vector(384) NOT NULL,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (job_id, skill_name, model_version)
    );
EXCEPTION
    WHEN undefined_object THEN
        -- "vector" type doesn't exist (pgvector not installed). Skip
        -- the embedding table; --embed pipeline step will also no-op.
        RAISE NOTICE 'Skipping extracted.job_skill_embedding (pgvector not installed)';
END$$;

-- Optional ANN index. ivfflat needs ANALYZE to build well; at 35 SCCC
-- jobs × ~12 skills = ~420 rows it's overkill (linear scan is fine).
-- Left commented; uncomment if the corpus grows past ~10k vectors.
-- CREATE INDEX IF NOT EXISTS ix_job_skill_embedding_ivf
--     ON extracted.job_skill_embedding
--     USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
