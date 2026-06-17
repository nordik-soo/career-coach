"""Pydantic models shared across routes.

Only DTOs here — domain logic lives in the engine and chat modules.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------- Envelope
class Meta(BaseModel):
    request_id: str
    dataset_version: str | None = None
    engine_version: str | None = None
    data_as_of: datetime | None = None
    confidence: str | None = None
    warnings: list[str] = Field(default_factory=list)


class Envelope(BaseModel):
    data: Any
    meta: Meta


# --------------------------------------------------------------- Jobs
class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    job_id: UUID
    source: str
    title: str
    employer: str | None = None
    location: str | None = None
    posted_date: date | None = None
    closing_date: date | None = None
    salary_text: str | None = None
    employment_type: str | None = None
    remote_flag: bool | None = None
    url: str | None = None
    noc_code: str | None = None
    is_active: bool


class JobSkillOut(BaseModel):
    skill_name: str
    skill_id: str | None = None
    confidence: float | None = None
    importance_rank: int | None = None


class JobDetailOut(JobOut):
    description: str | None = None
    skills: list[JobSkillOut] = Field(default_factory=list)


# --------------------------------------------------------------- Training
class TrainingResourceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    resource_id: UUID
    provider: str
    title: str
    description: str | None = None
    url: str | None = None
    location: str | None = None
    delivery_mode: str | None = None
    duration_text: str | None = None
    duration_band: str | None = None
    resource_type: str | None = None
    cost_text: str | None = None


# --------------------------------------------------------------- Profile
# Note: ProfileIn was removed in PR 3. Profile creation only happens through
# the chat → consent flow (POST /v1/consent). There is no direct profile
# creation endpoint.


class ProfilePatch(BaseModel):
    preferred_location: str | None = None
    target_role_text: str | None = None
    education_text: str | None = None
    experience_text: str | None = None
    skills_text: str | None = None
    work_type_preference: str | None = None
    language_preferences: list[str] | None = None
    # PR 10 intake fields — editable post-consent so users can refine
    # what the chat extractor captured.
    salary_expectation_text: str | None = None
    shift_preference: str | None = None
    transportation_text: str | None = None
    availability_text: str | None = None
    consent_purposes: list[str] | None = None


class ProfileOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    profile_id: UUID
    session_id: str | None = None
    created_at: datetime
    updated_at: datetime
    preferred_location: str | None = None
    target_role_text: str | None = None
    education_text: str | None = None
    experience_text: str | None = None
    skills_text: str | None = None
    work_type_preference: str | None = None
    language_preferences: list[str] | None = None
    # PR 10 intake fields
    salary_expectation_text: str | None = None
    shift_preference: str | None = None
    transportation_text: str | None = None
    availability_text: str | None = None
    consent_version: str | None = None
    consent_purposes: list[str] = Field(default_factory=list)


class UserSkillOut(BaseModel):
    skill_name: str
    skill_id: str | None = None
    source: str | None = None
    confidence: float | None = None


# --------------------------------------------------------------- Chat
# Note: ChatMessageIn was removed in PR 3. The chat route defines its own
# ChatMessageBody locally — profile_id is never accepted from the client
# body. Identity comes from the bearer token or staged session_id only.


class RecommendedJob(BaseModel):
    job_id: UUID
    title: str
    employer: str | None = None
    url: str | None = None
    location: str | None = None
    match_band: str
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    credential_warning: str | None = None
    suggested_resources: list[TrainingResourceOut] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str
    profile_id: UUID | None = None
    session_id: str | None = None
    recommended_jobs: list[RecommendedJob] = Field(default_factory=list)
    next_skill_suggestion: str | None = None
    requires_consent: bool = False


# --------------------------------------------------------------- Matches
class MatchRequest(BaseModel):
    # profile_id intentionally absent — derived from bearer token.
    top: int = Field(default=10, ge=1, le=50)


class MatchOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    match_id: UUID
    job_id: UUID
    title: str
    employer: str | None = None
    url: str | None = None
    match_score: float
    match_band: str
    match_eligible: bool
    matched_skills_count: int
    required_skills_count: int
    missing_skills_count: int
    credential_warning: str | None = None


# --------------------------------------------------------------- Feedback
class FeedbackIn(BaseModel):
    # profile_id intentionally absent — derived from bearer token.
    job_id: UUID
    match_id: UUID | None = None
    action: str = Field(pattern=r"^(saved|applied|not_relevant)$")
    note: str | None = None


# --------------------------------------------------------------- Admin
class DataStatusOut(BaseModel):
    active_jobs: int
    current_jobs: int
    total_jobs: int
    extracted_job_skills: int
    active_training_resources: int
    extracted_training_skills: int
    active_profiles: int
    employers: int = 0
    employer_designations: int = 0
    active_service_providers: int = 0
    knowledge_documents: int = 0
    last_publish_at: datetime | None = None
    current_dataset_version: str | None = None
