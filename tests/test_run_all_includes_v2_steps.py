"""Regression test for the matching v2 step 5 review finding.

A daily `python run_pipeline.py --all` run must include the two
matching v2 steps -- resolve_noc and embed_job_skills -- or they
silently go stale and semantic / occupation-boost matching breaks.

This is a source-inspection test (no DB, no Anthropic). It pins:
  1. Both new step functions appear in run_all()'s step list
  2. embed_job_skills comes AFTER extract_job_skills (dependency)
  3. Both run BEFORE quality_check (so the published dataset reflects them)
"""
from __future__ import annotations

import inspect
import os

os.environ.setdefault("LLM_ENABLED", "false")

import pytest

from skillbridge.pipeline import orchestrator as orch

pytestmark = pytest.mark.nodb


def test_run_all_includes_resolve_noc():
    src = inspect.getsource(orch.run_all)
    assert "step_resolve_noc" in src, (
        "run_all() must call step_resolve_noc -- matching v2 step 2 "
        "review fix. Without this, core.job_posting.noc_code goes "
        "stale on every daily pipeline run and the occupation-match "
        "boost silently produces zero for new jobs."
    )


def test_run_all_includes_embed_job_skills():
    src = inspect.getsource(orch.run_all)
    assert "step_embed_job_skills" in src, (
        "run_all() must call step_embed_job_skills -- matching v2 "
        "step 5 review fix. Without this, extracted.job_skill_embedding "
        "goes stale on every daily run and semantic matching silently "
        "regresses to lexical-only for any new/re-extracted job skills."
    )


def test_embed_runs_after_extract():
    """embed_job_skills depends on extracted.job_skill rows -- it MUST
    come after extract_job_skills in the step list. If the order ever
    flips, embed will silently produce 0 embeddings on a fresh run."""
    src = inspect.getsource(orch.run_all)
    extract_pos = src.find("step_extract_job_skills")
    embed_pos = src.find("step_embed_job_skills")
    assert extract_pos != -1, "step_extract_job_skills missing from run_all"
    assert embed_pos != -1, "step_embed_job_skills missing from run_all"
    assert extract_pos < embed_pos, (
        "step_embed_job_skills must appear AFTER step_extract_job_skills "
        "in run_all(). Embeddings read from extracted.job_skill, so "
        "embedding before extracting produces nothing useful."
    )


def test_quality_check_runs_after_v2_steps():
    """quality_check writes the dataset_version row that downstream
    readers consume. If it ran before resolve_noc / embed_job_skills,
    a fresh dataset_version would advertise data that isn't ready yet."""
    src = inspect.getsource(orch.run_all)
    noc_pos = src.find("step_resolve_noc")
    embed_pos = src.find("step_embed_job_skills")
    quality_pos = src.find("step_quality_check")
    assert quality_pos != -1, "step_quality_check missing from run_all"
    assert noc_pos < quality_pos, "resolve_noc must run before quality_check"
    assert embed_pos < quality_pos, "embed_job_skills must run before quality_check"


def test_skip_set_can_disable_v2_steps():
    """Both new steps must respect the `skip` parameter so operators
    can run a v2-step-free pipeline if needed (e.g. debugging a
    skill-extraction issue without re-embedding)."""
    src = inspect.getsource(orch.run_all)
    # The step-list tuples must use the canonical names 'resolve_noc'
    # and 'embed_job_skills' (matching the --skip CLI input). If a
    # contributor renames them in the tuple list, --skip resolve_noc
    # silently stops working.
    assert '"resolve_noc"' in src
    assert '"embed_job_skills"' in src
