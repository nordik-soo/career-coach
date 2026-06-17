"""Unit tests for `_skills_overlap` and `_word_bounded_in`.

These cover the slice-4 follow-up bug found in live chat: a single-
letter user skill (the R language, lowercased to "r") was matching
every job-side skill that *contained* the letter 'r' (e.g. "french
language fluency", "rent collection", "certification"), producing
absurd "12/12 strong match" results for a data analyst.

The fix replaced plain `in` substring containment with word-bounded
containment via `_word_bounded_in`. These tests pin both the regression
case and the legitimate matches the original substring check was added
for, so a future refactor cannot quietly re-introduce either bug.

Pure-function tests; no DB.
"""
from __future__ import annotations

import os

os.environ.setdefault("LLM_ENABLED", "false")

import pytest

from skillbridge.match.engine import _skills_overlap, _word_bounded_in

pytestmark = pytest.mark.nodb


# ---------------------------------------------------------------------------
# _word_bounded_in -- low-level helper
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("needle, haystack, expected", [
    # Plain word boundaries
    ("welding", "welding & fabrication", True),
    ("welding", "underwater welding", True),
    ("welding", "welding", True),
    # Single-letter needles must require standalone words
    ("r", "french language fluency", False),
    ("r", "rent collection and arrears management", False),
    ("r", "r programming", True),                 # standalone "r"
    ("r", "(r) library", True),                   # punct surrounds
    # Short skill names that ARE standalone in the haystack
    ("aws", "aws sagemaker", True),
    ("sql", "mysql database", False),             # sql inside mysql, not a word
    ("sql", "sql server", True),
    ("etl", "etl pipeline", True),
    ("etl", "completion of tasks", False),        # etl not a standalone word
    # Punctuation boundaries: hyphens and slashes count as non-word
    ("class g", "class g driver's license", True),
    ("class-g", "class-g license", True),
    # Empty inputs
    ("", "anything", False),
    ("something", "", False),
])
def test_word_bounded_in(needle, haystack, expected):
    assert _word_bounded_in(needle, haystack) is expected


def test_word_bounded_in_finds_match_when_first_occurrence_is_inside_word():
    """If the first occurrence is inside another word, the search must
    keep going and find any later standalone occurrence."""
    # 'aws' appears inside 'jigsaws' first, then as a standalone word
    assert _word_bounded_in("aws", "jigsaws plus aws core") is True


# ---------------------------------------------------------------------------
# _skills_overlap -- regression cases from the false-positive bug
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("user_skill, job_skill, expected", [
    # The exact bug that surfaced in live chat with Nazmul's resume.
    # "R" lowercased to "r" was matching every job skill containing letter r.
    ("r", "french language fluency", False),
    ("r", "rent collection and arrears management", False),
    ("r", "310t technician certification", False),
    ("r", "marketing principles", False),
    ("r", "early childhood education", False),
    # Other short / risky user skills
    ("aws", "lease enforcement and compliance", False),
    ("etl", "social-emotional learning facilitation", False),
    ("sql", "rent collection and arrears management", False),
])
def test_short_user_skill_no_longer_falsely_matches_unrelated_job(
    user_skill, job_skill, expected,
):
    """Regression: short user skills must NOT match unrelated job phrases
    just because the user-skill string happens to appear inside one of
    the job phrase's words."""
    assert _skills_overlap(user_skill, job_skill) is expected


@pytest.mark.parametrize("user_skill, job_skill", [
    # The legitimate matches the original substring check was added for --
    # these MUST still work after the word-boundary fix.
    ("welding", "welding & fabrication"),
    ("welding & fabrication", "welding"),
    ("class g", "class g driver's license"),
    ("class g driver's license", "class g"),
    ("aws", "aws sagemaker"),
    ("aws sagemaker", "aws"),
    ("python", "python programming"),
    ("python programming", "python"),
])
def test_legitimate_substring_matches_still_work(user_skill, job_skill):
    """Word-boundary fix must not break the original 'welding ⊂ welding
    & fabrication' use case the substring check was written for."""
    assert _skills_overlap(user_skill, job_skill) is True


def test_data_analyst_skill_set_produces_no_overlap_with_unrelated_jds():
    """End-to-end regression: the exact Nazmul-resume skill set must
    produce zero overlap matches against the truck-and-coach JD's
    top-12 skill phrases. Pre-fix, this returned 12 of 12 matches via
    the letter-'r' substring leak."""
    nazmul_skills = [
        "python", "r", "sql", "pytorch", "opencv", "scikit-learn",
        "pandas", "matplotlib", "langchain", "numpy", "power bi", "dax",
        "docker", "github", "mlflow", "aws", "aws sagemaker", "aws lambda",
        "aws s3", "aws ec2", "hugging face", "flask", "fastapi", "next.js",
        "faiss", "mysql", "nlp", "computer vision", "data science",
        "llm application", "convolutional neural networks", "xgboost",
        "machine learning model evaluation", "etl", "power query",
    ]
    truck_coach_skills = [
        "truck service and maintenance", "emergency repair",
        "motor vehicle inspection", "emissions testing preparation",
        "wheel end inspection", "welding", "parts fabrication",
        "310t certificate of qualification", "class g driver's license",
        "mto contract supervision", "driver hour tracking",
        "on-call availability",
    ]
    matches = [
        (u, j)
        for j in truck_coach_skills
        for u in nazmul_skills
        if _skills_overlap(u, j)
    ]
    assert matches == [], (
        f"data analyst should NOT match truck/coach skills; "
        f"got {len(matches)} false-positive overlap pairs: {matches[:5]}"
    )
