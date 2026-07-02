"""Step 6a: First-run matching engine runner for the eval corpus.

Executes the actual match engine against each corpus case with the
frozen posting bank and pinned embedding fixtures. No DB. No LLM.
No network.

Design (locked 2026-07-02, per Step 6 discussion):
  - Split from Step 6b+ resolution work: this file builds the runner
    and produces first-run disagreements. Disagreement RESOLUTION
    happens per case in separate follow-up commits.
  - Iterative monkeypatch discovery: patch what fails, don't
    exhaustively audit upfront. Runtime reveals the real dependency
    surface.
  - No skips, no xfails. Failures ARE the honest first-run report
    (per schema Design Goal 1). Pytest output is enough; no separate
    markdown report file.
  - Case c_market_data_unavailable_stub explicitly cannot produce its
    expected outcome with a frozen posting bank (per Step 5 note); it
    is expected to fail loudly in 6a and get the first Step 6b
    decision.

Expected first-run outcome (unknown until this runs):
  - 7 transcribed cases (mirror passes_today fixtures): likely pass
  - 21 authored cases: mixed; multiple disagreements likely.
  - Each disagreement becomes a Step 6b+ per-case investigation.

Loader reuses tests/test_matching_eval.load_corpus so the corpus is
a single source of truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

from tests.test_matching_eval import (
    Case,
    Corpus,
    EmbeddingPair,
    Posting,
    load_corpus,
)

pytestmark = pytest.mark.nodb


# ---------------------------------------------------------------------------
# Corpus loaded once for the parametrize
# ---------------------------------------------------------------------------
_CORPUS: Corpus = load_corpus()


# ---------------------------------------------------------------------------
# Frozen anchor: posted_date resolves against corpus.frozen_today
# ---------------------------------------------------------------------------
def _frozen_today() -> date:
    parts = _CORPUS.frozen_today.split("-")
    return date(int(parts[0]), int(parts[1]), int(parts[2]))


# ---------------------------------------------------------------------------
# Posting -> engine job row projection
# ---------------------------------------------------------------------------
def _posting_to_job_row(p: Posting) -> dict[str, Any]:
    """Project a Posting into the row shape core.v_current_job returns.

    Iteratively expanded as engine access patterns surface at runtime.
    """
    today = _frozen_today()
    posted_date = today - timedelta(days=p.posted_days_ago)
    return {
        "job_id": p.posting_id,
        "title": p.title,
        "employer": p.employer,
        "noc_code": p.noc_code,
        "location": p.location,
        "region_code": p.region_code,
        "employment_type": p.employment_type,
        "posted_date": posted_date,
        "url": None,
        "description": p.description_snippet or "",
        "shift_pattern": None,
    }


def _posting_to_skill_rows(p: Posting) -> list[dict[str, Any]]:
    """Project a Posting's skills into the rows _fetch_job_skills returns."""
    rows: list[dict[str, Any]] = []
    for i, s in enumerate(p.skills):
        rows.append({
            "skill_id": None,          # synthetic; not in reference.skill
            "skill_name": s.name,
            "importance_rank": i + 1,
            "source": s.requirement,   # required | preferred
        })
    return rows


# ---------------------------------------------------------------------------
# Semantic fixture lookup
# ---------------------------------------------------------------------------
def _cosine_from_fixtures(a: str, b: str, fixtures: tuple[EmbeddingPair, ...]) -> float:
    """Look up pinned cosine for a pair. Symmetric. Unlisted → 0.0."""
    a_lower = (a or "").strip().lower()
    b_lower = (b or "").strip().lower()
    for pair in fixtures:
        pa = pair.a.strip().lower()
        pb = pair.b.strip().lower()
        if (pa == a_lower and pb == b_lower) or (pa == b_lower and pb == a_lower):
            return pair.cosine
    return 0.0


# ---------------------------------------------------------------------------
# Monkeypatch fixture: silences every DB touchpoint we've discovered so far
# ---------------------------------------------------------------------------
@pytest.fixture
def patched_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patches every known DB / model touchpoint so the engine runs
    entirely off corpus content. Iteratively expanded during 6a."""
    postings = _CORPUS.posting_bank

    def _fake_fetch_eligible_jobs() -> list[dict[str, Any]]:
        return [_posting_to_job_row(p) for p in postings]

    _skills_by_job = {p.posting_id: _posting_to_skill_rows(p) for p in postings}

    def _fake_fetch_job_skills(job_id: str) -> list[dict[str, Any]]:
        return list(_skills_by_job.get(job_id, ()))

    def _fake_fetch_job_skill_embeddings(job_id: str):
        # No embeddings in the corpus. Semantic path is handled
        # separately by patching _semantic_match_strength.
        return []

    def _fake_regulated(noc_code: str | None, target_role: str | None) -> dict | None:
        # No regulated-occupation info in the corpus.
        return None

    def _fake_maybe_embed_user_skill_rows(user_rows: list) -> None:
        # Return None -> engine falls back to per-pair semantic scoring
        # which we intercept with _fake_semantic_match_strength below.
        return None

    def _fake_resolve_title_to_noc(title: str | None, **kwargs) -> str | None:
        # No taxonomy lookup available; engine handles None gracefully.
        # Step 6b may introduce a per-corpus title→NOC map if needed.
        return None

    def _fake_semantic_match_strength(
        job_skill_embedding,
        user_embeddings_matrix,
        *,
        tiebreak_keys: list[str] | None = None,
    ):
        # Without a real embedding path, no fixture-driven lookup is
        # possible via this signature (the engine passes embeddings,
        # not raw phrases). Step 6b will decide the right hook. For now:
        # zero strength, no winning index. Cases that depend on
        # semantic bridges will disagree loudly.
        return (0.0, None)

    engine_mod = "skillbridge.match.engine"
    monkeypatch.setattr(f"{engine_mod}._fetch_eligible_jobs", _fake_fetch_eligible_jobs)
    monkeypatch.setattr(f"{engine_mod}._fetch_job_skills", _fake_fetch_job_skills)
    monkeypatch.setattr(
        f"{engine_mod}._fetch_job_skill_embeddings", _fake_fetch_job_skill_embeddings,
    )
    monkeypatch.setattr(f"{engine_mod}._regulated", _fake_regulated)
    monkeypatch.setattr(
        f"{engine_mod}._maybe_embed_user_skill_rows", _fake_maybe_embed_user_skill_rows,
    )
    monkeypatch.setattr(
        f"{engine_mod}._semantic_match_strength", _fake_semantic_match_strength,
    )
    # resolve_title_to_noc is imported INTO the engine module; patch the
    # engine's local reference, not skillbridge.match.occupation.
    monkeypatch.setattr(
        f"{engine_mod}.resolve_title_to_noc", _fake_resolve_title_to_noc,
    )

    # Belt-and-braces: any un-mocked path that tries to open a DB
    # connection produces a clear error instead of a 30-second pool
    # timeout. Add a specific fake_* above and re-run when this fires.
    from contextlib import contextmanager

    @contextmanager
    def _trap_sync_cursor():
        raise RuntimeError(
            "test_matching_eval_runner: sync_cursor called from an "
            "un-monkeypatched engine path. Add a targeted fake_* fixture "
            "for the caller and re-run."
        )
        yield  # pragma: no cover

    monkeypatch.setattr("skillbridge.db.sync_cursor", _trap_sync_cursor)


# ---------------------------------------------------------------------------
# StagedProfile construction from Case
# ---------------------------------------------------------------------------
def _build_staged_profile(case: Case):
    """Construct a StagedProfile that the engine can score against."""
    from skillbridge.session.staging import StagedProfile, StagedSkill

    staged = StagedProfile.new(session_id=f"eval_{case.case_id}")
    staged.target_role_text = case.profile.target_role or None
    staged.experience_text = case.profile.experience_text
    staged.education_text = case.profile.education_text
    staged.work_type_preference = case.profile.work_type_preference

    for phrase in case.profile.skill_phrases:
        if not phrase:
            continue
        staged.skills.append(
            StagedSkill(
                skill_name=phrase,
                raw_phrase=phrase,
                confidence=0.9,
                source="chat",
            )
        )
    return staged


# ---------------------------------------------------------------------------
# Runner: engine + diagnosis for one case
# ---------------------------------------------------------------------------
@dataclass
class RunOutcome:
    """What the engine actually produced for one case."""
    matches: list = field(default_factory=list)
    diagnosis_outcome: str | None = None
    engine_error: str | None = None


def _run_engine_for_case(case: Case) -> RunOutcome:
    from skillbridge.chat.inventory_diagnosis import diagnose
    from skillbridge.match.engine import compute_matches_in_memory

    staged = _build_staged_profile(case)

    try:
        matches = compute_matches_in_memory(staged, top=20)
    except Exception as exc:  # noqa: BLE001
        return RunOutcome(engine_error=f"engine raised: {type(exc).__name__}: {exc}")

    # Diagnosis inputs (heuristics for Step 6a; Step 6b may refine):
    enough_to_match = bool(case.profile.skill_phrases) and bool(
        (case.profile.target_role or "").strip()
    )
    usable_evidence_present = bool(case.profile.skill_phrases) or bool(
        case.profile.experience_text
    )
    engine_completed = True
    snapshot_usable = True  # 6a always supplies the frozen bank
    target_posting_count: int | None = sum(
        1 for m in matches if m.match_eligible
    )

    try:
        dx = diagnose(
            enough_to_match=enough_to_match,
            usable_evidence_present=usable_evidence_present,
            engine_completed=engine_completed,
            snapshot_usable=snapshot_usable,
            direct_match_results=matches,
            skill_adjacent_results=[],
            target_posting_count=target_posting_count,
        )
        outcome = dx.outcome
    except Exception as exc:  # noqa: BLE001
        return RunOutcome(
            matches=matches,
            engine_error=f"diagnose raised: {type(exc).__name__}: {exc}",
        )

    return RunOutcome(matches=matches, diagnosis_outcome=outcome)


# ---------------------------------------------------------------------------
# The one parametrized test: run every case, compare diagnosis outcome
# ---------------------------------------------------------------------------
_CASE_IDS = [c.case_id for c in _CORPUS.cases]


@pytest.mark.parametrize("case_id", _CASE_IDS)
def test_case_diagnosis_matches_expected(
    case_id: str, patched_engine: None,
) -> None:
    """First-run engine check: does the engine produce the expected
    diagnosis outcome for this case? All 28 cases run parametrized.

    Failures are the honest first-run disagreement report. Each
    failure gets its own Step 6b+ investigation.

    Scope of THIS test: diagnosis outcome only. Per-job band /
    cap_reasons / matched_required assertions are Step 6c+ once the
    diagnosis-level runner stabilizes."""
    case = next(c for c in _CORPUS.cases if c.case_id == case_id)
    outcome = _run_engine_for_case(case)
    if outcome.engine_error:
        pytest.fail(
            f"[{case_id}] engine execution error: {outcome.engine_error}"
        )
    expected = case.expect.diagnosis
    actual = outcome.diagnosis_outcome
    assert actual == expected, (
        f"[{case_id}] diagnosis disagreement:\n"
        f"  expected: {expected}\n"
        f"  actual:   {actual}\n"
        f"  matches surfaced: {len(outcome.matches)}"
    )
