"""AR-9.feat.coach-tiers CP1 step 7 — adjacency enrichment via the
shared `build_skill_alignment` helper.

Pins:
  - `enrich_accepted_adjacency_jobs` operates ONLY on jobs that came
    out of `accept_candidates` (caller's contract; the function does
    not re-run the gate);
  - `exclude_job_ids` removes Strong/Stretch jobs before construction;
  - URL is validated through `url_policy.validate`; rejected URLs
    become None, never a raw string;
  - `why_adjacent` reuses the closed {"same_noc_minor_group",
    "skill_evidence"} set and matches handler.py's derivation;
  - `important_gaps` carries required NON-credential job_skills not in
    `skill_alignment`. Credentials never appear here — they're in
    `credential_warning_text` as an informational notice;
  - `skill_alignment` comes from the shared helper (no duplicate
    construction path);
  - sourceable job facts only (posted_date, posted_days_ago, location,
    employment_type, salary_text — per C3);
  - `transferable_pairs` is a per-alignment projection.

These tests do NOT touch the existing adjacency pipeline (`retrieve_candidates`
/ `accept_candidates`) — the enrichment function takes whatever the
caller has already accepted.
"""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from skillbridge.chat import tiered_evidence as te
from skillbridge.chat.tiered_evidence import (
    AdjacentJob,
    JobFacts,
    TransferablePair,
    _important_non_credential_gaps,
    _validate_url,
    _why_adjacent,
    enrich_accepted_adjacency_jobs,
)
from skillbridge.chat.url_policy import Validated
from skillbridge.match.alignment import UserSkillRow

pytestmark = pytest.mark.nodb


@pytest.fixture(autouse=True)
def _no_db_regulated_lookup(monkeypatch):
    """The enrichment function calls `_regulated` (DB) inside
    `_credential_warning_text`. Without a real DB the call eventually
    raises and we catch it, but each retry takes seconds. Short-circuit
    to None for nodb tests so the suite stays fast. Credential-warning
    semantics are covered by an integration test elsewhere."""
    monkeypatch.setattr(te, "_regulated", lambda noc, target: None)


# =========================================================================
# Fixtures
# =========================================================================
def _row(text, *, sid=None):
    from skillbridge.match.aliases import canonicalize_skill
    stripped = text.strip()
    return UserSkillRow(
        skill_id=sid,
        text=stripped,
        name=stripped.lower(),
        canon=canonicalize_skill(stripped) or "",
    )


def _sets(rows):
    ids = {r.skill_id for r in rows if r.skill_id}
    names = {r.name for r in rows}
    canons = {r.canon for r in rows if r.canon}
    return ids, names, canons


def _job(
    *,
    job_id="adj-1",
    title="Payroll Administrator",
    employer="North Star",
    url="https://example.com/job/1",
    location="Sault Ste. Marie, ON",
    noc_code="12102",
    employment_type="full-time",
    salary_text="$22-24/hr",
    posted_date=None,
    skills=None,
):
    return {
        "job_id": job_id,
        "title": title,
        "employer": employer,
        "url": url,
        "location": location,
        "noc_code": noc_code,
        "employment_type": employment_type,
        "salary_text": salary_text,
        "posted_date": posted_date,
        "skills": skills or [],
    }


def _js(name, *, skill_type=None):
    out = {"skill_id": None, "skill_name": name}
    if skill_type is not None:
        out["skill_type"] = skill_type
    return out


# =========================================================================
# _why_adjacent — closed vocabulary reused from handler.py
# =========================================================================
def test_why_adjacent_same_minor_group():
    assert _why_adjacent("12101", "12102") == "same_noc_minor_group"


def test_why_adjacent_skill_evidence_when_minors_differ():
    assert _why_adjacent("12101", "73402") == "skill_evidence"


def test_why_adjacent_skill_evidence_when_target_noc_absent():
    assert _why_adjacent(None, "12102") == "skill_evidence"
    assert _why_adjacent("", "12102") == "skill_evidence"


def test_why_adjacent_skill_evidence_when_job_noc_absent():
    assert _why_adjacent("12101", None) == "skill_evidence"


def test_why_adjacent_value_is_in_closed_set():
    """Closed-vocab contract: only the two known tokens may be returned."""
    valid = {"same_noc_minor_group", "skill_evidence"}
    assert _why_adjacent("12101", "12102") in valid
    assert _why_adjacent("12101", "73402") in valid


# =========================================================================
# _validate_url — accepts only Validated URLs from url_policy
# =========================================================================
def test_validate_url_accepts_well_formed_https():
    out = _validate_url("https://example.com/job/123")
    assert isinstance(out, Validated)
    assert out.raw_token == "https://example.com/job/123"


def test_validate_url_rejects_non_https():
    assert _validate_url("http://example.com/job/123") is None
    assert _validate_url("ftp://example.com/x") is None


def test_validate_url_rejects_malformed():
    assert _validate_url("not a url") is None
    assert _validate_url("https://") is None


def test_validate_url_rejects_none_and_non_str():
    assert _validate_url(None) is None
    assert _validate_url(123) is None
    assert _validate_url("") is None


# =========================================================================
# _important_non_credential_gaps
# =========================================================================
def test_gaps_exclude_credentials():
    """A required credential job-skill must NEVER appear in
    important_gaps. accept_candidates already ensured the user has
    every required credential, but defensive filtering catches any
    drift in upstream contracts."""
    job_skills = [
        _js("Class G driver's license", skill_type="required"),
        _js("forklift operation", skill_type="required"),
    ]
    # Neither requirement was matched → both should appear naively,
    # but the credential one must be filtered out.
    gaps = _important_non_credential_gaps(job_skills, set())
    assert "Class G driver's license" not in gaps
    assert "forklift operation" in gaps


def test_gaps_exclude_preferred_bucket():
    """Preferred skills are not 'important_gaps' for the prompt — the
    prompt's Sideways paragraph addresses required overlap, not nice-to-haves."""
    job_skills = [
        _js("payroll processing", skill_type="preferred"),
        _js("invoice processing", skill_type="required"),
    ]
    gaps = _important_non_credential_gaps(job_skills, set())
    assert "payroll processing" not in gaps
    assert "invoice processing" in gaps


def test_gaps_exclude_matched_requirements():
    """If a requirement was satisfied (its name appears in
    `matched_requirement_names`), it is NOT a gap."""
    job_skills = [
        _js("QuickBooks", skill_type="required"),
        _js("AP processing", skill_type="required"),
    ]
    gaps = _important_non_credential_gaps(
        job_skills, matched_requirement_names={"QuickBooks"}
    )
    assert gaps == ("AP processing",)


def test_gaps_preserve_input_order():
    job_skills = [
        _js("z-skill", skill_type="required"),
        _js("a-skill", skill_type="required"),
        _js("m-skill", skill_type="required"),
    ]
    gaps = _important_non_credential_gaps(job_skills, set())
    assert gaps == ("z-skill", "a-skill", "m-skill")


def test_gaps_skip_malformed_entries():
    job_skills = [
        "not a dict",                           # not a dict
        {"skill_type": "required"},             # no skill_name
        {"skill_name": "", "skill_type": "required"},   # empty
        {"skill_name": "real", "skill_type": "required"},
    ]
    gaps = _important_non_credential_gaps(job_skills, set())
    assert gaps == ("real",)


# =========================================================================
# enrich_accepted_adjacency_jobs — projection contract
# =========================================================================
def _baseline_inputs():
    """One user, one accepted adjacent job. Used as a shared fixture."""
    rows = [_row("QuickBooks"), _row("payroll")]
    ids, names, canons = _sets(rows)
    job = _job(
        skills=[
            _js("QuickBooks", skill_type="required"),
            _js("invoice processing", skill_type="required"),
            _js("payroll processing", skill_type="preferred"),
        ],
    )
    return rows, ids, names, canons, job


def test_enrich_returns_one_adjacent_job_per_accepted_input():
    rows, ids, names, canons, job = _baseline_inputs()
    out = enrich_accepted_adjacency_jobs(
        [job], rows, ids, names, canons,
        target_noc="12102",
    )
    assert len(out) == 1
    assert isinstance(out[0], AdjacentJob)


def test_enrich_excludes_job_ids_in_exclude_set():
    """Strong / Worth-a-try job IDs must be filtered before projection."""
    rows, ids, names, canons, job = _baseline_inputs()
    out = enrich_accepted_adjacency_jobs(
        [job], rows, ids, names, canons,
        target_noc="12102",
        exclude_job_ids={job["job_id"]},
    )
    assert out == []


def test_enrich_preserves_input_order():
    rows, ids, names, canons, _ = _baseline_inputs()
    j1 = _job(job_id="adj-1", title="A")
    j2 = _job(job_id="adj-2", title="B")
    j3 = _job(job_id="adj-3", title="C")
    out = enrich_accepted_adjacency_jobs(
        [j1, j2, j3], rows, ids, names, canons,
        target_noc=None,
    )
    assert [a.job_id for a in out] == ["adj-1", "adj-2", "adj-3"]


def test_enrich_uses_shared_alignment_helper():
    """alignment is populated from `build_skill_alignment` — i.e. the
    helper attributes 'QuickBooks' (user) to the literal 'QuickBooks'
    job-requirement record. Other matches (e.g. payroll → payroll
    processing via word-bounded substring) are also legitimate and
    not asserted-against here; this test only pins that the SHARED
    helper produced the QuickBooks attribution."""
    rows, ids, names, canons, job = _baseline_inputs()
    out = enrich_accepted_adjacency_jobs(
        [job], rows, ids, names, canons,
        target_noc=None,
    )
    a = out[0]
    matched = {al.job_requirement: al.user_skill for al in a.skill_alignment}
    assert matched.get("QuickBooks") == "QuickBooks"


def test_enrich_transferable_pairs_mirror_alignment():
    rows, ids, names, canons, job = _baseline_inputs()
    out = enrich_accepted_adjacency_jobs(
        [job], rows, ids, names, canons,
        target_noc=None,
    )
    a = out[0]
    assert len(a.transferable_pairs) == len(a.skill_alignment)
    p = a.transferable_pairs[0]
    assert isinstance(p, TransferablePair)
    assert p.user_skill == "QuickBooks"
    assert p.applies_to == "QuickBooks"
    assert p.stage == "exact"


def test_enrich_important_gaps_excludes_matched_and_credentials():
    rows, ids, names, canons, _ = _baseline_inputs()
    job = _job(
        skills=[
            _js("QuickBooks", skill_type="required"),                     # matched
            _js("invoice processing", skill_type="required"),             # gap
            _js("Class G driver's license", skill_type="required"),       # credential
            _js("payroll processing", skill_type="preferred"),            # preferred (excluded)
        ],
    )
    out = enrich_accepted_adjacency_jobs(
        [job], rows, ids, names, canons,
        target_noc=None,
    )
    a = out[0]
    assert a.important_gaps == ("invoice processing",)


def test_enrich_validated_url_becomes_sanitized():
    """AR-9.feat.coach-tiers CP1 step 7 (Fix 1): AdjacentJob.url is
    Validated, NOT SanitizedURL — the view layer (step 9) projects
    Validated → SanitizedURL so tiered_evidence.py stays free of any
    url_views.py import."""
    rows, ids, names, canons, job = _baseline_inputs()
    job["url"] = "https://www.sccc.example/job/1"
    out = enrich_accepted_adjacency_jobs(
        [job], rows, ids, names, canons,
        target_noc=None,
    )
    a = out[0]
    assert isinstance(a.url, Validated)
    assert a.url.raw_token == "https://www.sccc.example/job/1"


def test_enrich_unvalidated_url_becomes_none():
    rows, ids, names, canons, job = _baseline_inputs()
    job["url"] = "ftp://wrong-scheme.example/x"
    out = enrich_accepted_adjacency_jobs(
        [job], rows, ids, names, canons,
        target_noc=None,
    )
    assert out[0].url is None


def test_enrich_job_facts_carry_raw_source_fields():
    rows, ids, names, canons, _ = _baseline_inputs()
    posted = date.today() - timedelta(days=3)
    job = _job(
        posted_date=posted,
        location="Sault Ste. Marie, ON",
        employment_type="full-time",
        salary_text="$22-24/hr",
        skills=[_js("QuickBooks", skill_type="required")],
    )
    out = enrich_accepted_adjacency_jobs(
        [job], rows, ids, names, canons,
        target_noc=None,
    )
    facts = out[0].job_facts
    assert isinstance(facts, JobFacts)
    assert facts.posted_date == posted
    assert facts.posted_days_ago == 3
    assert facts.location == "Sault Ste. Marie, ON"
    assert facts.employment_type == "full-time"
    assert facts.salary_text == "$22-24/hr"


def test_enrich_job_facts_none_when_source_absent():
    rows, ids, names, canons, _ = _baseline_inputs()
    job = _job(
        posted_date=None,
        location=None,
        employment_type=None,
        salary_text=None,
        skills=[_js("QuickBooks", skill_type="required")],
    )
    out = enrich_accepted_adjacency_jobs(
        [job], rows, ids, names, canons,
        target_noc=None,
    )
    facts = out[0].job_facts
    assert facts.posted_date is None
    assert facts.posted_days_ago is None
    assert facts.location is None
    assert facts.employment_type is None
    assert facts.salary_text is None


def test_enrich_why_adjacent_same_minor_group():
    rows, ids, names, canons, _ = _baseline_inputs()
    job = _job(noc_code="12102", skills=[_js("QuickBooks", skill_type="required")])
    out = enrich_accepted_adjacency_jobs(
        [job], rows, ids, names, canons,
        target_noc="12101",
    )
    assert out[0].why_adjacent == "same_noc_minor_group"


def test_enrich_why_adjacent_skill_evidence():
    rows, ids, names, canons, _ = _baseline_inputs()
    job = _job(noc_code="73402", skills=[_js("QuickBooks", skill_type="required")])
    out = enrich_accepted_adjacency_jobs(
        [job], rows, ids, names, canons,
        target_noc="12101",
    )
    assert out[0].why_adjacent == "skill_evidence"


# =========================================================================
# AdjacentJob frozen / Literal smoke
# =========================================================================
def test_adjacent_job_is_frozen():
    rows, ids, names, canons, job = _baseline_inputs()
    out = enrich_accepted_adjacency_jobs(
        [job], rows, ids, names, canons,
        target_noc=None,
    )
    a = out[0]
    with pytest.raises((AttributeError, Exception)):
        a.title = "different"  # type: ignore


# =========================================================================
# Fix 1 (post-step-7 review) — dependency direction regression
# tiered_evidence MUST NOT import url_views; AdjacentJob.url is
# Validated, the view layer converts to SanitizedURL.
# =========================================================================
def test_tiered_evidence_module_does_not_import_url_views():
    """Regression for Fix 1. Storing SanitizedURL on AdjacentJob would
    couple tiered_evidence.py to url_views.py. Step 9 will make
    url_views.py consume tiered evidence, which would be a circular
    import. Storing Validated keeps the direction clean.

    This test guards the dependency by inspecting the module's actual
    imports — if a future refactor accidentally pulls url_views back
    in, this fails immediately."""
    import skillbridge.chat.tiered_evidence as te_mod
    import sys
    # After importing tiered_evidence, url_views should NOT have been
    # imported as a transitive consequence of importing tiered_evidence.
    # (It may be imported by OTHER modules in the suite — so we check
    # tiered_evidence's own module.__dict__ doesn't reference it.)
    referenced = {
        name for name, val in te_mod.__dict__.items()
        if hasattr(val, "__module__") and val.__module__ == "skillbridge.chat.url_views"
    }
    assert referenced == set(), (
        f"tiered_evidence pulled in url_views members: {referenced}. "
        "Storing SanitizedURL on AdjacentJob would reintroduce the "
        "circular-import risk Fix 1 closed."
    )
    # And no direct module-attribute reference either.
    assert "url_views" not in te_mod.__dict__
    # Sanity: url_policy IS allowed (it owns Validated).
    assert "validate" in te_mod.__dict__ or hasattr(te_mod, "_validate_url")
    _ = sys  # silence unused-import warning if linter checks


def test_adjacent_job_url_field_is_validated_not_sanitized():
    """The dataclass field is Validated, not SanitizedURL. Verifies
    via the actual class annotation."""
    annotations = AdjacentJob.__annotations__
    # The annotation is the string form because of `from __future__
    # import annotations`. Inspect the string.
    url_ann = annotations.get("url")
    assert url_ann is not None
    assert "Validated" in str(url_ann)
    assert "SanitizedURL" not in str(url_ann)


# =========================================================================
# Fix 2 (post-step-7 review) — credential warning must use ONLY
# the adjacent job's own NOC. Never fall back to the user's target_role.
# =========================================================================
def test_credential_warning_returns_none_when_job_noc_absent(monkeypatch):
    """Regression for Fix 2. Even when `_regulated` would return a
    warning row if queried with the user's target role, the adjacency
    enrichment path keeps `credential_warning_text` None for an
    adjacent job that has no noc_code. The misattribution that this
    test guards against is: 'the user wants to be a nurse → an
    adjacent retail job gets a nursing licensing warning.'"""
    # Smart mock: returns a warning row ONLY for a specific
    # noc_code OR a non-None target role. This would have fired
    # under the old code path; under Fix 2 the enrichment never
    # passes target_role through.
    def smart_regulated(noc, target):
        if noc == "31301":   # nursing NOC
            return {"regulator_name": "CNO", "licensing_note": "RN licence required."}
        if target:           # the OLD fallback we removed
            return {"regulator_name": "WRONG", "licensing_note": "wrong"}
        return None
    monkeypatch.setattr(te, "_regulated", smart_regulated)

    rows, ids, names, canons, _ = _baseline_inputs()
    # Adjacent job has NO noc_code. User's target role is the
    # regulated nursing role.
    job = _job(noc_code=None, skills=[_js("QuickBooks", skill_type="required")])
    out = enrich_accepted_adjacency_jobs(
        [job], rows, ids, names, canons,
        target_noc="31301",
    )
    # Under the OLD code, target_role_text would have leaked through
    # and pulled "wrong" into credential_warning_text. Fix 2 keeps it
    # None.
    assert out[0].credential_warning_text is None


def test_credential_warning_fires_only_for_job_own_noc(monkeypatch):
    """Same smart mock; this time the adjacent job HAS the regulated
    NOC. The lookup must fire on the job's own NOC and produce the
    correct wording."""
    def smart_regulated(noc, target):
        if noc == "31301":
            return {"regulator_name": "CNO", "licensing_note": "RN licence required."}
        return None
    monkeypatch.setattr(te, "_regulated", smart_regulated)

    rows, ids, names, canons, _ = _baseline_inputs()
    job = _job(noc_code="31301", skills=[_js("QuickBooks", skill_type="required")])
    out = enrich_accepted_adjacency_jobs(
        [job], rows, ids, names, canons,
        target_noc="31301",
    )
    text = out[0].credential_warning_text
    assert text is not None
    assert "CNO" in text
    assert "RN licence required" in text


def test_credential_helper_passes_none_for_target_role(monkeypatch):
    """Direct assertion on the arguments `_credential_warning_text`
    passes to `_regulated`. Fix 2 mandates target_role be None for
    adjacency lookups — guard the call site itself."""
    calls = []
    def recording_regulated(noc, target):
        calls.append((noc, target))
        return None
    monkeypatch.setattr(te, "_regulated", recording_regulated)

    rows, ids, names, canons, _ = _baseline_inputs()
    job = _job(noc_code="12102", skills=[_js("QuickBooks", skill_type="required")])
    enrich_accepted_adjacency_jobs(
        [job], rows, ids, names, canons,
        target_noc="31301",   # would have been the misattribution vector
    )
    # _regulated was called exactly once, with the JOB's NOC and a
    # None target.
    assert calls == [("12102", None)]


def test_enrich_signature_has_no_target_role_text():
    """Direct signature check: `enrich_accepted_adjacency_jobs` no
    longer accepts a `target_role_text` keyword. A future regression
    that added it back would re-open the misattribution surface."""
    import inspect
    sig = inspect.signature(enrich_accepted_adjacency_jobs)
    assert "target_role_text" not in sig.parameters


def test_enrich_skips_malformed_job_entries():
    rows, ids, names, canons, job = _baseline_inputs()
    bad = [
        "not a dict",
        {"job_id": "x"},                                  # no title
        {"title": "X"},                                   # no job_id
        {"job_id": "", "title": "X"},                     # empty job_id
        {"job_id": "y", "title": ""},                     # empty title
        job,
    ]
    out = enrich_accepted_adjacency_jobs(
        bad, rows, ids, names, canons,
        target_noc=None,
    )
    assert len(out) == 1
    assert out[0].job_id == job["job_id"]
