"""AR-9.bug.1: populate `presented_job_ids` on present_matches.

Live observation (2026-06-10):
  User profile produced direct matches including "Account Manager".
  User: "find jobs related to my skills"
  Adjacency engine ran, returned Account Manager AGAIN under
  "transferable roles." Same posting in two sections.

Root cause:
  `_capture_match_snapshot` at handler.py:1654 builds the snapshot
  dict with `captured_at_turn`, `lead_job`, `other_jobs_meta` -- but
  NEVER writes `presented_job_ids`. The staging sanitizer falls
  through to the empty-list default; `_run_adjacency_engine_and_persist`
  reads an empty exclusion set; `drop_excluded()` has nothing to
  exclude.

Fix:
  - Capture prior `last_match_snapshot.presented_job_ids` BEFORE
    overwriting the snapshot.
  - Build new `presented_job_ids` from this turn's results, prepended
    to the prior list, deduped + capped at MAX_PRESENTED_JOB_IDS,
    most-recent-first.
  - Include the list in the new snapshot dict.

Contract:
  - Order: this turn's results FIRST (most recent), then prior list.
  - Dedup by capped job_id string.
  - Cap at MAX_PRESENTED_JOB_IDS (16).
  - Each id capped at MAX_JOB_ID_CHARS (40).
  - Empty/non-string job_ids skipped silently.
  - Prior list with non-string entries filtered out.
  - Lifecycle: cleared when target_role changes
    (StagedProfile.__setattr__) and when present_no_match /
    present_near_miss fires (`_clear_match_snapshot`).
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.nodb


from skillbridge.chat.handler import _capture_match_snapshot
from skillbridge.session.staging import (
    MAX_JOB_ID_CHARS,
    MAX_PRESENTED_JOB_IDS,
    StagedProfile,
)


def _staged(message_count: int = 5) -> StagedProfile:
    sp = StagedProfile.new("sess-ar9bug1")
    sp.message_count = message_count
    sp.target_role_text = "accounting clerk"
    return sp


def _result(job_id: str, title: str = "Some Role", employer: str = "ACME") -> dict:
    return {
        "job_id": job_id,
        "title": title,
        "employer": employer,
        "score_explanation": {
            "credential_gap_skills": [],
        },
        "missing_skills": [],
    }


# =========================================================================
# Bug reproduction: empty snapshot before fix
# =========================================================================
def test_capture_writes_presented_job_ids_from_results():
    """The snapshot dict MUST contain `presented_job_ids` after a
    present_matches turn. Pre-fix the field was simply absent and
    the staging sanitizer defaulted to []."""
    sp = _staged()
    results = [
        _result("job-001", "Accounting Clerk", "Diamond J"),
        _result("job-002", "Billing Clerk", "Lock City"),
        _result("job-003", "Account Manager", "Diamond J"),
    ]
    _capture_match_snapshot(sp, results)

    assert sp.last_match_snapshot is not None
    presented = sp.last_match_snapshot.get("presented_job_ids")
    assert presented == ["job-001", "job-002", "job-003"]


def test_capture_preserves_results_order_most_recent_first():
    """The contract from AR-1 staging.py:647 -- most-recent first.
    Results are presented in band-order (best first); they're the
    most-recent presentation context."""
    sp = _staged()
    results = [_result(f"job-{i:03d}") for i in range(5)]
    _capture_match_snapshot(sp, results)

    assert sp.last_match_snapshot["presented_job_ids"] == [
        "job-000", "job-001", "job-002", "job-003", "job-004",
    ]


# =========================================================================
# Accumulation across consecutive present_matches turns
# =========================================================================
def test_accumulates_across_consecutive_present_matches_turns():
    """Turn N shows {A, B, C}; turn N+1 shows {D, E}. The snapshot
    after N+1 must carry all five, most-recent first. The adjacency
    engine then excludes everything the user has already seen, not
    just this turn's results."""
    sp = _staged()
    _capture_match_snapshot(sp, [
        _result("job-A"), _result("job-B"), _result("job-C"),
    ])
    # Simulate the next turn's message_count + a fresh result set.
    sp.message_count = 6
    _capture_match_snapshot(sp, [_result("job-D"), _result("job-E")])

    presented = sp.last_match_snapshot["presented_job_ids"]
    # Most-recent first: D, E from this turn, then prior A, B, C.
    assert presented == ["job-D", "job-E", "job-A", "job-B", "job-C"]


def test_dedups_when_same_job_appears_in_multiple_turns():
    """A posting that legitimately appears twice (e.g. same band-good
    role surfacing on turn N and N+1 because nothing better fits)
    should only count once in the exclusion list."""
    sp = _staged()
    _capture_match_snapshot(sp, [_result("job-A"), _result("job-B")])
    sp.message_count = 6
    _capture_match_snapshot(sp, [_result("job-B"), _result("job-C")])

    presented = sp.last_match_snapshot["presented_job_ids"]
    # job-B should appear once (this turn's position wins).
    assert presented == ["job-B", "job-C", "job-A"]


def test_caps_at_max_presented_job_ids():
    """Cap is MAX_PRESENTED_JOB_IDS = 16. After enough turns the
    oldest entries fall off (most-recent first; tail drops)."""
    sp = _staged()
    # Build up a prior list of MAX entries.
    prior_ids = [f"job-old-{i:02d}" for i in range(MAX_PRESENTED_JOB_IDS)]
    sp.last_match_snapshot = {
        "captured_at_turn": 4,
        "lead_job": {"job_id": "lead", "title": "Lead", "employer": None,
                     "credential_gaps": [], "core_skill_gaps": []},
        "other_jobs_meta": [],
        "presented_job_ids": prior_ids,
    }

    # New turn introduces three new ids.
    sp.message_count = 5
    _capture_match_snapshot(sp, [
        _result("job-new-A"), _result("job-new-B"), _result("job-new-C"),
    ])

    presented = sp.last_match_snapshot["presented_job_ids"]
    assert len(presented) == MAX_PRESENTED_JOB_IDS
    # New ids landed first; the oldest prior entries fell off the tail.
    assert presented[:3] == ["job-new-A", "job-new-B", "job-new-C"]
    # First 13 prior entries survived; last 3 fell off.
    assert presented[3:] == prior_ids[:MAX_PRESENTED_JOB_IDS - 3]


# =========================================================================
# Defensive: malformed inputs
# =========================================================================
def test_skips_results_without_string_job_id():
    """Non-string / empty / None job_ids drop silently rather than
    crash. The snapshot still gets the valid ones."""
    sp = _staged()
    results = [
        _result("job-A"),
        {"job_id": None, "title": "x", "score_explanation": {}, "missing_skills": []},
        {"job_id": "", "title": "x", "score_explanation": {}, "missing_skills": []},
        {"job_id": 123, "title": "x", "score_explanation": {}, "missing_skills": []},
        _result("job-B"),
    ]
    _capture_match_snapshot(sp, results)

    assert sp.last_match_snapshot["presented_job_ids"] == ["job-A", "job-B"]


def test_skips_prior_entries_that_are_not_strings():
    """Prior snapshot dict might have corrupt entries (forged blob,
    older schema). Filter rather than propagate."""
    sp = _staged()
    sp.last_match_snapshot = {
        "captured_at_turn": 4,
        "lead_job": {"job_id": "lead", "title": "Lead", "employer": None,
                     "credential_gaps": [], "core_skill_gaps": []},
        "other_jobs_meta": [],
        "presented_job_ids": ["job-A", None, 42, "", "job-B"],
    }

    sp.message_count = 5
    _capture_match_snapshot(sp, [_result("job-C")])

    presented = sp.last_match_snapshot["presented_job_ids"]
    assert presented == ["job-C", "job-A", "job-B"]


def test_truncates_long_job_ids():
    """Each id capped at MAX_JOB_ID_CHARS = 40. Matches the
    staging sanitizer's contract."""
    sp = _staged()
    long_id = "j" * 100
    _capture_match_snapshot(sp, [_result(long_id)])

    presented = sp.last_match_snapshot["presented_job_ids"]
    assert len(presented) == 1
    assert presented[0] == "j" * MAX_JOB_ID_CHARS


# =========================================================================
# Lifecycle: presented list resets when contract requires
# =========================================================================
def test_target_role_change_resets_accumulation():
    """StagedProfile.__setattr__ clears last_match_snapshot on
    target_role_text change. A new target = new presentation
    context; prior exclusions don't carry over."""
    sp = _staged()
    _capture_match_snapshot(sp, [_result("job-A"), _result("job-B")])
    assert sp.last_match_snapshot["presented_job_ids"] == ["job-A", "job-B"]

    sp.target_role_text = "welder"  # different target
    assert sp.last_match_snapshot is None

    # Next match turn under the new target starts fresh.
    sp.message_count = 6
    _capture_match_snapshot(sp, [_result("job-C")])
    assert sp.last_match_snapshot["presented_job_ids"] == ["job-C"]


def test_clear_match_snapshot_drops_accumulation():
    """_clear_match_snapshot (called on present_no_match /
    present_near_miss) wipes the snapshot entirely. Accumulation
    starts fresh on the next present_matches turn."""
    from skillbridge.chat.handler import _clear_match_snapshot

    sp = _staged()
    _capture_match_snapshot(sp, [_result("job-A"), _result("job-B")])
    _clear_match_snapshot(sp)
    assert sp.last_match_snapshot is None

    sp.message_count = 6
    _capture_match_snapshot(sp, [_result("job-C")])
    assert sp.last_match_snapshot["presented_job_ids"] == ["job-C"]


def test_empty_results_clears_snapshot():
    """The existing contract: empty results -> snapshot = None.
    Preserved by the fix (the early return at line 1697 still
    runs before the presented_job_ids build)."""
    sp = _staged()
    _capture_match_snapshot(sp, [])
    assert sp.last_match_snapshot is None


# =========================================================================
# Integration: adjacency engine reads the populated list
# =========================================================================
def test_adjacency_engine_reads_populated_presented_job_ids(monkeypatch):
    """End-to-end pin: after _capture_match_snapshot writes
    presented_job_ids, the adjacency engine's
    _run_adjacency_engine_and_persist sees the same list when it
    reads `staged.last_match_snapshot.get("presented_job_ids")`."""
    from skillbridge.chat import handler

    sp = _staged()
    _capture_match_snapshot(sp, [
        _result("job-direct-1", "Account Manager"),
        _result("job-direct-2", "Accounting Clerk"),
    ])
    # Confirm the snapshot was populated by _capture_match_snapshot.
    assert sp.last_match_snapshot["presented_job_ids"] == [
        "job-direct-1", "job-direct-2",
    ]

    # Capture what `_run_adjacency_engine_and_persist` would read.
    # We stub the engine pipeline to isolate the snapshot-read step.
    captured_exclusion: dict[str, tuple] = {}

    def _capture_drop_excluded(ranked, presented_job_ids):
        captured_exclusion["presented"] = tuple(presented_job_ids)
        return []

    monkeypatch.setattr(
        "skillbridge.match.adjacent._load_active_jobs_with_skills",
        lambda: [],
    )
    monkeypatch.setattr(
        "skillbridge.match.adjacent.retrieve_candidates",
        lambda *a, **kw: [],
    )
    monkeypatch.setattr(
        "skillbridge.match.adjacent.accept_candidates",
        lambda *a, **kw: ([], {
            "no_evidence": 0, "no_required_non_credential_skills": 0,
            "credential": 0, "coverage": 0, "transferable": 0,
        }),
    )
    monkeypatch.setattr(
        "skillbridge.match.adjacent.rank_adjacent",
        lambda *a, **kw: [],
    )
    monkeypatch.setattr(
        "skillbridge.match.adjacent.drop_excluded",
        _capture_drop_excluded,
    )

    handler._run_adjacency_engine_and_persist(sp, trigger="user_explicit")

    # The adjacency engine read the populated list, not an empty tuple.
    assert captured_exclusion["presented"] == ("job-direct-1", "job-direct-2")
