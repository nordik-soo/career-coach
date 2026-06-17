"""AR-7 acceptance tests: telemetry + multi-turn transcripts +
Redis-mode state bound.

Three pieces (per docs/adjacent-recommendations-design.md v11
§"Build slices: AR-7"):
  - aggregate telemetry log line emitted on adjacency turns: counts
    + trigger + threshold, no PII;
  - end-to-end transcripts at the locked dispatch level
    (handle_anonymous save-and-clear, soft-offer affirmative,
    ordinal follow-up, scope-violation digression + recovery);
  - Redis-mode worst-case state bound under realistic full
    population (resume_facts_json, R-1 caps, accumulated
    credentials, AR-1 caps).

Acceptance for `_maybe_append_soft_offer` reoffer suppression
itself lives in AR-6b (test_ar6b_soft_offer.py) -- this file
covers the *transcript* layer (entry-time pending state survives
all the way to the consume point).
"""
from __future__ import annotations

import logging
from typing import Any

import pytest

pytestmark = pytest.mark.nodb


from skillbridge.chat.adjacent_intent import AdjacentIntent
from skillbridge.session.staging import (
    MAX_ADJACENT_ITEMS,
    MAX_CANONICAL_CHARS,
    MAX_CRED_GAPS,
    MAX_EMPLOYER_CHARS,
    MAX_EVIDENCE_CHARS,
    MAX_JOB_ID_CHARS,
    MAX_MATCHED_SKILLS,
    MAX_OTHER_JOBS,
    MAX_PRESENTED_JOB_IDS,
    MAX_SKILL_CHARS,
    MAX_SKILL_GAPS,
    MAX_TITLE_CHARS,
    StagedProfile,
    StagedSkill,
)


# =========================================================================
# Helpers
# =========================================================================
class _FakeStore:
    """Persisting fake session store. Mirrors test_chat_handler_v2.FakeStore
    but actually round-trips state so multi-turn transcripts work."""

    def __init__(self) -> None:
        self.held: dict[str, StagedProfile] = {}

    def new_session(self) -> str:
        return "sess-1"

    def load(self, session_id):
        return self.held.get(session_id)

    def save(self, staged: StagedProfile) -> str:
        self.held[staged.session_id] = staged
        return staged.session_id

    def delete(self, session_id) -> None:
        self.held.pop(session_id, None)


def _staged_with_evidence() -> StagedProfile:
    sp = StagedProfile.new("sess-1")
    sp.message_count = 3
    sp.target_role_text = "warehouse worker"
    sp.intake_state = "intake_collecting"
    sp.skills = [
        StagedSkill(skill_name="welding", source="resume", confidence=0.8),
        StagedSkill(skill_name="blueprint reading", source="resume", confidence=0.8),
        StagedSkill(skill_name="forklift operation", source="resume", confidence=0.8),
    ]
    return sp


def _zero_drops() -> dict[str, int]:
    return {
        "no_evidence": 0, "no_required_non_credential_skills": 0,
        "credential": 0, "coverage": 0, "transferable": 0,
    }


def _stub_engine_pipeline(monkeypatch, *, drops=None) -> None:
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
        lambda *a, **kw: ([], drops if drops is not None else _zero_drops()),
    )
    monkeypatch.setattr(
        "skillbridge.match.adjacent.rank_adjacent",
        lambda *a, **kw: [],
    )


# =========================================================================
# AR-7 part 1: aggregate telemetry
# =========================================================================
def test_engine_emits_aggregate_telemetry(monkeypatch, caplog) -> None:
    """`_run_adjacency_engine_and_persist` must emit EXACTLY ONE
    aggregate log line per turn covering candidate-pool /
    candidates-returned / drop buckets / trigger / coverage
    threshold. Two emissions per turn would double-count the
    acceptance funnel."""
    from skillbridge.chat import handler

    # 10 candidates in the pool, all dropped (1 + 3 + 4 + 2 = 10),
    # zero returned. This wires the stub end-to-end so candidate_pool
    # (= len(retrieved)) is verifiable as a specific integer.
    monkeypatch.setattr(
        "skillbridge.match.adjacent._load_active_jobs_with_skills",
        lambda: [],
    )
    monkeypatch.setattr(
        "skillbridge.match.adjacent.retrieve_candidates",
        lambda *a, **kw: [{"job_id": f"j{i}"} for i in range(10)],
    )
    monkeypatch.setattr(
        "skillbridge.match.adjacent.accept_candidates",
        lambda *a, **kw: ([], {
            "no_evidence": 0, "no_required_non_credential_skills": 2,
            "credential": 1, "coverage": 3, "transferable": 4,
        }),
    )
    monkeypatch.setattr(
        "skillbridge.match.adjacent.rank_adjacent",
        lambda *a, **kw: [],
    )

    sp = _staged_with_evidence()
    with caplog.at_level(logging.INFO, logger="skillbridge.chat.handler"):
        handler._run_adjacency_engine_and_persist(sp, trigger="user_explicit")

    matching = [
        r for r in caplog.records
        if "adjacent_recommendations" in r.getMessage()
    ]
    assert len(matching) == 1, (
        f"expected exactly one aggregate telemetry record per turn, "
        f"got {len(matching)}: {[r.getMessage() for r in matching]}"
    )
    msg = matching[0].getMessage()
    assert "candidates_returned=0" in msg
    assert "candidate_pool=10" in msg
    assert "dropped_by_credential=1" in msg
    assert "dropped_by_coverage=3" in msg
    assert "dropped_by_transferable=4" in msg
    assert "dropped_no_required_non_credential_skills=2" in msg
    assert "trigger=user_explicit" in msg
    assert "adjacent_min_required_coverage=0.45" in msg


def test_engine_telemetry_uses_trigger_argument(monkeypatch, caplog) -> None:
    """`soft_offer_accepted` is the alternate trigger value -- it
    must surface verbatim in the telemetry."""
    from skillbridge.chat import handler

    _stub_engine_pipeline(monkeypatch)

    sp = _staged_with_evidence()
    with caplog.at_level(logging.INFO, logger="skillbridge.chat.handler"):
        handler._run_adjacency_engine_and_persist(
            sp, trigger="soft_offer_accepted",
        )

    rec = next(
        (r for r in caplog.records
         if "adjacent_recommendations" in r.getMessage()),
        None,
    )
    assert rec is not None
    assert "trigger=soft_offer_accepted" in rec.getMessage()


def test_engine_telemetry_omits_pii(monkeypatch, caplog) -> None:
    """`target_role_text` is user-typed and can contain anything.
    Verify it never appears in the telemetry line."""
    from skillbridge.chat import handler

    _stub_engine_pipeline(monkeypatch)

    sp = _staged_with_evidence()
    sp.target_role_text = "very-distinct-pii-token-9292"
    with caplog.at_level(logging.INFO, logger="skillbridge.chat.handler"):
        handler._run_adjacency_engine_and_persist(sp, trigger="user_explicit")

    rec = next(
        (r for r in caplog.records
         if "adjacent_recommendations" in r.getMessage()),
        None,
    )
    assert rec is not None
    assert "very-distinct-pii-token" not in rec.getMessage()


# =========================================================================
# AR-7 part 2: multi-turn transcripts
# =========================================================================
def _stub_v2_chain_for_transcript(monkeypatch) -> None:
    """Patch the v2 chain so the transcript test doesn't need a real
    DB / LLM. Stubs `compose_response_v2` deterministically; lets
    `_try_adjacency_dispatch` reach its branches."""
    from skillbridge.chat import handler
    from skillbridge.match import adjacent as adj_mod

    monkeypatch.setattr(handler, "CHAT_ORCHESTRATOR", "v2")
    monkeypatch.setattr(adj_mod, "_adjacency_enabled", lambda: True)
    monkeypatch.setattr(handler, "compose_response_v2", lambda inp: "stub-reply")
    monkeypatch.setattr(
        handler, "compose_reply",
        lambda inp: pytest.fail("v1 must not run"),
    )
    monkeypatch.setattr(
        handler, "_run_adjacency_engine_and_persist",
        lambda staged, *, trigger="user_explicit": _stub_run_engine(staged, trigger),
    )
    monkeypatch.setattr(handler, "plan_next_move", lambda truth_json: None)


def _stub_run_engine(staged: StagedProfile, trigger: str) -> dict[str, Any]:
    """Stand-in for `_run_adjacency_engine_and_persist` that persists
    a two-item snapshot and returns a matching payload."""
    items = [
        {"job_id": "job-1", "title": "Welder",
         "evidence_summary": "3 of 5 required skills",
         "why_adjacent": "skill_evidence",
         "matched_skills": ["welding"]},
        {"job_id": "job-2", "title": "Forklift Operator",
         "evidence_summary": "2 of 4 required skills",
         "why_adjacent": "skill_evidence",
         "matched_skills": ["forklift operation"]},
    ]
    staged.last_adjacent_snapshot = {
        "created_message_count": staged.message_count,
        "items": items,
    }
    return {
        "recommendations": [
            {"job_id": it["job_id"], "title": it["title"],
             "employer": "ACME", "location": "Sault Ste. Marie, ON",
             "evidence_summary": it["evidence_summary"],
             "why_adjacent": it["why_adjacent"],
             "matched_skills": list(it["matched_skills"])}
            for it in items
        ],
        "total_retrieved": 5,
        "total_dropped_by_credential_gap": 1,
        "total_dropped_by_coverage_floor": 1,
        "total_dropped_by_transferable_floor": 0,
        "total_dropped_by_no_required_non_credential_skills": 1,
    }


def test_transcript_affirmative_after_soft_offer_fires_recommend(monkeypatch) -> None:
    """Turn N+1 entry-time `pending_adjacent_offer=True` + user says
    'yes' -> AdjacentIntent fires (soft_offer_accepted trigger),
    dispatch synthesizes recommend_adjacent_roles, snapshot
    persisted."""
    from skillbridge.chat import handler

    _stub_v2_chain_for_transcript(monkeypatch)
    monkeypatch.setattr(
        "skillbridge.chat.adjacent_followup.resolve_adjacent_followup",
        lambda *a, **kw: None,
    )

    sp = _staged_with_evidence()
    response = handler._try_v2_path(
        staged=sp, message="yes please",
        uploaded_file=False, resume_info=None, store=_FakeStore(),
        pending_adjacent_offer=True,
    )
    assert response is not None
    assert response.get("final_move") == "recommend_adjacent_roles"
    assert sp.last_adjacent_snapshot is not None
    assert len(sp.last_adjacent_snapshot["items"]) == 2


def test_transcript_ordinal_followup_describes_role(monkeypatch) -> None:
    """Turn N+1: live snapshot from prior turn. User says 'tell me
    about the second one'. resolve_adjacent_followup returns item 2;
    describe_adjacent_role synthesized."""
    from skillbridge.chat import handler

    _stub_v2_chain_for_transcript(monkeypatch)
    monkeypatch.setattr(
        "skillbridge.chat.adjacent_followup.render_describe_adjacent_role",
        lambda item: {
            "job": {"title": item["title"], "employer": None,
                    "location": "Sault Ste. Marie", "url": None,
                    "posted_date": None},
            "evidence_summary": item.get("evidence_summary", ""),
            "matched_skills": list(item.get("matched_skills", [])),
            "expired": False,
        },
    )

    sp = _staged_with_evidence()
    sp.last_adjacent_snapshot = {
        "created_message_count": sp.message_count - 1,
        "items": [
            {"job_id": "j1", "title": "Welder",
             "evidence_summary": "3 of 5",
             "why_adjacent": "skill_evidence",
             "matched_skills": ["welding"]},
            {"job_id": "j2", "title": "Forklift Operator",
             "evidence_summary": "2 of 4",
             "why_adjacent": "skill_evidence",
             "matched_skills": ["forklift operation"]},
        ],
    }
    response = handler._try_v2_path(
        staged=sp, message="tell me about the second one",
        uploaded_file=False, resume_info=None, store=_FakeStore(),
        pending_adjacent_offer=False,
    )
    assert response is not None
    assert response.get("final_move") == "describe_adjacent_role"


# =========================================================================
# AR-7 part 2b: handle_anonymous save-and-clear
# =========================================================================
def test_handle_anonymous_consumes_pending_adjacent_offer_on_entry(monkeypatch) -> None:
    """Contract (handler.py:2819-2821): `pending_adjacent_offer` is
    captured and CLEARED at handler entry. The captured value is
    threaded into `_try_v2_path`. Even if downstream returns early,
    the flag ends the turn False. Without this, a 'no thanks' from
    the user would silently roll the offer forward.

    We drive a real `handle_anonymous` call with a persisted
    `pending_adjacent_offer=True` and assert:
      - the captured kwarg landed in `_try_v2_path` as True;
      - the staged profile's flag is False after the turn.
    """
    from skillbridge.chat import handler

    captured: dict[str, Any] = {}

    def _spy_try_v2(*, staged, message, uploaded_file, resume_info, store,
                    pending_adjacent_offer):
        captured["kwarg"] = pending_adjacent_offer
        captured["flag_after_save_and_clear"] = staged.pending_adjacent_offer
        store.save(staged)
        return {"reply": "stub", "session_id": staged.session_id,
                "final_move": "ask_one_clarifying_question",
                "ask_slot": None, "results": [],
                "training_by_job": {}, "next_skill_target": None,
                "band_signal": "none", "requires_consent": True}

    store = _FakeStore()
    sp = _staged_with_evidence()
    sp.pending_adjacent_offer = True
    store.save(sp)

    monkeypatch.setattr(handler, "get_store", lambda: store)
    monkeypatch.setattr(handler, "CHAT_ORCHESTRATOR", "v2")
    monkeypatch.setattr(handler, "_try_v2_path", _spy_try_v2)

    handler.handle_anonymous("hello", "sess-1")

    assert captured.get("kwarg") is True, (
        "handle_anonymous must thread the entry-time "
        "pending_adjacent_offer value into _try_v2_path"
    )
    assert captured.get("flag_after_save_and_clear") is False, (
        "handle_anonymous must CLEAR staged.pending_adjacent_offer "
        "BEFORE _try_v2_path runs so a downstream early return "
        "cannot accidentally roll the offer forward"
    )
    persisted = store.held["sess-1"]
    assert persisted.pending_adjacent_offer is False


def test_handle_anonymous_does_not_set_pending_when_already_false(monkeypatch) -> None:
    """Symmetric guard: when the entry-time flag is False, the
    captured kwarg into `_try_v2_path` is also False. Reoffer
    suppression upstream depends on this signal being faithful."""
    from skillbridge.chat import handler

    captured: dict[str, Any] = {}

    def _spy_try_v2(*, staged, message, uploaded_file, resume_info, store,
                    pending_adjacent_offer):
        captured["kwarg"] = pending_adjacent_offer
        store.save(staged)
        return {"reply": "stub", "session_id": staged.session_id,
                "final_move": "ask_one_clarifying_question",
                "ask_slot": None, "results": [],
                "training_by_job": {}, "next_skill_target": None,
                "band_signal": "none", "requires_consent": True}

    store = _FakeStore()
    sp = _staged_with_evidence()
    sp.pending_adjacent_offer = False
    store.save(sp)

    monkeypatch.setattr(handler, "get_store", lambda: store)
    monkeypatch.setattr(handler, "CHAT_ORCHESTRATOR", "v2")
    monkeypatch.setattr(handler, "_try_v2_path", _spy_try_v2)

    handler.handle_anonymous("hello", "sess-1")
    assert captured.get("kwarg") is False


# =========================================================================
# AR-7 part 2c: scope-violation digression + recovery
# =========================================================================
def _truth_with_scope_violation(user_message: str):
    """Build a truth-summary stand-in with scope_violations_detected
    non-empty. `_try_v2_path` only reads `.scope_violations_detected`
    and `.to_planner_json()` between truth-build and adjacency
    dispatch; the default TruthSummary handles both."""
    from skillbridge.chat.truth_summary import TruthSummary

    truth = TruthSummary(user_message=user_message)
    truth.scope_violations_detected = ["immigration"]
    return truth


def _truth_with_no_scope_violation(user_message: str):
    from skillbridge.chat.truth_summary import TruthSummary

    return TruthSummary(user_message=user_message)


def test_transcript_scope_digression_shifts_ttl_then_recovers(monkeypatch) -> None:
    """Three-turn transcript:
      - Turn A (prior): snapshot persisted with
        `created_message_count = 4`. `message_count = 5` -> TTL
        live (`current == created + 1`).
      - Turn B (digression): scope-violation forces TTL shift.
        `_try_v2_path` emits `redirect_scope` via the router;
        `created_message_count` advanced to 5 BEFORE that. The
        snapshot survives because redirect_scope is NOT in the
        snapshot-clearing branches (present_matches /
        present_near_miss).
        Then we manually advance `message_count` to 6 (handle_anonymous's
        touch() responsibility, which we bypass when calling
        `_try_v2_path` directly).
      - Turn C (recovery): ordinal follow-up. TTL: `6 == 5 + 1`
        live. resolve_adjacent_followup returns item 2;
        describe_adjacent_role synthesized.

    This proves the lifecycle the design promises: a single
    redirect_scope digression does not burn the followup window.
    """
    from skillbridge.chat import handler

    _stub_v2_chain_for_transcript(monkeypatch)

    sp = _staged_with_evidence()
    sp.message_count = 5
    sp.last_adjacent_snapshot = {
        "created_message_count": 4,
        "items": [
            {"job_id": "j1", "title": "Welder",
             "evidence_summary": "3 of 5",
             "why_adjacent": "skill_evidence",
             "matched_skills": ["welding"]},
            {"job_id": "j2", "title": "Forklift Operator",
             "evidence_summary": "2 of 4",
             "why_adjacent": "skill_evidence",
             "matched_skills": ["forklift operation"]},
        ],
    }
    store = _FakeStore()

    # --- Turn B: scope-violation digression. ---
    # On scope-violated turns the dispatch is short-circuited
    # BEFORE the resolver, so the real resolver is unreachable;
    # we don't stub it. The router synthesizes redirect_scope.
    monkeypatch.setattr(
        handler, "build_truth_summary",
        lambda **kw: _truth_with_scope_violation(kw.get("user_message", "")),
    )
    response_b = handler._try_v2_path(
        staged=sp, message="how do I get my PR card?",
        uploaded_file=False, resume_info=None, store=store,
        pending_adjacent_offer=False,
    )
    assert response_b is not None
    assert response_b.get("final_move") == "redirect_scope"
    # The shift advanced created_message_count by 1.
    assert sp.last_adjacent_snapshot is not None, (
        "snapshot must survive a scope-violation digression"
    )
    assert sp.last_adjacent_snapshot["created_message_count"] == 5, (
        "shift_adjacent_snapshot_ttl must advance "
        "created_message_count by 1 on a scope-violation turn"
    )

    # --- Simulate handle_anonymous's touch() between turns. ---
    sp.message_count = 6

    # --- Turn C: recovery / ordinal follow-up. ---
    # Real resolver runs: it must detect the "second one" ordinal AND
    # confirm TTL liveness (current=6 == created=5 + 1). Only the
    # live-job renderer is stubbed (the production version queries
    # core.v_current_job, which we don't have in tests).
    monkeypatch.setattr(
        handler, "build_truth_summary",
        lambda **kw: _truth_with_no_scope_violation(kw.get("user_message", "")),
    )
    monkeypatch.setattr(
        "skillbridge.chat.adjacent_followup.render_describe_adjacent_role",
        lambda item: {
            "job": {"title": item["title"], "employer": None,
                    "location": "Sault Ste. Marie", "url": None,
                    "posted_date": None},
            "evidence_summary": item.get("evidence_summary", ""),
            "matched_skills": list(item.get("matched_skills", [])),
            "expired": False,
        },
    )
    response_c = handler._try_v2_path(
        staged=sp, message="tell me about the second one",
        uploaded_file=False, resume_info=None, store=store,
        pending_adjacent_offer=False,
    )
    assert response_c is not None
    assert response_c.get("final_move") == "describe_adjacent_role"


# =========================================================================
# AR-7 part 3: Redis-mode worst-case state bound
# =========================================================================
def _redis_mode_worst_case_profile() -> StagedProfile:
    """Realistic Redis-mode worst case. Populates every field a
    long-running session can accumulate: resume_facts_json (8 skills
    + 2 jobs + 1 cert + 1 lang -- mirrors the R-1 cookie test
    fixture), R-1 snapshot at every cap, accumulated credentials
    list, AR-1 snapshot + flag at every cap. Catches accidental
    cap regressions across the entire serialization surface."""
    s = StagedProfile.new("sess-redis-worst-case")
    s.message_count = 24
    s.target_role_text = "warehouse worker"
    s.skills_text = "forklift, picking, packing, shipping, receiving"
    s.experience_text = "Three years at a Sault Ste. Marie distribution centre."
    s.education_text = "High school diploma."
    s.intake_state = "intake_collecting"
    s.skills = [
        StagedSkill(skill_name=f"skill-{i}", source="resume", confidence=0.8)
        for i in range(8)
    ]

    # Resume facts (compact form).
    s.resume_facts_json = {
        "skills": [{"name": f"Skill {i}", "fact_id": f"f{i}"} for i in range(8)],
        "work_history": [
            {"title": f"Job {i}", "employer": f"Employer {i}",
             "start_year": 2020 + i, "end_year": 2022 + i, "fact_id": f"w{i}"}
            for i in range(2)
        ],
        "certifications": [{"name": "Smart Serve", "fact_id": "c0"}],
        "languages": [{"name": "English", "fact_id": "l0"}],
    }
    s.suppressed_fact_ids = [f"f{i}" for i in range(4)]

    # Slice-8 conversation context (5 entries each).
    s.last_presented_job_titles = [f"Title {i}" for i in range(5)]
    s.last_presented_caps_applied = [f"cap_{i}" for i in range(5)]
    s.last_presented_credential_gaps = [f"cred_{i}" for i in range(5)]

    # R-1 snapshot at every cap, plus AR-1 presented_job_ids
    # at MAX_PRESENTED_JOB_IDS x MAX_JOB_ID_CHARS.
    s.last_match_snapshot = {
        "captured_at_turn": 23,
        "lead_job": {
            "job_id":   "j" * MAX_CANONICAL_CHARS,
            "title":    "x" * MAX_TITLE_CHARS,
            "employer": "y" * MAX_EMPLOYER_CHARS,
            "credential_gaps": [
                {"display":   "d" * MAX_CANONICAL_CHARS,
                 "canonical": "c" * MAX_CANONICAL_CHARS}
                for _ in range(MAX_CRED_GAPS)
            ],
            "core_skill_gaps": [
                "s" * MAX_CANONICAL_CHARS for _ in range(MAX_SKILL_GAPS)
            ],
        },
        "other_jobs_meta": [
            {"job_id": "j" * MAX_CANONICAL_CHARS,
             "title":  "t" * MAX_TITLE_CHARS}
            for _ in range(MAX_OTHER_JOBS)
        ],
        "presented_job_ids": [
            f"job-{i:03d}-" + "j" * (MAX_JOB_ID_CHARS - len(f"job-{i:03d}-"))
            for i in range(MAX_PRESENTED_JOB_IDS)
        ],
    }

    # Accumulated credentials -- conversation state, append-and-dedupe.
    # 5 entries is a reasonable upper bound for a long session.
    s.last_assumed_completed_credentials = [
        {"canonical": "c" * MAX_CANONICAL_CHARS, "mode": "claimed"},
        {"canonical": "d" * MAX_CANONICAL_CHARS, "mode": "hypothetical"},
        {"canonical": "e" * MAX_CANONICAL_CHARS, "mode": "claimed"},
        {"canonical": "f" * MAX_CANONICAL_CHARS, "mode": "hypothetical"},
        {"canonical": "g" * MAX_CANONICAL_CHARS, "mode": "claimed"},
    ]
    s.last_discussed_credential_canonical = "c" * MAX_CANONICAL_CHARS
    s.pending_credential_confirmation = {
        "canonical": "h" * MAX_CANONICAL_CHARS,
        "action": "add",
    }

    # AR-1 snapshot at full item / field caps + pending flag.
    s.last_adjacent_snapshot = {
        "created_message_count": 23,
        "items": [
            {
                "job_id":           f"adj-{i:02d}-" + "j" * (MAX_JOB_ID_CHARS - len(f"adj-{i:02d}-")),
                "title":            "T" * MAX_TITLE_CHARS,
                "evidence_summary": "E" * MAX_EVIDENCE_CHARS,
                "why_adjacent":     "same_noc_minor_group",
                "matched_skills":   [
                    f"s{k}-" + "s" * (MAX_SKILL_CHARS - len(f"s{k}-"))
                    for k in range(MAX_MATCHED_SKILLS)
                ],
            }
            for i in range(MAX_ADJACENT_ITEMS)
        ],
    }
    s.pending_adjacent_offer = True
    return s


def test_redis_mode_worst_case_state_stays_bounded() -> None:
    """Redis-mode has no per-cookie ceiling, but cap discipline must
    still bound the worst-case JSON size so a long session can't
    grow without limit. The bound below leaves comfortable headroom
    for incremental design additions while flagging any accidental
    cap regression."""
    sp = _redis_mode_worst_case_profile()
    blob = sp.to_json(redact_for_cookie=False).encode("utf-8")
    size = len(blob)
    assert size < 10_000, (
        f"Redis-mode worst-case StagedProfile is {size} bytes -- "
        f"cap discipline regressed. The bound exists to surface "
        f"accidental MAX_ blow-ups and unbounded list growth."
    )


def test_redis_mode_worst_case_state_round_trips() -> None:
    """Round-trip soundness: after to_json(False) + from_json, the
    full serialized form is BYTE-FOR-BYTE preserved. Catches any
    field the defensive deserializer silently drops or reshapes,
    not just the handful a per-field assertion would name."""
    sp = _redis_mode_worst_case_profile()
    serialized = sp.to_json(redact_for_cookie=False)
    sp2 = StagedProfile.from_json(serialized)
    assert sp2.to_json(redact_for_cookie=False) == serialized
