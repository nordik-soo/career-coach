"""AR-6c tests: lifecycle clear + ResponderV2Input payloads +
deterministic responder fallbacks.

Covers (per docs/adjacent-recommendations-design.md v11):
  - Lifecycle: new `present_matches` and `present_near_miss` decisions
    clear `last_adjacent_snapshot`. `present_no_match` does NOT.
  - ResponderV2Input has `adjacent_recommendations_payload` and
    `adjacent_role_description_payload` fields.
  - `_run_adjacency_engine_and_persist` returns the payload shape
    locked in v11 §"Locked StagedProfile / ResponderV2Input
    additions": recommendations list + drop counters.
  - `_build_user_block_v2` surfaces both payloads as tagged blocks
    (ADJACENT_RECOMMENDATIONS, ADJACENT_ROLE_DESCRIPTION) on the
    matching final_moves only.
  - `_fallback_reply_v2` handles both new outcomes deterministically:
    * recommend_adjacent_roles with empty list -> provider-free
      empty-result line ("From today's Sault Ste. Marie postings...").
    * recommend_adjacent_roles with recommendations -> "roles worth
      exploring" framing, lists titles + employers + evidence.
    * describe_adjacent_role with expired=True -> deterministic
      fallback ("that role's no longer on the board").
    * describe_adjacent_role with live job -> name + employer +
      location + evidence + next-step.
  - Forbidden vocabulary check on the fallback paths.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.nodb


from skillbridge.chat.arbiter import ArbiterDecision
from skillbridge.chat.responder import (
    ResponderV2Input,
    _build_user_block_v2,
    _describe_adjacent_role_fallback_v2,
    _fallback_reply_v2,
    _recommend_adjacent_roles_fallback_v2,
)
from skillbridge.chat.url_views import (
    build_sanitized_responder_view_v2 as _v_v2,
)


def _iv(inp):
    """Return (inp, view) so each test reuses one inp instance."""
    return inp, _v_v2(inp)
from skillbridge.session.staging import StagedProfile, StagedSkill


def _decision(move: str, **kw) -> ArbiterDecision:
    return ArbiterDecision(
        final_move=move,
        reason_code=kw.get("reason_code", "x"),
        tone=kw.get("tone", "brief_confident"),
        arbiter_action=kw.get("arbiter_action", "handler_synthesized_adjacent_recommendations"),
        ask_slot=kw.get("ask_slot"),
        caps_applied=kw.get("caps_applied", ()),
    )


def _input(move: str, **payloads) -> ResponderV2Input:
    return ResponderV2Input(
        user_message="hi",
        decision=_decision(move),
        results=[],
        training_by_job={},
        next_skill=(None, 0),
        band_signal="none",
        requires_consent=True,
        target_role_text="warehouse worker",
        resume_facts=None,
        conversation_context=None,
        **payloads,
    )


# =========================================================================
# ResponderV2Input field additions
# =========================================================================
def test_responder_v2_input_has_adjacent_recommendations_payload_field() -> None:
    inp = _input("recommend_adjacent_roles",
                 adjacent_recommendations_payload={"recommendations": []})
    assert inp.adjacent_recommendations_payload == {"recommendations": []}


def test_responder_v2_input_has_adjacent_role_description_payload_field() -> None:
    inp = _input("describe_adjacent_role",
                 adjacent_role_description_payload={"expired": True})
    assert inp.adjacent_role_description_payload == {"expired": True}


def test_responder_v2_input_payloads_default_to_none() -> None:
    inp = _input("present_matches")
    assert inp.adjacent_recommendations_payload is None
    assert inp.adjacent_role_description_payload is None


# =========================================================================
# Lifecycle clear: present_matches / present_near_miss clear the snapshot
# =========================================================================
def test_present_matches_clears_last_adjacent_snapshot() -> None:
    """The lifecycle clear is wired in `_try_v2_path`'s present-matches
    branch -- static audit on the source.

    AR-9.feat.coach-tiers CP2 step 4 update: the branch is now a
    set-membership check covering both `present_matches` AND the new
    `present_tiered_matches`, so the audit anchors on the
    set-membership form.

    CP2 step 6.1 (2026-06-14) refinement: the unconditional clear is
    now the `else` arm of a stamp/clear conditional. The stamp path
    fires only when `final.final_move == "present_tiered_matches"`
    AND the Sideways tier is populated (so ordinal follow-ups
    resolve). The clear path still covers `present_matches` (legacy)
    and any `present_tiered_matches` without Sideways records, so the
    invariant the test pins — a fresh direct-match decision
    invalidates any standing adjacent snapshot — remains true for
    present_matches. The slice is widened past the conditional body
    so the `else` arm is included."""
    import inspect

    from skillbridge.chat import handler

    src = inspect.getsource(handler._try_v2_path)
    # The set-membership branch must include "present_matches".
    capture_idx = src.find(
        'if final.final_move in {"present_matches", "present_tiered_matches"}'
    )
    assert capture_idx != -1, (
        "set-membership branch for present_matches lifecycle clear not found"
    )
    capture_block = src[capture_idx:capture_idx + 2400]
    assert "staged.last_adjacent_snapshot = None" in capture_block


def test_present_near_miss_clears_last_adjacent_snapshot() -> None:
    """Same lifecycle clear on present_near_miss."""
    import inspect

    from skillbridge.chat import handler

    src = inspect.getsource(handler._try_v2_path)
    # The branch handles present_no_match and present_near_miss; the
    # near_miss subset must clear the snapshot.
    assert 'present_near_miss' in src
    assert "staged.last_adjacent_snapshot = None" in src


def test_present_no_match_does_NOT_clear_last_adjacent_snapshot() -> None:
    """present_no_match preserves the snapshot -- no new match to
    override it. The audit checks the source for an explicit
    "if final.final_move == 'present_near_miss'" branch INSIDE the
    {present_no_match, present_near_miss} block, so the clear is
    only on near_miss (not on no_match)."""
    import inspect

    from skillbridge.chat import handler

    src = inspect.getsource(handler._try_v2_path)
    # The "present_no_match" branch shares its block with
    # present_near_miss; the clear is gated to near_miss only.
    no_match_block_idx = src.find(
        'final.final_move in {"present_no_match", "present_near_miss"}'
    )
    assert no_match_block_idx != -1
    block = src[no_match_block_idx:no_match_block_idx + 800]
    assert 'final.final_move == "present_near_miss"' in block, (
        "The clear is gated to present_near_miss only -- a bare clear "
        "in the no_match/near_miss block would wipe the snapshot on "
        "present_no_match too."
    )


# =========================================================================
# _build_user_block_v2 surfaces the new payloads
# =========================================================================
def test_user_block_surfaces_adjacent_recommendations_payload() -> None:
    payload = {
        "recommendations": [
            {"job_id": "j1", "title": "Welder", "employer": "ACME",
             "location": "Sault Ste. Marie, ON",
             "evidence_summary": "3 of 5", "why_adjacent": "skill_evidence",
             "matched_skills": ["welding"]},
        ],
        "total_retrieved": 7,
    }
    _inp = _input("recommend_adjacent_roles",
                  adjacent_recommendations_payload=payload)
    block = _build_user_block_v2(*_iv(_inp))
    assert "ADJACENT_RECOMMENDATIONS:" in block
    assert "Welder" in block
    assert "skill_evidence" in block


def test_user_block_surfaces_adjacent_role_description_payload() -> None:
    payload = {
        "job": {"title": "Welder", "employer": "ACME",
                "location": "Sault Ste. Marie", "url": "https://x.test",
                "posted_date": "2026-06-01"},
        "evidence_summary": "3 of 5",
        "matched_skills": ["welding"],
        "expired": False,
    }
    _inp = _input("describe_adjacent_role",
                  adjacent_role_description_payload=payload)
    block = _build_user_block_v2(*_iv(_inp))
    assert "ADJACENT_ROLE_DESCRIPTION:" in block
    assert "Welder" in block
    assert '"expired": false' in block


def test_user_block_does_NOT_surface_payload_on_other_moves() -> None:
    """ADJACENT_RECOMMENDATIONS block must only appear on
    recommend_adjacent_roles turns -- a misplaced payload on a
    different move wouldn't make it into the prompt."""
    block = _build_user_block_v2(*_iv(_input(
        "present_matches",
        adjacent_recommendations_payload={"recommendations": []},
    )))
    assert "ADJACENT_RECOMMENDATIONS:" not in block


# =========================================================================
# _recommend_adjacent_roles_fallback_v2
# =========================================================================
def test_fallback_recommend_with_recommendations_lists_titles() -> None:
    inp = _input("recommend_adjacent_roles",
                 adjacent_recommendations_payload={
                     "recommendations": [
                         {"title": "Welder", "employer": "ACME",
                          "evidence_summary": "3 of 5 required skills"},
                         {"title": "Forklift Operator", "employer": None,
                          "evidence_summary": "2 of 4 required skills"},
                     ],
                     "total_retrieved": 7,
                 })
    reply = _recommend_adjacent_roles_fallback_v2(*_iv(inp))
    assert "Welder at ACME" in reply
    assert "Forklift Operator" in reply
    assert "3 of 5" in reply
    # Approved framing tokens:
    assert "worth exploring" in reply
    # Forbidden vocabulary:
    for token in ("you qualify", "good fit", "great fit", "perfect for you"):
        assert token not in reply.lower()


def test_fallback_recommend_with_empty_list_uses_empty_result_line() -> None:
    inp = _input("recommend_adjacent_roles",
                 adjacent_recommendations_payload={"recommendations": []})
    reply = _recommend_adjacent_roles_fallback_v2(*_iv(inp))
    assert "not seeing other roles" in reply.lower()
    # Provider-free empty-result line (v11 lock):
    for forbidden_provider in ("sccc", "career centre", "algoma"):
        assert forbidden_provider not in reply.lower()


def test_fallback_recommend_with_missing_payload_uses_empty_result_line() -> None:
    inp = _input("recommend_adjacent_roles")
    reply = _recommend_adjacent_roles_fallback_v2(*_iv(inp))
    assert "not seeing other roles" in reply.lower()


def test_fallback_recommend_via_compose_dispatch(monkeypatch) -> None:
    """The dispatch in _fallback_reply_v2 routes
    recommend_adjacent_roles to the dedicated fallback."""
    inp = _input("recommend_adjacent_roles",
                 adjacent_recommendations_payload={"recommendations": []})
    reply = _fallback_reply_v2(*_iv(inp))
    assert "not seeing other roles" in reply.lower()


# =========================================================================
# _describe_adjacent_role_fallback_v2
# =========================================================================
def test_fallback_describe_with_live_job() -> None:
    inp = _input("describe_adjacent_role",
                 adjacent_role_description_payload={
                     "job": {"title": "Welder", "employer": "ACME",
                             "location": "Sault Ste. Marie, ON",
                             "url": "https://example.com/job/123",
                             "posted_date": "2026-06-01"},
                     "evidence_summary": "3 of 5 required skills",
                     "matched_skills": ["welding", "blueprint reading"],
                     "expired": False,
                 })
    reply = _describe_adjacent_role_fallback_v2(*_iv(inp))
    assert "Welder at ACME" in reply
    assert "Sault Ste. Marie" in reply
    assert "3 of 5" in reply
    # Bug.2b: URL is now rendered inline; the extra "Want the posting
    # URL?" round-trip is gone.
    assert "https://example.com/job/123" in reply
    assert "Want me to look at the path to apply?" in reply
    for token in ("you qualify", "good fit", "great fit"):
        assert token not in reply.lower()


def test_fallback_describe_expired_uses_deterministic_line() -> None:
    inp = _input("describe_adjacent_role",
                 adjacent_role_description_payload={
                     "job": None, "evidence_summary": "",
                     "matched_skills": [], "expired": True,
                 })
    reply = _describe_adjacent_role_fallback_v2(*_iv(inp))
    assert "no longer on the board" in reply


def test_fallback_describe_no_payload_uses_deterministic_line() -> None:
    inp = _input("describe_adjacent_role")
    reply = _describe_adjacent_role_fallback_v2(*_iv(inp))
    assert "no longer on the board" in reply


def test_fallback_describe_without_url_offers_path_to_apply() -> None:
    inp = _input("describe_adjacent_role",
                 adjacent_role_description_payload={
                     "job": {"title": "Welder", "employer": None,
                             "location": "Sault Ste. Marie",
                             "url": None, "posted_date": None},
                     "evidence_summary": "",
                     "matched_skills": [], "expired": False,
                 })
    reply = _describe_adjacent_role_fallback_v2(*_iv(inp))
    assert "path to apply" in reply.lower()


# =========================================================================
# _run_adjacency_engine_and_persist returns the payload shape
# =========================================================================
# =========================================================================
# AR-6c round-2: cap enforcement
# =========================================================================
def test_engine_caps_matched_skills_at_contract_limit(monkeypatch) -> None:
    """matched_skills capped at `MAX_MATCHED_SKILLS` (4) per the
    locked contract -- not 5. The display LIST is truncated; the
    EVIDENCE COUNT in the summary string uses the FULL match
    count."""
    from skillbridge.chat import handler
    from skillbridge.session.staging import MAX_MATCHED_SKILLS, MAX_SKILL_CHARS

    # 6 required, all-matching, non-credential skills.
    required_skills = [
        {
            "skill_name": name, "skill_id": None,
            "confidence": 0.9, "importance_rank": 1, "skill_type": "required",
        }
        for name in (
            "welding", "blueprint reading", "forklift operation",
            "fitting", "metal fabrication", "machine setup",
        )
    ]
    job = {
        "job_id": "j1", "title": "Welder",
        "noc_code": "72107", "skills": required_skills,
    }
    monkeypatch.setattr(
        "skillbridge.match.adjacent._load_active_jobs_with_skills",
        lambda: [job],
    )
    monkeypatch.setattr(
        "skillbridge.match.adjacent.retrieve_candidates",
        lambda *a, **kw: [job],
    )
    monkeypatch.setattr(
        "skillbridge.match.adjacent.accept_candidates",
        lambda *a, **kw: ([job], {
            "no_evidence": 0, "no_required_non_credential_skills": 0,
            "credential": 0, "coverage": 0, "transferable": 0,
        }),
    )
    monkeypatch.setattr(
        "skillbridge.match.adjacent.rank_adjacent",
        lambda accepted, *a, **kw: list(accepted),
    )

    sp = StagedProfile.new("s")
    # User has all 6 skills -> all 6 should match.
    sp.skills = [
        StagedSkill(skill_name=n, source="resume", confidence=0.9)
        for n in (
            "welding", "blueprint reading", "forklift operation",
            "fitting", "metal fabrication", "machine setup",
        )
    ]
    payload = handler._run_adjacency_engine_and_persist(sp)

    rec = payload["recommendations"][0]
    # Display cap: at most MAX_MATCHED_SKILLS.
    assert len(rec["matched_skills"]) == MAX_MATCHED_SKILLS
    # Evidence COUNT reflects total matches, NOT the displayed
    # subset. A 6-of-6 match must read "6 of 6", not "4 of 6".
    assert "6 of 6" in rec["evidence_summary"], (
        f"Evidence count must be pre-truncation. Got: "
        f"{rec['evidence_summary']!r}"
    )
    # Same in the snapshot:
    snap_item = sp.last_adjacent_snapshot["items"][0]
    assert len(snap_item["matched_skills"]) == MAX_MATCHED_SKILLS
    assert "6 of 6" in snap_item["evidence_summary"]
    # All displayed skills are bounded by MAX_SKILL_CHARS.
    for m in rec["matched_skills"]:
        assert len(m) <= MAX_SKILL_CHARS


def test_engine_caps_job_id_title_employer(monkeypatch) -> None:
    """job_id, title, employer all truncated to the staging caps."""
    from skillbridge.chat import handler
    from skillbridge.session.staging import (
        MAX_EMPLOYER_CHARS, MAX_JOB_ID_CHARS, MAX_TITLE_CHARS,
    )

    long_job_id = "j" * (MAX_JOB_ID_CHARS + 50)
    long_title = "T" * (MAX_TITLE_CHARS + 50)
    long_employer = "E" * (MAX_EMPLOYER_CHARS + 50)
    job = {
        "job_id": long_job_id,
        "title": long_title,
        "employer": long_employer,
        "noc_code": "72107",
        "skills": [{"skill_name": "welding", "skill_type": "required"}],
    }
    monkeypatch.setattr(
        "skillbridge.match.adjacent._load_active_jobs_with_skills",
        lambda: [job],
    )
    monkeypatch.setattr(
        "skillbridge.match.adjacent.retrieve_candidates",
        lambda *a, **kw: [job],
    )
    monkeypatch.setattr(
        "skillbridge.match.adjacent.accept_candidates",
        lambda *a, **kw: ([job], {
            "no_evidence": 0, "no_required_non_credential_skills": 0,
            "credential": 0, "coverage": 0, "transferable": 0,
        }),
    )
    monkeypatch.setattr(
        "skillbridge.match.adjacent.rank_adjacent",
        lambda accepted, *a, **kw: list(accepted),
    )

    sp = StagedProfile.new("s")
    sp.skills = [
        StagedSkill(skill_name="welding", source="resume", confidence=0.9),
        StagedSkill(skill_name="blueprint reading", source="resume", confidence=0.9),
        StagedSkill(skill_name="forklift operation", source="resume", confidence=0.9),
    ]
    payload = handler._run_adjacency_engine_and_persist(sp)
    rec = payload["recommendations"][0]
    assert len(rec["job_id"]) == MAX_JOB_ID_CHARS
    assert len(rec["title"]) == MAX_TITLE_CHARS
    assert len(rec["employer"]) == MAX_EMPLOYER_CHARS


# =========================================================================
# AR-6c round-2: forbidden-framing policy check for adjacency outcomes
# =========================================================================
@pytest.mark.parametrize("move", [
    "recommend_adjacent_roles", "describe_adjacent_role",
])
@pytest.mark.parametrize("forbidden", [
    "You qualify for this role",
    "you do qualify",
    "you would qualify",
    "you're qualified",
    "You are qualified for this",
    "This is a good fit",
    "great fit for your skills",
    "perfect fit",
    "strong match",
    "stretch match",
])
def test_policy_v2_rejects_forbidden_framing_on_adjacency_moves(
    monkeypatch, move, forbidden,
) -> None:
    """Adjacency surfaces eligibility-by-credential, NOT match-quality
    certification. The policy gate (`_policy_ok_v2`) MUST reject
    "you qualify" / "good fit" / "perfect fit" / "stretch match" on
    both adjacency outcomes."""
    from skillbridge.chat import responder

    # Stub is_enabled so the LLM path is taken (the policy check
    # runs on the LLM output). Stub `call` to return the forbidden
    # phrase. Policy fail -> fall back to deterministic renderer.
    monkeypatch.setattr(responder, "is_enabled", lambda: True)
    monkeypatch.setattr(
        responder, "call",
        lambda system, user, max_tokens=500: forbidden,
    )

    if move == "recommend_adjacent_roles":
        # AR-8a: empty `recommendations` now short-circuits to the
        # deterministic fallback BEFORE the LLM is called, so to keep
        # exercising the policy gate on the LLM output we pass a
        # non-empty valid recommendation. The policy gate is what's
        # under test here, not the empty-path early-return.
        inp = _input(move, adjacent_recommendations_payload={
            "recommendations": [{"title": "Maintenance Technician",
                                 "evidence_summary": "3 of 5"}],
            "total_retrieved": 1,
        })
    else:
        inp = _input(move, adjacent_role_description_payload={
            "job": None, "evidence_summary": "",
            "matched_skills": [], "expired": True,
        })
    reply = responder.compose_response_v2(inp)
    # The forbidden phrase MUST NOT survive into the final reply.
    assert forbidden not in reply, (
        f"Policy gate failed to reject {forbidden!r} on {move} -- "
        f"got reply: {reply!r}"
    )


# --- AR-6c round-3: "perfect for you" and friends ---
@pytest.mark.parametrize("move", [
    "recommend_adjacent_roles", "describe_adjacent_role",
])
@pytest.mark.parametrize("forbidden", [
    "This role is perfect for you",
    "perfect role for you",
    "perfect fit for you",
    "great for you",
    "ideal for you",
    "ideal role for you",
    "ideal fit for you",
    "You'd be a strong candidate",
    "you'd be an excellent candidate",
    "you would be a perfect candidate",
])
def test_policy_v2_rejects_perfect_for_you_and_candidate_framings(
    monkeypatch, move, forbidden,
) -> None:
    """The v11 §"Forbidden vocabulary" list names "perfect for you"
    and "ideal role for you" explicitly. Also catches the
    "you'd be a strong candidate" family. Each MUST be rejected by
    the policy gate on both adjacency moves."""
    from skillbridge.chat import responder

    monkeypatch.setattr(responder, "is_enabled", lambda: True)
    monkeypatch.setattr(
        responder, "call",
        lambda system, user, max_tokens=500: forbidden,
    )

    if move == "recommend_adjacent_roles":
        # AR-8a: empty `recommendations` now short-circuits to the
        # deterministic fallback BEFORE the LLM is called, so to keep
        # exercising the policy gate on the LLM output we pass a
        # non-empty valid recommendation. The policy gate is what's
        # under test here, not the empty-path early-return.
        inp = _input(move, adjacent_recommendations_payload={
            "recommendations": [{"title": "Maintenance Technician",
                                 "evidence_summary": "3 of 5"}],
            "total_retrieved": 1,
        })
    else:
        inp = _input(move, adjacent_role_description_payload={
            "job": None, "evidence_summary": "",
            "matched_skills": [], "expired": True,
        })
    reply = responder.compose_response_v2(inp)
    assert forbidden not in reply, (
        f"Policy gate failed to reject {forbidden!r} on {move} -- "
        f"got reply: {reply!r}"
    )


# --- AR-6c round-3: malformed-only recommendations fall back to empty-result ---
def test_fallback_recommend_with_only_malformed_entries_uses_empty_result_line() -> None:
    """A recommendations list whose entries are all malformed (forged
    blob / broken upstream) would render an empty heading plus the
    closing question -- a misleading "we found something" frame
    around nothing. The renderer filters to valid dicts FIRST and
    falls back to the empty-result line when none remain."""
    inp = _input("recommend_adjacent_roles",
                 adjacent_recommendations_payload={
                     "recommendations": [None, "garbage", 7, [], {}],  # type: ignore[list-item]
                     "total_retrieved": 0,
                 })
    reply = _recommend_adjacent_roles_fallback_v2(*_iv(inp))
    # Must use the empty-result line, NOT the "Here are a few..."
    # heading.
    assert "not seeing other roles" in reply.lower()
    assert "Here are a few" not in reply
    assert "Want me to look closer" not in reply


# --- AR-6c round-4: candidate framings the reviewer caught ---
@pytest.mark.parametrize("move", [
    "recommend_adjacent_roles", "describe_adjacent_role",
])
@pytest.mark.parametrize("forbidden", [
    "You'll be an ideal candidate",
    "you’d be a strong candidate",         # smart apostrophe (U+2019)
    "You’ll be an ideal candidate",        # smart apostrophe
    "You are a strong candidate",
    "You're an excellent candidate",
    "you’re a perfect candidate",          # smart apostrophe
    "you will be an ideal candidate",
])
def test_policy_v2_rejects_contraction_and_are_candidate_framings(
    monkeypatch, move, forbidden,
) -> None:
    """The reviewer's case: `'ll`, `'re`, `are` forms, plus smart
    apostrophes (U+2019). Each MUST be rejected by the policy gate."""
    from skillbridge.chat import responder

    monkeypatch.setattr(responder, "is_enabled", lambda: True)
    monkeypatch.setattr(
        responder, "call",
        lambda system, user, max_tokens=500: forbidden,
    )

    if move == "recommend_adjacent_roles":
        # AR-8a: empty `recommendations` now short-circuits to the
        # deterministic fallback BEFORE the LLM is called, so to keep
        # exercising the policy gate on the LLM output we pass a
        # non-empty valid recommendation. The policy gate is what's
        # under test here, not the empty-path early-return.
        inp = _input(move, adjacent_recommendations_payload={
            "recommendations": [{"title": "Maintenance Technician",
                                 "evidence_summary": "3 of 5"}],
            "total_retrieved": 1,
        })
    else:
        inp = _input(move, adjacent_role_description_payload={
            "job": None, "evidence_summary": "",
            "matched_skills": [], "expired": True,
        })
    reply = responder.compose_response_v2(inp)
    assert forbidden not in reply, (
        f"Policy gate failed to reject {forbidden!r} on {move} -- "
        f"got reply: {reply!r}"
    )


# --- AR-6c round-5: modals could/can + perception "seem/look like" ---
@pytest.mark.parametrize("move", [
    "recommend_adjacent_roles", "describe_adjacent_role",
])
@pytest.mark.parametrize("forbidden", [
    "You could be a strong candidate",
    "you could be an ideal candidate",
    "You can be a good candidate",
    "you can be an excellent candidate",
    "You seem like a strong candidate",
    "you seem like a good candidate",
    "You look like a strong candidate",
    "you look like an ideal candidate",
    "You sound like a perfect candidate",
    "you appear to be a strong candidate",
    "You seem to be an excellent candidate",
])
def test_policy_v2_rejects_modal_and_perception_candidate_framings(
    monkeypatch, move, forbidden,
) -> None:
    """The reviewer's final cases: modal `could` / `can` + the
    perception verbs `seem` / `look` / `sound` / `appear` (with
    `like` or `to be`). Each MUST be rejected by the policy gate."""
    from skillbridge.chat import responder

    monkeypatch.setattr(responder, "is_enabled", lambda: True)
    monkeypatch.setattr(
        responder, "call",
        lambda system, user, max_tokens=500: forbidden,
    )

    if move == "recommend_adjacent_roles":
        # AR-8a: empty `recommendations` now short-circuits to the
        # deterministic fallback BEFORE the LLM is called, so to keep
        # exercising the policy gate on the LLM output we pass a
        # non-empty valid recommendation. The policy gate is what's
        # under test here, not the empty-path early-return.
        inp = _input(move, adjacent_recommendations_payload={
            "recommendations": [{"title": "Maintenance Technician",
                                 "evidence_summary": "3 of 5"}],
            "total_retrieved": 1,
        })
    else:
        inp = _input(move, adjacent_role_description_payload={
            "job": None, "evidence_summary": "",
            "matched_skills": [], "expired": True,
        })
    reply = responder.compose_response_v2(inp)
    assert forbidden not in reply, (
        f"Policy gate failed to reject {forbidden!r} on {move} -- "
        f"got reply: {reply!r}"
    )


# --- AR-6c round-6: SEMANTIC rule for candidate framing ---
@pytest.mark.parametrize("move", [
    "recommend_adjacent_roles", "describe_adjacent_role",
])
@pytest.mark.parametrize("forbidden", [
    # Adverb-insertion forms the reviewer flagged:
    "You could potentially be a strong candidate",
    "You may well be a strong candidate",
    "You appear likely to be a strong candidate",
    # Other adverb / paraphrase forms the semantic rule must catch:
    "You're definitely a strong candidate",
    "I think you're a perfect candidate",
    "We see you as an ideal candidate",
    "Honestly, you're a great candidate",
    "Believe it or not, you'd be a perfect candidate",
    # Variants of the noun phrase alone (no "you ..."):
    "This is a strong candidate fit",   # also catches "good fit" via separate pattern, but this one is the noun-phrase
    "Your background makes a strong candidate",
])
def test_policy_v2_rejects_semantic_candidate_framing(
    monkeypatch, move, forbidden,
) -> None:
    """The semantic rule (`<positive-adjective> candidate`) catches
    adverb insertions, paraphrases, and other grammar variants the
    previous syntax-by-syntax regex missed. Adjacency outcomes never
    need candidate framing -- so the broader noun-phrase ban is
    safer and simpler than chasing modals/contractions/adverbs."""
    from skillbridge.chat import responder

    monkeypatch.setattr(responder, "is_enabled", lambda: True)
    monkeypatch.setattr(
        responder, "call",
        lambda system, user, max_tokens=500: forbidden,
    )

    if move == "recommend_adjacent_roles":
        # AR-8a: empty `recommendations` now short-circuits to the
        # deterministic fallback BEFORE the LLM is called, so to keep
        # exercising the policy gate on the LLM output we pass a
        # non-empty valid recommendation. The policy gate is what's
        # under test here, not the empty-path early-return.
        inp = _input(move, adjacent_recommendations_payload={
            "recommendations": [{"title": "Maintenance Technician",
                                 "evidence_summary": "3 of 5"}],
            "total_retrieved": 1,
        })
    else:
        inp = _input(move, adjacent_role_description_payload={
            "job": None, "evidence_summary": "",
            "matched_skills": [], "expired": True,
        })
    reply = responder.compose_response_v2(inp)
    assert forbidden not in reply, (
        f"Policy gate failed to reject {forbidden!r} on {move} -- "
        f"got reply: {reply!r}"
    )


def test_fallback_recommend_with_mixed_valid_and_malformed_uses_valid_only() -> None:
    """When the list has SOME valid recommendations and some malformed
    ones, render only the valid ones (and the closing question)."""
    inp = _input("recommend_adjacent_roles",
                 adjacent_recommendations_payload={
                     "recommendations": [
                         None,                                      # type: ignore[list-item]
                         {"title": "Welder", "employer": "ACME",
                          "evidence_summary": "3 of 5"},
                         "garbage",                                  # type: ignore[list-item]
                         {"title": "Forklift Operator"},
                     ],
                     "total_retrieved": 4,
                 })
    reply = _recommend_adjacent_roles_fallback_v2(*_iv(inp))
    assert "Welder at ACME" in reply
    assert "Forklift Operator" in reply
    assert "Want me to look closer" in reply
    # The empty-result line should NOT appear -- we have valid recs.
    assert "not seeing other roles" not in reply.lower()


def test_policy_v2_allows_approved_framing_on_adjacency_moves(monkeypatch) -> None:
    """Sanity: "roles worth exploring" / "where some of your existing
    skills transfer" must pass the policy check."""
    from skillbridge.chat import responder

    approved = (
        "Here are a few roles worth exploring where some of your "
        "existing skills transfer."
    )
    monkeypatch.setattr(responder, "is_enabled", lambda: True)
    monkeypatch.setattr(
        responder, "call",
        lambda system, user, max_tokens=500: approved,
    )

    inp = _input("recommend_adjacent_roles",
                 adjacent_recommendations_payload={
                     "recommendations": [
                         {"title": "Welder", "employer": "ACME",
                          "evidence_summary": "3 of 5"},
                     ],
                     "total_retrieved": 5,
                 })
    reply = responder.compose_response_v2(inp)
    assert "roles worth exploring" in reply


def test_run_engine_returns_payload_with_recommendation_shape(monkeypatch) -> None:
    """End-to-end: the engine helper persists the snapshot AND
    returns the responder payload. Locked shape per v11."""
    from skillbridge.chat import handler

    # Stub the engine pipeline so no DB is hit.
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

    sp = StagedProfile.new("s")
    sp.skills = [
        StagedSkill(skill_name="welding", source="resume", confidence=0.8),
        StagedSkill(skill_name="blueprint reading", source="resume", confidence=0.8),
        StagedSkill(skill_name="forklift operation", source="resume", confidence=0.8),
    ]
    payload = handler._run_adjacency_engine_and_persist(sp)

    # Payload shape (locked v11):
    assert set(payload.keys()) >= {
        "recommendations",
        "total_retrieved",
        "total_dropped_by_credential_gap",
        "total_dropped_by_coverage_floor",
        "total_dropped_by_transferable_floor",
        "total_dropped_by_no_required_non_credential_skills",
    }
    assert payload["recommendations"] == []
    assert payload["total_retrieved"] == 0
    # Snapshot persisted with empty items list.
    assert sp.last_adjacent_snapshot["items"] == []
    assert sp.last_adjacent_snapshot["created_message_count"] == sp.message_count
