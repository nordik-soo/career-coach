"""AR-6b tests: handler-side soft-offer append + flag setter +
reoffer suppression.

Covers (per docs/adjacent-recommendations-design.md v11):
  - Soft offer fires on `present_matches` with credential-only cap +
    evidence: reply ends with `_SOFT_OFFER_LINE`,
    `staged.pending_adjacent_offer = True`.
  - Soft offer fires on genuine `present_no_match` with evidence.
  - Suppressed when any other cap is present on lead result.
  - Suppressed when no usable evidence.
  - Suppressed when `pending_offer=True` at entry (reoffer
    suppression).
  - Suppressed when `_adjacency_enabled()` is False (flag OFF or
    cookie mode).
  - Never fires on non-match outcomes (`explain_gap`,
    `redirect_scope`, etc.).
  - Helper does NOT mutate `staged.pending_adjacent_offer` when the
    decision to emit is False.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.nodb


from skillbridge.chat.arbiter import ArbiterDecision
from skillbridge.match.adjacent import _SOFT_OFFER_LINE
from skillbridge.match.engine import MATCH
from skillbridge.session.staging import StagedProfile, StagedSkill


def _staged_with_evidence() -> StagedProfile:
    sp = StagedProfile.new("sess-1")
    sp.skills = [
        StagedSkill(skill_name="welding", source="resume", confidence=0.8),
        StagedSkill(skill_name="blueprint reading", source="resume", confidence=0.8),
        StagedSkill(skill_name="forklift operation", source="resume", confidence=0.8),
    ]
    return sp


def _staged_without_evidence() -> StagedProfile:
    return StagedProfile.new("sess-1")   # empty skills


def _decision(move: str, **kw) -> ArbiterDecision:
    return ArbiterDecision(
        final_move=move,
        reason_code=kw.get("reason_code", "x"),
        tone=kw.get("tone", "brief_confident"),
        arbiter_action=kw.get("arbiter_action", "passed_planner_through"),
        ask_slot=kw.get("ask_slot"),
        caps_applied=kw.get("caps_applied", ()),
    )


def _credential_capped_lead_result() -> dict:
    """Build a lead-result dict matching the engine.py:1042+ shape.
    Pre-cap band == good + sole cap == band_capped_by_credential."""
    return {
        "score_explanation": {
            "score_components": {
                "score_pre_caps": MATCH.band_good + 0.01,
            },
            "caps_applied": ["band_capped_by_credential"],
        },
    }


# =========================================================================
# Soft offer fires on credential-capped present_matches with evidence
# =========================================================================
def test_soft_offer_fires_on_credential_capped_matches(monkeypatch) -> None:
    from skillbridge.chat import handler
    from skillbridge.match import adjacent

    monkeypatch.setattr(adjacent, "_adjacency_enabled", lambda: True)

    sp = _staged_with_evidence()
    reply = handler._maybe_append_soft_offer(
        reply="here are your matches.",
        staged=sp,
        final=_decision("present_matches"),
        results=[_credential_capped_lead_result()],
        pending_offer=False,
    )
    assert reply.endswith(_SOFT_OFFER_LINE)
    assert sp.pending_adjacent_offer is True


def test_soft_offer_fires_on_present_no_match_with_evidence(monkeypatch) -> None:
    from skillbridge.chat import handler
    from skillbridge.match import adjacent

    monkeypatch.setattr(adjacent, "_adjacency_enabled", lambda: True)

    sp = _staged_with_evidence()
    reply = handler._maybe_append_soft_offer(
        reply="I don't see one in today's postings.",
        staged=sp,
        final=_decision("present_no_match"),
        results=[],
        pending_offer=False,
    )
    assert reply.endswith(_SOFT_OFFER_LINE)
    assert sp.pending_adjacent_offer is True


# =========================================================================
# Suppression cases
# =========================================================================
def test_soft_offer_suppressed_when_non_credential_cap_on_lead(monkeypatch) -> None:
    """Lead match capped by something other than credential -> the
    `is_credential_only_band_cap` predicate returns False so the
    soft offer doesn't fire."""
    from skillbridge.chat import handler
    from skillbridge.match import adjacent

    monkeypatch.setattr(adjacent, "_adjacency_enabled", lambda: True)
    sp = _staged_with_evidence()

    lead = {
        "score_explanation": {
            "score_components": {"score_pre_caps": MATCH.band_good + 0.01},
            "caps_applied": ["band_capped_by_no_experience"],
        },
    }
    reply = handler._maybe_append_soft_offer(
        reply="here are your matches.",
        staged=sp,
        final=_decision("present_matches"),
        results=[lead],
        pending_offer=False,
    )
    assert reply == "here are your matches."
    assert sp.pending_adjacent_offer is False


def test_soft_offer_suppressed_when_no_evidence_on_no_match(monkeypatch) -> None:
    from skillbridge.chat import handler
    from skillbridge.match import adjacent

    monkeypatch.setattr(adjacent, "_adjacency_enabled", lambda: True)
    sp = _staged_without_evidence()

    reply = handler._maybe_append_soft_offer(
        reply="no match.",
        staged=sp,
        final=_decision("present_no_match"),
        results=[],
        pending_offer=False,
    )
    assert reply == "no match."
    assert sp.pending_adjacent_offer is False


def test_soft_offer_suppressed_when_no_evidence_on_matches(monkeypatch) -> None:
    from skillbridge.chat import handler
    from skillbridge.match import adjacent

    monkeypatch.setattr(adjacent, "_adjacency_enabled", lambda: True)
    sp = _staged_without_evidence()

    reply = handler._maybe_append_soft_offer(
        reply="matches.",
        staged=sp,
        final=_decision("present_matches"),
        results=[_credential_capped_lead_result()],
        pending_offer=False,
    )
    assert reply == "matches."
    assert sp.pending_adjacent_offer is False


def test_soft_offer_suppressed_when_pending_offer_true(monkeypatch) -> None:
    """Reoffer suppression: if a soft offer was outstanding at handler
    entry, the consume-turn must NOT re-append the same offer even if
    the outcome conditions still hold."""
    from skillbridge.chat import handler
    from skillbridge.match import adjacent

    monkeypatch.setattr(adjacent, "_adjacency_enabled", lambda: True)
    sp = _staged_with_evidence()

    reply = handler._maybe_append_soft_offer(
        reply="here are your matches.",
        staged=sp,
        final=_decision("present_matches"),
        results=[_credential_capped_lead_result()],
        pending_offer=True,   # entry-time flag was True
    )
    assert reply == "here are your matches."
    assert sp.pending_adjacent_offer is False


def test_soft_offer_suppressed_when_adjacency_gate_off(monkeypatch) -> None:
    """Flag OFF (or cookie mode) -> no soft offer."""
    from skillbridge.chat import handler
    from skillbridge.match import adjacent

    monkeypatch.setattr(adjacent, "_adjacency_enabled", lambda: False)
    sp = _staged_with_evidence()

    reply = handler._maybe_append_soft_offer(
        reply="matches.",
        staged=sp,
        final=_decision("present_matches"),
        results=[_credential_capped_lead_result()],
        pending_offer=False,
    )
    assert reply == "matches."
    assert sp.pending_adjacent_offer is False


@pytest.mark.parametrize("move", [
    "explain_gap", "redirect_scope", "ask_one_clarifying_question",
    "offer_refinement", "acknowledge_and_continue", "present_near_miss",
    "confirm_resume_summary", "explain_remaining_gaps",
    "recommend_adjacent_roles", "describe_adjacent_role",
])
def test_soft_offer_never_fires_on_non_match_outcomes(monkeypatch, move) -> None:
    """The soft offer is specifically tied to `present_matches` and
    `present_no_match`. Every other outcome -- including the
    adjacency-synthesized ones -- must NOT trigger it."""
    from skillbridge.chat import handler
    from skillbridge.match import adjacent

    monkeypatch.setattr(adjacent, "_adjacency_enabled", lambda: True)
    sp = _staged_with_evidence()

    reply = handler._maybe_append_soft_offer(
        reply="response.",
        staged=sp,
        final=_decision(move),
        results=[_credential_capped_lead_result()],
        pending_offer=False,
    )
    assert reply == "response."
    assert sp.pending_adjacent_offer is False


def test_soft_offer_suppressed_when_present_matches_has_empty_results(monkeypatch) -> None:
    """`present_matches` with empty results is a weird shape (the
    arbiter shouldn't emit it) but defensive: lead_result is None →
    no offer."""
    from skillbridge.chat import handler
    from skillbridge.match import adjacent

    monkeypatch.setattr(adjacent, "_adjacency_enabled", lambda: True)
    sp = _staged_with_evidence()

    reply = handler._maybe_append_soft_offer(
        reply="matches.",
        staged=sp,
        final=_decision("present_matches"),
        results=[],
        pending_offer=False,
    )
    assert reply == "matches."
    assert sp.pending_adjacent_offer is False


# =========================================================================
# Placement: helper called between compose_response_v2 and staged.touch()
# =========================================================================
def test_soft_offer_helper_called_in_try_v2_path() -> None:
    """Static audit: the helper is wired into _try_v2_path, ABOVE
    `staged.touch()` and BELOW `compose_response_v2`. That ordering
    is the locked persistence contract -- the flag-set must happen
    before the save."""
    import inspect

    from skillbridge.chat import handler

    src = inspect.getsource(handler._try_v2_path)
    # The helper call must appear.
    assert "_maybe_append_soft_offer(" in src

    # Ordering: compose_response_v2 -> _maybe_append_soft_offer ->
    # staged.touch().
    compose_idx = src.find("compose_response_v2(")
    soft_idx = src.find("_maybe_append_soft_offer(")
    touch_idx = src.rfind("staged.touch()")
    assert 0 < compose_idx < soft_idx < touch_idx, (
        f"Soft-offer helper out of order. "
        f"compose={compose_idx} soft={soft_idx} touch={touch_idx}"
    )
