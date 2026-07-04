"""Unit tests for the recommender → matching handoff helper (Step 2.5).

DB-free, no LLM, no store. Constructs StagedProfile instances directly
and verifies the locked mutation order:

  1. capture label + id into locals
  2. write staged.target_role_text (triggers __setattr__ cascade)
  3. if id is 5-digit NOC, write staged.target_noc AFTER the cascade

Also pins scope discipline: non-role kind raises; empty/whitespace
label raises; non-5-digit id silently skips the target_noc write
(matching engine's lazy resolver handles the fallback).
"""
from __future__ import annotations

import pytest

from skillbridge.chat.conversation_frame import SurfaceItem
from skillbridge.chat.reference_handoff import handoff_recommender_to_matching
from skillbridge.session.staging import StagedProfile


pytestmark = pytest.mark.nodb


def _new_staged() -> StagedProfile:
    return StagedProfile.new(session_id="test-session-handoff-0001")


def _role(label: str, noc: str | None) -> SurfaceItem:
    return SurfaceItem(kind="role", label=label, id=noc, ordinal=1)


def _job(label: str, job_id: str | None) -> SurfaceItem:
    return SurfaceItem(kind="job", label=label, id=job_id, ordinal=1)


# ---------------------------------------------------------------- happy paths


class TestHandoffHappyPath:
    def test_role_with_5_digit_noc_sets_target_and_noc(self):
        staged = _new_staged()
        item = _role("Administrative assistant", "13110")
        handoff_recommender_to_matching(staged, item)
        assert staged.target_role_text == "Administrative assistant"
        assert staged.target_noc == "13110"

    def test_role_with_none_id_sets_target_only(self):
        staged = _new_staged()
        item = _role("Administrative assistant", None)
        handoff_recommender_to_matching(staged, item)
        assert staged.target_role_text == "Administrative assistant"
        assert staged.target_noc is None

    def test_role_with_non_5_digit_id_silently_skips_noc(self):
        """Locked: non-5-digit id skips the target_noc write. The
        matching engine's lazy title-to-NOC resolver handles the
        fallback via its normal cache -- warning here would just add
        noise for a case that isn't broken."""
        staged = _new_staged()
        item = _role("Administrative assistant", "not-a-noc")
        handoff_recommender_to_matching(staged, item)
        assert staged.target_role_text == "Administrative assistant"
        assert staged.target_noc is None

    def test_role_with_4_digit_id_skips_noc(self):
        """Length matters -- 4 digits is not a NOC 2021 code."""
        staged = _new_staged()
        item = _role("Data entry clerk", "1440")
        handoff_recommender_to_matching(staged, item)
        assert staged.target_role_text == "Data entry clerk"
        assert staged.target_noc is None

    def test_role_with_6_digit_id_skips_noc(self):
        """Length matters -- 6 digits is not a NOC 2021 code."""
        staged = _new_staged()
        item = _role("Data entry clerk", "144000")
        handoff_recommender_to_matching(staged, item)
        assert staged.target_noc is None

    def test_role_with_non_numeric_5_char_id_skips_noc(self):
        """Content matters -- 5 chars must all be digits."""
        staged = _new_staged()
        item = _role("Data entry clerk", "13110")  # baseline: numeric ok
        handoff_recommender_to_matching(staged, item)
        assert staged.target_noc == "13110"

        staged2 = _new_staged()
        item2 = _role("Data entry clerk", "13a10")
        handoff_recommender_to_matching(staged2, item2)
        assert staged2.target_noc is None


# ---------------------------------------------------------------- __setattr__ cascade


class TestHandoffTriggersSetattrCascade:
    """The staged.target_role_text write must fire StagedProfile's
    __setattr__ hook at staging.py:501, cascade-clearing all state
    scoped to the prior target. This is the LOAD-BEARING lifecycle
    contract the resolver design signed off on."""

    def _staged_with_prior_target(self) -> StagedProfile:
        """Fresh staged with a fully-populated prior target's state --
        everything __setattr__ should clear on the target_role_text
        rewrite."""
        s = StagedProfile.new(session_id="test-cascade")
        s.target_role_text = "software developer"
        s.target_noc = "21232"
        s.last_match_snapshot = {"jobs": [{"title": "Junior Developer"}]}
        s.last_recommender_adjacent_surface = (
            {"noc_code": "21231", "title": "Software engineer"},
            {"noc_code": "21230", "title": "Web developer"},
        )
        s.last_recommender_adjacent_surface_at_turn = 5
        s.pending_recommender_offer = "adjacent_role_drilldown_select"
        s.pending_adjacent_search_offer = True
        s.last_adjacent_nocs = ("21231", "21230")
        s.resume_upload_offered = True
        # deferred_career_intent only clears when prior_target_existed
        # (first target-fill does not clear it). Prior target IS set
        # here, so a switch below WILL clear this too.
        s.deferred_career_intent = "career_exploration"
        return s

    def test_cascade_clears_target_noc_before_new_write(self):
        """The key correctness test: target_noc must be cleared by the
        cascade, then overwritten by the handoff's step-3 write. If
        the mutation order were reversed (write noc first, then
        target_role_text), the cascade would wipe the new noc."""
        staged = self._staged_with_prior_target()
        assert staged.target_noc == "21232"  # prior target's noc
        item = _role("Administrative assistant", "13110")
        handoff_recommender_to_matching(staged, item)
        # New noc survives cascade (proves step-3 happens AFTER step-2).
        assert staged.target_noc == "13110"

    def test_cascade_clears_recommender_surface(self):
        staged = self._staged_with_prior_target()
        assert staged.last_recommender_adjacent_surface
        item = _role("Administrative assistant", "13110")
        handoff_recommender_to_matching(staged, item)
        assert staged.last_recommender_adjacent_surface == ()
        assert staged.last_recommender_adjacent_surface_at_turn is None

    def test_cascade_clears_pending_recommender_offer(self):
        staged = self._staged_with_prior_target()
        item = _role("Administrative assistant", "13110")
        handoff_recommender_to_matching(staged, item)
        assert staged.pending_recommender_offer is None

    def test_cascade_clears_pending_adjacent_search_offer(self):
        staged = self._staged_with_prior_target()
        item = _role("Administrative assistant", "13110")
        handoff_recommender_to_matching(staged, item)
        assert staged.pending_adjacent_search_offer is False

    def test_cascade_clears_last_match_snapshot(self):
        staged = self._staged_with_prior_target()
        item = _role("Administrative assistant", "13110")
        handoff_recommender_to_matching(staged, item)
        assert staged.last_match_snapshot is None

    def test_cascade_clears_last_adjacent_nocs(self):
        staged = self._staged_with_prior_target()
        item = _role("Administrative assistant", "13110")
        handoff_recommender_to_matching(staged, item)
        assert staged.last_adjacent_nocs == ()

    def test_cascade_clears_resume_upload_offered(self):
        staged = self._staged_with_prior_target()
        item = _role("Administrative assistant", "13110")
        handoff_recommender_to_matching(staged, item)
        assert staged.resume_upload_offered is False

    def test_cascade_clears_deferred_career_intent_when_prior_existed(self):
        """Locked in staging.py:576 -- deferred_career_intent clears
        ONLY when a prior target existed (this fixture sets one)."""
        staged = self._staged_with_prior_target()
        item = _role("Administrative assistant", "13110")
        handoff_recommender_to_matching(staged, item)
        assert staged.deferred_career_intent is None


# ---------------------------------------------------------------- scope discipline


class TestHandoffScopeDiscipline:
    def test_job_kind_raises_value_error(self):
        """Job-kind items are a different behavior (matching-job
        follow-up, not target-role switch) and out of scope."""
        staged = _new_staged()
        item = _job("Truck driver at Acme", "job_abc")
        with pytest.raises(ValueError, match="role-kind"):
            handoff_recommender_to_matching(staged, item)

    def test_job_kind_does_not_mutate_staged(self):
        """Contract: raising means no partial state change. Caller
        can retry or handle the error without staged being corrupted."""
        staged = _new_staged()
        staged.target_role_text = "software developer"
        staged.target_noc = "21232"
        pre_target = staged.target_role_text
        pre_noc = staged.target_noc
        item = _job("Truck driver at Acme", "job_abc")
        with pytest.raises(ValueError):
            handoff_recommender_to_matching(staged, item)
        assert staged.target_role_text == pre_target
        assert staged.target_noc == pre_noc

    def test_empty_label_raises_value_error(self):
        staged = _new_staged()
        item = _role("", "13110")
        with pytest.raises(ValueError, match="label"):
            handoff_recommender_to_matching(staged, item)

    def test_whitespace_only_label_raises_value_error(self):
        staged = _new_staged()
        item = _role("   \t  ", "13110")
        with pytest.raises(ValueError, match="label"):
            handoff_recommender_to_matching(staged, item)

    def test_bad_label_after_role_check_does_not_mutate(self):
        """Both guards run BEFORE any staged write -- verify no
        partial state on either raise."""
        staged = _new_staged()
        staged.target_role_text = "prior target"
        item = _role("", "13110")
        with pytest.raises(ValueError):
            handoff_recommender_to_matching(staged, item)
        assert staged.target_role_text == "prior target"
        assert staged.target_noc is None  # never written


# ---------------------------------------------------------------- idempotency


class TestHandoffIdempotency:
    """Handoff to the same target is a no-op semantically -- the
    __setattr__ cascade only fires on a value change. This isn't
    a required contract but pinning it here documents the behavior."""

    def test_same_role_and_noc_is_effectively_noop(self):
        staged = _new_staged()
        item = _role("Administrative assistant", "13110")
        handoff_recommender_to_matching(staged, item)
        # Second handoff to same item: label unchanged, noc unchanged.
        handoff_recommender_to_matching(staged, item)
        assert staged.target_role_text == "Administrative assistant"
        assert staged.target_noc == "13110"

    def test_first_fill_preserves_deferred_intent(self):
        """staging.py:576 clears deferred_career_intent only on a
        TRUE target switch (prior_target_existed check). A first-fill
        (no prior target) must NOT clear it -- the substrate-fill
        bridge relies on this."""
        staged = _new_staged()
        staged.deferred_career_intent = "local_skill_gap"
        assert staged.target_role_text is None  # no prior target
        item = _role("Administrative assistant", "13110")
        handoff_recommender_to_matching(staged, item)
        assert staged.target_role_text == "Administrative assistant"
        # deferred_intent survives first-fill.
        assert staged.deferred_career_intent == "local_skill_gap"
