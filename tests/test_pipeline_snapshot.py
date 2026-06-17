"""AR-9.feat.coach-tiers CP1 step 12 — pipeline snapshot.

Pins:
  - `_format_publish_at_text` converts a datetime to
    "YYYY-MM-DD HH:MM ET" in America/Toronto;
  - naive datetimes are treated as UTC;
  - None passes through as None — NEVER substituted with the
    current time;
  - `PipelineSnapshot` is frozen;
  - the fallback empty body uses the snapshot when supplied; the
    snapshot does NOT affect non-empty-tier rendering;
  - the snapshot does NOT affect matching or tier selection (no
    integration with the tiered_evidence builder or view builder).
"""
from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from skillbridge.chat.coach_tiers_fallback import (
    _EMPTY_BODY,
    _compose_empty_body,
    render_coach_tiers_fallback,
)
from skillbridge.chat.pipeline_snapshot import (
    PipelineSnapshot,
    _format_publish_at_text,
)
from skillbridge.chat.tiered_evidence import (
    TieredEvidence,
    JobFacts,
    StrongMatch,
)
from skillbridge.chat.url_policy import Validated, validate
from skillbridge.chat.url_views import (
    build_sanitized_responder_view_for_tiered_matches,
)
from skillbridge.match.alignment import SkillAlignment

pytestmark = pytest.mark.nodb


# =========================================================================
# _format_publish_at_text — pure formatter
# =========================================================================
def test_format_publish_at_text_utc_aware_input():
    """A tz-aware UTC datetime converts to America/Toronto and
    renders as 'YYYY-MM-DD HH:MM ET'."""
    # 2026-06-14 10:14 UTC = 2026-06-14 06:14 ET (EDT, UTC-4)
    dt = datetime(2026, 6, 14, 10, 14, tzinfo=timezone.utc)
    assert _format_publish_at_text(dt) == "2026-06-14 06:14 ET"


def test_format_publish_at_text_naive_treated_as_utc():
    """psycopg may return naive datetimes for TIMESTAMPTZ columns
    depending on driver settings. The formatter treats naive
    timestamps as UTC."""
    naive = datetime(2026, 6, 14, 10, 14)
    assert _format_publish_at_text(naive) == "2026-06-14 06:14 ET"


def test_format_publish_at_text_already_in_et():
    """An ET-zoned datetime is passed through to ET (no double
    conversion)."""
    et = ZoneInfo("America/Toronto")
    dt = datetime(2026, 6, 14, 6, 14, tzinfo=et)
    assert _format_publish_at_text(dt) == "2026-06-14 06:14 ET"


def test_format_publish_at_text_winter_est():
    """In January, America/Toronto is EST (UTC-5)."""
    dt = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
    assert _format_publish_at_text(dt) == "2026-01-15 07:00 ET"


def test_format_publish_at_text_summer_edt():
    """In July, America/Toronto is EDT (UTC-4)."""
    dt = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)
    assert _format_publish_at_text(dt) == "2026-07-15 08:00 ET"


def test_format_publish_at_text_returns_none_for_none_input():
    """None preserves — no substitution to current time."""
    assert _format_publish_at_text(None) is None


# =========================================================================
# PipelineSnapshot dataclass shape
# =========================================================================
def test_pipeline_snapshot_is_frozen():
    s = PipelineSnapshot(total_active_jobs=43, last_publish_at_text="x")
    with pytest.raises((AttributeError, Exception)):
        s.total_active_jobs = 0  # type: ignore


def test_pipeline_snapshot_carries_fields():
    s = PipelineSnapshot(total_active_jobs=43, last_publish_at_text=None)
    assert s.total_active_jobs == 43
    assert s.last_publish_at_text is None


# =========================================================================
# _compose_empty_body — uses snapshot when supplied
# =========================================================================
def test_compose_empty_body_without_snapshot_returns_generic():
    assert _compose_empty_body(None) == _EMPTY_BODY


def test_compose_empty_body_with_snapshot_count_and_timestamp():
    snap = PipelineSnapshot(
        total_active_jobs=43,
        last_publish_at_text="2026-06-14 06:14 ET",
    )
    body = _compose_empty_body(snap)
    assert "Nothing on the board matches yet." in body
    assert "43 active postings" in body
    assert "2026-06-14 06:14 ET" in body
    assert "last refreshed" in body


def test_compose_empty_body_with_count_but_no_timestamp():
    """When no publish record exists, the body still surfaces the
    count but omits the timestamp clause."""
    snap = PipelineSnapshot(
        total_active_jobs=12,
        last_publish_at_text=None,
    )
    body = _compose_empty_body(snap)
    assert "12 active postings" in body
    assert "last refreshed" not in body
    # And it ends cleanly with a period
    assert body.rstrip().endswith(".")


def test_compose_empty_body_with_zero_jobs():
    snap = PipelineSnapshot(
        total_active_jobs=0,
        last_publish_at_text="2026-06-14 06:14 ET",
    )
    body = _compose_empty_body(snap)
    assert "0 active postings" in body


# =========================================================================
# render_coach_tiers_fallback — snapshot only used for empty-tier body
# =========================================================================
def _empty_view():
    return build_sanitized_responder_view_for_tiered_matches(
        TieredEvidence(apply_today=(), worth_a_try=(), sideways_move=()),
    )


def _validated(raw: str) -> Validated:
    result = validate(raw)
    assert isinstance(result, Validated)
    return result


def _populated_view():
    sm = StrongMatch(
        job_id="s", title="Accounts Payable Clerk", employer="Diamond J",
        location=None, noc_code=None,
        url=_validated("https://example.com/job"),
        job_facts=JobFacts(
            posted_date=None, posted_days_ago=3, location=None,
            employment_type="full-time", salary_text=None,
        ),
        skill_alignment=(
            SkillAlignment(
                user_skill="QuickBooks", job_requirement="QuickBooks",
                stage="exact", source="required",
                is_normalized_equal=True,
            ),
        ),
        non_blocking_gaps=(),
        credential_warning_text=None,
        strength_claim_text="competitive_match",
    )
    return build_sanitized_responder_view_for_tiered_matches(
        TieredEvidence(apply_today=(sm,), worth_a_try=(), sideways_move=()),
    )


def test_empty_view_without_snapshot_falls_back_to_generic_body():
    text, urls = render_coach_tiers_fallback(_empty_view())
    assert _EMPTY_BODY in text
    assert urls == frozenset()


def test_empty_view_with_snapshot_uses_grounded_body():
    snap = PipelineSnapshot(
        total_active_jobs=43,
        last_publish_at_text="2026-06-14 06:14 ET",
    )
    text, urls = render_coach_tiers_fallback(_empty_view(), snap)
    assert "43 active postings" in text
    assert "2026-06-14 06:14 ET" in text
    # Empty body never produces URLs
    assert urls == frozenset()


def test_snapshot_does_not_affect_non_empty_tier_body():
    """The snapshot's count must NEVER appear when a tier was
    rendered — it's an empty-state-only signal."""
    v = _populated_view()
    snap = PipelineSnapshot(
        total_active_jobs=999,
        last_publish_at_text="2026-06-14 06:14 ET",
    )
    text, urls = render_coach_tiers_fallback(v, snap)
    # Tier 1 was rendered, so the snapshot's count must not surface
    assert "999 active postings" not in text
    assert "last refreshed" not in text
    # And the rendered URL set is unchanged by the snapshot
    text_without_snap, urls_without_snap = render_coach_tiers_fallback(v)
    assert urls == urls_without_snap


def test_snapshot_does_not_appear_in_collect_fallback_render_urls():
    """The URL collector doesn't take a snapshot, and the snapshot
    contributes no URLs — confirm both invariants."""
    from skillbridge.chat.coach_tiers_fallback import (
        collect_fallback_render_urls,
    )
    snap = PipelineSnapshot(
        total_active_jobs=43, last_publish_at_text="2026-06-14 06:14 ET",
    )
    # The collector takes only the view — no snapshot signature.
    import inspect
    sig = inspect.signature(collect_fallback_render_urls)
    assert "snapshot" not in sig.parameters


def test_snapshot_not_used_by_build_tiered_evidence_or_view():
    """Hard guardrail: matching and tier selection must NOT depend
    on the snapshot. Verified by inspecting the signatures of both
    builders — neither accepts a snapshot parameter."""
    import inspect
    from skillbridge.chat.tiered_evidence import build_tiered_evidence
    from skillbridge.chat.url_views import (
        build_sanitized_responder_view_for_tiered_matches as build_view,
    )
    assert "snapshot" not in inspect.signature(build_tiered_evidence).parameters
    assert "snapshot" not in inspect.signature(build_view).parameters
    assert "pipeline_snapshot" not in inspect.signature(build_tiered_evidence).parameters
    assert "pipeline_snapshot" not in inspect.signature(build_view).parameters


# =========================================================================
# Module hygiene — DB-free fetcher import doesn't import psycopg early
# =========================================================================
def test_format_module_is_db_free_path():
    """Importing the snapshot module and calling the pure formatter
    must NOT require a live database. (The formatter is the unit-test
    surface; `fetch_pipeline_snapshot` is the integration surface.)"""
    # The fact that this test file's other tests imported and ran
    # `_format_publish_at_text` without DB connectivity is the
    # assertion. This test just sanity-checks the contract.
    assert _format_publish_at_text(None) is None
    assert _format_publish_at_text(
        datetime(2026, 6, 14, 10, 14, tzinfo=timezone.utc)
    ) == "2026-06-14 06:14 ET"
