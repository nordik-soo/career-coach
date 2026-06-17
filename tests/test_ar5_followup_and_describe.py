"""AR-5 tests: ordinal/numeric/title-suffix resolver + describe render.

Covers (per docs/adjacent-recommendations-design.md v11):
  - `resolve_adjacent_followup`:
      * TTL gate (live only on +1 turn; cleared on +2)
      * ordinal patterns: first/1st/second/2nd/third/3rd (+/-"the", +/-"one")
      * numeric patterns: #1 / #2 / number 1 / item 2 / no. 3
      * title-suffix: distinctive token match (unique)
      * ambiguous → None
      * out-of-range → None
      * stale snapshot → None
      * defensive: malformed input doesn't crash
  - `render_describe_adjacent_role`:
      * fetches the live job by id; payload shape pinned
      * job expired → expired=True with deterministic fallback
      * snapshot evidence_summary + matched_skills carried through
      * defensive: malformed snapshot_item, missing job_id, non-str
        matched_skills entries
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.nodb


from skillbridge.chat.adjacent_followup import (
    _AMBIGUOUS,
    _match_numeric,
    _match_ordinal,
    _significant_title_tokens,
    render_describe_adjacent_role,
    resolve_adjacent_followup,
)


# =========================================================================
# Helpers
# =========================================================================
def _snap(items: list[dict], created_message_count: int = 5) -> dict:
    return {
        "created_message_count": created_message_count,
        "items": items,
    }


def _item(*, job_id: str = "job-1", title: str = "Welder",
          evidence_summary: str = "3 of 5 required skills",
          why_adjacent: str = "skill_evidence",
          matched_skills: list[str] | None = None) -> dict:
    return {
        "job_id": job_id,
        "title": title,
        "evidence_summary": evidence_summary,
        "why_adjacent": why_adjacent,
        "matched_skills": matched_skills or ["welding"],
    }


def _three_items() -> list[dict]:
    return [
        _item(job_id="j-1", title="Welder", matched_skills=["welding"]),
        _item(job_id="j-2", title="Forklift Operator",
              matched_skills=["forklift operation"]),
        _item(job_id="j-3", title="Truck Driver",
              matched_skills=["class g license"]),
    ]


# =========================================================================
# Ordinal resolver
# =========================================================================
@pytest.mark.parametrize("phrase,expected_idx", [
    ("the first one", 0),
    ("first", 0),
    ("the 1st", 0),
    ("1st one", 0),
    ("tell me about the second one", 1),
    ("the second", 1),
    ("2nd one", 1),
    ("third please", 2),
    ("the third one", 2),
    ("3rd", 2),
])
def test_ordinal_pattern_resolves_to_expected_item(phrase, expected_idx) -> None:
    items = _three_items()
    snap = _snap(items)
    result = resolve_adjacent_followup(phrase, snap, current_message_count=6)
    assert result is items[expected_idx]


def test_ordinal_out_of_range_returns_none() -> None:
    """Only two items but the user asked for the third."""
    items = _three_items()[:2]
    snap = _snap(items)
    result = resolve_adjacent_followup("the third one", snap, 6)
    assert result is None


# =========================================================================
# Numeric resolver
# =========================================================================
@pytest.mark.parametrize("phrase,expected_idx", [
    ("#1", 0),
    ("# 1", 0),
    ("number 1", 0),
    ("item 1", 0),
    ("no. 1", 0),
    ("no 2 looks good", 1),
    ("tell me about #3", 2),
    ("number 3", 2),
])
def test_numeric_pattern_resolves_to_expected_item(phrase, expected_idx) -> None:
    items = _three_items()
    snap = _snap(items)
    result = resolve_adjacent_followup(phrase, snap, 6)
    assert result is items[expected_idx]


def test_numeric_out_of_range_returns_none() -> None:
    items = _three_items()
    snap = _snap(items)
    result = resolve_adjacent_followup("#7 please", snap, 6)
    assert result is None


def test_numeric_zero_returns_none() -> None:
    items = _three_items()
    snap = _snap(items)
    # "#0" makes no sense; 1-indexed.
    assert resolve_adjacent_followup("#0", snap, 6) is None


# --- AR-5 round-3: invalid numeric must NOT silently yield to title ---
def test_invalid_numeric_short_circuits_title_path() -> None:
    """The user explicitly wrote "#0" alongside a title token. The
    invalid numeric reference is an explicit-but-malformed intent;
    the title path must NOT silently override and pick "welder"."""
    items = _three_items()
    snap = _snap(items)
    assert resolve_adjacent_followup("welder #0", snap, 6) is None


def test_invalid_zero_numeric_with_no_dot_short_circuits_title_path() -> None:
    """`_NUMERIC_PATTERN` captures the (optional) sign + digits and
    `_match_numeric` rejects any value < 1 via `_AMBIGUOUS`. The
    legacy 'no. <digit>' surface form still routes through correctly
    -- "welder no. 0" surfaces as an explicit-invalid numeric, not
    as a missing-numeric that lets the title path take over."""
    items = _three_items()
    snap = _snap(items)
    assert resolve_adjacent_followup("welder no. 0", snap, 6) is None


def test_match_numeric_zero_returns_ambiguous_sentinel() -> None:
    """Direct unit test: `#0` is invalid (1-indexed) and must produce
    the same sentinel as "multiple distinct refs" so the resolver
    short-circuits."""
    assert _match_numeric("#0") is _AMBIGUOUS


# --- AR-5 round-4: actual negative numerics (signed integers) ---
@pytest.mark.parametrize("phrase", [
    "#-1",
    "# -1",
    "number -1",
    "item -1",
    "no. -1",
    "no -1",
    "#-7",
])
def test_match_numeric_negative_returns_ambiguous_sentinel(phrase) -> None:
    """The regex must capture the minus sign so genuinely negative
    references surface as explicit-but-invalid. Otherwise the title
    path silently overrides."""
    assert _match_numeric(phrase) is _AMBIGUOUS


@pytest.mark.parametrize("phrase", [
    "welder #-1",
    "welder number -1",
    "welder no. -1",
    "welder no -1",
    "the welder #-1",
])
def test_negative_numeric_short_circuits_title_path(phrase) -> None:
    """The reviewer's case: a negative numeric reference paired with
    a title token must NOT resolve via title alone -- the explicit-
    but-invalid numeric flagged as `_AMBIGUOUS` and the resolver
    short-circuits to None."""
    items = _three_items()
    snap = _snap(items)
    assert resolve_adjacent_followup(phrase, snap, 6) is None


# =========================================================================
# Title-suffix resolver
# =========================================================================
@pytest.mark.parametrize("phrase,expected_idx", [
    ("the welder role", 0),
    ("tell me about welder", 0),
    ("more about the welder one", 0),
    ("forklift operator looks good", 1),
    ("the truck driver", 2),
])
def test_title_suffix_matches_unique_item(phrase, expected_idx) -> None:
    items = _three_items()
    snap = _snap(items)
    result = resolve_adjacent_followup(phrase, snap, 6)
    assert result is items[expected_idx], (
        f"phrase={phrase!r} expected {items[expected_idx]['title']!r} "
        f"got {result}"
    )


def test_title_suffix_ambiguous_returns_none() -> None:
    """Two items share the same distinctive token ('welder'). The
    resolver returns None so the planner asks for clarification."""
    items = [
        _item(job_id="j-1", title="Welder I"),
        _item(job_id="j-2", title="Welder II"),
    ]
    snap = _snap(items)
    result = resolve_adjacent_followup("the welder role", snap, 6)
    assert result is None


def test_title_suffix_no_match_returns_none() -> None:
    items = _three_items()
    snap = _snap(items)
    result = resolve_adjacent_followup("the dental hygienist one", snap, 6)
    assert result is None


def test_title_suffix_stopwords_dont_collide() -> None:
    """The user says 'the role'; the stopword filter strips 'role'
    so it's not a distinctive token matching every item."""
    items = _three_items()
    snap = _snap(items)
    result = resolve_adjacent_followup("the role please", snap, 6)
    assert result is None


# =========================================================================
# TTL behavior
# =========================================================================
def test_ttl_live_on_plus_one_turn() -> None:
    items = _three_items()
    snap = _snap(items, created_message_count=5)
    assert resolve_adjacent_followup("the second one", snap, current_message_count=6) is items[1]


def test_ttl_dead_on_plus_two_turns() -> None:
    """The snapshot is gone after one turn."""
    items = _three_items()
    snap = _snap(items, created_message_count=5)
    assert resolve_adjacent_followup("the second one", snap, current_message_count=7) is None


def test_ttl_dead_on_same_turn() -> None:
    """current == created -- the snapshot just got persisted; we
    haven't moved to the next turn yet."""
    items = _three_items()
    snap = _snap(items, created_message_count=5)
    assert resolve_adjacent_followup("the second one", snap, current_message_count=5) is None


def test_ttl_dead_on_backward_turn() -> None:
    items = _three_items()
    snap = _snap(items, created_message_count=5)
    assert resolve_adjacent_followup("the second one", snap, current_message_count=4) is None


# --- AR-5 round-3: negative TTL counts must NOT resolve ---
def test_ttl_rejects_negative_created_message_count() -> None:
    """A forged blob with `created_message_count=-1` and a real
    `current_message_count=0` would satisfy -1 + 1 == 0 and resolve
    item 1. The resolver enforces `created >= 0`, matching the
    sanitizer contract."""
    items = _three_items()
    snap = {"created_message_count": -1, "items": items}
    assert resolve_adjacent_followup("the first one", snap, current_message_count=0) is None


def test_ttl_rejects_negative_current_message_count() -> None:
    items = _three_items()
    snap = _snap(items, created_message_count=5)
    assert resolve_adjacent_followup("the first one", snap, current_message_count=-1) is None


def test_shift_adjacent_snapshot_ttl_rejects_negative_created_count() -> None:
    """The shifter mirrors the sanitizer/resolver contract; a forged
    snapshot with `created=-1` is left untouched, NOT promoted to 0."""
    from skillbridge.session.staging import (
        StagedProfile,
        shift_adjacent_snapshot_ttl,
    )

    sp = StagedProfile.new("sess-1")
    sp.__dict__["last_adjacent_snapshot"] = {
        "created_message_count": -1,
        "items": [],
    }
    shift_adjacent_snapshot_ttl(sp)
    assert sp.last_adjacent_snapshot["created_message_count"] == -1


# =========================================================================
# Defensive boundaries
# =========================================================================
def test_none_snap_returns_none() -> None:
    assert resolve_adjacent_followup("the second", None, 6) is None


def test_non_dict_snap_returns_none() -> None:
    assert resolve_adjacent_followup("the second", "garbage", 6) is None  # type: ignore[arg-type]


def test_missing_created_count_returns_none() -> None:
    snap = {"items": _three_items()}
    assert resolve_adjacent_followup("the second", snap, 6) is None


def test_non_int_created_count_returns_none() -> None:
    snap = {"created_message_count": "5", "items": _three_items()}
    assert resolve_adjacent_followup("the second", snap, 6) is None


def test_missing_items_returns_none() -> None:
    snap = {"created_message_count": 5}
    assert resolve_adjacent_followup("the second", snap, 6) is None


def test_empty_items_returns_none() -> None:
    snap = _snap([])
    assert resolve_adjacent_followup("the first", snap, 6) is None


def test_non_dict_item_at_index_returns_none() -> None:
    """A forged blob with a non-dict at the requested index → None."""
    snap = _snap([None, _item(job_id="j-2")])  # type: ignore[list-item]
    assert resolve_adjacent_followup("the first", snap, 6) is None
    # But the second item is a real dict and reachable.
    assert resolve_adjacent_followup("the second", snap, 6) is not None


def test_empty_message_returns_none() -> None:
    snap = _snap(_three_items())
    assert resolve_adjacent_followup("", snap, 6) is None
    assert resolve_adjacent_followup(None, snap, 6) is None


# =========================================================================
# AR-5 round-2: bool-as-int TTL rejection
# =========================================================================
def test_ttl_rejects_boolean_created_message_count() -> None:
    """`bool` is a subclass of `int`. A forged blob with
    `created_message_count=True` and a real `current_message_count=2`
    would coincidentally satisfy True + 1 == 2 and resolve item 1.
    The resolver must reject the boolean."""
    items = _three_items()
    snap = {"created_message_count": True, "items": items}
    result = resolve_adjacent_followup("the second one", snap, current_message_count=2)
    assert result is None


def test_ttl_rejects_boolean_false_created_message_count() -> None:
    items = _three_items()
    snap = {"created_message_count": False, "items": items}
    # False + 1 == 1; if accepted, current=1 would satisfy the TTL.
    result = resolve_adjacent_followup("the first one", snap, current_message_count=1)
    assert result is None


def test_ttl_rejects_boolean_current_message_count() -> None:
    items = _three_items()
    snap = _snap(items, created_message_count=0)
    result = resolve_adjacent_followup("the first one", snap, current_message_count=True)  # type: ignore[arg-type]
    assert result is None


def test_sanitize_adjacent_snapshot_rejects_boolean_created_count() -> None:
    """The AR-1a sanitizer must also reject booleans at the cookie
    boundary so a forged blob can't smuggle one in."""
    from skillbridge.session.staging import _sanitize_adjacent_snapshot

    snap = {
        "created_message_count": True,
        "items": [{"job_id": "j", "title": "t"}],
    }
    assert _sanitize_adjacent_snapshot(snap) is None
    snap["created_message_count"] = False
    assert _sanitize_adjacent_snapshot(snap) is None


def test_shift_adjacent_snapshot_ttl_rejects_boolean_created_count() -> None:
    """The shifter mustn't promote bool → bool+1 on a corrupted
    snapshot."""
    from skillbridge.session.staging import (
        StagedProfile,
        shift_adjacent_snapshot_ttl,
    )

    sp = StagedProfile.new("sess-1")
    sp.__dict__["last_adjacent_snapshot"] = {
        "created_message_count": True,
        "items": [],
    }
    shift_adjacent_snapshot_ttl(sp)
    # The boolean is left unchanged (defensive helper).
    assert sp.last_adjacent_snapshot["created_message_count"] is True


# =========================================================================
# AR-5 round-2: cross-format conflict detection
# =========================================================================
def test_conflict_ordinal_and_numeric_returns_none() -> None:
    """'the first one' AND '#2' point to different items -- the user
    is contradicting themselves; the resolver returns None so the
    planner asks for clarification."""
    items = _three_items()
    snap = _snap(items)
    result = resolve_adjacent_followup("the first one, #2", snap, 6)
    assert result is None


def test_conflict_ordinal_and_title_returns_none() -> None:
    """'the first one' is index 0 (Welder); 'forklift' is index 1.
    Conflict -> None."""
    items = _three_items()
    snap = _snap(items)
    result = resolve_adjacent_followup(
        "the first one, tell me about forklift", snap, 6,
    )
    assert result is None


def test_conflict_numeric_and_title_returns_none() -> None:
    """'welder' is index 0 (title); '#2' is index 1 (numeric).
    Conflict -> None."""
    items = _three_items()
    snap = _snap(items)
    result = resolve_adjacent_followup(
        "tell me about welder #2", snap, 6,
    )
    assert result is None


def test_agreement_ordinal_and_numeric_resolves() -> None:
    """'the first one' AND '#1' both point to index 0 -- the user
    is agreeing with themselves; the resolver picks that item."""
    items = _three_items()
    snap = _snap(items)
    result = resolve_adjacent_followup("the first one #1", snap, 6)
    assert result is items[0]


def test_agreement_ordinal_and_title_resolves() -> None:
    """'the first one' AND 'welder' both index 0 -> resolve."""
    items = _three_items()
    snap = _snap(items)
    result = resolve_adjacent_followup(
        "the first one, the welder role", snap, 6,
    )
    assert result is items[0]


def test_agreement_numeric_and_title_resolves() -> None:
    """'#1' AND 'welder' both index 0 -> resolve."""
    items = _three_items()
    snap = _snap(items)
    result = resolve_adjacent_followup(
        "tell me about welder #1", snap, 6,
    )
    assert result is items[0]


def test_title_ambiguity_resolved_by_ordinal() -> None:
    """Two items share the 'welder' token, but the ordinal narrows
    to the second one. Title's ambiguous set ∩ {1} = {1} -> resolve."""
    items = [
        _item(job_id="j-1", title="Welder I"),
        _item(job_id="j-2", title="Welder II"),
    ]
    snap = _snap(items)
    result = resolve_adjacent_followup(
        "the second welder one", snap, 6,
    )
    assert result is items[1]


# --- In-format ambiguity short-circuits even when ANOTHER path is precise ---
def test_in_format_ordinal_ambiguity_blocks_resolution() -> None:
    """'the first and second, #1' contains TWO ordinals within the
    ordinal format. Even though '#1' is precise, the user contradicted
    themselves in the ordinal path, so the resolver returns None for
    clarification."""
    items = _three_items()
    snap = _snap(items)
    result = resolve_adjacent_followup("the first and second, #1", snap, 6)
    assert result is None


def test_in_format_numeric_ambiguity_blocks_resolution() -> None:
    """'compare #1 and #2, the first one' has TWO numeric refs within
    the numeric format. Even though 'the first one' is precise, the
    resolver returns None."""
    items = _three_items()
    snap = _snap(items)
    result = resolve_adjacent_followup(
        "compare #1 and #2, the first one", snap, 6,
    )
    assert result is None


def test_in_format_ordinal_ambiguity_blocks_even_with_title_match() -> None:
    items = _three_items()
    snap = _snap(items)
    result = resolve_adjacent_followup(
        "the first and second, tell me about welder", snap, 6,
    )
    assert result is None


def test_title_ambiguity_with_no_other_disambiguator_returns_none() -> None:
    """Two title matches and no ordinal/numeric -> still None
    (existing contract preserved)."""
    items = [
        _item(job_id="j-1", title="Welder I"),
        _item(job_id="j-2", title="Welder II"),
    ]
    snap = _snap(items)
    result = resolve_adjacent_followup("the welder role", snap, 6)
    assert result is None


def test_smart_apostrophe_in_message_still_resolves() -> None:
    items = _three_items()
    snap = _snap(items)
    # U+2019 in "let's"; the normalizer folds it.
    result = resolve_adjacent_followup("let’s see #2", snap, 6)
    assert result is items[1]


# =========================================================================
# Direct helpers
# =========================================================================
def test_match_ordinal_returns_ambiguous_sentinel_on_multi() -> None:
    """If the message contains TWO ordinals ("first and second"),
    the helper returns the `_AMBIGUOUS` sentinel (NOT None) so the
    resolver short-circuits to None rather than letting another
    path's lone match win."""
    assert _match_ordinal("the first and the second") is _AMBIGUOUS


def test_match_ordinal_returns_none_on_absent() -> None:
    """`None` means "format absent" -- the path doesn't constrain."""
    assert _match_ordinal("show me jobs") is None


def test_match_numeric_returns_ambiguous_sentinel_on_multi() -> None:
    """Distinct numeric refs ('#1 and #2') -> _AMBIGUOUS sentinel."""
    assert _match_numeric("compare #1 and #2") is _AMBIGUOUS


def test_match_numeric_returns_none_on_absent() -> None:
    assert _match_numeric("show me jobs") is None


def test_match_numeric_repeated_same_value_is_not_ambiguous() -> None:
    """The user repeating "#2" twice isn't a contradiction; it
    resolves to 1 (zero-based)."""
    assert _match_numeric("#2 -- yes, the #2") == 1


def test_significant_title_tokens_strip_stopwords() -> None:
    assert _significant_title_tokens("The Welder Role") == {"welder"}
    assert _significant_title_tokens("Welder II — Custom Fab") == {
        "welder", "custom", "fab",
    }


def test_significant_title_tokens_non_string() -> None:
    assert _significant_title_tokens(None) == set()  # type: ignore[arg-type]
    assert _significant_title_tokens(7) == set()  # type: ignore[arg-type]


# =========================================================================
# render_describe_adjacent_role
# =========================================================================
class _FakeCursor:
    def __init__(self, row):
        self.row = row

    def execute(self, sql, params=()):
        self.last_sql = sql
        self.last_params = params

    def fetchone(self):
        return self.row


def _patch_fetch(monkeypatch, row):
    fake = _FakeCursor(row)

    class _Ctx:
        def __enter__(self_in):
            return fake
        def __exit__(self_in, *a):
            return False

    from skillbridge import db as db_mod
    monkeypatch.setattr(db_mod, "sync_cursor", lambda: _Ctx())
    return fake


def test_describe_render_returns_live_job_payload(monkeypatch) -> None:
    """The render path fetches the live job row and combines it with
    the snapshot's evidence + matched_skills."""
    live_row = {
        "job_id": "job-1",
        "title": "Welder",
        "employer": "ACME",
        "location": "Sault Ste. Marie, ON",
        "url": "https://example.com/posting/123",
        "posted_date": "2026-06-01",
    }
    _patch_fetch(monkeypatch, row=live_row)

    snapshot_item = _item(
        job_id="job-1", title="Welder",
        evidence_summary="3 of 5 required skills, 2 transferable",
        matched_skills=["welding", "blueprint reading"],
    )
    payload = render_describe_adjacent_role(snapshot_item)

    assert payload["expired"] is False
    assert payload["job"] is not None
    assert payload["job"]["title"] == "Welder"
    assert payload["job"]["employer"] == "ACME"
    assert payload["job"]["url"] == "https://example.com/posting/123"
    assert payload["evidence_summary"] == "3 of 5 required skills, 2 transferable"
    assert payload["matched_skills"] == ["welding", "blueprint reading"]


def test_describe_render_expired_when_no_row(monkeypatch) -> None:
    """Posting is no longer in core.v_current_job → expired=True."""
    _patch_fetch(monkeypatch, row=None)
    snapshot_item = _item(job_id="job-gone")
    payload = render_describe_adjacent_role(snapshot_item)
    assert payload["expired"] is True
    assert payload["job"] is None


def test_describe_render_evidence_survives_when_expired(monkeypatch) -> None:
    """Even when the job has expired, the evidence_summary and
    matched_skills from the snapshot still appear -- the responder
    may want to narrate them in the fallback line."""
    _patch_fetch(monkeypatch, row=None)
    snapshot_item = _item(
        job_id="job-gone",
        evidence_summary="3 of 5 required",
        matched_skills=["welding"],
    )
    payload = render_describe_adjacent_role(snapshot_item)
    assert payload["evidence_summary"] == "3 of 5 required"
    assert payload["matched_skills"] == ["welding"]
    assert payload["expired"] is True


def test_describe_render_handles_none_snapshot_item() -> None:
    payload = render_describe_adjacent_role(None)
    assert payload["expired"] is True
    assert payload["job"] is None
    assert payload["evidence_summary"] == ""
    assert payload["matched_skills"] == []


def test_describe_render_handles_non_dict_snapshot_item() -> None:
    payload = render_describe_adjacent_role("garbage")  # type: ignore[arg-type]
    assert payload["expired"] is True


def test_describe_render_handles_missing_job_id() -> None:
    """No job_id → can't fetch; expired=True with snapshot evidence
    still carried."""
    snapshot_item = {
        "title": "Welder",
        "evidence_summary": "evidence",
        "matched_skills": ["welding"],
    }
    payload = render_describe_adjacent_role(snapshot_item)
    assert payload["expired"] is True
    assert payload["evidence_summary"] == "evidence"
    assert payload["matched_skills"] == ["welding"]


def test_describe_render_filters_non_string_matched_skills(monkeypatch) -> None:
    """A forged snapshot blob with non-str entries in matched_skills
    is sanitized down to just the valid strings."""
    _patch_fetch(monkeypatch, row=None)
    snapshot_item = {
        "job_id": "job-1",
        "title": "Welder",
        "evidence_summary": "x",
        "matched_skills": ["welding", 7, None, "", "blueprint reading"],
    }
    payload = render_describe_adjacent_role(snapshot_item)
    assert payload["matched_skills"] == ["welding", "blueprint reading"]


def test_describe_render_query_targets_v_current_job(monkeypatch) -> None:
    """The fetch queries the active-jobs view, not the raw table.
    An expired posting (not in v_current_job) correctly drops out."""
    fake = _patch_fetch(monkeypatch, row={"job_id": "x"})
    render_describe_adjacent_role(_item(job_id="x"))
    assert "v_current_job" in fake.last_sql
    assert fake.last_params == ("x",)
