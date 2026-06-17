"""AR-1a tests: adjacent-recommendations state slots.

Covers (per docs/adjacent-recommendations-design.md):
  - pending_adjacent_offer field default + sanitizer
  - last_adjacent_snapshot field default + sanitizer
  - presented_job_ids dict-key contract on last_match_snapshot
  - shift_adjacent_snapshot_ttl defensive state helper
  - target_role_text change clears last_adjacent_snapshot
  - cookie round-trip under 3800 bytes at full caps
  - handler save-and-clear hook at handle_anonymous entry:
      * pure blank input does NOT consume the flag
      * upload with blank message DOES consume the flag
      * non-blank message consumes the flag

Activation-deferral guarantee: nothing in main sets
pending_adjacent_offer in AR-1 (the SETTER lives in AR-6). These tests
manually pre-set the field on fixture profiles to exercise the
defensive-deserialize, sanitizer, and handler-hook behaviors.
"""
from __future__ import annotations

import pytest

# AR-1a tests are pure-Python: sanitizers, defensive-deserialize, the
# shift state helper, dataclass field defaults, and a handler-entry hook
# whose downstream calls are stubbed. None of them need a database; opt
# out of the autouse DB fixture so the suite doesn't drag in a Postgres
# connection that doesn't exist locally.
pytestmark = pytest.mark.nodb


from skillbridge.session.staging import (
    MAX_ADJACENT_ITEMS,
    MAX_EVIDENCE_CHARS,
    MAX_JOB_ID_CHARS,
    MAX_MATCHED_SKILLS,
    MAX_PRESENTED_JOB_IDS,
    MAX_SKILL_CHARS,
    MAX_TITLE_CHARS,
    StagedProfile,
    _sanitize_adjacent_snapshot,
    _sanitize_pending_adjacent_offer,
    _sanitize_snapshot,
    shift_adjacent_snapshot_ttl,
)


# ---------------------------------------------------------------- defaults
def test_pending_adjacent_offer_defaults_false() -> None:
    s = StagedProfile.new("sess-1")
    assert s.pending_adjacent_offer is False


def test_last_adjacent_snapshot_defaults_none() -> None:
    s = StagedProfile.new("sess-1")
    assert s.last_adjacent_snapshot is None


# ---------------------------------------------------------------- sanitize_pending_adjacent_offer
def test_sanitize_pending_adjacent_offer_only_true_survives() -> None:
    assert _sanitize_pending_adjacent_offer(True) is True
    assert _sanitize_pending_adjacent_offer(False) is False
    assert _sanitize_pending_adjacent_offer(None) is False
    assert _sanitize_pending_adjacent_offer("yes") is False
    assert _sanitize_pending_adjacent_offer(1) is False  # 1 != True identity
    assert _sanitize_pending_adjacent_offer({"x": 1}) is False
    assert _sanitize_pending_adjacent_offer([True]) is False


# ---------------------------------------------------------------- sanitize_adjacent_snapshot
def _full_snapshot() -> dict:
    return {
        "created_message_count": 7,
        "items": [
            {
                "job_id": "job-A",
                "title": "Welder",
                "evidence_summary": "3 of 5 required skills, 2 transferable",
                "why_adjacent": "same_noc_minor_group",
                "matched_skills": ["welding", "blueprint reading"],
            },
            {
                "job_id": "job-B",
                "title": "Fabricator",
                "evidence_summary": "2 of 4 required skills, 1 transferable",
                "why_adjacent": "skill_evidence",
                "matched_skills": ["welding"],
            },
        ],
    }


def test_sanitize_adjacent_snapshot_happy_path() -> None:
    out = _sanitize_adjacent_snapshot(_full_snapshot())
    assert out is not None
    assert out["created_message_count"] == 7
    assert len(out["items"]) == 2
    assert out["items"][0]["job_id"] == "job-A"
    assert out["items"][0]["why_adjacent"] == "same_noc_minor_group"
    assert out["items"][0]["matched_skills"] == ["welding", "blueprint reading"]


def test_sanitize_adjacent_snapshot_rejects_top_level_garbage() -> None:
    assert _sanitize_adjacent_snapshot(None) is None
    assert _sanitize_adjacent_snapshot("not a dict") is None
    assert _sanitize_adjacent_snapshot([{"created_message_count": 1, "items": []}]) is None


def test_sanitize_adjacent_snapshot_rejects_bad_created_count() -> None:
    assert _sanitize_adjacent_snapshot({"created_message_count": "7", "items": []}) is None
    assert _sanitize_adjacent_snapshot({"created_message_count": -1, "items": []}) is None
    assert _sanitize_adjacent_snapshot({"created_message_count": None, "items": []}) is None


def test_sanitize_adjacent_snapshot_rejects_non_list_items() -> None:
    assert _sanitize_adjacent_snapshot(
        {"created_message_count": 1, "items": {"x": "y"}}
    ) is None


def test_sanitize_adjacent_snapshot_caps_items() -> None:
    too_many = {
        "created_message_count": 1,
        "items": [
            {
                "job_id": f"j{i}",
                "title": f"role {i}",
                "evidence_summary": "",
                "why_adjacent": "same_noc_minor_group",
                "matched_skills": [],
            }
            for i in range(MAX_ADJACENT_ITEMS + 5)
        ],
    }
    out = _sanitize_adjacent_snapshot(too_many)
    assert out is not None
    assert len(out["items"]) == MAX_ADJACENT_ITEMS


def test_sanitize_adjacent_snapshot_drops_items_missing_required_strings() -> None:
    snap = {
        "created_message_count": 1,
        "items": [
            {"job_id": "", "title": "ok", "why_adjacent": "skill_evidence"},
            {"job_id": "ok", "title": "", "why_adjacent": "skill_evidence"},
            {"job_id": "ok-3", "title": "ok-3", "why_adjacent": "skill_evidence"},
        ],
    }
    out = _sanitize_adjacent_snapshot(snap)
    assert out is not None
    assert [it["job_id"] for it in out["items"]] == ["ok-3"]


def test_sanitize_adjacent_snapshot_unknown_why_coerced_to_empty() -> None:
    snap = {
        "created_message_count": 1,
        "items": [{
            "job_id": "j", "title": "t",
            "why_adjacent": "secret_future_enum_value",
        }],
    }
    out = _sanitize_adjacent_snapshot(snap)
    assert out is not None
    assert out["items"][0]["why_adjacent"] == ""


def test_sanitize_adjacent_snapshot_truncates_strings_and_caps_matched() -> None:
    long_title = "T" * (MAX_TITLE_CHARS + 50)
    long_job_id = "J" * (MAX_JOB_ID_CHARS + 50)
    long_evidence = "E" * (MAX_EVIDENCE_CHARS + 50)
    long_skill = "S" * (MAX_SKILL_CHARS + 50)
    snap = {
        "created_message_count": 1,
        "items": [{
            "job_id": long_job_id,
            "title": long_title,
            "evidence_summary": long_evidence,
            "why_adjacent": "skill_evidence",
            "matched_skills": [long_skill] * (MAX_MATCHED_SKILLS + 5),
        }],
    }
    out = _sanitize_adjacent_snapshot(snap)
    assert out is not None
    item = out["items"][0]
    assert len(item["job_id"]) == MAX_JOB_ID_CHARS
    assert len(item["title"]) == MAX_TITLE_CHARS
    assert len(item["evidence_summary"]) == MAX_EVIDENCE_CHARS
    assert len(item["matched_skills"]) == MAX_MATCHED_SKILLS
    assert all(len(s) == MAX_SKILL_CHARS for s in item["matched_skills"])


# ---------------------------------------------------------------- presented_job_ids on last_match_snapshot
def _base_match_snapshot() -> dict:
    return {
        "captured_at_turn": 3,
        "lead_job": {
            "job_id": "lead-1",
            "title": "Truck and Coach Technician",
            "employer": "ACME",
            "credential_gaps": [],
            "core_skill_gaps": [],
        },
        "other_jobs_meta": [],
    }


def test_sanitize_snapshot_accepts_presented_job_ids() -> None:
    snap = _base_match_snapshot()
    snap["presented_job_ids"] = ["job-A", "job-B", "job-C"]
    out = _sanitize_snapshot(snap)
    assert out is not None
    assert out["presented_job_ids"] == ["job-A", "job-B", "job-C"]


def test_sanitize_snapshot_dedupes_presented_job_ids() -> None:
    snap = _base_match_snapshot()
    snap["presented_job_ids"] = ["A", "B", "A", "C", "B"]
    out = _sanitize_snapshot(snap)
    assert out is not None
    assert out["presented_job_ids"] == ["A", "B", "C"]


def test_sanitize_snapshot_caps_presented_job_ids() -> None:
    snap = _base_match_snapshot()
    snap["presented_job_ids"] = [f"job-{i}" for i in range(MAX_PRESENTED_JOB_IDS + 10)]
    out = _sanitize_snapshot(snap)
    assert out is not None
    assert len(out["presented_job_ids"]) == MAX_PRESENTED_JOB_IDS


def test_sanitize_snapshot_drops_non_str_presented_entries() -> None:
    snap = _base_match_snapshot()
    snap["presented_job_ids"] = ["A", 7, None, {"job_id": "B"}, "C"]
    out = _sanitize_snapshot(snap)
    assert out is not None
    assert out["presented_job_ids"] == ["A", "C"]


def test_sanitize_snapshot_wrong_type_presented_yields_empty() -> None:
    snap = _base_match_snapshot()
    snap["presented_job_ids"] = "A,B,C"   # string instead of list
    out = _sanitize_snapshot(snap)
    assert out is not None
    assert out["presented_job_ids"] == []


def test_sanitize_snapshot_truncates_oversized_job_id() -> None:
    snap = _base_match_snapshot()
    snap["presented_job_ids"] = ["X" * (MAX_JOB_ID_CHARS + 100)]
    out = _sanitize_snapshot(snap)
    assert out is not None
    assert len(out["presented_job_ids"][0]) == MAX_JOB_ID_CHARS


# ---------------------------------------------------------------- target_role_text invalidation
def test_target_role_text_change_clears_last_adjacent_snapshot() -> None:
    s = StagedProfile.new("sess-1")
    s.target_role_text = "Truck and Coach Technician"
    s.last_adjacent_snapshot = _full_snapshot()
    assert s.last_adjacent_snapshot is not None
    s.target_role_text = "Class G Driver"   # role pivot
    assert s.last_adjacent_snapshot is None


def test_target_role_text_change_does_NOT_clear_pending_adjacent_offer() -> None:
    """pending_adjacent_offer is about UI state in flight; the one-turn
    handler-entry save-and-clear already bounds it. A role change on the
    SAME turn shouldn't double-invalidate."""
    s = StagedProfile.new("sess-1")
    s.target_role_text = "Truck and Coach Technician"
    s.pending_adjacent_offer = True
    s.target_role_text = "Class G Driver"
    assert s.pending_adjacent_offer is True


def test_target_role_text_setting_same_value_preserves_snapshot() -> None:
    s = StagedProfile.new("sess-1")
    s.target_role_text = "Truck and Coach Technician"
    s.last_adjacent_snapshot = _full_snapshot()
    s.target_role_text = "Truck and Coach Technician"  # no-op
    assert s.last_adjacent_snapshot is not None


# ---------------------------------------------------------------- shift_adjacent_snapshot_ttl
def test_shift_adjacent_snapshot_ttl_advances_by_one() -> None:
    s = StagedProfile.new("sess-1")
    s.last_adjacent_snapshot = _full_snapshot()
    before = s.last_adjacent_snapshot["created_message_count"]
    shift_adjacent_snapshot_ttl(s)
    assert s.last_adjacent_snapshot["created_message_count"] == before + 1


def test_shift_adjacent_snapshot_ttl_idempotent_across_two_calls() -> None:
    """Two scope-violation turns shift the counter twice (one per call).
    The third on-topic turn should still find the snapshot live."""
    s = StagedProfile.new("sess-1")
    s.last_adjacent_snapshot = _full_snapshot()
    shift_adjacent_snapshot_ttl(s)
    shift_adjacent_snapshot_ttl(s)
    assert s.last_adjacent_snapshot["created_message_count"] == 9   # 7 + 2


def test_shift_adjacent_snapshot_ttl_noop_when_no_snapshot() -> None:
    s = StagedProfile.new("sess-1")
    assert s.last_adjacent_snapshot is None
    shift_adjacent_snapshot_ttl(s)   # must not raise
    assert s.last_adjacent_snapshot is None


def test_shift_adjacent_snapshot_ttl_noop_when_malformed_created_count() -> None:
    s = StagedProfile.new("sess-1")
    s.last_adjacent_snapshot = {"created_message_count": "junk", "items": []}
    shift_adjacent_snapshot_ttl(s)
    # Defensive helper: bail silently rather than mutate malformed state.
    assert s.last_adjacent_snapshot["created_message_count"] == "junk"


def test_shift_adjacent_snapshot_ttl_noop_when_snapshot_not_dict() -> None:
    s = StagedProfile.new("sess-1")
    # Bypass the type-checked field assignment to install a corrupted
    # in-memory shape directly (simulates round-trip from a forged blob
    # that bypassed the sanitizer).
    s.__dict__["last_adjacent_snapshot"] = "not a dict"
    shift_adjacent_snapshot_ttl(s)   # must not raise
    assert s.last_adjacent_snapshot == "not a dict"


# ---------------------------------------------------------------- cookie round-trip
def _r1_only_worst_case_profile() -> StagedProfile:
    """Cookie-mode worst case: R-1 fields at cap, AR-1 fields at
    dataclass defaults. Mirrors the existing R-1 cookie fixture at
    test_chat_handler_v2.py:1683+. Under the AR-1 contract, adjacency
    activation is Redis-gated, so in cookie mode the AR-1 fields are
    never set -- this is the only realistic worst case the signed
    cookie ever sees."""
    from skillbridge.session.staging import (
        MAX_CANONICAL_CHARS, MAX_CRED_GAPS, MAX_EMPLOYER_CHARS,
        MAX_OTHER_JOBS, MAX_SKILL_GAPS,
    )
    s = StagedProfile.new("sess-r1-only")
    s.target_role_text = "warehouse worker"
    s.skills_text = "forklift, picking, packing, shipping, receiving"
    s.experience_text = "Three years at a Sault Ste. Marie distribution centre."
    s.education_text = "High school diploma."
    s.skills = []
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
    s.last_match_snapshot = {
        "captured_at_turn": 99,
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
        # NOT setting presented_job_ids -- in cookie mode adjacency is
        # Redis-gated, so the matcher never populates this key.
    }
    s.last_assumed_completed_credentials = [
        {"canonical": "x" * MAX_CANONICAL_CHARS, "mode": "hypothetical"}
        for _ in range(MAX_CRED_GAPS)
    ]
    s.last_discussed_credential_canonical = "x" * MAX_CANONICAL_CHARS
    s.pending_credential_confirmation = {
        "canonical": "x" * MAX_CANONICAL_CHARS, "action": "add",
    }
    # AR-1 fields deliberately UNTOUCHED -- they stay at dataclass
    # defaults (pending_adjacent_offer=False, last_adjacent_snapshot=None).
    return s


def _worst_case_profile() -> StagedProfile:
    """Redis-mode worst case: R-1 fields at cap AND AR-1 fields at cap.
    Used by the Redis-mode round-trip / preservation tests. Cookie mode
    NEVER sees this state because adjacency is gated to Redis."""
    from skillbridge.session.staging import (
        MAX_CANONICAL_CHARS, MAX_CRED_GAPS, MAX_EMPLOYER_CHARS,
        MAX_OTHER_JOBS, MAX_SKILL_GAPS,
    )
    s = StagedProfile.new("sess-roundtrip-worst-case")
    s.target_role_text = "warehouse worker"
    s.skills_text = "forklift, picking, packing, shipping, receiving"
    s.experience_text = "Three years at a Sault Ste. Marie distribution centre."
    s.education_text = "High school diploma."
    s.skills = []

    # Same realistic compact resume_facts_json the existing R-1 cookie
    # test uses (8 skills + 2 jobs + 1 cert + 1 lang).
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

    # R-1 last_match_snapshot at every cap, plus AR-1 presented_job_ids
    # at MAX_PRESENTED_JOB_IDS × MAX_JOB_ID_CHARS. Index-tag the prefix
    # so capped entries remain unique.
    s.last_match_snapshot = {
        "captured_at_turn": 99,
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

    # AR-1 adjacent snapshot at full item / field caps. matched_skills
    # index-tagged so capped entries stay distinct.
    s.last_adjacent_snapshot = {
        "created_message_count": 12,
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


def test_signed_cookie_under_3800_bytes_with_r1_at_cap_and_ar1_at_default(
    monkeypatch,
) -> None:
    """The authoritative gate: the SIGNED CookieSessionStore.save() value
    (NOT raw JSON) must stay under the 3800-byte ceiling. Leaves ~300
    bytes margin for Set-Cookie attributes (Path / HttpOnly / SameSite /
    Secure / Max-Age) under the browser's 4 KB per-cookie limit.

    AR-1 activation contract: adjacency is Redis-gated, so cookie mode
    NEVER sees non-default AR-1 fields. Worst case for the cookie store
    is therefore R-1 at cap + AR-1 at default. to_json's lossless
    minification drops the at-default AR-1 keys so they don't eat the
    19-byte headroom the existing R-1 test relies on."""
    monkeypatch.setenv("SESSION_COOKIE_SECRET", "x" * 48)
    monkeypatch.setenv("SESSION_TTL_MINUTES", "30")
    import importlib
    import config as _cfg
    importlib.reload(_cfg)
    from skillbridge.session import cookie_store as _cs_mod
    importlib.reload(_cs_mod)

    s = _r1_only_worst_case_profile()
    store = _cs_mod.CookieSessionStore(secret="x" * 48)
    signed_value = store.save(s)
    size = len(signed_value.encode("utf-8"))
    assert size < 3800, (
        f"signed session value is {size} bytes; ceiling is 3800 to "
        f"leave ~300 bytes margin for Set-Cookie attributes. The AR-1 "
        f"keys must be dropped from cookie payloads when they hold "
        f"their dataclass defaults."
    )


def test_cookie_payload_omits_default_adjacent_keys() -> None:
    """Adjacency activation is gated to Redis mode (AR-6 contract), so
    the AR-1 fields are ALWAYS at their dataclass defaults in cookie
    mode. to_json(redact_for_cookie=True) performs LOSSLESS JSON
    minification: drops the keys when they hold defaults so they don't
    eat ~60 bytes of the 3800-byte signed-cookie budget.

    This is NOT a value-bearing redaction. If a non-default value
    somehow appears in cookie mode (bug), it stays in the payload and
    the cookie-size test trips -- the bug surfaces instead of hiding."""
    import json as _json
    s = StagedProfile.new("sess-1")
    # All AR-1 fields at default; no last_match_snapshot.presented_job_ids.
    blob = s.to_json(redact_for_cookie=True)
    data = _json.loads(blob)
    assert "pending_adjacent_offer" not in data
    assert "last_adjacent_snapshot" not in data
    # Round-trip restores the defaults from the dataclass:
    s2 = StagedProfile.from_json(blob)
    assert s2.pending_adjacent_offer is False
    assert s2.last_adjacent_snapshot is None


def test_cookie_payload_drops_empty_presented_job_ids_from_snapshot() -> None:
    """When last_match_snapshot is present but presented_job_ids is
    empty, the empty list key is dropped from the cookie payload."""
    import json as _json
    s = StagedProfile.new("sess-1")
    s.last_match_snapshot = {
        "captured_at_turn": 1,
        "lead_job": {
            "job_id": "j1", "title": "t", "employer": "e",
            "credential_gaps": [], "core_skill_gaps": [],
        },
        "other_jobs_meta": [],
        "presented_job_ids": [],
    }
    blob = s.to_json(redact_for_cookie=True)
    data = _json.loads(blob)
    assert "presented_job_ids" not in data["last_match_snapshot"]
    # Sanitizer rebuilds the empty default on the way back in:
    s2 = StagedProfile.from_json(blob)
    assert s2.last_match_snapshot["presented_job_ids"] == []


def test_cookie_payload_preserves_non_default_values_when_set() -> None:
    """Sanity: if AR-1 fields hold non-default values (which shouldn't
    happen in cookie mode under the Redis-mode gate, but might via a
    bug), to_json does NOT silently drop them. The cookie-size test is
    the catch."""
    import json as _json
    s = StagedProfile.new("sess-1")
    s.pending_adjacent_offer = True   # non-default
    blob = s.to_json(redact_for_cookie=True)
    data = _json.loads(blob)
    assert data.get("pending_adjacent_offer") is True


def test_round_trip_preserves_adjacent_fields_in_redis_mode() -> None:
    """In Redis mode (redact_for_cookie=False) the full adjacency state
    survives round-trip -- ordinal follow-up + cross-turn job exclusion
    both work. This is the contract surface for Redis-mode adjacency."""
    s = _worst_case_profile()
    blob = s.to_json(redact_for_cookie=False)
    s2 = StagedProfile.from_json(blob)
    assert s2.pending_adjacent_offer is True
    assert s2.last_adjacent_snapshot is not None
    assert s2.last_adjacent_snapshot["created_message_count"] == 12
    assert len(s2.last_adjacent_snapshot["items"]) == MAX_ADJACENT_ITEMS
    assert s2.last_match_snapshot is not None
    # presented_job_ids are index-tagged so all entries remain unique
    # after the per-entry length cap.
    assert len(s2.last_match_snapshot["presented_job_ids"]) == MAX_PRESENTED_JOB_IDS


def test_minification_does_not_mutate_original_profile() -> None:
    """to_json(redact_for_cookie=True) builds the cookie payload from
    asdict(self), so the minification touches only the serialized copy.
    The in-memory StagedProfile must still carry the full state for
    same-turn render and downstream code."""
    s = _worst_case_profile()
    _ = s.to_json(redact_for_cookie=True)
    assert s.last_adjacent_snapshot is not None
    assert s.last_match_snapshot is not None
    assert len(s.last_match_snapshot["presented_job_ids"]) == MAX_PRESENTED_JOB_IDS


def test_cookie_round_trip_legacy_blob_without_adjacent_fields() -> None:
    """A pre-AR-1 cookie blob (no pending_adjacent_offer /
    last_adjacent_snapshot keys) must still load — dataclass defaults
    apply and no key error reaches the handler."""
    s = StagedProfile.new("sess-1")
    import json as _json
    data = _json.loads(s.to_json(redact_for_cookie=True))
    data.pop("pending_adjacent_offer", None)
    data.pop("last_adjacent_snapshot", None)
    s2 = StagedProfile.from_json(_json.dumps(data))
    assert s2.pending_adjacent_offer is False
    assert s2.last_adjacent_snapshot is None


def test_cookie_round_trip_forged_pending_offer_collapses_to_false() -> None:
    """A forged cookie that smuggles 1 or "yes" into pending_adjacent_offer
    must NOT trick the handler into believing an offer was issued."""
    s = StagedProfile.new("sess-1")
    import json as _json
    data = _json.loads(s.to_json(redact_for_cookie=True))
    for bad in (1, "yes", "true", [True], {"flag": True}):
        data["pending_adjacent_offer"] = bad
        s2 = StagedProfile.from_json(_json.dumps(data))
        assert s2.pending_adjacent_offer is False, (
            f"Forged value {bad!r} survived as truthy"
        )


# ---------------------------------------------------------------- handler save-and-clear hook
class _PreloadedStubStore:
    """In-memory session store seeded with a single pre-built
    StagedProfile. Mirrors the FakeStore pattern in test_chat_handler_v2
    but supports load() returning a real fixture so we can pre-set
    pending_adjacent_offer = True before handle_anonymous runs."""

    def __init__(self, sid: str, seed: StagedProfile | None = None):
        self._sid = sid
        self._held: StagedProfile | None = seed

    def new_session(self) -> str:
        return self._sid

    def load(self, session_id):
        if session_id == self._sid:
            return self._held
        return None

    def save(self, staged: StagedProfile) -> str:
        self._held = staged
        return staged.session_id or self._sid

    def delete(self, session_id):
        if session_id == self._sid:
            self._held = None


def _preloaded_profile_with_flag_true(sid: str) -> StagedProfile:
    sp = StagedProfile.new(sid)
    sp.message_count = 3
    sp.target_role_text = "Truck and Coach Technician"
    sp.pending_adjacent_offer = True
    return sp


def test_handler_pure_blank_input_does_not_consume_flag(monkeypatch) -> None:
    """Pure blank input returns BEFORE session load (handler.py:2332) and
    must NOT consume pending_adjacent_offer. Otherwise an idle blank turn
    would burn an offer the user never declined."""
    from skillbridge.chat import handler

    sid = "sess-blank-test"
    profile = _preloaded_profile_with_flag_true(sid)
    store = _PreloadedStubStore(sid, seed=profile)
    monkeypatch.setattr(handler, "get_store", lambda: store)

    resp = handler.handle_anonymous("   ", sid)
    assert isinstance(resp, dict)

    # The hook never ran because the handler short-circuited BEFORE
    # session load. The held profile is untouched.
    assert store.load(sid).pending_adjacent_offer is True


def test_handler_upload_with_blank_message_consumes_flag(monkeypatch) -> None:
    """A resume upload with an empty message bypasses the blank
    short-circuit (uploaded_file=True at handler.py:2331-2332). Per the
    v8 lock, this turn DOES reach session load and the save-and-clear
    hook fires."""
    from skillbridge.chat import handler

    sid = "sess-upload-blank"
    profile = _preloaded_profile_with_flag_true(sid)
    store = _PreloadedStubStore(sid, seed=profile)
    monkeypatch.setattr(handler, "get_store", lambda: store)

    # Stub the resume pipeline so it claims a parsed resume AND
    # _resume_facts_have_content fires; this routes the handler into the
    # resume_review short-circuit, which performs its own save() with the
    # post-hook profile state. Avoids dragging the v2 chain into a hook
    # placement test.
    def _stub_apply_resume_upload(staged, file_bytes, filename):
        staged.resume_facts_json = {"work_history": [{"role": "stub"}]}
        return {"parsed": True, "warning": None}
    monkeypatch.setattr(handler, "_apply_resume_upload", _stub_apply_resume_upload)
    monkeypatch.setattr(handler, "_resume_facts_have_content", lambda facts: True)
    # The resume_review path renders via compose_reply; stub it so we
    # don't need an LLM key.
    monkeypatch.setattr(handler, "compose_reply", lambda inp: "stub resume review")

    resp = handler.handle_anonymous(
        "", sid, file_bytes=b"%PDF-fake", filename="resume.pdf",
    )
    assert isinstance(resp, dict)
    assert store.load(sid).pending_adjacent_offer is False


def test_handler_non_blank_message_consumes_flag(monkeypatch) -> None:
    """A non-blank turn must consume the flag exactly once. The handler
    calls _extract (LLM extraction) BEFORE _try_v2_path (handler.py:2525),
    so we stub BOTH to keep the test focused on the save-and-clear hook
    and to avoid the LLM call hanging for 30+s waiting for an unavailable
    API."""
    from skillbridge.chat import extractor as _extractor_mod
    from skillbridge.chat import handler

    sid = "sess-nonblank"
    profile = _preloaded_profile_with_flag_true(sid)
    store = _PreloadedStubStore(sid, seed=profile)
    monkeypatch.setattr(handler, "get_store", lambda: store)

    # Stub the LLM extractor so no network call is made. ExtractionResult
    # lives in skillbridge.chat.extractor (handler.py aliases it locally
    # as chat_extractor). Field shape: fields, skills, declined,
    # off_topic, raw_keys_dropped (extractor.py:121-127).
    empty_extraction = _extractor_mod.ExtractionResult(
        fields={}, skills=[], declined=[], off_topic=False,
        raw_keys_dropped=[],
    )
    monkeypatch.setattr(handler, "_extract", lambda message, *, asked_slots: empty_extraction)

    # _try_v2_path returns the response dict on the v2 happy path; None
    # signals fallback_to_legacy. The stub mirrors handler.py:996-997's
    # own touch+save so the post-hook profile actually persists.
    def _stub_try_v2_path(*, staged, message, **kwargs):
        staged.touch()
        new_sid = store.save(staged)
        return {"session_id": new_sid, "reply": "stub", "final_move": "ask"}
    monkeypatch.setattr(handler, "_try_v2_path", _stub_try_v2_path)

    resp = handler.handle_anonymous("hi there", sid)
    assert isinstance(resp, dict)
    assert store.load(sid).pending_adjacent_offer is False
