"""Unit tests for the StagedProfile additions in R-1
(remaining-gaps iteration).

Two concerns:
  1. The four new fields (last_match_snapshot,
     last_assumed_completed_credentials,
     last_discussed_credential_canonical,
     pending_credential_confirmation) round-trip cleanly through
     to_json / from_json (cookie session store path).
  2. from_json applies per-field defensive validation per
     docs/remaining-gaps-design.md §R-1. Malformed shapes drop /
     default; unknown enum values drop the entry; lists cap.

No DB. No fixtures beyond an in-memory StagedProfile.
"""
from __future__ import annotations

import json

import pytest

from skillbridge.session.staging import (
    MAX_CANONICAL_CHARS,
    MAX_CRED_GAPS,
    MAX_OTHER_JOBS,
    MAX_SKILL_GAPS,
    StagedProfile,
)

pytestmark = pytest.mark.nodb


def _fresh() -> StagedProfile:
    return StagedProfile.new("test-session")


# ---------------------------------------------------------------- defaults
def test_new_staged_profile_has_remaining_gaps_defaults():
    sp = _fresh()
    assert sp.last_match_snapshot is None
    assert sp.last_assumed_completed_credentials == []
    assert sp.last_discussed_credential_canonical is None
    assert sp.pending_credential_confirmation is None


def test_default_factories_are_independent_per_instance():
    """field(default_factory=list) must not share state across instances."""
    a = _fresh()
    b = _fresh()
    a.last_assumed_completed_credentials.append(
        {"canonical": "x", "mode": "claimed"}
    )
    assert b.last_assumed_completed_credentials == []


# ----------------------------------------------- target_role_text -> clear
def test_changing_target_role_text_clears_all_four_remaining_gaps_fields():
    sp = _fresh()
    sp.last_match_snapshot = {"lead_job": {"title": "X"}}
    sp.last_assumed_completed_credentials = [
        {"canonical": "x", "mode": "claimed"},
    ]
    sp.last_discussed_credential_canonical = "x"
    sp.pending_credential_confirmation = {"canonical": "x", "action": "add"}

    sp.target_role_text = "different role"

    assert sp.last_match_snapshot is None
    assert sp.last_assumed_completed_credentials == []
    assert sp.last_discussed_credential_canonical is None
    assert sp.pending_credential_confirmation is None


def test_setting_target_role_text_to_same_value_preserves_remaining_gaps():
    sp = _fresh()
    sp.target_role_text = "warehouse worker"
    sp.last_match_snapshot = {"lead_job": {"title": "X"}}
    sp.last_assumed_completed_credentials = [
        {"canonical": "x", "mode": "claimed"},
    ]
    # Same value -> no clear (StagedProfile already guards target_noc this way).
    sp.target_role_text = "warehouse worker"
    assert sp.last_match_snapshot == {"lead_job": {"title": "X"}}
    assert sp.last_assumed_completed_credentials == [
        {"canonical": "x", "mode": "claimed"},
    ]


# ----------------------------------------------- round-trip serialization
def test_to_json_from_json_round_trip_preserves_remaining_gaps_fields():
    sp = _fresh()
    sp.last_match_snapshot = {
        "captured_at_turn": 3,
        "lead_job": {
            "job_id":   "honda-uuid",
            "title":    "310S Licensed Automotive Technician",
            "employer": "Great Lakes Honda",
            "credential_gaps": [
                {"display":   "310S Automotive Technician License",
                 "canonical": "310s automotive technician certification"},
                {"display":   "G2/G driver's license",
                 "canonical": "class g driver s license"},
            ],
            "core_skill_gaps": [
                "Honda vehicle experience",
                "dealership experience",
            ],
        },
        "other_jobs_meta": [
            {"job_id": "other-1", "title": "Truck Tech"},
        ][:MAX_OTHER_JOBS],   # MAX_OTHER_JOBS=0 in v1 -- the field is
                              # reserved for future job-pivot; v1 stores
                              # an empty list so roundtrip stays clean.
        # AR-1: the snapshot sanitizer defaults this list when absent,
        # so the original must include the empty form to round-trip
        # by dict-equality.
        "presented_job_ids": [],
    }
    sp.last_assumed_completed_credentials = [
        {"canonical": "310s automotive technician certification",
         "mode": "hypothetical"},
        {"canonical": "class g driver s license", "mode": "claimed"},
    ]
    sp.last_discussed_credential_canonical = "class g driver s license"
    sp.pending_credential_confirmation = {
        "canonical": "310s automotive technician certification",
        "action": "add",
    }

    blob = sp.to_json()
    restored = StagedProfile.from_json(blob)

    assert restored.last_match_snapshot == sp.last_match_snapshot
    assert restored.last_assumed_completed_credentials == \
        sp.last_assumed_completed_credentials
    assert restored.last_discussed_credential_canonical == \
        sp.last_discussed_credential_canonical
    assert restored.pending_credential_confirmation == \
        sp.pending_credential_confirmation


def test_round_trip_preserves_ordered_accumulation():
    """Ordered append-and-dedupe semantics depend on serialization order."""
    sp = _fresh()
    sp.last_assumed_completed_credentials = [
        {"canonical": f"cred_{i}", "mode": "claimed"} for i in range(5)
    ]
    restored = StagedProfile.from_json(sp.to_json())
    assert [
        c["canonical"] for c in restored.last_assumed_completed_credentials
    ] == [f"cred_{i}" for i in range(5)]


# ----------------------------------------------- defensive deserialization
def test_from_json_drops_non_dict_snapshot():
    blob = json.loads(_fresh().to_json())
    blob["last_match_snapshot"] = "not a dict"
    restored = StagedProfile.from_json(json.dumps(blob))
    assert restored.last_match_snapshot is None


def test_from_json_drops_snapshot_with_missing_lead_job():
    blob = json.loads(_fresh().to_json())
    blob["last_match_snapshot"] = {"captured_at_turn": 1, "lead_job": None}
    restored = StagedProfile.from_json(json.dumps(blob))
    assert restored.last_match_snapshot is None


def test_from_json_handles_credential_gaps_supplied_as_dict():
    """Round-10 R-1 review: a forged cookie that puts a dict where the
    schema expects a list MUST NOT crash from_json. Pre-fix code did
    `dict[:MAX_CRED_GAPS]` which raises KeyError because dict's
    __getitem__ doesn't accept slice keys. Correct behavior: treat the
    wrong-typed field as empty and continue."""
    blob = json.loads(_fresh().to_json())
    blob["last_match_snapshot"] = {
        "captured_at_turn": 1,
        "lead_job": {
            "job_id": "x", "title": "T", "employer": None,
            "credential_gaps": {"display": "x", "canonical": "y"},  # dict
            "core_skill_gaps": [],
        },
        "other_jobs_meta": [],
    }
    restored = StagedProfile.from_json(json.dumps(blob))
    assert restored.last_match_snapshot is not None
    assert restored.last_match_snapshot["lead_job"]["credential_gaps"] == []


def test_from_json_handles_core_skill_gaps_supplied_as_dict():
    blob = json.loads(_fresh().to_json())
    blob["last_match_snapshot"] = {
        "captured_at_turn": 1,
        "lead_job": {
            "job_id": "x", "title": "T", "employer": None,
            "credential_gaps": [],
            "core_skill_gaps": {"skill": "Honda experience"},  # dict, not list
        },
        "other_jobs_meta": [],
    }
    restored = StagedProfile.from_json(json.dumps(blob))
    assert restored.last_match_snapshot is not None
    assert restored.last_match_snapshot["lead_job"]["core_skill_gaps"] == []


def test_from_json_handles_other_jobs_meta_supplied_as_dict():
    blob = json.loads(_fresh().to_json())
    blob["last_match_snapshot"] = {
        "captured_at_turn": 1,
        "lead_job": {
            "job_id": "x", "title": "T", "employer": None,
            "credential_gaps": [],
            "core_skill_gaps": [],
        },
        "other_jobs_meta": {"job_id": "x", "title": "T"},  # dict, not list
    }
    restored = StagedProfile.from_json(json.dumps(blob))
    assert restored.last_match_snapshot is not None
    assert restored.last_match_snapshot["other_jobs_meta"] == []


def test_from_json_handles_snapshot_list_fields_supplied_as_strings():
    """Similar shape problem with a different wrong type. Strings ARE
    sliceable but iterating produces single characters, none of which
    are dicts -> empty result, no crash."""
    blob = json.loads(_fresh().to_json())
    blob["last_match_snapshot"] = {
        "captured_at_turn": 1,
        "lead_job": {
            "job_id": "x", "title": "T", "employer": None,
            "credential_gaps": "credential string",
            "core_skill_gaps": "skill string",
        },
        "other_jobs_meta": "meta string",
    }
    restored = StagedProfile.from_json(json.dumps(blob))
    assert restored.last_match_snapshot is not None
    lead = restored.last_match_snapshot["lead_job"]
    assert lead["credential_gaps"] == []
    assert lead["core_skill_gaps"] == []
    assert restored.last_match_snapshot["other_jobs_meta"] == []


def test_from_json_drops_credential_gap_entries_with_wrong_shape():
    blob = json.loads(_fresh().to_json())
    blob["last_match_snapshot"] = {
        "captured_at_turn": 1,
        "lead_job": {
            "job_id": "x", "title": "T", "employer": None,
            "credential_gaps": [
                {"display": "good", "canonical": "good-c"},
                "not-a-dict",
                {"display": 123, "canonical": "missing-display"},
                {"display": "missing-canonical"},
                {"display": "ok-second", "canonical": "ok-second-c"},
            ],
            "core_skill_gaps": [],
        },
        "other_jobs_meta": [],
    }
    restored = StagedProfile.from_json(json.dumps(blob))
    gaps = restored.last_match_snapshot["lead_job"]["credential_gaps"]
    assert [g["canonical"] for g in gaps] == ["good-c", "ok-second-c"]


def test_from_json_caps_credential_gaps_at_max():
    blob = json.loads(_fresh().to_json())
    blob["last_match_snapshot"] = {
        "captured_at_turn": 1,
        "lead_job": {
            "job_id": "x", "title": "T", "employer": None,
            "credential_gaps": [
                {"display": f"d{i}", "canonical": f"c{i}"}
                for i in range(10)
            ],
            "core_skill_gaps": [],
        },
        "other_jobs_meta": [],
    }
    restored = StagedProfile.from_json(json.dumps(blob))
    gaps = restored.last_match_snapshot["lead_job"]["credential_gaps"]
    assert len(gaps) == MAX_CRED_GAPS


def test_from_json_caps_core_skill_gaps_at_max():
    blob = json.loads(_fresh().to_json())
    blob["last_match_snapshot"] = {
        "captured_at_turn": 1,
        "lead_job": {
            "job_id": "x", "title": "T", "employer": None,
            "credential_gaps": [],
            "core_skill_gaps": [f"skill {i}" for i in range(10)],
        },
        "other_jobs_meta": [],
    }
    restored = StagedProfile.from_json(json.dumps(blob))
    assert len(restored.last_match_snapshot["lead_job"]["core_skill_gaps"]) == \
        MAX_SKILL_GAPS


def test_from_json_truncates_long_title_and_employer():
    blob = json.loads(_fresh().to_json())
    blob["last_match_snapshot"] = {
        "captured_at_turn": 1,
        "lead_job": {
            "job_id": "x",
            "title":  "Z" * 500,
            "employer": "E" * 500,
            "credential_gaps": [],
            "core_skill_gaps": [],
        },
        "other_jobs_meta": [],
    }
    restored = StagedProfile.from_json(json.dumps(blob))
    lead = restored.last_match_snapshot["lead_job"]
    assert len(lead["title"]) == 80
    assert len(lead["employer"]) == 60


def test_from_json_drops_accumulated_entries_with_unknown_mode():
    blob = json.loads(_fresh().to_json())
    blob["last_assumed_completed_credentials"] = [
        {"canonical": "x", "mode": "claimed"},
        {"canonical": "y", "mode": "future_extension"},
        {"canonical": "z", "mode": "hypothetical"},
    ]
    restored = StagedProfile.from_json(json.dumps(blob))
    canonicals = [c["canonical"] for c in restored.last_assumed_completed_credentials]
    assert canonicals == ["x", "z"]


def test_from_json_drops_accumulated_entries_with_non_string_canonical():
    blob = json.loads(_fresh().to_json())
    blob["last_assumed_completed_credentials"] = [
        {"canonical": "ok", "mode": "claimed"},
        {"canonical": 123,  "mode": "claimed"},
        {"canonical": "",   "mode": "claimed"},   # empty drops too
    ]
    restored = StagedProfile.from_json(json.dumps(blob))
    assert [c["canonical"] for c in restored.last_assumed_completed_credentials] \
        == ["ok"]


def test_from_json_drops_accumulated_when_not_a_list():
    blob = json.loads(_fresh().to_json())
    blob["last_assumed_completed_credentials"] = "not a list"
    restored = StagedProfile.from_json(json.dumps(blob))
    assert restored.last_assumed_completed_credentials == []


def test_from_json_dedupes_duplicate_canonicals_promoting_to_claimed():
    """Round-18 cookie-boundary defense: a cookie that survived
    signature verification with duplicate canonicals MUST be deduped
    at deserialization with hypothetical -> claimed promotion when
    any duplicate is claimed."""
    blob = json.loads(_fresh().to_json())
    blob["last_assumed_completed_credentials"] = [
        {"canonical": "X", "mode": "hypothetical"},
        {"canonical": "X", "mode": "claimed"},
        {"canonical": "Y", "mode": "hypothetical"},
    ]
    restored = StagedProfile.from_json(json.dumps(blob))
    # X collapsed to one entry, promoted to claimed; Y stays as its
    # own entry; order preserved.
    assert restored.last_assumed_completed_credentials == [
        {"canonical": "X", "mode": "claimed"},
        {"canonical": "Y", "mode": "hypothetical"},
    ]


def test_from_json_late_duplicate_after_cap_still_promotes():
    """Round-19 cookie-boundary fix: the dedupe scan MUST run over the
    FULL input before the cap is applied. A late duplicate sitting
    past the MAX_CRED_GAPS-th position should still promote an earlier
    hypothetical entry to claimed.

    Persisted input (6 entries, 5 unique with A repeated at the end):
        [A hypothetical, B, C, D, E, A claimed]

    Pre-fix behavior: the loop broke after collecting A..E (5 unique),
    skipping the second A; A stayed hypothetical.
    Correct behavior: A is promoted to claimed; the result is the
    five entries [A claimed, B, C, D, E].
    """
    blob = json.loads(_fresh().to_json())
    blob["last_assumed_completed_credentials"] = [
        {"canonical": "A", "mode": "hypothetical"},
        {"canonical": "B", "mode": "claimed"},
        {"canonical": "C", "mode": "claimed"},
        {"canonical": "D", "mode": "claimed"},
        {"canonical": "E", "mode": "claimed"},
        {"canonical": "A", "mode": "claimed"},   # past the cap; must
                                                  # still promote A
    ]
    restored = StagedProfile.from_json(json.dumps(blob))
    assert restored.last_assumed_completed_credentials == [
        {"canonical": "A", "mode": "claimed"},
        {"canonical": "B", "mode": "claimed"},
        {"canonical": "C", "mode": "claimed"},
        {"canonical": "D", "mode": "claimed"},
        {"canonical": "E", "mode": "claimed"},
    ]


def test_from_json_dedupe_keeps_first_mode_when_no_promotion_warranted():
    """Same-mode duplicates collapse without changing mode; a claimed-
    then-hypothetical sequence stays claimed (first wins, no demotion)."""
    blob = json.loads(_fresh().to_json())
    blob["last_assumed_completed_credentials"] = [
        {"canonical": "X", "mode": "claimed"},
        {"canonical": "X", "mode": "hypothetical"},
    ]
    restored = StagedProfile.from_json(json.dumps(blob))
    assert restored.last_assumed_completed_credentials == [
        {"canonical": "X", "mode": "claimed"},
    ]


def test_from_json_caps_accumulated_at_max_cred_gaps_preserving_order():
    blob = json.loads(_fresh().to_json())
    blob["last_assumed_completed_credentials"] = [
        {"canonical": f"c_{i}", "mode": "claimed"} for i in range(10)
    ]
    restored = StagedProfile.from_json(json.dumps(blob))
    canonicals = [c["canonical"] for c in restored.last_assumed_completed_credentials]
    assert canonicals == [f"c_{i}" for i in range(MAX_CRED_GAPS)]


def test_from_json_drops_pending_with_unknown_action():
    blob = json.loads(_fresh().to_json())
    blob["pending_credential_confirmation"] = {
        "canonical": "x", "action": "modify",
    }
    restored = StagedProfile.from_json(json.dumps(blob))
    assert restored.pending_credential_confirmation is None


def test_from_json_drops_pending_with_non_string_canonical():
    blob = json.loads(_fresh().to_json())
    blob["pending_credential_confirmation"] = {"canonical": None, "action": "add"}
    restored = StagedProfile.from_json(json.dumps(blob))
    assert restored.pending_credential_confirmation is None


def test_from_json_drops_pending_when_not_a_dict():
    blob = json.loads(_fresh().to_json())
    blob["pending_credential_confirmation"] = ["canonical", "x"]
    restored = StagedProfile.from_json(json.dumps(blob))
    assert restored.pending_credential_confirmation is None


def test_from_json_drops_last_discussed_when_not_a_string():
    blob = json.loads(_fresh().to_json())
    blob["last_discussed_credential_canonical"] = 42
    restored = StagedProfile.from_json(json.dumps(blob))
    assert restored.last_discussed_credential_canonical is None


def test_from_json_drops_last_discussed_when_empty_string():
    blob = json.loads(_fresh().to_json())
    blob["last_discussed_credential_canonical"] = ""
    restored = StagedProfile.from_json(json.dumps(blob))
    assert restored.last_discussed_credential_canonical is None


def test_from_json_truncates_long_last_discussed():
    blob = json.loads(_fresh().to_json())
    blob["last_discussed_credential_canonical"] = "x" * 500
    restored = StagedProfile.from_json(json.dumps(blob))
    assert len(restored.last_discussed_credential_canonical) == MAX_CANONICAL_CHARS


# ----------------------------------------------- backward-compatibility
def test_legacy_session_blob_without_new_fields_loads_with_defaults():
    """A pre-R-1 cookie should deserialize with the four new fields at
    their defaults (None / []), not raise KeyError."""
    sp = _fresh()
    blob = json.loads(sp.to_json())
    for k in (
        "last_match_snapshot",
        "last_assumed_completed_credentials",
        "last_discussed_credential_canonical",
        "pending_credential_confirmation",
    ):
        blob.pop(k, None)
    restored = StagedProfile.from_json(json.dumps(blob))
    assert restored.last_match_snapshot is None
    assert restored.last_assumed_completed_credentials == []
    assert restored.last_discussed_credential_canonical is None
    assert restored.pending_credential_confirmation is None
