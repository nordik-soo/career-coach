"""Unit tests for near-miss gap classification (chat orchestration v2.2).

Three concerns:

  1. Every registry entry maps to the right near-miss bucket via the
     existing `Gap.category` field. The 13 live YAML entries are the
     pinning ground truth -- if any map differently after a YAML
     edit, this test fails fast.

  2. Heuristic table behavior: each branch of `_classify_by_heuristic`
     fires on canonical example phrases. Includes both true positives
     (e.g. "Class A driver certification" -> credential) and the
     negative cases (e.g. "brake repair" -> core_skill, NOT credential).

  3. Telemetry: heuristic classifications emit ONE INFO log per call.
     Registry hits emit NONE. The backlog flow depends on the log
     being a faithful record of unregistered gaps surfaced.

This module is DEAD CODE until Slice N-2/3/5 wire it. These tests
prove correctness before integration.
"""
from __future__ import annotations

import logging
import os

os.environ.setdefault("LLM_ENABLED", "false")

import pytest

from skillbridge.match.engine import MatchResult
from skillbridge.match.near_miss import (
    DEFAULT_CORE_SKILL_CAP,
    DEFAULT_CREDENTIAL_CAP,
    TITLE_MATCH_SIMILARITY_THRESHOLD,
    _REGISTRY_TO_NEAR_MISS,
    _canonical_or_raw,
    _classify_by_heuristic,
    _qualifies_as_near_miss,
    build_near_miss_payload,
    classify_gap,
    filter_near_miss_candidates,
)
from skillbridge.training.registry import TrainingRegistry


pytestmark = pytest.mark.nodb


# ===========================================================================
# Shared registry fixture (load once per test session; .from_yaml is
# pure-file and ~5ms but no point repeating it)
# ===========================================================================
@pytest.fixture(scope="module")
def registry() -> TrainingRegistry:
    return TrainingRegistry.from_yaml()


# ===========================================================================
# 1. Registry mapping: every canonical_name + every alias lands in the
#    expected near-miss bucket per the locked design (Q2).
# ===========================================================================
# The truth table: (registry category) -> (near-miss bucket).
# Anything credential-like collapses to "credential"; "skill" -> "core_skill".
# If a future YAML edit introduces a NEW category value, the registry
# helper raises -- the production safety net.
def test_registry_mapping_table_is_complete():
    """The `_REGISTRY_TO_NEAR_MISS` table MUST cover every category
    value currently present in the YAML. Otherwise an entry would
    silently raise at runtime."""
    reg = TrainingRegistry.from_yaml()
    yaml_categories = {g.category for g in reg.gaps}
    missing = yaml_categories - set(_REGISTRY_TO_NEAR_MISS)
    assert not missing, (
        f"YAML has category values not handled by _REGISTRY_TO_NEAR_MISS: "
        f"{missing}. Add them to the mapping or delete the entry."
    )


def test_every_yaml_canonical_name_classifies_via_registry(registry):
    """Each of the 13 live YAML entries, looked up by canonical_name,
    maps to the expected near-miss bucket per the locked design
    mapping table. This is the ground-truth assertion: if anyone
    edits the YAML in a way that breaks the mapping, this fails fast."""
    for gap in registry.gaps:
        expected = _REGISTRY_TO_NEAR_MISS[gap.category]
        actual = classify_gap(gap.canonical_name, registry)
        assert actual == expected, (
            f"Registry entry {gap.canonical_name!r} has "
            f"category={gap.category!r}, mapping table expects "
            f"{expected!r}, classify_gap returned {actual!r}"
        )


@pytest.mark.parametrize("alias,expected", [
    # Sample aliases across all three credential-mapped categories
    ("310T",              "credential"),    # credential
    ("Class G",           "credential"),    # license -> credential
    ("WHMIS",             "credential"),    # safety_training -> credential
    ("first aid",         "credential"),    # safety_training -> credential
    ("forklift",          "credential"),    # safety_training -> credential
    # And the skill bucket
    ("Excel",             "core_skill"),
    ("MS Excel",          "core_skill"),
    ("QuickBooks",        "core_skill"),
])
def test_aliases_route_via_registry_lookup(alias, expected, registry):
    """Aliases (not just canonical names) must classify the same as
    their canonical entry. `registry.lookup` does the alias matching;
    this test pins that classify_gap uses lookup correctly."""
    assert classify_gap(alias, registry) == expected


# ===========================================================================
# 2. Heuristic fallback: non-YAML gap names route via keyword rules.
# ===========================================================================
# Each row is a (name, expected_bucket, reason) triple. The reason is
# documentation only -- pytest doesn't see it, but it makes the
# parametrize block readable when a future contributor reviews this.
@pytest.mark.parametrize("name,expected", [
    # CREDENTIAL keywords
    ("Class A driver certification",      "credential"),
    ("Smart Serve certificate",           "credential"),
    ("fall protection ticket",            "credential"),
    ("certificate of qualification",      "credential"),
    ("food handler licence",              "credential"),  # licence (UK spelling)
    ("food handler license",              "credential"),  # license (US spelling)
    ("OSHA credential",                   "credential"),
    # OPERATIONAL keywords
    ("MTO contract supervision",          "operational"),
    ("driver hour tracking",              "operational"),
    ("on-call availability",              "operational"),
    ("on call rotation",                  "operational"),
    ("weekend availability",              "operational"),
    ("shift willingness",                 "operational"),
    # CORE_SKILL fallback (no keyword matches)
    ("brake system inspection",           "core_skill"),
    ("truck service and maintenance",     "core_skill"),
    ("transmission diagnostics",          "core_skill"),
    ("welding and fabrication",           "core_skill"),
    ("customer escalation handling",      "core_skill"),
])
def test_heuristic_classification(name, expected, registry):
    """Each non-YAML gap name routes via the keyword heuristic to
    the expected bucket. Covers all three branches + the negative
    case (core_skill fallback when no keyword matches)."""
    assert classify_gap(name, registry) == expected


def test_pure_heuristic_function_does_not_consult_registry():
    """`_classify_by_heuristic` is the pure inner function. It MUST
    NOT consult the registry -- the public `classify_gap` does that
    first. Testing it in isolation lets us pin the keyword tables
    without registry-coupling."""
    assert _classify_by_heuristic("Class A certification") == "credential"
    assert _classify_by_heuristic("on-call availability") == "operational"
    assert _classify_by_heuristic("transmission repair") == "core_skill"


def test_heuristic_priority_credential_beats_operational():
    """When a gap matches BOTH a credential keyword and an
    operational keyword (rare but possible), credential wins.
    Reasoning in the design: silently dropping a real credential gap
    as operational would lose useful info; mis-flagging an
    operational item as credential surfaces but doesn't mislead.

    Input choice: "supervision certification" hits BOTH keyword
    sets -- "certif" (credential) AND "supervision" (operational).
    "supervisor certification" would only hit credential because
    the operational keyword is "supervision", not "supervisor", so
    it wouldn't actually prove priority. Pinning the genuine
    both-hit case so future contributors can't accidentally weaken
    the priority guarantee."""
    assert _classify_by_heuristic("supervision certification") == "credential"
    # Sanity check: the OPERATIONAL-only sibling phrase classifies
    # as operational, confirming the keyword genuinely fires.
    assert _classify_by_heuristic("supervision rotation") == "operational"


# ===========================================================================
# 3. Empty / None / whitespace input: defensive fallback to core_skill,
#    with WARNING log. Should never happen in production but the engine
#    contract isn't formally narrowed against None.
# ===========================================================================
@pytest.mark.parametrize("bad", ["", "   ", "\t\n"])
def test_empty_input_returns_core_skill_with_warning(bad, registry, caplog):
    caplog.set_level(logging.WARNING, logger="skillbridge.match.near_miss")
    assert classify_gap(bad, registry) == "core_skill"
    assert any(
        "empty name" in r.message for r in caplog.records
    ), "expected a WARNING log for empty-name input"


# ===========================================================================
# 4. Telemetry: heuristic calls log ONCE at INFO; registry hits log nothing.
#    The backlog flow depends on logs being a faithful record.
# ===========================================================================
def test_heuristic_call_emits_one_info_log(registry, caplog):
    """A non-YAML gap name routes via heuristic and MUST emit exactly
    one INFO log mentioning the gap name and the category. This is
    the telemetry the design promises for backlog triage."""
    caplog.set_level(logging.INFO, logger="skillbridge.match.near_miss")
    classify_gap("driver hour tracking", registry)
    matching = [
        r for r in caplog.records
        if "heuristic_classified" in r.message
        and "driver hour tracking" in r.message
        and "operational" in r.message
    ]
    assert len(matching) == 1, (
        f"expected exactly one heuristic_classified INFO log; got "
        f"{[r.message for r in caplog.records]}"
    )


def test_registry_hit_emits_no_heuristic_log(registry, caplog):
    """When the registry classifies the gap, no `heuristic_classified`
    log should appear. Telemetry must not falsely report known
    entries as unregistered."""
    caplog.set_level(logging.INFO, logger="skillbridge.match.near_miss")
    classify_gap("310T technician certification", registry)
    assert not any(
        "heuristic_classified" in r.message for r in caplog.records
    ), (
        "registry hits must not emit the heuristic_classified log; "
        f"got {[r.message for r in caplog.records]}"
    )


# ===========================================================================
# 5. Defense-in-depth: an unknown registry `category` value raises with
#    a clear error pointing at the offending gap. This catches a YAML
#    edit that introduces a new category before it ships.
# ===========================================================================
def test_unknown_registry_category_raises_with_clear_error(registry):
    """Synthesize a Gap with a category value not in the mapping table.
    classify_gap MUST raise ValueError naming the gap and the bad
    category. Production safety: a silent fall-through to core_skill
    would mask a YAML config bug."""
    from skillbridge.training.registry import Gap

    bad_gap = Gap(
        canonical_name="synthetic_bad_gap",
        aliases=("synthetic-bad-gap",),
        category="some_new_category",  # not in _REGISTRY_TO_NEAR_MISS
        description="",
        resources=(),
    )
    # Inject the bad gap into a synthetic registry. We don't mutate
    # the shared registry. TrainingRegistry is a frozen dataclass with
    # `version` and `registry_verified_at` fields the YAML loader sets
    # -- we mirror those positionally here.
    synth_registry = TrainingRegistry(
        version=1, registry_verified_at=None, gaps=(bad_gap,),
    )

    with pytest.raises(ValueError) as excinfo:
        classify_gap("synthetic_bad_gap", synth_registry)
    msg = str(excinfo.value)
    assert "synthetic_bad_gap" in msg
    assert "some_new_category" in msg


# ===========================================================================
# 6. Live-bug regression: the exact Michael Carter case from the
#    2026-06-05 design doc. After classification, the 11 gaps from
#    the truck-tech job MUST split:
#      2 credentials, 6 core_skills, 3 operational  (per the design
#      doc's worked example).
# ===========================================================================
def test_michael_truck_tech_case_classifies_per_design(registry):
    """The 11 gaps from the live truck-tech job MUST split into the
    exact counts the design doc names. This is the worked-example
    pin -- if classification drifts, the doc and the code disagree."""
    truck_gaps = [
        "310T certificate of qualification",
        "Class G driver's license",
        "truck service and maintenance",
        "emergency repair",
        "emissions testing preparation",
        "wheel end inspection",
        "parts fabrication",
        "motor vehicle inspection",
        "MTO contract supervision",
        "driver hour tracking",
        "on-call availability",
    ]
    counts = {"credential": 0, "core_skill": 0, "operational": 0}
    for g in truck_gaps:
        counts[classify_gap(g, registry)] += 1

    assert counts == {
        "credential":  2,   # 310T + Class G
        "core_skill":  6,   # six skill-shaped gaps
        "operational": 3,   # MTO supervision + hour tracking + on-call
    }, (
        f"Michael truck-tech classification drifted from the design "
        f"doc's worked example: got {counts}"
    )


# ===========================================================================
# 7. filter_near_miss_candidates -- per the locked design (Slice N-2)
#
# Reviewer's pre-build checklist:
#   - only low-band eligible candidates considered
#   - title_match_override=True passes
#   - title_match_similarity >= 0.85 passes
#   - noc_code == target_noc passes
#   - score alone does NOT pass
#   - generic skill overlap does NOT pass
#   - empty target role / empty NOC does not accidentally pass
#   - output order preserves engine ranking
#
# Each bullet has at least one parametrized or named test below.
# ===========================================================================
def _mk(
    *,
    job_id: str = "j1",
    band: str = "low",
    eligible: bool = True,
    score: float = 0.30,
    title: str = "Truck and Coach Technician",
    noc: str | None = None,
    override: bool = False,
    similarity: float = 0.0,
) -> MatchResult:
    """Minimal MatchResult factory for filter tests. Defaults match the
    'plausible low-band candidate' shape so tests can override only
    the field under examination -- keeps the parametrize blocks readable."""
    return MatchResult(
        job_id=job_id, profile_id="p", title=title,
        employer="Acme", url="https://example.com", location="SSM",
        match_score=score, match_band=band, match_eligible=eligible,
        ineligibility_reason=None,
        matched_skills=[], missing_skills=[],
        matched_skill_ids=[], missing_skill_ids=[],
        required_skills_count=10, credential_warning=None,
        posted_date=None, noc_code=noc,
        score_explanation={
            "title_match_override": override,
            "title_match_similarity": similarity,
        },
    )


# ---- Positive cases: each of the 3 qualifying conditions ----
def test_filter_qualifies_on_title_match_override():
    cands = [_mk(job_id="j-override", override=True)]
    out = filter_near_miss_candidates(cands, "truck", target_noc=None)
    assert [m.job_id for m in out] == ["j-override"]


def test_filter_qualifies_on_high_title_similarity():
    """At-threshold (== 0.85) should pass. Above threshold (0.9) passes.
    Below threshold (0.84) does not. Boundary pinned."""
    at_threshold = _mk(job_id="j-edge", similarity=TITLE_MATCH_SIMILARITY_THRESHOLD)
    above_threshold = _mk(job_id="j-above", similarity=0.92)
    out = filter_near_miss_candidates(
        [at_threshold, above_threshold], "truck", target_noc=None,
    )
    assert [m.job_id for m in out] == ["j-edge", "j-above"]


def test_filter_qualifies_on_noc_match():
    cands = [_mk(job_id="j-noc", noc="7321")]
    out = filter_near_miss_candidates(cands, target_role_text=None, target_noc="7321")
    assert [m.job_id for m in out] == ["j-noc"]


# ---- Negative cases: none of the 3 conditions hold ----
@pytest.mark.parametrize("name,kwargs", [
    ("score_only_proximity",      {"score": 0.42, "similarity": 0.30}),
    ("similarity_below_threshold", {"similarity": 0.84}),
    ("noc_mismatch",              {"noc": "1234"}),
    ("override_false_plus_low_sim", {"override": False, "similarity": 0.50}),
    ("everything_default_zero",    {}),
])
def test_filter_rejects_when_no_condition_holds(name, kwargs):
    """Score alone, low similarity, NOC mismatch, and the 'no signal'
    case all fail. None of these is a near-miss."""
    cands = [_mk(job_id=f"j-{name}", **kwargs)]
    out = filter_near_miss_candidates(cands, "truck", target_noc="7321")
    assert out == [], f"{name} should NOT qualify; got {[m.job_id for m in out]}"


def test_filter_rejects_generic_skill_overlap_without_title_match():
    """The MOST IMPORTANT negative test: a low-band 'marketing
    coordinator' job that scored 0.15 because of generic
    'communication' or 'teamwork' overlap is NOT a near-miss for a
    truck technician. No title_match_override, no high similarity,
    no NOC match -- the engine surfaced it because some skill overlapped,
    but the role is wrong."""
    coordinator = _mk(
        job_id="j-coord",
        title="Marketing Coordinator",
        score=0.15,
        override=False,
        similarity=0.10,   # very low title similarity
        noc="1123",        # different NOC family
    )
    out = filter_near_miss_candidates([coordinator], "truck and coach technician", target_noc="7321")
    assert out == []


# ---- Defensive guards: caller-error paths ----
def test_filter_rejects_non_low_band_with_warning(caplog):
    """If a stretch / good / strong match slips into the input list,
    the filter MUST reject it -- silently promoting a stretch to
    near-miss would demote the user's existing skill foundation."""
    caplog.set_level(logging.WARNING, logger="skillbridge.match.near_miss")
    stretch = _mk(job_id="j-stretch", band="stretch", override=True)
    good    = _mk(job_id="j-good",    band="good",    override=True)
    out = filter_near_miss_candidates(
        [stretch, good], "truck", target_noc=None,
    )
    assert out == []
    assert sum(1 for r in caplog.records if "non-low band" in r.message) == 2


def test_filter_rejects_ineligible_with_warning(caplog):
    """Ineligible matches MUST be dropped. The engine excludes
    ineligible-band-X matches from the recommended pool; the
    near-miss filter MUST honor that."""
    caplog.set_level(logging.WARNING, logger="skillbridge.match.near_miss")
    bad = _mk(job_id="j-inelig", eligible=False, override=True)
    out = filter_near_miss_candidates([bad], "truck", target_noc="7321")
    assert out == []
    assert any("ineligible" in r.message for r in caplog.records)


# ---- Empty target signals must not accidentally pass ----
@pytest.mark.parametrize("role,noc", [
    (None, None),
    ("",   None),
    ("   ", None),
    (None, ""),
    (None, "   "),
    ("",   ""),
])
def test_filter_returns_empty_when_no_target_signal(role, noc):
    """When BOTH target_role_text and target_noc are absent/empty,
    near-miss has no anchor -- return [] regardless of candidate
    quality. Prevents the 'override=True is always a near-miss'
    foot-gun if the handler ever forgets the precondition gate."""
    cands = [_mk(job_id="j-anchor", override=True)]  # would otherwise pass
    out = filter_near_miss_candidates(cands, role, noc)
    assert out == []


# ---- Output order preserves engine ranking ----
def test_filter_preserves_input_order_which_is_engine_ranking():
    """The engine sorts matches by (eligible, score) descending before
    they reach us. The filter must preserve that order so the
    highest-scoring near-miss is index [0] -- Slice N-4's responder
    uses [0] as the 'first/strongest' per locked Q6."""
    a = _mk(job_id="rank-1-override", override=True,  score=0.45)
    b = _mk(job_id="rank-2-sim",      similarity=0.91, score=0.40)
    c = _mk(job_id="rank-3-noc",      noc="7321",     score=0.30)
    # Interleave a non-qualifying candidate to confirm filtering
    # doesn't mangle order
    skip = _mk(job_id="skip-no-signal", score=0.38)
    out = filter_near_miss_candidates(
        [a, b, skip, c], "truck", target_noc="7321",
    )
    assert [m.job_id for m in out] == ["rank-1-override", "rank-2-sim", "rank-3-noc"]


def test_filter_returns_empty_when_input_is_empty():
    assert filter_near_miss_candidates([], "truck", target_noc="7321") == []


# ---- Pure predicate helper: per-candidate logic without list mechanics ----
def test_qualifies_predicate_each_branch():
    """_qualifies_as_near_miss is the per-candidate decision. Testing
    it directly lets the list-level filter tests focus on filtering
    behavior (skip/drop/order) without re-asserting the predicate
    truth table for every list-mode test."""
    assert _qualifies_as_near_miss(_mk(override=True), target_noc=None) is True
    assert _qualifies_as_near_miss(_mk(similarity=0.85), target_noc=None) is True
    assert _qualifies_as_near_miss(_mk(noc="7321"), target_noc="7321") is True
    # All three off -> False
    assert _qualifies_as_near_miss(_mk(), target_noc="7321") is False
    # NOC None disables the NOC rule even if candidate has noc_code set
    assert _qualifies_as_near_miss(_mk(noc="7321"), target_noc=None) is False
    # Non-numeric similarity is defensively ignored
    bad = _mk()
    bad.score_explanation = {"title_match_similarity": "0.99"}  # string, not float
    assert _qualifies_as_near_miss(bad, target_noc=None) is False
    # None score_explanation also handled
    blank = _mk()
    blank.score_explanation = None
    assert _qualifies_as_near_miss(blank, target_noc=None) is False


# ---- Live regression: Michael truck-tech case end-to-end through filter ----
def test_filter_michael_truck_tech_scenario():
    """Synthetic equivalent of the live-test scenario: 25 jobs, only
    one (Truck and Coach Technician) has title_match_override=True.
    Filter must return exactly that one job, regardless of input
    order or other candidates' scores."""
    truck = _mk(
        job_id="truck", title="Truck and Coach Technician",
        score=0.30, override=True, noc="7321",
    )
    noise = [
        _mk(job_id=f"noise-{i}", title=f"Unrelated Role {i}",
            score=0.10 + i * 0.005, override=False, similarity=0.12, noc="9999")
        for i in range(24)
    ]
    out = filter_near_miss_candidates(
        noise + [truck], "truck and coach technician", target_noc="7321",
    )
    assert [m.job_id for m in out] == ["truck"]


# ===========================================================================
# 8. build_near_miss_payload -- Slice N-5 (payload-shape helper)
#
# Five concerns the reviewer's pre-build checklist cares about:
#   - classify+drop: operational gaps are filtered out
#   - cap 3+3: at most 3 credentials + 3 core skills
#   - credential-first ordering (alphabetical for v1 per locked Q4)
#   - canonical-name alignment: engine names map to registry canonical
#     so _find_grounded_provider in the responder matches reliably
#     (Slice N-4 reviewer note)
#   - empty candidates list raises (caller bug)
# ===========================================================================
def _mk_for_payload(
    *,
    job_id: str = "j-payload",
    title: str = "Truck and Coach Technician",
    employer: str | None = "Garden River First Nation",
    required_missing: list[str] | None = None,
    credential_gap_skills: list[str] | None = None,
    override: bool = True,
    noc: str | None = "7321",
) -> MatchResult:
    """MatchResult factory for build_near_miss_payload tests."""
    return MatchResult(
        job_id=job_id, profile_id="p", title=title, employer=employer,
        url="https://example.com", location="SSM",
        match_score=0.30, match_band="low", match_eligible=True,
        ineligibility_reason=None,
        matched_skills=[], missing_skills=[],
        matched_skill_ids=[], missing_skill_ids=[],
        required_skills_count=12, credential_warning=None,
        posted_date=None, noc_code=noc,
        score_explanation={
            "title_match_override": override,
            "title_match_similarity": 1.0 if override else 0.0,
            "required_missing": required_missing or [],
            "credential_gap_skills": credential_gap_skills or [],
        },
    )


def test_build_payload_michael_truck_tech_full_scenario(registry):
    """Live-bug scenario: the engine surfaces 11 gaps -- 2 registry
    credentials, 6 heuristic core_skills, 3 operational. Payload must
    contain canonical-aligned credentials, alphabetical core_skills
    capped at 3, and ZERO operational entries."""
    candidate = _mk_for_payload(required_missing=[
        # registry canonical
        "310T technician certification",
        # registry canonical
        "Class G driver's license",
        # heuristic core_skill
        "truck service and maintenance",
        "emergency repair",
        "emissions testing preparation",
        "wheel end inspection",
        "parts fabrication",
        "motor vehicle inspection",
        # heuristic operational (must be filtered out)
        "MTO contract supervision",
        "driver hour tracking",
        "on-call availability",
    ])
    payload = build_near_miss_payload([candidate], registry)
    assert payload["role"] == "Truck and Coach Technician"
    assert payload["employer"] == "Garden River First Nation"
    assert payload["job_count"] == 1
    assert payload["credential_gaps"] == [
        "310T technician certification",   # alpha order: 3 < C
        "Class G driver's license",
    ]
    assert payload["core_skill_gaps"] == [
        "emergency repair",
        "emissions testing preparation",
        "motor vehicle inspection",
    ]
    # Operational MUST NOT appear anywhere
    for op_gap in ("MTO contract supervision", "driver hour tracking",
                   "on-call availability"):
        assert op_gap not in payload["credential_gaps"]
        assert op_gap not in payload["core_skill_gaps"]


def test_build_payload_canonical_name_alignment_per_n4_review(registry):
    """The Slice N-4 reviewer's concern: if the engine surfaces a
    registry ALIAS rather than the canonical name, the payload must
    surface the CANONICAL name (so the responder's
    _find_grounded_provider lookup matches against
    training_by_job['for_gap'] reliably).

    Setup: synthesize an engine gap name that's a registry ALIAS
    (not the canonical). Expect canonical name in the payload."""
    # First find an actual alias in the registry to use
    # Class G driver's license has the alias "Class G"
    candidate = _mk_for_payload(required_missing=[
        "Class G",           # alias, NOT canonical -- must be aligned
        "Microsoft Excel",   # canonical itself, also valid input
        "WHMIS",             # canonical, also valid input
    ])
    payload = build_near_miss_payload([candidate], registry)
    # Both Class G (alias) AND WHMIS (canonical) classify as credential.
    # Microsoft Excel is core_skill.
    assert "Class G driver's license" in payload["credential_gaps"]
    assert "Class G" not in payload["credential_gaps"]
    assert "WHMIS" in payload["credential_gaps"]
    assert "Microsoft Excel" in payload["core_skill_gaps"]


def test_build_payload_caps_at_three_each(registry):
    """A candidate with > 3 credentials + > 3 core_skills must produce
    a payload capped at 3+3 (locked Q4). Excess entries are dropped
    in alphabetical order."""
    candidate = _mk_for_payload(required_missing=[
        # 5 heuristic credentials
        "alpha certification",
        "bravo certification",
        "charlie certification",
        "delta certification",
        "echo certification",
        # 5 heuristic core_skills
        "alpha repair",
        "bravo repair",
        "charlie repair",
        "delta repair",
        "echo repair",
    ])
    payload = build_near_miss_payload([candidate], registry)
    assert len(payload["credential_gaps"]) == DEFAULT_CREDENTIAL_CAP == 3
    assert len(payload["core_skill_gaps"]) == DEFAULT_CORE_SKILL_CAP == 3
    # Alphabetical -- first three of each bucket
    assert payload["credential_gaps"] == [
        "alpha certification", "bravo certification", "charlie certification",
    ]
    assert payload["core_skill_gaps"] == [
        "alpha repair", "bravo repair", "charlie repair",
    ]


def test_build_payload_falls_back_to_credential_gap_skills_when_required_missing_empty(
    registry,
):
    """Some engine code paths populate credential_gap_skills but leave
    required_missing empty. Helper must read both."""
    candidate = _mk_for_payload(
        required_missing=[],
        credential_gap_skills=["310T technician certification"],
    )
    payload = build_near_miss_payload([candidate], registry)
    assert payload["credential_gaps"] == ["310T technician certification"]


def test_build_payload_uses_top_candidate_for_header_fields(registry):
    """When passed multiple candidates, the payload's role/employer
    come from index [0] (engine-ranking top). job_count is the total
    list length, NOT just the top. Locked Q6."""
    top = _mk_for_payload(
        job_id="top", title="Truck and Coach Technician",
        employer="Garden River FN",
        required_missing=["310T technician certification"],
    )
    runner_up = _mk_for_payload(
        job_id="runner", title="Diesel Mechanic",   # different title
        employer="Northern Garage",
        required_missing=["welding certification"],
    )
    third = _mk_for_payload(
        job_id="third", title="Heavy Equip Tech",
        employer="Acme",
        required_missing=["forklift certification"],
    )
    payload = build_near_miss_payload([top, runner_up, third], registry)
    # Header from top
    assert payload["role"] == "Truck and Coach Technician"
    assert payload["employer"] == "Garden River FN"
    # Count is total
    assert payload["job_count"] == 3
    # Gaps come from top only (not unioned across all candidates)
    assert payload["credential_gaps"] == ["310T technician certification"]
    # The runner-up's gap MUST NOT appear -- Q6 says first/strongest only
    for foreign_gap in ("welding certification", "forklift certification"):
        assert foreign_gap not in payload["credential_gaps"]
        assert foreign_gap not in payload["core_skill_gaps"]


def test_build_payload_empty_required_missing_returns_valid_payload(registry):
    """A candidate with zero surfaced gaps (rare; thin job_skill
    coverage). Payload still returned with role/employer/job_count
    but empty gap lists. The responder's defensive fallback handles
    the no-gap rendering."""
    candidate = _mk_for_payload(required_missing=[])
    payload = build_near_miss_payload([candidate], registry)
    assert payload["role"] == "Truck and Coach Technician"
    assert payload["credential_gaps"] == []
    assert payload["core_skill_gaps"] == []
    assert payload["job_count"] == 1


def test_build_payload_raises_on_empty_candidate_list(registry):
    """Caller is contracted to short-circuit before calling this --
    an empty list signals a caller bug, not a no-data case. Raise
    so the bug surfaces immediately rather than producing a payload
    with role=None that lies to the responder."""
    with pytest.raises(ValueError) as excinfo:
        build_near_miss_payload([], registry)
    assert "empty" in str(excinfo.value).lower()


def test_build_payload_dedup_within_bucket(registry):
    """If the engine surfaces an alias AND its canonical name for the
    same gap (rare but possible), the payload dedupes to one entry.
    Canonical-name alignment is what enables this -- both versions
    collapse to the same canonical string after alignment."""
    candidate = _mk_for_payload(required_missing=[
        "Class G",                       # alias for "Class G driver's license"
        "Class G driver's license",      # canonical
        "Microsoft Excel",               # canonical core_skill
        "Excel",                         # alias for Microsoft Excel
    ])
    payload = build_near_miss_payload([candidate], registry)
    # Dedup: only one of each
    assert payload["credential_gaps"].count("Class G driver's license") == 1
    assert payload["core_skill_gaps"].count("Microsoft Excel") == 1


def test_canonical_or_raw_returns_canonical_for_registry_hit(registry):
    """Pure helper sanity: alias goes in, canonical comes out."""
    assert _canonical_or_raw("Class G", registry) == "Class G driver's license"
    assert _canonical_or_raw("310T", registry) == "310T technician certification"


def test_canonical_or_raw_returns_raw_for_registry_miss(registry):
    """Unknown gap: pass through unchanged. No invention."""
    assert _canonical_or_raw("snake oil license", registry) == "snake oil license"
    assert _canonical_or_raw("brake system repair", registry) == "brake system repair"
