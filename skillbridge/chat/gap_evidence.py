"""Three-layer gap evidence subsystem (locked 2026-06-18).

LOCKED DESIGN (see project_three_layer_gap_evidence memory):
  - Three independent gap detectors emit a unified GapEvidence shape:
      Layer A -- target NOC standard gap     (OaSIS reference.noc_skill)
      Layer B -- local posting gap            (engine MatchResult.missing_*)
      Layer C -- adjacent NOC standard gap   (OaSIS, fan-out over CP5
                                              tier_evidence.sideways_move
                                              unique NOCs)
  - Same blocker classifier across all three: `is_credential_skill_name`
    from skillbridge.match.engine. No fork.
  - First release: ephemeral computation per turn. No StagedProfile
    field, no DB persistence, no cookie-budget impact.
  - Recommender-side only. Does NOT affect matching, scoring, tier
    (Strong/Good/Stretch/Explore later), or CP5 behavior.
  - User skill source: the same canonical set the matcher uses
    (build_user_skill_rows -> derive_user_skill_sets). No parallel
    user-skill authority.

This module ships in named slices:
  Slice 1               -- shared `GapEvidence` shape (this file).
  Slice 2               -- Layer A detector.
  Slice 3               -- Layer B detector.
  Slice 4 (this slice)  -- Layer C detector + Layer A refactor to share
                           the OaSIS row-processing helper.
  Slice 5 (deferred)    -- wiring into the recommender once that
                           consumer design is locked.

Until Slice 5, no production code imports from this module. A guard
test in tests/test_gap_evidence.py pins that contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal

from skillbridge.db import sync_cursor
from skillbridge.match.engine import is_credential_skill_name

Layer = Literal[
    "target_noc_standard",   # Layer A
    "local_posting",         # Layer B
    "adjacent_noc_standard", # Layer C
]

GapSource = Literal[
    "reference.noc_skill",   # OaSIS-driven (Layer A and Layer C)
    "extracted.job_skill",   # SCCC postings (Layer B)
]

LayerAStatus = Literal[
    "ok",                            # query succeeded, gaps may be empty
                                     # (user has all NOC skills) or non-empty
    "no_target_noc",                 # target_noc was None / empty
    "no_reference_skill_profile",    # NOC valid but reference.noc_skill
                                     # has no rows for it
]


@dataclass(frozen=True, slots=True)
class GapEvidence:
    """One skill record for the recommender.

    Deliberately NEUTRAL framing: this dataclass carries records from
    all three layers, but the layers describe different things:
      - Layer B records ARE missing-skills (the user lacks them per
        a specific posting's requirements).
      - Layer A / Layer C records are occupation-standard skills the
        user does not match against by canonical skill_id -- which is
        NOT the same as "the user lacks them." A user can clearly
        demonstrate Reading Comprehension via work history without the
        canonical skill_id ever appearing on their staged profile.
    The voice that distinguishes "missing posting requirement" from
    "occupation-standard development area" lives in the recommender's
    LLM prompt sections, NOT in this data shape. The `layer`
    discriminator gates which voice is used.

    Emitted independently by each of the three layer detectors. The
    `layer` discriminator tells the recommender which evidence stream
    the record belongs to. Shape is identical across layers so the
    recommender consumes a flat list[GapEvidence] and groups by
    `layer` in the LLM prompt sections (TARGET_NOC_STANDARD_GAPS /
    LOCAL_POSTING_GAPS / ADJACENT_NOC_STANDARD_GAPS).

    Nullability contract
    --------------------
    `skill_id` is always populated for Layer A and Layer C
    records -- the OaSIS source (reference.noc_skill JOIN reference.skill)
    yields a canonical skill_id by construction. It is NULLABLE for
    Layer B records: the SCCC job-posting extractor may not have
    resolved an extracted skill name to reference.skill at job
    ingestion time, so MatchResult.missing_skill_ids contains
    `str | None` entries (see engine.py:81). The unresolved name
    still matters for the recommender; we don't drop these records.

    `importance` is populated where the source data carries it.
    Per-layer status in this slice:
      - Layer A: reference.noc_skill.importance is a 0.0-5.0
        absolute scale per OaSIS. Populated.
      - Layer C: same OaSIS source as Layer A. Populated.
      - Layer B: ALWAYS None in this release. The engine's MatchResult
        does not expose per-missing-skill importance through its
        public shape (see engine.py:67-104). Populating it would
        require either modifying engine.py (forbidden -- matching-side
        change) or a separate SQL query per missing skill per
        posting. The unified shape ALLOWS importance for Layer B;
        the implementation simply does not populate it yet.
    When populated, the shape carries the RAW number; the recommender
    prompt is responsible for explaining the per-layer scale to the
    LLM (per-section prompt copy).

    Blocker classification
    ----------------------
    The `blocker` flag mirrors the same binary semantics Layer B
    already uses in production: True iff the skill name passes
    `is_credential_skill_name` (licence/cert keywords or Ontario
    trade-code regex). Same classifier for all three layers -- no
    fork.

    Storage
    -------
    Ephemeral. Computed per turn, passed to the recommender, then
    discarded. No StagedProfile field; no cookie impact. If a later
    release wants a compact follow-up snapshot for cross-turn
    reasoning, it would mirror the `last_match_snapshot` lifecycle
    pattern -- but first release does not need it.
    """
    layer: Layer
    source_id: str
    source_label: str
    skill_id: str | None
    skill_name: str
    blocker: bool
    importance: float | None
    source: GapSource


# =========================================================================
# Slice 2 (2026-06-18) -- Layer A: target NOC standard gap detector
# =========================================================================
# Compares the user's canonical skill IDs against the OaSIS NOC standard
# skill profile (`reference.noc_skill` JOIN `reference.skill` JOIN
# `reference.occupation` for the title). Returns a LayerAResult with:
#   - gaps:   tuple of GapEvidence records for every OaSIS standard
#             profile skill not matched against the user's resolved
#             skill_ids, deduplicated by skill_id. NOT framed as "skills
#             the user lacks" -- the OaSIS canonical may name broad
#             competencies (Reading Comprehension, Critical Thinking)
#             that a user demonstrates through experience without ever
#             carrying the canonical skill_id on their staged profile.
#             The recommender prompt frames these as occupation-standard
#             development areas, NOT as deficits.
#   - status: one of LayerAStatus values; honest about empty cases so the
#             recommender can distinguish "no OaSIS data" from "user has
#             all needed skills".
#
# Per the locked design:
#   - Exact 5-digit NOC match only. NO 4-digit fallback in first release.
#   - Same `is_credential_skill_name` classifier for the blocker flag.
#   - importance comes through as 0.0-5.0 from reference.noc_skill, or
#     None when the source row's importance is NULL.
#   - User skill source is the same canonical set the matcher uses;
#     callers pass the resolved `user_skill_ids` set so this module
#     stays decoupled from staging / alignment helpers.
#   - No side effects, no staged mutation. Pure read-and-derive.


@dataclass(frozen=True, slots=True)
class LayerAResult:
    """Result of one Layer A invocation. Honest about the empty cases:

      - status="ok" with empty gaps: the NOC has an OaSIS profile and
        the user already has every standard skill (or every standard
        skill resolves through the user_skill_ids set).
      - status="no_target_noc": caller did not supply a target NOC.
        No gaps to compute.
      - status="no_reference_skill_profile": target NOC is set but
        `reference.noc_skill` has no rows for it. The OaSIS data
        either doesn't include that NOC or hasn't been ingested.
    """
    gaps: tuple[GapEvidence, ...]
    status: LayerAStatus


# Single-query fetch: one round-trip for skill rows + skill names +
# the occupation title used as source_label. ORDER BY importance DESC
# is stable + makes the output deterministic for tests.
_LAYER_A_SQL: str = """
    SELECT ns.skill_id      AS skill_id,
           s.skill_name     AS skill_name,
           ns.importance    AS importance,
           o.title          AS noc_title
      FROM reference.noc_skill ns
      JOIN reference.skill s         ON s.skill_id = ns.skill_id
      JOIN reference.occupation o    ON o.noc_code = ns.noc_code
     WHERE ns.noc_code = %s
     ORDER BY ns.importance DESC NULLS LAST, s.skill_name
"""


def _fetch_noc_skill_rows(noc_code: str) -> list[dict[str, Any]]:
    """Run the OaSIS SQL and return row dicts. Used by both Layer A
    (target NOC) and Layer C (per adjacent NOC). Extracted as its own
    function so unit tests can monkeypatch the DB layer cleanly. No
    error handling here -- any DB exception propagates so the caller
    can decide whether to degrade gracefully."""
    with sync_cursor() as cur:
        cur.execute(_LAYER_A_SQL, (noc_code,))
        return list(cur.fetchall())


def _is_valid_noc_code(noc: str) -> bool:
    """Enforce the locked-design rule: exact 5-digit NOC, no 4-digit
    fallback, no alphanumeric values. NOC 2021 codes (which OaSIS
    keys on) are all 5 digits. Anything else either represents
    malformed cookie state, a planner bug, or a value the caller
    really shouldn't pass to the OaSIS layer.

    Used by Layer A (returns no_target_noc on invalid input) and
    Layer C (silently skips invalid noc_code entries from
    sideways_move). Inline validation rather than a hard exception
    keeps the gap-evidence subsystem honest-and-degrading rather
    than crashing on dirty input.
    """
    return len(noc) == 5 and noc.isdigit()


def _process_oasis_skill_rows(
    *,
    rows: list[dict[str, Any]],
    user_skill_ids: set[str],
    source_id: str,
    layer: Layer,
) -> tuple[list[GapEvidence], str]:
    """Shared row-processing logic for Layer A and Layer C. Both
    layers consume `reference.noc_skill JOIN reference.skill JOIN
    reference.occupation` rows identically -- the only differences
    are which `layer` discriminator each emits and what `source_id`
    each carries. Extracted here so Slice 4 doesn't duplicate
    Slice 2's logic.

    Returns:
        Tuple of (gaps, noc_title). The noc_title is captured from
        the first row's `noc_title` column (every row joins through
        the same occupation row, so any non-empty value will do).
        Empty string when no row had a usable noc_title.

    Defensive rules (mirrored from Slice 2 to preserve behavior):
      - skip rows with missing/empty skill_id or skill_name;
      - skip skills the user already has (via canonical skill_id);
      - dedupe by skill_id within the result set;
      - coerce importance to float; None / unparseable -> None.
    """
    gaps: list[GapEvidence] = []
    seen_skill_ids: set[str] = set()
    noc_title = ""
    for row in rows:
        skill_id = row.get("skill_id")
        skill_name = row.get("skill_name")
        if not isinstance(skill_id, str) or not skill_id:
            continue
        if not isinstance(skill_name, str) or not skill_name:
            continue
        if not noc_title:
            title_raw = row.get("noc_title")
            if isinstance(title_raw, str):
                noc_title = title_raw
        if skill_id in user_skill_ids:
            continue
        if skill_id in seen_skill_ids:
            continue
        seen_skill_ids.add(skill_id)

        importance_raw = row.get("importance")
        importance: float | None
        if importance_raw is None:
            importance = None
        else:
            try:
                importance = float(importance_raw)
            except (TypeError, ValueError):
                importance = None

        gaps.append(GapEvidence(
            layer=layer,
            source_id=source_id,
            source_label=noc_title,
            skill_id=skill_id,
            skill_name=skill_name,
            blocker=is_credential_skill_name(skill_name),
            importance=importance,
            source="reference.noc_skill",
        ))
    return gaps, noc_title


def compute_target_noc_standard_gaps(
    *,
    user_skill_ids: Iterable[str],
    target_noc: str | None,
) -> LayerAResult:
    """Layer A detector. See module-level Slice 2 comment for contract.

    Args:
        user_skill_ids: canonical skill IDs the user is known to have.
            Source must be the same as the matcher uses
            (build_user_skill_rows -> derive_user_skill_sets). Pass as
            any iterable; this function dedups via set() internally.
        target_noc: the user's resolved target NOC code (5 digits).
            None / empty / whitespace returns status="no_target_noc"
            with no gaps.

    Returns:
        LayerAResult with `gaps` populated by every OaSIS standard skill
        the user lacks, and `status` distinguishing the empty cases.
    """
    if target_noc is None:
        return LayerAResult(gaps=(), status="no_target_noc")
    noc = target_noc.strip() if isinstance(target_noc, str) else ""
    if not noc:
        return LayerAResult(gaps=(), status="no_target_noc")
    # Locked design: exact 5-digit NOC only. NO 4-digit fallback.
    # An invalid format (wrong length, non-digit characters) is
    # treated as no_target_noc -- honest about the unusable input
    # rather than firing a SQL query that's guaranteed to return
    # no rows.
    if not _is_valid_noc_code(noc):
        return LayerAResult(gaps=(), status="no_target_noc")

    rows = _fetch_noc_skill_rows(noc)
    if not rows:
        return LayerAResult(gaps=(), status="no_reference_skill_profile")

    user_ids: set[str] = {sid for sid in user_skill_ids if sid}
    gaps, _noc_title = _process_oasis_skill_rows(
        rows=rows,
        user_skill_ids=user_ids,
        source_id=noc,
        layer="target_noc_standard",
    )
    return LayerAResult(gaps=tuple(gaps), status="ok")


# =========================================================================
# Slice 3 (2026-06-18) -- Layer B: local SSM posting gap detector
# =========================================================================
# Reads `MatchResult.missing_skills` + `MatchResult.missing_skill_ids`
# directly from engine output (no new SQL, no engine changes). Emits
# one GapEvidence per missing skill per posting.
#
# Per the locked design:
#   - `skill_id` is nullable -- engine.py:81 returns
#     `list[str | None]` because the SCCC posting extractor may not
#     have resolved an extracted skill name to reference.skill at
#     job-ingestion time. The raw name still matters for the
#     recommender.
#   - `source_label` formats as "{title} @ {employer}" when employer
#     is set, falling back to title alone when not (matches the
#     pattern used in James's session live-verify logs).
#   - `importance` is None for Layer B in first release: the engine's
#     MatchResult doesn't carry per-missing-skill importance through
#     to its public shape. Adding it would require either touching
#     engine.py (forbidden by lock) or a separate SQL query per
#     missing skill per posting (expensive). The locked design
#     explicitly allows null importance.
#   - Same `is_credential_skill_name` classifier for the blocker flag.
#   - Returns a plain `tuple[GapEvidence, ...]` -- no status wrapper
#     needed because the empty-match-results case is naturally an
#     empty tuple. (Contrast with Layer A's three-status result:
#     OaSIS-missing vs user-has-everything are distinguishable cases
#     for Layer A but irrelevant for Layer B.)


def compute_local_posting_gaps(
    *,
    match_results: Iterable[Any],
) -> tuple[GapEvidence, ...]:
    """Layer B detector. See module-level Slice 3 comment for contract.

    Args:
        match_results: any iterable of engine MatchResult instances
            (or duck-typed objects exposing the same attributes).
            None entries and malformed entries are silently skipped.

    Returns:
        Tuple of GapEvidence records, one per (posting, missing-skill)
        pair, deduplicated within each posting by (skill_id, lowercase
        name). Order is preserved from the engine's missing list per
        posting; postings come in the iteration order of `match_results`.
    """
    out: list[GapEvidence] = []
    for mr in match_results:
        if mr is None:
            continue
        job_id = getattr(mr, "job_id", None)
        if not isinstance(job_id, str) or not job_id:
            continue

        title_raw = getattr(mr, "title", None)
        title = title_raw if isinstance(title_raw, str) and title_raw else ""
        employer_raw = getattr(mr, "employer", None)
        employer = (
            employer_raw
            if isinstance(employer_raw, str) and employer_raw
            else ""
        )
        if title and employer:
            source_label = f"{title} @ {employer}"
        elif title:
            source_label = title
        elif employer:
            source_label = employer
        else:
            source_label = "(untitled posting)"

        names_raw = getattr(mr, "missing_skills", None) or []
        ids_raw = getattr(mr, "missing_skill_ids", None) or []
        names = list(names_raw) if isinstance(names_raw, (list, tuple)) else []
        ids = list(ids_raw) if isinstance(ids_raw, (list, tuple)) else []

        # Dedupe per posting on (skill_id, lowercase name). Two records
        # with the same skill_id are duplicates by construction; when
        # skill_id is None we fall back to the name. The same skill
        # surfacing on two DIFFERENT postings is NOT deduplicated --
        # the recommender wants to know which postings each gap appears
        # in.
        seen: set[tuple[str | None, str]] = set()
        for i, name in enumerate(names):
            if not isinstance(name, str):
                continue
            name_stripped = name.strip()
            if not name_stripped:
                continue
            skill_id_raw = ids[i] if i < len(ids) else None
            skill_id = (
                skill_id_raw
                if isinstance(skill_id_raw, str) and skill_id_raw
                else None
            )
            key = (skill_id, name_stripped.lower())
            if key in seen:
                continue
            seen.add(key)

            out.append(GapEvidence(
                layer="local_posting",
                source_id=job_id,
                source_label=source_label,
                skill_id=skill_id,
                skill_name=name_stripped,
                blocker=is_credential_skill_name(name_stripped),
                importance=None,
                source="extracted.job_skill",
            ))
    return tuple(out)


# =========================================================================
# Slice 4 (2026-06-18) -- Layer C: adjacent NOC standard gap detector
# =========================================================================
# Fan-out version of Layer A. For every UNIQUE NOC code present in
# CP5's tier_evidence.sideways_move (the final surfaced 1-3 adjacent
# jobs, NOT raw `accepted_adjacent`), compare the user's skills
# against that NOC's OaSIS standard profile.
#
# Per the locked design:
#   - Source: same as Layer A -- reference.noc_skill JOIN reference.skill
#     JOIN reference.occupation. Reuses the _process_oasis_skill_rows
#     helper.
#   - Exact 5-digit NOC match only. NO 4-digit fallback in first
#     release. NOCs with no rows in reference.noc_skill surface as
#     a per-NOC slice with status="no_reference_skill_profile".
#   - Cap: no separate cap. CP5 already caps surfaced jobs at
#     MAX_ADJACENT_ITEMS=3, so the unique-NOC count is naturally
#     bounded at 1-3.
#   - Layer C does NOT modify or feed back into CP5. Input is
#     read-only.
#
# Output shape: tuple of per-NOC slices. Each slice carries the
# NOC code, the OaSIS-resolved NOC title, the gaps for that NOC,
# and a status mirroring LayerAStatus. The recommender groups the
# whole tuple under ADJACENT_NOC_STANDARD_GAPS in the prompt.


@dataclass(frozen=True, slots=True)
class LayerCNocSlice:
    """Per-NOC slice of Layer C output. One per unique NOC surfaced
    in CP5's sideways_move.

    `status` mirrors LayerAStatus semantics:
      - "ok": OaSIS profile exists for this NOC; `gaps` may be empty
        (user already has every standard skill) or non-empty.
      - "no_reference_skill_profile": NOC was in sideways_move but
        `reference.noc_skill` has no rows for it. `gaps` is empty;
        the recommender should narrate honestly ("no OaSIS profile
        available for that adjacent occupation").
      - "no_target_noc": never produced by Layer C (the NOC is
        always present by construction -- extracted from
        sideways_move). Included in the type for symmetry with
        LayerAStatus only.
    """
    noc_code: str
    noc_label: str
    gaps: tuple[GapEvidence, ...]
    status: LayerAStatus


def compute_adjacent_noc_standard_gaps(
    *,
    user_skill_ids: Iterable[str],
    sideways_move: Iterable[Any],
) -> tuple[LayerCNocSlice, ...]:
    """Layer C detector. See module-level Slice 4 comment for contract.

    Args:
        user_skill_ids: same canonical skill ID source as Layer A
            (build_user_skill_rows -> derive_user_skill_sets).
        sideways_move: the final CP5-surfaced adjacent jobs (typically
            `tier_evidence.sideways_move`). Each entry should expose
            a `noc_code` attribute. None entries and entries with
            None / empty / non-string noc_code are silently skipped.

    Returns:
        Tuple of LayerCNocSlice records, one per UNIQUE NOC code,
        in first-seen order. Empty tuple when sideways_move is empty
        or carries no valid NOC codes.
    """
    unique_nocs: list[str] = []
    seen_nocs: set[str] = set()
    for job in sideways_move:
        if job is None:
            continue
        noc_raw = getattr(job, "noc_code", None)
        if not isinstance(noc_raw, str):
            continue
        noc = noc_raw.strip()
        if not noc or noc in seen_nocs:
            continue
        # Locked design: exact 5-digit NOC only. NO 4-digit fallback.
        # Adjacent jobs with malformed or non-5-digit noc_code are
        # silently skipped here -- a job from sideways_move with a
        # bad NOC code can still appear in the surfaced tier, but
        # Layer C cannot produce a per-NOC slice for it.
        if not _is_valid_noc_code(noc):
            continue
        seen_nocs.add(noc)
        unique_nocs.append(noc)

    if not unique_nocs:
        return ()

    user_ids: set[str] = {sid for sid in user_skill_ids if sid}

    out: list[LayerCNocSlice] = []
    for noc in unique_nocs:
        rows = _fetch_noc_skill_rows(noc)
        if not rows:
            out.append(LayerCNocSlice(
                noc_code=noc,
                noc_label="",
                gaps=(),
                status="no_reference_skill_profile",
            ))
            continue
        gaps, noc_title = _process_oasis_skill_rows(
            rows=rows,
            user_skill_ids=user_ids,
            source_id=noc,
            layer="adjacent_noc_standard",
        )
        out.append(LayerCNocSlice(
            noc_code=noc,
            noc_label=noc_title,
            gaps=tuple(gaps),
            status="ok",
        ))
    return tuple(out)
