"""JobMatchEngine v1.0.0 — the core of the SkillBridge product.

Auditable, deterministic, config-driven. Formula:
  skill_score = matched_top_n / required_top_n
  + recency boost (if posted_date within recency_boost_days)
  + location boost (if region matches local CSD)
  + small target-role similarity boost

Eligibility:
  - Job skills below min_extraction_confidence are dropped.
  - Top-N highest-importance remaining skills become the scoring set.
  - If fewer than min_required_skills_for_eligibility skills remain,
    match_eligible = False (excluded from ranked recommendations, still
    shown on dashboard).

Bands:
  >= band_strong  -> strong
  >= band_good    -> good
  >= band_stretch -> stretch
  else            -> low

Credential warning is attached when the job's NOC or target_role maps to a
core.regulated_occupation row.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date
from typing import Iterable, NamedTuple

from rapidfuzz import fuzz

from config import (
    EMBEDDING_MODEL_VERSION,
    LOCAL_CITIES,
    MATCH,
    MIN_EXTRACTION_CONFIDENCE,
    SEMANTIC_COSINE_THRESHOLD,
    SEMANTIC_STRENGTH_CAP,
)
from skillbridge.db import sync_cursor
from skillbridge.match.aliases import canonicalize_skill
from skillbridge.match.alignment import (
    SkillAlignment,
    UserSkillRow,
    build_user_skill_rows,
    derive_user_skill_sets,
    is_normalized_equal,
)
from skillbridge.match.occupation import resolve_title_to_noc
from skillbridge.session.staging import StagedProfile
from skillbridge.versions import ENGINE_VERSION_JOB_MATCH

# Matching v2 step 2: occupation-match boost. Applied when the user's
# target NOC equals the job's NOC. +0.10 is large enough to matter
# (about the same as the location boost) but small enough that a
# strong skill ratio still dominates the score -- you can't get a
# "strong" match purely on NOC match without skill overlap.
_TARGET_NOC_BOOST = 0.10

log = logging.getLogger(__name__)


@dataclass
class MatchResult:
    job_id: str
    profile_id: str
    title: str
    employer: str | None
    url: str | None
    location: str | None
    match_score: float
    match_band: str
    match_eligible: bool
    ineligibility_reason: str | None
    matched_skills: list[str]
    missing_skills: list[str]
    matched_skill_ids: list[str | None]
    missing_skill_ids: list[str | None]
    required_skills_count: int
    credential_warning: str | None
    posted_date: date | None
    noc_code: str | None
    # Sprint 1: structured "why this score" components. Surfaced to the
    # responder so any "because" clause it generates must reference one
    # of these signals, not invent causality. See docs/resume-design.md
    # §5 — coaching is allowed, but justifying coaching with grounded
    # facts only.
    score_explanation: dict | None = None
    # AR-9.feat.coach-tiers CP1: per-requirement provenance. One entry
    # per satisfied job requirement; the responder reads `user_skill`
    # for "your X" and `job_requirement` for "their Y", and gates the
    # reserved phrasing "they ask for X, which you have" on
    # `is_normalized_equal`. Empty tuple when the matcher was called
    # without user_rows (legacy / tests).
    skill_alignment: tuple[SkillAlignment, ...] = ()
    # AR-9.feat.coach-tiers CP1: raw job-source fields pulled through
    # for tier prose. Both default to None; ingestion populates them
    # when the source carries structured values.
    employment_type: str | None = None
    salary_text: str | None = None


# ----------------------------------------------------- Internal SQL helpers
def _fetch_user_skills(profile_id: str) -> list[dict]:
    with sync_cursor() as cur:
        cur.execute(
            "SELECT skill_id, skill_name FROM profile.user_skill WHERE profile_id = %s",
            (profile_id,),
        )
        return list(cur.fetchall())


def _fetch_profile(profile_id: str) -> dict | None:
    with sync_cursor() as cur:
        cur.execute(
            """
            SELECT profile_id, preferred_location, target_role_text,
                   work_type_preference, shift_preference,
                   language_preferences, experience_text
              FROM profile.user_profile
             WHERE profile_id = %s AND deleted_at IS NULL
            """,
            (profile_id,),
        )
        return cur.fetchone()


def _fetch_eligible_jobs() -> list[dict]:
    with sync_cursor() as cur:
        cur.execute("SELECT * FROM core.v_current_job ORDER BY posted_date DESC NULLS LAST")
        return list(cur.fetchall())


def _maybe_embed_user_skill_rows(user_rows: list[UserSkillRow]):
    """Encode the user's skill phrases into a (N, dim) matrix in
    the order of `user_rows`.

    AR-9.feat.coach-tiers CP1: ordering matters. The semantic-argmax
    path in `_classify_match` reads back `user_rows[winning_index]`
    to attribute the win, so row i in the matrix MUST correspond to
    `user_rows[i].name`. Prior to CP1 the matrix was built from
    `sorted(user_skill_names)` which made argmax-to-row mapping
    impossible.

    Returns None when:
      - user_rows is empty
      - sentence-transformers isn't installed (get_embedder returns None)
      - encoding fails for any reason (logged as warning; engine falls
        back to lexical-only without disrupting the chat)

    On success: an L2-normalized float32 numpy array. Consumed by
    `_semantic_match_strength` via one dot product per job-side skill.
    """
    if not user_rows:
        return None
    # Lazy import: get_embedder probes for sentence-transformers and
    # returns None when missing, so importing the service module itself
    # is cheap.
    from skillbridge.embed.service import get_embedder
    embedder = get_embedder()
    if embedder is None:
        return None
    try:
        return embedder.encode_many([r.name for r in user_rows])
    except Exception as e:
        log.warning("user-side embedding failed: %s -- falling back to lexical-only", e)
        return None


def _fetch_job_skill_embeddings(job_id: str):
    """Step 5d: pull pre-computed embeddings for one job's skills.

    Returns a dict keyed by lowercased skill_name -> numpy array.
    Empty dict when no embeddings exist (e.g. before --embed has run,
    or when sentence-transformers wasn't installed at embed time).

    Only fetches the CURRENT model_version. If a future release ships
    a new embedding model, old vectors stay in the table but the
    engine ignores them (clean rollover).
    """
    # The embedding table is itself optional -- the schema migration
    # skips creating it when pgvector isn't installed (see
    # sql/schema.sql DO $$ EXCEPTION wrap). Catching UndefinedTable
    # here lets the engine fall back to lexical-only when semantic
    # infrastructure isn't deployed yet. Same graceful-degradation
    # pattern as the sentence-transformers soft dependency.
    import psycopg
    try:
        with sync_cursor() as cur:
            cur.execute(
                """
                SELECT skill_name, embedding
                  FROM extracted.job_skill_embedding
                 WHERE job_id = %s AND model_version = %s
                """,
                (job_id, EMBEDDING_MODEL_VERSION),
            )
            rows = list(cur.fetchall())
    except psycopg.errors.UndefinedTable:
        return {}
    if not rows:
        return {}
    # Local import so engine module imports don't pay the numpy cost
    # for callers that don't use semantic matching at all.
    import numpy as np
    out: dict[str, "np.ndarray"] = {}
    for r in rows:
        emb = r["embedding"]
        # pgvector returns either a list or a string depending on
        # driver settings; normalize to numpy float32 vector.
        if isinstance(emb, str):
            # "[0.1,0.2,...]" -> list of floats
            emb = [float(x) for x in emb.strip("[]").split(",") if x.strip()]
        out[(r["skill_name"] or "").lower()] = np.asarray(emb, dtype=np.float32)
    return out


def _fetch_job_skills(job_id: str) -> list[dict]:
    with sync_cursor() as cur:
        cur.execute(
            """
            SELECT skill_id, skill_name, confidence, importance_rank, skill_type
              FROM extracted.job_skill
             WHERE job_id = %s
             ORDER BY importance_rank NULLS LAST, confidence DESC
            """,
            (job_id,),
        )
        return list(cur.fetchall())


def _required_or_preferred(skill: dict) -> str:
    """Bucket a JD-side skill into required/preferred.

    NULL/missing/unknown values become 'required' -- the Sprint 5 step 3
    contract: when in doubt, treat as required. Keeps legacy rows (pre-
    skill_type column) scoring exactly as v1.1 did.
    """
    raw = (skill.get("skill_type") or "").strip().lower()
    return "preferred" if raw == "preferred" else "required"


# Required/preferred scoring weights -- Sprint 5 step 3.
#
# skill_score = REQ_WEIGHT * required_match_ratio
#             + PREF_WEIGHT * preferred_match_ratio
#
# Required is the dominant signal (80%); preferred is a nudge (20%). Edge
# cases:
#   - Job has only required skills (all NULL on legacy rows count here):
#       base = required_match_ratio   -- identical to v1.1.
#   - Job has only preferred skills (unusual but possible):
#       base = preferred_match_ratio  -- a candidate matching every
#       preferred-only skill should still rank.
#   - Job has neither (empty extraction):
#       base = 0.0.
_REQ_WEIGHT = 0.8
_PREF_WEIGHT = 0.2


def _weighted_skill_base(
    matched_req: float, total_req: int,
    matched_pref: float, total_pref: int,
) -> tuple[float, float, float]:
    """Compute the weighted base score plus the two component ratios.

    Step 4a refactor: the `matched_*` args are now strength SUMS (floats),
    not match COUNTS (ints). When every match fires through a stage-1
    path (exact / canonical / word-bounded substring) the strength is
    1.0 and the sum equals the count -- pre-refactor numbers are
    preserved exactly. Stages that contribute partial strength (token-
    overlap fuzzy = 0.85; future semantic re-ranker = variable) yield
    lower ratios, which is the whole point of the refactor: fuzzy
    matches contribute proportionally less to the band.

    Returns (base, required_ratio, preferred_ratio). The component ratios
    are surfaced separately in score_explanation so the responder can
    quote them honestly.
    """
    req_ratio = (matched_req / total_req) if total_req else 0.0
    pref_ratio = (matched_pref / total_pref) if total_pref else 0.0

    if total_req == 0 and total_pref == 0:
        base = 0.0
    elif total_req == 0:
        base = pref_ratio
    elif total_pref == 0:
        base = req_ratio
    else:
        base = _REQ_WEIGHT * req_ratio + _PREF_WEIGHT * pref_ratio

    return base, req_ratio, pref_ratio


# Per-stage match strengths. Step 4a refactor (continuous strength).
# Stage 1 (exact / canonical / word-bounded substring) is "definitely
# this skill"; stage 2 (token-overlap fuzzy) is "probably this skill,
# but the user's phrasing differs". Step 5 will add a Stage 3 from
# embedding cosine similarity with variable strength.
_STRENGTH_STAGE_1 = 1.0
_STRENGTH_TOKEN_OVERLAP = 0.85


# AR-9.feat.coach-tiers CP1: internal-only sub-stage rungs for the
# `skill_alignment` attribution layer. Public stage taxonomy collapses
# the first four into "exact"; these labels exist so attribution can
# pick a deterministic winner when multiple user skills satisfy the
# same requirement (rung priority first, then lex-asc on row.text).
_RUNG_NO_MATCH = -1
_RUNG_SKILL_ID = 0
_RUNG_NAME_EQ = 1
_RUNG_CANON = 2
_RUNG_SUBSTRING = 3
_RUNG_FUZZY = 4
_RUNG_SEMANTIC = 5


def _semantic_match_strength(
    job_skill_embedding,
    user_embeddings_matrix,
    *,
    tiebreak_keys: list[str] | None = None,
) -> tuple[float, int | None]:
    """Compute the best-cosine semantic strength between one job-side
    skill embedding and the user's full embedding matrix, and report
    the WINNING ROW INDEX.

    Returns `(strength, winning_index)`:
      - strength is SEMANTIC_STRENGTH_CAP when max cosine >=
        SEMANTIC_COSINE_THRESHOLD, else 0.0.
      - winning_index is the row index into user_embeddings_matrix
        where the maximum cosine occurred. None when strength is 0.0.

    Tie-break (AR-9.feat.coach-tiers): when multiple rows share the
    max cosine value, `tiebreak_keys` (parallel-indexed to the matrix
    rows) picks the lex-smallest. When `tiebreak_keys` is None, the
    lowest row index wins (numpy argmax default).

    The flat strength mapping (cap vs zero, no continuous gradient)
    keeps the contract simple and capped well below the lexical fuzzy
    tier (0.85) so semantic never overrides a lexical fuzzy match.

    Inputs are assumed L2-normalized; cosine reduces to dot product.
    Returns (0.0, None) when either embedding is None or empty.
    """
    if job_skill_embedding is None or user_embeddings_matrix is None:
        return 0.0, None
    # numpy import is local so the engine itself doesn't pay the
    # cost when semantic isn't in use.
    import numpy as np
    if user_embeddings_matrix.shape[0] == 0:
        return 0.0, None
    sims = user_embeddings_matrix @ np.asarray(job_skill_embedding)
    best = float(sims.max())
    if best < SEMANTIC_COSINE_THRESHOLD:
        return 0.0, None
    # Identify every row tied at the max cosine value. Float32 equality
    # comparison is sufficient because the matrix values are derived
    # from the same normalized encoder pass — ties only happen when
    # two user skills truly share an identical embedding (e.g. the
    # user typed the same skill twice).
    tied: list[int] = [
        i for i in range(int(sims.shape[0])) if float(sims[i]) == best
    ]
    if len(tied) == 1:
        return SEMANTIC_STRENGTH_CAP, tied[0]
    if tiebreak_keys is not None and len(tiebreak_keys) > max(tied):
        winner = min(tied, key=lambda i: tiebreak_keys[i])
        return SEMANTIC_STRENGTH_CAP, winner
    return SEMANTIC_STRENGTH_CAP, tied[0]


def _pick_winner_lex(
    user_rows: list[UserSkillRow] | None,
    predicate,
) -> str | None:
    """Walk `user_rows` (preserving caller-supplied order), collect
    rows for which `predicate(row)` is True, return the lex-smallest
    row.text. Returns None when `user_rows` is None or no row matches.
    """
    if user_rows is None:
        return None
    candidates = [r for r in user_rows if predicate(r)]
    if not candidates:
        return None
    return min(candidates, key=lambda r: r.text).text


def _classify_match(
    job_skill: dict,
    user_skill_ids: set[str],
    user_skill_names: set[str],
    user_skill_names_canon: set[str],
    *,
    user_rows: list[UserSkillRow] | None = None,
    job_skill_embedding=None,
    user_embeddings_matrix=None,
) -> tuple[float, str, str | None, int]:
    """Single attribution-aware matcher. Returns
    `(strength, stage, matched_user_text, internal_rung)`.

    When `user_rows` is provided, `matched_user_text` is the original
    user-typed text of the row that produced the win (UserSkillRow.text
    — already .strip()ed by build_user_skill_rows) and `internal_rung`
    identifies which sub-stage produced it. When `user_rows` is None,
    both default to None / _RUNG_NO_MATCH and behavior is byte-stable
    for legacy callers.

    Internal rung order for attribution tie-breaking (NOT visible in
    the public stage label):
      _RUNG_SKILL_ID < _RUNG_NAME_EQ < _RUNG_CANON < _RUNG_SUBSTRING
        < _RUNG_FUZZY < _RUNG_SEMANTIC
    Within a rung: lex-asc on row.text. For semantic specifically, the
    HIGHEST cosine wins first; lex-asc on row.text only breaks ties at
    the max cosine.

    Public stage taxonomy (unchanged from prior shape):
      "exact"     — skill_id / name eq / canonical alias / word-bounded substring
      "fuzzy"     — token-overlap >= threshold
      "semantic"  — cosine >= threshold
      "no_match"  — strength 0.0

    CREDENTIAL-STRICT PATH (round-31 R-2 credential safety, preserved):
    When the job-side skill is a credential (Class G, 310S, WHMIS,
    etc.), only the first three rungs are allowed. Substring / fuzzy /
    semantic are DISABLED so "G2/G driver's license" cannot silently
    satisfy a mandatory Class G requirement.
    """
    sid = job_skill.get("skill_id")
    sname = (job_skill.get("skill_name") or "").lower()

    # ---------- rung 0: skill_id ----------
    if sid and sid in user_skill_ids:
        text = _pick_winner_lex(
            user_rows, lambda r: r.skill_id is not None and r.skill_id == sid
        )
        return _STRENGTH_STAGE_1, "exact", text, _RUNG_SKILL_ID

    # ---------- rung 1: normalized name equality ----------
    if sname in user_skill_names:
        text = _pick_winner_lex(user_rows, lambda r: r.name == sname)
        return _STRENGTH_STAGE_1, "exact", text, _RUNG_NAME_EQ

    # ---------- rung 2: canonical alias ----------
    sname_canon = canonicalize_skill(sname)
    if sname_canon and sname_canon in user_skill_names_canon:
        text = _pick_winner_lex(
            user_rows, lambda r: bool(r.canon) and r.canon == sname_canon
        )
        return _STRENGTH_STAGE_1, "exact", text, _RUNG_CANON

    # Credential-strict gate: stop here for credential-class skills.
    if is_credential_skill_name(sname):
        return 0.0, "no_match", None, _RUNG_NO_MATCH

    # ---------- rung 3: word-bounded substring ----------
    for uname in user_skill_names:
        if _word_bounded_in(uname, sname) or _word_bounded_in(sname, uname):
            text = _pick_winner_lex(
                user_rows,
                lambda r: _word_bounded_in(r.name, sname)
                or _word_bounded_in(sname, r.name),
            )
            return _STRENGTH_STAGE_1, "exact", text, _RUNG_SUBSTRING

    # ---------- rung 4: token-overlap fuzzy ----------
    j_tokens = _skill_tokens(sname)
    if j_tokens:
        fuzzy_names: set[str] = set()
        for uname in user_skill_names:
            u_tokens = _skill_tokens(uname)
            if not u_tokens:
                continue
            overlap = u_tokens & j_tokens
            if len(overlap) / len(j_tokens) >= _SKILL_TOKEN_OVERLAP_THRESHOLD:
                fuzzy_names.add(uname)
        if fuzzy_names:
            text = _pick_winner_lex(user_rows, lambda r: r.name in fuzzy_names)
            return _STRENGTH_TOKEN_OVERLAP, "fuzzy", text, _RUNG_FUZZY

    # ---------- rung 5: semantic fallback ----------
    tiebreak_keys = (
        [r.text for r in user_rows] if user_rows is not None else None
    )
    sem_strength, sem_index = _semantic_match_strength(
        job_skill_embedding, user_embeddings_matrix,
        tiebreak_keys=tiebreak_keys,
    )
    if sem_strength > 0:
        text = (
            user_rows[sem_index].text
            if (user_rows is not None and sem_index is not None
                and sem_index < len(user_rows))
            else None
        )
        return sem_strength, "semantic", text, _RUNG_SEMANTIC

    return 0.0, "no_match", None, _RUNG_NO_MATCH


def _skill_match_strength(
    job_skill: dict,
    user_skill_ids: set[str],
    user_skill_names: set[str],
    user_skill_names_canon: set[str],
    *,
    job_skill_embedding=None,
    user_embeddings_matrix=None,
) -> tuple[float, str]:
    """Legacy 2-tuple wrapper over `_classify_match`. Keeps every
    pre-CP1 caller byte-stable while the attribution-aware path
    coexists. Both functions share a single internal implementation
    so scoring and attribution cannot disagree.
    """
    strength, stage, _user_text, _rung = _classify_match(
        job_skill,
        user_skill_ids,
        user_skill_names,
        user_skill_names_canon,
        job_skill_embedding=job_skill_embedding,
        user_embeddings_matrix=user_embeddings_matrix,
    )
    return strength, stage


class _ClassifiedRequirement(NamedTuple):
    """One job-side requirement's matcher output, kept alongside the
    original job_skill dict so `_score_one_job` can populate its
    scoring dicts without re-invoking the matcher. Internal contract.
    """
    skill: dict          # the job-side skill row, verbatim
    strength: float      # 0.0 .. 1.0
    stage: str           # "exact" | "fuzzy" | "semantic" | "no_match"


def build_skill_alignment(
    job_skills: list[dict],
    user_rows: list[UserSkillRow] | None,
    user_skill_ids: set[str],
    user_skill_names: set[str],
    user_skill_names_canon: set[str],
    *,
    user_embeddings_matrix=None,
    job_skill_embeddings: dict | None = None,
) -> tuple[list[SkillAlignment], list[_ClassifiedRequirement]]:
    """Shared alignment-building helper. ONE matcher pass per requirement.

    AR-9.feat.coach-tiers CP1: the single source of truth for both
    `_score_one_job` branches AND the adjacency-enrichment path. The
    duplicated inline construction that used to live in two scoring
    branches lives here; adjacency reuses this helper directly without
    re-running scoring, boosts, hard gates, or the regulated-occupation
    lookup.

    Returns `(alignments, classifications)`:
      - alignments: SkillAlignment list, populated only when user_rows
        is provided (legacy callers get an empty list).
      - classifications: per-requirement (skill, strength, stage) in
        input order. _score_one_job consumes these to populate its
        existing scoring dicts without re-invoking the matcher.

    The helper does NOT compute score, boosts, bands, hard gates,
    credential warnings, or any non-alignment side effect. Those
    belong to `_score_one_job` and stay there.
    """
    alignments: list[SkillAlignment] = []
    classifications: list[_ClassifiedRequirement] = []
    embeddings = job_skill_embeddings or {}
    for s in job_skills:
        key = (s.get("skill_name") or "").lower()
        job_emb = embeddings.get(key) if embeddings else None
        strength, stage, matched_text, _rung = _classify_match(
            s,
            user_skill_ids,
            user_skill_names,
            user_skill_names_canon,
            user_rows=user_rows,
            job_skill_embedding=job_emb,
            user_embeddings_matrix=user_embeddings_matrix,
        )
        classifications.append(_ClassifiedRequirement(s, strength, stage))
        if (
            user_rows is not None
            and strength > 0
            and matched_text is not None
            and stage in ("exact", "fuzzy", "semantic")
        ):
            bucket = _required_or_preferred(s)
            job_req_name = s.get("skill_name") or ""
            alignments.append(SkillAlignment(
                user_skill=matched_text,
                job_requirement=job_req_name,
                stage=stage,  # type: ignore[arg-type]
                source="required" if bucket == "required" else "preferred",
                is_normalized_equal=is_normalized_equal(matched_text, job_req_name),
            ))
    return alignments, classifications


def _skill_base_mode(total_req: int, total_pref: int) -> str:
    """Name the branch _weighted_skill_base took, for score_explanation.

    'weighted'        -- standard 0.8/0.2 mix (both required and preferred present)
    'required_only'   -- job has only required skills (legacy NULL rows land here)
    'preferred_only'  -- job has only preferred skills (rare but possible)
    'empty'           -- job had no skills extracted at all
    """
    if total_req == 0 and total_pref == 0:
        return "empty"
    if total_pref == 0:
        return "required_only"
    if total_req == 0:
        return "preferred_only"
    return "weighted"


def _regulated(noc_code: str | None, target_role: str | None) -> dict | None:
    if not noc_code and not target_role:
        return None
    with sync_cursor() as cur:
        if noc_code:
            cur.execute(
                "SELECT * FROM core.regulated_occupation WHERE noc_code = %s",
                (noc_code,),
            )
            row = cur.fetchone()
            if row:
                return row
        if target_role:
            cur.execute(
                """
                SELECT * FROM core.regulated_occupation
                 WHERE similarity(occupation_title, %s) > 0.5
                 ORDER BY similarity(occupation_title, %s) DESC
                 LIMIT 1
                """,
                (target_role, target_role),
            )
            return cur.fetchone()
    return None


# ----------------------------------------------------------- Scoring
# CP3 step 2 (2026-06-15): soft-trait skill names the job-skill
# extractor sometimes captures as `required` ("Phil's Delivery Service"
# lists "reliability" as a required skill at confidence 0.85). The
# chat-side extractor refuses to credit these as user claims because
# they are not verifiable hard skills, so the matcher ends up showing
# them as `required_missing` regardless of what the user says. Filter
# them out of the job's required set entirely — they should not be
# barriers to Apply today, they should not be gaps in Worth a try.
#
# Limited to clear work-attitude items. Borderline cases (time
# management, organizational skills, written communication, problem
# solving) are KEPT because they are teachable and demonstrable.
_SOFT_TRAIT_SKILL_NAMES: frozenset[str] = frozenset({
    "reliability",
    "punctuality",
    "attention to detail",
    "multitasking",
    "multitasking and prioritization",
    "working under pressure",
    "initiative",
    "self-motivated",
    "self motivated",
    "self-motivation",
    "positive attitude",
})


def _is_soft_trait_skill_name(name: str | None) -> bool:
    if not isinstance(name, str):
        return False
    return name.strip().lower() in _SOFT_TRAIT_SKILL_NAMES


def _filter_eligible_skills(job_skills: list[dict]) -> list[dict]:
    """Drop low-confidence skills; take top N by importance.

    Sprint 5 step 4 carve-out: skills that look like licences or
    certifications (Class G, WHMIS, First Aid, 310T, etc.) are kept
    regardless of importance_rank. The v1.2.0 extraction frequently
    ranks credentials lower than core duties (rank 9 for "Class G
    driver's license" on the truck/coach posting), which would let
    them fall off the top-N cutoff and silently break the credential
    cap. Confidence threshold still applies to credentials.

    CP3 step 2: soft traits (reliability, attention to detail, etc.)
    are dropped here. The job-skill extractor captures them at ingest
    because postings call them out, but the chat-side extractor will
    not credit them as user claims (they are not verifiable hard
    skills), so they would otherwise appear as permanent gaps that
    the user can never close. Removing them from the required set
    makes Apply today reachable when the user covers every actual
    teachable skill.
    """
    kept = [
        s for s in job_skills
        if (s.get("confidence") is None
            or float(s["confidence"]) >= MIN_EXTRACTION_CONFIDENCE)
        and not _is_soft_trait_skill_name(s.get("skill_name"))
    ]
    kept.sort(key=lambda s: (s.get("importance_rank") or 99, -(float(s.get("confidence") or 0))))
    top_n = kept[: MATCH.top_n_required_skills]

    # Carve-out: any credential anywhere in the (confidence-filtered)
    # set is kept, even if it ranked below top_n. Dedup by skill_name
    # so a credential that ALSO made top_n isn't counted twice.
    #
    # Round-33: gate on `is_credential_skill_name(...)` rather than the
    # bare keyword substring. The keyword check misses bare trade
    # codes (a job_skill row stored as "310S" / "310T" / "433A"
    # contains no licence/cert vocabulary). A low-ranked bare trade
    # code would otherwise fall off the top-N cutoff and the
    # credential cap would silently fail to fire on the engine
    # downstream. Same classifier the strict-match gate uses.
    top_n_names = {s.get("skill_name") for s in top_n}
    carved = [
        s for s in kept
        if s not in top_n
        and is_credential_skill_name(s.get("skill_name"))
        and s.get("skill_name") not in top_n_names
    ]
    return top_n + carved


def _band(score: float) -> str:
    if score >= MATCH.band_strong:
        return "strong"
    if score >= MATCH.band_good:
        return "good"
    if score >= MATCH.band_stretch:
        return "stretch"
    return "low"


# Licence / certification keywords. If a missing skill matches one of
# these, the match band is capped at "stretch" — you can't be a "strong"
# match for a job you literally don't have a required licence for, no
# matter how many other skills overlap. This catches the common case of
# a Class G license being a hard gate for transport / delivery roles.
_LICENCE_OR_CERT_KEYWORDS: tuple[str, ...] = (
    "licence", "license", "certification", "certificate", "cert",
    "ticket", "endorsement", "permit",
    "whmis", "first aid", "cpr",
    "class g", "class a", "class b", "class c", "class d", "class e",
    "class f", "class z", "air brake", "z endorsement",
)


# Round-32 credential-safety rule: Ontario apprenticeship trade codes
# are 3 digits followed by one letter (310S auto tech, 310T truck/coach,
# 309A electrician, 433A boilermaker, 442A industrial electrician,
# 313A plumber, etc.). A job skill listed as the bare code -- "310S"
# rather than "310S license" -- otherwise misses the keyword check
# above and falls through to fuzzy / substring matching. That lets
# a non-credential user skill like "310S automotive experience"
# silently satisfy the requirement via word-bounded substring.
#
# The pattern requires word boundaries so common product / version
# strings like "Python 3", "Windows 10", "AWS S3", "ISO 9001" do NOT
# match (insufficient digits, wrong shape, or no isolated 3-digit+letter
# token). Conservative even for novel codes: a false positive only
# means "this skill is treated as credential-strict" which is the
# safer default for ambiguous tokens.
_TRADE_CODE_PATTERN = re.compile(r"\b\d{3}[A-Za-z]\b")


def is_credential_skill_name(name: str | None) -> bool:
    """True if a skill name looks like a licence / certification.

    Shared by the matching engine (credential cap + filter carve-out),
    the responder (narration cap force-include), and any future caller
    that needs to identify credential-class skills consistently. Single
    source of truth for the keyword set.

    Detection has two arms:
      1. Word-substring against `_LICENCE_OR_CERT_KEYWORDS` (the
         long-standing vocabulary set).
      2. Regex match against `_TRADE_CODE_PATTERN` -- Ontario
         apprenticeship trade codes like 310S / 310T / 309A / 433A.
         A bare trade code IS the credential's identity even without
         "license" / "certification" appended; the matcher's
         credential-strict gate must catch it.

    Longer term, the engine should carry an explicit `is_credential`
    flag on job_skill rows. For now this keyword + pattern set is the
    single source of truth.
    """
    if not name:
        return False
    low = name.lower()
    if any(kw in low for kw in _LICENCE_OR_CERT_KEYWORDS):
        return True
    if _TRADE_CODE_PATTERN.search(name):
        return True
    return False


def _has_critical_credential_gap(missing_skill_names: list[str]) -> bool:
    """True if any missing skill looks like a licence / cert the user
    would need before they can do the job at all (not just to do it
    well). Used to cap the band at stretch for these cases."""
    return any(is_credential_skill_name(n) for n in missing_skill_names)


def _apply_hard_gates(
    band: str,
    score: float,
    missing_names: list[str],
    profile: dict,
    job: dict,
    score_explanation: dict,
) -> tuple[str, float]:
    """Apply Sprint 5 step 4 hard gates. Returns (band, score) post-demote.

    Mutates score_explanation in place to record which gates fired. Both
    scoring paths (main eligible + direct-title early-return) call this
    so the responder sees the same honesty floors regardless of how the
    posting was surfaced. A user typing an exact job title shouldn't
    bypass the same caps a normal-path match would hit.

    Demotes from 'strong'/'good' to 'stretch' with score floored at
    band_stretch + 0.01. An already-'stretch'/'low' band keeps its band
    but still records the flag, so the responder can narrate the reason.
    """
    stretch_floor = MATCH.band_stretch + 0.01
    # caps_applied is a flat ordered list of cap-flag names that fired.
    # Empty list when nothing fires. The responder uses this in Step 6
    # to pick which cap to narrate first without having to discover the
    # flag set by scanning every "band_capped_by_*" key.
    caps_applied = score_explanation.setdefault("caps_applied", [])

    def demote(flag: str, extra: dict | None = None) -> None:
        nonlocal band, score
        if band in ("strong", "good"):
            band = "stretch"
            score = stretch_floor
        score_explanation[flag] = True
        caps_applied.append(flag)
        if extra:
            score_explanation.update(extra)

    # 1. Credential cap: missing a licence/cert that the job lists as a
    #    required skill (e.g. Class G for a driving role).
    if _has_critical_credential_gap(missing_names):
        demote(
            "band_capped_by_credential",
            {
                "credential_gap_skills": [
                    n for n in missing_names
                    if any(kw in n.lower() for kw in _LICENCE_OR_CERT_KEYWORDS)
                ],
            },
        )

    # 2. No-experience floor: skills without experience_text don't prove
    #    fit. A blank-resume user with the right vocabulary should not
    #    outrank a candidate who has actually done the work.
    if not (profile.get("experience_text") or "").strip():
        demote("band_capped_by_no_experience")

    # 3. Work-type mismatch cap: user explicitly said full-time and the
    #    job's employment_type is part-time only (or vice versa). The
    #    soft -0.04 nudge from _work_type_fit isn't enough on its own.
    canon_pref = _normalize_work_type(profile.get("work_type_preference"))
    canon_job = _normalize_work_type(job.get("employment_type"))
    if (
        canon_pref and canon_job and canon_pref != "flexible"
        and {canon_pref, canon_job} == {"full-time", "part-time"}
    ):
        demote(
            "band_capped_by_work_type_mismatch",
            {
                "work_type_user": canon_pref,
                "work_type_job": canon_job,
            },
        )

    return band, score


def _location_boost(job_location: str | None, region_code: str | None) -> float:
    """Boost jobs whose location resolves to a city in LOCAL_CITIES.

    Post-PR-7A: every job is SSM-native by source approval, so this boost
    no longer discriminates "local vs national" (national isn't allowed).
    It still differentiates "Sault Ste. Marie proper" from nearby Algoma
    communities (Wawa, Blind River, etc.) when the user has signalled a
    preferred location.
    """
    loc = (job_location or "").lower()
    if any(city in loc for city in LOCAL_CITIES):
        return MATCH.location_boost_local_csd
    return 0.0


def _recency_boost(posted: date | None) -> float:
    if not posted:
        return 0.0
    age = (date.today() - posted).days
    if age < 0:
        return 0.0
    if age <= MATCH.recency_boost_days:
        # Linear: full boost on day 0 → 0 at recency_boost_days. Boost max 0.05.
        return 0.05 * (1 - age / max(1, MATCH.recency_boost_days))
    return 0.0


# =========================================================================
# Skill name overlap — fuzzy comparison between user and job skills
# =========================================================================
#
# Resumes and job postings rarely use the same exact phrasing for the same
# skill. Haiku might extract "welding" from a job and "welding &
# fabrication" from a resume — same skill, but the previous exact-string
# match would have missed it. This helper handles two real-world patterns:
#
#   1. SUBSTRING — one phrase contains the other ("welding" ⊂ "welding &
#      fabrication"). Word-level, after lowercase normalization.
#
#   2. TOKEN OVERLAP — the job's significant tokens are mostly present in
#      the user's phrasing ("class g license" vs "class g driver's
#      license"). Threshold: ≥60% of the job-side tokens (>= 2 chars,
#      non-stopword) appear in the user phrase.
#
# Both checks are conservative: short tokens and common words are filtered
# so we don't falsely match "a", "the", "and" between unrelated phrases.

_SKILL_STOPWORDS = frozenset({
    "and", "the", "of", "a", "an", "in", "for", "with", "to", "or",
    "&", "/", "-",
})
_SKILL_TOKEN_OVERLAP_THRESHOLD = 0.60


def _skill_tokens(phrase: str) -> set[str]:
    """Lowercased significant tokens from a skill phrase."""
    parts = re.split(r"[^a-z0-9]+", phrase.lower())
    return {t for t in parts if len(t) >= 2 and t not in _SKILL_STOPWORDS}


def _word_bounded_in(needle: str, haystack: str) -> bool:
    """True if `needle` appears in `haystack` at word boundaries.

    A plain `needle in haystack` check treats single-letter / short skill
    names as wildcards: "r" (the R programming language) matches "french
    language fluency" because the letter 'r' appears inside "french".
    This produces false-positive strong matches against unrelated jobs.

    Word-bounded containment requires the surrounding characters to be
    non-alphanumeric (or string edges), so "r" only matches a standalone
    "r" word, while "welding" still matches "welding & fabrication".
    """
    if not needle or not haystack:
        return False
    idx = haystack.find(needle)
    n = len(needle)
    while idx != -1:
        before_ok = idx == 0 or not haystack[idx - 1].isalnum()
        end = idx + n
        after_ok = end >= len(haystack) or not haystack[end].isalnum()
        if before_ok and after_ok:
            return True
        idx = haystack.find(needle, idx + 1)
    return False


def _skills_overlap(user_name: str, job_name: str) -> bool:
    """True if the user's skill phrase covers the job's skill phrase.

    Used to compare extracted resume skills (often longer phrases) against
    extracted job skills (often single words). Both are already lowercased
    when this is called from _score_one_job.
    """
    u = user_name.strip()
    j = job_name.strip()
    if not u or not j:
        return False
    # Word-bounded substring. Catches "welding" ⊂ "welding & fabrication".
    # The word-boundary requirement prevents single-letter skills like "r"
    # (R language) from matching every job phrase that contains the letter.
    if _word_bounded_in(j, u) or _word_bounded_in(u, j):
        return True
    # Token overlap. Catches "class g license" ≈ "class g driver's license".
    u_tokens = _skill_tokens(u)
    j_tokens = _skill_tokens(j)
    if not j_tokens:
        return False
    overlap = u_tokens & j_tokens
    return len(overlap) / len(j_tokens) >= _SKILL_TOKEN_OVERLAP_THRESHOLD


def _target_role_boost(title: str, target_role: str | None) -> float:
    if not target_role:
        return 0.0
    sim = _target_role_similarity(title, target_role)
    if sim >= 80:
        return 0.05
    if sim >= 60:
        return 0.02
    return 0.0


def _target_noc_boost(job_noc: str | None, user_noc: str | None) -> float:
    """+_TARGET_NOC_BOOST when both sides resolve to the same NOC code.

    NOC equality is the strongest occupation-similarity signal we have
    (it's not derived from title-token similarity; it's an authoritative
    taxonomy lookup). Boost is small enough that skills still dominate
    -- a strong NOC match with low skill overlap can still be just a
    stretch.

    Sibling NOCs (e.g., 21232 and 21233) would also reasonably deserve
    a smaller boost, but OaSIS doesn't ship explicit sibling edges and
    NOC-prefix proximity is too noisy without curation. Deferred.

    Returns 0.0 when either side is None (graceful when resolver missed
    or OaSIS CSVs aren't loaded yet).
    """
    if not job_noc or not user_noc:
        return 0.0
    return _TARGET_NOC_BOOST if job_noc == user_noc else 0.0


def _target_role_similarity(title: str | None, target_role: str | None) -> float:
    if not title or not target_role:
        return 0.0
    return float(fuzz.token_set_ratio(title.lower(), target_role.lower()))


def _direct_title_match_score(title: str | None, target_role: str | None) -> float | None:
    """Return a displayable score for a near-exact title request.

    Skill extraction is allowed to be incomplete. If a user types a specific
    SCCC title and that title exists in the current-job view, the system must
    show the posting instead of saying "no matches" merely because the job has
    fewer than the configured extracted skills.
    """
    sim = _target_role_similarity(title, target_role)
    if sim >= 92:
        return max(MATCH.band_good + 0.01, 0.61)
    if sim >= 82:
        return MATCH.band_stretch + 0.01
    return None


# --------------------------------- PR 10: work_type_fit + shift_fit ----
#
# These are SOFT signals: they don't filter jobs out (we don't have clean
# shift data per posting), they just nudge ranking. Both default to 0
# when the user hasn't told us their preference yet — match score is
# unchanged from PR 9A behaviour in that case.

# Map free-text work-type values to canonical token sets.
_WORK_TYPE_CANON: dict[str, frozenset[str]] = {
    "full-time": frozenset({"full-time", "full time", "fulltime", "ft"}),
    "part-time": frozenset({"part-time", "part time", "parttime", "pt"}),
    "flexible":  frozenset({"flexible", "hybrid"}),
    "remote":    frozenset({"remote", "work from home", "wfh"}),
    "casual":    frozenset({"casual", "on-call", "as-needed"}),
    "contract":  frozenset({"contract", "fixed-term", "term"}),
    "seasonal":  frozenset({"seasonal", "summer", "winter season"}),
}

# Cheap shift detector — looks for these phrases anywhere in title+description.
_SHIFT_HINTS: dict[str, tuple[str, ...]] = {
    "evenings": ("evening", "evenings", "evening shift", "pm shift"),
    "nights":   ("night", "nights", "overnight", "graveyard", "night shift"),
    "weekends": ("weekend", "weekends", "saturday", "sunday"),
    "days":     ("day shift", "daytime", "weekday", "mornings"),
}


def _normalize_work_type(value: str | None) -> str | None:
    if not value:
        return None
    v = value.strip().lower().replace(" ", "-")
    for canon, aliases in _WORK_TYPE_CANON.items():
        if v == canon or v in aliases:
            return canon
    return None


def _work_type_fit(job_employment_type: str | None,
                   pref: str | None) -> float:
    """+0.05 when the job's employment_type matches the user's preference,
    -0.04 when it directly contradicts (PT user vs FT-only job, etc.).
    0 when either side is unknown OR user said 'flexible'.
    """
    canon_pref = _normalize_work_type(pref)
    canon_job = _normalize_work_type(job_employment_type)
    if not canon_pref or canon_pref == "flexible":
        return 0.0
    if not canon_job:
        return 0.0
    if canon_pref == canon_job:
        return 0.05
    # PT user vs FT-only or vice versa = mild penalty.
    if {canon_pref, canon_job} == {"full-time", "part-time"}:
        return -0.04
    if canon_pref == "remote" and canon_job not in {"remote", "flexible"}:
        return -0.02
    return 0.0


def _shift_fit(job_title: str, job_description: str | None,
               pref: str | None) -> float:
    """+0.03 when the job mentions a shift the user prefers.
    No penalty if the job doesn't mention a shift at all (no signal).
    0 if user said 'any', 'flexible', 'rotating', or didn't say.
    """
    if not pref:
        return 0.0
    p = pref.strip().lower()
    if p in {"any", "flexible", "rotating", ""}:
        return 0.0
    # Map user phrasing to canonical shift bucket.
    canon: str | None = None
    if "evening" in p or "pm" in p:    canon = "evenings"
    elif "night" in p or "overnight" in p: canon = "nights"
    elif "weekend" in p:               canon = "weekends"
    elif "day" in p or "morning" in p: canon = "days"
    if canon is None:
        return 0.0
    haystack = (job_title or "").lower() + " " + (job_description or "").lower()
    hints = _SHIFT_HINTS.get(canon, ())
    if any(h in haystack for h in hints):
        return 0.03
    return 0.0


def _score_one_job(
    job: dict,
    job_skills: list[dict],
    user_skill_ids: set[str],
    user_skill_names: set[str],
    profile: dict,
    user_skill_names_canon: set[str] | None = None,
    *,
    user_rows: list[UserSkillRow] | None = None,
    user_embeddings_matrix=None,
    job_skill_embeddings: dict | None = None,
) -> MatchResult | None:
    # Canonical-form set lets alias variants ("Class G licence" vs
    # "class g license") match exactly without depending on the fuzzy
    # token-overlap fallback. Build a parallel set if the caller didn't
    # supply one -- keeps older callers working.
    if user_skill_names_canon is None:
        user_skill_names_canon = {canonicalize_skill(n) for n in user_skill_names}
    # AR-9.feat.coach-tiers CP1: per-requirement attribution. Empty when
    # the caller didn't supply user_rows (legacy / batch-scoring callers).
    skill_alignment_list: list[SkillAlignment] = []
    required = _filter_eligible_skills(job_skills)
    if len(required) < MATCH.min_required_skills_for_eligibility:
        direct_score = _direct_title_match_score(
            job.get("title"),
            profile.get("target_role_text"),
        )
        if direct_score is not None:
            cred = _regulated(job.get("noc_code"), profile.get("target_role_text"))
            credential_warning = None
            if cred:
                credential_warning = (
                    f"This occupation may require Canadian/Ontario licensing or certification "
                    f"({cred['regulator_name']}). {cred.get('licensing_note') or ''}"
                ).strip()

            # Compute matched / missing the same way the eligible-job
            # branch does. Even though the JOB has too few extracted
            # skills to be ranked by skills alone, the user may still
            # have some of those skills — saying "missing: [all of them]"
            # when the user actually has welding (or whatever) on their
            # resume is a misleading "because" clause for the responder.
            tm_matched: list[dict] = []
            tm_missing: list[dict] = []
            tm_matched_req: list[dict] = []
            tm_missing_req: list[dict] = []
            tm_matched_pref: list[dict] = []
            tm_missing_pref: list[dict] = []
            tm_match_strength: dict[str, float] = {}
            tm_match_stage: dict[str, str] = {}
            tm_alignments, tm_classifications = build_skill_alignment(
                required,
                user_rows,
                user_skill_ids,
                user_skill_names,
                user_skill_names_canon,
                user_embeddings_matrix=user_embeddings_matrix,
                job_skill_embeddings=job_skill_embeddings,
            )
            skill_alignment_list.extend(tm_alignments)
            for c in tm_classifications:
                s = c.skill
                key = (s.get("skill_name") or "").lower()
                bucket = _required_or_preferred(s)
                if c.strength > 0:
                    tm_matched.append(s)
                    # max-wins on lowercase collision (same rule as the
                    # main eligible path).
                    prev = tm_match_strength.get(key, 0.0)
                    if c.strength > prev:
                        tm_match_strength[key] = c.strength
                        tm_match_stage[key] = c.stage
                    (tm_matched_req if bucket == "required" else tm_matched_pref).append(s)
                else:
                    tm_missing.append(s)
                    (tm_missing_req if bucket == "required" else tm_missing_pref).append(s)
            tm_total_req = len(tm_matched_req) + len(tm_missing_req)
            tm_total_pref = len(tm_matched_pref) + len(tm_missing_pref)
            tm_matched_req_strength = sum(
                tm_match_strength.get((s.get("skill_name") or "").lower(), 0.0)
                for s in tm_matched_req
            )
            tm_matched_pref_strength = sum(
                tm_match_strength.get((s.get("skill_name") or "").lower(), 0.0)
                for s in tm_matched_pref
            )
            tm_base, tm_req_ratio, tm_pref_ratio = _weighted_skill_base(
                tm_matched_req_strength, tm_total_req,
                tm_matched_pref_strength, tm_total_pref,
            )

            # Build a score_explanation for the direct-title-match early
            # return, mirroring the eligible-job path's shape. The
            # responder needs the same "why" payload for these matches
            # (otherwise it can't ground a "because" clause for a job we
            # surfaced purely on title similarity).
            target_role = profile.get("target_role_text")
            title_sim_pct = (
                float(_target_role_similarity(job.get("title"), target_role))
                if target_role else 0.0
            )
            posted = job.get("posted_date")
            recency_days: int | None = (
                max(0, (date.today() - posted).days)
                if isinstance(posted, date) else None
            )

            score_explanation = {
                "matched_skills": [s["skill_name"] for s in tm_matched],
                "missing_skills": [s["skill_name"] for s in tm_missing],
                "skill_match_ratio": (
                    round(len(tm_matched) / len(required), 3) if required else 0.0
                ),
                # Required/preferred split (Sprint 5 step 3). Mirrors the
                # main eligible-path shape so the responder doesn't have
                # to handle two payload schemas.
                "required_matched": [s["skill_name"] for s in tm_matched_req],
                "required_missing": [s["skill_name"] for s in tm_missing_req],
                # Step 4a: per-skill strength + sum, same shape as main path
                "required_match_strengths": [
                    round(tm_match_strength.get((s.get("skill_name") or "").lower(), 0.0), 3)
                    for s in tm_matched_req
                ],
                # Step 5e: parallel stage labels (exact | fuzzy | semantic)
                "required_match_stages": [
                    tm_match_stage.get((s.get("skill_name") or "").lower(), "no_match")
                    for s in tm_matched_req
                ],
                "required_match_strength_sum": round(tm_matched_req_strength, 3),
                "required_match_ratio": round(tm_req_ratio, 3),
                "required_total": tm_total_req,
                "preferred_matched": [s["skill_name"] for s in tm_matched_pref],
                "preferred_missing": [s["skill_name"] for s in tm_missing_pref],
                "preferred_match_strengths": [
                    round(tm_match_strength.get((s.get("skill_name") or "").lower(), 0.0), 3)
                    for s in tm_matched_pref
                ],
                "preferred_match_stages": [
                    tm_match_stage.get((s.get("skill_name") or "").lower(), "no_match")
                    for s in tm_matched_pref
                ],
                "preferred_match_strength_sum": round(tm_matched_pref_strength, 3),
                "preferred_match_ratio": round(tm_pref_ratio, 3),
                "preferred_total": tm_total_pref,
                "title_match_similarity": (
                    round(title_sim_pct / 100.0, 3) if target_role else None
                ),
                "title_match_override": True,
                "recency_days": recency_days,
                "location_boosted": (
                    _location_boost(job.get("location"), job.get("region_code")) > 0
                ),
                "work_type_fit": "no_signal",
                "shift_fit": "no_signal",
                "credential_warning_present": bool(credential_warning),
                # Sprint 5 step 5: structured breakdown. Mode is 'direct_title'
                # because the score was driven by title similarity rather
                # than the skill-base formula (the JD had too few skills
                # for normal scoring). Boosts are all zero on this path --
                # the direct-title score is used as-is.
                "score_components": {
                    "skill_base": {
                        # Use the helper's computed base so the all-required
                        # and all-preferred edge cases collapse correctly
                        # (an all-required JD with full match -> 1.0, not
                        # the weighted blend 0.8 * 1.0 + 0.2 * 0.0).
                        "value": round(tm_base, 3),
                        "mode": "direct_title",
                        "required_match_ratio": round(tm_req_ratio, 3),
                        "required_weight": _REQ_WEIGHT,
                        "preferred_match_ratio": round(tm_pref_ratio, 3),
                        "preferred_weight": _PREF_WEIGHT,
                    },
                    "boosts": {
                        "location": 0.0,
                        "recency": 0.0,
                        "target_role": 0.0,
                        "target_noc_match": 0.0,
                        "work_type_fit": 0.0,
                        "shift_fit": 0.0,
                    },
                    "title_match": {
                        "applied": True,
                        "raw_similarity": (
                            round(title_sim_pct / 100.0, 3)
                            if target_role else None
                        ),
                    },
                    "score_pre_caps": round(min(1.0, direct_score), 3),
                    "score_post_caps": round(min(1.0, direct_score), 3),
                },
            }

            # Hard gates also apply on the direct-title surface. A user
            # who types an exact job title should not bypass the credential
            # cap, no-experience floor, or work-type mismatch cap just
            # because the JD had too few extracted skills for normal
            # scoring.
            tm_band = _band(direct_score)
            tm_score = min(1.0, direct_score)
            tm_missing_names = [s["skill_name"] for s in tm_missing]
            tm_band, tm_score = _apply_hard_gates(
                tm_band, tm_score, tm_missing_names,
                profile, job, score_explanation,
            )
            score_explanation["score_components"]["score_post_caps"] = round(tm_score, 3)

            return MatchResult(
                job_id=str(job["job_id"]),
                profile_id=str(profile["profile_id"]),
                title=job["title"],
                employer=job.get("employer"),
                url=job.get("url"),
                location=job.get("location"),
                match_score=round(tm_score, 3),
                match_band=tm_band,
                match_eligible=True,
                ineligibility_reason=None,
                matched_skills=[s["skill_name"] for s in tm_matched],
                missing_skills=[s["skill_name"] for s in tm_missing],
                matched_skill_ids=[s.get("skill_id") for s in tm_matched],
                missing_skill_ids=[s.get("skill_id") for s in tm_missing],
                required_skills_count=len(required),
                credential_warning=credential_warning,
                posted_date=job.get("posted_date"),
                noc_code=job.get("noc_code"),
                score_explanation=score_explanation,
                skill_alignment=tuple(skill_alignment_list),
                employment_type=job.get("employment_type"),
                salary_text=job.get("salary_text"),
            )

        return MatchResult(
            job_id=str(job["job_id"]),
            profile_id=str(profile["profile_id"]),
            title=job["title"],
            employer=job.get("employer"),
            url=job.get("url"),
            location=job.get("location"),
            match_score=0.0,
            match_band="low",
            match_eligible=False,
            ineligibility_reason=f"too_few_required_skills (have {len(required)})",
            matched_skills=[],
            missing_skills=[s["skill_name"] for s in required],
            matched_skill_ids=[],
            missing_skill_ids=[s.get("skill_id") for s in required],
            required_skills_count=len(required),
            credential_warning=None,
            posted_date=job.get("posted_date"),
            noc_code=job.get("noc_code"),
            employment_type=job.get("employment_type"),
            salary_text=job.get("salary_text"),
        )

    matched: list[dict] = []
    missing: list[dict] = []
    # Track required vs preferred separately so the weighted base score
    # can give required skills the larger pull. The flat `matched` and
    # `missing` lists are still maintained for UI compatibility (existing
    # callers expect the aggregated lists).
    matched_req: list[dict] = []
    missing_req: list[dict] = []
    matched_pref: list[dict] = []
    missing_pref: list[dict] = []
    # Step 4a: per-skill continuous match strength. Keyed by skill_name
    # (lowercased) so the score_explanation can surface strength alongside
    # the existing flat name list without changing its shape.
    #
    # Dedup safety: extracted.job_skill PRIMARY KEY is (job_id, skill_name),
    # so two rows in `required` can't share a skill_name within one job.
    # Lowercasing here is the only collision risk -- e.g. if the extractor
    # ever produced both "Python" and "python" for the same job. Today
    # that doesn't happen (the LLM-side extractor and the schema's PK
    # both effectively enforce case-stable names per job), so dict
    # collisions are not a concern. If they ever become one, the dedup
    # rule would be "max strength wins" -- enforced literally by the
    # max(...) assignment below.
    match_strength: dict[str, float] = {}
    # Step 5e: parallel dict tracking which STAGE produced the match.
    # Same key as match_strength. Values: "exact" | "fuzzy" | "semantic".
    # The responder uses this to narrate "you have X" vs "your X is
    # related to Y" without inferring stage from magic-number strength.
    match_stage: dict[str, str] = {}
    alignments, classifications = build_skill_alignment(
        required,
        user_rows,
        user_skill_ids,
        user_skill_names,
        user_skill_names_canon,
        user_embeddings_matrix=user_embeddings_matrix,
        job_skill_embeddings=job_skill_embeddings,
    )
    skill_alignment_list.extend(alignments)
    for c in classifications:
        s = c.skill
        key = (s.get("skill_name") or "").lower()
        bucket = _required_or_preferred(s)
        if c.strength > 0:
            matched.append(s)
            # max-wins on the (unlikely) case where lowercasing causes a
            # collision. If the colliding rows had different stages, the
            # winning stage is the one whose strength wins.
            prev_strength = match_strength.get(key, 0.0)
            if c.strength > prev_strength:
                match_strength[key] = c.strength
                match_stage[key] = c.stage
            (matched_req if bucket == "required" else matched_pref).append(s)
        else:
            missing.append(s)
            (missing_req if bucket == "required" else missing_pref).append(s)

    total_req = len(matched_req) + len(missing_req)
    total_pref = len(matched_pref) + len(missing_pref)
    # Strength sums replace counts in _weighted_skill_base. When all
    # matches are stage-1 (strength=1.0), the sum equals the count and
    # the resulting ratio is identical to the pre-refactor count-based
    # ratio -- backwards compatible.
    matched_req_strength = sum(
        match_strength.get((s.get("skill_name") or "").lower(), 0.0)
        for s in matched_req
    )
    matched_pref_strength = sum(
        match_strength.get((s.get("skill_name") or "").lower(), 0.0)
        for s in matched_pref
    )
    base, req_ratio, pref_ratio = _weighted_skill_base(
        matched_req_strength, total_req,
        matched_pref_strength, total_pref,
    )

    # Compute boost components individually so we can surface each one
    # in score_explanation. This is what lets the responder say
    # "your forklift and inventory line up with the top requirements"
    # (quoting matched_skills) instead of inventing "because Sault employers
    # value warehouse experience" (no grounded source).
    loc_boost = _location_boost(job.get("location"), job.get("region_code"))
    rec_boost = _recency_boost(job.get("posted_date"))
    role_boost = _target_role_boost(job["title"], profile.get("target_role_text"))
    # Step 2 occupation-match boost. Independent of role_boost (which is
    # title-token similarity); when both fire, they stack. NOC match is
    # the authoritative occupation signal; title-token similarity stays
    # as the graceful fallback when the resolver has no OaSIS data.
    noc_boost = _target_noc_boost(job.get("noc_code"), profile.get("target_noc"))
    wt_fit = _work_type_fit(job.get("employment_type"),
                            profile.get("work_type_preference"))
    sh_fit = _shift_fit(job["title"], job.get("description"),
                        profile.get("shift_preference"))
    boost = loc_boost + rec_boost + role_boost + noc_boost + wt_fit + sh_fit

    # work_type_fit can be negative; clamp the floor to 0 so negative boosts
    # don't drop us below the skill-match base.
    score = max(0.0, min(1.0, base + boost))

    # Title-match fast path. When the user named a specific role and a job's
    # title strongly matches it, surface the posting at least at the
    # 'stretch' band — even if we have no skill overlap yet. Otherwise a
    # zero-skill user gets "no matches" for a job they literally typed by
    # title, which is the worst possible answer the chat can give.
    #
    # The threshold (>=80 token_set_ratio) is tight enough that generic
    # role words like "manager" alone don't trip it; the user has to
    # match most of the title's tokens.
    target_role = profile.get("target_role_text")
    title_sim = 0.0
    title_match_override = False
    if target_role:
        title_sim = float(fuzz.token_set_ratio(
            (job.get("title") or "").lower(),
            target_role.lower(),
        ))
        if title_sim >= 80 and score < MATCH.band_stretch:
            # Nudge into stretch band so it shows. Use a small offset above
            # band_stretch so the responder labels it correctly.
            score = MATCH.band_stretch + 0.01
            title_match_override = True

    cred = _regulated(job.get("noc_code"), profile.get("target_role_text"))
    credential_warning = None
    if cred:
        credential_warning = (
            f"This occupation may require Canadian/Ontario licensing or certification "
            f"({cred['regulator_name']}). {cred.get('licensing_note') or ''}"
        ).strip()

    # Build the explanation payload that the responder will quote in
    # "because" clauses. Each entry is something the LLM is allowed to
    # cite verbatim — no invented signals.
    posted = job.get("posted_date")
    recency_days: int | None
    if isinstance(posted, date):
        recency_days = max(0, (date.today() - posted).days)
    else:
        recency_days = None

    score_explanation = {
        "matched_skills": [s["skill_name"] for s in matched],
        "missing_skills": [s["skill_name"] for s in missing],
        # Step 4a refactor: this is now a STRENGTH-WEIGHTED ratio, not a
        # pure matched-count ratio. When all matches are stage-1
        # (strength=1.0) the value equals the old count-based ratio
        # (backwards-compat). When token-overlap fuzzy matches (0.85)
        # or future semantic matches contribute, the ratio is lower
        # because the strength sum is < the match count. Treat as
        # "skill-base score", not "X of Y skills matched". For explicit
        # count-style narration ("you matched 4 of 5 required skills"),
        # use len(required_matched) / required_total instead.
        "skill_match_ratio": round(base, 3),
        # Sprint 5 step 3: required/preferred split. Required skills are
        # the dominant signal (80% weight); preferred is a nudge (20%).
        # Legacy rows with skill_type=NULL bucket into 'required'.
        "required_matched": [s["skill_name"] for s in matched_req],
        "required_missing": [s["skill_name"] for s in missing_req],
        # Matching v2 step 4a: per-skill strength alongside the flat
        # name list. Same indexing as required_matched. Strength=1.0
        # means stage-1 (exact / canonical / word-bounded substring);
        # 0.85 means token-overlap fuzzy; 0.75 means semantic (Step 5).
        "required_match_strengths": [
            round(match_strength.get((s.get("skill_name") or "").lower(), 0.0), 3)
            for s in matched_req
        ],
        # Matching v2 step 5e: parallel stage labels so the responder
        # can narrate "you have X" for exact vs "your X is related to
        # the Y requirement" for semantic. Values: "exact" | "fuzzy"
        # | "semantic". Same indexing as required_matched / strengths.
        "required_match_stages": [
            match_stage.get((s.get("skill_name") or "").lower(), "no_match")
            for s in matched_req
        ],
        "required_match_strength_sum": round(matched_req_strength, 3),
        "required_match_ratio": round(req_ratio, 3),
        "required_total": total_req,
        "preferred_matched": [s["skill_name"] for s in matched_pref],
        "preferred_missing": [s["skill_name"] for s in missing_pref],
        "preferred_match_strengths": [
            round(match_strength.get((s.get("skill_name") or "").lower(), 0.0), 3)
            for s in matched_pref
        ],
        "preferred_match_stages": [
            match_stage.get((s.get("skill_name") or "").lower(), "no_match")
            for s in matched_pref
        ],
        "preferred_match_strength_sum": round(matched_pref_strength, 3),
        "preferred_match_ratio": round(pref_ratio, 3),
        "preferred_total": total_pref,
        "title_match_similarity": round(title_sim / 100.0, 3) if target_role else None,
        "title_match_override": title_match_override,
        "recency_days": recency_days,
        "location_boosted": loc_boost > 0,
        "work_type_fit": (
            "matched" if wt_fit > 0
            else ("mismatched" if wt_fit < 0 else "no_signal")
        ),
        "shift_fit": "matched" if sh_fit > 0 else "no_signal",
        "credential_warning_present": bool(credential_warning),
        # Sprint 5 step 5: structured breakdown so the responder narrates
        # from numbers, not invented "because" clauses. The nested dict
        # is additive -- flat fields above stay for backwards compat.
        "score_components": {
            "skill_base": {
                "value": round(base, 3),
                "mode": _skill_base_mode(total_req, total_pref),
                "required_match_ratio": round(req_ratio, 3),
                "required_weight": _REQ_WEIGHT,
                "preferred_match_ratio": round(pref_ratio, 3),
                "preferred_weight": _PREF_WEIGHT,
            },
            "boosts": {
                "location": round(loc_boost, 3),
                "recency": round(rec_boost, 3),
                "target_role": round(role_boost, 3),
                "target_noc_match": round(noc_boost, 3),
                "work_type_fit": round(wt_fit, 3),
                "shift_fit": round(sh_fit, 3),
            },
            "title_match": {
                "applied": title_match_override,
                "raw_similarity": (
                    round(title_sim / 100.0, 3) if target_role else None
                ),
            },
            "score_pre_caps": round(score, 3),
            # score_post_caps is filled in after _apply_hard_gates runs.
            "score_post_caps": round(score, 3),
        },
    }

    # ----- Hard gates (Sprint 5 step 4) ---------------------------------
    # The shared helper applies all three caps and mutates
    # score_explanation in place. Both the main eligible path and the
    # direct-title early-return path call it -- the responder must see
    # the same honesty floors regardless of which scoring path produced
    # the result.
    missing_names = [s["skill_name"] for s in missing]
    band = _band(score)
    band, score = _apply_hard_gates(
        band, score, missing_names, profile, job, score_explanation,
    )
    score_explanation["score_components"]["score_post_caps"] = round(score, 3)

    return MatchResult(
        job_id=str(job["job_id"]),
        profile_id=str(profile["profile_id"]),
        title=job["title"],
        employer=job.get("employer"),
        url=job.get("url"),
        location=job.get("location"),
        match_score=round(score, 3),
        match_band=band,
        match_eligible=True,
        ineligibility_reason=None,
        matched_skills=[s["skill_name"] for s in matched],
        missing_skills=[s["skill_name"] for s in missing],
        matched_skill_ids=[s.get("skill_id") for s in matched],
        missing_skill_ids=[s.get("skill_id") for s in missing],
        required_skills_count=len(required),
        credential_warning=credential_warning,
        posted_date=job.get("posted_date"),
        noc_code=job.get("noc_code"),
        score_explanation=score_explanation,
        skill_alignment=tuple(skill_alignment_list),
        employment_type=job.get("employment_type"),
        salary_text=job.get("salary_text"),
    )


# --------------------------------------------------------- Public entrypoint
def compute_matches(profile_id: str, dataset_version: str | None = None) -> list[MatchResult]:
    """Score every current eligible job against this profile.

    Persists results to analytics.job_match (+ job_match_skill). Returns the
    list ordered by score descending.
    """
    profile = _fetch_profile(profile_id)
    if profile is None:
        return []
    user_skills = _fetch_user_skills(profile_id)
    # AR-9.feat.coach-tiers CP1: build UserSkillRow list from persisted
    # rows so the matcher can attribute matches. profile.user_skill has
    # already cleared the consent boundary; no evidence gate needed.
    user_rows: list[UserSkillRow] = []
    for s in user_skills:
        raw = (s.get("skill_name") or "").strip()
        if not raw:
            continue
        user_rows.append(UserSkillRow(
            skill_id=str(s["skill_id"]) if s.get("skill_id") else None,
            text=raw,
            name=raw.lower(),
            canon=canonicalize_skill(raw) or "",
        ))
    user_skill_ids, user_skill_names, user_skill_names_canon = (
        derive_user_skill_sets(user_rows)
    )

    # Step 2: resolve target_role_text -> NOC for the persistent profile.
    # _fetch_profile doesn't currently SELECT target_noc (it isn't a
    # persisted column on profile.user_profile); resolve at match time
    # and inject. If the persisted profile gains a target_noc column
    # later, this becomes a noop.
    profile["target_noc"] = resolve_title_to_noc(profile.get("target_role_text"))

    # Step 5d: encode user skills ONCE per batch (same pattern as the
    # in-memory variant). None when sentence-transformers absent.
    user_embeddings_matrix = _maybe_embed_user_skill_rows(user_rows)

    results: list[MatchResult] = []
    for job in _fetch_eligible_jobs():
        skills = _fetch_job_skills(str(job["job_id"]))
        job_skill_embeddings = (
            _fetch_job_skill_embeddings(str(job["job_id"]))
            if user_embeddings_matrix is not None else None
        )
        m = _score_one_job(
            job, skills, user_skill_ids, user_skill_names, profile,
            user_skill_names_canon=user_skill_names_canon,
            user_rows=user_rows,
            user_embeddings_matrix=user_embeddings_matrix,
            job_skill_embeddings=job_skill_embeddings,
        )
        if m is not None:
            results.append(m)

    _persist_matches(profile_id, results, dataset_version)
    results.sort(key=lambda r: (r.match_eligible, r.match_score), reverse=True)
    return results


def top_matches(profile_id: str, top: int = 10, only_eligible: bool = True) -> list[MatchResult]:
    """Return the top-N matches for a profile, recomputing if needed."""
    all_matches = compute_matches(profile_id)
    filtered = [m for m in all_matches if (m.match_eligible if only_eligible else True)]
    return filtered[:top]


# ---------------------------------------------------------- Persistence
def _persist_matches(profile_id: str, results: Iterable[MatchResult],
                     dataset_version: str | None) -> None:
    with sync_cursor() as cur:
        # Clear previous matches for this profile so the table reflects the
        # latest dataset version cleanly. Cheap at MVP scale.
        cur.execute("DELETE FROM analytics.job_match WHERE profile_id = %s", (profile_id,))
        for r in results:
            cur.execute(
                """
                INSERT INTO analytics.job_match
                    (profile_id, job_id, match_score, match_band, match_eligible,
                     ineligibility_reason, matched_skills_count, required_skills_count,
                     missing_skills_count, credential_warning, dataset_version,
                     engine_version)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                RETURNING match_id
                """,
                (
                    profile_id, r.job_id, r.match_score, r.match_band, r.match_eligible,
                    r.ineligibility_reason, len(r.matched_skills), r.required_skills_count,
                    len(r.missing_skills), r.credential_warning, dataset_version,
                    ENGINE_VERSION_JOB_MATCH,
                ),
            )
            match_id = cur.fetchone()["match_id"]
            for name, sid in zip(r.matched_skills, r.matched_skill_ids):
                cur.execute(
                    "INSERT INTO analytics.job_match_skill (match_id, skill_id, skill_name, status) "
                    "VALUES (%s, %s, %s, 'matched') ON CONFLICT DO NOTHING",
                    (match_id, sid, name),
                )
            for name, sid in zip(r.missing_skills, r.missing_skill_ids):
                cur.execute(
                    "INSERT INTO analytics.job_match_skill (match_id, skill_id, skill_name, status) "
                    "VALUES (%s, %s, %s, 'missing') ON CONFLICT DO NOTHING",
                    (match_id, sid, name),
                )


# --------------------------------------- In-memory variant (pre-consent)
def compute_matches_in_memory(staged: StagedProfile, top: int = 5) -> list[MatchResult]:
    """Score current jobs against a staged (un-persisted) profile.

    Reads jobs/skills/regulated_occupations from Postgres but writes nothing
    user-specific. Used by the chat handler on the pre-consent path.
    """
    # AR-9.feat.coach-tiers CP1: UserSkillRow is the authority. Sets
    # the matcher consumes are derived from rows so scoring and
    # attribution cannot disagree about which user skills were
    # evidence-eligible this turn.
    user_rows = build_user_skill_rows(staged.skills)
    user_skill_ids, user_skill_names, user_skill_names_canon = (
        derive_user_skill_sets(user_rows)
    )

    # Step 2: resolve target_role_text -> NOC once per match-batch and
    # cache on the staged profile. Subsequent turns with the same role
    # skip the DB round-trip. Returns None when the resolver has nothing
    # (empty occupation_title_synonym table or unresolvable title) --
    # _target_noc_boost handles that gracefully (boost = 0).
    if staged.target_role_text and not staged.target_noc:
        staged.target_noc = resolve_title_to_noc(staged.target_role_text)

    # Step 5d: encode user skills ONCE per match-batch (~150ms for ~30
    # skills) so per-job semantic comparisons reduce to a single matrix
    # multiply. The matrix is built from `user_rows` IN ROW ORDER so
    # the semantic-argmax index maps directly back to user_rows[i].text
    # for attribution. None when sentence-transformers is missing OR
    # the user has no skills.
    user_embeddings_matrix = _maybe_embed_user_skill_rows(user_rows)

    profile_dict = {
        "profile_id": staged.session_id,  # placeholder; nothing is persisted
        "preferred_location": staged.preferred_location,
        "target_role_text": staged.target_role_text,
        "target_noc": staged.target_noc,
        "work_type_preference": staged.work_type_preference,
        "shift_preference": staged.shift_preference,
        "experience_text": staged.experience_text,
    }
    results: list[MatchResult] = []
    for job in _fetch_eligible_jobs():
        skills = _fetch_job_skills(str(job["job_id"]))
        job_skill_embeddings = (
            _fetch_job_skill_embeddings(str(job["job_id"]))
            if user_embeddings_matrix is not None else None
        )
        m = _score_one_job(
            job, skills, user_skill_ids, user_skill_names, profile_dict,
            user_skill_names_canon=user_skill_names_canon,
            user_rows=user_rows,
            user_embeddings_matrix=user_embeddings_matrix,
            job_skill_embeddings=job_skill_embeddings,
        )
        if m is not None:
            results.append(m)
    results.sort(key=lambda r: (r.match_eligible, r.match_score), reverse=True)
    return results[:top]


def next_skill_to_unlock_in_memory(staged: StagedProfile) -> tuple[str | None, int]:
    """In-memory variant of next_skill_to_unlock. Same algorithm; no DB writes."""
    matches = compute_matches_in_memory(staged, top=100)
    if not matches:
        return None, 0
    candidates: dict[str, int] = {}
    for m in matches:
        if not m.match_eligible or m.match_band in {"good", "strong"}:
            continue
        denom = m.required_skills_count or 1
        gain_per = 1 / denom
        if m.match_score + gain_per < MATCH.band_good:
            continue
        for s in m.missing_skills:
            candidates[s] = candidates.get(s, 0) + 1
    if not candidates:
        return None, 0
    best = max(candidates.items(), key=lambda kv: kv[1])
    return best[0], best[1]


# --------------------------------------------- Next-skill-to-unlock helper
def next_skill_to_unlock(profile_id: str, candidate_pool_size: int = 50) -> tuple[str | None, int]:
    """Given current profile skills, find the single missing skill that, if
    acquired, would push the most additional current jobs into 'good' or
    'strong' band. Returns (skill_name, jobs_unlocked).

    Used by the empty-recommendation case ("If you add Excel, 12 more jobs
    become reachable").
    """
    matches = compute_matches(profile_id)
    if not matches:
        return None, 0
    # Count, per missing skill, how many currently-non-good matches would
    # cross 0.60 if that single skill became matched.
    candidates: dict[str, int] = {}
    for m in matches:
        if not m.match_eligible or m.match_band in {"good", "strong"}:
            continue
        denom = m.required_skills_count or 1
        gain_per = 1 / denom
        new_score_if_added = m.match_score + gain_per
        if new_score_if_added < MATCH.band_good:
            continue
        for s in m.missing_skills:
            candidates[s] = candidates.get(s, 0) + 1
    if not candidates:
        return None, 0
    best = max(candidates.items(), key=lambda kv: kv[1])
    return best[0], best[1]
