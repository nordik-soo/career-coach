"""Occupation normalization -- title -> NOC 2021 code resolver.

Matching v2 step 2 (see docs/matching-v2-design.md §3). Bridges
"Software Developer" / "Application Programmer" / "Coding Specialist"
to a single NOC code so jobs posted under different titles cluster
together in the matcher.

Lookup strategy (in order, first hit wins):
  1. Exact case-insensitive match against `reference.occupation_title_synonym.title`
  2. Trigram similarity (postgres pg_trgm via the GIN index we built in
     step 1), threshold 0.6 -- catches small spelling drift like
     "front desk agent" vs "front desk agents".

When nothing reaches the threshold, returns None. Callers must treat
None as "no occupation signal". (Step 2 cutover 2026-07-16:
`_target_role_boost` was retired — the title-token similarity boost
that this resolver's None branch used to fall back to no longer
exists. NOC-match remains the sole occupation-relevance signal
contributing to V1 fit.)

There is no LLM fallback. We defer that until real chat exposes a
title we can't resolve with the OaSIS lexicon -- premature LLM calls
on every chat turn would burn tokens for marginal recall.

Round-34 / round-35 / round-36 NOC observability:
  - Four-state distinction (empty_table | exact | fuzzy | unresolved)
    is exposed via `resolve_title_to_noc_with_state`. Successful
    resolutions DON'T run the presence check -- the empty-table
    discriminator only fires when both passes miss, keeping the hot
    path at one query for exact and two for fuzzy.
  - `candidate_similarity` is the highest similarity returned by
    pg_trgm's `%` operator at the session threshold (typically 0.3).
    Values below that threshold are filtered server-side and are
    NOT visible to this function; the field name is "candidate"
    rather than "best" to be honest about that limit.
  - Miss telemetry is split by call source:
      job_backfill: titles come from public partner-feed job postings
                    we already store unredacted. Logs the normalized
                    title (length-capped) so the vocabulary backlog
                    can be built from real data.
      user_target:  the input is user-typed and can be ANYTHING --
                    name, email, free-text rant. SHA-256 over a
                    constrained input space (English job titles +
                    common email domains) is dictionary-recoverable
                    and offers no real protection (round-36 review).
                    We log ONLY length + token count + score; no
                    derived fingerprint, hashed or otherwise.
                    A future iteration can add keyed HMAC with a
                    rotatable telemetry secret if the operator
                    explicitly wants the fingerprint signal back.
  - Audit callers pass `emit_telemetry=False` so the report itself
    is the canonical record and the miss log isn't double-flooded.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from skillbridge.db import sync_cursor

log = logging.getLogger(__name__)
# Dedicated child logger for resolver misses so production can route
# them separately (e.g. structured-log sink that builds the vocabulary
# backlog). On the job_backfill side it logs the normalized title
# (public partner-feed data). On the user_target side it logs ONLY
# length / token count / score -- no derived fingerprint, because
# SHA-256 over a constrained input space (job titles, email
# patterns) is dictionary-recoverable and offers no real protection.
_miss_log = logging.getLogger("skillbridge.match.occupation.miss")


# Trigram similarity threshold. The pg_trgm operator `%` returns rows
# above the session-configured threshold (default 0.3 in pg_trgm). We
# pass our own threshold via similarity() comparison so the lookup is
# explicit and tunable without altering the session setting.
_TRIGRAM_THRESHOLD = 0.6


# Length cap on titles included in miss telemetry (job_backfill side
# only). The full title may legitimately be 100+ chars; capping at 80
# keeps log lines bounded without obscuring the vocabulary signal.
_MISS_TITLE_CAP = 80


CallSource = Literal["user_target", "job_backfill"]


class NocResolutionState(str, Enum):
    """Four-state resolver outcome. Distinguishes deployment-prereq
    failures (empty_table) from real vocabulary misses (unresolved)
    so the inspection command can report them separately."""
    EMPTY_TABLE = "empty_table"      # reference.occupation_title_synonym is empty
    EXACT = "exact"                  # case-insensitive exact match (sim = 1.0)
    FUZZY = "fuzzy"                  # trigram >= threshold (sim recorded)
    UNRESOLVED = "unresolved"        # no row reached threshold


@dataclass(frozen=True)
class NocResolution:
    """Structured resolver result."""
    state: NocResolutionState
    noc_code: str | None             # populated on EXACT / FUZZY, None otherwise
    similarity: float                # 1.0 exact, [threshold, 1.0] fuzzy, 0.0 otherwise
    # Best similarity returned by pg_trgm's `%` operator. The operator
    # applies its session-configured threshold (default 0.3) at the
    # SQL layer; rows below that threshold are filtered out before
    # they reach us. So this is "the best candidate the SQL layer
    # SURFACED," NOT a guarantee of the actual highest similarity
    # across the corpus. Round-35: renamed from "best_fuzzy_similarity"
    # to be honest about that limit. For an exhaustive best-score
    # audit, run a separate KNN query that bypasses the operator
    # filter.
    candidate_similarity: float = 0.0


def _log_miss(needle: str, call_source: CallSource, candidate_sim: float) -> None:
    """Emit one PII-conscious miss log line.

    user_target side (round-36): log ONLY length + token count + score.
        target_role_text can legitimately contain a name, email, or
        any free-text the user typed. A raw SHA-256 fingerprint over
        such a constrained input space (English job titles + common
        email domains) is dictionary-recoverable and offers no real
        protection -- so we log no fingerprint at all. The
        length+token signals are enough to distinguish "one-word
        title" from "long sentence" patterns when triaging the
        backlog; the score tells us how close pg_trgm got.

    job_backfill side: log the normalized title (length-capped).
        Job-posting titles are public partner-feed data we already
        store unredacted. Capturing them here lets us build the
        vocabulary backlog directly.
    """
    if call_source == "user_target":
        token_count = len(needle.split())
        _miss_log.info(
            "noc_resolver_miss length=%d token_count=%d "
            "call_source=user_target candidate_score=%.3f",
            len(needle), token_count, candidate_sim,
        )
    else:
        _miss_log.info(
            "noc_resolver_miss title=%r call_source=%s candidate_score=%.3f",
            needle[:_MISS_TITLE_CAP], call_source, candidate_sim,
        )


def resolve_title_to_noc_with_state(
    title: str | None,
    *,
    call_source: CallSource = "user_target",
    emit_telemetry: bool = True,
) -> NocResolution:
    """Full four-state resolution + miss telemetry.

    Returns a `NocResolution` with state + noc_code + similarity +
    candidate_similarity. On UNRESOLVED with a non-empty synonym
    table, emits a single PII-conscious log line (see `_log_miss`)
    unless the caller passes `emit_telemetry=False` -- the audit
    suppresses it because the report itself is the canonical record.

    Query ordering (round-35): exact -> fuzzy -> presence check. The
    presence check only runs when BOTH passes return nothing, so the
    hot path stays at one query for exact resolution and two for
    fuzzy. The empty-table discriminator is still preserved -- it
    just fires lazily.

    `call_source` distinguishes user-target misses (where the input
    can be PII) from job-backfill misses (public job-posting data).
    """
    if not title:
        return NocResolution(
            state=NocResolutionState.UNRESOLVED,
            noc_code=None, similarity=0.0, candidate_similarity=0.0,
        )
    needle = title.strip().lower()
    if not needle:
        return NocResolution(
            state=NocResolutionState.UNRESOLVED,
            noc_code=None, similarity=0.0, candidate_similarity=0.0,
        )

    with sync_cursor() as cur:
        # ---- Pass 1: exact case-insensitive match ----
        # Single query for the happy path. Skips the presence check;
        # finding a row means the table is non-empty by construction.
        cur.execute(
            """
            SELECT noc_code
              FROM reference.occupation_title_synonym
             WHERE LOWER(title) = %s
             LIMIT 1
            """,
            (needle,),
        )
        row = cur.fetchone()
        if row:
            return NocResolution(
                state=NocResolutionState.EXACT,
                noc_code=row["noc_code"],
                similarity=1.0,
                candidate_similarity=1.0,
            )

        # ---- Pass 2: trigram fuzzy match ----
        # The `%` operator applies pg_trgm's session threshold
        # (default 0.3). We compute similarity() in SELECT so we can
        # apply our own _TRIGRAM_THRESHOLD (0.6) for matching, while
        # still surfacing the best candidate the SQL layer returned
        # for telemetry / audit. Note: `candidate_similarity` is NOT
        # the corpus-wide best; values below 0.3 never reach us.
        cur.execute(
            """
            SELECT noc_code, similarity(title, %s) AS sim
              FROM reference.occupation_title_synonym
             WHERE title %% %s
             ORDER BY sim DESC
             LIMIT 1
            """,
            (title, title),
        )
        row = cur.fetchone()
        best = float(row["sim"]) if row else 0.0
        if row and best >= _TRIGRAM_THRESHOLD:
            return NocResolution(
                state=NocResolutionState.FUZZY,
                noc_code=row["noc_code"],
                similarity=best,
                candidate_similarity=best,
            )

        # ---- Pass 3: empty-table discriminator (only on full miss) ----
        # Both passes missed. Distinguish "the table is empty -- nobody
        # could have matched" from "table is populated, this title is
        # a real vocabulary miss." Running this LAST means successful
        # exact/fuzzy paths pay zero queries for the discriminator.
        cur.execute("SELECT 1 FROM reference.occupation_title_synonym LIMIT 1")
        if cur.fetchone() is None:
            return NocResolution(
                state=NocResolutionState.EMPTY_TABLE,
                noc_code=None, similarity=0.0, candidate_similarity=0.0,
            )

    # Real vocabulary miss. Emit PII-conscious telemetry unless the
    # caller (e.g. audit sweep) explicitly suppressed it.
    if emit_telemetry:
        _log_miss(needle, call_source, best)
    return NocResolution(
        state=NocResolutionState.UNRESOLVED,
        noc_code=None, similarity=0.0, candidate_similarity=best,
    )


def resolve_title_to_noc(
    title: str | None,
    *,
    call_source: CallSource = "user_target",
) -> str | None:
    """Return the NOC 2021 code for a free-text job title, or None.

    Thin wrapper around `resolve_title_to_noc_with_state` preserved
    for callers that only need the code. Telemetry on misses still
    fires (the wrapper just discards state + similarity).

    Empty / whitespace input returns None without hitting the DB.
    DB-side, the lookup is one query for exact, two for fuzzy, and
    three for a structural-failure check. If
    `reference.occupation_title_synonym` is empty -- the user hasn't
    loaded the OaSIS CSVs yet, see docs/oasis-download.md -- returns
    None and the matcher falls through to today's behavior.
    """
    return resolve_title_to_noc_with_state(
        title, call_source=call_source,
    ).noc_code


def resolve_title_to_noc_with_score(
    title: str | None,
    *,
    call_source: CallSource = "user_target",
) -> tuple[str | None, float]:
    """Same as resolve_title_to_noc, but also returns the match score.

    For debug/inspection paths (e.g. logging in the SCCC backfill).
    The chat hot path uses the simpler resolver. Returns (None, 0.0)
    on miss.
    """
    res = resolve_title_to_noc_with_state(title, call_source=call_source)
    return res.noc_code, res.similarity
