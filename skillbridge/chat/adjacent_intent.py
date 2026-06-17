"""Adjacent-recommendations intent detector (AR-2).

Pure function. Classifies a user message into one of three terminal
states for the adjacency dispatch:

  - `AdjacentIntent`: surface adjacency engine; the message is a
    different-role discovery ask AND the user has usable evidence.
  - `NeedsEvidenceIntent`: the message looks like an adjacency ask
    but the user doesn't have enough evidence to anchor a
    recommendation; AR-6 routes to a clarification.
  - `None`: falls through to the standard planner / arbiter chain.

This module is DEAD CODE until AR-6 wires the call site into
`_try_v2_path`. The activation-safety audit in
tests/test_ar1c_parity_and_activation.py confirms no production
caller exists yet. The Redis-mode activation gate
(`_adjacency_enabled`) short-circuits ALL of this in cookie-mode
sessions.

Trigger precedence locked at v11 design (with corrections through
v12 amendment):

  - "what else FOR THIS JOB?"        -> remaining-gaps (R-3 owns
                                       it; runs BEFORE the adjacency
                                       hook in `_try_v2_path`).
  - "what other jobs / roles?"      -> adjacency (this module).
  - bare "what else?"                -> None (planner falls through
                                       to a focused clarification).
  - soft-offer accepted (the prior
    responder turn appended the
    soft-offer line; the handler
    threads `pending_offer=True`)   -> adjacency on the locked
                                       affirmative set; anything
                                       else -> None.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from skillbridge.session.staging import StagedProfile


@dataclass(frozen=True)
class AdjacentIntent:
    """Terminal state: dispatch the adjacency engine."""
    trigger: str   # one of "user_explicit" | "soft_offer_accepted"


@dataclass(frozen=True)
class NeedsEvidenceIntent:
    """Terminal state: handler emits a clarification asking for the
    user's experience or skills. We can't recommend roles when we have
    nothing to anchor against."""
    trigger: str


# Locked affirmative set for the soft-offer-accepted path (v11 QF
# lock). Anything outside this set -- "I guess?", "maybe", "ok I guess
# why not" -- falls through to None so the planner emits a
# clarification rather than blowing into adjacency. Normalized form:
# lowercased, surrounding punctuation stripped, internal whitespace
# collapsed.
_AFFIRMATIVE_REPLIES: frozenset[str] = frozenset({
    "yes",
    "yes please",
    "sure",
    "ok",
    "okay",
    "show me",
    "go ahead",
    "please do",
})


# Explicit different-role phrasings. Compiled with word boundaries so
# substring leakage (e.g. "what other things should I bring?") doesn't
# trip a false positive. Patterns are deliberately conservative -- the
# detector returns None on ambiguous input and the planner handles it.
_ADJACENCY_EXPLICIT_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p) for p in [
        r"\bwhat other (jobs?|roles?|positions?|work|kinds? of work)\b",
        r"\bany other (jobs?|roles?|positions?|openings?)\b",
        r"\b(different|other) (jobs?|roles?|positions?)\b",
        r"\bother (postings?|openings?)\b",
        r"\broles? like (this|that)( one)?\b",
        r"\bshow me other (jobs?|roles?|positions?)\b",
        r"\bwhat am i close to (in other|with other|on other)\b",
        r"\banything else i could do\b",
        r"\banything else i can do\b",
        # AR-8b: jobs-related-to-my-skills, "related" connector.
        # Live-observed misses (2026-06-10):
        #   "find any job related my skill"     -- typo, no "to"
        #   "find jobs related to my skills"    -- canonical
        # Verb-led (find/find me/show/look for/get me) OR question-led
        # (what/any/which) + job-noun + " related (to)? " + my/the skills.
        # Anchors: leading verb/question word + trailing "my/the
        # skill(s) (set)?" so "skills related to this job" (subject
        # reversed) and "find any job" alone (no skills connector)
        # both correctly drop to None.
        # AR-8b round-2: optional adverb slot (closely/directly/most)
        # before "related" so "find jobs closely related to my skills"
        # also matches.
        r"\b(?:find(?:\s+me)?|show(?:\s+me)?|look\s+for|get\s+me|what|any|which)\s+"
        r"(?:any\s+|some\s+|the\s+|other\s+|more\s+|additional\s+)?"
        r"(?:jobs?|roles?|work|postings?|positions?|openings?)\b"
        r"(?:\s+(?:are|that\s+are))?"
        r"\s+(?:closely\s+|directly\s+|most\s+)?related(?:\s+to)?\s+"
        r"(?:my|the)\s+skills?(?:\s+set)?\b",
        # AR-8b: jobs-related-to-my-skills, "match/fit/use/suit"
        # connectors. Same shape as above but with the explicit
        # transfer verbs (matching/that match, fitting/that fit,
        # using/that use, suiting/that suit). Inflections cover
        # 3rd-person and gerund forms.
        # AR-8b round-2: optional modal/adverb slot (would/could/
        # should/might/best/really) between optional "that" and the
        # connector verb so "what roles would match my skills" and
        # "which jobs best match my skills" also match.
        r"\b(?:find(?:\s+me)?|show(?:\s+me)?|look\s+for|get\s+me|what|any|which)\s+"
        r"(?:any\s+|some\s+|the\s+|other\s+|more\s+|additional\s+)?"
        r"(?:jobs?|roles?|work|postings?|positions?|openings?)\b"
        r"(?:\s+that)?"
        r"\s+(?:would\s+|could\s+|should\s+|might\s+|best\s+|really\s+|closely\s+)?"
        # "use" needs special handling: "using" = "us"+"ing" (silent
        # 'e' drops), so `use(?:s|ing)?` would NOT match "using".
        # `(?:uses?|using)` covers "use" / "uses" / "using" correctly.
        r"(?:match(?:es|ing)?|fit(?:s|ting)?|(?:uses?|using)|suit(?:s|ing)?)\s+"
        r"(?:my|the)\s+skills?(?:\s+set)?\b",
    ]
)


# AR-8b round-3: clause-scoped decline guard. The two AR-8b explicit
# patterns above match the *shape* of a skill-related job request
# (verb + job-noun + connector + my/the skills). When the user puts
# negation in front of the verb the shape still matches, so the
# existing `_OTHER_ROLE_DECLINE_PATTERNS` (which require "other")
# don't cover it.
#
# Round-3 fixes two opposite false-flags from round-2:
#   (a) INDIRECT negation in the same clause must coerce decline.
#       Round-2 only caught direct negation (`don't show...`); these
#       slipped through:
#         - "I don't want you to show me jobs related to my skills"
#         - "I do not want you to find jobs related to my skills"
#         - "I would rather you not show me jobs related to my skills"
#         - "I don't want jobs related to my skills"  (want-object)
#   (b) Message-wide suppression broke compound pivots. Round-2 ran
#       the guard over the whole message; a decline in clause 1
#       poisoned a clean request in clause 2:
#         - "Don't show me this role; find jobs related to my skills"
#         - "Never show me this posting; instead find jobs related to my skills"
#         - "Don't show me that one. Show me roles that use my skills"
#
# Fix: split the normalized message into clauses on `;`, `.`, `!`,
# `?` or `,` + transition conjunction; run BOTH the explicit-pattern
# check and the decline guard per-clause; explicit_match is True iff
# at least one clause matches an explicit pattern without a same-
# clause decline.
#
# Smart apostrophes (U+2019 etc.) fold to ASCII `'` in `_normalize`
# before these patterns run.

_SEARCH_VERB_DECLINE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p) for p in [
        # (1) Direct negation of the search verb itself.
        #   "don't show me ..." / "do not find ..." / "never show me"
        #   "please don't look for" / "won't search for"
        r"\b(?:don'?t|do not|never|won'?t|will not|"
        r"please\s+(?:don'?t|do not))\s+"
        r"(?:find|show(?:\s+me)?|look\s+for|look\s+up|get\s+me|"
        r"search(?:\s+for)?)\b",
        # (2) Indirect negation with a complement-clause verb:
        #   "<don't|...> <want|wish|...> (you|me|...)? to <search verb>"
        # The optional pronoun slot covers "you/me/us/them/him/her"
        # but is NOT required, so "I don't want to find" also matches.
        r"\b(?:don'?t|do not|never|won'?t|will not)\s+"
        r"(?:want|wish|care|like|need|ask|expect|prefer|intend)\s+"
        r"(?:(?:you|me|us|them|him|her|anyone)\s+)?"
        r"to\s+"
        r"(?:find|show(?:\s+me)?|look\s+for|look\s+up|get\s+me|"
        r"search(?:\s+for)?)\b",
        # (3) "would rather (you|...)? not <search verb>"
        # AR-8b round-4: contraction `'d rather` ("I'd / we'd / they'd
        # / you'd / he'd / she'd / it'd rather you not show me ...").
        # `'d` always stands for "would" in this construction. Smart
        # apostrophes (U+2019 etc.) fold to ASCII `'` in `_normalize`
        # before this pattern runs.
        r"\b(?:would|(?:i|we|they|you|he|she|it)'?d)\s+rather\s+"
        r"(?:(?:you|i|we|they|he|she|anyone)\s+)?"
        r"not\s+"
        r"(?:find|show(?:\s+me)?|look\s+for|look\s+up|get\s+me|"
        r"search(?:\s+for)?)\b",
        # (4) "prefer (you|...)? not <search verb>" (without "would")
        r"\bprefer\s+"
        r"(?:(?:you|me|us|them|anyone)\s+)?"
        r"not\s+"
        r"(?:to\s+)?"
        r"(?:find|show(?:\s+me)?|look\s+for|look\s+up|get\s+me|"
        r"search(?:\s+for)?)\b",
        # (5) Want-object form: "<don't|...> <want|wish|...> <job-noun>
        #     ... related/matching my/the skills". The user negates
        #     wanting THE jobs themselves (not "you to show" them).
        #     Mirrors `_OTHER_ROLE_DECLINE`'s want-object branch but
        #     keyed on the skill-connector shape instead of "other".
        r"\b(?:don'?t|do not|never|won'?t|will not)\s+"
        r"(?:want|wish|care\s+for|like|need)\s+"
        r"(?:any\s+|some\s+|the\s+|those\s+|these\s+|that\s+|this\s+)?"
        r"(?:jobs?|roles?|work|postings?|positions?|openings?)\b"
        r"(?:\s+(?:are|that\s+are))?"
        r"\s+(?:closely\s+|directly\s+|most\s+)?"
        r"(?:related(?:\s+to)?|"
        r"(?:would\s+|could\s+|should\s+|might\s+|best\s+|really\s+|closely\s+)?"
        r"(?:match(?:es|ing)?|fit(?:s|ting)?|(?:uses?|using)|suit(?:s|ing)?))\s+"
        r"(?:my|the)\s+skills?(?:\s+set)?\b",
    ]
)


# AR-8b round-6: clause-level retraction phrases. Standalone refusals
# at the END of a compound message (after an earlier positive clause)
# must flip the polarity to decline. The message-level
# `_PURE_DECLINE_PATTERNS` only fires on whole-message or
# leading-position "no" / "not now" / "maybe later", so a trailing
# "never mind" / "cancel that" / "forget it" / "I changed my mind"
# slipped through round-5.
#
# Each pattern is anchored to clause-end (`\s*[.!?]?\s*$`) so a
# substring usage doesn't false-positive ("I never mind helping" /
# "I want to cancel that one" / "I changed my mind about this").
_CLAUSE_LEVEL_RETRACTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.I) for p in [
        # "never mind" / "nevermind" at clause end.
        r"\bnever\s*mind\s*[.!?]?\s*$",
        # Round-7: compound "no thanks" / "no thank you" / "no i'm
        # good" at clause end. The compound disambiguates "no" so
        # this matches even after explicit content in the same
        # clause ("find jobs ... no thanks").
        r"\bno\s+(?:thanks?|thank\s+you|i'?m\s+good)\s*[.!?]?\s*$",
        # Round-7: softener-compound "actually no" / "sorry no" at
        # clause end. The softener disambiguates "no" without needing
        # a preceding punctuation boundary, so "find jobs ... sorry
        # no" matches even without a comma. Optional comma allowed
        # between softener and "no" ("actually, no" / "sorry, no").
        r"\b(?:actually|sorry)[,!.]?\s+no\s*[.!?]?\s*$",
        # Round-8: bare "no" at clause end. Relaxed from round-7's
        # `(?:^|[,.!?])\s*no` to `\bno` so that "Show me jobs related
        # to my skills no" (no comma) is treated the same as the
        # comma form. The substring guard "I have no idea" still
        # holds because the pattern requires `\s*[.!?]?\s*$` AFTER
        # "no" -- mid-clause "no" followed by content fails the
        # end-anchor check.
        r"\bno\s*[.!?]?\s*$",
        # "cancel that" / "cancel it" at clause end.
        r"\bcancel\s+(?:that|it)\s*[.!?]?\s*$",
        # "forget it" / "forget that" / "forget about it/that" at
        # clause end.
        r"\bforget\s+(?:it|that|about\s+(?:it|that))\s*[.!?]?\s*$",
        # "(i)? (have|just)? changed my mind" at clause end. The "i"
        # is optional because clause splitting may have stripped a
        # leading subject-pronoun-only fragment.
        r"\b(?:i\s+)?(?:have\s+|just\s+)?changed\s+my\s+mind\s*[.!?]?\s*$",
    ]
)


# AR-8b round-9: PER-MATCH retraction suppressors.
#
# Round-8 set has_retraction=False whenever ANY suppressor matched
# the clause. That over-broadened in two ways:
#   - "Show me jobs related to my skills, don't forget it, actually
#     no" -- the "don't forget" suppressor incorrectly cancelled the
#     trailing "actually no" which is a genuine retraction.
#   - "Show me jobs related to my skills. I said never mind" -- the
#     "I + <up to 2 words> + never mind" suppressor matched "I said
#     never mind", but "said" is a verb of reporting, so "never mind"
#     IS the retraction the user said.
#
# Round-9 fix:
#   1. Each retraction match is checked independently against the
#      suppressors. A suppressor that targets one phrase doesn't
#      affect another phrase elsewhere in the clause.
#   2. Suppressor (b)'s `.{0,40}?` window becomes `[^,]{0,40}?`: a
#      comma between conjunction and retraction means they're in
#      separate clauses semantically; suppression doesn't carry across.
#   3. Suppressor (c)'s word-window becomes a strict modal-only
#      window (would/will/do/does/did/might/could/should). Verbs of
#      saying/thinking ("said", "mean", "guess", "think") no longer
#      consume the pronoun slot, so genuine retractions surface.
_NEGATION_BEFORE_FORGET_CANCEL = re.compile(
    r"\b(?:don'?t|do not|never|won'?t|will not|"
    r"please\s+(?:don'?t|do not))\s+$",
    re.I,
)
_SUBORDINATING_CONJUNCTION_NO_COMMA = re.compile(
    r"\b(?:because|while|since|if|when|although|though|after|"
    r"before|as|until|unless)\b[^,]{0,40}?$",
    re.I,
)
# Round-10: prefix-only form, anchored to end of preceding text via
# `$`. Round-9 ran the suppressor against the whole clause, so an
# earlier "i would never mind waiting" (positive sentiment phrase)
# cancelled a later, separate "never mind" retraction. Round-10
# checks `preceding` (text up to the current match start) and
# requires it to END with the pronoun-modal-adverb prefix so the
# suppressor only fires when this specific "never mind" match is
# preceded by the right structure.
#
# Round-11: adverb allowance. Round-10's prefix slot only permitted
# modals (would/will/do/does/did/might/could/should), so positive
# "never mind" constructions with adverbs ("I really never mind",
# "I honestly never mind", "We generally never mind") fell through
# and incorrectly produced retraction. Round-11 adds a controlled
# adverb allowlist; up to 2 prefix words total (any combination of
# modal and adverb) covers "I would really never mind", "They might
# honestly never mind", etc. Reporting verbs (said/mean/guess/think)
# are deliberately NOT in the allowlist so genuine "I said never
# mind" retractions still surface.
_PRONOUN_MODAL_PREFIX_BEFORE_NEVER_MIND = re.compile(
    r"\b(?:i|you|we|they|he|she|it)(?:'?(?:m|s|d|ll|ve|re))?\b"
    r"\s+"
    r"(?:"
    r"(?:would|will|do|does|did|might|could|should"
    r"|really|honestly|truly|absolutely|definitely|certainly"
    r"|generally|usually|normally|typically|frankly|genuinely"
    r"|seriously|sincerely|particularly)\s+"
    r"){0,2}"
    r"$",
    re.I,
)


def _retraction_match_suppressed(clause: str, match: "re.Match[str]") -> bool:
    """Return True iff the given retraction `match` in `clause` is
    suppressed by the surrounding syntax. Per-match scope: a
    suppressor that targets a different retraction phrase in the
    same clause does NOT suppress this one.
    """
    matched = match.group(0).lower()
    preceding = clause[: match.start()]

    # (a) Negation directly before forget/cancel. Only fires when
    # THIS match itself is forget/cancel-shaped.
    if "forget" in matched or "cancel" in matched:
        if _NEGATION_BEFORE_FORGET_CANCEL.search(preceding):
            return True

    # (b) Subordinating conjunction within ~40 non-comma chars
    # before this retraction. A comma between would put the
    # conjunction and the retraction in separate clauses.
    if _SUBORDINATING_CONJUNCTION_NO_COMMA.search(preceding):
        return True

    # (c) Pronoun + optional modal directly before THIS "never
    # mind" match. Round-10: checked against `preceding` (not the
    # whole clause) so an earlier positive "I would never mind X"
    # doesn't cancel a later separate retraction. Modal-only window
    # excludes "said"/"mean"/"guess"/"think".
    if "never" in matched and "mind" in matched:
        if _PRONOUN_MODAL_PREFIX_BEFORE_NEVER_MIND.search(preceding):
            return True

    return False


def _clause_has_active_retraction(clause: str) -> bool:
    """Return True iff the clause has at least one retraction
    phrase whose meaning is NOT suppressed by surrounding syntax."""
    for pat in _CLAUSE_LEVEL_RETRACTION_PATTERNS:
        m = pat.search(clause)
        if m is None:
            continue
        if _retraction_match_suppressed(clause, m):
            continue
        return True
    return False


# Clause boundaries used by `_has_explicit_adjacency_clause`.
# Terminal punctuation (`;`, `.`, `!`, `?`) ends a clause WITH OR
# WITHOUT trailing whitespace (round-4 fix: previously required `\s+`
# after punctuation, so "role;find" / "role.Show" passed through as a
# single clause). Bare comma is NOT a boundary (too noisy: lists,
# parentheticals, "no, do X" patterns) -- only `, but|instead|however|
# rather|though` introduces a transition that clearly opens a new
# clause. Bare " but " (without preceding comma) IS a boundary --
# common in informal writing ("don't show me this role but find ...").
_CLAUSE_BOUNDARY = re.compile(
    r"(?:[;.!?]+\s*"
    r"|\s*,\s+(?:but|instead|however|rather|though)\s+"
    r"|\s+but\s+)",
    re.I,
)


def _has_explicit_adjacency_clause(norm: str) -> bool:
    """Clause-scoped explicit-pattern check with same-clause decline
    guard AND last-clause-wins ordering.

    Iterates ALL clauses (not just up to the first positive). Each
    intent-bearing clause updates a running polarity flag:

      - Explicit pattern matches AND no same-clause search-verb
        decline -> positive.
      - Explicit pattern matches AND same-clause search-verb decline
        -> negative.
      - No explicit pattern AND a clause-level retraction phrase
        ("never mind", "no thanks", "cancel that", "forget it",
        "i changed my mind") -> negative. Round-6 add: trailing
        standalone refusals must flip an earlier positive even
        though they don't carry the explicit shape.
      - Else: clause isn't intent-bearing; flag unchanged.

    Returns True iff the LAST intent-bearing clause was positive.

    Review history:
      - Round 3 (clause splitting): a decline in clause 1 must not
        poison a clean request in clause 2.
      - Round 3 (same-clause indirect): an indirect decline in the
        SAME clause must still suppress the request.
      - Round 5 (last-clause-wins): a search-verb-shaped retraction
        at the END of a compound message must override an earlier
        request.
      - Round 6 (trailing pure-decline): a standalone retraction
        clause ("never mind") must also participate in last-clause
        ordering, even though it has no adjacency shape.

    Pure-decline whole-message inputs ("no thanks" alone) are still
    caught by the message-level `_PURE_DECLINE_PATTERNS` check in
    `detect_adjacent_intent`; this function's retraction handling
    is specifically for trailing clauses after a positive clause.
    """
    last_positive: bool | None = None
    for raw in _CLAUSE_BOUNDARY.split(norm):
        clause = raw.strip()
        if not clause:
            continue
        has_explicit = any(
            pat.search(clause) for pat in _ADJACENCY_EXPLICIT_PATTERNS
        )
        has_search_decline = any(
            pat.search(clause) for pat in _SEARCH_VERB_DECLINE_PATTERNS
        )
        # Round-9: per-match retraction check. Each retraction
        # phrase is checked individually against the suppressors so a
        # suppressor that targets one phrase doesn't cancel another
        # phrase elsewhere in the same clause. See
        # `_clause_has_active_retraction` and
        # `_retraction_match_suppressed`.
        has_retraction = _clause_has_active_retraction(clause)
        if has_explicit:
            if has_search_decline or has_retraction:
                last_positive = False
            else:
                last_positive = True
        elif has_retraction:
            last_positive = False
    return last_positive is True


# Phrases that look adjacency-ish but are R-3's territory (same-role
# gap discovery). The detector returns None on these so R-3 can fire
# at the dispatch layer. R-3 ALSO runs first in `_try_v2_path` per the
# locked design, so these phrases shouldn't reach the adjacency
# detector at all -- this is belt-and-braces.
_SAME_ROLE_GAP_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p) for p in [
        r"\bwhat else (do i need|for this (job|role)|am i missing)\b",
        r"\bwhat (am i|do i) (still missing|need to add|need to do)\b",
    ]
)


# Same-role anchor. "anything else I could do for this job?" hits the
# explicit-pattern set BUT the "for this job" anchor means it's R-3
# territory. Mirrors `_SAME_ROLE_ANCHOR` in remaining_gaps.py.
_SAME_ROLE_ANCHOR = re.compile(
    r"\b(?:for|with|about|in|on|toward(?:s)?)\s+"
    r"(?:this|that|the\s+current)\s+"
    r"(?:job|role|position|one|opening|posting)\b"
)


# Object-scoped decline: the user explicitly rejects OTHER roles (the
# object of the negation). These return None even when paired with an
# adjacency-shaped clause. Examples that match:
#   - "no other jobs please"
#   - "I don't want other roles"
#   - "I don't think other jobs are right for me"
#   - "I'm not looking for other roles"
#   - "don't show me other postings"
# Examples that do NOT match (handled by the next group below):
#   - "I don't want this job, show me other roles"  (object is THIS role)
_OTHER_ROLE_DECLINE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p) for p in [
        # "no other (jobs|roles|...)"
        r"\bno\s+other\s+(?:jobs?|roles?|positions?|openings?|postings?)\b",
        # "i (don't|do not) (want|need|care for|like|see myself in) other ..."
        r"\bi\s+(?:don'?t|do not)\s+"
        r"(?:want|need|care\s+for|like|see\s+myself\s+in)\s+"
        r"other\s+(?:jobs?|roles?|positions?|openings?|postings?)\b",
        # "i (don't|do not) think other (jobs|roles) are right/good/for me/..."
        r"\bi\s+(?:don'?t|do not)\s+think\s+"
        r"other\s+(?:jobs?|roles?|positions?)\b",
        # "don't show|recommend|suggest me other (jobs|roles)"
        r"\b(?:don'?t|do not)\s+"
        r"(?:show|recommend|suggest|give)\s+(?:me\s+)?"
        r"(?:any\s+)?other\s+(?:jobs?|roles?|positions?|openings?|postings?)\b",
        # "i'm not looking|interested for|in other (jobs|roles|postings|...)"
        r"\bi(?:'?m|\s+am)\s+not\s+"
        r"(?:looking|interested)\s+(?:in|for)\s+"
        r"(?:any\s+)?other\s+(?:jobs?|roles?|positions?|openings?|postings?)\b",
        # "i'm not looking|interested for|in different (jobs|roles)"
        r"\bi(?:'?m|\s+am)\s+not\s+"
        r"(?:looking\s+for|interested\s+in)\s+"
        r"different\s+(?:jobs?|roles?|positions?)\b",
    ]
)


# Pure decline phrases. Treated as adjacency declines regardless of
# anything else in the message because they're standalone refusals:
#   - "no" alone
#   - "no thanks", "no thank you"
#   - "no, ..." or "no. ..." or "no! ..." (the comma/period seals the
#     "no" as a refusal; whatever follows is a separate statement)
#   - "not now" / "maybe later"
_PURE_DECLINE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p) for p in [
        r"^\s*no(?:\s+(?:thanks|thank\s+you|i'?m\s+good|i\s+am\s+good))?\s*[.!?]?\s*$",
        r"^\s*no\s*[,.;!]\s+",   # "no, ..." / "no. ..." / "no! ..."
        r"\b(?:not\s+now|maybe\s+later)\b",
    ]
)


# Current-role decline + adjacency pivot. The user explicitly rejects
# THIS / THE CURRENT role, NOT other roles. When the message ALSO
# contains an adjacency request, the explicit pivot wins.
#
# Covers two syntactic shapes:
#   (a) Object-style: "I (don't|do not|'m not|am not) <verb> this/that/
#       the current role". Verb set includes both want-style (want,
#       like, need, care for) AND interest-style (interested in, see
#       myself in).
#   (b) Subject-style: "this/that/the current role (isn't|is not|ain't)
#       for me / a (good) fit / right for me / what I want". The role
#       is the SUBJECT of the negation, not the object.
#
# The shared design constraint: the negation must clearly attach to
# THIS / THE CURRENT role, never to "other" roles -- those object-
# scoped declines are caught by `_OTHER_ROLE_DECLINE_PATTERNS` and
# block adjacency entirely.
_CURRENT_ROLE_DECLINE_PATTERN = re.compile(
    # (a) Object-style.
    r"\bi(?:\s+(?:don'?t|do not)|\s+am not|'?m not)\s+"
    r"(?:want|like|need|care\s+for|interested\s+in|see\s+myself\s+in)\s+"
    r"(?:any\s+part\s+of\s+)?(?:this|that|the\s+current)\s+"
    r"(?:job|role|position|one|opening|posting)\b"
    r"|"
    # (b) Person-subject form: the user negates THEIR fit for the
    # role ("I'm not a fit for this role", "I am not a good fit for
    # the current position"). The role is the object of "for".
    r"\bi(?:\s+am|'?m)\s+not\s+a\s+(?:good\s+)?fit\s+for\s+"
    r"(?:this|that|the\s+current)\s+"
    r"(?:job|role|position|one|opening|posting)\b"
    r"|"
    # (c) Role-subject form.
    r"\b(?:this|that|the\s+current)\s+"
    r"(?:job|role|position|one|opening|posting)\s+"
    r"(?:is\s+not|isn'?t|ain'?t)\s+"
    r"(?:for\s+me|a\s+(?:good\s+)?fit|right(?:\s+for\s+me)?|what\s+i\s+want)\b"
)


# Smart-apostrophe normalization table. Mobile autocorrect and many
# editors emit U+2019 (right single quotation mark) where users type
# `'`. The detector's patterns use straight apostrophes, so all four
# smart-apostrophe code points collapse to `'` before pattern matching.
_APOSTROPHE_NORMALIZER = str.maketrans({
    "‘": "'",   # left single quotation mark
    "’": "'",   # right single quotation mark
    "‚": "'",   # single low-9 quotation mark
    "‛": "'",   # single high-reversed-9 quotation mark
})


def _normalize(message: str | None) -> str:
    """Lowercase, smart-apostrophe-fold, strip surrounding punctuation,
    collapse whitespace."""
    if not message:
        return ""
    # Smart-apostrophe normalization BEFORE lowering so the table
    # works on the raw glyph (translate() doesn't depend on case but
    # ordering matters for downstream re-use).
    s = message.translate(_APOSTROPHE_NORMALIZER).lower().strip()
    # Strip surrounding punctuation (./,!?;:) so "yes," normalizes to
    # "yes". Internal punctuation is left alone -- the affirmative
    # check uses an exact-string equality after this normalization.
    s = re.sub(r"^[\s.,!?;:'\"]+", "", s)
    s = re.sub(r"[\s.,!?;:'\"]+$", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def detect_adjacent_intent(
    message: str | None,
    staged: StagedProfile,
    user_has_evidence: bool,
    pending_offer: bool,
) -> AdjacentIntent | NeedsEvidenceIntent | None:
    """Classify the user's message into the adjacency intent space.

    Args:
        message: the raw user message.
        staged: the live StagedProfile (UNUSED in the v1 detector --
            kept in the signature for forward compatibility and to
            make the audit easier to reason about. Reserved for
            future heuristics that need profile context).
        user_has_evidence: the result of
            `match.adjacent.has_usable_skill_evidence(staged)`,
            threaded in by the caller so the detector stays pure (no
            DB access, no engine import).
        pending_offer: True iff the prior responder turn appended the
            soft-offer line. Threaded in from the handler's
            save-and-clear hook in `handle_anonymous` (AR-1a).

    Returns:
        AdjacentIntent on a clean different-role discovery match;
        NeedsEvidenceIntent when the ask matched but evidence is
        missing; None on ambiguous / same-role / unrelated input.

    PURE -- does not mutate `staged`. Idempotent given the same
    inputs.
    """
    _ = staged   # currently unused; preserved for forward-compat.

    norm = _normalize(message)
    if not norm:
        return None

    # R-3 precedence: same-role gap phrasings yield None so R-3 can
    # handle them at the dispatch layer. (In `_try_v2_path` R-3 also
    # runs first; this guard is a defense against accidental ordering
    # drift.)
    for pat in _SAME_ROLE_GAP_PATTERNS:
        if pat.search(norm):
            return None

    # Same-role anchor: "for this job", "with this role", etc. Even
    # if the explicit-pattern set would match, the anchor means it's
    # R-3 territory. Mirrors the counter-exclusion in remaining_gaps.py.
    # EXCEPTION: the compound "I don't want this job, show me other
    # roles" carries the anchor inside a current-role decline; that's
    # still a valid pivot (handled below), so we only fall through
    # when the anchor stands alone.
    has_current_role_anchor = bool(_SAME_ROLE_ANCHOR.search(norm))
    has_current_role_decline = bool(_CURRENT_ROLE_DECLINE_PATTERN.search(norm))

    if has_current_role_anchor and not has_current_role_decline:
        return None

    # Object-scoped negation: the user explicitly rejects OTHER
    # roles. Cannot be coerced into adjacency by an adjacency-shaped
    # token elsewhere in the message.
    #   - "I don't want other jobs"
    #   - "no other roles please"
    #   - "I don't think other jobs are right for me"
    for pat in _OTHER_ROLE_DECLINE_PATTERNS:
        if pat.search(norm):
            return None

    # Pure decline phrases: "no", "no thanks", "no, ...", "not now".
    # Override any adjacency-shaped follow-up clause -- the leading
    # refusal seals the intent ("no, show me other jobs" → decline).
    # EXCEPTION: when the compound carries an explicit current-role
    # decline ("I don't want this job, show me other roles"), the
    # pivot survives. Pure decline doesn't apply because the message
    # isn't a refusal -- it's a redirect.
    if not has_current_role_decline:
        for pat in _PURE_DECLINE_PATTERNS:
            if pat.search(norm):
                return None

    # AR-8b round-3: clause-scoped explicit-pattern check. A decline
    # in one clause must not poison a clean request in another clause
    # ("Don't show me this role; find jobs related to my skills" ->
    # AdjacentIntent on the second clause). An indirect decline in
    # the SAME clause must still coerce decline ("I don't want you to
    # show me jobs related to my skills" -> None). See
    # `_has_explicit_adjacency_clause` for the contract.
    explicit_match = _has_explicit_adjacency_clause(norm)

    affirmative_match = pending_offer and norm in _AFFIRMATIVE_REPLIES

    if not (explicit_match or affirmative_match):
        return None

    if not user_has_evidence:
        trigger = "user_explicit" if explicit_match else "soft_offer_accepted"
        return NeedsEvidenceIntent(trigger=trigger)

    trigger = "user_explicit" if explicit_match else "soft_offer_accepted"
    return AdjacentIntent(trigger=trigger)
