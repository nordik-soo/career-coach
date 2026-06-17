"""AR-8b tests: intent patterns for "jobs related to my skills" and
the match/fit/use/suit family.

Live observation (2026-06-10):
  Bot soft-offer: "...just say *what other roles?*"
  User: "other role"                        → AdjacentIntent  (matched)
  User: "find any job related my skill"     → None            (MISS)
  User: "find jobs related to my skills"    → None            (MISS)

The miss caused the user to bounce off the soft-offer path back into
the standard match → no-match → re-offer loop. Detector was treating
the phrasing as ambiguous because `_ADJACENCY_EXPLICIT_PATTERNS` had
no shape that paired a job-noun with a my-skills connector.

AR-8b adds two patterns:
  - `<find|show|look for|get me|what|any|which> <jobs|roles|work|...>
     related (to)? my/the skills`
  - `<find|show|look for|get me|what|any|which> <jobs|roles|work|...>
     (that)? <match|matching|fit|fitting|use|using|suit|suiting>
     my/the skills`

Tests cover:
  - positive: the two live-observed misses + natural variants;
  - negative: "find any job" alone, "skills related to this job"
    (subject reversed), "what skills do I need for this job",
    "find a developer" (no job-noun in the noun set);
  - regression: every prior AR-2 positive still matches.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.nodb


from skillbridge.chat.adjacent_intent import (
    AdjacentIntent,
    NeedsEvidenceIntent,
    detect_adjacent_intent,
)
from skillbridge.session.staging import StagedProfile


def _staged() -> StagedProfile:
    return StagedProfile.new("sess-ar8b")


# =========================================================================
# Positive: live-observed misses + natural variants
# =========================================================================
@pytest.mark.parametrize("phrase", [
    # The two live-observed misses.
    "find any job related my skill",
    "find jobs related to my skills",
    # "related" connector variants.
    "show me jobs related to my skills",
    "show jobs related to my skills",
    "look for roles related to my skills",
    "get me work related to my skills",
    "what jobs are related to my skills",
    "what jobs related to my skills",
    "any roles related to my skills",
    "which positions are related to the skills",
    "find postings related to my skill set",
    "show me openings related to my skill set",
    # match/fit/use/suit connectors, with and without "that".
    "show me jobs matching my skills",
    "find jobs that match my skills",
    "find work that matches my skills",
    "look for roles that fit my skills",
    "show me roles fitting my skills",
    "any jobs that use my skills",
    "what jobs use my skills",
    "show me jobs using my skills",
    "find positions that suit my skills",
    "any roles suiting my skills",
    "find jobs that fit my skill set",
])
def test_skills_phrasing_with_evidence_returns_adjacent_intent(phrase) -> None:
    """Verb/question-led job-to-skills request must produce
    AdjacentIntent when the user has usable evidence."""
    result = detect_adjacent_intent(
        phrase, _staged(),
        user_has_evidence=True,
        pending_offer=False,
    )
    assert isinstance(result, AdjacentIntent), (
        f"phrase={phrase!r} returned {result!r}; expected AdjacentIntent"
    )
    assert result.trigger == "user_explicit"


def test_live_observed_miss_now_caught_without_evidence_returns_needs_evidence() -> None:
    """When the phrasing matches but the user has no evidence yet,
    the detector returns NeedsEvidenceIntent so the handler can ask
    for skills instead of running the engine on nothing."""
    result = detect_adjacent_intent(
        "find any job related my skill", _staged(),
        user_has_evidence=False,
        pending_offer=False,
    )
    assert isinstance(result, NeedsEvidenceIntent)
    assert result.trigger == "user_explicit"


# =========================================================================
# Negative: phrasings that should NOT trigger adjacency
# =========================================================================
@pytest.mark.parametrize("phrase", [
    # Verb + job-noun but NO skills connector. The user is just
    # asking generically; the planner / clarifier handles it.
    "find any job",
    "show me jobs",
    "look for roles",
    "get me work",
    "any jobs around",
    "what jobs are there",
    # Subject reversed -- "skills" leads, not "jobs". User is asking
    # about skills, not about jobs that match their skills.
    "skills related to this job",
    "skills that match this position",
    # Skill-gap question -- R-3 territory (same-role gap discovery),
    # not adjacency.
    "what skills do i need for this job",
    "what skills should i learn",
    # No job-noun at all.
    "tell me about my skills",
    "what can i do",
    "find a developer",
    # "matches" but not paired with my/the skills.
    "find jobs that match the description",
    "show me roles that fit the criteria",
    # Verb + skills but no job-noun.
    "use my skills",
    "find my skills",
])
def test_phrases_without_proper_job_to_skills_link_return_none(phrase) -> None:
    """Each phrase lacks the verb-led job-noun + skills-connector
    shape AR-8b targets. Detector must return None so the planner
    handles the ambiguity instead of fabricating adjacency intent."""
    result = detect_adjacent_intent(
        phrase, _staged(),
        user_has_evidence=True,
        pending_offer=False,
    )
    assert result is None, (
        f"phrase={phrase!r} returned {result!r}; expected None"
    )


# =========================================================================
# Negative: direct search-verb negation (reviewer-flagged blocking)
# =========================================================================
@pytest.mark.parametrize("phrase", [
    # Reviewer's blocking list.
    "don't show me jobs related to my skills",
    "do not show me jobs matching my skills",
    "please don't show me jobs that use my skills",
    "don't find jobs related to my skills",
    "never show me roles matching my skills",
    # Adjacent natural forms.
    "won't find jobs related to my skills",
    "will not show me jobs that match my skills",
    "please do not look for jobs related to my skills",
    "don't get me roles matching my skills",
    "dont show me jobs related to my skills",   # no apostrophe
])
def test_search_verb_negation_overrides_explicit_match_returns_none(phrase) -> None:
    """The AR-8b explicit patterns match on shape (verb + job-noun +
    skills connector). A leading negation flips the intent from
    request to decline. `_OTHER_ROLE_DECLINE_PATTERNS` only handles
    "other" forms; the new `_SEARCH_VERB_DECLINE_PATTERNS` guard
    fills the skill-connector gap."""
    result = detect_adjacent_intent(
        phrase, _staged(),
        user_has_evidence=True,
        pending_offer=False,
    )
    assert result is None, (
        f"phrase={phrase!r} matched adjacency despite leading "
        f"search-verb negation; got {result!r}"
    )


@pytest.mark.parametrize("phrase", [
    # Smart right-single-quote U+2019 (mobile autocorrect).
    "don’t show me jobs related to my skills",
    "please don’t show me jobs that use my skills",
    # Smart left-single-quote U+2018 (some editors).
    "don‘t find jobs related to my skills",
    # Single low-9 U+201A.
    "don‚t show me roles matching my skills",
])
def test_search_verb_negation_with_smart_apostrophes_also_caught(phrase) -> None:
    """Smart apostrophes from mobile autocorrect / various editors
    fold to ASCII `'` in `_normalize` before the decline pattern
    runs. The negation guard must reject all four code points in the
    `_APOSTROPHE_NORMALIZER` table the same way it rejects the
    straight apostrophe."""
    result = detect_adjacent_intent(
        phrase, _staged(),
        user_has_evidence=True,
        pending_offer=False,
    )
    assert result is None, (
        f"phrase={phrase!r} (smart apostrophe) matched adjacency "
        f"despite leading negation; got {result!r}"
    )


# =========================================================================
# AR-8b round-3: indirect negation (blocking)
# =========================================================================
@pytest.mark.parametrize("phrase", [
    # Reviewer's blocking round-3 reproductions.
    "I don't want you to show me jobs related to my skills",
    "I do not want you to find jobs related to my skills",
    "I would rather you not show me jobs related to my skills",
    # Adjacent natural forms.
    "I don't want to find jobs related to my skills",        # no pronoun
    "I would rather not show me jobs related to my skills",  # no pronoun
    "I prefer you not show me jobs related to my skills",
    "I prefer not to find jobs matching my skills",
    "I'd prefer you not find jobs related to my skills",
    # Want-object form (negate wanting the JOBS themselves).
    "I don't want jobs related to my skills",
    "I don't want any roles matching my skills",
    "I do not want those jobs that use my skills",
    "I don't need roles related to my skills",
])
def test_indirect_negation_in_same_clause_returns_none(phrase) -> None:
    """Indirect negation in the same clause must coerce decline.
    Round-2's direct-negation guard caught "don't <search verb>" but
    let "don't want you to <search verb>" / "would rather you not
    <search verb>" / "don't want <job-noun> ... my skills" slip
    through. Round-3 broadened decline patterns plus clause-scoped
    application close the gap."""
    result = detect_adjacent_intent(
        phrase, _staged(),
        user_has_evidence=True,
        pending_offer=False,
    )
    assert result is None, (
        f"phrase={phrase!r} matched adjacency despite indirect "
        f"negation in same clause; got {result!r}"
    )


# =========================================================================
# AR-8b round-3: compound pivots (decline + request) preserve the
# request clause
# =========================================================================
@pytest.mark.parametrize("phrase", [
    # Reviewer's blocking round-3 reproductions.
    "Don't show me this role; find jobs related to my skills",
    "Never show me this posting; instead find jobs related to my skills",
    "Don't show me that one. Show me roles that use my skills",
    # Adjacent natural pivots.
    "I don't want this role; find jobs related to my skills",
    "Not this one. Show me jobs that match my skills",
    "Skip this posting, instead find jobs related to my skills",
    "Don't show me this. However find jobs matching my skills",
])
def test_compound_pivot_preserves_request_clause(phrase) -> None:
    """A decline in one clause must NOT poison a clean adjacency
    request in another clause. Clause-scoped decline lets the second
    clause produce AdjacentIntent."""
    result = detect_adjacent_intent(
        phrase, _staged(),
        user_has_evidence=True,
        pending_offer=False,
    )
    assert isinstance(result, AdjacentIntent), (
        f"phrase={phrase!r} did not produce AdjacentIntent on the "
        f"second clause; got {result!r}"
    )
    assert result.trigger == "user_explicit"


# =========================================================================
# AR-8b round-4: `'d rather` contraction
# =========================================================================
@pytest.mark.parametrize("phrase", [
    # Reviewer's round-4 blocking: contraction of "would rather".
    "I'd rather you not show me jobs related to my skills",
    "we'd rather you not find jobs related to my skills",
    "they'd rather you not show me roles matching my skills",
    "you'd rather not show me jobs matching my skills",
    "he'd rather not find jobs related to my skills",
    "she'd rather you not show me jobs related to my skills",
    # Without subject pronoun in the not-clause.
    "I'd rather not find jobs related to my skills",
])
def test_contracted_would_rather_caught_as_decline(phrase) -> None:
    """`'d rather` is the natural contraction of `would rather`.
    Round-3's pattern only matched "would rather" literally, so
    "I'd rather you not show me jobs related to my skills" slipped
    through. Round-4 broadens the pattern to (i|we|they|you|he|she|
    it)'?d as well as "would"."""
    result = detect_adjacent_intent(
        phrase, _staged(),
        user_has_evidence=True,
        pending_offer=False,
    )
    assert result is None, (
        f"phrase={phrase!r} matched adjacency despite 'd-rather "
        f"contraction; got {result!r}"
    )


@pytest.mark.parametrize("phrase", [
    # Smart right-single-quote U+2019.
    "I’d rather you not show me jobs related to my skills",
    "we’d rather not find jobs matching my skills",
    # Smart left-single-quote U+2018.
    "I‘d rather you not show me jobs related to my skills",
])
def test_contracted_would_rather_with_smart_apostrophe_also_caught(phrase) -> None:
    """Smart apostrophes on the contraction (mobile autocorrect)
    fold to ASCII `'` in `_normalize`. The pattern `'?d` covers
    both."""
    result = detect_adjacent_intent(
        phrase, _staged(),
        user_has_evidence=True,
        pending_offer=False,
    )
    assert result is None, (
        f"phrase={phrase!r} (smart apostrophe) matched adjacency; "
        f"got {result!r}"
    )


# =========================================================================
# AR-8b round-4: terminal punctuation without trailing whitespace
# =========================================================================
@pytest.mark.parametrize("phrase", [
    # Reviewer's round-4 blocking reproductions.
    "Don't show me this role;find jobs related to my skills",
    "Don't show me that one.Show me roles that use my skills",
    # Adjacent natural forms.
    "Don't show me this role.Find jobs related to my skills",
    "Never show me this posting;find jobs related to my skills",
    "Don't show me that one!Show me roles that use my skills",
])
def test_terminal_punctuation_without_whitespace_still_splits(phrase) -> None:
    """Round-3's `_CLAUSE_BOUNDARY` ended with `\\s+`, so "role;find"
    (no space after `;`) was a single clause and the decline in the
    first half poisoned the request in the second half. Round-4
    drops the trailing whitespace requirement on terminal punctuation
    so the split fires either way."""
    result = detect_adjacent_intent(
        phrase, _staged(),
        user_has_evidence=True,
        pending_offer=False,
    )
    assert isinstance(result, AdjacentIntent), (
        f"phrase={phrase!r} did not produce AdjacentIntent after "
        f"terminal-punctuation-only split; got {result!r}"
    )
    assert result.trigger == "user_explicit"


# =========================================================================
# AR-8b round-4: bare "but" without preceding comma
# =========================================================================
@pytest.mark.parametrize("phrase", [
    # Reviewer's round-4 residual gap (now closed).
    "Don't show me this role but find jobs related to my skills",
    "Skip this posting but find jobs related to my skills",
    "Never mind this one but show me roles that use my skills",
])
def test_bare_but_separates_clauses(phrase) -> None:
    """Round-3 required a leading comma before "but" to recognize it
    as a clause boundary. Informal writing often omits the comma --
    "don't show me this role but find jobs ..." is a clear pivot.
    Round-4 adds bare " but " as a clause separator."""
    result = detect_adjacent_intent(
        phrase, _staged(),
        user_has_evidence=True,
        pending_offer=False,
    )
    assert isinstance(result, AdjacentIntent), (
        f"phrase={phrase!r} did not produce AdjacentIntent after "
        f"bare-but split; got {result!r}"
    )
    assert result.trigger == "user_explicit"


# =========================================================================
# AR-8b round-5: request-then-retraction ordering (blocking)
# =========================================================================
@pytest.mark.parametrize("phrase", [
    # Reviewer's round-5 blocking reproductions.
    "Show me jobs related to my skills, but don't show me jobs related to my skills.",
    "Find jobs related to my skills. Actually, don't show me any jobs related to my skills.",
    "Find jobs related to my skills; I would rather you not show me jobs related to my skills.",
    # Adjacent natural retractions.
    "Show me roles that match my skills. Actually don't show me roles matching my skills.",
    "Find jobs related to my skills, but I don't want you to show me jobs related to my skills",
    "Find jobs related to my skills; I'd rather you not show me jobs related to my skills",
])
def test_later_retraction_overrides_earlier_request(phrase) -> None:
    """The user's LAST intent-bearing clause wins. An earlier
    positive clause must be overridden by a later retraction so
    the final decision matches what the user just said. Round-4's
    `_has_explicit_adjacency_clause` short-circuited on the first
    positive clause and never considered the retraction; round-5
    iterates ALL clauses and tracks the last polarity."""
    result = detect_adjacent_intent(
        phrase, _staged(),
        user_has_evidence=True,
        pending_offer=False,
    )
    assert result is None, (
        f"phrase={phrase!r} matched adjacency despite later "
        f"retraction clause; got {result!r}"
    )


@pytest.mark.parametrize("phrase", [
    # The mirror of retraction: decline then pivot to positive.
    # Last clause is positive, so AdjacentIntent wins. Round-5's
    # last-clause-wins ordering must preserve this direction too.
    "Don't show me this role; find jobs related to my skills",
    "Don't show me that one. Show me roles that use my skills",
    "I would rather not show me this. Find jobs related to my skills",
    "Don't show me jobs related to my skills; actually, show me jobs related to my skills",
])
def test_later_request_overrides_earlier_decline(phrase) -> None:
    """Symmetric to retraction: a positive clause AFTER a decline
    must produce AdjacentIntent. Compound pivots in either direction
    work the same way under last-clause-wins."""
    result = detect_adjacent_intent(
        phrase, _staged(),
        user_has_evidence=True,
        pending_offer=False,
    )
    assert isinstance(result, AdjacentIntent), (
        f"phrase={phrase!r} did not produce AdjacentIntent on the "
        f"trailing positive clause; got {result!r}"
    )
    assert result.trigger == "user_explicit"


# =========================================================================
# AR-8b round-6: trailing standalone retractions (blocking)
# =========================================================================
@pytest.mark.parametrize("phrase", [
    # Reviewer's round-6 blocking reproductions.
    "Show me jobs related to my skills. Never mind.",
    "Find jobs related to my skills; no thanks.",
    "Show me roles matching my skills. Actually, no.",
    "Show me jobs related to my skills. Cancel that.",
    "Find jobs related to my skills. Forget it.",
    "Show me roles matching my skills. I changed my mind.",
    # Adjacent natural retractions.
    "Show me jobs related to my skills. Nevermind.",       # no space
    "Show me roles matching my skills. No thanks.",        # no comma form
    "Find jobs related to my skills. Sorry, no.",          # "sorry" softener
    "Show me jobs related to my skills. Forget about it.", # "forget about it"
    "Show me jobs related to my skills. Forget that.",
    "Show me jobs related to my skills. Cancel it.",
    "Show me roles matching my skills. Changed my mind.",  # no "i"
    "Show me roles matching my skills. I've changed my mind.",  # contraction
    "Show me roles matching my skills. I have just changed my mind.",
])
def test_trailing_standalone_retraction_overrides_earlier_request(phrase) -> None:
    """A standalone refusal at the END of a compound message must
    override an earlier positive clause. Round-5 only flipped on
    same-clause search-verb declines; round-6 adds clause-level
    pure-decline patterns ("never mind", "no thanks", "cancel that",
    "forget it", "i changed my mind") that participate in last-
    clause polarity tracking."""
    result = detect_adjacent_intent(
        phrase, _staged(),
        user_has_evidence=True,
        pending_offer=False,
    )
    assert result is None, (
        f"phrase={phrase!r} matched adjacency despite trailing "
        f"standalone retraction; got {result!r}"
    )


# =========================================================================
# AR-8b round-6: retraction phrases as substrings must NOT flip
# =========================================================================
@pytest.mark.parametrize("phrase", [
    # "never mind" inside a phrase, not at clause end.
    "Show me jobs related to my skills. I never mind helping",
    # "cancel that" inside a phrase, not at clause end.
    "Show me jobs related to my skills. I want to cancel that one",
    # "forget it" inside a phrase, not at clause end.
    "Show me jobs related to my skills. Don't forget it's still important",
    # "changed my mind" inside a phrase, not at clause end.
    "Show me roles matching my skills. I changed my mind about lunch",
])
def test_retraction_substring_does_not_flip_polarity(phrase) -> None:
    """Round-6 retraction patterns are anchored to clause-end
    (`\\s*[.!?]?\\s*$`) so a substring occurrence doesn't false-
    positive. The earlier positive clause wins -- the retraction
    phrase appearing mid-clause is irrelevant."""
    result = detect_adjacent_intent(
        phrase, _staged(),
        user_has_evidence=True,
        pending_offer=False,
    )
    assert isinstance(result, AdjacentIntent), (
        f"phrase={phrase!r} lost AdjacentIntent because a retraction "
        f"substring incorrectly fired; got {result!r}"
    )
    assert result.trigger == "user_explicit"


# =========================================================================
# AR-8b round-6: pivot back to positive after a retraction clause
# =========================================================================
@pytest.mark.parametrize("phrase", [
    "Never mind. Show me jobs related to my skills",
    "Cancel that. Find jobs related to my skills",
    "Forget it. Show me roles matching my skills",
])
def test_retraction_then_request_pivots_back_to_positive(phrase) -> None:
    """A standalone retraction in clause 1 followed by a positive
    clause 2 must produce AdjacentIntent. Last-clause-wins works in
    both directions: retraction -> request and request -> retraction."""
    result = detect_adjacent_intent(
        phrase, _staged(),
        user_has_evidence=True,
        pending_offer=False,
    )
    assert isinstance(result, AdjacentIntent), (
        f"phrase={phrase!r} did not produce AdjacentIntent on the "
        f"trailing positive clause; got {result!r}"
    )
    assert result.trigger == "user_explicit"


# =========================================================================
# AR-8b round-7: same-clause trailing retractions (blocking)
# =========================================================================
@pytest.mark.parametrize("phrase", [
    # Reviewer's round-7 blocking reproductions: explicit pattern
    # AND retraction in the SAME clause (no terminal punctuation
    # between them, so clause split doesn't fire).
    "Show me jobs related to my skills, never mind",
    "Find jobs related to my skills, no thanks",
    "Show me roles matching my skills, actually no",
    "Show me jobs related to my skills cancel that",
    "Find jobs related to my skills forget it",
    "Show me roles matching my skills I changed my mind",
    # Adjacent natural same-clause forms.
    "Show me roles matching my skills, no thank you",
    "Find jobs related to my skills sorry no",
    "Show me jobs related to my skills, sorry no",
    "Find jobs related to my skills, forget about it",
    "Show me roles matching my skills, I've changed my mind",
])
def test_same_clause_trailing_retraction_overrides_explicit(phrase) -> None:
    """Round-6 used `elif` for clause-level retraction; once the
    explicit branch was taken, retraction patterns were skipped.
    Round-7 evaluates retraction alongside search-verb decline
    within the explicit branch. Together with the "no" pattern's
    relaxed start anchor (start-OR-`[,.!?]`-boundary), this catches
    same-clause trailing retractions without losing the substring
    guard."""
    result = detect_adjacent_intent(
        phrase, _staged(),
        user_has_evidence=True,
        pending_offer=False,
    )
    assert result is None, (
        f"phrase={phrase!r} matched adjacency despite same-clause "
        f"trailing retraction; got {result!r}"
    )


# =========================================================================
# AR-8b round-7: substring guards still hold in same-clause case
# =========================================================================
@pytest.mark.parametrize("phrase", [
    # "no" mid-clause, not at end -- "I have no idea" shape.
    "Show me jobs related to my skills, I have no idea what to do",
    # "no" with non-boundary leading char -- shouldn't fire.
    "Show me jobs related to my skills with no education required",
    # "never mind" mid-clause.
    "Show me jobs related to my skills, I never mind helping out",
    # "cancel that" mid-clause.
    "Show me jobs related to my skills, I want to cancel that subscription",
    # "forget it" mid-clause.
    "Show me jobs related to my skills, don't forget it matters",
    # "changed my mind" mid-clause.
    "Show me jobs related to my skills, I changed my mind about lunch",
    # Compound "no thanks" inside a longer trailing phrase isn't end.
    "Show me jobs related to my skills, I said no thanks needed for now",
])
def test_round_7_substring_guards_preserve_explicit_match(phrase) -> None:
    """Round-7's same-clause retraction check MUST keep the existing
    clause-end anchors. A retraction phrase appearing mid-clause
    (with further content after) must NOT flip the polarity."""
    result = detect_adjacent_intent(
        phrase, _staged(),
        user_has_evidence=True,
        pending_offer=False,
    )
    assert isinstance(result, AdjacentIntent), (
        f"phrase={phrase!r} lost AdjacentIntent because a mid-clause "
        f"retraction substring incorrectly fired; got {result!r}"
    )
    assert result.trigger == "user_explicit"


# =========================================================================
# AR-8b round-8: retraction polarity guards (blocking)
# =========================================================================
@pytest.mark.parametrize("phrase", [
    # Negation flips "forget" / "cancel".
    "Show me jobs related to my skills but don't forget it.",
    "Show me jobs related to my skills; don't cancel it.",
    "Show me jobs related to my skills, do not forget it.",
    "Show me jobs related to my skills. Please don't forget it.",
    "Show me jobs related to my skills. Never forget it.",
    "Show me jobs related to my skills. Don't forget that.",
    # Subordinating conjunction introduces a reason, not the action.
    "Show me jobs related to my skills because I changed my mind.",
    "Show me jobs related to my skills while I cancel it.",
    "Show me jobs related to my skills since I changed my mind.",
    "Show me jobs related to my skills since I have changed my mind.",
    "Show me jobs related to my skills as I have changed my mind.",
    "Show me jobs related to my skills although I changed my mind.",
    # Subject pronoun before "never mind" -- positive sentiment.
    "Show me jobs related to my skills. I never mind.",
    "Show me jobs related to my skills. We never mind.",
    "Show me jobs related to my skills. I'll never mind.",
    "Show me jobs related to my skills. I would never mind.",
])
def test_polarity_inverting_context_suppresses_retraction(phrase) -> None:
    """Retraction shape alone is not enough -- semantics must be a
    retraction. Round-7 caught the literal phrases at clause end but
    didn't check the surrounding syntax. Round-8 adds three
    suppressor patterns:
       (a) Negation directly before "forget"/"cancel" inverts the
           verb. "Don't forget it" = "remember it".
       (b) Subordinating conjunction makes the retraction a reason.
           "because I changed my mind" = reason, not retraction.
       (c) Subject pronoun + "never mind" = "I don't object" =
           positive.
    All three cases must preserve the earlier positive request."""
    result = detect_adjacent_intent(
        phrase, _staged(),
        user_has_evidence=True,
        pending_offer=False,
    )
    assert isinstance(result, AdjacentIntent), (
        f"phrase={phrase!r} lost AdjacentIntent because retraction "
        f"suppressor failed to fire; got {result!r}"
    )
    assert result.trigger == "user_explicit"


# =========================================================================
# AR-8b round-8: suppressors do NOT block genuine retraction
# =========================================================================
@pytest.mark.parametrize("phrase", [
    # Genuine retraction with no inverting context -- round-6/7 tests
    # already cover the canonical forms; round-8 re-pins a sample to
    # prove the suppressor introduction didn't over-broaden.
    "Show me jobs related to my skills. Forget it.",
    "Show me jobs related to my skills. Cancel that.",
    "Show me jobs related to my skills. Never mind.",
    "Show me jobs related to my skills. I changed my mind.",
    "Show me jobs related to my skills, no thanks.",
    "Show me jobs related to my skills, actually no.",
])
def test_genuine_retraction_still_fires_after_round_8(phrase) -> None:
    """Round-8's polarity-inverting suppressors must NOT block plain
    retractions. Each of these has no negation before forget/cancel,
    no subordinating conjunction, no subject pronoun before never
    mind -- they're canonical retractions and must flip to None."""
    result = detect_adjacent_intent(
        phrase, _staged(),
        user_has_evidence=True,
        pending_offer=False,
    )
    assert result is None, (
        f"phrase={phrase!r} unexpectedly produced {result!r}; "
        f"round-8 suppressor over-broadened"
    )


# =========================================================================
# AR-8b round-8: bare "no" without leading punctuation also flips
# =========================================================================
@pytest.mark.parametrize("phrase", [
    # Reviewer's round-7 secondary gap (now closed).
    "Show me jobs related to my skills no",
    "Find jobs related to my skills no",
    "Show me roles matching my skills no",
])
def test_bare_no_at_clause_end_without_punctuation_flips(phrase) -> None:
    """Round-7 required `(?:^|[,.!?])\\s*` before bare "no" to avoid
    "I have no idea" false positives. The substring guard still works
    because the end-anchor `\\s*[.!?]?\\s*$` rejects "no" with content
    after. Round-8 relaxes the leading anchor to `\\b` so the comma
    form and the unpunctuated form converge on the same answer."""
    result = detect_adjacent_intent(
        phrase, _staged(),
        user_has_evidence=True,
        pending_offer=False,
    )
    assert result is None, (
        f"phrase={phrase!r} matched adjacency despite trailing "
        f"bare 'no'; got {result!r}"
    )


@pytest.mark.parametrize("phrase", [
    # Substring guard still holds with the relaxed bare "no" pattern.
    "Show me jobs related to my skills with no education required",
    "Show me jobs related to my skills, I have no idea what to do",
    "Show me roles matching my skills and I have no preferences",
])
def test_relaxed_bare_no_does_not_match_mid_clause(phrase) -> None:
    """The end-anchor (`\\s*[.!?]?\\s*$`) still prevents "no" mid-
    clause from firing. Only "no" at the literal end of the clause
    counts as retraction."""
    result = detect_adjacent_intent(
        phrase, _staged(),
        user_has_evidence=True,
        pending_offer=False,
    )
    assert isinstance(result, AdjacentIntent), (
        f"phrase={phrase!r} lost AdjacentIntent to relaxed bare "
        f"'no' pattern; got {result!r}"
    )


# =========================================================================
# AR-8b round-9: per-match suppressor scope (blocking)
# =========================================================================
@pytest.mark.parametrize("phrase", [
    # Reviewer's round-9 reproductions: clause has BOTH a
    # suppressed phrase AND a genuine trailing retraction. The
    # suppressor must apply only to its associated phrase; the
    # trailing retraction must still fire.
    "Show me jobs related to my skills, don't forget it, actually no.",
    "Show me jobs related to my skills because I need work, actually no.",
    "Show me jobs related to my skills while I think, no thanks.",
    # Adjacent natural shapes.
    "Show me jobs related to my skills, don't cancel it, actually no.",
    "Show me jobs related to my skills since I'm bored, no thanks.",
])
def test_unrelated_suppressor_does_not_cancel_trailing_retraction(phrase) -> None:
    """Round-8 dropped has_retraction for the whole clause whenever
    ANY suppressor matched. That over-broadened to cancel genuine
    trailing retractions ("don't forget it ... actually no" -- the
    "don't forget" suppressor is for "forget it", not for "actually
    no"). Round-9 per-match scoping: each retraction is checked
    against suppressors targeting that specific match."""
    result = detect_adjacent_intent(
        phrase, _staged(),
        user_has_evidence=True,
        pending_offer=False,
    )
    assert result is None, (
        f"phrase={phrase!r} matched adjacency despite trailing "
        f"retraction; got {result!r}"
    )


# =========================================================================
# AR-8b round-9: pronoun-window suppressor excludes verbs of speech
# =========================================================================
@pytest.mark.parametrize("phrase", [
    # Reviewer's round-9 reproductions.
    "Show me jobs related to my skills. I said never mind.",
    "I mean never mind.",
    "I guess never mind.",
    "Show me jobs related to my skills. We said never mind.",
    # Adjacent natural shapes.
    "Show me jobs related to my skills. I think never mind.",
    "Show me jobs related to my skills. You said never mind.",
    "Show me jobs related to my skills. They guess never mind.",
])
def test_verbs_of_speech_do_not_suppress_never_mind_retraction(phrase) -> None:
    """Round-8 allowed any pronoun + up-to-2-word window before
    "never mind" to suppress. That swallowed "I said never mind" /
    "I mean never mind" / "I guess never mind" -- but "said"/"mean"/
    "guess" are verbs of speech: the user is reporting what they
    (or someone else) said, which IS a retraction.

    Round-9 restricts the pronoun-window to MODAL verbs only
    (would/will/do/does/did/might/could/should). "I never mind",
    "I'll never mind", "I would never mind" still suppress because
    "I/you" is the grammatical subject of "never mind"."""
    result = detect_adjacent_intent(
        phrase, _staged(),
        user_has_evidence=True,
        pending_offer=False,
    )
    assert result is None, (
        f"phrase={phrase!r} matched adjacency despite genuine "
        f"reported retraction; got {result!r}"
    )


# =========================================================================
# AR-8b round-9: pronoun + modal still suppresses (round-8 positives)
# =========================================================================
@pytest.mark.parametrize("phrase", [
    # Round-8 positives -- these must still produce AdjacentIntent
    # after the round-9 modal-only window tightening.
    "Show me jobs related to my skills. I never mind.",
    "Show me jobs related to my skills. I'll never mind.",
    "Show me jobs related to my skills. I would never mind.",
    "Show me jobs related to my skills. We never mind.",
    # Round-8 negation+forget/cancel positives must still survive.
    "Show me jobs related to my skills but don't forget it.",
    "Show me jobs related to my skills; don't cancel it.",
    # Round-8 subordinating-conjunction positives must still survive
    # when the retraction is in the SAME subordinate (no comma).
    "Show me jobs related to my skills because I changed my mind.",
    "Show me jobs related to my skills while I cancel it.",
])
def test_round_8_positives_preserved_under_per_match_suppression(phrase) -> None:
    """The shift to per-match suppressor scoping must not regress
    round-8 cases where a single suppressor legitimately invalidates
    a single retraction. These are the canonical "good" shapes from
    round-8 -- they must all still produce AdjacentIntent."""
    result = detect_adjacent_intent(
        phrase, _staged(),
        user_has_evidence=True,
        pending_offer=False,
    )
    assert isinstance(result, AdjacentIntent), (
        f"phrase={phrase!r} lost AdjacentIntent under round-9 "
        f"per-match suppressor; got {result!r}"
    )
    assert result.trigger == "user_explicit"


# =========================================================================
# AR-8b round-10: pronoun-modal suppressor must be anchored to THIS
# match's preceding context (blocking)
# =========================================================================
@pytest.mark.parametrize("phrase", [
    # Reviewer's round-10 reproductions: earlier "pronoun + modal +
    # never mind" used as positive sentiment ("I would never mind
    # waiting") followed by a LATER, separate "never mind"
    # retraction. The suppressor must not fire for the trailing
    # match because the pronoun-modal prefix doesn't immediately
    # precede THAT match.
    "Show me jobs related to my skills, I would never mind waiting, actually never mind.",
    "Show me jobs related to my skills, I'd never mind commuting, sorry never mind.",
    "Show me jobs related to my skills, we will never mind the hours, now never mind.",
    # Adjacent natural shapes.
    "Show me jobs related to my skills. I would never mind waiting. Actually never mind.",
    "Show me jobs related to my skills, you'd never mind the commute, never mind",
])
def test_pronoun_modal_suppressor_is_anchored_to_match_preceding_text(phrase) -> None:
    """Round-9 ran `_PRONOUN_MODAL_NEVER_MIND.search(clause)` over
    the whole clause -- an earlier positive "I would never mind X"
    incorrectly cancelled a later, separate "never mind" retraction.
    Round-10 anchors the suppressor pattern to the END of the
    preceding text (preceding = clause[:match.start()]) so the
    pronoun+modal prefix must immediately precede THIS match."""
    result = detect_adjacent_intent(
        phrase, _staged(),
        user_has_evidence=True,
        pending_offer=False,
    )
    assert result is None, (
        f"phrase={phrase!r} matched adjacency despite trailing "
        f"genuine 'never mind' retraction; got {result!r}"
    )


# =========================================================================
# AR-8b round-10: legitimate single-occurrence "I never mind" cases
# preserved
# =========================================================================
@pytest.mark.parametrize("phrase", [
    # Single-occurrence "pronoun + (optional modal) + never mind"
    # at clause end MUST still suppress (positive sentiment, not
    # retraction). Round-8 positives re-pinned under round-10.
    "Show me jobs related to my skills. I never mind.",
    "Show me jobs related to my skills. We never mind.",
    "Show me jobs related to my skills. I'll never mind.",
    "Show me jobs related to my skills. I would never mind.",
    "Show me jobs related to my skills. You'll never mind.",
    "Show me jobs related to my skills. They might never mind.",
])
def test_pronoun_modal_suppressor_still_fires_on_direct_preceding(phrase) -> None:
    """The prefix-only suppressor must STILL fire when the pronoun-
    modal prefix immediately precedes "never mind" -- this is the
    case the suppressor exists for. Round-8 positives must survive
    the round-10 refactor."""
    result = detect_adjacent_intent(
        phrase, _staged(),
        user_has_evidence=True,
        pending_offer=False,
    )
    assert isinstance(result, AdjacentIntent), (
        f"phrase={phrase!r} lost AdjacentIntent under round-10 "
        f"anchored suppressor; got {result!r}"
    )
    assert result.trigger == "user_explicit"


# =========================================================================
# AR-8b round-11: adverbs between pronoun/modal and "never mind"
# (blocking regression)
# =========================================================================
@pytest.mark.parametrize("phrase", [
    # Reviewer's round-11 reproductions (prepended with the
    # adjacency request so the assertion has something to preserve;
    # the reviewer's standalone forms imply they trail an earlier
    # explicit request).
    "Show me jobs related to my skills. I really never mind.",
    "Show me jobs related to my skills. I honestly never mind.",
    "Show me jobs related to my skills. We generally never mind.",
    "Show me jobs related to my skills. I would really never mind.",
    "Show me jobs related to my skills. They might honestly never mind.",
    # Adjacent natural shapes.
    "Show me jobs related to my skills. I truly never mind.",
    "Show me jobs related to my skills. I absolutely never mind.",
    "Show me jobs related to my skills. You definitely never mind.",
    "Show me jobs related to my skills. We usually never mind.",
    "Show me jobs related to my skills. They genuinely never mind.",
    "Show me jobs related to my skills. I'll really never mind.",
    "Show me jobs related to my skills. We would honestly never mind.",
])
def test_adverb_between_pronoun_and_never_mind_still_suppresses(phrase) -> None:
    """Round-10's prefix-only pattern only allowed `(?:modal\\s+)?`
    between pronoun and "never mind". Natural English includes
    adverbs ("I really never mind", "I honestly never mind", "We
    generally never mind") that broke the suppressor. Round-11 adds
    a controlled adverb allowlist; up to 2 prefix words total covers
    "I would really never mind" (modal + adverb) too."""
    result = detect_adjacent_intent(
        phrase, _staged(),
        user_has_evidence=True,
        pending_offer=False,
    )
    assert isinstance(result, AdjacentIntent), (
        f"phrase={phrase!r} lost AdjacentIntent because adverb "
        f"between pronoun and 'never mind' broke suppression; "
        f"got {result!r}"
    )
    assert result.trigger == "user_explicit"


# =========================================================================
# AR-8b round-11: reporting verbs still excluded
# =========================================================================
@pytest.mark.parametrize("phrase", [
    # Reporting verbs MUST NOT consume the prefix slot. Round-9's
    # cases must still produce None.
    "Show me jobs related to my skills. I said never mind.",
    "Show me jobs related to my skills. I mean never mind.",
    "Show me jobs related to my skills. I guess never mind.",
    "Show me jobs related to my skills. We said never mind.",
    # Adverb followed by reporting verb is still a genuine
    # retraction -- the reporting verb is the head of the prefix
    # and "never mind" is being reported.
    "Show me jobs related to my skills. I really said never mind.",
    "Show me jobs related to my skills. I honestly mean never mind.",
    # Modal followed by reporting verb is also a genuine retraction.
    "Show me jobs related to my skills. I would say never mind.",
])
def test_reporting_verbs_still_excluded_from_suppressor_under_round_11(phrase) -> None:
    """The round-11 allowlist must NOT include reporting verbs
    (said/mean/guess/think/say). Even when an adverb appears in the
    prefix, a following reporting verb breaks suppression because
    `(?:...){0,2}` can only consume words in the allowlist and the
    reporting verb fails."""
    result = detect_adjacent_intent(
        phrase, _staged(),
        user_has_evidence=True,
        pending_offer=False,
    )
    assert result is None, (
        f"phrase={phrase!r} matched adjacency despite a reporting "
        f"verb between pronoun and 'never mind'; got {result!r}"
    )


# =========================================================================
# Sanity: empty-shape message returns None for the right reason
# =========================================================================
def test_unrelated_negation_message_returns_none_without_match() -> None:
    """A message that starts with "don't show me" but contains no
    adjacency-shaped clause returns None. The clause-scoped helper
    iterates clauses without finding any explicit-pattern match, so
    the decline guard is effectively unreachable on this input --
    None comes from "no explicit clause", not from coerced decline."""
    result = detect_adjacent_intent(
        "don't show me your homework",
        _staged(),
        user_has_evidence=True,
        pending_offer=False,
    )
    assert result is None


# =========================================================================
# AR-8b round-2 coverage-gap closures (non-blocking but worth catching)
# =========================================================================
@pytest.mark.parametrize("phrase", [
    "find me jobs related to my skills",
    "what roles would match my skills",
    "which jobs best match my skills",
    "find me roles matching my skills",
    "what jobs could match my skills",
    "what positions should fit my skills",
    "find jobs closely related to my skills",
])
def test_round_2_coverage_phrases_match_adjacent_intent(phrase) -> None:
    """Reviewer-flagged non-blocking coverage gaps from AR-8b round-2:
       - "find me" verb form (previously only "find" or "get me");
       - modal between job-noun and connector (would/could/should/might);
       - "best" / "closely" adverb in the same slot.
    All three should produce AdjacentIntent."""
    result = detect_adjacent_intent(
        phrase, _staged(),
        user_has_evidence=True,
        pending_offer=False,
    )
    assert isinstance(result, AdjacentIntent), (
        f"phrase={phrase!r} returned {result!r}; expected AdjacentIntent"
    )
    assert result.trigger == "user_explicit"


# =========================================================================
# Negative: same-role anchor still blocks new patterns
# =========================================================================
@pytest.mark.parametrize("phrase", [
    "find jobs related to my skills for this role",
    "find jobs related to my skills for this position",
    "show me roles that match my skills for this job",
])
def test_same_role_anchor_overrides_new_patterns(phrase) -> None:
    """Even when an AR-8b pattern matches, an explicit "for this
    role/job/position" anchor pulls the message into R-3 territory.
    The detector must return None so R-3 handles it."""
    result = detect_adjacent_intent(
        phrase, _staged(),
        user_has_evidence=True,
        pending_offer=False,
    )
    assert result is None, (
        f"phrase={phrase!r} matched adjacency despite same-role "
        f"anchor; got {result!r}"
    )


# =========================================================================
# Negative: object-scoped decline still blocks new patterns
# =========================================================================
@pytest.mark.parametrize("phrase", [
    "i don't want other jobs that match my skills",
    "don't show me other roles related to my skills",
])
def test_other_role_decline_overrides_new_patterns(phrase) -> None:
    """When the user explicitly rejects OTHER roles, an AR-8b
    pattern in the same message does not coerce adjacency."""
    result = detect_adjacent_intent(
        phrase, _staged(),
        user_has_evidence=True,
        pending_offer=False,
    )
    assert result is None, (
        f"phrase={phrase!r} matched adjacency despite "
        f"object-scoped decline; got {result!r}"
    )


# =========================================================================
# Regression: every prior AR-2 explicit phrasing still matches
# =========================================================================
@pytest.mark.parametrize("phrase", [
    "what other jobs are there?",
    "What other roles could I look at?",
    "what other positions",
    "any other roles?",
    "any other jobs",
    "any other openings I should know about",
    "different roles I could try",
    "other postings worth considering",
    "show me other jobs",
    "show me other roles",
    "what am I close to in other roles?",
    "anything else I could do",
    "anything else I can do",
    "roles like this one",
    "any role like that",
])
def test_prior_ar2_phrases_still_match_after_new_patterns_added(phrase) -> None:
    """AR-8b adds patterns; it MUST NOT regress any AR-2 positives.
    Mirrors the AR-2 happy-path parametrize list verbatim so a future
    pattern reorder or split is caught here as well as in AR-2."""
    result = detect_adjacent_intent(
        phrase, _staged(),
        user_has_evidence=True,
        pending_offer=False,
    )
    assert isinstance(result, AdjacentIntent), (
        f"AR-2 phrase={phrase!r} no longer matches after AR-8b "
        f"additions; got {result!r}"
    )
