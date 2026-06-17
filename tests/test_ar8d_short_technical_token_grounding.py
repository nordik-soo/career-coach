"""AR-8d tests: controlled short technical-token grounding.

Live observation (2026-06-10):
  User: "I know python, React, SQL, I have two years experience"
  Log:  extractor_dropped=['skill_ungrounded:SQL']
  User: "I know javascript, API, database, git version control"
  Log:  extractor_dropped=['skill_ungrounded:API']

The >=4-char grounding floor in `_is_grounded` was dropping every
short technical token (SQL=3, API=3, Git=3, C#=2, C++=3, JS=2,
TS=2) because the >=4 char rule treats them as hallucination
shortcuts. The fix is a controlled allowlist + contextual validation:
  - Allowlist (locked): SQL, API, Git, C#, C++, JS, TS.
  - Name AND evidence must normalize to the SAME approved token.
  - Token must occur in the message with alphanumeric boundaries on
    both sides (`(?<![A-Za-z0-9])` / `(?![A-Za-z0-9])`, NOT `\\b`
    because C++/C# end in non-word chars).
  - Deliberately excluded: R, Go, AI, UI, CD (high false-positive
    risk against ordinary words).
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.nodb


from skillbridge.chat.extractor import (
    _APPROVED_SHORT_TECHNICAL_TOKENS,
    _is_approved_short_technical_token,
    _is_grounded,
)


# =========================================================================
# Allowlist contents
# =========================================================================
def test_approved_set_contains_locked_tokens() -> None:
    """The set MUST contain exactly the seven reviewer-locked
    tokens. Adding or removing a member is a contract change that
    needs explicit review."""
    assert _APPROVED_SHORT_TECHNICAL_TOKENS == frozenset({
        "sql", "api", "git", "c#", "c++", "js", "ts",
    })


@pytest.mark.parametrize("excluded", ["r", "go", "ai", "ui", "cd"])
def test_excluded_tokens_not_in_allowlist(excluded) -> None:
    """R / Go / AI / UI / CD are intentionally OUT. False-positive
    risk against ordinary words is too high."""
    assert excluded not in _APPROVED_SHORT_TECHNICAL_TOKENS


# =========================================================================
# Approved tokens accepted at sensible boundaries
# =========================================================================
@pytest.mark.parametrize("name, evidence, message", [
    ("SQL",  "SQL",  "I know python and SQL"),
    ("sql",  "sql",  "I know python and sql"),
    ("SQL",  "SQL",  "SQL is my best skill"),                  # at start
    ("SQL",  "SQL",  "my best skill is SQL"),                  # at end
    ("SQL",  "SQL",  "I do SQL, Python, and React."),          # comma-bounded
    ("API",  "API",  "I worked on API integrations"),
    ("Git",  "Git",  "I use Git daily"),
    ("Git",  "Git",  "Git/version control basics."),
    ("C#",   "C#",   "I write C# code"),
    ("C#",   "C#",   "My languages: C#, Java"),
    ("C++",  "C++",  "I do C++ embedded work"),
    ("C++",  "C++",  "I write C++ and C# code"),
    ("JS",   "JS",   "I use JS and CSS"),
    ("TS",   "TS",   "I use TS for type safety"),
])
def test_approved_short_tokens_accepted_with_boundary(
    name, evidence, message,
) -> None:
    """Each approved short token grounds when it appears in the
    message with alphanumeric boundaries on both sides."""
    assert _is_grounded(
        evidence, message.lower(), skill_name=name,
    ), f"name={name!r} evidence={evidence!r} message={message!r}"


# =========================================================================
# Compound tokens (C++ / C#) handle non-word boundary correctly
# =========================================================================
@pytest.mark.parametrize("message", [
    "I write C++ for embedded systems",
    "C++ programmer here",
    "Strong C++ background",
    "I do C++ and Python.",
    "Languages: C++, C#",     # comma after C++
])
def test_cpp_accepted_in_natural_contexts(message) -> None:
    assert _is_grounded("C++", message.lower(), skill_name="C++")


@pytest.mark.parametrize("message", [
    "I code in C#",
    "C# is my main language",
    "I'm a C# developer.",
    "Languages: C#, Java",
])
def test_csharp_accepted_in_natural_contexts(message) -> None:
    assert _is_grounded("C#", message.lower(), skill_name="C#")


# =========================================================================
# Boundary rejection: compound words / no-boundary cases
# =========================================================================
@pytest.mark.parametrize("name, message", [
    # "SQL" inside "MySQL" -- alphanumeric on left.
    ("SQL", "I use MySQL daily"),
    # "API" inside "APIs" -- alphanumeric on right.
    ("API", "I built APIs all year"),
    # "Git" inside "Github" -- alphanumeric on right.
    ("Git", "Github is my home"),
    # "JS" inside "JSON" -- alphanumeric on right.
    ("JS", "I parse JSON daily"),
    # "TS" inside "TSV" -- alphanumeric on right.
    ("TS", "I work with TSV files"),
    # "C++" followed by digit -- alphanumeric on right ("C++14").
    ("C++", "I use C++14 features"),
    # "C#" followed by digit -- alphanumeric on right ("C#7").
    ("C#", "C#7 introduced patterns"),
    # Approved token but NOT in the message at all.
    ("SQL", "I know python and react"),
    ("API", "I know python and react"),
])
def test_approved_tokens_rejected_without_word_boundary(
    name, message,
) -> None:
    """Allowlist alone isn't enough: the token must occur with
    non-word boundaries on both sides (or message start/end).
    `MySQL` doesn't ground `SQL`; `APIs` doesn't ground `API`;
    `C++14` doesn't ground `C++`."""
    assert not _is_grounded(name, message.lower(), skill_name=name), (
        f"name={name!r} message={message!r} should NOT have grounded"
    )


# =========================================================================
# Round-2: underscore boundary -- `\w` (not `[A-Za-z0-9]`) so
# identifier suffixes like `API_v2` correctly reject the token
# =========================================================================
@pytest.mark.parametrize("name, message", [
    # The reviewer's blocking reproductions: underscore boundary.
    ("API", "I built API_v2 endpoints"),
    ("SQL", "I work in SQL_mode every day"),
    ("JS",  "app.JS_bundle is huge"),
    ("TS",  "TS_config sets the options"),
    ("C++", "C++_library is the tag"),
    ("C#",  "C#_service handles auth"),
    ("Git", "I run Git_blame regularly"),
    # Underscore on the LEFT side too.
    ("API", "I push to _APIs there"),
    ("SQL", "the _SQL prefix is special"),
])
def test_underscore_boundary_rejected_per_w_class(name, message) -> None:
    """Round-1 used `[A-Za-z0-9]` for boundaries which treats
    underscore as a separator -- so `API_v2` incorrectly grounded
    `API`. Round-2 uses `\\w` (which includes underscore and Unicode
    letters), matching the locked contract. `API_v2` is now
    correctly an identifier, not a free-standing API mention."""
    assert not _is_grounded(name, message.lower(), skill_name=name), (
        f"name={name!r} message={message!r} grounded under "
        f"underscore boundary; round-2 should reject"
    )


@pytest.mark.parametrize("name, message", [
    # Unicode letter adjacent to the token -- `\w` includes Unicode
    # letters by default in Python regex, so naïveSQL or SQLé
    # correctly reject.
    ("SQL", "I love naïveSQL libraries"),
    ("SQL", "I write SQLé queries"),
    ("API", "I build APIé endpoints"),
    ("JS",  "I love réactJS"),
    ("JS",  "JSè bundles are huge"),
    ("Git", "I run Gitü daily"),
])
def test_unicode_letter_boundary_rejected(name, message) -> None:
    """`\\w` covers Unicode letters too. A token adjacent to an
    accented character is part of a larger identifier, not a
    free-standing skill mention."""
    assert not _is_grounded(name, message.lower(), skill_name=name), (
        f"name={name!r} message={message!r} grounded with Unicode "
        f"adjacency; should reject"
    )


# =========================================================================
# Name / evidence mismatch rejection
# =========================================================================
@pytest.mark.parametrize("name, evidence, message", [
    # Both in allowlist but different members -- still reject.
    ("JavaScript", "JS",  "I know JS"),
    ("TypeScript", "TS",  "I use TS daily"),
    ("JS",         "SQL", "I know SQL and JS"),
    # Evidence is a non-allowlist token even though name is allowed.
    ("SQL",        "DB",  "I work on DB and SQL"),
    # Name is non-allowlist, evidence is allowlisted -- name is what
    # gets stored; mismatch means we can't trust the LLM's claim.
    ("Structured Query Language", "SQL", "I know SQL"),
])
def test_name_and_evidence_must_match_same_approved_token(
    name, evidence, message,
) -> None:
    """The validator requires name AND evidence to normalize to the
    SAME approved token. JS != JavaScript even though both refer to
    the same skill; the LLM's canonicalization isn't accepted."""
    assert not _is_grounded(
        evidence, message.lower(), skill_name=name,
    ), f"name={name!r} evidence={evidence!r} should NOT have grounded"


# =========================================================================
# Deliberately-excluded tokens are still rejected
# =========================================================================
@pytest.mark.parametrize("name, message", [
    # The reviewer's specific exclusion -- R as a language.
    ("R",  "I know R the language"),
    ("R",  "I program in R"),
    ("R",  "I use R for stats"),
    # Other exclusions.
    ("Go", "I write Go services"),
    ("AI", "I work on AI projects"),
    ("UI", "I design UI components"),
    ("CD", "I run CI/CD pipelines"),
])
def test_excluded_tokens_still_rejected(name, message) -> None:
    """R, Go, AI, UI, CD remain rejected even when the user
    legitimately means them as technical tokens. The false-positive
    risk against ordinary words is too high to include them in the
    allowlist."""
    assert not _is_grounded(name, message.lower(), skill_name=name), (
        f"name={name!r} should remain rejected"
    )


# =========================================================================
# Existing slot escapes still work
# =========================================================================
def test_shift_preference_escape_still_works() -> None:
    """The closed-vocabulary slot escape ('day', 'ft', 'pt') runs
    BEFORE the AR-8d escape. Unchanged behavior."""
    assert _is_grounded("day", "i prefer day shifts", slot="shift_preference")
    assert _is_grounded("ft",  "ft work please", slot="work_type_preference")


def test_skill_name_kwarg_default_preserves_old_behavior() -> None:
    """When `_is_grounded` is called WITHOUT `skill_name`, the
    AR-8d escape doesn't fire and short tokens drop as before.
    This protects callers that don't pass skill_name (e.g. slot
    grounding calls)."""
    # SQL would ground with skill_name=SQL, but without skill_name
    # it falls through to the floor and is dropped.
    assert not _is_grounded("SQL", "i know sql")


def test_long_evidence_path_unchanged() -> None:
    """Evidence with len >= 4 takes the existing substring-match
    path and never consults the AR-8d escape."""
    assert _is_grounded("python", "i know python")
    assert not _is_grounded("python", "i know java")
    # JavaScript (full form) is >= 4 chars so passes naturally.
    assert _is_grounded("javascript", "i love javascript")


# =========================================================================
# _is_approved_short_technical_token unit
# =========================================================================
@pytest.mark.parametrize("name, evidence, msg, expected", [
    # Direct positives.
    ("SQL",  "SQL",  "i know sql",                True),
    ("c#",   "c#",   "i write c# code",           True),
    ("c++",  "c++",  "languages: c++, c#",        True),
    # Case insensitivity.
    ("SqL",  "sQl",  "I know SQL",                True),
    # Whitespace tolerance on name/evidence.
    ("  SQL  ", "SQL", "i know sql",              True),
    ("SQL", "  SQL  ", "i know sql",              True),
    # Non-string inputs rejected defensively.
    (None,   "SQL",  "i know sql",                False),
    ("SQL",  None,   "i know sql",                False),
    (123,    "SQL",  "i know sql",                False),
    # Empty / whitespace-only after normalization.
    ("",     "",     "i know sql",                False),
    ("   ",  "   ",  "i know sql",                False),
])
def test_approved_short_token_predicate(
    name, evidence, msg, expected,
) -> None:
    """Direct unit on the predicate."""
    assert _is_approved_short_technical_token(
        name, evidence, msg,
    ) is expected
