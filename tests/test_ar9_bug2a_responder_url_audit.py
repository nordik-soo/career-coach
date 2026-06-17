"""AR-9.bug.2a sub-step 6: static guard against raw URL access in responder.

ONE focused AST check, not a broad framework. After sub-step 4
migrated every URL-consuming function to read from SanitizedResponderView's
projected items, responder.py must contain ZERO direct raw-URL
dict-idiom reads.

Two idioms checked:
  - .get("url")
  - ["url"]

The third historical idiom — `.url` attribute access — is intentionally
NOT checked, because it's now the legitimate way consumers read a
SanitizedURL from projected view items (e.g., `r.url.raw`).

Failures indicate a missed migration or a regression. The error message
lists every violation with line numbers so the fix is mechanical.

Sub-step 4 migration target functions (12) that previously contained
these patterns are all now reading from view:
  _build_user_block (v1), _build_user_block_v2
  _policy_ok, _policy_ok_v2
  _fallback_reply, _fallback_reply_v2
  _present_matches_fallback, _present_matches_fallback_v2
  _recommend_adjacent_roles_fallback_v2
  _describe_adjacent_role_fallback_v2
  _explain_gap_fallback_v2
  _registry_grounded_explain_gap_fallback
"""
from __future__ import annotations

import ast
import pathlib

import pytest

pytestmark = pytest.mark.nodb


_RESPONDER_PATH = (
    pathlib.Path(__file__).parent.parent
    / "skillbridge" / "chat" / "responder.py"
)

# AR-9.feat.coach-tiers step 9: the tier projection layer is now part
# of the responder URL pipeline. Its builder consumes a TieredEvidence
# (frozen dataclasses) — never a raw job dict — and projects
# Validated → SanitizedURL. Any regression that introduces a raw URL
# read inside the builder must fail this guard.
_URL_VIEWS_PATH = (
    pathlib.Path(__file__).parent.parent
    / "skillbridge" / "chat" / "url_views.py"
)
_TIERED_BUILDER_NAMES: frozenset[str] = frozenset({
    "build_sanitized_responder_view_for_tiered_matches",
    "_project_validated_to_sanitized",
    "_project_job_facts",
    "_project_training_option",
    "_project_prioritized_gap",
    "_project_non_blocking_gap",
    "_project_transferable_pair",
    "_project_strong_match",
    "_project_stretch_match",
    "_project_adjacent_job",
})


def _collect_url_dict_accesses(source: str) -> list[tuple[int, str, str]]:
    """Walk the AST and find .get("url") and ["url"] accesses.

    Returns a list of (lineno, idiom, function_name) tuples.
    function_name is the enclosing function (or "<module>" if at
    module level).
    """
    tree = ast.parse(source)
    violations: list[tuple[int, str, str]] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.func_stack: list[str] = ["<module>"]

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.func_stack.append(node.name)
            self.generic_visit(node)
            self.func_stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self.func_stack.append(node.name)
            self.generic_visit(node)
            self.func_stack.pop()

        def _current_func(self) -> str:
            return self.func_stack[-1]

        def visit_Call(self, node: ast.Call) -> None:
            # .get("url")
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "url"
            ):
                violations.append(
                    (node.lineno, '.get("url")', self._current_func())
                )
            self.generic_visit(node)

        def visit_Subscript(self, node: ast.Subscript) -> None:
            # ["url"]
            slice_node = node.slice
            if (
                isinstance(slice_node, ast.Constant)
                and slice_node.value == "url"
            ):
                violations.append(
                    (node.lineno, '["url"]', self._current_func())
                )
            self.generic_visit(node)

    Visitor().visit(tree)
    return violations


def test_responder_has_no_raw_url_dict_accesses():
    """Sub-step 6 static guard.

    responder.py must contain ZERO `.get("url")` or `["url"]` accesses.
    After sub-step 4, every URL read goes through
    SanitizedResponderView projected items (where `url` is a
    SanitizedURL dataclass attribute, not a dict key).

    AR-9.feat.coach-tiers step 9: when CP2 wires the new
    `present_tiered_matches` responder function into responder.py,
    this same guard automatically catches any raw URL access inside
    it. The CP2 function MUST read URLs from the SanitizedResponderView's
    `prompt_tiered_*` projections (where `url` is a SanitizedURL),
    never via `.get("url")` on the underlying TieredEvidence dict.

    A non-empty violations list means either:
      - A new consumer was added without migrating to view (regression);
      - A migrated function regressed to raw access; OR
      - A non-URL `url` field is being read (should be renamed or refactored).
    """
    source = _RESPONDER_PATH.read_text(encoding="utf-8")
    violations = _collect_url_dict_accesses(source)
    if violations:
        rendered = "\n".join(
            f"  responder.py:{lineno} in {func}: {idiom}"
            for lineno, idiom, func in violations
        )
        pytest.fail(
            f"responder.py contains {len(violations)} raw URL "
            f"dict-style access(es):\n{rendered}\n\n"
            f"After sub-step 4 migration, all URL reads must go "
            f"through SanitizedResponderView projected items "
            f"(.url attribute on SanitizedURL). Migrate the call site "
            f"to read from view, or — if the 'url' field is a non-URL "
            f"identifier — rename it."
        )


def test_tiered_view_builder_has_no_raw_url_dict_accesses():
    """AR-9.feat.coach-tiers step 9 — extension of the static guard.

    The new tier projection layer in `url_views.py` consumes a frozen
    `TieredEvidence` and emits frozen `Prompt*` records with
    `SanitizedURL` (not str / not dict). The 10 functions in
    `_TIERED_BUILDER_NAMES` must NEVER reach into a raw dict for a
    URL. Future regressions where someone changes the builder to take
    a raw dict — and reaches in with `.get("url")` or `["url"]` — must
    fail here, just as they would in responder.py.

    Other functions in url_views.py legitimately read raw URLs (they
    drive the original validation pipeline); this check ONLY scopes to
    the tier-projection functions, which never touch raw dicts.
    """
    source = _URL_VIEWS_PATH.read_text(encoding="utf-8")
    all_violations = _collect_url_dict_accesses(source)
    in_tier_builder = [
        v for v in all_violations if v[2] in _TIERED_BUILDER_NAMES
    ]
    if in_tier_builder:
        rendered = "\n".join(
            f"  url_views.py:{lineno} in {func}: {idiom}"
            for lineno, idiom, func in in_tier_builder
        )
        pytest.fail(
            f"Tier projection layer contains "
            f"{len(in_tier_builder)} raw URL dict-style access(es):\n"
            f"{rendered}\n\nThe builder consumes TieredEvidence "
            f"(frozen dataclasses), not dicts. URLs flow as Validated "
            f"→ SanitizedURL via `_project_validated_to_sanitized`."
        )


def test_audit_helper_detects_seeded_violations():
    """Self-test: the audit helper actually flags violations when given
    source that contains them. Without this, a bug in the visitor
    could silently let real violations slip through.
    """
    seeded = (
        "def consumer(r):\n"
        "    a = r.get('url')\n"            # .get("url")
        "    b = r['url']\n"                # ["url"]
        "    return a, b\n"
    )
    violations = _collect_url_dict_accesses(seeded)
    idioms = {v[1] for v in violations}
    assert '.get("url")' in idioms
    assert '["url"]' in idioms
    assert len(violations) == 2
    # Both in the consumer function
    assert all(v[2] == "consumer" for v in violations)


def test_audit_helper_ignores_other_keys():
    """Confirm the helper only flags the literal "url" key — other
    string keys are not mistakenly flagged.
    """
    other_keys = (
        "def consumer(r):\n"
        "    a = r.get('title')\n"
        "    b = r['employer']\n"
        "    c = r.get('url_count')\n"     # similar string, not literal "url"
        "    return a, b, c\n"
    )
    violations = _collect_url_dict_accesses(other_keys)
    assert violations == []


def test_audit_helper_ignores_attribute_url():
    """The third historical idiom — `.url` attribute access — is NOT
    flagged because it's the legitimate way consumers read SanitizedURL
    from projected view items.
    """
    attribute_access = (
        "def consumer(view):\n"
        "    for r in view.fallback_results:\n"
        "        if r.url is not None:\n"
        "            yield r.url.raw\n"
    )
    violations = _collect_url_dict_accesses(attribute_access)
    assert violations == []
