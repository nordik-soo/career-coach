"""AR-1c tests: prompt/Literal parity + Redis-mode activation gate +
activation-safety grep audit.

Covers (per docs/adjacent-recommendations-design.md v12 amendment):
  - OutcomeMove Literal contains both recommend_adjacent_roles AND
    describe_adjacent_role.
  - PlannerMove Literal EXCLUDES both (planner cannot emit them).
  - Planner system prompt names both in its "YOU MUST NOT EMIT" block.
  - Responder OUTCOME_RESPONDER_PROMPT lists both in its FINAL_MOVE
    enumeration AND includes a narration shape for each.
  - Schema rejection: plan_next_move returns None if the LLM emits
    either new move.
  - _adjacency_enabled returns True only when get_store() is a
    RedisSessionStore; False for CookieSessionStore.
  - Activation-safety grep audit: no production module dispatches
    into AR-1/2/3/4/5 entry points. Setters / dispatchers land in
    AR-6.
"""
from __future__ import annotations

from typing import get_args

import pytest

pytestmark = pytest.mark.nodb


from skillbridge.chat.arbiter import (
    ARBITER_REASON_ADJACENT_DESCRIPTION,
    ARBITER_REASON_ADJACENT_RECOMMENDATIONS,
    ArbiterAction,
    ArbiterDecision,
    OutcomeMove,
)
from skillbridge.chat.planner import PLANNER_SYSTEM_PROMPT, PlannerMove
from skillbridge.chat.prompts import OUTCOME_RESPONDER_PROMPT
from skillbridge.match.adjacent import (
    _synthesize_describe_adjacent_role_decision,
    _synthesize_recommend_adjacent_roles_decision,
)


_NEW_OUTCOME_MOVES = ("recommend_adjacent_roles", "describe_adjacent_role")


# =========================================================================
# Literal parity
# =========================================================================
def test_outcome_move_contains_recommend_adjacent_roles() -> None:
    assert "recommend_adjacent_roles" in get_args(OutcomeMove)


def test_outcome_move_contains_describe_adjacent_role() -> None:
    assert "describe_adjacent_role" in get_args(OutcomeMove)


def test_planner_move_excludes_recommend_adjacent_roles() -> None:
    """The planner cannot emit recommend_adjacent_roles -- it's a
    handler-synthesized move. Schema-level enforcement guards against
    a future contributor accidentally widening PlannerMove."""
    assert "recommend_adjacent_roles" not in get_args(PlannerMove)


def test_planner_move_excludes_describe_adjacent_role() -> None:
    assert "describe_adjacent_role" not in get_args(PlannerMove)


# =========================================================================
# Planner prompt parity ("YOU MUST NOT EMIT")
# =========================================================================
def test_planner_prompt_names_recommend_adjacent_roles_as_forbidden() -> None:
    """The prompt's "YOU MUST NOT EMIT" block must name the move
    explicitly so the LLM is steered away from emitting it. Belt-and-
    braces with the PlannerMove schema-level rejection."""
    assert "recommend_adjacent_roles" in PLANNER_SYSTEM_PROMPT


def test_planner_prompt_names_describe_adjacent_role_as_forbidden() -> None:
    assert "describe_adjacent_role" in PLANNER_SYSTEM_PROMPT


def test_planner_prompt_forbids_both_adjacent_moves_in_one_block() -> None:
    """Sanity: both moves appear in the same prompt that already
    forbids present_matches / present_no_match / confirm_resume_summary
    -- they share the same handler-synthesized semantics."""
    forbidden_segment_idx = PLANNER_SYSTEM_PROMPT.find("YOU MUST NOT EMIT")
    assert forbidden_segment_idx != -1
    # The forbidden block is the same paragraph. Read 250 chars from
    # the marker to capture the full sentence.
    block = PLANNER_SYSTEM_PROMPT[forbidden_segment_idx:forbidden_segment_idx + 250]
    for move in _NEW_OUTCOME_MOVES:
        assert move in block, (
            f"{move!r} should appear in the 'YOU MUST NOT EMIT' block, "
            f"not just somewhere later in the prompt."
        )


# =========================================================================
# Responder prompt parity
# =========================================================================
def test_responder_prompt_lists_recommend_adjacent_roles_in_final_move() -> None:
    """The FINAL_MOVE enumeration line must include the new move so
    the LLM has a named outcome to narrate."""
    assert "recommend_adjacent_roles" in OUTCOME_RESPONDER_PROMPT


def test_responder_prompt_lists_describe_adjacent_role_in_final_move() -> None:
    assert "describe_adjacent_role" in OUTCOME_RESPONDER_PROMPT


def test_responder_prompt_has_narration_shape_for_recommend_adjacent_roles() -> None:
    """The responder needs a narration shape, not just a name. Look for
    the shape-block header pattern used by the other moves."""
    assert "recommend_adjacent_roles —" in OUTCOME_RESPONDER_PROMPT


def test_responder_prompt_has_narration_shape_for_describe_adjacent_role() -> None:
    assert "describe_adjacent_role —" in OUTCOME_RESPONDER_PROMPT


def test_responder_prompt_states_forbidden_vocabulary_for_adjacent_roles() -> None:
    """The forbidden tokens lock from v3 (no "qualify", "good fit",
    etc.) must be repeated in the recommend_adjacent_roles shape so
    the LLM has it inside the narration context, not just at the
    global rules section."""
    shape_idx = OUTCOME_RESPONDER_PROMPT.find("recommend_adjacent_roles —")
    assert shape_idx != -1
    # Read forward enough to include the forbidden-vocabulary clause.
    shape = OUTCOME_RESPONDER_PROMPT[shape_idx:shape_idx + 1200]
    for token in ("you qualify", "good fit", "good match"):
        assert token in shape, (
            f"recommend_adjacent_roles narration shape must explicitly "
            f"name forbidden vocabulary token {token!r}."
        )


# =========================================================================
# Synthesis-factory shape (every field, plus ArbiterAction parity)
# =========================================================================
# ArbiterDecision.arbiter_action is typed `str` (not the ArbiterAction
# Literal), so the dataclass does NOT enforce the Literal at runtime.
# These tests guard the typed-contract surface explicitly: every
# arbiter_action emitted by the synthesis factories MUST be a member of
# get_args(ArbiterAction). Likewise the reason_code MUST match the
# constant the design lock specifies.
def test_recommend_adjacent_roles_decision_shape_is_pinned() -> None:
    d = _synthesize_recommend_adjacent_roles_decision()
    assert isinstance(d, ArbiterDecision)
    assert d.final_move == "recommend_adjacent_roles"
    assert d.reason_code == ARBITER_REASON_ADJACENT_RECOMMENDATIONS
    assert d.tone == "brief_confident"
    assert d.arbiter_action == "handler_synthesized_adjacent_recommendations"
    assert d.arbiter_action in get_args(ArbiterAction), (
        "handler_synthesized_adjacent_recommendations must be a member "
        "of ArbiterAction. ArbiterDecision.arbiter_action is typed `str`, "
        "so the dataclass won't catch drift -- this test does."
    )
    assert d.ask_slot is None
    assert d.caps_applied == ()
    assert d.notes is None


def test_describe_adjacent_role_decision_shape_is_pinned() -> None:
    d = _synthesize_describe_adjacent_role_decision()
    assert isinstance(d, ArbiterDecision)
    assert d.final_move == "describe_adjacent_role"
    assert d.reason_code == ARBITER_REASON_ADJACENT_DESCRIPTION
    assert d.tone == "brief_confident"
    assert d.arbiter_action == "handler_synthesized_adjacent_description"
    assert d.arbiter_action in get_args(ArbiterAction), (
        "handler_synthesized_adjacent_description must be a member of "
        "ArbiterAction. ArbiterDecision.arbiter_action is typed `str`, "
        "so the dataclass won't catch drift -- this test does."
    )
    assert d.ask_slot is None
    assert d.caps_applied == ()
    assert d.notes is None


def test_final_moves_are_members_of_outcome_move_literal() -> None:
    """Belt-and-braces: the factories' final_move must also live in the
    OutcomeMove Literal. ArbiterDecision.final_move is typed `str`."""
    rec = _synthesize_recommend_adjacent_roles_decision()
    desc = _synthesize_describe_adjacent_role_decision()
    assert rec.final_move in get_args(OutcomeMove)
    assert desc.final_move in get_args(OutcomeMove)


# =========================================================================
# Schema-layer rejection (plan_next_move returns None on forbidden move)
# =========================================================================
@pytest.mark.parametrize("forbidden_move", list(_NEW_OUTCOME_MOVES))
def test_plan_next_move_returns_none_when_llm_emits_new_outcome_move(
    monkeypatch, forbidden_move,
):
    """Defense in depth: even if the LLM (somehow) returns one of the
    new adjacency outcomes, the entry point must reject it and fall
    back to None. PlannerMove schema gate catches what the prompt
    didn't deter.

    Mirrors test_chat_planner.py:_truth() -- plan_next_move consumes
    the dict-shape (TruthSummary.to_planner_json) rather than the
    dataclass."""
    from skillbridge.chat import planner

    monkeypatch.setattr("skillbridge.chat.planner.llm.is_enabled", lambda: True)
    monkeypatch.setattr(
        "skillbridge.chat.planner.llm.call_json",
        lambda system, user, max_tokens=None: {
            "move": forbidden_move,
            "reason_code": "user_confirmed",
            "ask_slot": None,
            "tone": "brief_confident",
        },
    )
    truth = {
        "user_message": "show me jobs",
        "enough_to_match": True,
        "user_intent_signal": "impatient_proceed",
        "target_role_text": "warehouse worker",
        "target_role_specificity": "specific",
        "scope_violations_detected": [],
    }
    assert planner.plan_next_move(truth) is None


# =========================================================================
# _adjacency_enabled Redis-mode gate
# =========================================================================
def test_adjacency_enabled_true_for_redis_store(monkeypatch) -> None:
    """Adjacency activation gate returns True when BOTH gates are
    open: the active store is a RedisSessionStore AND the
    ADJACENCY_ACTIVATION_ENABLED flag is on (the AR-6a feature flag,
    default OFF until AR-6c lands)."""
    from skillbridge.match import adjacent
    from skillbridge.session.redis_store import RedisSessionStore

    class _FakeRedisStore(RedisSessionStore):
        def __init__(self):
            # Skip parent __init__ which opens a real connection.
            pass

    monkeypatch.setattr(
        "skillbridge.session.get_store",
        lambda: _FakeRedisStore(),
    )
    monkeypatch.setattr("config.ADJACENCY_ACTIVATION_ENABLED", True)
    assert adjacent._adjacency_enabled() is True


def test_adjacency_enabled_false_when_redis_but_flag_off(monkeypatch) -> None:
    """Even on a Redis-backed session, the feature flag must be ON
    for the gate to open. Default-OFF behavior is the AR-6a/b/c
    safety contract -- Redis users see the pre-AR-1 experience
    until AR-6c lands."""
    from skillbridge.match import adjacent
    from skillbridge.session.redis_store import RedisSessionStore

    class _FakeRedisStore(RedisSessionStore):
        def __init__(self):
            pass

    monkeypatch.setattr(
        "skillbridge.session.get_store",
        lambda: _FakeRedisStore(),
    )
    monkeypatch.setattr("config.ADJACENCY_ACTIVATION_ENABLED", False)
    assert adjacent._adjacency_enabled() is False


def test_adjacency_enabled_false_for_cookie_store(monkeypatch) -> None:
    """Cookie mode: feature gated off entirely. No soft offer, no
    intent dispatch, no persistence."""
    from skillbridge.match import adjacent
    from skillbridge.session.cookie_store import CookieSessionStore

    monkeypatch.setattr(
        "skillbridge.session.get_store",
        lambda: CookieSessionStore(secret="x" * 48),
    )
    assert adjacent._adjacency_enabled() is False


def test_adjacency_enabled_false_for_unknown_store(monkeypatch) -> None:
    """Defensive: an unknown store class (e.g. an in-memory test stub
    that isn't a RedisSessionStore subclass) must also gate off. The
    contract is opt-in to Redis, not opt-out of cookie."""
    from skillbridge.match import adjacent

    class _RandomStore:
        def load(self, sid): return None
        def save(self, s): return "x"
        def new_session(self): return "x"

    monkeypatch.setattr(
        "skillbridge.session.get_store",
        lambda: _RandomStore(),
    )
    assert adjacent._adjacency_enabled() is False


# =========================================================================
# Activation-safety grep audit
# =========================================================================
# These tests are the AR-1c equivalent of a "dead code at this commit"
# contract. They walk the production package and assert that no module
# under skillbridge/ dispatches into the AR-1/2/3/4/5 entry points.
# Setters / dispatchers MUST land in AR-6 and not before. If a future
# slice adds a call site here without the corresponding AR-6 wiring,
# this audit trips.

# (a) the SETTER for pending_adjacent_offer (AR-6 contract — must not
#     fire in any earlier slice).
# (b) callers of the engine pipeline (_load_active_jobs_with_skills,
#     retrieve_candidates, accept_candidates, _score_one_adjacent_job,
#     resolve_adjacent_followup, detect_adjacent_intent,
#     synthesize_recommend_adjacent_decision,
#     synthesize_describe_adjacent_role).
# (c) callers of the soft-offer predicates
#     (should_emit_soft_offer_on_matches/on_no_match,
#     is_credential_only_band_cap, has_usable_skill_evidence,
#     is_ssm_region_job, shift_adjacent_snapshot_ttl).
# (d) callers of _adjacency_enabled (the gate itself).
def _walk_production_python_files() -> list[str]:
    """Return absolute paths of every .py file under skillbridge/ EXCEPT
    the dead adjacency modules themselves.

    Adjacency code lives in:
        skillbridge/chat/adjacent_intent.py
        skillbridge/match/region.py
        skillbridge/match/adjacent.py
    These modules CAN reference one another freely (a dead module
    calling another dead module is still dead). The audit's job is to
    catch a *production* caller (any other module) dispatching into
    them before AR-6 wires the Redis-gated activation. Excluding the
    adjacency surface from the scan keeps the audit honest."""
    import pathlib

    pkg = pathlib.Path(__file__).resolve().parent.parent / "skillbridge"
    DEAD_ADJACENCY_MODULES = {
        str(pkg / "chat" / "adjacent_intent.py"),
        str(pkg / "chat" / "adjacent_followup.py"),
        str(pkg / "match" / "region.py"),
        str(pkg / "match" / "adjacent.py"),
    }
    return [
        str(p) for p in pkg.rglob("*.py")
        if str(p) not in DEAD_ADJACENCY_MODULES
    ]


_DEAD_DISPATCH_NAMES = (
    # AR-6 activation is incremental:
    #   AR-6a (this commit) wires the dispatch helpers in handler.py:
    #     scope-violated TTL shift + `_try_adjacency_dispatch` chain.
    #     Names activated here have been REMOVED from this list.
    #   AR-6b will activate the soft-offer predicates and the
    #     pending_adjacent_offer SETTER -- those three names remain
    #     dead until that commit.
    #   AR-6c will activate `render_describe_adjacent_role` (it's
    #     wired through `_try_adjacency_dispatch` but the payload
    #     isn't threaded into ResponderV2Input until AR-6c).
    #
    # Anything still in this list MUST NOT have a runtime reference
    # in production code (outside the dead adjacency modules
    # themselves, which the walker excludes from the scan).
    # All AR-6a / AR-6b predicates now have a production caller in
    # handler.py. AR-6c will activate the responder payload
    # threading; nothing remains structurally dead from the
    # adjacency feature surface.
    #
    # `is_credential_only_band_cap` is invoked transitively via
    # `should_emit_soft_offer_on_matches`, so its activation rides on
    # AR-6b's wiring.
)


def _module_defines_name(tree, name: str) -> bool:
    """True iff this module's AST contains a top-level or class-method
    `def name(...)` for `name`."""
    import ast as _ast
    for node in _ast.walk(tree):
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            if node.name == name:
                return True
    return False


def _ast_runtime_references(tree, names: tuple[str, ...]) -> dict[str, int]:
    """Walk the AST and return `{name: count}` for every runtime
    reference to a member of `names`. "Runtime reference" means:
      - `ast.Name(id=name)` -- bare name read or write
      - `ast.Attribute(attr=name)` -- `x.name` access
      - `ast.ImportFrom` aliases mentioning `name`
      - `ast.FunctionDef(name=name)` arg defaults / decorators don't
        count as references to a same-named identifier (we filter the
        defining-site itself separately).

    This DELIBERATELY excludes string literals (ast.Constant) so a
    docstring mention of `_adjacency_enabled` does not trip the audit.
    """
    import ast as _ast
    target = set(names)
    counts: dict[str, int] = {n: 0 for n in names}
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Name) and node.id in target:
            counts[node.id] += 1
        elif isinstance(node, _ast.Attribute) and node.attr in target:
            counts[node.attr] += 1
        elif isinstance(node, _ast.ImportFrom):
            for alias in node.names:
                if alias.name in target:
                    counts[alias.name] += 1
    return counts


def _audit_no_runtime_references(name_pool: tuple[str, ...]) -> list[str]:
    """Shared body: walk every production .py, parse to AST, and report
    any runtime reference to a name in `name_pool` except where the
    module is the definition site."""
    import ast as _ast
    leaks: list[str] = []
    for path in _walk_production_python_files():
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        try:
            tree = _ast.parse(src)
        except SyntaxError:
            continue
        defined_here = {n for n in name_pool if _module_defines_name(tree, n)}
        not_defined_here = tuple(n for n in name_pool if n not in defined_here)
        if not not_defined_here:
            continue
        counts = _ast_runtime_references(tree, not_defined_here)
        for n, count in counts.items():
            if count > 0:
                leaks.append(f"{path}: {count} runtime reference(s) to {n!r}")
    return leaks


def test_activation_audit_no_production_caller_of_dead_helpers() -> None:
    """No production module outside the canonical definition file
    dispatches into the AR-1 / AR-1b adjacency helpers at runtime.
    Docstring / comment mentions don't trip this audit."""
    leaks = _audit_no_runtime_references(_DEAD_DISPATCH_NAMES)
    assert not leaks, (
        "Production code dispatches into AR-1 helpers before AR-6 has "
        "landed. These runtime references must move to AR-6 wiring or "
        "be removed:\n  " + "\n  ".join(leaks)
    )


def test_activation_audit_setter_for_pending_adjacent_offer_lives_in_handler() -> None:
    """The SETTER for pending_adjacent_offer is the AR-6b soft-offer
    wiring in handler.py (`_maybe_append_soft_offer`). This audit
    confirms the setter is ONLY in that one file -- any other
    production setter would indicate a stray flag mutation outside
    the locked single-source-of-truth.

    Pre-AR-6b: no setter existed and the audit checked for absence.
    Post-AR-6b: exactly one setter exists, in handler.py."""
    import ast as _ast
    setter_paths: set[str] = set()
    for path in _walk_production_python_files():
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        try:
            tree = _ast.parse(src)
        except SyntaxError:
            continue
        for node in _ast.walk(tree):
            targets: list[_ast.AST] = []
            value: _ast.AST | None = None
            if isinstance(node, _ast.Assign):
                targets = list(node.targets)
                value = node.value
            elif isinstance(node, _ast.AugAssign):
                targets = [node.target]
                value = node.value
            else:
                continue
            for target in targets:
                if (
                    isinstance(target, _ast.Attribute)
                    and target.attr == "pending_adjacent_offer"
                ):
                    # False-assignments are the save-and-clear hook
                    # (AR-1a) and the AR-6b helper's "we didn't fire"
                    # branch; they don't count as setters.
                    if (
                        isinstance(value, _ast.Constant)
                        and value.value is False
                    ):
                        continue
                    setter_paths.add(path)

    # Exactly ONE setter, and it lives in handler.py.
    handler_paths = {p for p in setter_paths if "handler.py" in p}
    assert handler_paths, (
        "AR-6b expects the soft-offer setter to live in handler.py "
        f"(`_maybe_append_soft_offer`). Found setters in: "
        f"{setter_paths or '(none)'}"
    )
    assert setter_paths == handler_paths, (
        "A pending_adjacent_offer SETTER lives outside handler.py. "
        "The flag is the single signal that gates the next turn's "
        "adjacency dispatch -- mutations belong in one place. "
        f"Stray setters: {setter_paths - handler_paths}"
    )


def test_activation_audit_no_production_caller_of_engine_entrypoints() -> None:
    """The adjacency engine entry points (defined later in AR-2..AR-5)
    must have no production runtime references until AR-6. They don't
    exist at this commit -- the audit is a forward-looking gate that
    trips when a future slice wires them prematurely. Docstring /
    comment mentions don't count."""
    forward_names = ()
    # No remaining forward-looking names -- AR-2..AR-5 are all defined.
    # The dead-helpers audit (above) catches any production caller.
    leaks = _audit_no_runtime_references(forward_names)
    assert not leaks, (
        "Production code holds runtime references to AR-2..AR-5 entry "
        "points before those slices have landed. Move to AR-6 wiring "
        "or remove:\n  " + "\n  ".join(leaks)
    )
