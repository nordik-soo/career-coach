"""The three deterministic gates for chat orchestration v2.

See docs/chat-orchestration-v2-design.md §4. A gate fires when the
current turn is **not a routing decision** -- it's either a state
transition (resume upload review) or has no input to route (empty
input, first-turn greeting). When a gate fires, the handler skips
the planner and arbiter entirely; the gate's output is the final
move.

Discipline: three gates. No more.

If a contributor finds themselves adding a fourth, ask hard whether
the new "gate" is actually a routing decision in disguise. The whole
architecture of v2 hinges on the planner being good enough to handle
the small cases naturally -- shortcuts re-introduce the rigidity v2
was built to escape.

Evaluation order matters (the dispatcher honors it strictly):

    1. empty input (no upload, no text)
    2. resume just uploaded this turn
    3. first-turn greeting -- *only if the message is greeting-like*

A first-turn resume upload routes to gate 2 (review the resume), not
gate 3 (greet) -- showing what we parsed is more useful than asking
the user to introduce themselves.

A first-turn message that already carries job intent (e.g. "I'm
looking for warehouse work", "truck and coach technician") must NOT
fire the greeting gate. The canned greeting asks "What kind of work
are you looking for?" -- emitting that when the user already said
what they want is a regression. Gate 3 therefore requires both
`message_count == 0` AND a normalized whitelist match against a small
set of bare greeting phrases (hi/hello/hey/good morning/can you help me/...).
Everything else on turn zero falls through to the planner.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GateDecision:
    """Output of a fired gate.

    The handler uses this to bypass the planner + arbiter pipeline. If
    `canned_response` is set, the responder LLM is also skipped and
    `canned_response` is returned verbatim -- saves an LLM call for
    trivial cases (empty input, first-turn greeting). Resume-upload
    gate leaves `canned_response=None` and routes through the
    existing RESUME_REVIEW responder flow because the response needs
    to quote actual parsed facts.
    """
    final_move: str            # one of the outcome moves from the v2 taxonomy
    gate_name: str             # which gate fired (for logs + transcript tests)
    canned_response: str | None = None
    ask_slot: str | None = None
    tone: str = "warm_supportive"


# =========================================================================
# Gate 1: empty / whitespace input
# =========================================================================
# Fires when the user submits an empty turn (no text + no file). The
# canned response is a soft re-prompt; the move tag is
# `ask_one_clarifying_question` with no specific slot, signalling
# "generic re-prompt." Transcript tests assert against the gate_name.
_EMPTY_INPUT_RESPONSE = (
    "Tell me a bit about the kind of work you're looking for "
    "and I'll find local matches."
)


def _is_empty_input_gate(
    *, user_message: str, uploaded_file: bool,
) -> GateDecision | None:
    """Fires when there's literally nothing to act on this turn.

    Returns None when:
      - the user sent any non-whitespace text, OR
      - the user uploaded a file (handled by gate 2)
    """
    if uploaded_file:
        return None
    if user_message and user_message.strip():
        return None
    return GateDecision(
        final_move="ask_one_clarifying_question",
        gate_name="empty_input",
        canned_response=_EMPTY_INPUT_RESPONSE,
        ask_slot=None,            # generic; not slot-specific
        tone="warm_supportive",
    )


# =========================================================================
# Gate 2: resume just uploaded
# =========================================================================
def _is_resume_upload_gate(
    *, uploaded_file: bool,
) -> GateDecision | None:
    """Fires when the current turn includes a file upload.

    Wins over the first-turn greeting gate by design: if a user opens
    the chat and immediately drops a resume, the right behavior is to
    show what we parsed -- not to greet them and ask them to introduce
    themselves while their resume sits there unread.

    `canned_response` is left None because the response has to quote
    the actual parsed facts. The handler routes through the existing
    RESUME_REVIEW responder flow, which is already grounded.
    """
    if not uploaded_file:
        return None
    return GateDecision(
        final_move="confirm_resume_summary",
        gate_name="resume_upload",
        canned_response=None,      # responder will narrate parsed facts
        ask_slot=None,
        tone="warm_supportive",
    )


# =========================================================================
# Gate 3: first-turn greeting (content-aware)
# =========================================================================
_FIRST_TURN_GREETING = (
    "Hey there! I'm SkillBridge SSM — I help folks in "
    "Sault Ste. Marie find local work that fits what they can do. "
    "What kind of work are you looking for?"
)

# Bare greeting phrases. Conservative whitelist by design: anything
# outside this set is treated as routable content and handed to the
# planner. Includes the obvious openers plus the "can you help me"
# family the user called out in review. Time-of-day greetings cover
# common variants; "morning"/"afternoon"/"evening" are accepted alone
# because that's how many people open chats. NO job-domain words --
# adding one here would re-create the bug this set is fixing.
_GREETING_PHRASES: frozenset[str] = frozenset({
    # bare greetings
    "hi", "hii", "hiii", "hello", "hey", "heyy", "heya",
    "yo", "hola", "howdy", "sup",
    # extended greetings
    "hi there", "hello there", "hey there",
    # time-of-day greetings
    "good morning", "good afternoon", "good evening", "good day",
    "morning", "afternoon", "evening",
    # politeness check-ins (no job content)
    "how are you", "how are you doing", "hows it going", "how's it going",
    # generic help-asks with no specifics (user-supplied examples)
    "help", "help me", "can you help", "can you help me",
    "i need help", "i need some help", "please help", "please help me",
})

# Trailing/leading punctuation to peel off before whitelist match.
# Keep tight -- we WANT exact matches, not fuzzy ones.
_GREETING_PUNCT = ".!?,;:"


def _normalize_for_greeting_match(message: str) -> str:
    """Reduce a first-turn message to its bare greeting form for an
    exact whitelist lookup.

    Steps: lowercase, trim whitespace, peel off trailing/leading
    punctuation (handles "Hi!", "hello.", "hey!!!"), collapse internal
    whitespace. Deliberately conservative -- if the message has
    substantive content beyond a greeting (e.g. "hi I need a job"),
    normalization leaves it intact and the whitelist lookup fails,
    routing it to the planner.
    """
    s = (message or "").strip().lower()
    while s and s[-1] in _GREETING_PUNCT:
        s = s[:-1]
    while s and s[0] in _GREETING_PUNCT:
        s = s[1:]
    return " ".join(s.split())


def _is_first_turn_greeting_gate(
    *, user_message: str, message_count: int,
) -> GateDecision | None:
    """Fires only when (a) it's the user's very first turn, AND
    (b) the message normalizes to a bare greeting phrase.

    The content guard was added in Slice 2 review: previously the gate
    fired on any non-empty first-turn message, which short-circuited
    legitimate first-turn job intent (e.g. "I'm looking for warehouse
    manager work") into the canned "What kind of work are you looking
    for?" reply. Now those messages fall through to the planner.

    The canned welcome mirrors the prior NEXT_ACTION_RESPONDER fallback
    text so the first impression on true greetings is unchanged from v1.
    """
    if message_count != 0:
        return None
    if _normalize_for_greeting_match(user_message) not in _GREETING_PHRASES:
        return None
    return GateDecision(
        final_move="acknowledge_and_continue",
        gate_name="first_turn_greeting",
        canned_response=_FIRST_TURN_GREETING,
        ask_slot=None,
        tone="warm_supportive",
    )


# =========================================================================
# Dispatcher -- enforces the explicit precedence
# =========================================================================
def evaluate_gates(
    *,
    user_message: str,
    uploaded_file: bool,
    message_count: int,
) -> GateDecision | None:
    """Returns the first gate that fires, or None to pass through to
    the planner. Evaluation order: empty -> resume -> first-turn.

    The order resolves compound turns (e.g. first turn AND resume
    upload AND empty text) without ambiguity:
      - Empty + upload         -> upload wins (gate 1 short-circuits on uploaded_file)
      - First-turn + upload    -> upload wins (gate 2 fires before gate 3)
      - Empty + first-turn     -> empty wins (gate 1 fires before gate 3)
    """
    decision = _is_empty_input_gate(
        user_message=user_message, uploaded_file=uploaded_file,
    )
    if decision is not None:
        return decision

    decision = _is_resume_upload_gate(uploaded_file=uploaded_file)
    if decision is not None:
        return decision

    decision = _is_first_turn_greeting_gate(
        user_message=user_message, message_count=message_count,
    )
    if decision is not None:
        return decision

    return None
