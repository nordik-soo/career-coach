"""Training-provider deny list — shared between policy paths.

Originally defined inline in `responder.py` for the existing
present_matches policy gate. Moved here as a leaf module so the new
tiered-matches policy gate (`coach_tiers_policy.py`) can import the
deny list without depending on its future consumer (`responder.py`).
Both policy paths now import these constants from the same source
of truth.

Module discipline:
  - No internal imports from other `skillbridge.chat` modules.
  - Both constants stay underscore-prefixed: they are internal-to-
    policy-paths and should not be re-exported as a public surface.
"""
from __future__ import annotations


# Post-Slice-9 grounding fix: a conservative deny-list of training
# provider names. The policy rejects the reply when one of these
# appears AND it is NOT present in this turn's TRAINING block. The
# allowlist (allowed providers when in TRAINING) is implicit -- if
# the registry put it in the prompt for this turn, the LLM may quote
# it; otherwise it may not.
#
# The list is NOT meant to be exhaustive named-entity detection. It
# catches the specific provider-shaped names Haiku has been seen to
# improvise (TAC) plus the canonical training providers we recognize
# from the allowlist (so that "Sault College" mentioned without TRAINING
# context still trips the check). Adding to this list is cheap; the
# point is defense-in-depth against the LLM supplementing TRAINING
# with outside-knowledge providers.
_KNOWN_TRAINING_PROVIDERS: tuple[str, ...] = (
    # Local SSM
    "sault college",
    "algoma university",
    "sault community career centre",
    "sccc",
    "northland adult learning centre",
    # Ontario credential authorities
    "skilled trades ontario",
    "drivetest",
    "serviceontario",
    "ministry of labour",
    "ministry of transportation",
    # National MOOCs / vendor certifications
    "microsoft learn",
    "aws skill builder",
    "google career certificates",
    "coursera",
    "edx",
    "comptia",
    "intuit",
    "quickbooks",
    # Health / safety
    "canadian red cross",
    "st. john ambulance",
    "ccohs",
    "traincan",
    "algoma public health",
    # Specific hallucination-prone organizations Haiku has been seen to
    # invent or supplement without grounding. Add new ones here when
    # observed in policy-fail logs.
    "transportation association of canada",
    "tac",
    "canadian welding bureau",
    "wsib",
    "ifsac",
    "national fire protection association",
    "nfpa",
)


# Canonical-full-name -> {set of accepted abbreviations / shorthand forms}.
# When a canonical name is grounded in this turn's TRAINING block, every
# entry in its set is ALSO treated as grounded. Without this, the LLM
# writing "SCCC" while TRAINING only carried "Sault Community Career
# Centre" got rejected as "ungrounded SCCC mention" and the deterministic
# fallback fired with stitched-together YAML prose ("For X for Y, here
# are the next steps:" — the unnatural reply 2026-06-08 surfaced).
#
# Keys MUST be lowercase to match the `grounded` set membership. Add
# new pairs as we observe the LLM naturally using a shorthand the
# policy rejected.
_PROVIDER_ABBREVIATIONS: dict[str, frozenset[str]] = {
    "sault community career centre": frozenset({"sccc"}),
    # Future additions go here as the live-test telemetry surfaces them:
    # "canadian centre for occupational health and safety": frozenset({"ccohs"}),
    # "skilled trades ontario": frozenset({"sto"}),
    # ...
}
