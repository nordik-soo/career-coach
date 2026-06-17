"""Resume parsing and extraction package.

See docs/resume-design.md for the design rationale. This package is the
first half of the resume pipeline: bytes → plain text. The second half
(text → evidence-bound structured facts) lives in resume.extract.

Hard invariant: file bytes are never persisted. Bytes come in, text comes
out, bytes are discarded. The route layer reads multipart bytes into
memory, hands them to parse_resume(), and lets them go.
"""
from skillbridge.resume.derive import (
    compact_facts,
    derive_staged_slots,
    derive_with_suppressions,
)
from skillbridge.resume.extract import (
    ResumeExtractionResult,
    extract_resume_facts,
)
from skillbridge.resume.parse import (
    MAX_RESUME_BYTES,
    MIN_EXTRACTED_CHARS,
    ParseResult,
    parse_resume,
)

__all__ = [
    # parse.py
    "MAX_RESUME_BYTES",
    "MIN_EXTRACTED_CHARS",
    "ParseResult",
    "parse_resume",
    # extract.py
    "ResumeExtractionResult",
    "extract_resume_facts",
    # derive.py
    "compact_facts",
    "derive_staged_slots",
    "derive_with_suppressions",
]
