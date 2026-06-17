"""Chat endpoint.

Two paths the route picks between, never mixes:
- Authenticated (bearer token present) -> handler.handle_authenticated
- Anonymous (no token) -> handler.handle_anonymous (pre-consent, no DB writes)

The route never accepts profile_id from the client body. profile_id is
ONLY resolved from the bearer token. This prevents the "anyone with a UUID
can read any profile" hole.

Sprint 1 adds a separate multipart endpoint at /v1/chat/messages/upload
that accepts an optional resume file alongside the message. The JSON
endpoint at /v1/chat/messages is unchanged, so existing clients keep
working until the frontend wires the paperclip widget (item 9).
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from skillbridge.auth import optional_profile
from skillbridge.chat import handler as chat_handler
from skillbridge.resume import MAX_RESUME_BYTES
from skillbridge.routes._envelope import envelope
from skillbridge.versions import CHAT_PROMPT_VERSION

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/chat", tags=["chat"])


class ChatMessageBody(BaseModel):
    message: str
    session_id: str | None = None  # ignored if Authorization header is valid


@router.post("/messages")
async def post_message(
    request: Request,
    body: ChatMessageBody,
    profile_id: str | None = Depends(optional_profile),
):
    if profile_id is not None:
        result = await asyncio.to_thread(
            chat_handler.handle_authenticated, body.message, profile_id
        )
    else:
        result = await asyncio.to_thread(
            chat_handler.handle_anonymous, body.message, body.session_id
        )
    return await envelope(request, result, engine_version=CHAT_PROMPT_VERSION)


@router.post("/messages/upload")
async def post_message_with_upload(
    request: Request,
    file: UploadFile = File(..., description="Resume file (PDF, DOCX, or TXT, ≤5MB)"),
    message: str = Form("", description="Optional chat message accompanying the upload"),
    session_id: str | None = Form(None),
    profile_id: str | None = Depends(optional_profile),
):
    """Multipart endpoint for chats that include a resume upload.

    Returns the same response shape as /v1/chat/messages plus a
    `resume_info` block describing what was parsed.

    Limits:
      - File size ≤ 5 MB (HTTP 413 if exceeded).
      - Authenticated path is currently rejected; upload is anonymous-
        only in this sprint. Authenticated resume re-upload is a future
        story since post-consent users already have a profile row to
        update against.
    """
    # Read once, into memory. 5 MB cap so streaming isn't worth the
    # complexity. If you raise the cap above ~50 MB later, switch to
    # streaming chunks into a temp file.
    file_bytes = await file.read()
    if len(file_bytes) > MAX_RESUME_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Resume file exceeds {MAX_RESUME_BYTES // (1024 * 1024)} MB limit.",
        )

    if profile_id is not None:
        # Authenticated upload is intentionally not wired in this sprint —
        # the data model for "replace existing resume on a consented
        # profile" needs its own design pass. Anonymous users (or
        # authenticated users who want to re-upload) can sign out first.
        raise HTTPException(
            status_code=409,
            detail=(
                "Resume upload is anonymous-only in this build. "
                "Sign out first, upload, then sign back in."
            ),
        )

    filename = file.filename or None

    result = await asyncio.to_thread(
        chat_handler.handle_anonymous,
        message,
        session_id,
        file_bytes=file_bytes,
        filename=filename,
    )
    return await envelope(request, result, engine_version=CHAT_PROMPT_VERSION)
