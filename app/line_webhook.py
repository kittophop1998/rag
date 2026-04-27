"""
LINE Messaging API webhook.

Flow:
1. Verify the ``X-Line-Signature`` header against the raw request body
   using the channel secret (HMAC-SHA256).
2. For every text message event, run the question through the RAG engine.
3. Reply via LINE Messaging API ``/v2/bot/message/reply``.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
from typing import List

import httpx
from fastapi import APIRouter, Header, HTTPException, Request

from app.config import settings
from app.rag import rag_engine

logger = logging.getLogger(__name__)

router = APIRouter()

LINE_REPLY_URL = "https://api.line.me/v2/bot/message/reply"
MAX_LINE_TEXT = 4900  # LINE limit is 5000 chars per text message; leave headroom


# ---------------------------------------------------------------------------
# Signature verification
# ---------------------------------------------------------------------------
def _verify_signature(channel_secret: str, body: bytes, signature: str) -> bool:
    """Constant-time HMAC-SHA256 verification of the LINE webhook payload."""
    if not channel_secret or not signature:
        return False
    digest = hmac.new(channel_secret.encode("utf-8"), body, hashlib.sha256).digest()
    expected = base64.b64encode(digest).decode("utf-8")
    return hmac.compare_digest(expected, signature)


# ---------------------------------------------------------------------------
# LINE reply
# ---------------------------------------------------------------------------
async def _reply_text(reply_token: str, texts: List[str]) -> None:
    """Send 1..5 text messages back to the user via the reply API."""
    if not settings.line_channel_access_token:
        logger.warning("LINE_CHANNEL_ACCESS_TOKEN not configured, skipping reply.")
        return

    messages = [{"type": "text", "text": t[:MAX_LINE_TEXT]} for t in texts[:5]]
    payload = {"replyToken": reply_token, "messages": messages}
    headers = {
        "Authorization": f"Bearer {settings.line_channel_access_token}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(LINE_REPLY_URL, json=payload, headers=headers)
        if resp.status_code >= 300:
            logger.error("LINE reply failed: %s %s", resp.status_code, resp.text)


# ---------------------------------------------------------------------------
# Webhook endpoint
# ---------------------------------------------------------------------------
@router.post("/webhook")
async def line_webhook(
    request: Request,
    x_line_signature: str = Header(default=""),
):
    body = await request.body()

    if not _verify_signature(settings.line_channel_secret, body, x_line_signature):
        logger.warning("Invalid LINE signature.")
        raise HTTPException(status_code=400, detail="Invalid signature")

    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    events = payload.get("events", []) or []
    for event in events:
        if event.get("type") != "message":
            continue
        message = event.get("message", {}) or {}
        if message.get("type") != "text":
            continue

        user_text: str = message.get("text", "")
        reply_token: str = event.get("replyToken", "")
        if not reply_token:
            continue

        logger.info("LINE question: %s", user_text)
        try:
            result = rag_engine.ask(user_text)
            await _reply_text(reply_token, [result.answer])
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to handle LINE event: %s", exc)
            await _reply_text(
                reply_token,
                ["ขออภัยครับ ระบบมีปัญหาชั่วคราว กรุณาลองใหม่อีกครั้ง"],
            )

    return {"status": "ok"}
