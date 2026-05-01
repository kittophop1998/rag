"""Singleton OpenAI Python SDK clients.

Import ``get_client()`` for synchronous calls and ``get_async_client()`` for
async/streaming calls.  Both return a cached instance built from
``settings.openai_api_key`` so credentials are only read once.

Usage
-----
from app.openai_client import get_client, get_async_client

# Sync (e.g. background threads)
response = get_client().chat.completions.create(...)

# Async (FastAPI route / async generator)
stream = await get_async_client().chat.completions.create(..., stream=True)
"""

from __future__ import annotations

import openai

from app.config import settings

_sync_client: openai.OpenAI | None = None
_async_client: openai.AsyncOpenAI | None = None


def get_client() -> openai.OpenAI:
    """Return the process-global synchronous OpenAI client."""
    global _sync_client
    if _sync_client is None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured.")
        _sync_client = openai.OpenAI(api_key=settings.openai_api_key)
    return _sync_client


def get_async_client() -> openai.AsyncOpenAI:
    """Return the process-global asynchronous OpenAI client."""
    global _async_client
    if _async_client is None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured.")
        _async_client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
    return _async_client


def reset_clients() -> None:
    """Force re-creation of clients (useful after API key change at runtime)."""
    global _sync_client, _async_client
    _sync_client = None
    _async_client = None
