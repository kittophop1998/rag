"""Simple token-based authentication for the RAG admin UI."""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta
from typing import Dict

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings

_security = HTTPBearer(auto_error=False)
_active_tokens: Dict[str, datetime] = {}
TOKEN_TTL_HOURS = 24


def create_token(username: str, password: str) -> str:
    """Validate credentials and return a new session token."""
    if username != settings.admin_username or password != settings.admin_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง",
        )
    token = secrets.token_urlsafe(32)
    _active_tokens[token] = datetime.utcnow() + timedelta(hours=TOKEN_TTL_HOURS)
    return token


def revoke_token(token: str) -> None:
    _active_tokens.pop(token, None)


def require_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(_security),
) -> str:
    """FastAPI dependency — raises 401 when token is missing or expired."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="กรุณาเข้าสู่ระบบก่อนใช้งาน",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = credentials.credentials
    expires = _active_tokens.get(token)
    if not expires or datetime.utcnow() > expires:
        _active_tokens.pop(token, None)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session หมดอายุ กรุณาเข้าสู่ระบบอีกครั้ง",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # Sliding window — extend TTL on each valid request
    _active_tokens[token] = datetime.utcnow() + timedelta(hours=TOKEN_TTL_HOURS)
    return token
