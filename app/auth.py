"""Token-based authentication with RBAC (admin / user roles).

Token store is in-memory (does not survive restarts).
Each token entry holds: (expires_at, username, role).
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Tuple

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.user_store import UserRole, verify_user_password

_security = HTTPBearer(auto_error=False)
# token -> (expires_at, username, role)
_active_tokens: Dict[str, Tuple[datetime, str, UserRole]] = {}
TOKEN_TTL_HOURS = 24


@dataclass
class UserSession:
    """Carries the authenticated user's identity through request handlers."""
    token: str
    username: str
    role: UserRole


def create_token(username: str, password: str) -> UserSession:
    """Validate credentials against the user store and return a new session."""
    user = verify_user_password(username, password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง",
        )
    token = secrets.token_urlsafe(32)
    _active_tokens[token] = (
        datetime.utcnow() + timedelta(hours=TOKEN_TTL_HOURS),
        user.username,
        user.role,
    )
    return UserSession(token=token, username=user.username, role=user.role)


def revoke_token(token: str) -> None:
    _active_tokens.pop(token, None)


def require_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(_security),
) -> UserSession:
    """FastAPI dependency — raises 401 when token is missing or expired."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="กรุณาเข้าสู่ระบบก่อนใช้งาน",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = credentials.credentials
    entry = _active_tokens.get(token)
    if not entry or datetime.utcnow() > entry[0]:
        _active_tokens.pop(token, None)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session หมดอายุ กรุณาเข้าสู่ระบบอีกครั้ง",
            headers={"WWW-Authenticate": "Bearer"},
        )
    expires, username, role = entry
    # Sliding window — extend TTL on each authenticated request
    _active_tokens[token] = (
        datetime.utcnow() + timedelta(hours=TOKEN_TTL_HOURS),
        username,
        role,
    )
    return UserSession(token=token, username=username, role=role)


def require_admin(session: UserSession = Depends(require_auth)) -> UserSession:
    """FastAPI dependency — raises 403 when the user is not an admin."""
    if session.role != UserRole.admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="คุณไม่มีสิทธิ์เข้าถึงส่วนนี้ (ต้องการสิทธิ์ Admin)",
        )
    return session
