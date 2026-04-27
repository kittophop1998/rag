"""User store with RBAC roles: admin and user.

Users are persisted in users.json (same approach as db_connections.json).
Passwords are hashed with PBKDF2-HMAC-SHA256 using Python's built-in hashlib.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

_USERS_PATH = Path("./users.json")


class UserRole(str, Enum):
    admin = "admin"
    user = "user"


class User(BaseModel):
    id: str
    username: str
    password_hash: str  # format: pbkdf2sha256:<salt>:<hex_digest>
    role: UserRole
    enabled: bool = True
    display_name: str = ""
    created_at: str


class UserCreate(BaseModel):
    username: str
    password: str
    role: UserRole = UserRole.user
    display_name: str = ""


class UserUpdate(BaseModel):
    password: Optional[str] = None
    role: Optional[UserRole] = None
    enabled: Optional[bool] = None
    display_name: Optional[str] = None


class UserResponse(BaseModel):
    id: str
    username: str
    role: UserRole
    enabled: bool
    display_name: str
    created_at: str


# ---------------------------------------------------------------------------
# Password helpers
# ---------------------------------------------------------------------------
_ITERATIONS = 260_000


def _hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), _ITERATIONS)
    return f"pbkdf2sha256:{salt}:{dk.hex()}"


def _verify_password(password: str, password_hash: str) -> bool:
    try:
        _, salt, dk_hex = password_hash.split(":")
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), _ITERATIONS)
        return secrets.compare_digest(dk.hex(), dk_hex)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# JSON persistence
# ---------------------------------------------------------------------------
def _load() -> List[User]:
    if not _USERS_PATH.exists():
        return []
    try:
        raw = json.loads(_USERS_PATH.read_text(encoding="utf-8"))
        return [User(**item) for item in raw]
    except Exception:
        return []


def _save(users: List[User]) -> None:
    _USERS_PATH.write_text(
        json.dumps([u.model_dump() for u in users], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def bootstrap_default_admin(username: str, password: str) -> None:
    """Ensure at least one admin user exists.  Called once at startup."""
    users = _load()
    if not any(u.role == UserRole.admin for u in users):
        admin = User(
            id=str(uuid.uuid4()),
            username=username,
            password_hash=_hash_password(password),
            role=UserRole.admin,
            enabled=True,
            display_name="Administrator",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        users.insert(0, admin)
        _save(users)


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------
def list_users() -> List[User]:
    return _load()


def get_user_by_id(user_id: str) -> Optional[User]:
    return next((u for u in _load() if u.id == user_id), None)


def get_user_by_username(username: str) -> Optional[User]:
    return next((u for u in _load() if u.username == username), None)


def verify_user_password(username: str, password: str) -> Optional[User]:
    """Return User if credentials are valid and the account is enabled."""
    user = get_user_by_username(username)
    if user and user.enabled and _verify_password(password, user.password_hash):
        return user
    return None


def create_user(data: UserCreate) -> User:
    users = _load()
    if any(u.username == data.username for u in users):
        raise ValueError(f"ชื่อผู้ใช้ '{data.username}' มีอยู่แล้วในระบบ")
    user = User(
        id=str(uuid.uuid4()),
        username=data.username,
        password_hash=_hash_password(data.password),
        role=data.role,
        enabled=True,
        display_name=data.display_name or data.username,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    users.append(user)
    _save(users)
    return user


def update_user(user_id: str, data: UserUpdate) -> Optional[User]:
    users = _load()
    for i, u in enumerate(users):
        if u.id == user_id:
            patch: dict = {}
            if data.password is not None:
                patch["password_hash"] = _hash_password(data.password)
            if data.role is not None:
                patch["role"] = data.role
            if data.enabled is not None:
                patch["enabled"] = data.enabled
            if data.display_name is not None:
                patch["display_name"] = data.display_name
            updated = u.model_copy(update=patch)
            users[i] = updated
            _save(users)
            return updated
    return None


def delete_user(user_id: str) -> bool:
    users = _load()
    new_list = [u for u in users if u.id != user_id]
    if len(new_list) == len(users):
        return False
    _save(new_list)
    return True
