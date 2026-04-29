"""User store with RBAC roles: admin and user.

Users are now persisted in the shared SQLite database (chat.db).
Pydantic models and all public function signatures are unchanged from the
previous JSON-based version, so callers (auth.py, main.py) require no edits.

Migration
---------
On the first startup after this change, ``bootstrap_default_admin`` calls
``_migrate_from_json()`` which reads the old users.json (if it exists) and
imports every user into SQLite.  The JSON file is renamed to users.json.bak
after a successful migration so it is not re-processed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import sqlite3
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

from app.chat_store import _conn

logger = logging.getLogger(__name__)

_USERS_JSON = Path("./users.json")


# ---------------------------------------------------------------------------
# Models (unchanged)
# ---------------------------------------------------------------------------

class UserRole(str, Enum):
    admin = "admin"
    user = "user"


class User(BaseModel):
    id: str
    username: str
    password_hash: str  # pbkdf2sha256:<salt>:<hex_digest>
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
# Password helpers (unchanged)
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
# SQLite helpers
# ---------------------------------------------------------------------------

def _row_to_user(row: sqlite3.Row) -> User:
    return User(
        id=row["id"],
        username=row["username"],
        password_hash=row["password_hash"],
        role=UserRole(row["role"]),
        enabled=bool(row["enabled"]),
        display_name=row["display_name"] or "",
        created_at=row["created_at"],
    )


def _insert_user(con: sqlite3.Connection, user: User) -> None:
    con.execute(
        "INSERT INTO users(id, username, password_hash, role, enabled, display_name, created_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (
            user.id,
            user.username,
            user.password_hash,
            user.role.value,
            1 if user.enabled else 0,
            user.display_name,
            user.created_at,
        ),
    )


# ---------------------------------------------------------------------------
# One-time JSON → SQLite migration
# ---------------------------------------------------------------------------

def _migrate_from_json() -> None:
    """Import users.json into SQLite if the table is empty.

    Runs silently when there is nothing to migrate.  After a successful import
    the JSON file is renamed to users.json.bak so it cannot be re-read.
    """
    if not _USERS_JSON.exists():
        return

    with _conn() as con:
        count = con.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        if count > 0:
            return  # DB already populated; nothing to do

    try:
        raw: list = json.loads(_USERS_JSON.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read users.json for migration: %s", exc)
        return

    migrated = 0
    with _conn() as con:
        for item in raw:
            try:
                user = User(
                    id=item["id"],
                    username=item["username"],
                    password_hash=item["password_hash"],
                    role=UserRole(item.get("role", "user")),
                    enabled=bool(item.get("enabled", True)),
                    display_name=item.get("display_name", ""),
                    created_at=item.get("created_at", datetime.now(timezone.utc).isoformat()),
                )
                _insert_user(con, user)
                migrated += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping user during migration: %s", exc)

    if migrated:
        bak = _USERS_JSON.with_suffix(".json.bak")
        _USERS_JSON.rename(bak)
        logger.info(
            "Migrated %d user(s) from users.json → SQLite.  "
            "Original file renamed to %s.",
            migrated,
            bak.name,
        )


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_default_admin(username: str, password: str) -> None:
    """Ensure at least one admin exists.  Called once at startup.

    The migration from users.json runs here so it happens before the admin
    check — preserving any admin users that were in the old JSON file.
    """
    _migrate_from_json()

    with _conn() as con:
        has_admin = con.execute(
            "SELECT 1 FROM users WHERE role='admin' LIMIT 1"
        ).fetchone()

    if not has_admin:
        admin = User(
            id=str(uuid.uuid4()),
            username=username,
            password_hash=_hash_password(password),
            role=UserRole.admin,
            enabled=True,
            display_name="Administrator",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        with _conn() as con:
            con.execute(
                "INSERT OR IGNORE INTO users"
                "(id, username, password_hash, role, enabled, display_name, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (admin.id, admin.username, admin.password_hash, admin.role.value,
                 1, admin.display_name, admin.created_at),
            )


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def list_users() -> List[User]:
    with _conn() as con:
        rows = con.execute("SELECT * FROM users ORDER BY created_at").fetchall()
    return [_row_to_user(r) for r in rows]


def get_user_by_id(user_id: str) -> Optional[User]:
    with _conn() as con:
        row = con.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    return _row_to_user(row) if row else None


def get_user_by_username(username: str) -> Optional[User]:
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM users WHERE username=?", (username,)
        ).fetchone()
    return _row_to_user(row) if row else None


def verify_user_password(username: str, password: str) -> Optional[User]:
    """Return User if credentials are valid and account is enabled."""
    user = get_user_by_username(username)
    if user and user.enabled and _verify_password(password, user.password_hash):
        return user
    return None


def create_user(data: UserCreate) -> User:
    user = User(
        id=str(uuid.uuid4()),
        username=data.username,
        password_hash=_hash_password(data.password),
        role=data.role,
        enabled=True,
        display_name=data.display_name or data.username,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    try:
        with _conn() as con:
            _insert_user(con, user)
    except sqlite3.IntegrityError as exc:
        raise ValueError(f"ชื่อผู้ใช้ '{data.username}' มีอยู่แล้วในระบบ") from exc
    return user


def update_user(user_id: str, data: UserUpdate) -> Optional[User]:
    fields: dict = {}
    if data.password is not None:
        fields["password_hash"] = _hash_password(data.password)
    if data.role is not None:
        fields["role"] = data.role.value
    if data.enabled is not None:
        fields["enabled"] = 1 if data.enabled else 0
    if data.display_name is not None:
        fields["display_name"] = data.display_name

    if not fields:
        return get_user_by_id(user_id)

    set_clause = ", ".join(f"{k}=?" for k in fields)
    with _conn() as con:
        cur = con.execute(
            f"UPDATE users SET {set_clause} WHERE id=?",
            (*fields.values(), user_id),
        )
        if cur.rowcount == 0:
            return None

    return get_user_by_id(user_id)


def delete_user(user_id: str) -> bool:
    with _conn() as con:
        cur = con.execute("DELETE FROM users WHERE id=?", (user_id,))
    return cur.rowcount > 0
