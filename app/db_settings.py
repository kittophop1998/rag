"""Persistent storage for external database connection configurations — backed by SQLite.

Previously stored in ``./db_connections.json``; data is automatically migrated
to the ``db_connections`` table in ``chat.db`` on first startup.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel

from app.chat_store import _conn


class DatabaseConnection(BaseModel):
    id: str
    name: str
    db_type: str          # mysql | postgresql | mssql | mongodb | rest | other
    url: str
    description: str = ""
    enabled: bool = True
    created_at: str


class DatabaseConnectionCreate(BaseModel):
    name: str
    db_type: str
    url: str
    description: str = ""
    enabled: bool = True


class DatabaseConnectionUpdate(BaseModel):
    name: Optional[str] = None
    db_type: Optional[str] = None
    url: Optional[str] = None
    description: Optional[str] = None
    enabled: Optional[bool] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SELECT = (
    "SELECT id, name, db_type, url, description, enabled, created_at "
    "FROM db_connections"
)


def _row_to_conn(row) -> DatabaseConnection:
    return DatabaseConnection(
        id=row["id"],
        name=row["name"],
        db_type=row["db_type"],
        url=row["url"],
        description=row["description"],
        enabled=bool(row["enabled"]),
        created_at=row["created_at"],
    )


# ---------------------------------------------------------------------------
# Public CRUD API
# ---------------------------------------------------------------------------

def list_connections() -> List[DatabaseConnection]:
    with _conn() as con:
        rows = con.execute(
            f"{_SELECT} ORDER BY created_at"
        ).fetchall()
    return [_row_to_conn(r) for r in rows]


def get_connection(conn_id: str) -> Optional[DatabaseConnection]:
    with _conn() as con:
        row = con.execute(
            f"{_SELECT} WHERE id=?", (conn_id,)
        ).fetchone()
    return _row_to_conn(row) if row else None


def add_connection(data: DatabaseConnectionCreate) -> DatabaseConnection:
    conn = DatabaseConnection(
        id=str(uuid.uuid4()),
        name=data.name,
        db_type=data.db_type,
        url=data.url,
        description=data.description,
        enabled=data.enabled,
        created_at=datetime.utcnow().isoformat(),
    )
    with _conn() as con:
        con.execute(
            "INSERT INTO db_connections"
            "(id, name, db_type, url, description, enabled, vanna_trained, created_at)"
            " VALUES (?,?,?,?,?,?,0,?)",
            (
                conn.id, conn.name, conn.db_type, conn.url,
                conn.description, int(conn.enabled), conn.created_at,
            ),
        )
    return conn


def update_connection(
    conn_id: str, data: DatabaseConnectionUpdate
) -> Optional[DatabaseConnection]:
    with _conn() as con:
        row = con.execute(f"{_SELECT} WHERE id=?", (conn_id,)).fetchone()
        if not row:
            return None
        current = _row_to_conn(row)
        patch = {k: v for k, v in data.model_dump().items() if v is not None}
        updated = current.model_copy(update=patch)
        con.execute(
            "UPDATE db_connections"
            " SET name=?, db_type=?, url=?, description=?, enabled=?"
            " WHERE id=?",
            (
                updated.name, updated.db_type, updated.url,
                updated.description, int(updated.enabled), conn_id,
            ),
        )
    return updated


def delete_connection(conn_id: str) -> bool:
    with _conn() as con:
        cur = con.execute("DELETE FROM db_connections WHERE id=?", (conn_id,))
    return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Vanna training state
# ---------------------------------------------------------------------------

def is_vanna_trained(conn_id: str) -> bool:
    """Return True if the connection has been schema-trained for Vanna.ai."""
    with _conn() as con:
        row = con.execute(
            "SELECT vanna_trained FROM db_connections WHERE id=?", (conn_id,)
        ).fetchone()
    return bool(row["vanna_trained"]) if row else False


def mark_vanna_trained(conn_id: str) -> None:
    """Mark a DB connection as schema-trained for Vanna.ai."""
    with _conn() as con:
        con.execute(
            "UPDATE db_connections SET vanna_trained=1 WHERE id=?", (conn_id,)
        )


def get_trained_connection_ids() -> List[str]:
    """Return IDs of all connections that have been Vanna schema-trained."""
    with _conn() as con:
        rows = con.execute(
            "SELECT id FROM db_connections WHERE vanna_trained=1"
        ).fetchall()
    return [r["id"] for r in rows]
