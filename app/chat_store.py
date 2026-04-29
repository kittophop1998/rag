"""SQLite-backed persistence for the entire application.

Tables
------
users           : authentication & authorization
chat_sessions   : per-user chat sessions
chat_messages   : messages within a session
db_connections  : external database connection configs (replaces db_connections.json)
url_sources     : web crawl source configs (replaces url_sources.json)

WAL mode is enabled so concurrent readers don't block the writer.
All public functions open and close their own connection, making them safe
to call from FastAPI async routes without a connection pool.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, List, Optional

from app.config import settings

logger = logging.getLogger(__name__)


def _get_db_path() -> Path:
    """Return the SQLite database path, ensuring the parent directory exists."""
    p = settings.data_dir / "chat.db"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


@contextmanager
def _conn() -> Generator[sqlite3.Connection, None, None]:
    con = sqlite3.connect(str(_get_db_path()), check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def init_db() -> None:
    """Create all tables and indexes. Safe to call multiple times (idempotent)."""
    with _conn() as con:
        con.executescript("""
            -- Users ------------------------------------------------------------
            CREATE TABLE IF NOT EXISTS users (
                id            TEXT PRIMARY KEY,
                username      TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role          TEXT NOT NULL DEFAULT 'user',
                enabled       INTEGER NOT NULL DEFAULT 1,
                display_name  TEXT NOT NULL DEFAULT '',
                created_at    TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);

            -- Chat sessions ----------------------------------------------------
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id         TEXT PRIMARY KEY,
                username   TEXT NOT NULL,
                title      TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_user
                ON chat_sessions(username, created_at DESC);

            -- Chat messages ----------------------------------------------------
            CREATE TABLE IF NOT EXISTS chat_messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL
                           REFERENCES chat_sessions(id) ON DELETE CASCADE,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                sources    TEXT,
                ts         INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON chat_messages(session_id, id);

            -- External DB connections ------------------------------------------
            CREATE TABLE IF NOT EXISTS db_connections (
                id            TEXT PRIMARY KEY,
                name          TEXT NOT NULL,
                db_type       TEXT NOT NULL,
                url           TEXT NOT NULL,
                description   TEXT NOT NULL DEFAULT '',
                enabled       INTEGER NOT NULL DEFAULT 1,
                vanna_trained INTEGER NOT NULL DEFAULT 0,
                created_at    TEXT NOT NULL
            );

            -- URL crawl sources -----------------------------------------------
            CREATE TABLE IF NOT EXISTS url_sources (
                id              TEXT PRIMARY KEY,
                name            TEXT NOT NULL,
                url             TEXT NOT NULL,
                description     TEXT NOT NULL DEFAULT '',
                enabled         INTEGER NOT NULL DEFAULT 1,
                crawl_depth     INTEGER NOT NULL DEFAULT 0,
                created_at      TEXT NOT NULL,
                last_indexed_at TEXT
            );

            -- Per-group index state (enabled / disabled + last indexed time) ------
            CREATE TABLE IF NOT EXISTS db_group_states (
                conn_id         TEXT NOT NULL,
                group_name      TEXT NOT NULL,
                enabled         INTEGER NOT NULL DEFAULT 1,
                last_indexed_at TEXT,
                PRIMARY KEY (conn_id, group_name)
            );
            CREATE INDEX IF NOT EXISTS idx_dgs_conn
                ON db_group_states(conn_id);
        """)


# ---------------------------------------------------------------------------
# JSON → SQLite one-time migration
# ---------------------------------------------------------------------------

def migrate_json_to_db() -> None:
    """Import legacy JSON config files into SQLite.

    Safe to call on every startup — only imports if the target table is empty.
    After a successful import the original JSON file is left in place so the
    admin can verify the data before removing it manually.
    """
    _migrate_db_connections()
    _migrate_url_sources()
    _migrate_vanna_trained()


def _migrate_db_connections() -> None:
    json_path = Path("./db_connections.json")
    if not json_path.exists():
        return
    with _conn() as con:
        count = con.execute("SELECT COUNT(*) FROM db_connections").fetchone()[0]
        if count > 0:
            return  # already populated — skip
        try:
            items = json.loads(json_path.read_text(encoding="utf-8"))
            for item in items:
                con.execute(
                    "INSERT OR IGNORE INTO db_connections"
                    "(id, name, db_type, url, description, enabled, vanna_trained, created_at)"
                    " VALUES (?,?,?,?,?,?,0,?)",
                    (
                        item["id"],
                        item["name"],
                        item["db_type"],
                        item["url"],
                        item.get("description", ""),
                        int(item.get("enabled", True)),
                        item["created_at"],
                    ),
                )
            logger.info("Migrated %d DB connection(s) from db_connections.json → SQLite.", len(items))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to migrate db_connections.json: %s", exc)


def _migrate_url_sources() -> None:
    json_path = Path("./url_sources.json")
    if not json_path.exists():
        return
    with _conn() as con:
        count = con.execute("SELECT COUNT(*) FROM url_sources").fetchone()[0]
        if count > 0:
            return
        try:
            items = json.loads(json_path.read_text(encoding="utf-8"))
            for item in items:
                con.execute(
                    "INSERT OR IGNORE INTO url_sources"
                    "(id, name, url, description, enabled, crawl_depth, created_at, last_indexed_at)"
                    " VALUES (?,?,?,?,?,?,?,?)",
                    (
                        item["id"],
                        item["name"],
                        item["url"],
                        item.get("description", ""),
                        int(item.get("enabled", True)),
                        int(item.get("crawl_depth", 0)),
                        item["created_at"],
                        item.get("last_indexed_at"),
                    ),
                )
            logger.info("Migrated %d URL source(s) from url_sources.json → SQLite.", len(items))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to migrate url_sources.json: %s", exc)


def _migrate_vanna_trained() -> None:
    json_path = Path("./vanna_trained.json")
    if not json_path.exists():
        return
    try:
        trained_ids: list = json.loads(json_path.read_text(encoding="utf-8"))
        with _conn() as con:
            for conn_id in trained_ids:
                con.execute(
                    "UPDATE db_connections SET vanna_trained=1 WHERE id=?",
                    (conn_id,),
                )
        logger.info("Migrated %d Vanna trained ID(s) from vanna_trained.json → SQLite.", len(trained_ids))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to migrate vanna_trained.json: %s", exc)


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

def create_session(username: str, title: str) -> dict:
    """Insert a new chat session and return its dict representation."""
    sid = uuid.uuid4().hex
    now = int(time.time() * 1000)
    with _conn() as con:
        con.execute(
            "INSERT INTO chat_sessions(id, username, title, created_at) VALUES (?,?,?,?)",
            (sid, username, title, now),
        )
    return {"id": sid, "title": title, "createdAt": now}


def list_sessions(username: str) -> List[dict]:
    """Return all sessions for a user, newest first."""
    with _conn() as con:
        rows = con.execute(
            "SELECT id, title, created_at FROM chat_sessions "
            "WHERE username=? ORDER BY created_at DESC",
            (username,),
        ).fetchall()
    return [{"id": r["id"], "title": r["title"], "createdAt": r["created_at"]} for r in rows]


def get_session(session_id: str, username: str) -> Optional[dict]:
    """Return session meta-data or None if it doesn't belong to the user."""
    with _conn() as con:
        row = con.execute(
            "SELECT id, title, created_at FROM chat_sessions WHERE id=? AND username=?",
            (session_id, username),
        ).fetchone()
    if not row:
        return None
    return {"id": row["id"], "title": row["title"], "createdAt": row["created_at"]}


def delete_session(session_id: str, username: str) -> bool:
    """Delete a session (and its messages via CASCADE). Returns False if not found."""
    with _conn() as con:
        cur = con.execute(
            "DELETE FROM chat_sessions WHERE id=? AND username=?",
            (session_id, username),
        )
    return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

def get_messages(session_id: str, username: str) -> Optional[List[dict]]:
    """Return messages for a session, or None if the session is not owned by the user."""
    with _conn() as con:
        owns = con.execute(
            "SELECT 1 FROM chat_sessions WHERE id=? AND username=?",
            (session_id, username),
        ).fetchone()
        if not owns:
            return None
        rows = con.execute(
            "SELECT role, content, sources, ts FROM chat_messages "
            "WHERE session_id=? ORDER BY id",
            (session_id,),
        ).fetchall()
    return [
        {
            "role": r["role"],
            "content": r["content"],
            "sources": json.loads(r["sources"]) if r["sources"] else [],
            "ts": r["ts"],
        }
        for r in rows
    ]


def add_message(
    session_id: str,
    username: str,
    role: str,
    content: str,
    sources: Optional[list] = None,
) -> Optional[dict]:
    """Append a message to a session.  Returns None if session not owned by user."""
    now = int(time.time() * 1000)
    sources_json = json.dumps(sources, ensure_ascii=False) if sources else None
    with _conn() as con:
        owns = con.execute(
            "SELECT 1 FROM chat_sessions WHERE id=? AND username=?",
            (session_id, username),
        ).fetchone()
        if not owns:
            return None
        con.execute(
            "INSERT INTO chat_messages(session_id, role, content, sources, ts) "
            "VALUES (?,?,?,?,?)",
            (session_id, role, content, sources_json, now),
        )
    return {"role": role, "content": content, "sources": sources or [], "ts": now}


# ---------------------------------------------------------------------------
# DB Group States — per-(conn_id, group_name) enabled flag
# ---------------------------------------------------------------------------

def get_group_states(conn_id: str) -> dict[str, dict]:
    """Return {group_name: {enabled, last_indexed_at}} for a connection."""
    with _conn() as con:
        rows = con.execute(
            "SELECT group_name, enabled, last_indexed_at FROM db_group_states WHERE conn_id=?",
            (conn_id,),
        ).fetchall()
    return {
        r["group_name"]: {
            "enabled": bool(r["enabled"]),
            "last_indexed_at": r["last_indexed_at"],
        }
        for r in rows
    }


def set_group_enabled(conn_id: str, group_name: str, enabled: bool) -> None:
    """Upsert the enabled flag for a group within a connection."""
    with _conn() as con:
        con.execute(
            """
            INSERT INTO db_group_states(conn_id, group_name, enabled)
            VALUES (?, ?, ?)
            ON CONFLICT(conn_id, group_name) DO UPDATE SET enabled=excluded.enabled
            """,
            (conn_id, group_name, int(enabled)),
        )


def touch_group_indexed(conn_id: str, group_name: str) -> None:
    """Update last_indexed_at timestamp and ensure the row exists."""
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with _conn() as con:
        con.execute(
            """
            INSERT INTO db_group_states(conn_id, group_name, enabled, last_indexed_at)
            VALUES (?, ?, 1, ?)
            ON CONFLICT(conn_id, group_name) DO UPDATE SET last_indexed_at=excluded.last_indexed_at
            """,
            (conn_id, group_name, now),
        )


def delete_group_states(conn_id: str) -> None:
    """Remove all group state rows for a connection (called on DB delete)."""
    with _conn() as con:
        con.execute("DELETE FROM db_group_states WHERE conn_id=?", (conn_id,))
