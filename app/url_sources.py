"""Persistent storage for website URL source configurations — backed by SQLite.

Previously stored in ``./url_sources.json``; data is automatically migrated
to the ``url_sources`` table in ``chat.db`` on first startup.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel

from app.chat_store import _conn


class UrlSource(BaseModel):
    id: str
    name: str
    url: str
    description: str = ""
    enabled: bool = True
    crawl_depth: int = 0  # 0 = single page, 1 = follow internal links on the page
    created_at: str
    last_indexed_at: Optional[str] = None


class UrlSourceCreate(BaseModel):
    name: str
    url: str
    description: str = ""
    enabled: bool = True
    crawl_depth: int = 0


class UrlSourceUpdate(BaseModel):
    name: Optional[str] = None
    url: Optional[str] = None
    description: Optional[str] = None
    enabled: Optional[bool] = None
    crawl_depth: Optional[int] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SELECT = (
    "SELECT id, name, url, description, enabled, crawl_depth, created_at, last_indexed_at "
    "FROM url_sources"
)


def _row_to_source(row) -> UrlSource:
    return UrlSource(
        id=row["id"],
        name=row["name"],
        url=row["url"],
        description=row["description"],
        enabled=bool(row["enabled"]),
        crawl_depth=row["crawl_depth"],
        created_at=row["created_at"],
        last_indexed_at=row["last_indexed_at"],
    )


# ---------------------------------------------------------------------------
# Public CRUD API
# ---------------------------------------------------------------------------

def list_url_sources() -> List[UrlSource]:
    with _conn() as con:
        rows = con.execute(f"{_SELECT} ORDER BY created_at").fetchall()
    return [_row_to_source(r) for r in rows]


def get_url_source(source_id: str) -> Optional[UrlSource]:
    with _conn() as con:
        row = con.execute(f"{_SELECT} WHERE id=?", (source_id,)).fetchone()
    return _row_to_source(row) if row else None


def add_url_source(data: UrlSourceCreate) -> UrlSource:
    source = UrlSource(
        id=str(uuid.uuid4()),
        name=data.name,
        url=data.url,
        description=data.description,
        enabled=data.enabled,
        crawl_depth=data.crawl_depth,
        created_at=datetime.utcnow().isoformat(),
    )
    with _conn() as con:
        con.execute(
            "INSERT INTO url_sources"
            "(id, name, url, description, enabled, crawl_depth, created_at, last_indexed_at)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (
                source.id, source.name, source.url, source.description,
                int(source.enabled), source.crawl_depth,
                source.created_at, source.last_indexed_at,
            ),
        )
    return source


def update_url_source(source_id: str, data: UrlSourceUpdate) -> Optional[UrlSource]:
    with _conn() as con:
        row = con.execute(f"{_SELECT} WHERE id=?", (source_id,)).fetchone()
        if not row:
            return None
        current = _row_to_source(row)
        patch = {k: v for k, v in data.model_dump().items() if v is not None}
        updated = current.model_copy(update=patch)
        con.execute(
            "UPDATE url_sources"
            " SET name=?, url=?, description=?, enabled=?, crawl_depth=?"
            " WHERE id=?",
            (
                updated.name, updated.url, updated.description,
                int(updated.enabled), updated.crawl_depth, source_id,
            ),
        )
    return updated


def delete_url_source(source_id: str) -> bool:
    with _conn() as con:
        cur = con.execute("DELETE FROM url_sources WHERE id=?", (source_id,))
    return cur.rowcount > 0


def mark_url_indexed(source_id: str) -> None:
    """Update last_indexed_at timestamp for a URL source."""
    with _conn() as con:
        con.execute(
            "UPDATE url_sources SET last_indexed_at=? WHERE id=?",
            (datetime.utcnow().isoformat(), source_id),
        )
