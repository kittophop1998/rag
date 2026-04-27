"""Persistent storage for external database connection configurations."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

_CONFIG_PATH = Path("./db_connections.json")


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


def _load() -> List[DatabaseConnection]:
    if not _CONFIG_PATH.exists():
        return []
    try:
        raw = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        return [DatabaseConnection(**item) for item in raw]
    except Exception:
        return []


def _save(connections: List[DatabaseConnection]) -> None:
    _CONFIG_PATH.write_text(
        json.dumps([c.model_dump() for c in connections], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def list_connections() -> List[DatabaseConnection]:
    return _load()


def get_connection(conn_id: str) -> Optional[DatabaseConnection]:
    return next((c for c in _load() if c.id == conn_id), None)


def add_connection(data: DatabaseConnectionCreate) -> DatabaseConnection:
    connections = _load()
    conn = DatabaseConnection(
        id=str(uuid.uuid4()),
        name=data.name,
        db_type=data.db_type,
        url=data.url,
        description=data.description,
        enabled=data.enabled,
        created_at=datetime.utcnow().isoformat(),
    )
    connections.append(conn)
    _save(connections)
    return conn


def update_connection(conn_id: str, data: DatabaseConnectionUpdate) -> Optional[DatabaseConnection]:
    connections = _load()
    for i, c in enumerate(connections):
        if c.id == conn_id:
            updated = c.model_copy(update={k: v for k, v in data.model_dump().items() if v is not None})
            connections[i] = updated
            _save(connections)
            return updated
    return None


def delete_connection(conn_id: str) -> bool:
    connections = _load()
    new_list = [c for c in connections if c.id != conn_id]
    if len(new_list) == len(connections):
        return False
    _save(new_list)
    return True
