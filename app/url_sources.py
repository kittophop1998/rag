"""Persistent storage for website URL source configurations."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

_CONFIG_PATH = Path("./url_sources.json")


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


def _load() -> List[UrlSource]:
    if not _CONFIG_PATH.exists():
        return []
    try:
        raw = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        return [UrlSource(**item) for item in raw]
    except Exception:
        return []


def _save(sources: List[UrlSource]) -> None:
    _CONFIG_PATH.write_text(
        json.dumps([s.model_dump() for s in sources], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def list_url_sources() -> List[UrlSource]:
    return _load()


def get_url_source(source_id: str) -> Optional[UrlSource]:
    return next((s for s in _load() if s.id == source_id), None)


def add_url_source(data: UrlSourceCreate) -> UrlSource:
    sources = _load()
    source = UrlSource(
        id=str(uuid.uuid4()),
        name=data.name,
        url=data.url,
        description=data.description,
        enabled=data.enabled,
        crawl_depth=data.crawl_depth,
        created_at=datetime.utcnow().isoformat(),
    )
    sources.append(source)
    _save(sources)
    return source


def update_url_source(source_id: str, data: UrlSourceUpdate) -> Optional[UrlSource]:
    sources = _load()
    for i, s in enumerate(sources):
        if s.id == source_id:
            patch = {k: v for k, v in data.model_dump().items() if v is not None}
            updated = s.model_copy(update=patch)
            sources[i] = updated
            _save(sources)
            return updated
    return None


def delete_url_source(source_id: str) -> bool:
    sources = _load()
    new_list = [s for s in sources if s.id != source_id]
    if len(new_list) == len(sources):
        return False
    _save(new_list)
    return True


def mark_url_indexed(source_id: str) -> None:
    """Update last_indexed_at timestamp for a URL source."""
    sources = _load()
    for i, s in enumerate(sources):
        if s.id == source_id:
            sources[i] = s.model_copy(update={"last_indexed_at": datetime.utcnow().isoformat()})
            _save(sources)
            return
