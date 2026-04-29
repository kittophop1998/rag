"""DB Auto-Index pipeline.

Scans all tables from a connected database, asks OpenAI to group them by
business domain, then embeds and stores the data in a per-connection
ChromaDB collection so the RAG engine can search across DB content.

Public API
----------
index_database(conn_id, db_type, db_url, conn_name)
    Launch indexing in a background daemon thread.  Returns immediately.
get_index_status(conn_id) → dict
    Return current status dict: {status, message, progress?, tables?, chunks?, groups?}
delete_index(conn_id) → bool
    Remove the ChromaDB directory for this connection.
load_db_store(conn_id) → Chroma | None
    Load the persisted Chroma collection (None if not indexed yet).
list_indexed_conn_ids() → list[str]
    List conn_ids that have a completed ChromaDB index on disk.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import settings
from app.db_inspector import _SQL_TYPES, _normalize_sql_url, get_schema_description

logger = logging.getLogger(__name__)

MAX_ROWS_PER_TABLE = 250
ROWS_PER_CHUNK = 25
STANDARD_GROUPS = ["ยอดขาย", "สินค้า", "ลูกค้า", "สินทรัพย์", "ผู้ใช้และแผนก", "ทั่วไป"]

_SYSTEM_TABLE_PATTERNS = (
    r"^kysely_migration(_lock)?$",
    r"^alembic_version$",
    r"^django_migrations$",
    r"^flyway_schema_history$",
    r"^schema_migrations$",
    r"^sqlite_",
)
_SYSTEM_TABLE_REGEXES = [re.compile(p, re.IGNORECASE) for p in _SYSTEM_TABLE_PATTERNS]

_GROUP_ALIASES: dict[str, tuple[str, ...]] = {
    "ยอดขาย": ("ยอดขาย", "การขาย", "คำสั่งซื้อ", "ขาย", "sales", "orders", "invoice"),
    "สินค้า": ("สินค้า", "หมวดสินค้า", "สต็อก", "คลังสินค้า", "products", "product", "items", "catalog"),
    "ลูกค้า": ("ลูกค้า", "สมาชิก", "ผู้ติดต่อ", "customers", "customer", "crm"),
    "สินทรัพย์": ("สินทรัพย์", "อุปกรณ์", "asset", "assets", "equipment"),
    "ผู้ใช้และแผนก": ("ผู้ใช้", "แผนก", "พนักงาน", "สิทธิ์", "users", "departments", "roles", "staff"),
}

_status_lock = threading.Lock()
_index_status: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def _chroma_dir(conn_id: str) -> Path:
    return settings.chroma_base_dir / "db" / conn_id


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------
def _meta_file(conn_id: str) -> Path:
    return _chroma_dir(conn_id) / "meta.json"


def _load_meta(conn_id: str) -> dict | None:
    """Load persisted index metadata (groups, table_rows, etc.) from disk."""
    f = _meta_file(conn_id)
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cannot read meta.json for %s: %s", conn_id, exc)
        return None


def _save_meta(conn_id: str, meta: dict) -> None:
    """Persist index metadata next to the Chroma collection."""
    try:
        d = _chroma_dir(conn_id)
        d.mkdir(parents=True, exist_ok=True)
        _meta_file(conn_id).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cannot save meta.json for %s: %s", conn_id, exc)


def get_index_status(conn_id: str) -> dict:
    with _status_lock:
        mem = _index_status.get(conn_id)
    if mem:
        return mem.copy()
    if (_chroma_dir(conn_id) / "chroma.sqlite3").exists():
        meta = _load_meta(conn_id) or {}
        return {
            "status": "indexed",
            "message": meta.get("message", "พร้อมใช้งาน"),
            "tables": meta.get("tables"),
            "rows": meta.get("rows"),
            "chunks": meta.get("chunks"),
            "groups": meta.get("groups", []),
            "group_map": meta.get("group_map", {}),
            "table_rows": meta.get("table_rows", {}),
            "indexed_at": meta.get("indexed_at"),
        }
    return {"status": "none", "message": "ยังไม่ได้ Index"}


def _set_status(conn_id: str, **kwargs: Any) -> None:
    with _status_lock:
        _index_status[conn_id] = dict(kwargs)


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------
def _serialize_value(v: Any) -> str | None:
    """Return a string representation, or None to skip the field."""
    if v is None:
        return None
    if isinstance(v, bytes):
        return None  # skip binary blobs
    return str(v)


def _serialize_row(row: dict) -> dict:
    result = {}
    for k, v in row.items():
        s = _serialize_value(v)
        if s is not None:
            result[k] = s
    return result


def _fetch_sql_tables(url: str) -> dict[str, list[dict]]:
    """Return {table_name: [row_dict, ...]} for every SQL table."""
    from sqlalchemy import MetaData, create_engine, select  # type: ignore[import-untyped]

    engine = create_engine(_normalize_sql_url(url), pool_pre_ping=True)
    try:
        meta = MetaData()
        meta.reflect(bind=engine)
        result: dict[str, list[dict]] = {}
        with engine.connect() as conn:
            for table_name, table in meta.tables.items():
                try:
                    rows = conn.execute(select(table).limit(MAX_ROWS_PER_TABLE))
                    result[table_name] = [_serialize_row(dict(r._mapping)) for r in rows]
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Cannot read table %s: %s", table_name, exc)
                    result[table_name] = []
        return result
    finally:
        engine.dispose()


def _fetch_sql_selected_tables(url: str, table_names: list[str]) -> dict[str, list[dict]]:
    """Return rows for the requested SQL table names only."""
    from sqlalchemy import MetaData, create_engine, select  # type: ignore[import-untyped]

    selected = [t for t in dict.fromkeys(table_names) if t]
    if not selected:
        return {}

    engine = create_engine(_normalize_sql_url(url), pool_pre_ping=True)
    try:
        meta = MetaData()
        meta.reflect(bind=engine)

        result: dict[str, list[dict]] = {}
        with engine.connect() as conn:
            for table_name in selected:
                table = meta.tables.get(table_name)
                if table is None:
                    # Handle schema-qualified names reflected as "schema.table".
                    table = next(
                        (t for n, t in meta.tables.items() if n.endswith(f".{table_name}")),
                        None,
                    )
                if table is None:
                    logger.warning("Cannot find table '%s' while group indexing", table_name)
                    result[table_name] = []
                    continue
                try:
                    rows = conn.execute(select(table).limit(MAX_ROWS_PER_TABLE))
                    result[table_name] = [_serialize_row(dict(r._mapping)) for r in rows]
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Cannot read table %s: %s", table_name, exc)
                    result[table_name] = []
        return result
    finally:
        engine.dispose()


def _fetch_mongo_collections(url: str) -> dict[str, list[dict]]:
    """Return {collection_name: [doc_dict, ...]} for every MongoDB collection."""
    from pymongo import MongoClient  # type: ignore[import-untyped]

    from app.db_inspector import _extract_mongo_dbname

    db_name = _extract_mongo_dbname(url)
    client: MongoClient = MongoClient(url, serverSelectionTimeoutMS=5000)
    try:
        db = client[db_name]
        result: dict[str, list[dict]] = {}
        for coll_name in db.list_collection_names():
            docs = list(db[coll_name].find({}, limit=MAX_ROWS_PER_TABLE))
            result[coll_name] = [
                _serialize_row({k: v for k, v in doc.items() if k != "_id"})
                for doc in docs
            ]
        return result
    finally:
        client.close()


def _fetch_mongo_selected_collections(url: str, names: list[str]) -> dict[str, list[dict]]:
    """Return rows for the requested MongoDB collections only."""
    from pymongo import MongoClient  # type: ignore[import-untyped]

    from app.db_inspector import _extract_mongo_dbname

    selected = [n for n in dict.fromkeys(names) if n]
    if not selected:
        return {}

    db_name = _extract_mongo_dbname(url)
    client: MongoClient = MongoClient(url, serverSelectionTimeoutMS=5000)
    try:
        db = client[db_name]
        existing = set(db.list_collection_names())
        result: dict[str, list[dict]] = {}
        for coll_name in selected:
            if coll_name not in existing:
                logger.warning("Cannot find collection '%s' while group indexing", coll_name)
                result[coll_name] = []
                continue
            docs = list(db[coll_name].find({}, limit=MAX_ROWS_PER_TABLE))
            result[coll_name] = [
                _serialize_row({k: v for k, v in doc.items() if k != "_id"})
                for doc in docs
            ]
        return result
    finally:
        client.close()


def _is_system_table(table_name: str) -> bool:
    return any(rx.search(table_name) for rx in _SYSTEM_TABLE_REGEXES)


def _normalize_group_name(group_name: str, table_names: list[str]) -> str:
    """Map free-form LLM group names into a standard set used by the UI."""
    name = (group_name or "").strip()
    if not name:
        name = "ทั่วไป"

    if name in STANDARD_GROUPS:
        return name

    lowered = name.lower()
    for canonical, keywords in _GROUP_ALIASES.items():
        if any(k.lower() in lowered for k in keywords):
            return canonical

    table_blob = " ".join(table_names).lower()
    for canonical, keywords in _GROUP_ALIASES.items():
        if any(k.lower() in table_blob for k in keywords):
            return canonical

    return "ทั่วไป"


# ---------------------------------------------------------------------------
# OpenAI grouping
# ---------------------------------------------------------------------------
def _ask_openai_groups(schema: str, table_names: list[str]) -> dict[str, list[str]]:
    """Ask OpenAI to cluster tables by business domain.

    Returns {group_name: [table_name, ...]} or falls back to a single group.
    """
    llm = ChatOpenAI(
        model=settings.openai_chat_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )
    schema_preview = schema[:4_000]
    prompt = (
        "คุณคือผู้เชี่ยวชาญด้านฐานข้อมูล\n\n"
        f"Schema:\n{schema_preview}\n\n"
        f"ตาราง: {json.dumps(table_names, ensure_ascii=False)}\n\n"
        "จงจัดกลุ่มตารางเหล่านี้ตามโดเมนทางธุรกิจ\n"
        f"ต้องใช้ชื่อกลุ่มจากรายการนี้เท่านั้น: {json.dumps(STANDARD_GROUPS, ensure_ascii=False)}\n"
        'ถ้าไม่แน่ใจให้ใช้ "ทั่วไป"\n'
        "ตอบเป็น JSON object เท่านั้น (ไม่มี markdown) โดย key คือชื่อกลุ่ม value คือ array ชื่อตาราง\n"
        'ตัวอย่าง: {"ยอดขาย": ["orders", "order_lines"], "สินค้า": ["products"]}'
    )
    try:
        resp = llm.invoke(prompt)
        text = resp.content.strip()
        if "```" in text:
            parts = text.split("```")
            text = parts[1] if len(parts) > 1 else parts[0]
            if text.lower().startswith("json"):
                text = text[4:]
        groups: dict = json.loads(text.strip())
        if isinstance(groups, dict) and groups:
            return groups
    except Exception as exc:  # noqa: BLE001
        logger.warning("OpenAI grouping failed (%s) — using flat group", exc)
    return {"ทั่วไป": table_names}


# ---------------------------------------------------------------------------
# Document conversion
# ---------------------------------------------------------------------------
def _rows_to_docs(
    table_name: str,
    rows: list[dict],
    group_name: str,
    conn_name: str,
) -> list[Document]:
    """Convert table rows into LangChain Documents (batched by ROWS_PER_CHUNK)."""
    if not rows:
        return []
    docs: list[Document] = []
    for i in range(0, len(rows), ROWS_PER_CHUNK):
        batch = rows[i : i + ROWS_PER_CHUNK]
        header = f"[DB: {conn_name}] [กลุ่ม: {group_name}] [ตาราง: {table_name}]"
        lines = [header]
        for j, row in enumerate(batch, i + 1):
            row_str = " | ".join(f"{k}: {v}" for k, v in row.items())
            lines.append(f"  แถว {j}: {row_str}")
        docs.append(
            Document(
                page_content="\n".join(lines),
                metadata={
                    "source": f"db:{conn_name}/{table_name}",
                    "table": table_name,
                    "group": group_name,
                    "db_name": conn_name,
                    "type": "db_data",
                },
            )
        )
    return docs


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------
def delete_index(conn_id: str) -> bool:
    """Remove the ChromaDB directory for a connection.  Returns True if deleted."""
    d = _chroma_dir(conn_id)
    if d.exists():
        shutil.rmtree(d)
        with _status_lock:
            _index_status.pop(conn_id, None)
        return True
    with _status_lock:
        _index_status.pop(conn_id, None)
    return False


def load_db_store(conn_id: str) -> Chroma | None:
    """Load the persisted Chroma collection.  Returns None if not indexed."""
    d = _chroma_dir(conn_id)
    if not (d / "chroma.sqlite3").exists():
        return None
    return Chroma(
        persist_directory=str(d),
        embedding_function=OpenAIEmbeddings(
            model=settings.openai_embed_model,
            api_key=settings.openai_api_key,
        ),
        collection_name=f"db_{conn_id}",
    )


def list_indexed_conn_ids() -> list[str]:
    """Return conn_ids that have a completed index under chroma/db/."""
    db_root = settings.chroma_base_dir / "db"
    if not db_root.exists():
        return []
    return [
        d.name
        for d in db_root.iterdir()
        if d.is_dir() and (d / "chroma.sqlite3").exists()
    ]


# ---------------------------------------------------------------------------
# Main indexing pipeline (runs in background thread)
# ---------------------------------------------------------------------------
def _run_index(conn_id: str, db_type: str, db_url: str, conn_name: str) -> None:
    try:
        _set_status(conn_id, status="indexing",
                    message="กำลังอ่านข้อมูลจากฐานข้อมูล...", progress=5)

        t = db_type.lower()
        if t in _SQL_TYPES:
            table_data = _fetch_sql_tables(db_url)
        elif t == "mongodb":
            table_data = _fetch_mongo_collections(db_url)
        else:
            raise ValueError(f"ไม่รองรับ db_type '{db_type}'")

        removed_system_tables = sorted([tname for tname in table_data if _is_system_table(tname)])
        if removed_system_tables:
            table_data = {k: v for k, v in table_data.items() if k not in removed_system_tables}

        if not table_data:
            raise RuntimeError("ไม่พบตารางธุรกิจสำหรับทำ Index (system tables ถูกกรองออกทั้งหมด)")

        total_rows = sum(len(v) for v in table_data.values())
        _set_status(
            conn_id, status="indexing",
            message=f"พบ {len(table_data)} ตาราง / {total_rows} rows — OpenAI กำลังจัดกลุ่ม...",
            progress=30,
        )

        schema = get_schema_description(db_type, db_url)
        groups = _ask_openai_groups(schema, list(table_data.keys()))
        groups_norm: dict[str, list[str]] = {}
        for g_name, g_tables in groups.items():
            std_name = _normalize_group_name(g_name, g_tables)
            groups_norm.setdefault(std_name, []).extend(g_tables)

        table_to_group: dict[str, str] = {}
        for g_name, g_tables in groups_norm.items():
            for t_name in g_tables:
                table_to_group[t_name] = g_name

        _set_status(conn_id, status="indexing",
                    message="แปลงข้อมูลเป็น Documents...", progress=55)

        all_docs: list[Document] = []
        for table_name, rows in table_data.items():
            group = table_to_group.get(table_name, "ทั่วไป")
            all_docs.extend(_rows_to_docs(table_name, rows, group, conn_name))

        if not all_docs:
            raise RuntimeError("ไม่มีข้อมูลที่จะ Index (ทุกตารางว่างเปล่า)")

        _set_status(
            conn_id, status="indexing",
            message=f"Embedding {len(all_docs)} chunks ด้วย OpenAI...", progress=70,
        )

        chroma_dir = _chroma_dir(conn_id)
        tmp_dir = Path(str(chroma_dir) + "_tmp")

        # Write to a fresh tmp path so ChromaDB never reuses a cached connection
        # that pointed to the old (now-deleted) chroma.sqlite3 file, which would
        # cause SQLITE_READONLY_DBMOVED (code 1032).
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        Chroma.from_documents(
            all_docs,
            OpenAIEmbeddings(
                model=settings.openai_embed_model,
                api_key=settings.openai_api_key,
            ),
            persist_directory=str(tmp_dir),
            collection_name=f"db_{conn_id}",
        )

        # Atomic swap: replace old index only after the new write succeeds.
        # The old data remains queryable during the entire embedding phase.
        if chroma_dir.exists():
            shutil.rmtree(chroma_dir)
        shutil.move(str(tmp_dir), str(chroma_dir))

        # Build a clean {group_name: [tables]} mapping that only contains
        # tables we actually saw in the database (defends against LLM
        # hallucinations).  Tables not assigned to any group fall under "ทั่วไป".
        actual_table_set = set(table_data.keys())
        groups_clean: dict[str, list[str]] = {}
        assigned: set[str] = set()
        for g_name, g_tables in groups_norm.items():
            valid = [t for t in g_tables if t in actual_table_set]
            if valid:
                normalized_name = _normalize_group_name(g_name, valid)
                groups_clean.setdefault(normalized_name, [])
                groups_clean[normalized_name].extend(valid)
                assigned.update(valid)
        for g_name in list(groups_clean.keys()):
            groups_clean[g_name] = sorted(set(groups_clean[g_name]))
        unassigned = sorted(actual_table_set - assigned)
        if unassigned:
            groups_clean.setdefault("ทั่วไป", []).extend(unassigned)

        # Per-table row counts (handy for the UI)
        table_rows = {t: len(rs) for t, rs in table_data.items()}

        from datetime import datetime, timezone  # noqa: PLC0415

        message = (
            f"สำเร็จ: {len(table_data)} ตาราง, "
            f"{total_rows} rows, "
            f"{len(all_docs)} chunks, "
            f"{len(groups_clean)} กลุ่ม"
        )
        if removed_system_tables:
            message += f" (กรอง system tables {len(removed_system_tables)} ตาราง)"
        indexed_at = datetime.now(timezone.utc).isoformat()

        _set_status(
            conn_id,
            status="indexed",
            message=message,
            tables=len(table_data),
            rows=total_rows,
            chunks=len(all_docs),
            groups=list(groups_clean.keys()),
            group_map=groups_clean,
            table_rows=table_rows,
            indexed_at=indexed_at,
            progress=100,
        )

        _save_meta(conn_id, {
            "message": message,
            "tables": len(table_data),
            "rows": total_rows,
            "chunks": len(all_docs),
            "groups": list(groups_clean.keys()),
            "group_map": groups_clean,
            "table_rows": table_rows,
            "indexed_at": indexed_at,
        })
        logger.info(
            "DB index complete [%s]: %d tables, %d rows, %d chunks, %d groups",
            conn_name, len(table_data), total_rows, len(all_docs), len(groups_clean),
        )

    except Exception as exc:  # noqa: BLE001
        logger.exception("DB indexing failed [%s]: %s", conn_id, exc)
        _set_status(conn_id, status="error", message=f"Index ล้มเหลว: {exc}", progress=0)


def index_database(conn_id: str, db_type: str, db_url: str, conn_name: str) -> None:
    """Launch the indexing pipeline in a daemon thread.  Returns immediately.

    If an indexing job is already running for this connection, this call is
    a no-op (prevents accidental duplicate jobs).
    """
    with _status_lock:
        if _index_status.get(conn_id, {}).get("status") == "indexing":
            return
        _index_status[conn_id] = {"status": "indexing", "message": "เริ่มต้น...", "progress": 0}

    t = threading.Thread(
        target=_run_index,
        args=(conn_id, db_type, db_url, conn_name),
        daemon=True,
        name=f"db-index-{conn_id[:8]}",
    )
    t.start()


# ---------------------------------------------------------------------------
# Per-group re-index pipeline
# ---------------------------------------------------------------------------
_group_status_lock = threading.Lock()
_group_status: dict[str, dict] = {}   # key = f"{conn_id}|{group_name}"


def _gkey(conn_id: str, group_name: str) -> str:
    return f"{conn_id}|{group_name}"


def get_group_index_status(conn_id: str, group_name: str) -> dict:
    """Return {status, message, progress?} for a single group reindex job."""
    with _group_status_lock:
        s = _group_status.get(_gkey(conn_id, group_name))
    if s:
        return s.copy()
    # Check meta.json for last successful group reindex timestamp.
    meta = _load_meta(conn_id) or {}
    gm = meta.get("group_map", {})
    group_indexed_at = meta.get("group_indexed_at", {})
    last_success_at = group_indexed_at.get(group_name)
    if group_name in gm:
        return {"status": "idle", "message": "", "last_success_at": last_success_at}
    return {"status": "idle", "message": "", "last_success_at": last_success_at}


def _run_group_index(
    conn_id: str, group_name: str, db_type: str, db_url: str, conn_name: str
) -> None:
    key = _gkey(conn_id, group_name)
    phase_started = time.perf_counter()

    def _gs(**kw: object) -> None:
        from datetime import datetime, timezone  # noqa: PLC0415

        with _group_status_lock:
            payload = dict(kw)
            payload.setdefault("updated_at", datetime.now(timezone.utc).isoformat())
            _group_status[key] = payload

    def _mark_phase(phase: str, *, message: str, progress: int) -> None:
        nonlocal phase_started
        now = time.perf_counter()
        elapsed = now - phase_started
        phase_started = now
        logger.info(
            "Group reindex phase [%s / %s] %s (+%.2fs): %s",
            conn_name, group_name, phase, elapsed, message,
        )
        _gs(status="indexing", phase=phase, message=message, progress=progress)

    try:
        _mark_phase(phase="init", message="กำลังเตรียมงาน...", progress=5)

        # Fetch only tables that belong to this group according to meta.json
        meta = _load_meta(conn_id)
        if not meta:
            raise RuntimeError("ยังไม่มี Index หลัก — กรุณา Index ทั้ง connection ก่อน")

        group_tables: list[str] = meta.get("group_map", {}).get(group_name, [])
        if not group_tables:
            raise RuntimeError(f"ไม่พบกลุ่ม '{group_name}' ใน Index")

        _mark_phase(
            phase="fetch",
            message=f"โหลดข้อมูลเฉพาะกลุ่ม ({len(group_tables)} ตาราง)...",
            progress=20,
        )

        t = db_type.lower()
        if t in _SQL_TYPES:
            table_data = _fetch_sql_selected_tables(db_url, group_tables)
        elif t == "mongodb":
            table_data = _fetch_mongo_selected_collections(db_url, group_tables)
        else:
            raise ValueError(f"ไม่รองรับ db_type '{db_type}'")

        if not table_data:
            raise RuntimeError("ดึงข้อมูลตารางไม่ได้ (อาจถูกลบไปแล้ว)")

        _mark_phase(
            phase="transform",
            message=f"แปลงข้อมูล {len(table_data)} ตารางเป็น Documents...",
            progress=45,
        )

        all_docs: list[Document] = []
        group_rows = 0
        group_table_rows: dict[str, int] = {}
        total_tables = len(table_data)
        for idx, (tname, rows) in enumerate(table_data.items(), start=1):
            group_rows += len(rows)
            group_table_rows[tname] = len(rows)
            all_docs.extend(_rows_to_docs(tname, rows, group_name, conn_name))
            table_progress = min(65, 45 + int((idx / max(total_tables, 1)) * 20))
            _gs(
                status="indexing",
                phase="transform",
                message=(
                    f"แปลงตาราง {idx}/{total_tables} ({tname}) "
                    f"สะสม {len(all_docs)} chunks"
                ),
                progress=table_progress,
            )

        if not all_docs:
            raise RuntimeError("ไม่มีข้อมูลในตารางกลุ่มนี้")

        _mark_phase(
            phase="load_store",
            message=f"เตรียม Vector Store สำหรับ {len(all_docs)} chunks...",
            progress=70,
        )

        # Load existing Chroma store and delete old docs for this group then add new ones
        chroma_dir = _chroma_dir(conn_id)
        if not (chroma_dir / "chroma.sqlite3").exists():
            raise RuntimeError("ยังไม่มี Index หลัก — กรุณา Index ทั้ง connection ก่อน")

        embeddings = OpenAIEmbeddings(
            model=settings.openai_embed_model,
            api_key=settings.openai_api_key,
        )
        store = Chroma(
            persist_directory=str(chroma_dir),
            embedding_function=embeddings,
            collection_name=f"db_{conn_id}",
        )
        # Remove all existing docs for this group by metadata filter
        _mark_phase(
            phase="delete_old",
            message=f"ลบข้อมูลเก่าของกลุ่ม '{group_name}'...",
            progress=78,
        )
        try:
            old = store.get(where={"group": group_name})
            if old and old.get("ids"):
                store.delete(ids=old["ids"])
                logger.info("Deleted %d old docs for group '%s'", len(old["ids"]), group_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not delete old docs for group '%s': %s", group_name, exc)

        _mark_phase(
            phase="write_new",
            message=f"กำลัง Embed/เขียน {len(all_docs)} chunks ใหม่...",
            progress=85,
        )
        store.add_documents(all_docs)

        # Update meta.json: refresh table_rows for this group's tables
        _mark_phase(
            phase="save_meta",
            message="บันทึก checkpoint และสรุปผล...",
            progress=95,
        )
        existing_meta = _load_meta(conn_id) or {}
        tr = existing_meta.get("table_rows", {})
        tr.update(group_table_rows)
        existing_meta["table_rows"] = tr

        from datetime import datetime, timezone  # noqa: PLC0415
        now_iso = datetime.now(timezone.utc).isoformat()
        existing_meta["group_indexed_at"] = existing_meta.get("group_indexed_at", {})
        existing_meta["group_indexed_at"][group_name] = now_iso
        _save_meta(conn_id, existing_meta)

        # Touch SQLite group state
        from app.chat_store import touch_group_indexed  # noqa: PLC0415
        touch_group_indexed(conn_id, group_name)

        _gs(
            status="done",
            message=f"สำเร็จ: {len(table_data)} ตาราง, {group_rows} rows, {len(all_docs)} chunks",
            progress=100,
            phase="done",
            last_success_at=now_iso,
        )
        logger.info(
            "Group reindex done [%s / %s]: %d tables, %d rows, %d chunks",
            conn_name, group_name, len(table_data), group_rows, len(all_docs),
        )

    except Exception as exc:  # noqa: BLE001
        logger.exception("Group reindex failed [%s / %s]: %s", conn_id, group_name, exc)
        _gs(status="error", phase="error", message=f"ล้มเหลว: {exc}", progress=0)


def index_group(
    conn_id: str, group_name: str, db_type: str, db_url: str, conn_name: str
) -> None:
    """Launch a per-group reindex job in a daemon thread.  No-op if already running."""
    key = _gkey(conn_id, group_name)
    with _group_status_lock:
        if _group_status.get(key, {}).get("status") == "indexing":
            return
        _group_status[key] = {"status": "indexing", "message": "เริ่มต้น...", "progress": 0}

    t = threading.Thread(
        target=_run_group_index,
        args=(conn_id, group_name, db_type, db_url, conn_name),
        daemon=True,
        name=f"grp-idx-{conn_id[:8]}-{group_name[:10]}",
    )
    t.start()
