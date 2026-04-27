"""Read schema metadata from supported databases.

Supports:
- SQL (MySQL, PostgreSQL, MSSQL) via SQLAlchemy
- MongoDB via pymongo (samples documents to infer field names)
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_SQL_TYPES = {"mysql", "postgresql", "mssql", "other"}
_MAX_SAMPLE_DOCS = 10   # documents to sample per MongoDB collection

# Map bare scheme → scheme+driver (use our installed drivers)
_SCHEME_MAP = {
    "mysql":      "mysql+pymysql",
    "postgresql": "postgresql+psycopg2",
    "postgres":   "postgresql+psycopg2",
    "mssql":      "mssql+pyodbc",
}


def _normalize_sql_url(url: str) -> str:
    """Rewrite the SQLAlchemy URL dialect to use the drivers we have installed,
    and percent-encode any special characters in the password component so that
    SQLAlchemy can parse the URL correctly.

    e.g.  mysql://user:p@ss=1@host/db  →  mysql+pymysql://user:p%40ss%3D1@host/db
    """
    from urllib.parse import quote, urlparse, urlunparse

    # 1. Swap dialect prefix
    new_url = url
    for bare, with_driver in _SCHEME_MAP.items():
        prefix = f"{bare}://"
        if url.startswith(prefix):
            new_url = with_driver + url[len(prefix) - 3:]
            break

    # 2. Re-encode password (handles @, =, /, +, etc. inside passwords)
    try:
        parsed = urlparse(new_url)
        if parsed.password and any(c in parsed.password for c in "@=+/ "):
            safe_pass = quote(parsed.password, safe="")
            netloc = parsed.netloc.replace(
                f":{parsed.password}@", f":{safe_pass}@", 1
            )
            new_url = urlunparse(parsed._replace(netloc=netloc))
    except Exception:  # noqa: BLE001
        pass  # return as-is if parsing fails

    return new_url


def get_schema_description(db_type: str, url: str) -> str:
    """Return a human-readable schema string to use as LLM context.

    Args:
        db_type: One of mysql / postgresql / mssql / mongodb / other
        url:     Connection string / URL

    Returns:
        Multi-line string describing tables/collections and their fields.
    """
    t = db_type.lower()
    if t in _SQL_TYPES:
        return _sql_schema(url)
    if t == "mongodb":
        return _mongo_schema(url)
    return f"(ประเภทฐานข้อมูล '{db_type}' ไม่รองรับการอ่าน schema อัตโนมัติ)"


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------
def _sql_schema(url: str) -> str:
    from sqlalchemy import create_engine, inspect  # type: ignore[import-untyped]

    engine = create_engine(_normalize_sql_url(url), pool_pre_ping=True)
    try:
        insp = inspect(engine)
        parts: list[str] = []
        for table in insp.get_table_names():
            cols = insp.get_columns(table)
            col_str = ", ".join(f"{c['name']} ({c['type']})" for c in cols)
            try:
                pk = insp.get_pk_constraint(table)
                pk_cols = pk.get("constrained_columns", [])
                pk_str = f"  [PK: {', '.join(pk_cols)}]" if pk_cols else ""
            except Exception:  # noqa: BLE001
                pk_str = ""
            parts.append(f"Table `{table}`: {col_str}{pk_str}")
        return "\n".join(parts) if parts else "(ไม่พบตารางใดในฐานข้อมูลนี้)"
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# MongoDB
# ---------------------------------------------------------------------------
def _mongo_schema(url: str) -> str:
    from pymongo import MongoClient  # type: ignore[import-untyped]

    db_name = _extract_mongo_dbname(url)
    client: MongoClient = MongoClient(url, serverSelectionTimeoutMS=5000)
    try:
        db = client[db_name]
        parts: list[str] = []
        for coll_name in db.list_collection_names():
            coll = db[coll_name]
            sample = list(coll.find({}, limit=_MAX_SAMPLE_DOCS))
            if sample:
                keys: set[str] = set()
                for doc in sample:
                    keys.update(k for k in doc if k != "_id")
                field_str = ", ".join(sorted(keys))
                parts.append(f"Collection `{coll_name}`: fields = [{field_str}]")
            else:
                parts.append(f"Collection `{coll_name}`: (ว่างเปล่า)")
        return "\n".join(parts) if parts else "(ไม่พบ collection ใน database นี้)"
    finally:
        client.close()


def _extract_mongo_dbname(url: str) -> str:
    """Extract the database name from a MongoDB connection string."""
    # mongodb://user:pass@host:port/dbname?options
    m = re.search(r"/([^/?]+)(\?|$)", url.split("//", 1)[-1])
    if m:
        return m.group(1)
    raise ValueError(
        "กรุณาระบุชื่อ database ใน MongoDB URL เช่น mongodb://host:27017/mydb"
    )
