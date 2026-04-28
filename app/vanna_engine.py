"""
Vanna.ai engine for Text-to-SQL with ChromaDB persistence.

Vanna works by storing three types of training data in a vector database:
  - DDL      : CREATE TABLE statements (give Vanna the exact schema)
  - SQL      : Example question → SQL pairs (teach Vanna your query patterns)
  - Docs     : Business context strings (explain domain-specific terms)

At query time Vanna retrieves the most similar training examples and feeds
them as few-shot context to the LLM, generating significantly more accurate
SQL than a plain schema-in-prompt approach.

Architecture:
  - ChromaDB_VectorStore  : persists training data in ``chroma_base_dir/vanna/``
  - OpenAI_Chat           : uses the same ChatOpenAI model as the rest of the app

Directory layout (within the shared ChromaDB root):
    chroma_base_dir/
        rag/        ← document RAG (managed by indexer.py)
        vanna/      ← Vanna SQL training data  (managed here)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

_TRAINED_IDS_FILE = Path("vanna_trained.json")
"""Flat JSON file that records which DB connection IDs have had their schema
trained into Vanna's ChromaDB.  Lives next to the app working directory."""


# ---------------------------------------------------------------------------
# Persistence helpers for tracking which DB connections have been schema-trained
# ---------------------------------------------------------------------------
def _load_trained_ids() -> set[str]:
    if _TRAINED_IDS_FILE.exists():
        try:
            return set(json.loads(_TRAINED_IDS_FILE.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            return set()
    return set()


def _save_trained_ids(ids: set[str]) -> None:
    _TRAINED_IDS_FILE.write_text(
        json.dumps(sorted(ids), ensure_ascii=False), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# DDL extraction via SQLAlchemy (reuses db_inspector's URL normaliser)
# ---------------------------------------------------------------------------
def _extract_ddl(db_type: str, db_url: str) -> list[str]:
    """Generate CREATE TABLE DDL strings from a live SQL database.

    Returns an empty list for MongoDB or on any connection error.
    """
    if db_type.lower() == "mongodb":
        return []

    try:
        from sqlalchemy import create_engine, MetaData  # type: ignore[import-untyped]
        from sqlalchemy.schema import CreateTable  # type: ignore[import-untyped]
        from app.db_inspector import _normalize_sql_url  # internal but stable

        engine = create_engine(_normalize_sql_url(db_url), pool_pre_ping=True)
        meta = MetaData()
        meta.reflect(bind=engine)

        ddl_list: list[str] = []
        for tbl in meta.tables.values():
            try:
                ddl_list.append(str(CreateTable(tbl).compile(engine)))
            except Exception as exc:  # noqa: BLE001
                logger.warning("DDL compile failed for table %s: %s", tbl.name, exc)

        engine.dispose()
        logger.info("Extracted %d DDL statement(s) from %s DB", len(ddl_list), db_type)
        return ddl_list

    except Exception as exc:  # noqa: BLE001
        logger.warning("DDL extraction failed (%s): %s", db_type, exc)
        return []


# ---------------------------------------------------------------------------
# VannaEngine
# ---------------------------------------------------------------------------
class VannaEngine:
    """Wraps Vanna.ai (ChromaDB + OpenAI) for persistent Text-to-SQL generation."""

    def __init__(self) -> None:
        self._vn: Any | None = None
        self._trained_ids: set[str] = _load_trained_ids()

    # -- Lazy initialisation -------------------------------------------------
    @property
    def vn(self) -> Any:
        if self._vn is None:
            self._vn = self._init_vanna()
        return self._vn

    def _init_vanna(self) -> Any:
        try:
            from vanna.openai import OpenAI_Chat  # type: ignore[import-untyped]
            from vanna.chromadb import ChromaDB_VectorStore  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "Vanna.ai ไม่ได้ติดตั้ง — กรุณาติดตั้ง: pip install vanna"
            ) from exc

        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        class _CompanyVanna(ChromaDB_VectorStore, OpenAI_Chat):
            def __init__(self, config: dict | None = None) -> None:
                ChromaDB_VectorStore.__init__(self, config=config)
                OpenAI_Chat.__init__(self, config=config)

        vanna_dir = settings.chroma_vanna_dir  # chroma_base_dir/vanna/
        vanna_dir.mkdir(parents=True, exist_ok=True)

        cfg: dict[str, Any] = {
            "model": settings.openai_chat_model,
            "api_key": settings.openai_api_key,
            "path": str(vanna_dir),
        }
        logger.info("Initialising Vanna.ai — ChromaDB at %s", vanna_dir)
        return _CompanyVanna(config=cfg)

    # -- Training ------------------------------------------------------------
    def train_on_connection(
        self,
        conn_id: str,
        db_type: str,
        db_url: str,
        schema_description: str,
    ) -> int:
        """Train Vanna on a DB connection's DDL + schema description.

        Returns the number of training items successfully stored.
        Auto-skips connections that are already trained (idempotent).
        """
        count = 0

        # 1. DDL statements (SQL databases only)
        for ddl in _extract_ddl(db_type, db_url):
            try:
                self.vn.train(ddl=ddl)
                count += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Vanna DDL train failed: %s", exc)

        # 2. Schema description as documentation (covers MongoDB too)
        if schema_description.strip():
            try:
                self.vn.train(documentation=schema_description)
                count += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Vanna doc train failed: %s", exc)

        self._trained_ids.add(conn_id)
        _save_trained_ids(self._trained_ids)
        logger.info(
            "Vanna trained on connection '%s' — %d item(s) added", conn_id, count
        )
        return count

    def is_trained(self, conn_id: str) -> bool:
        return conn_id in self._trained_ids

    def add_sql_example(self, question: str, sql: str) -> str:
        """Add an example question → SQL pair to Vanna's training data."""
        result = self.vn.train(question=question, sql=sql)
        return str(result) if result else "ok"

    def add_ddl(self, ddl: str) -> str:
        """Add a DDL statement to Vanna's training data."""
        result = self.vn.train(ddl=ddl)
        return str(result) if result else "ok"

    def add_documentation(self, documentation: str) -> str:
        """Add business-context documentation to Vanna's training data."""
        result = self.vn.train(documentation=documentation)
        return str(result) if result else "ok"

    # -- Introspection -------------------------------------------------------
    def get_training_data(self) -> list[dict]:
        """Return all stored training entries as a list of dicts."""
        try:
            df = self.vn.get_training_data()
            if df is None or df.empty:
                return []
            return df.fillna("").to_dict(orient="records")
        except Exception as exc:  # noqa: BLE001
            logger.warning("get_training_data failed: %s", exc)
            return []

    def remove_training_data(self, training_id: str) -> bool:
        """Remove a single training entry by its ID."""
        try:
            self.vn.remove_training_data(id=training_id)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("remove_training_data(%s) failed: %s", training_id, exc)
            return False

    def get_trained_connections(self) -> list[str]:
        """Return the list of DB connection IDs that have been schema-trained."""
        return sorted(self._trained_ids)

    # -- SQL Generation ------------------------------------------------------
    def generate_sql(self, question: str) -> str | None:
        """Generate SQL via Vanna's RAG pipeline.

        Returns the SQL string on success, or None if generation fails
        (caller should fall back to the direct LLM approach).
        """
        try:
            sql = self.vn.generate_sql(question)
            return sql.strip() if sql and sql.strip() else None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Vanna generate_sql failed: %s", exc)
            return None


# Singleton used by text_to_sql.py and main.py
vanna_engine = VannaEngine()
