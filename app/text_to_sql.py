"""Text-to-Query pipeline.

Flow:
    1. Read DB schema  (db_inspector)
    2. LLM generates a query  (SQL or MongoDB JSON)
    3. Execute the query  (db_query)
    4. LLM summarises the result in Thai

Supported backends: MySQL, PostgreSQL, MSSQL, MongoDB
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import sqlparse
import sqlparse.tokens as T

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.config import settings
from app.db_inspector import get_schema_description
from app.db_query import execute_query

logger = logging.getLogger(__name__)

# Maximum characters of schema text forwarded to the LLM.
# ~4 chars ≈ 1 token, so 20 000 chars ≈ 5 000 tokens — leaves plenty of
# headroom under the default 30 000 TPM limit.
_MAX_SCHEMA_CHARS = 20_000

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
_SQL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a read-only SQL expert specialised in {dialect}.
Your ONLY job is to generate SELECT queries. You are STRICTLY FORBIDDEN from generating any other statement.

ABSOLUTE RULES — violation will cause the query to be rejected:
- ONLY generate SELECT statements. NEVER generate DROP, DELETE, INSERT, UPDATE, ALTER, TRUNCATE, CREATE, REPLACE, or MERGE.
- If the user asks to delete, drop, modify, or change data in any way, respond ONLY with: SELECT 'คำขอนี้ไม่ได้รับอนุญาต — ระบบอนุญาตเฉพาะการสืบค้นข้อมูลเท่านั้น' AS message
- Output raw SQL only — no explanation, no markdown code fences.
- Use ONLY column names that appear in the Schema below. Never guess or invent column names.
- Add LIMIT 200 if the question does not specify a result count.

Schema:
{schema}

กฎเพิ่มเติม:
- ก่อนใช้ column ใด ให้ตรวจสอบให้มั่นใจว่า column นั้นมีอยู่จริงใน Schema
- หา column ที่ต้องการใน Schema ไม่เจอ ให้ใช้เฉพาะ column ที่มั่นใจว่ามีจริง เช่น id, name"""),
    ("human", "{question}"),
])

_MONGO_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a read-only MongoDB expert.
Your ONLY job is to generate find/aggregation queries for reading data.

ABSOLUTE RULES — violation will cause the query to be rejected:
- NEVER generate queries that write, modify, or delete data.
- FORBIDDEN operations: drop, remove, deleteOne, deleteMany, updateOne, updateMany, findOneAndDelete, findOneAndUpdate, findOneAndReplace, insertOne, insertMany, bulkWrite, $out, $merge.
- If the user asks to delete, drop, or modify data, respond ONLY with: {{"error": "คำขอนี้ไม่ได้รับอนุญาต — ระบบอนุญาตเฉพาะการสืบค้นข้อมูลเท่านั้น"}}
- Output raw JSON only — no explanation, no markdown code fences.
- Accepted formats:
    Simple find  → {{"collection":"name","filter":{{...}},"sort":{{...}},"projection":{{...}}}}
    Aggregation  → {{"collection":"name","pipeline":[{{...}},...] }}
- Use ONLY field names that appear in the Schema below. Never guess or invent field names.
- Add {{"$limit": 200}} in the pipeline if the question does not specify a result count.

Schema (collections & fields):
{schema}"""),
    ("human", "{question}"),
])

_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """คุณคือผู้ช่วยวิเคราะห์ข้อมูลของบริษัท
ตอบเป็นภาษาไทย กระชับ ชัดเจน เน้นสรุปตัวเลขและข้อมูลสำคัญ
ถ้าผลลัพธ์ว่างเปล่า ให้บอกว่าไม่พบข้อมูลตามเงื่อนไขที่ระบุ"""),
    ("human", """คำถาม: {question}

Query ที่ใช้:
{query}

ผลลัพธ์ ({row_count} แถว — แสดงสูงสุด 20 แถวแรก):
{result}

กรุณาสรุปคำตอบ"""),
])

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
_DIALECT_MAP = {
    "mysql": "MySQL",
    "postgresql": "PostgreSQL",
    "mssql": "MS SQL Server",
    "other": "SQL",
}


class TextToQueryEngine:
    """Converts a natural-language question into a DB query and returns
    a Thai-language answer together with the raw query and result rows."""

    def __init__(self) -> None:
        self._llm: ChatOpenAI | None = None

    @property
    def llm(self) -> ChatOpenAI:
        if self._llm is None:
            if not settings.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY is not configured.")
            self._llm = ChatOpenAI(
                model=settings.openai_chat_model,
                api_key=settings.openai_api_key,
                temperature=0,
                max_tokens=1024,  # queries are short; cap output to save TPM
            )
        return self._llm

    def ask(self, question: str, db_type: str, db_url: str) -> dict[str, Any]:
        """Run the full pipeline and return a result dict.

        Keys in the returned dict:
            answer    – Thai-language natural-language answer
            query     – The generated SQL / MongoDB JSON string
            db_type   – The DB type used
            row_count – Number of rows retrieved
            rows      – List of row dicts (max 200)
        """
        db_type_l = db_type.lower()

        # ── 1. Read schema ────────────────────────────────────────────────
        try:
            schema = get_schema_description(db_type_l, db_url)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"ไม่สามารถอ่าน schema ได้: {exc}") from exc

        if len(schema) > _MAX_SCHEMA_CHARS:
            schema = schema[:_MAX_SCHEMA_CHARS] + "\n...(schema ถูกตัดทอนเนื่องจากมีขนาดใหญ่เกินไป)"
            logger.warning("[text_to_sql] schema truncated to %d chars", _MAX_SCHEMA_CHARS)

        # ── 2. Generate query ─────────────────────────────────────────────
        try:
            if db_type_l == "mongodb":
                chain = _MONGO_PROMPT | self.llm | StrOutputParser()
                raw_query = chain.invoke({"schema": schema, "question": question}).strip()
            else:
                dialect = _DIALECT_MAP.get(db_type_l, "SQL")
                chain = _SQL_PROMPT | self.llm | StrOutputParser()
                raw_query = chain.invoke(
                    {"dialect": dialect, "schema": schema, "question": question}
                ).strip()
            raw_query = _strip_fence(raw_query)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"OpenAI สร้าง query ไม่ได้: {exc}") from exc

        logger.info("[text_to_sql] generated %s query: %.120s", db_type, raw_query)

        # ── 3. Security validation (Layer 2) ──────────────────────────────
        try:
            if db_type_l == "mongodb":
                parsed_mongo = json.loads(raw_query)
                _validate_mongo_query(parsed_mongo)
            else:
                _validate_sql_query(raw_query)
        except (ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"คำขอถูกปฏิเสธโดยระบบความปลอดภัย: {exc}") from exc

        # ── 4. Execute ────────────────────────────────────────────────────
        try:
            rows = execute_query(db_type_l, db_url, raw_query)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"รัน query ไม่สำเร็จ: {exc}") from exc

        # ── 5. Summarise ──────────────────────────────────────────────────
        result_preview = json.dumps(rows[:20], ensure_ascii=False, default=str)
        try:
            answer_chain = _ANSWER_PROMPT | self.llm | StrOutputParser()
            answer = answer_chain.invoke(
                {
                    "question": question,
                    "query": raw_query,
                    "row_count": len(rows),
                    "result": result_preview,
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("answer summarisation failed: %s", exc)
            answer = f"(ไม่สามารถสรุปผลได้: {exc})"

        return {
            "answer": answer,
            "query": raw_query,
            "db_type": db_type,
            "row_count": len(rows),
            "rows": rows,
        }


def _strip_fence(text: str) -> str:
    """Remove ``` code fences that the LLM sometimes wraps around output."""
    text = text.strip()
    text = re.sub(r"^```[\w]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Security validators (Layer 2 — after LLM generation, before execution)
# ---------------------------------------------------------------------------

_FORBIDDEN_SQL_TYPES = frozenset({
    "DROP", "DELETE", "INSERT", "UPDATE", "CREATE",
    "ALTER", "TRUNCATE", "REPLACE", "MERGE", "RENAME",
    "CALL", "EXEC", "EXECUTE",
})

_FORBIDDEN_MONGO_OPS = frozenset({
    "drop", "remove",
    "deleteOne", "deleteMany",
    "updateOne", "updateMany", "replaceOne",
    "findOneAndDelete", "findOneAndUpdate", "findOneAndReplace",
    "insertOne", "insertMany", "bulkWrite",
    "$out", "$merge",
})


def _validate_sql_query(sql: str) -> None:
    """Raise ValueError if *sql* is not a pure SELECT query.

    Uses sqlparse to parse the statement type, then falls back to a keyword
    scan so that obfuscated or multi-statement payloads are still caught.
    """
    statements = sqlparse.parse(sql.strip())
    if not statements or not any(str(s).strip() for s in statements):
        raise ValueError("ไม่พบ SQL query ในผลลัพธ์")

    for stmt in statements:
        if not str(stmt).strip():
            continue

        stmt_type = (stmt.get_type() or "").upper()

        if stmt_type and stmt_type != "SELECT":
            raise ValueError(
                f"คำสั่ง {stmt_type} ไม่ได้รับอนุญาต — อนุญาตเฉพาะ SELECT เท่านั้น"
            )

        # Fallback: scan every DML/DDL token regardless of get_type()
        for token in stmt.flatten():
            if token.ttype in (T.Keyword.DML, T.Keyword.DDL):
                kw = token.normalized.upper()
                if kw in _FORBIDDEN_SQL_TYPES:
                    raise ValueError(
                        f"คำสั่ง {kw} ไม่ได้รับอนุญาต — อนุญาตเฉพาะ SELECT เท่านั้น"
                    )

    logger.debug("[security] SQL validation passed")


def _validate_mongo_query(query: dict) -> None:
    """Raise ValueError if the MongoDB query contains any write/delete operation."""
    query_text = json.dumps(query)
    for op in _FORBIDDEN_MONGO_OPS:
        # Word-boundary check: look for the op as a JSON key or value
        if re.search(r'(?<!")"' + re.escape(op) + r'"', query_text):
            raise ValueError(
                f"การดำเนินการ '{op}' ไม่ได้รับอนุญาต — อนุญาตเฉพาะการสืบค้นข้อมูลเท่านั้น"
            )
    logger.debug("[security] MongoDB validation passed")


# Singleton used by main.py
text_to_query_engine = TextToQueryEngine()
