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
    ("system", """คุณคือ SQL expert เชี่ยวชาญ {dialect}
สร้าง SQL query จากคำถามของผู้ใช้ โดยใช้ Schema ด้านล่างเท่านั้น

Schema:
{schema}

กฎเข้มงวด — ต้องปฏิบัติตามทุกข้อ:
- ใช้เฉพาะ SELECT statement เท่านั้น ห้าม INSERT / UPDATE / DELETE / DROP ทุกกรณี
- ตอบเป็น SQL query ล้วนๆ ไม่มีคำอธิบาย ไม่มี markdown code fence
- ห้ามสมมติหรือเดาชื่อ column เด็ดขาด ใช้ได้เฉพาะชื่อ column ที่ปรากฏใน Schema ด้านบนเท่านั้น
- ก่อนใช้ column ใด ให้ตรวจสอบให้มั่นใจว่า column นั้นมีอยู่จริงใน Schema
- ถ้าต้องการ column ที่แทนรหัสสินค้า / ชื่อสินค้า ฯลฯ ให้ดูจาก Schema ก่อน อย่าเดาชื่อเอง
- หา column ที่ต้องการใน Schema ไม่เจอ ให้ใช้เฉพาะ column ที่มั่นใจว่ามีจริง เช่น id, name
- เพิ่ม LIMIT 200 ถ้าคำถามไม่ได้ระบุจำนวนผลลัพธ์"""),
    ("human", "{question}"),
])

_MONGO_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """คุณคือ MongoDB expert
สร้าง MongoDB query จากคำถามของผู้ใช้ โดยใช้ Schema ด้านล่างเท่านั้น

Schema (collections & fields):
{schema}

กฎเข้มงวด — ต้องปฏิบัติตามทุกข้อ:
- ตอบเป็น JSON object เท่านั้น ไม่มีคำอธิบาย ไม่มี markdown code fence
- รูปแบบที่ยอมรับ:
    Simple find  → {{"collection":"name","filter":{{...}},"sort":{{...}},"projection":{{...}}}}
    Aggregation  → {{"collection":"name","pipeline":[{{...}},...] }}
- ห้ามสมมติหรือเดาชื่อ field เด็ดขาด ใช้ได้เฉพาะชื่อ field ที่ปรากฏใน Schema ด้านบนเท่านั้น
- ก่อนใช้ field ใด ให้ตรวจสอบให้มั่นใจว่า field นั้นมีอยู่จริงใน Schema
- ถ้าต้องการ field ที่แทนรหัสสินค้า / ชื่อสินค้า ฯลฯ ให้ดูจาก Schema ก่อน อย่าเดาชื่อเอง
- หา field ที่ต้องการใน Schema ไม่เจอ ให้ใช้เฉพาะ field ที่มั่นใจว่ามีจริง เช่น _id, name
- ห้ามใช้ $out หรือ $merge ใน pipeline
- ใส่ {{"$limit": 200}} ใน pipeline ถ้าไม่มีการระบุจำนวน"""),
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

        # ── 3. Execute ────────────────────────────────────────────────────
        try:
            rows = execute_query(db_type_l, db_url, raw_query)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"รัน query ไม่สำเร็จ: {exc}") from exc

        # ── 4. Summarise ──────────────────────────────────────────────────
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


# Singleton used by main.py
text_to_query_engine = TextToQueryEngine()
