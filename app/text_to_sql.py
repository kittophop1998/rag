"""Text-to-Query pipeline using the official OpenAI Python SDK.

Flow:
    1. Read DB schema  (db_inspector)
    2. Generate query — Vanna.ai RAG first, OpenAI direct fallback
    3. Security-validate the query
    4. Execute the query  (db_query)
    5. OpenAI summarises the result in Thai (with the SQL embedded in the answer)

Supported backends: MySQL, PostgreSQL, MSSQL, MongoDB
"""

from __future__ import annotations

from datetime import date
import json
import logging
import numbers
import re
from typing import Any

import sqlparse
import sqlparse.tokens as T

from app.config import settings
from app.constants import (
    MAX_SCHEMA_CHARS,
    MAX_PLANNER_SCHEMA_CHARS,
    MAX_SEMANTIC_CONTEXT_CHARS,
    MAX_RESULT_PREVIEW_ROWS,
    MAX_RESULT_PREVIEW_CHARS,
)
from app.db_inspector import get_schema_description
from app.db_query import execute_query
from app.openai_client import get_client

logger = logging.getLogger(__name__)

_MAX_SCHEMA_CHARS = MAX_SCHEMA_CHARS

# ---------------------------------------------------------------------------
# Prompt templates (plain strings for the OpenAI SDK)
# ---------------------------------------------------------------------------

# Default few-shot examples shown to the model when no connection-specific
# examples are stored.  They teach JOIN patterns, aggregates, LIKE, and date
# filtering — the four most common query shapes.
_DEFAULT_FEW_SHOTS = """
--- ตัวอย่าง SQL ที่ถูกต้อง (Few-Shot Examples) ---
Q: "แสดงรายการสินค้าทั้งหมด"
SQL: SELECT * FROM products p LIMIT 200;

Q: "ยอดขายรวมเดือนนี้เท่าไร"
SQL: SELECT SUM(o.total_amount) AS total_sales
     FROM orders o
     WHERE MONTH(o.created_at) = MONTH(CURDATE())
       AND YEAR(o.created_at) = YEAR(CURDATE());

Q: "ลูกค้าคนไหนซื้อของมากที่สุด 10 อันดับแรก"
SQL: SELECT c.name, COUNT(o.id) AS order_count, SUM(o.total_amount) AS total_spent
     FROM customers c
     JOIN orders o ON c.id = o.customer_id
     GROUP BY c.id, c.name
     ORDER BY total_spent DESC
     LIMIT 10;

Q: "สินค้าไหนยังมีสต็อกน้อยกว่า 10 ชิ้น"
SQL: SELECT p.name, p.stock_quantity
     FROM products p
     WHERE p.stock_quantity < 10
     ORDER BY p.stock_quantity ASC;

Q: "ค้นหาพนักงานชื่อ สมชาย"
SQL: SELECT * FROM employees e WHERE e.name LIKE '%สมชาย%' LIMIT 200;
--- จบตัวอย่าง ---"""

_SQL_SYSTEM = """You are a senior read-only SQL expert specialised in {dialect}.
Your ONLY job is to generate the most accurate and efficient SELECT queries.
You are STRICTLY FORBIDDEN from generating any other statement.

ABSOLUTE RULES — violation will cause the query to be rejected:
- ONLY generate SELECT statements. NEVER generate DROP, DELETE, INSERT, UPDATE, ALTER, TRUNCATE, CREATE, REPLACE, or MERGE.
- If the user asks to delete, drop, modify, or change data in any way, respond ONLY with: SELECT 'คำขอนี้ไม่ได้รับอนุญาต — ระบบอนุญาตเฉพาะการสืบค้นข้อมูลเท่านั้น' AS message
- Output raw SQL only — no explanation, no markdown code fences.
- Use ONLY column names that appear in the Schema below. Never guess or invent column names.
- Add LIMIT 200 if the question does not specify a result count.
- Current Date: {current_date}
- Always use table aliases and qualify columns with aliases (e.g. u.id, o.created_at) to avoid ambiguous column errors.
- Use JOIN hints from [FK: ...] annotations in the Schema to connect related tables correctly.

QUERY QUALITY RULES:
- For aggregation questions (sum, count, average, top-N), always use GROUP BY and ORDER BY appropriately.
- For date/time filtering, use the most precise condition possible (YEAR + MONTH, DATE_FORMAT, BETWEEN, etc.).
- For "top-N" questions, always include ORDER BY with the relevant metric DESC and LIMIT N.
- For JOIN queries, select only meaningful columns — avoid SELECT * when joining multiple tables.
- Prefer column aliases (AS) in SELECT for readability (e.g. SUM(o.total) AS total_sales).
- If the question is ambiguous, write the query that best matches the most common interpretation.

{semantic_context}

Schema (relevant tables only):
{schema}

{few_shots}

กฎเพิ่มเติม:
- ก่อนใช้ column ใด ให้ตรวจสอบให้มั่นใจว่า column นั้นมีอยู่จริงใน Schema
- หา column ที่ต้องการใน Schema ไม่เจอ ให้ใช้เฉพาะ column ที่มั่นใจว่ามีจริง เช่น id, name
- ใช้ FK annotations ใน Schema เป็นคำแนะนำในการ JOIN ตารางที่ถูกต้อง
- ถ้า Schema ที่ให้มาไม่มีข้อมูลเพียงพอตอบคำถามเลย ให้ตอบว่า: CLARIFY: <อธิบายสั้นๆ ว่าต้องการข้อมูลเพิ่มเติมอะไร>

{entity_hints}

ขั้นตอนการทำงาน (ให้เขียน reasoning เป็น SQL comment ก่อน SQL จริง):
-- Step 1: Domain: <ระบุกลุ่มข้อมูลที่คำถามต้องการ>
-- Step 2: Tables: <ชื่อตาราง> เพราะ <เหตุผลสั้นๆ>
-- Step 3: JOIN: <FK ที่ใช้เชื่อม หรือ "ไม่มี JOIN">
จากนั้นสร้าง SQL SELECT statement เท่านั้น"""

# Prompt for Step 1 of two-step querying: identify relevant tables only.
_TABLE_PLANNER_SYSTEM = """คุณคือผู้เชี่ยวชาญฐานข้อมูลที่วิเคราะห์คำถามภาษาไทยเพื่อระบุตารางที่จำเป็น

{semantic_context}

Schema ของฐานข้อมูล (ทุกตาราง):
{schema}

กฎสำคัญในการเลือกตาราง:
1. ถ้าคำถามระบุชื่อตารางตรงๆ (เช่น "coupons", "orders", "products") ให้รวมตารางนั้นทุกกรณี
2. ถ้าคำถามมีคำที่ตรงกับชื่อตาราง (แม้บางส่วน เช่น "coupon" ↔ "coupons") ให้รวมตารางนั้น
3. ให้ดูจาก FK annotations เพื่อรวมตารางที่ต้อง JOIN ด้วย
4. ห้ามเลือกตารางที่ดูเหมือน "เกี่ยวข้อง" แต่ไม่ได้ถูกถามโดยตรง

{mandatory_hint}

จากคำถามของผู้ใช้ ให้ระบุ **เฉพาะชื่อตาราง** ที่จำเป็นสำหรับการสร้าง SQL query
- ตอบเป็น JSON array ของชื่อตารางเท่านั้น เช่น: ["coupons", "coupons_code", "coupons_product"]
- ห้ามตอบอย่างอื่น ห้ามมีคำอธิบาย


ขั้นตอนการทำงาน:
1. วิเคราะห์ว่าคำถามต้องการข้อมูลจากกลุ่ม (Domain) ใด
2. เลือก Table ที่จำเป็นและตรวจสอบ Schema ว่ามี Column นั้นจริงหรือไม่
3. เขียนคำอธิบายสั้นๆ ว่าทำไมถึง Join ตารางเหล่านี้
4. สร้าง SQL SELECT statement เท่านั้น"""

_MONGO_SYSTEM = """You are a read-only MongoDB expert.
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
{schema}"""

_ANSWER_SYSTEM = """คุณคือนักวิเคราะห์ข้อมูลของบริษัทที่มีความเชี่ยวชาญสูง
ตอบเป็นภาษาไทย กระชับ ชัดเจน เน้นสรุปตัวเลขและข้อมูลสำคัญ
ถ้าผลลัพธ์ว่างเปล่า ให้บอกว่าไม่พบข้อมูลตามเงื่อนไขที่ระบุ และแนะนำให้ลองค้นหาด้วยคำที่กว้างขึ้นหรือปรับช่วงเวลา

รูปแบบการตอบ:
- ถ้ามีข้อมูลหลายรายการให้สรุปเป็น bullet list หรือตาราง Markdown
- ถ้ามี URL รูปภาพในผลลัพธ์ ให้แสดงด้วย syntax ![ชื่อ](url)
- ตัวเลขสำคัญให้ **ตัวหนา** และจัดรูปแบบให้อ่านง่าย (เช่น ใส่ comma คั่นหลักพัน)
- ถ้าผลลัพธ์มีมากกว่า 20 แถว ให้วิเคราะห์สถิติสรุป (รวม, เฉลี่ย, สูงสุด, ต่ำสุด) แทนการแสดงทุก row
- ถ้ามีข้อมูลเชิงแนวโน้ม (เช่น ยอดขายตามเดือน) ให้วิเคราะห์แนวโน้มเพิ่มเติมด้วย
- ท้ายคำตอบให้แสดง query ที่ใช้ในรูปแบบ code block เสมอ เพื่อให้ผู้ใช้ตรวจสอบได้"""

_ANSWER_USER = """คำถาม: {question}

ผลลัพธ์ ({row_count} แถว — แสดงสูงสุด 20 แถวแรก):
{result}

สถิติสรุปเพิ่มเติม (สำหรับกรณีข้อมูลมีจำนวนมาก):
{aggregate_summary}

กรุณาสรุปคำตอบจากข้อมูลด้านบน (ไม่ต้องแสดง SQL ในคำตอบ)"""

# Prompt for detecting named entities (proper nouns) in the user question.
_ENTITY_DETECT_SYSTEM = """คุณคือผู้ช่วยวิเคราะห์คำถามภาษาไทยเพื่อดึง "ชื่อเฉพาะ" (Named Entities) ที่อาจเป็นค่าข้อมูลจริงใน Database
ชื่อเฉพาะ ได้แก่: ชื่อบริษัท, ชื่อสินค้า, ชื่อบุคคล, ชื่อแบรนด์, ชื่อหมวดหมู่, รหัสอ้างอิง

กฎ:
- ตอบเป็น JSON array ของ string เท่านั้น เช่น: ["ABC", "สมชาย", "iPhone 15"]
- ถ้าคำถามมีแต่คำทั่วไป (เช่น "ยอดขาย", "สินค้าทั้งหมด") ให้ตอบ: []
- ห้ามตอบอย่างอื่นนอกจาก JSON array"""

# Prompt for broadening a query that returned 0 rows.
_BROADEN_SYSTEM = """You are a SQL expert. A query returned 0 rows, which means the WHERE conditions may be too strict.
Analyse the query and rewrite it to be broader so the user can see nearby/similar data.

Strategies:
- Replace exact match (=) with LIKE '%value%' for string columns
- Widen date ranges (e.g. last 90 days instead of last 30 days)
- Remove low-confidence filter conditions that might be wrong
- Keep the core aggregation / JOIN structure intact
- Add LIMIT 200 if not present

Output raw SQL only — no explanation, no markdown fences.
ABSOLUTE RULES: Only SELECT statements. Use only columns that exist in the schema."""

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
_DIALECT_MAP = {
    "mysql":      "MySQL",
    "postgresql": "PostgreSQL",
    "mssql":      "MS SQL Server",
    "other":      "SQL",
}


class TextToQueryEngine:
    """Converts a natural-language question into a DB query and returns
    a Thai-language answer together with the raw query and result rows."""

    def ask(
        self,
        question: str,
        db_type: str,
        db_url: str,
        conn_id: str = "",
    ) -> dict[str, Any]:
        """Run the full pipeline and return a result dict.

        Keys in the returned dict:
            answer       – Thai-language natural-language answer (includes SQL block)
            query        – The generated SQL / MongoDB JSON string
            db_type      – The DB type used
            row_count    – Number of rows retrieved
            rows         – List of row dicts (max 200)
            using_vanna  – True when Vanna.ai generated the SQL
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

        # ── 2a. Load Semantic Catalog + build context string ──────────────
        semantic_context = ""
        few_shots = ""
        if conn_id:
            try:
                from app.db_indexer import get_semantic_catalog  # noqa: PLC0415
                catalog = get_semantic_catalog(conn_id)
                semantic_context = _build_semantic_context(catalog)
                # Future extension: per-connection few-shot examples could be
                # loaded from meta.json["few_shots"] here.
            except Exception as exc:  # noqa: BLE001
                logger.warning("[text_to_sql] Could not load semantic catalog: %s", exc)

        # ── 2b. Generate query — try Vanna first, fall back to direct OpenAI ─
        raw_query: str | None = None
        using_vanna = False

        if conn_id and db_type_l != "mongodb":
            raw_query = _try_vanna(question, conn_id, db_type_l, db_url, schema)
            if raw_query:
                using_vanna = True
                logger.info("[text_to_sql] Vanna generated %s query: %.120s", db_type, raw_query)

        if raw_query is None:
            # ── 2c. Entity Resolution: discover actual DB values ──────────
            entity_hints = ""
            if db_type_l != "mongodb":
                entity_hints = _resolve_entity_hints(
                    question, schema, db_type_l, db_url
                )
                if entity_hints:
                    logger.info("[text_to_sql] Entity hints resolved: %.200s", entity_hints)

            raw_query = _generate_query_with_openai(
                question, db_type_l, schema,
                semantic_context=semantic_context,
                few_shots=few_shots,
                entity_hints=entity_hints,
            )
            logger.info("[text_to_sql] OpenAI generated %s query: %.120s", db_type, raw_query)

        # ── 2d. Clarification detection ───────────────────────────────────
        if isinstance(raw_query, str) and raw_query.upper().startswith("CLARIFY:"):
            clarification_msg = raw_query[len("CLARIFY:"):].strip()
            logger.info("[text_to_sql] LLM requested clarification: %s", clarification_msg)
            return {
                "answer":      f"❓ {clarification_msg}",
                "query":       "",
                "db_type":     db_type,
                "row_count":   0,
                "rows":        [],
                "using_vanna": using_vanna,
                "needs_clarification": True,
            }

        # ── 3. Security validation ─────────────────────────────────────────
        def _sec_validate(q: str) -> None:
            try:
                if db_type_l == "mongodb":
                    _validate_mongo_query(json.loads(q))
                else:
                    _validate_sql_query(q)
            except (ValueError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"คำขอถูกปฏิเสธโดยระบบความปลอดภัย: {exc}") from exc

        _sec_validate(raw_query)

        # ── 3b. Pre-execution column validation (SQL only) ─────────────────
        if db_type_l != "mongodb":
            col_issues = _check_columns_against_schema(raw_query, schema)
            if col_issues:
                logger.warning(
                    "[text_to_sql] Column validation found issues, retrying: %s", col_issues
                )
                raw_query = _fix_query_with_feedback(
                    raw_query, col_issues, question, db_type_l, schema
                )
                _sec_validate(raw_query)

        # ── 4. Execute (retry once on column-not-found error) ─────────────
        rows: list[dict[str, Any]] = []
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                rows = execute_query(db_type_l, db_url, raw_query)
                last_exc = None
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt == 0 and _is_column_error(str(exc)):
                    logger.warning(
                        "[text_to_sql] Column error on attempt %d, retrying with feedback: %s",
                        attempt + 1, exc,
                    )
                    try:
                        raw_query = _fix_query_with_feedback(
                            raw_query, str(exc), question, db_type_l, schema
                        )
                        _sec_validate(raw_query)
                    except Exception as fix_exc:  # noqa: BLE001
                        logger.warning("[text_to_sql] Fix attempt failed: %s", fix_exc)
                        break
                else:
                    break

        if last_exc is not None:
            raise RuntimeError(f"รัน query ไม่สำเร็จ: {last_exc}") from last_exc

        # ── 4b. Empty-result self-correction (SQL only, one retry) ────────
        if len(rows) == 0 and db_type_l != "mongodb":
            logger.info("[text_to_sql] Zero rows returned — attempting broader query")
            try:
                broader_query = _broaden_empty_query(raw_query, question, db_type_l, schema)
                _sec_validate(broader_query)
                broader_rows = execute_query(db_type_l, db_url, broader_query)
                if broader_rows:
                    logger.info(
                        "[text_to_sql] Broader query found %d rows", len(broader_rows)
                    )
                    rows = broader_rows
                    raw_query = broader_query  # surface the broader query to the user
            except Exception as exc:  # noqa: BLE001
                logger.warning("[text_to_sql] Broader query attempt failed: %s", exc)

        # ── 5. Summarise with OpenAI (SQL embedded in the answer) ──────────
        preview_rows = rows[:MAX_RESULT_PREVIEW_ROWS]
        result_preview = json.dumps(preview_rows, ensure_ascii=False, default=str)
        # Hard char cap: truncate long JSON blobs (e.g. rows with big text fields).
        if len(result_preview) > MAX_RESULT_PREVIEW_CHARS:
            result_preview = result_preview[:MAX_RESULT_PREVIEW_CHARS] + "... (ตัดทอนเพื่อประหยัด token)"
            logger.warning(
                "[text_to_sql] result_preview truncated to %d chars", MAX_RESULT_PREVIEW_CHARS
            )
        aggregate_summary = _build_aggregate_summary(rows, MAX_RESULT_PREVIEW_ROWS)

        try:
            client = get_client()
            resp = client.chat.completions.create(
                model=settings.openai_chat_model,
                temperature=0.2,
                max_tokens=2048,
                messages=[
                    {"role": "system", "content": _ANSWER_SYSTEM},
                    {
                        "role": "user",
                        "content": _ANSWER_USER.format(
                            question=question,
                            row_count=len(rows),
                            result=result_preview,
                            aggregate_summary=aggregate_summary,
                        ),
                    },
                ],
            )
            answer = _strip_sql_from_answer(resp.choices[0].message.content or "")
        except Exception as exc:  # noqa: BLE001
            logger.exception("answer summarisation failed: %s", exc)
            answer = f"(ไม่สามารถสรุปผลได้: {exc})\n\nพบข้อมูล {len(rows)} แถว"

        return {
            "answer":      answer,
            "query":       raw_query,
            "db_type":     db_type,
            "row_count":   len(rows),
            "rows":        rows,
            "using_vanna": using_vanna,
        }


# ---------------------------------------------------------------------------
# OpenAI query generation
# ---------------------------------------------------------------------------
def _generate_query_with_openai(
    question: str,
    db_type_l: str,
    schema: str,
    semantic_context: str = "",
    few_shots: str = "",
    entity_hints: str = "",
) -> str:
    """Generate SQL or MongoDB query using the OpenAI Python SDK.

    For SQL backends with many tables, runs the Two-Step Querying strategy:
      Step 1 — Table Planner: identify which tables are relevant.
      Step 2 — SQL Generator: write SQL using only those tables + semantic context.
    """
    client = get_client()
    try:
        if db_type_l == "mongodb":
            system_msg = _MONGO_SYSTEM.format(schema=schema)
        else:
            dialect = _DIALECT_MAP.get(db_type_l, "SQL")

            # ── Step 1: Two-Step — pick relevant tables ────────────────────
            planned_tables = _plan_relevant_tables(question, schema, semantic_context, db_type_l)
            focused_schema = schema
            if planned_tables:
                focused_schema = _build_focused_schema(schema, planned_tables)

            # Minify types in the focused schema to cut token usage by ~40 %.
            focused_schema_min = _minify_schema(focused_schema)
            logger.debug(
                "[text_to_sql] Schema size: full=%d → focused=%d → minified=%d chars",
                len(schema), len(focused_schema), len(focused_schema_min),
            )

            # ── Step 2: Generate SQL with focused + minified context ───────
            system_msg = _SQL_SYSTEM.format(
                dialect=dialect,
                schema=focused_schema_min,
                current_date=date.today().isoformat(),
                semantic_context=semantic_context or "(ไม่มีคำอธิบายเพิ่มเติม)",
                few_shots=few_shots or _DEFAULT_FEW_SHOTS,
                entity_hints=entity_hints or "",
            )

        resp = client.chat.completions.create(
            model=settings.openai_chat_model,
            temperature=0,
            max_tokens=2048,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": question},
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
        return _strip_fence(raw)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"OpenAI สร้าง query ไม่ได้: {exc}") from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _strip_fence(text: str) -> str:
    """Remove ``` code fences that the LLM sometimes wraps around output."""
    text = text.strip()
    text = re.sub(r"^```[\w]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Schema minification — reduce token count before sending to LLM
# ---------------------------------------------------------------------------

# Ordered list of (pattern, replacement) applied globally to the schema text.
# db_inspector renders types as str(col['type']) which gives e.g. INTEGER(11),
# VARCHAR(255), DECIMAL(10,2).  We collapse the size argument and unify aliases.
_MINIFY_RULES: list[tuple[re.Pattern[str], str]] = [
    # Integer family → INT
    (re.compile(r"\b(TINY|SMALL|MEDIUM|BIG)?INTEGER\s*\(\d+\)", re.I), "INT"),
    (re.compile(r"\bBIGINT\s*\(\d+\)", re.I), "INT"),
    (re.compile(r"\b(TINY|SMALL|MEDIUM)?INT\s*\(\d+\)", re.I), "INT"),
    # Variable-length strings
    (re.compile(r"\bVARCHAR\s*\(\d+\)", re.I), "VARCHAR"),
    (re.compile(r"\bNVARCHAR\s*\(\d+\)", re.I), "VARCHAR"),
    (re.compile(r"\bCHAR\s*\(\d+\)", re.I), "CHAR"),
    # Decimal / numeric
    (re.compile(r"\bDECIMAL\s*\([\d,\s]+\)", re.I), "DECIMAL"),
    (re.compile(r"\bNUMERIC\s*\([\d,\s]+\)", re.I), "NUMERIC"),
    # Float with optional precision
    (re.compile(r"\bFLOAT\s*\(\d+\)", re.I), "FLOAT"),
    # Datetime with optional precision
    (re.compile(r"\bDATETIME\s*\(\d+\)", re.I), "DATETIME"),
    (re.compile(r"\bTIMESTAMP\s*\(\d+\)", re.I), "TIMESTAMP"),
    # Long/medium/tiny text+blob variants
    (re.compile(r"\b(LONG|MEDIUM|TINY)TEXT\b", re.I), "TEXT"),
    (re.compile(r"\b(LONG|MEDIUM|TINY)BLOB\b", re.I), "BLOB"),
    # Boolean aliases
    (re.compile(r"\bBOOLEAN\b", re.I), "BOOL"),
    (re.compile(r"\bTINYINT\s*\(1\)", re.I), "BOOL"),
    # PostgreSQL-specific
    (re.compile(r"\bCHARACTER VARYING\s*\(\d+\)", re.I), "VARCHAR"),
    (re.compile(r"\bDOUBLE PRECISION\b", re.I), "DOUBLE"),
]

# Additional noise patterns that appear in some drivers but carry no SQL info.
_NOISE_SUBS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\s+NOT\s+NULL\b", re.I), ""),
    (re.compile(r"\s+NULL\b", re.I), ""),
    (re.compile(r"\s+AUTO_INCREMENT\b", re.I), ""),
    (re.compile(r"\s+UNSIGNED\b", re.I), ""),
    (re.compile(r"\s+SIGNED\b", re.I), ""),
    (re.compile(r"\s+DEFAULT\s+'[^']*'", re.I), ""),
    (re.compile(r"\s+DEFAULT\s+\S+", re.I), ""),
    (re.compile(r"\s+COLLATE\s+\S+", re.I), ""),
    (re.compile(r"\s+CHARACTER\s+SET\s+\S+", re.I), ""),
]


def _minify_schema(schema: str) -> str:
    """Return a schema string with verbose type info collapsed to short names.

    Applies globally to the entire schema text — no line-by-line parsing needed.
    Preserves table names, column names, PK and FK annotations.

    Example:
      Before: id (INTEGER(11)), total (DECIMAL(10,2)), name (VARCHAR(255))
      After:  id (INT), total (DECIMAL), name (VARCHAR)
    """
    s = schema
    for pattern, repl in _MINIFY_RULES:
        s = pattern.sub(repl, s)
    for pattern, repl in _NOISE_SUBS:
        s = pattern.sub(repl, s)
    # Collapse repeated spaces produced by noise removal
    s = re.sub(r"  +", " ", s)
    return s


def _make_planner_schema(schema: str) -> str:
    """Ultra-compact schema for the Table Planner: only table + column names.

    No types, no PK/FK annotations — the planner only needs to know which
    tables exist and what columns they have in order to route the question.

    Output (one line per table):
        orders: id, customer_id, total_amount, created_at
        customers: id, name, email, phone

    Handles multi-digit type arguments such as DECIMAL(10,2) correctly by
    extracting column names via regex (word before ' (') rather than naive
    comma-splitting which would break on nested commas.
    """
    lines: list[str] = []
    for line in schema.splitlines():
        m = re.match(r"Table\s+`([^`]+)`:\s*(.+)", line)
        if not m:
            continue
        table    = m.group(1)
        col_block = m.group(2)
        # Extract column names: each entry is "colname (TYPE...)".
        # Use lookbehind-for-comma OR start-of-string so we match the word
        # right before " (" while skipping digits/words inside nested parens.
        col_names = re.findall(r"(?:(?<=,)|^)\s*([A-Za-z_]\w*)\s+\(", col_block)
        # Strip annotation keywords that can appear as "words before ("
        col_names = [c for c in col_names if c.upper() not in ("PK", "FK", "CONSTRAINT")]
        if col_names:
            lines.append(f"{table}: {', '.join(col_names)}")
    return "\n".join(lines) if lines else schema


# ---------------------------------------------------------------------------
# Two-step query planning helpers
# ---------------------------------------------------------------------------
_MIN_TABLES_FOR_PLANNING = 5
"""Only run the table-planner step when the schema has at least this many tables.
Smaller databases don't need the extra LLM round-trip."""


def _count_schema_tables(schema: str) -> int:
    """Count table/collection entries in a schema string."""
    return len(re.findall(r"^Table `", schema, re.MULTILINE))


def _extract_literal_table_matches(question: str, schema: str) -> list[str]:
    """Return table names that literally appear in the question text.

    Handles both exact matches (e.g. "coupons") and singular/plural variants
    (e.g. "coupon" matches "coupons").  Case-insensitive.
    This acts as a hard constraint passed to the Table Planner to prevent it
    from substituting a semantically similar but incorrect table.
    """
    # Collect all table names from schema
    all_tables = re.findall(r"Table\s+`([^`]+)`", schema)
    q_lower = question.lower()
    matched: list[str] = []
    for tbl in all_tables:
        tbl_lower = tbl.lower()
        # Exact match or plural/singular variant: "coupon" in "coupons", vice versa
        if (
            tbl_lower in q_lower
            or tbl_lower.rstrip("s") in q_lower
            or (tbl_lower + "s") in q_lower
        ):
            matched.append(tbl)
    return matched


def _plan_relevant_tables(
    question: str,
    schema: str,
    semantic_context: str,
    db_type_l: str,
) -> list[str] | None:
    """Step 1 of Two-Step Querying: ask AI which tables are needed.

    Strategy:
    1. First do a cheap literal-match pass to find tables explicitly named in
       the question (e.g. "coupons" → `coupons`, `coupons_code`, `coupons_product`).
       These become *mandatory* tables the planner must include.
    2. Call the LLM to pick additional related tables (for JOINs, FK chains).
    3. Merge mandatory + AI-chosen tables and return the union.

    Returns a list of table names to focus on, or None to skip (small schema
    or LLM call failed).
    """
    if db_type_l == "mongodb":
        return None  # MongoDB queries are schemaless; skip planning

    if _count_schema_tables(schema) < _MIN_TABLES_FOR_PLANNING:
        return None  # Schema is small enough to send in full

    # ── Step 1a: Literal keyword match (cheap, no AI needed) ─────────────
    mandatory = _extract_literal_table_matches(question, schema)
    mandatory_hint = ""
    if mandatory:
        mandatory_hint = (
            f"\n⚠️ คำถามระบุตารางเหล่านี้โดยตรง — ต้องรวมทุกตารางนี้เสมอ: "
            f"{json.dumps(mandatory, ensure_ascii=False)}"
        )
        logger.info("[text_to_sql] Mandatory tables from literal match: %s", mandatory)

    # ── Step 1b: AI Planner for JOIN / FK resolution ──────────────────────
    client = get_client()
    planner_schema = _make_planner_schema(schema)[:MAX_PLANNER_SCHEMA_CHARS]
    system_msg = _TABLE_PLANNER_SYSTEM.format(
        semantic_context=semantic_context or "(ไม่มีคำอธิบายเพิ่มเติม)",
        schema=planner_schema,
        mandatory_hint=mandatory_hint,
    )
    try:
        resp = client.chat.completions.create(
            model=settings.openai_chat_model,
            temperature=0,
            max_tokens=256,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": question},
            ],
        )
        raw = _strip_fence((resp.choices[0].message.content or "").strip())
        parsed = json.loads(raw)
        if isinstance(parsed, list) and parsed:
            ai_tables = [str(t) for t in parsed if t]
            # Merge: mandatory always wins, AI adds FK-related extras
            merged = list(dict.fromkeys(mandatory + [t for t in ai_tables if t not in mandatory]))
            logger.info(
                "[text_to_sql] Table plan: mandatory=%s ai=%s merged=%s",
                mandatory, ai_tables, merged,
            )
            return merged
    except Exception as exc:  # noqa: BLE001
        logger.warning("[text_to_sql] Table planner failed (%s) — using literal matches", exc)

    # Fallback: return mandatory matches only (no AI)
    if mandatory:
        return mandatory
    return None


def _build_focused_schema(schema: str, table_names: list[str]) -> str:
    """Return only the schema lines for the given tables.

    Falls back to the full schema if no lines match (safety net).
    """
    if not table_names:
        return schema
    target = {t.lower() for t in table_names}
    lines = [
        line for line in schema.splitlines()
        if re.match(r"Table\s+`([^`]+)`", line) and
           re.match(r"Table\s+`([^`]+)`", line).group(1).lower() in target  # type: ignore[union-attr]
    ]
    return "\n".join(lines) if lines else schema


def _build_semantic_context(
    catalog: dict[str, str],
    table_names: list[str] | None = None,
) -> str:
    """Build a formatted semantic-context block from the catalog dict.

    If *table_names* is provided only those tables are included (used after
    the Two-Step planner narrows down the relevant set).
    The result is always capped at MAX_SEMANTIC_CONTEXT_CHARS to stay within
    the token budget.
    """
    if not catalog:
        return ""
    items = (
        [(t, catalog[t]) for t in table_names if t in catalog]
        if table_names is not None
        else list(catalog.items())
    )
    if not items:
        return ""
    lines = ["คำอธิบายตารางตามโดเมนธุรกิจ (Semantic Catalog):"]
    total_chars = len(lines[0])
    for table, desc in items:
        entry = f"  - `{table}`: {desc}"
        if total_chars + len(entry) > MAX_SEMANTIC_CONTEXT_CHARS:
            lines.append("  - (... ตัดคำอธิบายที่เหลือเพื่อประหยัด token)")
            break
        lines.append(entry)
        total_chars += len(entry)
    return "\n".join(lines) if len(lines) > 1 else ""


def _strip_sql_from_answer(text: str) -> str:
    """Remove any SQL/code blocks the LLM accidentally put in the answer.

    The SQL is always shown in a dedicated panel by the frontend, so it must
    not appear a second time inside the answer text.
    Strips both fenced blocks (```...```) and bare SELECT/WITH statements
    that the model sometimes emits as plain text.
    """
    # Remove fenced code blocks (```sql … ```, ``` … ```, etc.)
    text = re.sub(r"```[\w]*\n?[\s\S]*?```", "", text)
    # Remove trailing bare SELECT / WITH statement lines (Vanna sometimes adds these)
    text = re.sub(
        r"\n?(SELECT|WITH)\s+[\s\S]*?(?:LIMIT\s+\d+|;)?\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip()


def _build_aggregate_summary(rows: list[dict[str, Any]], preview_limit: int) -> str:
    """Build lightweight aggregate stats to reduce summary bias from truncated previews."""
    total_rows = len(rows)
    if total_rows == 0:
        return "ไม่พบข้อมูล"

    if total_rows <= preview_limit:
        return "ข้อมูลมีจำนวนไม่เกินช่วงที่แสดงตัวอย่าง จึงไม่จำเป็นต้องคำนวณสถิติเพิ่มเติม"

    numeric_cols: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, numbers.Number):
                numeric_cols.setdefault(key, []).append(float(value))

    summary: dict[str, Any] = {
        "preview_row_count": min(total_rows, preview_limit),
        "total_row_count":   total_rows,
    }

    if numeric_cols:
        summary["numeric_aggregates"] = {
            col: {
                "count": len(values),
                "sum":   round(sum(values), 4),
                "avg":   round(sum(values) / len(values), 4) if values else None,
                "min":   round(min(values), 4) if values else None,
                "max":   round(max(values), 4) if values else None,
            }
            for col, values in numeric_cols.items()
            if values
        }
    else:
        summary["numeric_aggregates"] = "ไม่พบคอลัมน์ตัวเลขสำหรับคำนวณสถิติ"

    return json.dumps(summary, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Entity Resolution helpers
# ---------------------------------------------------------------------------

def _parse_text_columns(schema: str) -> dict[str, list[str]]:
    """Parse schema to return {table: [text_column_names]}.

    Identifies columns with string types (VARCHAR, CHAR, TEXT, NVARCHAR)
    suitable for LIKE-based value discovery.
    """
    result: dict[str, list[str]] = {}
    _TEXT_TYPES = re.compile(r"\b(VAR)?CHAR|NVARCHAR|TEXT\b", re.I)
    for line in schema.splitlines():
        m = re.match(r"Table\s+`([^`]+)`\s*:\s*(.+)", line)
        if not m:
            continue
        table = m.group(1)
        col_block = m.group(2)
        text_cols: list[str] = []
        # Each column: "colname (TYPE...)"
        for entry in re.finditer(r"([A-Za-z_]\w*)\s+\(([^)]+)\)", col_block):
            col_name = entry.group(1)
            col_type = entry.group(2)
            if col_name.upper() in ("PK", "FK"):
                continue
            if _TEXT_TYPES.search(col_type):
                text_cols.append(col_name)
        if text_cols:
            result[table] = text_cols
    return result


def _detect_named_entities(question: str) -> list[str]:
    """Use OpenAI to extract named entities (proper nouns) from the question.

    Returns a list of strings; empty list if no entities found or LLM call fails.
    """
    try:
        client = get_client()
        resp = client.chat.completions.create(
            model=settings.openai_chat_model,
            temperature=0,
            max_tokens=128,
            messages=[
                {"role": "system", "content": _ENTITY_DETECT_SYSTEM},
                {"role": "user",   "content": question},
            ],
        )
        raw = _strip_fence((resp.choices[0].message.content or "").strip())
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(e) for e in parsed if e]
    except Exception as exc:  # noqa: BLE001
        logger.debug("[entity_resolution] entity detect failed: %s", exc)
    return []


def _resolve_entity_hints(
    question: str,
    schema: str,
    db_type_l: str,
    db_url: str,
) -> str:
    """Detect named entities in the question and search the DB for actual values.

    Strategy:
    1. Ask LLM to extract named entities from the question.
    2. For each entity, run `SELECT DISTINCT col FROM table WHERE col LIKE '%entity%' LIMIT 5`
       on every text column across all tables.
    3. Return a formatted hint string for the SQL generator to use.
    """
    entities = _detect_named_entities(question)
    if not entities:
        return ""

    logger.info("[entity_resolution] Detected entities: %s", entities)
    text_cols = _parse_text_columns(schema)
    if not text_cols:
        return ""

    hints: list[str] = []

    # Determine LIKE syntax by dialect
    dialect_like = "LIKE"  # standard

    for entity in entities[:5]:  # cap at 5 entities to limit DB round-trips
        found_any = False
        for table, cols in list(text_cols.items())[:20]:  # cap tables
            for col in cols[:4]:  # cap columns per table
                try:
                    if db_type_l == "mssql":
                        discovery_sql = (
                            f"SELECT DISTINCT TOP 5 [{col}] FROM [{table}] "
                            f"WHERE [{col}] LIKE N'%{entity}%'"
                        )
                    else:
                        discovery_sql = (
                            f"SELECT DISTINCT `{col}` FROM `{table}` "
                            f"WHERE `{col}` LIKE '%{entity}%' LIMIT 5"
                        )
                    rows = execute_query(db_type_l, db_url, discovery_sql)
                    if rows:
                        values = [str(list(r.values())[0]) for r in rows if r]
                        hint = (
                            f"- ค้นหา '{entity}' พบใน `{table}`.`{col}`: "
                            + ", ".join(f'"{v}"' for v in values[:5])
                        )
                        hints.append(hint)
                        found_any = True
                        logger.debug("[entity_resolution] %s", hint)
                        break  # found in this col, move to next table
                except Exception:  # noqa: BLE001
                    pass
            if found_any:
                break  # found entity in some table, stop scanning

    if not hints:
        return ""
    return (
        "\n--- Entity Resolution Hints (ค่าจริงใน Database) ---\n"
        + "\n".join(hints)
        + "\nให้ใช้ค่าที่พบข้างต้นใน WHERE clause แทนการเดา\n"
        + "--- สิ้นสุด Entity Hints ---"
    )


# ---------------------------------------------------------------------------
# Empty-result broadening helper
# ---------------------------------------------------------------------------

def _broaden_empty_query(
    bad_query: str,
    original_question: str,
    db_type_l: str,
    schema: str,
) -> str:
    """Ask OpenAI to rewrite a query that returned 0 rows with looser conditions."""
    client = get_client()
    dialect = _DIALECT_MAP.get(db_type_l, "SQL")
    schema_min = _minify_schema(schema)[:MAX_SCHEMA_CHARS]
    user_msg = (
        f"Original question: {original_question}\n\n"
        f"Query that returned 0 rows:\n{bad_query}\n\n"
        f"Schema ({dialect}):\n{schema_min}\n\n"
        "Rewrite with broader conditions so the user can see nearby data."
    )
    resp = get_client().chat.completions.create(
        model=settings.openai_chat_model,
        temperature=0,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": _BROADEN_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
    )
    return _strip_fence((resp.choices[0].message.content or "").strip())


def _try_vanna(    question: str,
    conn_id: str,
    db_type: str,
    db_url: str,
    schema: str,
) -> str | None:
    """Attempt SQL generation via Vanna.ai RAG.

    Auto-trains on the connection's schema on first use.
    Returns the SQL string on success, or None so the caller falls back
    to the direct OpenAI approach.
    """
    try:
        from app.vanna_engine import vanna_engine  # noqa: PLC0415

        if not vanna_engine.is_trained(conn_id):
            logger.info("[vanna] Auto-training on connection '%s' ...", conn_id)
            vanna_engine.train_on_connection(conn_id, db_type, db_url, schema)

        sql = vanna_engine.generate_sql(question)
        if sql:
            return _strip_fence(sql)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("[vanna] Unavailable — falling back to direct OpenAI: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Column validation helpers
# ---------------------------------------------------------------------------
_SCHEMA_ERROR_PATTERNS = re.compile(
    # column errors
    r"unknown column|column .* (not found|doesn't exist|does not exist)"
    r"|invalid column name|no such column|undefined column"
    r"|field .* (not found|does not exist)"
    r"|ER_BAD_FIELD_ERROR"
    # table errors
    r"|table .* doesn'?t exist|no such table|relation .* does not exist"
    r"|ER_NO_SUCH_TABLE|invalid object name",
    re.IGNORECASE,
)


def _is_column_error(error_msg: str) -> bool:
    """Return True when the DB error looks like a missing table/column problem."""
    return bool(_SCHEMA_ERROR_PATTERNS.search(error_msg))


def _parse_schema_columns(schema: str) -> dict[str, set[str]]:
    """Parse schema string produced by db_inspector into {table: {col, ...}}.

    Expected line format (from _sql_schema):
        Table `orders`: id (INTEGER), total (DECIMAL), ...  [PK: id]
    """
    result: dict[str, set[str]] = {}
    for line in schema.splitlines():
        m = re.match(r"Table\s+`([^`]+)`\s*:\s*(.+)", line)
        if not m:
            continue
        table_name = m.group(1).lower()
        col_part = m.group(2)
        # strip trailing PK annotation
        col_part = re.sub(r"\s*\[PK:.*?\]\s*$", "", col_part)
        cols: set[str] = set()
        for col_def in col_part.split(","):
            col_name = col_def.strip().split()[0] if col_def.strip() else ""
            if col_name:
                cols.add(col_name.lower())
        result[table_name] = cols
    return result


def _extract_table_alias_map(sql: str) -> dict[str, str]:
    """Return {alias_lower: table_lower} from FROM / JOIN clauses.

    Handles:
        FROM orders o
        FROM orders AS o
        JOIN customers c ON ...
        JOIN customers AS c ON ...
    Does NOT handle sub-queries or CTEs (they are intentionally skipped).
    """
    alias_map: dict[str, str] = {}
    # Match: (FROM|JOIN) <table> [AS] [alias]
    # The alias is optional; if absent, the table name itself is the key.
    pattern = re.compile(
        r"\b(?:FROM|JOIN)\s+`?(\w+)`?"          # table name
        r"(?:\s+(?:AS\s+)?`?(\w+)`?)?",         # optional alias
        re.IGNORECASE,
    )
    # keywords that should NOT be treated as aliases
    _SQL_KEYWORDS = frozenset({
        "on", "where", "set", "left", "right", "inner", "outer",
        "cross", "full", "join", "from", "select", "group", "order",
        "having", "limit", "union", "with", "as",
    })
    for m in pattern.finditer(sql):
        table = m.group(1).lower()
        alias = m.group(2).lower() if m.group(2) else None
        if alias and alias in _SQL_KEYWORDS:
            alias = None
        alias_map[table] = table            # table name always maps to itself
        if alias:
            alias_map[alias] = table
    return alias_map


def _check_schema_issues(sql: str, schema: str) -> str:
    """Return a description of table / column problems found, or empty string if clean.

    Checks (in order):
    1. Tables used in FROM / JOIN exist in the schema.
    2. Qualified references  alias.column — column exists in the resolved table.
    3. Unqualified column references — column exists in *any* schema table
       (best-effort; alias resolution is not guaranteed for unqualified names).
    """
    schema_cols = _parse_schema_columns(schema)
    if not schema_cols:
        return ""

    problems: list[str] = []
    alias_map = _extract_table_alias_map(sql)

    # 1. Validate table names
    for alias, table in alias_map.items():
        if table not in schema_cols:
            problems.append(f"table `{table}` does not exist in schema")

    all_valid_cols: set[str] = set()
    for cols in schema_cols.values():
        all_valid_cols.update(cols)

    # 2. Validate qualified alias.column references
    for m in re.finditer(r"\b(\w+)\.(\w+)\b", sql):
        qualifier = m.group(1).lower()
        col = m.group(2).lower()
        if col == "*":
            continue
        if qualifier in alias_map:
            actual_table = alias_map[qualifier]
            table_cols = schema_cols.get(actual_table, set())
            if table_cols and col not in table_cols:
                problems.append(
                    f"column `{m.group(2)}` does not exist in table `{actual_table}`"
                )
        # else: subquery alias or CTE — skip

    if problems:
        unique = list(dict.fromkeys(problems))
        return "Schema issues: " + "; ".join(unique[:12])
    return ""


# Keep old name as alias so existing call-sites don't break
_check_columns_against_schema = _check_schema_issues


_FIX_SYSTEM = """You are a SQL expert. A SQL query failed because it references tables or columns that do not exist in the schema.
Rewrite ONLY the SQL to use valid tables and columns from the schema provided.
Output raw SQL only — no explanation, no markdown fences.
ABSOLUTE RULES:
- Only SELECT statements allowed.
- Use ONLY table names listed in the Schema.
- Use ONLY column names that appear inside the matching table definition in the Schema.
- Keep the original query intent as close as possible.
- Add LIMIT 200 if not already present."""


def _fix_query_with_feedback(
    bad_query: str,
    error_or_issue: str,
    original_question: str,
    db_type_l: str,
    schema: str,
) -> str:
    """Ask OpenAI to rewrite *bad_query* after a column error.

    The schema passed in should already be the focused + minified version so
    this extra LLM call stays within the token budget.
    """
    client = get_client()
    dialect = _DIALECT_MAP.get(db_type_l, "SQL")
    # Minify the schema one more time in case a raw schema was passed in.
    schema_for_fix = _minify_schema(schema)[:MAX_SCHEMA_CHARS]
    user_msg = (
        f"Original question: {original_question}\n\n"
        f"Failed query:\n{bad_query}\n\n"
        f"Error / issue:\n{error_or_issue}\n\n"
        f"Schema ({dialect}):\n{schema_for_fix}\n\n"
        "Rewrite the query using only columns that exist in the schema above."
    )
    try:
        resp = client.chat.completions.create(
            model=settings.openai_chat_model,
            temperature=0,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": _FIX_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
        )
        fixed = (resp.choices[0].message.content or "").strip()
        return _strip_fence(fixed)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[text_to_sql] _fix_query_with_feedback failed: %s", exc)
        return bad_query  # return original; outer code will raise


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
    """Raise ValueError if *sql* is not a pure SELECT query."""
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
        if re.search(r'(?<!")"' + re.escape(op) + r'"', query_text):
            raise ValueError(
                f"การดำเนินการ '{op}' ไม่ได้รับอนุญาต — อนุญาตเฉพาะการสืบค้นข้อมูลเท่านั้น"
            )
    logger.debug("[security] MongoDB validation passed")


# Singleton used by main.py
text_to_query_engine = TextToQueryEngine()
