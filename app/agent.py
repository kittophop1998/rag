"""Agentic Workflow Engine.

The agent receives a user question and autonomously decides:
1. **Intent classification** – smalltalk, knowledge-base search, or DB query.
2. **Multi-source retrieval** – searches document RAG + all indexed DB stores.
3. **Reflection** – if no results found, tries a broader search or suggests alternatives.
4. **Generation** – synthesises a clear Thai-language answer with citations.

The agent uses the official OpenAI Python SDK directly (not LangChain wrappers)
for all LLM calls, while keeping LangChain/ChromaDB for vector retrieval.

Public API
----------
agent_engine.ask_stream(question) → AsyncGenerator[str, None]
    Yields SSE-formatted strings (same protocol as rag.py ask_stream).
"""

from __future__ import annotations

import json
import logging
import re
from typing import AsyncGenerator, List

from langchain_core.documents import Document

from app.config import settings
from app.openai_client import get_async_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NOT_FOUND_REPLY = "ไม่พบข้อมูลที่ตรงกับคำถาม ลองพิมพ์ใหม่ให้เฉพาะเจาะจงขึ้นอีกนิดนะครับ"
SMALLTALK_FALLBACK = "ได้เลยครับ ผมพร้อมคุยด้วยเสมอ มีอะไรอยากคุยหรืออยากให้ช่วยเพิ่มเติมไหมครับ"
# NOTE: Thai text / OCR-heavy PDFs typically score 0.05–0.15; keep threshold
#       low enough to surface relevant chunks while still filtering random noise.
MIN_RELEVANCE_SCORE_DOC = 0.05   # Documents / URL sources
MIN_RELEVANCE_SCORE_DB  = 0.05   # DB data rows (structured text; lower threshold)

SMALLTALK_PATTERNS = (
    r"^(hi|hello|hey)\b",
    r"\bhow are you\b",
    r"^(สวัสดี|หวัดดี|ดีจ้า|ดีครับ|ดีค่ะ)",
    r"(เป็นไงบ้าง|เป็นอย่างไรบ้าง|เป็นยังไงบ้าง)",
    r"(ขอบคุณ|thank you|ขอบคุณมาก)",
    r"(คุยเล่น|ชวนคุย|ทักทาย)",
)


# ---------------------------------------------------------------------------
# Intent helpers
# ---------------------------------------------------------------------------
def _is_smalltalk(text: str) -> bool:
    t = (text or "").strip().lower()
    return bool(t) and any(re.search(p, t) for p in SMALLTALK_PATTERNS)


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------
def _format_docs(docs: List[Document]) -> str:
    if not docs:
        return "(ไม่มีข้อมูลที่เกี่ยวข้อง)"
    parts = []
    for i, d in enumerate(docs, 1):
        if d.metadata.get("type") == "db_data":
            db_name = d.metadata.get("db_name", "?")
            table   = d.metadata.get("table", "?")
            group   = d.metadata.get("group", "")
            gstr    = f" / กลุ่ม: {group}" if group else ""
            header  = f"[{i}] ที่มา: DB {db_name} / ตาราง {table}{gstr}"
        else:
            source = d.metadata.get("source", "unknown")
            page   = d.metadata.get("page")
            header = f"[{i}] ที่มา: {source}" + (f" (หน้า {page + 1})" if isinstance(page, int) else "")
        parts.append(f"{header}\n{d.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


def _retrieve_all_docs(question: str) -> List[Document]:
    """Multi-source retrieval: document RAG + every indexed DB store.

    Returns an empty list (not raises) when no store is available.
    Applies separate relevance thresholds for documents vs DB rows.
    """
    from app.chat_store import get_group_states     # noqa: PLC0415
    from app.db_indexer import list_indexed_conn_ids, load_db_store  # noqa: PLC0415
    from app.indexer import build_or_load_vectorstore  # noqa: PLC0415

    all_docs: List[Document] = []

    # 1. Document / URL store
    try:
        vs = build_or_load_vectorstore()
        scored = vs.similarity_search_with_relevance_scores(question, k=settings.top_k)
        passed = [doc for doc, score in scored if score >= MIN_RELEVANCE_SCORE_DOC]
        # Fallback: if nothing passed threshold (e.g. short Thai queries with
        # negative cosine scores), include all top-K so context is not empty.
        all_docs.extend(passed if passed else [doc for doc, _ in scored])
    except FileNotFoundError:
        pass
    except Exception as exc:  # noqa: BLE001
        logger.warning("Agent: doc store retrieval failed: %s", exc)

    # 2. Indexed DB stores (all connections)
    for conn_id in list_indexed_conn_ids():
        try:
            db_vs = load_db_store(conn_id)
            if not db_vs:
                continue
            group_states   = get_group_states(conn_id)
            disabled       = {g for g, r in group_states.items() if not r.get("enabled", True)}
            try:
                scored = db_vs.similarity_search_with_relevance_scores(question, k=settings.top_k)
                db_docs = [doc for doc, score in scored if score >= MIN_RELEVANCE_SCORE_DB]
                if not db_docs:
                    db_docs = [doc for doc, _ in scored]
            except Exception:
                db_docs = db_vs.similarity_search(question, k=settings.top_k)
            if disabled:
                db_docs = [d for d in db_docs if d.metadata.get("group") not in disabled]
            all_docs.extend(db_docs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Agent: DB store %s failed: %s", conn_id, exc)

    return all_docs


# ---------------------------------------------------------------------------
# Prompts (as plain strings for the openai SDK)
# ---------------------------------------------------------------------------
_SMALLTALK_SYSTEM = (
    "คุณคือผู้ช่วยแชตภาษาไทยที่เป็นกันเอง สุภาพ และตอบสั้นกระชับ\n"
    "- ตอบเหมือนคุยกับคนทั่วไปได้เลย ไม่ต้องอ้างอิงเอกสาร/ฐานข้อมูล\n"
    "- ถ้าอีกฝ่ายยังไม่ระบุโจทย์งาน ให้ชวนถามต่อแบบธรรมชาติ"
)

_RAG_SYSTEM = """คุณคือผู้ช่วยตอบคำถามภายในของบริษัท ตอบเป็นภาษาไทยที่สุภาพ กระชับ และเข้าใจง่าย

กฎสำคัญ:
1. ใช้ข้อมูลจาก "เอกสารอ้างอิง" ด้านล่างเท่านั้น ห้ามเดาหรือใช้ความรู้ภายนอก
   แหล่งข้อมูลอาจเป็นได้ทั้งไฟล์ PDF เนื้อหาจากเว็บไซต์ และข้อมูลจากฐานข้อมูล
2. ถ้าข้อมูลในเอกสารอ้างอิงไม่เพียงพอ ให้ตอบกลับเพียงประโยคเดียวว่า:
   "ไม่พบข้อมูลที่ตรงกับคำถาม ลองพิมพ์ใหม่ให้เฉพาะเจาะจงขึ้นอีกนิดนะครับ"
3. ถ้ามีข้อมูล ให้สรุปคำตอบให้ชัดเจน และอ้างอิงแหล่งที่มาในวงเล็บท้ายประโยค
   - ถ้าเป็นไฟล์: (ที่มา: ชื่อไฟล์)
   - ถ้าเป็นเว็บ: (ที่มา: URL)
   - ถ้าเป็น DB: (ที่มา: DB ชื่อ / ตาราง)

รูปแบบการตอบ:
- ตอบเป็น Markdown
- ถ้ามี URL รูปภาพในข้อมูล ให้แสดงด้วย syntax ![ชื่อรูป](url)"""

_PLAN_SYSTEM = """คุณคือ AI ที่ช่วยวางแผนการตอบคำถาม

จงวิเคราะห์คำถามและตอบเป็น JSON ตามรูปแบบนี้เท่านั้น:
{
  "intent": "smalltalk" | "knowledge_search" | "no_context_needed",
  "reason": "เหตุผลสั้นๆ"
}

- "smalltalk": ทักทาย คุยเล่น ขอบคุณ ไม่ใช่คำถามด้านงาน
- "knowledge_search": คำถามที่ต้องการข้อมูลจากเอกสารหรือฐานข้อมูล
- "no_context_needed": คำถามทั่วไปที่ตอบได้โดยไม่ต้องค้นหา (เช่น คำนวณ, แปลภาษา)"""


# ---------------------------------------------------------------------------
# Agent Engine
# ---------------------------------------------------------------------------
class AgentEngine:
    """Agentic RAG engine using the OpenAI Python SDK directly."""

    # ── Planning ─────────────────────────────────────────────────────────
    def _classify_intent(self, question: str) -> str:
        """Return 'smalltalk', 'knowledge_search', or 'no_context_needed'."""
        if _is_smalltalk(question):
            return "smalltalk"
        try:
            client = get_async_client()  # type: ignore[assignment]
            # Use sync client here (called from sync context if needed)
            from app.openai_client import get_client  # noqa: PLC0415
            resp = get_client().chat.completions.create(
                model=settings.openai_chat_model,
                temperature=0,
                max_tokens=80,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _PLAN_SYSTEM},
                    {"role": "user", "content": question},
                ],
            )
            data = json.loads(resp.choices[0].message.content or "{}")
            return data.get("intent", "knowledge_search")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Agent: intent classification failed (%s) — defaulting to knowledge_search", exc)
            return "knowledge_search"

    # ── Streaming generation ──────────────────────────────────────────────
    async def _stream_smalltalk(self, question: str) -> AsyncGenerator[str, None]:
        try:
            client = get_async_client()
            stream = await client.chat.completions.create(
                model=settings.openai_chat_model,
                temperature=0.5,
                stream=True,
                messages=[
                    {"role": "system", "content": _SMALLTALK_SYSTEM},
                    {"role": "user",   "content": question},
                ],
            )
            async for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                if token:
                    yield f'data: {json.dumps({"type": "token", "content": token})}\n\n'
            yield f'data: {json.dumps({"type": "sources", "content": []})}\n\n'
            yield "data: [DONE]\n\n"
        except Exception as exc:  # noqa: BLE001
            logger.warning("Agent: smalltalk stream failed: %s", exc)
            yield f'data: {json.dumps({"type": "token", "content": SMALLTALK_FALLBACK})}\n\n'
            yield f'data: {json.dumps({"type": "sources", "content": []})}\n\n'
            yield "data: [DONE]\n\n"

    async def _stream_no_context(self, question: str) -> AsyncGenerator[str, None]:
        """Direct LLM answer for questions that don't need retrieval."""
        try:
            client = get_async_client()
            stream = await client.chat.completions.create(
                model=settings.openai_chat_model,
                temperature=0.3,
                stream=True,
                messages=[
                    {"role": "system", "content": "คุณคือผู้ช่วย AI ที่ตอบคำถามเป็นภาษาไทย ตอบกระชับและชัดเจน ใช้ Markdown"},
                    {"role": "user",   "content": question},
                ],
            )
            async for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                if token:
                    yield f'data: {json.dumps({"type": "token", "content": token})}\n\n'
            yield f'data: {json.dumps({"type": "sources", "content": []})}\n\n'
            yield "data: [DONE]\n\n"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Agent: no-context stream failed: %s", exc)
            yield f'data: {json.dumps({"type": "error", "content": str(exc)})}\n\n'

    async def _stream_rag(self, question: str, docs: List[Document]) -> AsyncGenerator[str, None]:
        """Stream a RAG answer given pre-retrieved docs."""
        context = _format_docs(docs)
        sources = [
            {
                "source":  d.metadata.get("source", "unknown"),
                "page":    d.metadata.get("page"),
                "snippet": d.page_content[:300].strip(),
                "type":    d.metadata.get("type", "document"),
                "group":   d.metadata.get("group"),
            }
            for d in docs
        ]

        try:
            client = get_async_client()
            stream = await client.chat.completions.create(
                model=settings.openai_chat_model,
                temperature=0.2,
                stream=True,
                messages=[
                    {"role": "system", "content": _RAG_SYSTEM},
                    {
                        "role": "user",
                        "content": f"คำถาม:\n{question}\n\nเอกสารอ้างอิง:\n{context}",
                    },
                ],
            )
            async for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                if token:
                    yield f'data: {json.dumps({"type": "token", "content": token})}\n\n'
            yield f'data: {json.dumps({"type": "sources", "content": sources})}\n\n'
            yield "data: [DONE]\n\n"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Agent: RAG stream failed: %s", exc)
            yield f'data: {json.dumps({"type": "error", "content": str(exc)})}\n\n'

    # ── Main entry point ──────────────────────────────────────────────────
    async def ask_stream(self, question: str) -> AsyncGenerator[str, None]:
        """Agentic streaming endpoint.

        Phases:
        1. Fast smalltalk guard (regex, no LLM call)
        2. Intent classification via LLM
        3. Retrieval from all available stores
        4. Reflection: if empty retrieval, attempt broader search
        5. Streaming generation
        """
        question = (question or "").strip()
        if not question:
            yield f'data: {json.dumps({"type": "error", "content": "คำถามว่างเปล่า"})}\n\n'
            return

        # Phase 1: quick smalltalk guard
        if _is_smalltalk(question):
            async for event in self._stream_smalltalk(question):
                yield event
            return

        # Phase 2: intent classification (async wrapper around sync call)
        import asyncio  # noqa: PLC0415
        try:
            intent = await asyncio.get_event_loop().run_in_executor(
                None, self._classify_intent, question
            )
        except Exception:  # noqa: BLE001
            intent = "knowledge_search"

        logger.info("Agent: intent=%s for question='%.60s'", intent, question)

        if intent == "smalltalk":
            async for event in self._stream_smalltalk(question):
                yield event
            return

        if intent == "no_context_needed":
            async for event in self._stream_no_context(question):
                yield event
            return

        # Phase 3: knowledge_search — retrieve from all stores
        try:
            docs = await asyncio.get_event_loop().run_in_executor(
                None, _retrieve_all_docs, question
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Agent: retrieval error: %s", exc)
            docs = []

        # Phase 4: reflection — if nothing found, try a shorter query
        if not docs and len(question) > 20:
            logger.info("Agent: reflection — retrying with shorter query")
            try:
                short_q = await asyncio.get_event_loop().run_in_executor(
                    None, _retrieve_all_docs, question[:60]
                )
                docs = short_q
            except Exception:  # noqa: BLE001
                pass

        if not docs:
            yield f'data: {json.dumps({"type": "token", "content": NOT_FOUND_REPLY})}\n\n'
            yield f'data: {json.dumps({"type": "sources", "content": []})}\n\n'
            yield "data: [DONE]\n\n"
            return

        # Phase 5: generate streaming answer
        async for event in self._stream_rag(question, docs):
            yield event


agent_engine = AgentEngine()
