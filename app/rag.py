"""
The RAG chain.

Uses the official OpenAI Python SDK for all LLM calls and LangChain/ChromaDB
for vector retrieval only.

Flow:
* Receive a user question
* Retrieve top-K snippets from all ChromaDB stores (documents + DB indexes)
* Prompt the LLM to answer ONLY using that context
* Return "ไม่พบข้อมูล..." when the answer is not in the context

Vector store: ChromaDB  (``chroma_base_dir/rag/``, collection ``rag_documents``)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import AsyncGenerator, List, Optional

from langchain_chroma import Chroma

from langchain_core.documents import Document

from app.config import settings
from app.indexer import build_or_load_vectorstore
from app.openai_client import get_async_client, get_client

logger = logging.getLogger(__name__)

NOT_FOUND_REPLY = "ไม่พบข้อมูลที่ตรงกับคำถาม ลองพิมพ์ใหม่ให้เฉพาะเจาะจงขึ้นอีกนิดนะครับ"
SMALLTALK_REPLY_FALLBACK = "ได้เลยครับ ผมพร้อมคุยด้วยเสมอ มีอะไรอยากคุยหรืออยากให้ช่วยเพิ่มเติมไหมครับ"

# Separate thresholds: DB rows are structured text and tend to have lower
# cosine similarity even when highly relevant.
# NOTE: Thai text / OCR-heavy PDFs typically score 0.05–0.15; keep threshold
#       low enough to surface relevant chunks while still filtering random noise.
MIN_RELEVANCE_SCORE_DOC = 0.05
MIN_RELEVANCE_SCORE_DB  = 0.05

SMALLTALK_PATTERNS = (
    r"^(hi|hello|hey)\b",
    r"\bhow are you\b",
    r"^(สวัสดี|หวัดดี|ดีจ้า|ดีครับ|ดีค่ะ)",
    r"(เป็นไงบ้าง|เป็นอย่างไรบ้าง|เป็นยังไงบ้าง)",
    r"(ขอบคุณ|thank you)",
    r"(คุยเล่น|ชวนคุย|ทักทาย)",
)

SMALLTALK_SYSTEM = (
    "คุณคือผู้ช่วยแชตภาษาไทยที่เป็นกันเอง สุภาพ และตอบสั้นกระชับ\n"
    "- ตอบเหมือนคุยกับคนทั่วไปได้เลย ไม่ต้องอ้างอิงเอกสาร/ฐานข้อมูล\n"
    "- ถ้าอีกฝ่ายยังไม่ระบุโจทย์งาน ให้ชวนถามต่อแบบธรรมชาติ"
)

RAG_SYSTEM = """คุณคือผู้ช่วยตอบคำถามภายในของบริษัท
ตอบเป็นภาษาไทยที่สุภาพ กระชับ และเข้าใจง่าย

กฎสำคัญ:
1. ใช้ข้อมูลจาก "เอกสารอ้างอิง" ด้านล่างเท่านั้น ห้ามเดาหรือใช้ความรู้ภายนอก
   แหล่งข้อมูลอาจเป็นได้ทั้งไฟล์ PDF เนื้อหาจากเว็บไซต์ และข้อมูลจากฐานข้อมูล
2. ถ้าข้อมูลในเอกสารอ้างอิงไม่เพียงพอที่จะตอบ ให้ตอบกลับเพียงประโยคเดียวว่า:
   "ไม่พบข้อมูลที่ตรงกับคำถาม ลองพิมพ์ใหม่ให้เฉพาะเจาะจงขึ้นอีกนิดนะครับ"
3. ถ้ามีข้อมูล ให้สรุปคำตอบให้ชัดเจน และอ้างอิงแหล่งที่มาในวงเล็บท้ายประโยค
   - ถ้าเป็นไฟล์ PDF เช่น (ที่มา: hr_policy.pdf)
   - ถ้าเป็นเว็บไซต์ เช่น (ที่มา: https://example.com/page)
   - ถ้าเป็นข้อมูลจากฐานข้อมูล เช่น (ที่มา: DB ชื่อบริษัท / ตาราง orders)

รูปแบบการตอบ:
- ตอบเป็น Markdown
- ถ้ามี URL รูปภาพในข้อมูล ให้แสดงด้วย syntax ![ชื่อรูป](url)
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _format_docs(docs: List[Document]) -> str:
    """Render retrieved documents as a numbered context block."""
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


def _is_smalltalk(question: str) -> bool:
    text = (question or "").strip().lower()
    if not text:
        return False
    return any(re.search(p, text) for p in SMALLTALK_PATTERNS)


@dataclass
class RAGAnswer:
    """Response wrapper containing the answer and the citations used."""
    answer: str
    sources: List[dict]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class RAGEngine:
    """Encapsulates the vector store retrieval + OpenAI LLM as a single callable."""

    def __init__(self) -> None:
        self._vectorstore: Optional[Chroma] = None

    # -- lazy initialisation -------------------------------------------------
    @property
    def vectorstore(self) -> Chroma:
        if self._vectorstore is None:
            self._vectorstore = build_or_load_vectorstore()
        return self._vectorstore

    def reload(self) -> None:
        """Release the current Chroma client so the next call re-loads from disk.

        Runs a GC cycle to close the SQLite connection before the directory is
        wiped (avoids SQLITE_READONLY_DBMOVED / code 1032 on ChromaDB).
        """
        import gc
        self._vectorstore = None
        gc.collect()

    # -- multi-store retrieval -----------------------------------------------
    def _search_store(
        self,
        store: Chroma,
        question: str,
        threshold: float = MIN_RELEVANCE_SCORE_DOC,
    ) -> List[Document]:
        """Search with a relevance threshold to reduce unrelated context.

        Falls back to un-filtered top-K when ALL scores are below threshold
        (common for short Thai queries where cosine similarity can be negative).
        In that case the LLM prompt already guards against hallucination.
        """
        try:
            scored = store.similarity_search_with_relevance_scores(
                question, k=settings.top_k
            )
            filtered = [doc for doc, score in scored if score >= threshold]
            logger.debug(
                "Store search: %d/%d docs passed threshold %.2f",
                len(filtered), len(scored), threshold,
            )
            # If nothing passed the threshold, fall back to raw top-K so that
            # short/ambiguous Thai queries still get context.
            if not filtered and scored:
                logger.debug(
                    "No docs passed threshold %.2f — returning raw top-%d results",
                    threshold, len(scored),
                )
                return [doc for doc, _ in scored]
            return filtered
        except Exception:  # noqa: BLE001
            return store.similarity_search(question, k=settings.top_k)

    def _retrieve_docs(self, question: str) -> List[Document]:
        """Retrieve relevant documents from all available stores.

        Sources searched (in order):
        1. Document RAG vector store (PDFs + crawled URLs)
        2. Every indexed DB connection under chroma/db/

        Uses a LOWER threshold (MIN_RELEVANCE_SCORE_DB) for DB row documents
        because structured row text typically has lower cosine similarity.
        """
        from app.chat_store import get_group_states  # noqa: PLC0415
        from app.db_indexer import list_indexed_conn_ids, load_db_store  # noqa: PLC0415

        all_docs: List[Document] = []
        has_any_store = False

        # 1. Document RAG store
        try:
            vs = self.vectorstore
            has_any_store = True
            all_docs.extend(
                self._search_store(vs, question, threshold=MIN_RELEVANCE_SCORE_DOC)
            )
        except FileNotFoundError:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.warning("Doc store retrieval failed: %s", exc)

        # 2. All indexed DB stores
        for conn_id in list_indexed_conn_ids():
            try:
                db_vs = load_db_store(conn_id)
                if db_vs:
                    has_any_store = True
                    group_states    = get_group_states(conn_id)
                    disabled_groups = {
                        g for g, row in group_states.items() if not row.get("enabled", True)
                    }
                    db_docs = self._search_store(
                        db_vs, question, threshold=MIN_RELEVANCE_SCORE_DB
                    )
                    if disabled_groups:
                        db_docs = [
                            d for d in db_docs
                            if d.metadata.get("group") not in disabled_groups
                        ]
                    logger.info(
                        "DB store %s: %d docs retrieved (disabled groups: %s)",
                        conn_id, len(db_docs), disabled_groups or "none",
                    )
                    all_docs.extend(db_docs)
            except Exception as exc:  # noqa: BLE001
                logger.warning("DB store %s retrieval failed: %s", conn_id, exc)

        if not all_docs and not has_any_store:
            raise FileNotFoundError(
                "ยังไม่มี index ใดเลย — กรุณาอัปโหลดเอกสาร PDF หรือ Index ฐานข้อมูลก่อนครับ"
            )

        return all_docs

    # -- streaming entry point -----------------------------------------------
    async def ask_stream(self, question: str) -> AsyncGenerator[str, None]:
        """Async generator that yields Server-Sent Event strings.

        Uses the OpenAI Python SDK directly for streaming completions.

        Event types emitted:
        * ``{"type":"token","content":"..."}``  – one LLM token
        * ``{"type":"sources","content":[...]}`` – retrieved source list
        * ``{"type":"error","content":"..."}``   – error message
        * ``[DONE]``                              – stream finished
        """
        question = (question or "").strip()
        if not question:
            yield f'data: {json.dumps({"type": "error", "content": "คำถามว่างเปล่า"})}\n\n'
            return

        if not settings.openai_api_key:
            yield f'data: {json.dumps({"type": "error", "content": "OPENAI_API_KEY ยังไม่ได้ตั้งค่า"})}\n\n'
            return

        client = get_async_client()

        # Smalltalk path
        if _is_smalltalk(question):
            try:
                stream = await client.chat.completions.create(
                    model=settings.openai_chat_model,
                    temperature=0.5,
                    stream=True,
                    messages=[
                        {"role": "system", "content": SMALLTALK_SYSTEM},
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
                logger.warning("smalltalk stream failed: %s", exc)
                yield f'data: {json.dumps({"type": "token", "content": SMALLTALK_REPLY_FALLBACK})}\n\n'
                yield f'data: {json.dumps({"type": "sources", "content": []})}\n\n'
                yield "data: [DONE]\n\n"
            return

        # RAG path
        try:
            docs: List[Document] = self._retrieve_docs(question)
            if not docs:
                yield f'data: {json.dumps({"type": "token", "content": NOT_FOUND_REPLY})}\n\n'
                yield f'data: {json.dumps({"type": "sources", "content": []})}\n\n'
                yield "data: [DONE]\n\n"
                return

            context = _format_docs(docs)
            stream = await client.chat.completions.create(
                model=settings.openai_chat_model,
                temperature=0.2,
                stream=True,
                messages=[
                    {"role": "system", "content": RAG_SYSTEM},
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
            yield f'data: {json.dumps({"type": "sources", "content": sources})}\n\n'
            yield "data: [DONE]\n\n"

        except FileNotFoundError as exc:
            yield f'data: {json.dumps({"type": "error", "content": str(exc)})}\n\n'
        except RuntimeError as exc:
            yield f'data: {json.dumps({"type": "error", "content": str(exc)})}\n\n'
        except Exception as exc:  # noqa: BLE001
            logger.exception("ask_stream failed: %s", exc)
            yield f'data: {json.dumps({"type": "error", "content": "ระบบมีปัญหาชั่วคราว กรุณาลองใหม่อีกครั้ง"})}\n\n'

    # -- sync entry point (kept for backward compatibility) -------------------
    def ask(self, question: str) -> RAGAnswer:
        """Synchronous RAG query using the OpenAI Python SDK."""
        question = (question or "").strip()
        if not question:
            return RAGAnswer(answer=NOT_FOUND_REPLY, sources=[])

        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        client = get_client()

        if _is_smalltalk(question):
            try:
                resp = client.chat.completions.create(
                    model=settings.openai_chat_model,
                    temperature=0.5,
                    messages=[
                        {"role": "system", "content": SMALLTALK_SYSTEM},
                        {"role": "user",   "content": question},
                    ],
                )
                answer = (resp.choices[0].message.content or "").strip()
                return RAGAnswer(answer=answer or SMALLTALK_REPLY_FALLBACK, sources=[])
            except Exception as exc:  # noqa: BLE001
                logger.warning("smalltalk reply failed: %s", exc)
                return RAGAnswer(answer=SMALLTALK_REPLY_FALLBACK, sources=[])

        docs: List[Document] = self._retrieve_docs(question)
        if not docs:
            return RAGAnswer(answer=NOT_FOUND_REPLY, sources=[])

        context = _format_docs(docs)
        try:
            resp = client.chat.completions.create(
                model=settings.openai_chat_model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": RAG_SYSTEM},
                    {
                        "role": "user",
                        "content": f"คำถาม:\n{question}\n\nเอกสารอ้างอิง:\n{context}",
                    },
                ],
            )
            answer = (resp.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.exception("RAG chain failed: %s", exc)
            return RAGAnswer(
                answer="ขออภัยครับ ระบบมีปัญหาชั่วคราว กรุณาลองใหม่อีกครั้ง",
                sources=[],
            )

        sources = [
            {
                "source":  d.metadata.get("source", "unknown"),
                "page":    d.metadata.get("page"),
                "snippet": d.page_content[:240].strip(),
                "type":    d.metadata.get("type", "document"),
                "group":   d.metadata.get("group"),
            }
            for d in docs
        ]
        return RAGAnswer(answer=answer or NOT_FOUND_REPLY, sources=sources)


rag_engine = RAGEngine()
