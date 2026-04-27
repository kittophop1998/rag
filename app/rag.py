"""
The RAG chain.

The chain:

* receives a user question,
* retrieves the top-K most relevant snippets from the FAISS vector store,
* prompts the LLM to answer **only** using that context,
* returns ``"ไม่พบข้อมูลในเอกสารครับ"`` when the answer is not in the context.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import AsyncGenerator, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

from app.config import settings
from app.indexer import build_or_load_vectorstore

logger = logging.getLogger(__name__)

NOT_FOUND_REPLY = "ไม่พบข้อมูลในเอกสารครับ"

SYSTEM_PROMPT = """คุณคือผู้ช่วยตอบคำถามภายในของบริษัท
ตอบเป็นภาษาไทยที่สุภาพ กระชับ และเข้าใจง่าย

กฎสำคัญ:
1. ใช้ข้อมูลจาก "เอกสารอ้างอิง" ด้านล่างเท่านั้น ห้ามเดาหรือใช้ความรู้ภายนอก
   แหล่งข้อมูลอาจเป็นได้ทั้งไฟล์ PDF และเนื้อหาจากเว็บไซต์
2. ถ้าข้อมูลในเอกสารอ้างอิงไม่เพียงพอที่จะตอบ ให้ตอบกลับเพียงประโยคเดียวว่า:
   "{not_found}"
3. ถ้ามีข้อมูล ให้สรุปคำตอบให้ชัดเจน และอ้างอิงแหล่งที่มาในวงเล็บท้ายประโยค
   - ถ้าเป็นไฟล์ PDF เช่น (ที่มา: hr_policy.pdf)
   - ถ้าเป็นเว็บไซต์ เช่น (ที่มา: https://example.com/page)
"""

USER_PROMPT = """คำถาม:
{question}

เอกสารอ้างอิง:
{context}
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
        source = d.metadata.get("source", "unknown")
        page = d.metadata.get("page")
        header = f"[{i}] ที่มา: {source}" + (f" (หน้า {page + 1})" if isinstance(page, int) else "")
        parts.append(f"{header}\n{d.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


@dataclass
class RAGAnswer:
    """Response wrapper containing the answer and the citations used."""

    answer: str
    sources: List[dict]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class RAGEngine:
    """Encapsulates the vector store + LLM + prompt as a single callable."""

    def __init__(self) -> None:
        self._vectorstore: Optional[FAISS] = None
        self._llm: Optional[ChatOpenAI] = None
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT.replace("{not_found}", NOT_FOUND_REPLY)),
                ("human", USER_PROMPT),
            ]
        )

    # -- lazy initialisation -------------------------------------------------
    @property
    def vectorstore(self) -> FAISS:
        if self._vectorstore is None:
            self._vectorstore = build_or_load_vectorstore()
        return self._vectorstore

    @property
    def llm(self) -> ChatOpenAI:
        if self._llm is None:
            if not settings.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY is not configured.")
            self._llm = ChatOpenAI(
                model=settings.openai_chat_model,
                api_key=settings.openai_api_key,
                temperature=0.2,
            )
        return self._llm

    def reload(self) -> None:
        """Force the next call to re-load the FAISS index from disk."""
        self._vectorstore = None

    # -- streaming entry point -----------------------------------------------
    async def ask_stream(self, question: str) -> AsyncGenerator[str, None]:
        """Async generator that yields Server-Sent Event strings.

        Event types emitted:
        * ``{"type":"token","content":"..."}``  – one LLM token at a time
        * ``{"type":"sources","content":[...]}`` – retrieved source list
        * ``{"type":"error","content":"..."}``   – error message
        * ``[DONE]``                              – stream finished
        """
        question = (question or "").strip()
        if not question:
            yield f'data: {json.dumps({"type": "error", "content": "คำถามว่างเปล่า"})}\n\n'
            return

        try:
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": settings.top_k})
            docs: List[Document] = retriever.invoke(question)

            stream_llm = ChatOpenAI(
                model=settings.openai_chat_model,
                api_key=settings.openai_api_key,
                temperature=0.2,
                streaming=True,
            )
            chain = (
                {
                    "context": RunnableLambda(lambda _: _format_docs(docs)),
                    "question": RunnablePassthrough(),
                }
                | self._prompt
                | stream_llm
                | StrOutputParser()
            )

            async for chunk in chain.astream(question):
                if chunk:
                    yield f'data: {json.dumps({"type": "token", "content": chunk})}\n\n'

            sources = [
                {
                    "source": d.metadata.get("source", "unknown"),
                    "page": d.metadata.get("page"),
                    "snippet": d.page_content[:300].strip(),
                }
                for d in docs
            ]
            yield f'data: {json.dumps({"type": "sources", "content": sources})}\n\n'
            yield "data: [DONE]\n\n"

        except FileNotFoundError:
            yield f'data: {json.dumps({"type": "error", "content": "ยังไม่มี FAISS index — กรุณากด Rebuild Index ก่อนครับ"})}\n\n'
        except RuntimeError as exc:
            yield f'data: {json.dumps({"type": "error", "content": str(exc)})}\n\n'
        except Exception as exc:  # noqa: BLE001
            logger.exception("ask_stream failed: %s", exc)
            yield f'data: {json.dumps({"type": "error", "content": "ระบบมีปัญหาชั่วคราว กรุณาลองใหม่อีกครั้ง"})}\n\n'

    # -- main entry point ----------------------------------------------------
    def ask(self, question: str) -> RAGAnswer:
        question = (question or "").strip()
        if not question:
            return RAGAnswer(answer=NOT_FOUND_REPLY, sources=[])

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": settings.top_k})
        docs: List[Document] = retriever.invoke(question)

        chain = (
            {
                "context": RunnableLambda(lambda _: _format_docs(docs)),
                "question": RunnablePassthrough(),
            }
            | self._prompt
            | self.llm
            | StrOutputParser()
        )

        try:
            answer = chain.invoke(question).strip()
        except Exception as exc:  # noqa: BLE001
            logger.exception("RAG chain failed: %s", exc)
            return RAGAnswer(
                answer="ขออภัยครับ ระบบมีปัญหาชั่วคราว กรุณาลองใหม่อีกครั้ง",
                sources=[],
            )

        sources = [
            {
                "source": d.metadata.get("source", "unknown"),
                "page": d.metadata.get("page"),
                "snippet": d.page_content[:240].strip(),
            }
            for d in docs
        ]
        return RAGAnswer(answer=answer or NOT_FOUND_REPLY, sources=sources)


rag_engine = RAGEngine()
