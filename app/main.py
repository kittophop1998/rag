"""
FastAPI entry point.

Endpoints:
* ``GET  /``         -> Chat web UI (static HTML).
* ``POST /api/chat`` -> Ask a question, return JSON answer + sources.
* ``POST /api/reindex`` -> Rebuild the FAISS index from ./documents.
* ``POST /webhook``  -> LINE Messaging API webhook.
* ``GET  /healthz``  -> Liveness probe.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import settings
from app.indexer import build_vectorstore
from app.line_webhook import router as line_router
from app.rag import rag_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("rag")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Company RAG",
    description="ระบบถาม-ตอบเอกสารภายในบริษัท ด้วย FastAPI + LangChain + OpenAI + FAISS + LINE",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Register the LINE webhook (POST /webhook).
app.include_router(line_router)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question in Thai or English")


class SourceItem(BaseModel):
    source: str
    page: Optional[int] = None
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem] = []


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    """Serve the chat UI."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/healthz", include_in_schema=False)
async def healthz():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
async def api_chat(req: ChatRequest) -> ChatResponse:
    """Ask the RAG a question and get back an answer plus its citations."""
    try:
        result = rag_engine.ask(req.question)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "ยังไม่มี FAISS index — กรุณาอัปโหลดไฟล์ PDF ลงในโฟลเดอร์ "
                f"{settings.documents_dir} แล้วเรียก POST /api/reindex"
            ),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatResponse(answer=result.answer, sources=result.sources)


@app.get("/api/chat/stream")
async def api_chat_stream(q: str = "") -> StreamingResponse:
    """Stream a RAG answer token-by-token via Server-Sent Events (GET ?q=...)."""
    if not q.strip():

        async def _empty_err():
            yield 'data: {"type":"error","content":"คำถามว่างเปล่า"}\n\n'

        return StreamingResponse(_empty_err(), media_type="text/event-stream")

    return StreamingResponse(
        rag_engine.ask_stream(q.strip()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/documents")
async def api_documents():
    """List PDF documents in the documents directory with their index status."""
    docs_dir = settings.documents_dir
    index_dir = settings.faiss_index_dir
    indexed = index_dir.exists() and any(index_dir.iterdir())

    if not docs_dir.exists():
        return {"documents": [], "indexed": indexed, "count": 0}

    pdfs = sorted(docs_dir.rglob("*.pdf"))
    return {
        "documents": [
            {
                "name": p.name,
                "path": str(p.relative_to(docs_dir)),
                "size": p.stat().st_size,
                "indexed": indexed,
            }
            for p in pdfs
        ],
        "indexed": indexed,
        "count": len(pdfs),
    }


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    """Upload a PDF file into the documents directory."""
    fname = file.filename or ""
    if not fname.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="รองรับเฉพาะไฟล์ PDF เท่านั้น")

    docs_dir = settings.documents_dir
    docs_dir.mkdir(parents=True, exist_ok=True)

    dest = docs_dir / fname
    content = await file.read()
    dest.write_bytes(content)
    logger.info("Uploaded PDF: %s (%d bytes)", fname, len(content))
    return {"status": "ok", "filename": fname, "size": len(content)}


@app.post("/api/reindex")
async def api_reindex():
    """Force a rebuild of the FAISS index from ./documents."""
    try:
        build_vectorstore(force=True)
        rag_engine.reload()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok", "message": "Re-indexed successfully."}


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def on_startup() -> None:
    logger.info("Starting Company RAG service ...")
    if not settings.openai_api_key:
        logger.warning("OPENAI_API_KEY is not set. /api/chat will fail until you configure it.")

    # Try to warm up the vector store, but don't crash if there are no docs yet.
    try:
        rag_engine.vectorstore  # noqa: B018 - triggers lazy init
        logger.info("FAISS index loaded successfully.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Vector store not ready: %s", exc)


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
