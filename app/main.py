"""
FastAPI entry point.

Endpoints:
* ``GET  /``                    -> Chat web UI (static HTML).
* ``POST /api/chat``            -> Ask a question, return JSON answer + sources.
* ``GET  /api/chat/stream``     -> Stream answer via SSE.
* ``POST /api/reindex``         -> Rebuild the FAISS index (admin only).
* ``POST /api/upload``          -> Upload PDF (admin only).
* ``POST /webhook``             -> LINE Messaging API webhook.
* ``GET  /healthz``             -> Liveness probe.
* ``GET  /api/users``           -> List users (admin only).
* ``POST /api/users``           -> Create user (admin only).
* ``PATCH /api/users/{id}``     -> Update user (admin only).
* ``DELETE /api/users/{id}``    -> Delete user (admin only).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.auth import UserSession, create_token, require_admin, require_auth, revoke_token
from app.config import settings
from app.db_settings import (
    DatabaseConnectionCreate,
    DatabaseConnectionUpdate,
    add_connection,
    delete_connection,
    list_connections,
    update_connection,
)
from app.indexer import build_vectorstore
from app.line_webhook import router as line_router
from app.rag import rag_engine
from app.user_store import (
    UserCreate,
    UserResponse,
    UserRole,
    UserUpdate,
    bootstrap_default_admin,
    create_user,
    delete_user,
    get_user_by_id,
    list_users,
    update_user,
)

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

app.include_router(line_router)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class LoginRequest(BaseModel):
    username: str
    password: str


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
    """Serve the chat UI (login check is done client-side)."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/healthz", include_in_schema=False)
async def healthz():
    return {"status": "ok"}


# ── Auth ──────────────────────────────────────────────────────────────────────
@app.post("/api/auth/login", tags=["auth"])
async def api_login(req: LoginRequest):
    """Exchange username + password for a session token."""
    session = create_token(req.username, req.password)
    return {"token": session.token, "username": session.username, "role": session.role}


@app.post("/api/auth/logout", tags=["auth"])
async def api_logout(session: UserSession = Depends(require_auth)):
    revoke_token(session.token)
    return {"status": "ok"}


@app.get("/api/auth/me", tags=["auth"])
async def api_me(session: UserSession = Depends(require_auth)):
    """Verify that the current token is still valid and return identity."""
    return {"authenticated": True, "username": session.username, "role": session.role}


# ── User Management (admin only) ──────────────────────────────────────────────
@app.get("/api/users", tags=["users"])
async def api_list_users(_: UserSession = Depends(require_admin)):
    """List all users (admin only)."""
    return {
        "users": [
            UserResponse(
                id=u.id,
                username=u.username,
                role=u.role,
                enabled=u.enabled,
                display_name=u.display_name,
                created_at=u.created_at,
            ).model_dump()
            for u in list_users()
        ]
    }


@app.post("/api/users", tags=["users"])
async def api_create_user(data: UserCreate, _: UserSession = Depends(require_admin)):
    """Create a new user (admin only)."""
    if not data.username.strip():
        raise HTTPException(status_code=400, detail="ชื่อผู้ใช้ต้องไม่ว่างเปล่า")
    if len(data.password) < 6:
        raise HTTPException(status_code=400, detail="รหัสผ่านต้องมีอย่างน้อย 6 ตัวอักษร")
    try:
        user = create_user(data)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return UserResponse(
        id=user.id,
        username=user.username,
        role=user.role,
        enabled=user.enabled,
        display_name=user.display_name,
        created_at=user.created_at,
    ).model_dump()


@app.patch("/api/users/{user_id}", tags=["users"])
async def api_update_user(
    user_id: str,
    data: UserUpdate,
    session: UserSession = Depends(require_admin),
):
    """Update a user's role, password, or status (admin only)."""
    target = get_user_by_id(user_id)
    if not target:
        raise HTTPException(status_code=404, detail="ไม่พบผู้ใช้ที่ระบุ")

    # Prevent the last admin from losing admin role or being disabled
    if target.role == UserRole.admin and session.username == target.username:
        if data.role == UserRole.user:
            raise HTTPException(
                status_code=400,
                detail="ไม่สามารถลดสิทธิ์ของตัวเองได้",
            )
        if data.enabled is False:
            raise HTTPException(
                status_code=400,
                detail="ไม่สามารถปิดใช้งานบัญชีของตัวเองได้",
            )

    if data.password is not None and len(data.password) < 6:
        raise HTTPException(status_code=400, detail="รหัสผ่านต้องมีอย่างน้อย 6 ตัวอักษร")

    updated = update_user(user_id, data)
    if not updated:
        raise HTTPException(status_code=404, detail="ไม่พบผู้ใช้ที่ระบุ")
    return UserResponse(
        id=updated.id,
        username=updated.username,
        role=updated.role,
        enabled=updated.enabled,
        display_name=updated.display_name,
        created_at=updated.created_at,
    ).model_dump()


@app.delete("/api/users/{user_id}", tags=["users"])
async def api_delete_user(user_id: str, session: UserSession = Depends(require_admin)):
    """Delete a user (admin only).  Cannot delete yourself."""
    target = get_user_by_id(user_id)
    if not target:
        raise HTTPException(status_code=404, detail="ไม่พบผู้ใช้ที่ระบุ")
    if target.username == session.username:
        raise HTTPException(status_code=400, detail="ไม่สามารถลบบัญชีของตัวเองได้")
    if not delete_user(user_id):
        raise HTTPException(status_code=404, detail="ไม่พบผู้ใช้ที่ระบุ")
    return {"status": "ok"}


# ── Database Settings (admin only) ────────────────────────────────────────────
@app.get("/api/settings/databases", tags=["settings"])
async def api_list_databases(_: UserSession = Depends(require_admin)):
    return {"databases": [c.model_dump() for c in list_connections()]}


@app.post("/api/settings/databases", tags=["settings"])
async def api_add_database(
    data: DatabaseConnectionCreate,
    _: UserSession = Depends(require_admin),
):
    conn = add_connection(data)
    return conn.model_dump()


@app.patch("/api/settings/databases/{conn_id}", tags=["settings"])
async def api_update_database(
    conn_id: str,
    data: DatabaseConnectionUpdate,
    _: UserSession = Depends(require_admin),
):
    conn = update_connection(conn_id, data)
    if not conn:
        raise HTTPException(status_code=404, detail="ไม่พบ Database ที่ระบุ")
    return conn.model_dump()


@app.delete("/api/settings/databases/{conn_id}", tags=["settings"])
async def api_delete_database(conn_id: str, _: UserSession = Depends(require_admin)):
    if not delete_connection(conn_id):
        raise HTTPException(status_code=404, detail="ไม่พบ Database ที่ระบุ")
    return {"status": "ok"}


# ── Chat (all authenticated users) ────────────────────────────────────────────
@app.post("/api/chat", response_model=ChatResponse)
async def api_chat(
    req: ChatRequest,
    _: UserSession = Depends(require_auth),
) -> ChatResponse:
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
async def api_chat_stream(
    q: str = "",
    _: UserSession = Depends(require_auth),
) -> StreamingResponse:
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


# ── Documents (all authenticated users) ───────────────────────────────────────
@app.get("/api/documents")
async def api_documents(_: UserSession = Depends(require_auth)):
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


# ── Upload & Reindex (admin only) ─────────────────────────────────────────────
@app.post("/api/upload")
async def api_upload(
    file: UploadFile = File(...),
    _: UserSession = Depends(require_admin),
):
    """Upload a PDF file into the documents directory (admin only)."""
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
async def api_reindex(_: UserSession = Depends(require_admin)):
    """Force a rebuild of the FAISS index from ./documents (admin only)."""
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

    # Ensure a default admin account always exists
    bootstrap_default_admin(settings.admin_username, settings.admin_password)
    logger.info(
        "Default admin bootstrapped (username: %s) — update via User Management.",
        settings.admin_username,
    )

    if not settings.openai_api_key:
        logger.warning("OPENAI_API_KEY is not set. /api/chat will fail until you configure it.")

    try:
        rag_engine.vectorstore  # noqa: B018 – triggers lazy init
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
