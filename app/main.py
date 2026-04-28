"""
FastAPI entry point.

Endpoints:
* ``GET  /``                                    -> Chat web UI (static HTML).
* ``POST /api/chat``                            -> Ask a question, return JSON answer + sources.
* ``GET  /api/chat/stream``                     -> Stream answer via SSE.
* ``POST /api/reindex``                         -> Rebuild the ChromaDB RAG index (admin only).
* ``POST /api/upload``                          -> Upload PDF (admin only).
* ``GET  /healthz``                             -> Liveness probe.
* ``POST /api/db-query``                        -> NL → SQL via Vanna.ai + LLM fallback.
* ``POST /api/vanna/train``                     -> Add Vanna training data (admin only).
* ``GET  /api/vanna/training-data``             -> List Vanna training entries (admin only).
* ``DELETE /api/vanna/training-data/{id}``      -> Remove Vanna training entry (admin only).
* ``POST /api/vanna/train-connection/{conn_id}``-> Train Vanna on a DB schema (admin only).
* ``GET  /api/users``                           -> List users (admin only).
* ``POST /api/users``                           -> Create user (admin only).
* ``PATCH /api/users/{id}``                     -> Update user (admin only).
* ``DELETE /api/users/{id}``                    -> Delete user (admin only).
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
    get_connection,
    list_connections,
    update_connection,
)
from app.url_sources import (
    UrlSourceCreate,
    UrlSourceUpdate,
    add_url_source,
    delete_url_source,
    get_url_source,
    list_url_sources,
    update_url_source,
)
from app.indexer import build_vectorstore
from app.rag import rag_engine
from app.text_to_sql import text_to_query_engine
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
    description="ระบบถาม-ตอบเอกสารภายในบริษัท ด้วย FastAPI + LangChain + OpenAI + ChromaDB + Vanna.ai",
    version="2.0.0",
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


class DBQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural-language question about the database")
    db_connection_id: str = Field(..., description="ID of the DatabaseConnection to query")


class VannaTrainRequest(BaseModel):
    type: str = Field(..., description="Training type: 'sql', 'ddl', or 'documentation'")
    question: str = Field(default="", description="Natural-language question (for type='sql')")
    sql: str = Field(default="", description="SQL query paired with the question (for type='sql')")
    ddl: str = Field(default="", description="DDL CREATE TABLE statement (for type='ddl')")
    documentation: str = Field(default="", description="Business context text (for type='documentation')")


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


# ── URL Source Settings (admin only) ──────────────────────────────────────────
@app.get("/api/settings/urls", tags=["settings"])
async def api_list_urls(_: UserSession = Depends(require_admin)):
    return {"urls": [s.model_dump() for s in list_url_sources()]}


@app.post("/api/settings/urls", tags=["settings"])
async def api_add_url(
    data: UrlSourceCreate,
    _: UserSession = Depends(require_admin),
):
    source = add_url_source(data)
    return source.model_dump()


@app.patch("/api/settings/urls/{source_id}", tags=["settings"])
async def api_update_url(
    source_id: str,
    data: UrlSourceUpdate,
    _: UserSession = Depends(require_admin),
):
    source = update_url_source(source_id, data)
    if not source:
        raise HTTPException(status_code=404, detail="ไม่พบ URL source ที่ระบุ")
    return source.model_dump()


@app.delete("/api/settings/urls/{source_id}", tags=["settings"])
async def api_delete_url(source_id: str, _: UserSession = Depends(require_admin)):
    if not delete_url_source(source_id):
        raise HTTPException(status_code=404, detail="ไม่พบ URL source ที่ระบุ")
    return {"status": "ok"}


# ── Database Query via Natural Language (all authenticated users) ──────────────
@app.post("/api/db-query", tags=["database"])
async def api_db_query(
    req: DBQueryRequest,
    _: UserSession = Depends(require_auth),
):
    """Convert a natural-language question into a DB query and return an answer.

    Steps performed server-side:
    1. Look up the saved DatabaseConnection by *db_connection_id*.
    2. Read the database schema via SQLAlchemy / pymongo.
    3. Ask GPT to generate a SQL or MongoDB query.
    4. Execute the query (read-only, max 200 rows).
    5. Ask GPT to summarise the result in Thai.
    """
    conn = get_connection(req.db_connection_id)
    if not conn:
        raise HTTPException(status_code=404, detail="ไม่พบ Database connection ที่ระบุ")
    if not conn.enabled:
        raise HTTPException(status_code=400, detail="Database connection นี้ถูกปิดใช้งาน")

    try:
        result = text_to_query_engine.ask(
            question=req.question,
            db_type=conn.db_type,
            db_url=conn.url,
            conn_id=req.db_connection_id,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("db-query failed: %s", exc)
        raise HTTPException(status_code=500, detail="ระบบมีปัญหาชั่วคราว กรุณาลองใหม่") from exc

    return result


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
                "ยังไม่มี ChromaDB index — กรุณาอัปโหลดไฟล์ PDF ลงในโฟลเดอร์ "
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
    indexed = (settings.chroma_rag_dir / "chroma.sqlite3").exists()

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
    """Force a rebuild of the ChromaDB RAG index from ./documents (admin only)."""
    try:
        build_vectorstore(force=True)
        rag_engine.reload()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok", "message": "Re-indexed successfully."}


# ── Vanna.ai — Text-to-SQL training (admin only) ──────────────────────────────
@app.post("/api/vanna/train", tags=["vanna"])
async def api_vanna_train(
    req: VannaTrainRequest,
    _: UserSession = Depends(require_admin),
):
    """Add training data to Vanna.ai's ChromaDB vector store (admin only).

    - type='sql'           → requires question + sql
    - type='ddl'           → requires ddl
    - type='documentation' → requires documentation
    """
    from app.vanna_engine import vanna_engine  # noqa: PLC0415

    try:
        if req.type == "sql":
            if not req.question.strip() or not req.sql.strip():
                raise HTTPException(status_code=400, detail="'question' และ 'sql' จำเป็นต้องระบุ")
            training_id = vanna_engine.add_sql_example(req.question.strip(), req.sql.strip())
        elif req.type == "ddl":
            if not req.ddl.strip():
                raise HTTPException(status_code=400, detail="'ddl' จำเป็นต้องระบุ")
            training_id = vanna_engine.add_ddl(req.ddl.strip())
        elif req.type == "documentation":
            if not req.documentation.strip():
                raise HTTPException(status_code=400, detail="'documentation' จำเป็นต้องระบุ")
            training_id = vanna_engine.add_documentation(req.documentation.strip())
        else:
            raise HTTPException(
                status_code=400,
                detail="'type' ต้องเป็นหนึ่งใน: 'sql', 'ddl', 'documentation'",
            )
        return {"status": "ok", "training_id": training_id, "type": req.type}
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("vanna/train failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"การ train ล้มเหลว: {exc}") from exc


@app.get("/api/vanna/training-data", tags=["vanna"])
async def api_vanna_training_data(_: UserSession = Depends(require_admin)):
    """List all Vanna.ai training entries stored in ChromaDB (admin only)."""
    from app.vanna_engine import vanna_engine  # noqa: PLC0415

    try:
        data = vanna_engine.get_training_data()
        return {
            "training_data": data,
            "count": len(data),
            "trained_connections": vanna_engine.get_trained_connections(),
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("vanna/training-data failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"ดึงข้อมูล training ไม่สำเร็จ: {exc}") from exc


@app.delete("/api/vanna/training-data/{training_id}", tags=["vanna"])
async def api_vanna_delete_training(
    training_id: str,
    _: UserSession = Depends(require_admin),
):
    """Remove a single Vanna.ai training entry by its ID (admin only)."""
    from app.vanna_engine import vanna_engine  # noqa: PLC0415

    try:
        ok = vanna_engine.remove_training_data(training_id)
        if not ok:
            raise HTTPException(status_code=404, detail="ไม่พบข้อมูล training ที่ระบุ")
        return {"status": "ok"}
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"ลบข้อมูล training ไม่สำเร็จ: {exc}") from exc


@app.post("/api/vanna/train-connection/{conn_id}", tags=["vanna"])
async def api_vanna_train_connection(
    conn_id: str,
    _: UserSession = Depends(require_admin),
):
    """Manually trigger Vanna.ai schema training for a saved DB connection (admin only).

    This extracts DDL and schema description from the live database and stores
    them in Vanna's ChromaDB.  The same training happens automatically on first
    use of a connection via /api/db-query.
    """
    from app.vanna_engine import vanna_engine  # noqa: PLC0415
    from app.db_inspector import get_schema_description  # noqa: PLC0415

    conn = get_connection(conn_id)
    if not conn:
        raise HTTPException(status_code=404, detail="ไม่พบ Database connection ที่ระบุ")
    if not conn.enabled:
        raise HTTPException(status_code=400, detail="Database connection นี้ถูกปิดใช้งาน")

    try:
        schema = get_schema_description(conn.db_type.lower(), conn.url)
        count = vanna_engine.train_on_connection(
            conn_id=conn_id,
            db_type=conn.db_type,
            db_url=conn.url,
            schema_description=schema,
        )
        return {
            "status": "ok",
            "connection_name": conn.name,
            "items_trained": count,
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("vanna/train-connection failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"การ train ล้มเหลว: {exc}") from exc


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def on_startup() -> None:
    logger.info("Starting Company RAG service (ChromaDB + Vanna.ai) ...")

    bootstrap_default_admin(settings.admin_username, settings.admin_password)
    logger.info(
        "Default admin bootstrapped (username: %s) — update via User Management.",
        settings.admin_username,
    )

    if not settings.openai_api_key:
        logger.warning("OPENAI_API_KEY is not set. /api/chat will fail until you configure it.")

    try:
        rag_engine.vectorstore  # noqa: B018 – triggers lazy init
        logger.info(
            "ChromaDB RAG index loaded (path: %s).", settings.chroma_rag_dir
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Vector store not ready (run Rebuild Index): %s", exc)


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
