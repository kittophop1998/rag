"""Application configuration loaded from environment variables / .env file."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from app.constants import CHROMA_RAG_SUBDIR, CHROMA_VANNA_SUBDIR, CHROMA_DB_SUBDIR


class Settings(BaseSettings):
    """Strongly-typed settings for the RAG service.

    All values can be overridden via environment variables or the ``.env`` file.
    Variable names are case-insensitive.

    Directory layout (all relative to the working directory):

        chroma_base_dir/           ← single ChromaDB root  (CHROMA_BASE_DIR)
            rag/                   ← document vector store  (chroma_rag_dir)
            db/<conn_id>/          ← per-connection DB indexes (chroma_db_dir)
            vanna/                 ← Vanna.ai SQL training  (chroma_vanna_dir)
        data/
            chat.db                ← SQLite: users, sessions, messages, configs
        documents/                 ← source PDFs            (DOCUMENTS_DIR)
    """

    # ── OpenAI ────────────────────────────────────────────────────────────
    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o"
    openai_embed_model: str = "text-embedding-3-large"

    # ── ChromaDB ──────────────────────────────────────────────────────────
    chroma_base_dir: Path = Path("./chroma")
    """Root directory for all ChromaDB data.
    Sub-folders 'rag/' and 'vanna/' are created automatically.
    """

    # ── Data directory ─────────────────────────────────────────────────────
    data_dir: Path = Path("./data")
    """Directory where SQLite database (chat.db) is stored.
    Override via DATA_DIR env var to point at a persistent volume on the server.
    Example: DATA_DIR=/var/lib/rag/data
    """

    # ── RAG ───────────────────────────────────────────────────────────────
    documents_dir: Path = Path("./documents")
    chunk_size: int = 1_000
    chunk_overlap: int = 150
    top_k: int = 5

    # ── Auth ──────────────────────────────────────────────────────────────
    admin_username: str = "admin"
    admin_password: str = "admin1234"
    auth_token_ttl_hours: int = 24 #24 hours

    # ── Server ────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Computed paths (derived from chroma_base_dir) ─────────────────────
    @property
    def chroma_rag_dir(self) -> Path:
        """ChromaDB persist directory for document RAG."""
        return self.chroma_base_dir / CHROMA_RAG_SUBDIR

    @property
    def chroma_vanna_dir(self) -> Path:
        """ChromaDB persist directory for Vanna.ai SQL training data."""
        return self.chroma_base_dir / CHROMA_VANNA_SUBDIR

    @property
    def chroma_db_dir(self) -> Path:
        """ChromaDB persist directory root for per-connection DB indexes."""
        return self.chroma_base_dir / CHROMA_DB_SUBDIR


settings = Settings()
