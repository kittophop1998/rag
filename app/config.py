"""Application configuration loaded from environment variables / .env file."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed settings for the RAG service."""

    # --- OpenAI ---
    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o"
    openai_embed_model: str = "text-embedding-3-small"

    # --- RAG ---
    documents_dir: Path = Path("./documents")
    faiss_index_dir: Path = Path("./faiss_index")
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 3

    # --- LINE ---
    line_channel_access_token: str = ""
    line_channel_secret: str = ""

    # --- Auth ---
    admin_username: str = "admin"
    admin_password: str = "admin1234"

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
