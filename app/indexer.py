"""
Indexing pipeline for the company RAG — backed by ChromaDB.

Steps performed by :func:`build_or_load_vectorstore`:

1. Load every PDF from ``settings.documents_dir``.
2. Fetch content from all enabled URL sources.
3. Split everything into overlapping chunks.
4. Embed the chunks with OpenAIEmbeddings.
5. Persist the vectors in a local ChromaDB collection so we don't re-embed
   on every server restart.

ChromaDB persist path: ``settings.chroma_rag_dir``  (= ``chroma_base_dir/rag/``)
Collection name      : ``constants.CHROMA_RAG_COLLECTION``
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.constants import CHROMA_RAG_COLLECTION

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading & splitting
# ---------------------------------------------------------------------------
def _load_pdfs(documents_dir: Path) -> List[Document]:
    """Load every PDF in *documents_dir* (recursively) into LangChain Documents."""
    if not documents_dir.exists():
        documents_dir.mkdir(parents=True, exist_ok=True)
        logger.warning("Documents directory %s did not exist — created empty one.", documents_dir)
        return []

    pdf_paths = sorted(documents_dir.rglob("*.pdf"))
    if not pdf_paths:
        logger.warning("No PDF files found in %s.", documents_dir)
        return []

    docs: List[Document] = []
    for path in pdf_paths:
        logger.info("Loading PDF: %s", path)
        try:
            loader = PyPDFLoader(str(path))
            file_docs = loader.load()
            for d in file_docs:
                d.metadata["source"] = str(path.relative_to(documents_dir))
            docs.extend(file_docs)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to load %s: %s", path, exc)

    logger.info("Loaded %d page(s) from %d PDF file(s).", len(docs), len(pdf_paths))
    return docs


def _load_url_sources() -> List[Document]:
    """Crawl all enabled URL sources and return their content as Documents."""
    try:
        from app.url_sources import list_url_sources, mark_url_indexed  # noqa: PLC0415
        from app.web_crawler import crawl_url  # noqa: PLC0415
    except ImportError as exc:
        logger.warning("URL source modules unavailable: %s", exc)
        return []

    enabled = [s for s in list_url_sources() if s.enabled]
    if not enabled:
        return []

    all_docs: List[Document] = []
    for source in enabled:
        logger.info(
            "Crawling URL source '%s': %s (depth=%d)",
            source.name, source.url, source.crawl_depth,
        )
        try:
            crawled = crawl_url(source.url, crawl_depth=source.crawl_depth, source_name=source.name)
            all_docs.extend(crawled)
            mark_url_indexed(source.id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to crawl '%s' (%s): %s", source.name, source.url, exc)

    logger.info("Loaded %d page(s) from %d URL source(s).", len(all_docs), len(enabled))
    return all_docs


def _split_documents(docs: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks suitable for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    logger.info("Split into %d chunk(s).", len(chunks))
    return chunks


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _embeddings() -> OpenAIEmbeddings:
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. Add it to your .env before indexing."
        )
    return OpenAIEmbeddings(
        model=settings.openai_embed_model,
        api_key=settings.openai_api_key,
    )


def _chroma_is_initialised(persist_dir: Path) -> bool:
    """Return True when ChromaDB has been written to *persist_dir*."""
    return (persist_dir / "chroma.sqlite3").exists()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_vectorstore(force: bool = False) -> Chroma:
    """Build a ChromaDB vector store from source documents and persist it.

    * ``force=False`` → raises :class:`FileExistsError` if an index already exists.
    * ``force=True``  → wipes the existing collection and rebuilds from scratch.

    Persist path : ``settings.chroma_rag_dir``  (``chroma_base_dir/rag/``)
    Collection   : ``CHROMA_RAG_COLLECTION``
    """
    persist_dir = settings.chroma_rag_dir

    if _chroma_is_initialised(persist_dir) and not force:
        raise FileExistsError(
            f"ChromaDB index already exists at {persist_dir}. "
            "Pass force=True to rebuild."
        )

    # Load source material
    pdf_docs = _load_pdfs(settings.documents_dir)
    url_docs = _load_url_sources()
    docs = pdf_docs + url_docs

    if not docs:
        raise RuntimeError(
            "ไม่พบข้อมูลที่จะ Index — กรุณาอัปโหลดไฟล์ PDF ลงในโฟลเดอร์ "
            f"{settings.documents_dir} หรือเพิ่ม URL แหล่งข้อมูลในหน้าตั้งค่า"
        )

    chunks = _split_documents(docs)
    if not chunks:
        raise RuntimeError(
            "ไม่สามารถดึงข้อความจากไฟล์ PDF ได้เลย — "
            "ไฟล์อาจเป็น scanned PDF (รูปภาพ) ที่ไม่มี text layer "
            "กรุณาแปลงไฟล์ให้เป็น PDF ที่มีข้อความ (searchable PDF) ก่อนอัปโหลด"
        )

    # Wipe old index before rebuilding
    if force and persist_dir.exists():
        shutil.rmtree(persist_dir)
        logger.info("Removed old ChromaDB RAG index at %s", persist_dir)

    persist_dir.mkdir(parents=True, exist_ok=True)

    vs = Chroma.from_documents(
        chunks,
        _embeddings(),
        persist_directory=str(persist_dir),
        collection_name=CHROMA_RAG_COLLECTION,
    )
    logger.info(
        "ChromaDB RAG index built at %s — collection '%s', %d chunks",
        persist_dir, CHROMA_RAG_COLLECTION, len(chunks),
    )
    return vs


def load_vectorstore() -> Chroma:
    """Load the persisted ChromaDB collection from disk.

    Raises :class:`FileNotFoundError` if no index has been built yet.
    """
    persist_dir = settings.chroma_rag_dir

    if not _chroma_is_initialised(persist_dir):
        raise FileNotFoundError(
            f"ไม่พบ ChromaDB index ที่ {persist_dir} — กรุณากด Rebuild Index ก่อนครับ"
        )

    logger.info("Loading ChromaDB RAG index from %s", persist_dir)
    return Chroma(
        persist_directory=str(persist_dir),
        embedding_function=_embeddings(),
        collection_name=CHROMA_RAG_COLLECTION,
    )


def build_or_load_vectorstore() -> Chroma:
    """Convenience helper used at server start-up.

    * Existing index found → load it.
    * No index → build from documents folder.
    * No documents either → raise so the server keeps running without a store.
    """
    try:
        return load_vectorstore()
    except FileNotFoundError:
        logger.info("No ChromaDB RAG index found — building a new one ...")
        return build_vectorstore(force=True)
