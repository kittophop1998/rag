"""
Indexing pipeline for the company RAG.

Steps performed by :func:`build_or_load_vectorstore`:

1. Load every PDF from ``settings.documents_dir``.
2. Split them into overlapping chunks.
3. Embed the chunks with OpenAIEmbeddings.
4. Store / load them as a local FAISS index so we don't re-index
   on every server restart.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading & splitting
# ---------------------------------------------------------------------------
def _load_pdfs(documents_dir: Path) -> List[Document]:
    """Load every PDF in *documents_dir* (recursively) into LangChain Documents."""
    if not documents_dir.exists():
        documents_dir.mkdir(parents=True, exist_ok=True)
        logger.warning("Documents directory %s did not exist, created empty one.", documents_dir)
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
# Public API
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


def build_vectorstore(force: bool = False) -> FAISS:
    """
    Build a FAISS vector store from the PDFs in ``documents_dir`` and persist
    it to ``faiss_index_dir``.

    If *force* is False and a FAISS index already exists, this raises
    :class:`FileExistsError` to make the caller decide.
    """
    index_dir = settings.faiss_index_dir
    if index_dir.exists() and any(index_dir.iterdir()) and not force:
        raise FileExistsError(
            f"A FAISS index already exists at {index_dir}. "
            f"Pass force=True to rebuild."
        )

    docs = _load_pdfs(settings.documents_dir)
    if not docs:
        raise RuntimeError(
            "No documents to index. Put PDF files into "
            f"{settings.documents_dir} and try again."
        )

    chunks = _split_documents(docs)
    vs = FAISS.from_documents(chunks, _embeddings())

    index_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(index_dir))
    logger.info("FAISS index saved to %s", index_dir)
    return vs


def load_vectorstore() -> FAISS:
    """Load a previously persisted FAISS index from disk."""
    index_dir = settings.faiss_index_dir
    if not index_dir.exists() or not any(index_dir.iterdir()):
        raise FileNotFoundError(
            f"No FAISS index found in {index_dir}. Build it first via "
            f"build_vectorstore() or POST /reindex."
        )

    logger.info("Loading FAISS index from %s", index_dir)
    return FAISS.load_local(
        str(index_dir),
        _embeddings(),
        allow_dangerous_deserialization=True,  # we trust our own pickle
    )


def build_or_load_vectorstore() -> FAISS:
    """
    Convenience helper used at server start-up:

    * If an index already exists on disk -> load it.
    * Otherwise build one from the documents folder.
    * If there are no documents either, raise so the caller can keep running
      the server without a vector store.
    """
    try:
        return load_vectorstore()
    except FileNotFoundError:
        logger.info("No persisted FAISS index found, building a new one ...")
        return build_vectorstore(force=True)
