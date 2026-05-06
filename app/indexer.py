"""
Indexing pipeline for the Ruangthong RAG — backed by ChromaDB.

Supported document types
------------------------
- PDF  (.pdf)   — via pypdf / PyPDFLoader
- Word (.docx)  — via python-docx
- CSV  (.csv)   — built-in csv module

Steps performed by :func:`build_or_load_vectorstore`:

1. Load every supported document from ``settings.documents_dir``.
2. Fetch content from all enabled URL sources.
3. Split everything into overlapping chunks.
4. Embed the chunks with OpenAIEmbeddings (via langchain-openai).
5. Persist the vectors in a local ChromaDB collection.

ChromaDB persist path: ``settings.chroma_rag_dir``  (= ``chroma_base_dir/rag/``)
Collection name      : ``constants.CHROMA_RAG_COLLECTION``
"""

from __future__ import annotations

import csv
import io
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

# Supported file extensions
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".csv"}

# ---------------------------------------------------------------------------
# Thai PUA → Standard Unicode normalisation
# ---------------------------------------------------------------------------
# Old Thai PDF fonts (e.g. Angsana, Cordia, TH SarabunPSK in Windows 9x era)
# map combining vowels / tone-marks to the Unicode Private-Use-Area (U+F700–
# U+F7FF) instead of the standard Thai block (U+0E00–U+0E7F).  PyPDF extracts
# these as-is, causing a mismatch between indexed text and user queries.
_THAI_PUA_MAP: dict[str, str] = {
    "\uf700": "\u0e40",  # sara e (alternate)
    "\uf701": "\u0e41",  # sara ae (alternate)
    "\uf702": "\u0e35",  # sara ii  ี
    "\uf703": "\u0e36",  # sara ue  ึ
    "\uf704": "\u0e38",  # sara u   ุ
    "\uf705": "\u0e48",  # mai ek   ่
    "\uf706": "\u0e49",  # mai tho  ้
    "\uf707": "\u0e4a",  # mai tri  ๊
    "\uf708": "\u0e4b",  # mai jattawa ๋
    "\uf709": "\u0e47",  # maitaikhu ็
    "\uf70a": "\u0e48",  # mai ek   ่ (positional variant)
    "\uf70b": "\u0e49",  # mai tho  ้ (positional variant)
    "\uf70c": "\u0e4a",  # mai tri  ๊ (positional variant)
    "\uf70d": "\u0e4b",  # mai jattawa ๋ (positional variant)
    "\uf70e": "\u0e47",  # maitaikhu ็ (positional variant)
    "\uf70f": "\u0e47",  # maitaikhu ็ (positional variant)
    "\uf710": "\u0e31",  # sara a   ั
    "\uf711": "\u0e34",  # sara i   ิ
    "\uf712": "\u0e47",  # maitaikhu ็ (another variant)
    "\uf713": "\u0e4c",  # thanthakat ์
    "\uf714": "\u0e4d",  # nikhahit  ํ
    "\uf715": "\u0e32",  # sara aa  า (alternate)
}
_THAI_PUA_TABLE = str.maketrans(_THAI_PUA_MAP)


def _normalize_thai_pua(text: str) -> str:
    """Replace Thai PUA characters with their standard Unicode equivalents.

    Needed for PDFs created with old Thai Type-1 fonts (Angsana, Cordia, etc.)
    whose combining marks land in U+F700–U+F7FF instead of U+0E00–U+0E7F.
    Without this step the embeddings of indexed text won't match embeddings of
    normal Thai queries typed by users.
    """
    return text.translate(_THAI_PUA_TABLE)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def _load_pdfs(documents_dir: Path) -> List[Document]:
    """Load every PDF in *documents_dir* (recursively) into LangChain Documents."""
    if not documents_dir.exists():
        documents_dir.mkdir(parents=True, exist_ok=True)
        logger.warning("Documents directory %s did not exist — created empty one.", documents_dir)
        return []

    pdf_paths = sorted(documents_dir.rglob("*.pdf"))
    if not pdf_paths:
        return []

    docs: List[Document] = []
    for path in pdf_paths:
        logger.info("Loading PDF: %s", path)
        try:
            loader = PyPDFLoader(str(path))
            file_docs = loader.load()
            for d in file_docs:
                d.metadata["source"] = str(path.relative_to(documents_dir))
                d.metadata["file_type"] = "pdf"
                # Normalise Thai PUA characters (old Thai fonts) so embeddings
                # match standard Unicode queries typed by users.
                d.page_content = _normalize_thai_pua(d.page_content)
            docs.extend(file_docs)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to load PDF %s: %s", path, exc)

    logger.info("Loaded %d page(s) from %d PDF file(s).", len(docs), len(pdf_paths))
    return docs


def _load_word_docs(documents_dir: Path) -> List[Document]:
    """Load every .docx file in *documents_dir* into LangChain Documents."""
    if not documents_dir.exists():
        return []

    docx_paths = sorted(documents_dir.rglob("*.docx"))
    if not docx_paths:
        return []

    try:
        import docx as python_docx  # type: ignore[import]
    except ImportError:
        logger.warning("python-docx not installed — skipping .docx files. Run: pip install python-docx")
        return []

    docs: List[Document] = []
    for path in docx_paths:
        logger.info("Loading Word document: %s", path)
        try:
            doc = python_docx.Document(str(path))
            # Extract paragraphs + table cells
            parts: list[str] = []
            for para in doc.paragraphs:
                text = para.text.strip()
                if text:
                    parts.append(text)
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                    if row_text:
                        parts.append(row_text)

            full_text = "\n".join(parts)
            if not full_text.strip():
                logger.warning("Word document %s appears to be empty — skipping.", path)
                continue

            docs.append(
                Document(
                    page_content=full_text,
                    metadata={
                        "source":    str(path.relative_to(documents_dir)),
                        "file_type": "docx",
                    },
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to load Word document %s: %s", path, exc)

    logger.info("Loaded %d Word document(s).", len(docs))
    return docs


def _load_csv_docs(documents_dir: Path) -> List[Document]:
    """Load every .csv file in *documents_dir* into LangChain Documents.

    Each CSV is converted to a text representation where each row is a
    key=value line, grouped into chunks of 50 rows per Document so that
    the splitter can still produce appropriate chunk sizes.
    """
    if not documents_dir.exists():
        return []

    csv_paths = sorted(documents_dir.rglob("*.csv"))
    if not csv_paths:
        return []

    docs: List[Document] = []
    ROWS_PER_DOC = 50  # group rows before handing to the text splitter

    for path in csv_paths:
        logger.info("Loading CSV: %s", path)
        try:
            content = path.read_bytes()
            # Try UTF-8, fall back to TIS-620 (common for Thai CSVs)
            for encoding in ("utf-8-sig", "utf-8", "tis-620", "cp874", "latin-1"):
                try:
                    text = content.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                text = content.decode("utf-8", errors="replace")

            reader = csv.DictReader(io.StringIO(text))
            headers = reader.fieldnames or []
            rows    = list(reader)

            if not rows:
                logger.warning("CSV %s is empty — skipping.", path)
                continue

            rel_source = str(path.relative_to(documents_dir))
            # Split into chunks of ROWS_PER_DOC rows
            for chunk_start in range(0, len(rows), ROWS_PER_DOC):
                chunk = rows[chunk_start : chunk_start + ROWS_PER_DOC]
                lines = [f"ไฟล์: {path.name} | คอลัมน์: {', '.join(headers)}"]
                for i, row in enumerate(chunk, chunk_start + 1):
                    row_str = " | ".join(f"{k}: {v}" for k, v in row.items() if v)
                    lines.append(f"แถว {i}: {row_str}")
                docs.append(
                    Document(
                        page_content="\n".join(lines),
                        metadata={
                            "source":    rel_source,
                            "file_type": "csv",
                        },
                    )
                )

        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to load CSV %s: %s", path, exc)

    logger.info("Loaded %d document chunk(s) from %d CSV file(s).", len(docs), len(csv_paths))
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


def list_documents(documents_dir: Path | None = None) -> list[dict]:
    """Return a list of all supported documents in *documents_dir*.

    Each entry: {name, path, size, file_type, extension}
    """
    d = documents_dir or settings.documents_dir
    if not d.exists():
        return []
    result = []
    for ext in SUPPORTED_EXTENSIONS:
        for p in sorted(d.rglob(f"*{ext}")):
            result.append({
                "name":      p.name,
                "path":      str(p.relative_to(d)),
                "size":      p.stat().st_size,
                "file_type": ext.lstrip("."),
                "extension": ext,
            })
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_vectorstore(force: bool = False) -> Chroma:
    """Build a ChromaDB vector store from source documents and persist it.

    * ``force=False`` → raises :class:`FileExistsError` if an index already exists.
    * ``force=True``  → wipes the existing collection and rebuilds from scratch.

    Uses a **tmp-then-move** strategy so the write never touches the live
    persist_dir while a cached chromadb PersistentClient may still hold an
    open SQLite handle.  Avoids SQLITE_READONLY_DBMOVED (error 1032).

    Persist path : ``settings.chroma_rag_dir``  (``chroma_base_dir/rag/``)
    Collection   : ``CHROMA_RAG_COLLECTION``
    """
    import uuid  # noqa: PLC0415

    persist_dir = settings.chroma_rag_dir

    if _chroma_is_initialised(persist_dir) and not force:
        raise FileExistsError(
            f"ChromaDB index already exists at {persist_dir}. "
            "Pass force=True to rebuild."
        )

    # Load all source material
    pdf_docs  = _load_pdfs(settings.documents_dir)
    word_docs = _load_word_docs(settings.documents_dir)
    csv_docs  = _load_csv_docs(settings.documents_dir)
    url_docs  = _load_url_sources()
    docs      = pdf_docs + word_docs + csv_docs + url_docs

    if not docs:
        raise RuntimeError(
            "ไม่พบข้อมูลที่จะ Index — กรุณาอัปโหลดไฟล์ PDF / Word / CSV "
            f"ลงในโฟลเดอร์ {settings.documents_dir} หรือเพิ่ม URL แหล่งข้อมูลในหน้าตั้งค่า"
        )

    chunks = _split_documents(docs)
    if not chunks:
        raise RuntimeError(
            "ไม่สามารถดึงข้อความจากไฟล์ได้เลย — "
            "ไฟล์ PDF อาจเป็น scanned PDF (รูปภาพ) ที่ไม่มี text layer "
            "กรุณาแปลงให้เป็น searchable PDF หรือลองไฟล์ .docx / .csv แทน"
        )

    # ── Write to a unique tmp directory first ────────────────────────────
    # ChromaDB (0.5+) uses a process-level SharedSystemClient cache keyed
    # by persist_directory.  If we rmtree the live directory while that
    # cached client still holds an open SQLite connection, the subsequent
    # write (to the re-created directory) triggers SQLITE_READONLY_DBMOVED
    # (code 1032).  Writing to a fresh tmp path avoids the conflict entirely.
    tmp_dir = persist_dir.parent / f"{persist_dir.name}_tmp_{uuid.uuid4().hex}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    Chroma.from_documents(
        chunks,
        _embeddings(),
        persist_directory=str(tmp_dir),
        collection_name=CHROMA_RAG_COLLECTION,
    )

    # ── Atomic swap: delete old → move tmp → done ─────────────────────────
    if persist_dir.exists():
        shutil.rmtree(persist_dir)
        logger.info("Removed old ChromaDB RAG index at %s", persist_dir)
    shutil.move(str(tmp_dir), str(persist_dir))

    logger.info(
        "ChromaDB RAG index built at %s — collection '%s', %d chunks (%d PDF, %d Word, %d CSV, %d URL)",
        persist_dir, CHROMA_RAG_COLLECTION, len(chunks),
        len(pdf_docs), len(word_docs), len(csv_docs), len(url_docs),
    )

    # Return a fresh Chroma client pointing to the final (moved) path
    return Chroma(
        persist_directory=str(persist_dir),
        embedding_function=_embeddings(),
        collection_name=CHROMA_RAG_COLLECTION,
    )


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
