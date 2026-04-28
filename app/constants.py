"""
Project-wide constants.

Keep all magic numbers, collection names, and shared limits here so that
every module imports from one place instead of repeating string literals.
"""

# ---------------------------------------------------------------------------
# ChromaDB collection names
# ---------------------------------------------------------------------------
CHROMA_RAG_COLLECTION = "rag_documents"
"""ChromaDB collection that stores document chunks for RAG retrieval."""

# ---------------------------------------------------------------------------
# ChromaDB sub-directories (relative to Settings.chroma_base_dir)
# ---------------------------------------------------------------------------
CHROMA_RAG_SUBDIR = "rag"
CHROMA_VANNA_SUBDIR = "vanna"

# ---------------------------------------------------------------------------
# Text-to-SQL
# ---------------------------------------------------------------------------
MAX_SCHEMA_CHARS = 20_000
"""Maximum characters of DB schema forwarded to the LLM (~5 000 tokens)."""

MAX_RESULT_PREVIEW_ROWS = 20
"""Maximum rows included in the LLM summarisation prompt."""
