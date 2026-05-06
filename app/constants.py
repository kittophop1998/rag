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
CHROMA_DB_SUBDIR = "db"
"""Sub-directory under chroma_base_dir that holds per-connection DB indexes."""

# ---------------------------------------------------------------------------
# Text-to-SQL  ·  Token budget
# ---------------------------------------------------------------------------
MAX_SCHEMA_CHARS = 20_000
"""Maximum characters of the full schema sent to the SQL generator (~5 000 tokens).
Increased to cover large databases with 300+ tables."""

MAX_PLANNER_SCHEMA_CHARS = 25_000
"""Maximum characters of the ultra-compact (names-only) schema sent to the
Table Planner step.  Must be large enough to include ALL tables in the DB.
For a 394-table DB the compact schema is ~23 000 chars; set to 25 000 to cover it."""

MAX_SEMANTIC_CONTEXT_CHARS = 4_000
"""Maximum characters of semantic-catalog text injected into prompts.
Increased to provide richer table descriptions for large databases."""

MAX_RESULT_PREVIEW_ROWS = 20
"""Maximum rows included in the LLM summarisation prompt.
Increased from 10 → 20 to give the model more data for analysis."""

MAX_RESULT_PREVIEW_CHARS = 6_000
"""Hard character cap on the JSON result blob sent to the summariser.
Guards against rows with very long text fields."""
