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
# Text-to-SQL  ·  Token budget
# ---------------------------------------------------------------------------
MAX_SCHEMA_CHARS = 12_000
"""Maximum characters of the full schema sent to the SQL generator (~3 000 tokens).
Reduced from 20 000 to cut per-query input cost."""

MAX_PLANNER_SCHEMA_CHARS = 3_500
"""Maximum characters of the ultra-compact (names-only) schema sent to the
Table Planner step.  Much smaller because we only need table + column names."""

MAX_SEMANTIC_CONTEXT_CHARS = 2_500
"""Maximum characters of semantic-catalog text injected into prompts.
Prevents large catalogs from dominating the token budget."""

MAX_RESULT_PREVIEW_ROWS = 20
"""Maximum rows included in the LLM summarisation prompt.
Increased from 10 → 20 to give the model more data for analysis."""

MAX_RESULT_PREVIEW_CHARS = 6_000
"""Hard character cap on the JSON result blob sent to the summariser.
Guards against rows with very long text fields."""
