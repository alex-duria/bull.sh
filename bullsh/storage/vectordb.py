"""Vector database for RAG over SEC filings using DuckDB."""

from pathlib import Path
from typing import Any

import numpy as np

from bullsh.logging import log

# Lazy imports for optional dependencies
_duckdb = None
_model = None


def _get_duckdb():
    """Lazy load duckdb."""
    global _duckdb
    if _duckdb is None:
        try:
            import duckdb

            _duckdb = duckdb
        except ImportError:
            raise ImportError("duckdb not installed. Run: pip install duckdb")
    return _duckdb


_model_loading_shown = False


def _get_embedding_model():
    """Lazy load sentence-transformers model."""
    global _model, _model_loading_shown
    if _model is None:
        try:
            # Suppress HuggingFace noise during model load
            import os
            import sys

            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
            os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

            # Check if model needs download (first time)
            from pathlib import Path

            # HuggingFace cache location varies by OS
            hf_cache = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
            if hf_cache:
                cache_dir = Path(hf_cache)
            elif os.name == "nt":  # Windows
                cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            else:
                cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

            model_cached = False
            try:
                if cache_dir.exists():
                    model_cached = any(
                        d.name.startswith("models--sentence-transformers--all-MiniLM-L6-v2")
                        for d in cache_dir.iterdir()
                        if d.is_dir()
                    )
            except (OSError, PermissionError):
                pass  # Assume not cached if we can't check

            if not model_cached and not _model_loading_shown:
                _model_loading_shown = True
                print(
                    "  ◐ Downloading embedding model (first time only)...",
                    file=sys.stderr,
                    flush=True,
                )

            from sentence_transformers import SentenceTransformer

            # all-MiniLM-L6-v2 is fast and effective (384 dimensions)
            _model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
            log("tools", "Loaded embedding model: all-MiniLM-L6-v2")

            if not model_cached:
                print("  ✓ Embedding model ready", file=sys.stderr, flush=True)

        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )
    return _model


class VectorDB:
    """
    Local vector database for SEC filings using DuckDB.

    Stores chunked filing text with embeddings for semantic search.
    Based on tenk's approach: https://github.com/ralliesai/tenk
    """

    # Chunking parameters (from tenk)
    CHUNK_SIZE = 3000  # Characters per chunk
    CHUNK_OVERLAP = 500  # Overlap between chunks
    EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 dimension

    def __init__(self, db_path: Path):
        """Initialize vector database."""
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        duckdb = _get_duckdb()
        self.conn = duckdb.connect(str(db_path))
        self._init_schema()

    def _init_schema(self) -> None:
        """Create database schema if not exists."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS filings (
                ticker VARCHAR,
                form VARCHAR,
                year INTEGER,
                quarter INTEGER,
                chunk_index INTEGER,
                chunk_text TEXT,
                embedding FLOAT[384],
                url VARCHAR,
                filing_date VARCHAR,
                PRIMARY KEY (ticker, form, year, quarter, chunk_index)
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_filings_ticker
            ON filings(ticker)
        """)
        log("tools", f"VectorDB initialized at {self.db_path}")

    def add_filing(
        self,
        ticker: str,
        form: str,
        year: int,
        text: str,
        url: str | None = None,
        filing_date: str | None = None,
        quarter: int = 0,
    ) -> dict[str, Any]:
        """
        Add a filing to the vector database.

        Chunks the text, generates embeddings, and stores in DuckDB.
        """
        ticker = ticker.upper()

        # Check if already indexed
        existing = self.conn.execute(
            "SELECT COUNT(*) FROM filings WHERE ticker = ? AND form = ? AND year = ? AND quarter = ?",
            [ticker, form, year, quarter],
        ).fetchone()[0]

        if existing > 0:
            log("tools", f"Filing already indexed: {ticker} {form} {year}")
            return {"status": "already_indexed", "chunks": existing}

        # Chunk the text
        chunks = self._chunk_text(text)
        log("tools", f"Chunked {ticker} {form} {year} into {len(chunks)} chunks")

        if not chunks:
            return {"status": "no_content", "chunks": 0}

        # Generate embeddings
        model = _get_embedding_model()
        embeddings = model.encode(chunks, show_progress_bar=False)

        # Insert into database
        data = [
            (ticker, form, year, quarter, i, chunk, emb.tolist(), url, filing_date)
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
        ]

        self.conn.executemany("INSERT INTO filings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", data)

        log("tools", f"Indexed {len(chunks)} chunks for {ticker} {form} {year}")
        return {"status": "indexed", "chunks": len(chunks)}

    def search(
        self,
        query: str,
        k: int = 5,
        ticker: str | None = None,
        form: str | None = None,
        year: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Semantic search over indexed filings.

        Returns top-k most relevant chunks based on cosine similarity.
        """
        # Build WHERE clause for filtering
        conditions = []
        params = []

        if ticker:
            conditions.append("ticker = ?")
            params.append(ticker.upper())
        if form:
            conditions.append("form = ?")
            params.append(form)
        if year:
            conditions.append("year = ?")
            params.append(year)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        # Fetch all matching chunks
        rows = self.conn.execute(
            f"""
            SELECT chunk_text, ticker, form, year, quarter, embedding, url, filing_date
            FROM filings {where_clause}
        """,
            params,
        ).fetchall()

        if not rows:
            log("tools", "No indexed filings found for search filters")
            return []

        # Extract embeddings and metadata
        texts = [r[0] for r in rows]
        embeddings = np.array([r[5] for r in rows])
        metadata = [
            {
                "ticker": r[1],
                "form": r[2],
                "year": r[3],
                "quarter": r[4],
                "url": r[6],
                "filing_date": r[7],
            }
            for r in rows
        ]

        # Encode query
        model = _get_embedding_model()
        query_emb = model.encode(query)

        # Calculate cosine similarity
        # Normalize embeddings
        emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        query_norm = np.linalg.norm(query_emb)

        # Avoid division by zero
        emb_norms = np.where(emb_norms == 0, 1e-8, emb_norms)
        query_norm = query_norm if query_norm > 0 else 1e-8

        similarities = np.dot(embeddings / emb_norms, query_emb / query_norm)

        # Get top-k results
        top_indices = np.argsort(similarities)[-k:][::-1]

        results = []
        for i in top_indices:
            results.append(
                {
                    "text": texts[i],
                    "score": float(similarities[i]),
                    **metadata[i],
                }
            )

        log(
            "tools",
            f"Search returned {len(results)} results, top score: {results[0]['score']:.3f}"
            if results
            else "No results",
        )
        return results

    def list_indexed(self, ticker: str | None = None) -> list[dict[str, Any]]:
        """List all indexed filings, optionally filtered by ticker."""
        if ticker:
            rows = self.conn.execute(
                """
                SELECT DISTINCT ticker, form, year, quarter, COUNT(*) as chunks, MAX(url) as url
                FROM filings
                WHERE ticker = ?
                GROUP BY ticker, form, year, quarter
                ORDER BY year DESC
            """,
                [ticker.upper()],
            ).fetchall()
        else:
            rows = self.conn.execute("""
                SELECT DISTINCT ticker, form, year, quarter, COUNT(*) as chunks, MAX(url) as url
                FROM filings
                GROUP BY ticker, form, year, quarter
                ORDER BY ticker, year DESC
            """).fetchall()

        return [
            {
                "ticker": r[0],
                "form": r[1],
                "year": r[2],
                "quarter": r[3],
                "chunks": r[4],
                "url": r[5],
            }
            for r in rows
        ]

    def delete_filing(self, ticker: str, form: str, year: int, quarter: int = 0) -> int:
        """Delete a filing from the index. Returns number of chunks deleted."""
        result = self.conn.execute(
            "DELETE FROM filings WHERE ticker = ? AND form = ? AND year = ? AND quarter = ?",
            [ticker.upper(), form, year, quarter],
        )
        deleted = result.rowcount
        log("tools", f"Deleted {deleted} chunks for {ticker} {form} {year}")
        return deleted

    def clear(self) -> int:
        """Clear all indexed filings. Returns number of chunks deleted."""
        result = self.conn.execute("DELETE FROM filings")
        deleted = result.rowcount
        log("tools", f"Cleared vector database: {deleted} chunks deleted")
        return deleted

    def _chunk_text(self, text: str) -> list[str]:
        """
        Split text into overlapping chunks.

        Uses tenk's approach: 3000 char chunks with 500 char overlap.
        Filters out low-quality chunks (boilerplate, headers-only, etc.)
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.CHUNK_SIZE
            chunk = text[start:end]

            # Only add substantive chunks
            if self._is_quality_chunk(chunk):
                chunks.append(chunk)

            # Move forward with overlap
            start = end - self.CHUNK_OVERLAP

        return chunks

    def _is_quality_chunk(self, chunk: str) -> bool:
        """Filter out low-quality chunks (boilerplate, headers, etc.)."""
        stripped = chunk.strip()

        # Skip empty or very short chunks
        if len(stripped) < 300:
            return False

        # Skip chunks that are mostly whitespace/formatting
        alpha_chars = sum(1 for c in stripped if c.isalpha())
        if alpha_chars < len(stripped) * 0.3:
            return False

        # Skip chunks that look like pure table of contents
        lines = stripped.split("\n")
        if len(lines) > 10:
            short_lines = sum(1 for line in lines if len(line.strip()) < 30)
            if short_lines > len(lines) * 0.8:
                return False

        # Skip ONLY pure boilerplate - not section content
        boilerplate_only = [
            "page intentionally left blank",
            "exhibit index",
            "table of contents",
        ]
        lower_chunk = stripped.lower()
        if any(phrase in lower_chunk and len(stripped) < 800 for phrase in boilerplate_only):
            return False

        return True

    def get_stats(self) -> dict[str, Any]:
        """Get database statistics."""
        total_chunks = self.conn.execute("SELECT COUNT(*) FROM filings").fetchone()[0]
        unique_filings = self.conn.execute(
            "SELECT COUNT(DISTINCT ticker || form || CAST(year AS VARCHAR)) FROM filings"
        ).fetchone()[0]
        unique_tickers = self.conn.execute("SELECT COUNT(DISTINCT ticker) FROM filings").fetchone()[
            0
        ]

        return {
            "total_chunks": total_chunks,
            "unique_filings": unique_filings,
            "unique_tickers": unique_tickers,
            "db_path": str(self.db_path),
        }


# Global instance
_vectordb: VectorDB | None = None


def get_vectordb() -> VectorDB:
    """Get the global VectorDB instance."""
    global _vectordb
    if _vectordb is None:
        from bullsh.config import get_config

        config = get_config()
        db_path = config.data_dir / "vectordb" / "filings.db"
        _vectordb = VectorDB(db_path)
    return _vectordb


def reset_vectordb() -> None:
    """Reset the global VectorDB instance (for testing)."""
    global _vectordb
    _vectordb = None
