"""RAG (Retrieval-Augmented Generation) tools for semantic search over SEC filings."""

import asyncio
from typing import Any

from bullsh.logging import log
from bullsh.tools.base import ToolResult, ToolStatus


# Lazy import to handle optional dependencies
def _get_vectordb():
    """Get VectorDB instance, handling missing dependencies."""
    try:
        from bullsh.storage.vectordb import get_vectordb

        return get_vectordb()
    except ImportError as e:
        raise ImportError("RAG dependencies not installed. Run: pip install bullsh[rag]") from e


async def rag_search(
    query: str,
    ticker: str | None = None,
    form: str | None = None,
    year: int | None = None,
    k: int = 5,
) -> ToolResult:
    """
    Semantic search over indexed SEC filings.

    Args:
        query: Natural language search query
        ticker: Optional ticker filter
        form: Optional form type filter (10-K, 10-Q)
        year: Optional year filter
        k: Number of results to return

    Returns:
        ToolResult with relevant text chunks and metadata
    """
    try:
        log("tools", f"rag_search: query='{query}' ticker={ticker} form={form} year={year}")

        # Run in executor since VectorDB operations are sync
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            _rag_search_sync,
            query,
            ticker,
            form,
            year,
            k,
        )

        if not results:
            return ToolResult(
                data={
                    "query": query,
                    "results": [],
                    "message": "No indexed filings found. Use sec_fetch first to index filings.",
                },
                confidence=0.0,
                status=ToolStatus.PARTIAL,
                tool_name="rag_search",
                ticker=ticker or "",
            )

        # Format results for the agent
        formatted_results = []
        for r in results:
            formatted_results.append(
                {
                    "text": r["text"],
                    "score": round(r["score"], 3),
                    "ticker": r["ticker"],
                    "form": r["form"],
                    "year": r["year"],
                    "url": r.get("url"),
                }
            )

        avg_score = sum(r["score"] for r in results) / len(results)

        log("tools", f"rag_search: Found {len(results)} results, avg score {avg_score:.3f}")

        return ToolResult(
            data={
                "query": query,
                "results": formatted_results,
                "total_results": len(results),
            },
            confidence=min(avg_score, 1.0),
            status=ToolStatus.SUCCESS,
            tool_name="rag_search",
            ticker=ticker or "",
        )

    except ImportError as e:
        return ToolResult(
            data={},
            confidence=0.0,
            status=ToolStatus.FAILED,
            tool_name="rag_search",
            ticker=ticker or "",
            error_message=str(e),
        )
    except Exception as e:
        log("tools", f"rag_search: Error: {e}", level="error")
        return ToolResult(
            data={},
            confidence=0.0,
            status=ToolStatus.FAILED,
            tool_name="rag_search",
            ticker=ticker or "",
            error_message=str(e),
        )


def _rag_search_sync(
    query: str,
    ticker: str | None,
    form: str | None,
    year: int | None,
    k: int,
) -> list[dict[str, Any]]:
    """Synchronous RAG search."""
    vectordb = _get_vectordb()
    return vectordb.search(
        query=query,
        k=k,
        ticker=ticker,
        form=form,
        year=year,
    )


async def rag_index(
    ticker: str,
    form: str,
    year: int,
    text: str,
    url: str | None = None,
    filing_date: str | None = None,
    quarter: int = 0,
) -> ToolResult:
    """
    Index a filing into the vector database for RAG search.

    This is typically called automatically after sec_fetch.
    """
    ticker = ticker.upper()

    try:
        log("tools", f"rag_index: Indexing {ticker} {form} {year}")

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            _rag_index_sync,
            ticker,
            form,
            year,
            text,
            url,
            filing_date,
            quarter,
        )

        if result["status"] == "already_indexed":
            log("tools", f"rag_index: {ticker} {form} {year} already indexed")
            return ToolResult(
                data=result,
                confidence=1.0,
                status=ToolStatus.CACHED,
                tool_name="rag_index",
                ticker=ticker,
            )

        log("tools", f"rag_index: Indexed {result['chunks']} chunks for {ticker} {form} {year}")

        return ToolResult(
            data=result,
            confidence=1.0,
            status=ToolStatus.SUCCESS,
            tool_name="rag_index",
            ticker=ticker,
        )

    except ImportError as e:
        return ToolResult(
            data={},
            confidence=0.0,
            status=ToolStatus.FAILED,
            tool_name="rag_index",
            ticker=ticker,
            error_message=str(e),
        )
    except Exception as e:
        log("tools", f"rag_index: Error: {e}", level="error")
        return ToolResult(
            data={},
            confidence=0.0,
            status=ToolStatus.FAILED,
            tool_name="rag_index",
            ticker=ticker,
            error_message=str(e),
        )


def _rag_index_sync(
    ticker: str,
    form: str,
    year: int,
    text: str,
    url: str | None,
    filing_date: str | None,
    quarter: int,
) -> dict[str, Any]:
    """Synchronous RAG indexing."""
    vectordb = _get_vectordb()
    return vectordb.add_filing(
        ticker=ticker,
        form=form,
        year=year,
        text=text,
        url=url,
        filing_date=filing_date,
        quarter=quarter,
    )


async def rag_list(ticker: str | None = None) -> ToolResult:
    """
    List all indexed filings in the vector database.
    """
    try:
        loop = asyncio.get_event_loop()
        indexed = await loop.run_in_executor(
            None,
            _rag_list_sync,
            ticker,
        )

        return ToolResult(
            data={
                "indexed_filings": indexed,
                "total": len(indexed),
            },
            confidence=1.0,
            status=ToolStatus.SUCCESS,
            tool_name="rag_list",
            ticker=ticker or "",
        )

    except ImportError as e:
        return ToolResult(
            data={},
            confidence=0.0,
            status=ToolStatus.FAILED,
            tool_name="rag_list",
            ticker=ticker or "",
            error_message=str(e),
        )
    except Exception as e:
        log("tools", f"rag_list: Error: {e}", level="error")
        return ToolResult(
            data={},
            confidence=0.0,
            status=ToolStatus.FAILED,
            tool_name="rag_list",
            ticker=ticker or "",
            error_message=str(e),
        )


def _rag_list_sync(ticker: str | None) -> list[dict[str, Any]]:
    """Synchronous RAG list."""
    vectordb = _get_vectordb()
    return vectordb.list_indexed(ticker=ticker)
