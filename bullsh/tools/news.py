"""News and web search tools using DuckDuckGo."""

import asyncio
import warnings
from typing import Any

from bullsh.logging import log
from bullsh.storage.cache import get_cache
from bullsh.tools.base import ToolResult, ToolStatus

# Suppress the rename warning
warnings.filterwarnings("ignore", message=".*duckduckgo_search.*renamed.*")

# Try new package name first, fall back to old
DDGS_AVAILABLE = False
try:
    from ddgs import DDGS

    DDGS_AVAILABLE = True
except ImportError:
    try:
        from duckduckgo_search import DDGS

        DDGS_AVAILABLE = True
    except ImportError:
        pass


def _result_from_cache(cached: dict[str, Any], tool_name: str, ticker: str) -> ToolResult:
    """Reconstruct ToolResult from cached data."""
    return ToolResult(
        data=cached.get("data", {}),
        confidence=cached.get("confidence", 1.0),
        status=ToolStatus(cached.get("status", "success")),
        source_url=cached.get("source_url"),
        tool_name=tool_name,
        ticker=ticker,
        cached=True,
    )


async def search_news(query: str, days_back: int = 30) -> ToolResult:
    """
    Search financial news via DuckDuckGo.

    Args:
        query: Search query (usually company name or ticker)
        days_back: How many days back to search

    Returns:
        ToolResult with news articles
    """
    query_upper = query.upper()
    cache = get_cache()

    # Check cache first (news cached for 4 hours)
    cached = cache.get("news", query_upper, days_back=days_back)
    if cached:
        return _result_from_cache(cached, "search_news", query_upper)

    if not DDGS_AVAILABLE:
        return ToolResult(
            data={},
            confidence=0.0,
            status=ToolStatus.FAILED,
            tool_name="search_news",
            ticker=query_upper,
            error_message="duckduckgo-search not installed. Run: pip install duckduckgo-search",
        )

    try:
        log("tools", f"search_news: Searching for '{query}'")

        # DDGS is synchronous - run in executor
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, _search_ddg_sync, query)

        if not results:
            log("tools", f"search_news: No results for '{query}'")
            return ToolResult(
                data={"error": f"No news found for '{query}'"},
                confidence=0.0,
                status=ToolStatus.PARTIAL,
                tool_name="search_news",
                ticker=query,
            )

        articles = []
        for r in results:
            articles.append(
                {
                    "title": r.get("title", ""),
                    "source": r.get("source", "Unknown"),
                    "date": r.get("date", ""),
                    "snippet": r.get("body", "")[:300],
                    "url": r.get("url", ""),
                }
            )

        result_data = {
            "query": query,
            "article_count": len(articles),
            "articles": articles,
        }

        log("tools", f"search_news: Found {len(articles)} articles for '{query}'")

        # Cache successful results
        cache.set(
            "news",
            query_upper,
            {
                "data": result_data,
                "confidence": 0.8,
                "status": "success",
                "source_url": f"https://duckduckgo.com/?q={query}+stock&t=h_&iar=news",
            },
            days_back=days_back,
        )

        return ToolResult(
            data=result_data,
            confidence=0.8,
            status=ToolStatus.SUCCESS,
            source_url=f"https://duckduckgo.com/?q={query}+stock&t=h_&iar=news",
            tool_name="search_news",
            ticker=query_upper,
        )

    except Exception as e:
        log("tools", f"search_news: Error - {e}", level="error")
        return ToolResult(
            data={},
            confidence=0.0,
            status=ToolStatus.FAILED,
            tool_name="search_news",
            ticker=query,
            error_message=str(e),
        )


def _search_ddg_sync(query: str) -> list[dict[str, Any]]:
    """Synchronous DuckDuckGo search."""
    try:
        with DDGS() as ddgs:
            search_query = f"{query} stock"
            return list(ddgs.news(search_query, max_results=10))
    except Exception as e:
        log("tools", f"search_news sync: Error - {e}", level="error")
        return []


async def web_search(query: str, max_results: int = 10) -> ToolResult:
    """
    General web search for current information not found in SEC filings.

    Use this to:
    - Get current stock prices and market data
    - Find recent company announcements
    - Research competitors and market trends
    - Fill gaps when filing data is outdated or missing

    Args:
        query: Search query
        max_results: Maximum number of results to return

    Returns:
        ToolResult with search results
    """
    cache = get_cache()
    cache_key = query.lower().replace(" ", "_")[:50]

    # Check cache (web search cached for 1 hour)
    cached = cache.get("web_search", cache_key)
    if cached:
        return _result_from_cache(cached, "web_search", query)

    if not DDGS_AVAILABLE:
        return ToolResult(
            data={},
            confidence=0.0,
            status=ToolStatus.FAILED,
            tool_name="web_search",
            ticker=query,
            error_message="duckduckgo-search not installed. Run: pip install duckduckgo-search",
        )

    try:
        log("tools", f"web_search: Searching for '{query}'")

        # Run in executor since DDGS is synchronous
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, _web_search_sync, query, max_results)

        if not results:
            log("tools", f"web_search: No results for '{query}'")
            return ToolResult(
                data={"error": f"No results found for '{query}'"},
                confidence=0.0,
                status=ToolStatus.PARTIAL,
                tool_name="web_search",
                ticker=query,
            )

        search_results = []
        for r in results:
            search_results.append(
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", r.get("link", "")),
                    "snippet": r.get("body", r.get("snippet", ""))[:500],
                }
            )

        result_data = {
            "query": query,
            "result_count": len(search_results),
            "results": search_results,
        }

        log("tools", f"web_search: Found {len(search_results)} results for '{query}'")

        # Cache successful results
        cache.set(
            "web_search",
            cache_key,
            {
                "data": result_data,
                "confidence": 0.7,
                "status": "success",
                "source_url": f"https://duckduckgo.com/?q={query}",
            },
            ttl_hours=1,
        )

        return ToolResult(
            data=result_data,
            confidence=0.7,
            status=ToolStatus.SUCCESS,
            source_url=f"https://duckduckgo.com/?q={query}",
            tool_name="web_search",
            ticker=query,
        )

    except Exception as e:
        log("tools", f"web_search: Error - {e}", level="error")
        return ToolResult(
            data={},
            confidence=0.0,
            status=ToolStatus.FAILED,
            tool_name="web_search",
            ticker=query,
            error_message=str(e),
        )


def _web_search_sync(query: str, max_results: int = 10) -> list[dict[str, Any]]:
    """Synchronous DuckDuckGo web search."""
    try:
        with DDGS() as ddgs:
            # Use text search for general web results
            return list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        log("tools", f"web_search sync: Error - {e}", level="error")
        return []
