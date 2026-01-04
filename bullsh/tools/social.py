"""Social sentiment tools - StockTwits and Reddit."""

import asyncio
from typing import Any

import httpx
from bs4 import BeautifulSoup

from bullsh.storage.cache import get_cache
from bullsh.tools.base import ToolResult, ToolStatus
from bullsh.logging import log


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


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


async def search_stocktwits(symbol: str) -> ToolResult:
    """
    Search StockTwits for stock discussions and sentiment.

    Primary social sentiment source.
    """
    symbol = symbol.upper()
    cache = get_cache()

    # Check cache first (social data cached for 1 hour)
    cached = cache.get("stocktwits", symbol)
    if cached:
        return _result_from_cache(cached, "search_stocktwits", symbol)

    try:
        log("tools", f"search_stocktwits: Fetching sentiment for {symbol}")
        async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
            # StockTwits has a public API for streams
            api_url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"

            resp = await client.get(api_url, headers=HEADERS)

            if resp.status_code == 404:
                log("tools", f"search_stocktwits: Symbol not found: {symbol}")
                return ToolResult(
                    data={"error": f"Symbol not found: {symbol}"},
                    confidence=0.0,
                    status=ToolStatus.FAILED,
                    tool_name="search_stocktwits",
                    ticker=symbol,
                    error_message=f"Symbol not found: {symbol}",
                )

            if resp.status_code != 200:
                log("tools", f"search_stocktwits: API returned {resp.status_code}, falling back to Reddit")
                # Fall back to Reddit if StockTwits fails
                return await search_reddit(symbol)

            data = resp.json()

            messages = []
            sentiment_counts = {"bullish": 0, "bearish": 0, "neutral": 0}

            for msg in data.get("messages", [])[:20]:  # Last 20 messages
                sentiment = msg.get("entities", {}).get("sentiment", {}).get("basic")
                if sentiment:
                    sentiment_counts[sentiment.lower()] = sentiment_counts.get(sentiment.lower(), 0) + 1

                messages.append({
                    "text": msg.get("body", "")[:200],
                    "sentiment": sentiment,
                    "created_at": msg.get("created_at"),
                    "likes": msg.get("likes", {}).get("total", 0),
                })

            # Calculate overall sentiment
            total = sum(sentiment_counts.values())
            if total > 0:
                bullish_pct = sentiment_counts["bullish"] / total
                bearish_pct = sentiment_counts["bearish"] / total

                if bullish_pct > 0.6:
                    overall_sentiment = "Bullish"
                elif bearish_pct > 0.6:
                    overall_sentiment = "Bearish"
                else:
                    overall_sentiment = "Mixed"
            else:
                overall_sentiment = "Unknown"

            result_data = {
                "symbol": symbol,
                "source": "StockTwits",
                "overall_sentiment": overall_sentiment,
                "sentiment_breakdown": sentiment_counts,
                "message_count": len(messages),
                "sample_messages": messages[:5],
            }

            confidence = 0.8 if len(messages) >= 10 else 0.5

            # Cache successful results
            cache.set("stocktwits", symbol, {
                "data": result_data,
                "confidence": confidence,
                "status": "success",
                "source_url": f"https://stocktwits.com/symbol/{symbol}",
            })

            return ToolResult(
                data=result_data,
                confidence=confidence,
                status=ToolStatus.SUCCESS,
                source_url=f"https://stocktwits.com/symbol/{symbol}",
                tool_name="search_stocktwits",
                ticker=symbol,
            )

    except httpx.TimeoutException:
        log("tools", f"search_stocktwits: Timeout for {symbol}, falling back to Reddit")
        # Fall back to Reddit
        return await search_reddit(symbol)
    except Exception as e:
        log("tools", f"search_stocktwits: Error for {symbol}: {e}", level="error")
        return ToolResult(
            data={},
            confidence=0.0,
            status=ToolStatus.FAILED,
            tool_name="search_stocktwits",
            ticker=symbol,
            error_message=str(e),
        )


async def search_reddit(
    query: str,
    subreddits: list[str] | None = None,
) -> ToolResult:
    """
    Search Reddit for stock discussions.

    Fallback if StockTwits unavailable.
    """
    query_upper = query.upper()
    cache = get_cache()

    # Check cache first
    cached = cache.get("reddit", query_upper)
    if cached:
        return _result_from_cache(cached, "search_reddit", query_upper)

    subreddits = subreddits or ["stocks", "investing", "wallstreetbets", "SecurityAnalysis"]

    try:
        log("tools", f"search_reddit: Searching for '{query}' in {subreddits}")
        async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
            results = []

            # Try JSON API first (more reliable than scraping)
            for sub in subreddits[:4]:
                try:
                    # Reddit JSON API endpoint
                    url = f"https://www.reddit.com/r/{sub}/search.json"
                    params = {
                        "q": query,
                        "restrict_sr": "on",
                        "sort": "relevance",
                        "t": "month",
                        "limit": 5,
                    }

                    resp = await client.get(url, params=params, headers=HEADERS, timeout=10)
                    log("tools", f"search_reddit: {sub} returned {resp.status_code}")

                    if resp.status_code == 429:
                        log("tools", f"search_reddit: Rate limited on {sub}")
                        await asyncio.sleep(2)
                        continue

                    if resp.status_code != 200:
                        continue

                    data = resp.json()
                    posts = data.get("data", {}).get("children", [])

                    for post in posts[:3]:
                        post_data = post.get("data", {})
                        results.append({
                            "subreddit": sub,
                            "title": post_data.get("title", "")[:200],
                            "score": post_data.get("score", 0),
                            "comments": post_data.get("num_comments", 0),
                            "url": f"https://reddit.com{post_data.get('permalink', '')}",
                        })

                    # Small delay between requests
                    await asyncio.sleep(0.5)

                except Exception as e:
                    log("tools", f"search_reddit: Error on {sub}: {e}", level="warning")
                    continue

            # Fallback to old.reddit.com scraping if JSON API failed
            if not results:
                log("tools", "search_reddit: JSON API failed, trying scraping fallback")
                for sub in subreddits[:2]:
                    try:
                        url = f"https://old.reddit.com/r/{sub}/search"
                        params = {"q": query, "restrict_sr": "on", "sort": "relevance", "t": "month"}
                        resp = await client.get(url, params=params, headers=HEADERS, timeout=10)

                        if resp.status_code != 200:
                            continue

                        soup = BeautifulSoup(resp.text, "html.parser")
                        for post in soup.select(".thing.link")[:3]:
                            title_el = post.select_one("a.title")
                            if title_el:
                                results.append({
                                    "subreddit": sub,
                                    "title": title_el.text.strip()[:200],
                                    "score": "?",
                                    "comments": "?",
                                })
                        await asyncio.sleep(0.5)
                    except Exception:
                        continue

            if not results:
                return ToolResult(
                    data={"error": f"No Reddit discussions found for '{query}'"},
                    confidence=0.0,
                    status=ToolStatus.FAILED,
                    tool_name="search_reddit",
                    ticker=query_upper,
                    error_message=f"No results found",
                )

            result_data = {
                "query": query,
                "source": "Reddit",
                "post_count": len(results),
                "posts": results,
            }

            # Cache successful results
            cache.set("reddit", query_upper, {
                "data": result_data,
                "confidence": 0.6,
                "status": "success",
                "source_url": f"https://www.reddit.com/search/?q={query}",
            })

            return ToolResult(
                data=result_data,
                confidence=0.6,  # Reddit sentiment is less structured
                status=ToolStatus.SUCCESS,
                source_url=f"https://www.reddit.com/search/?q={query}",
                tool_name="search_reddit",
                ticker=query_upper,
            )

    except httpx.TimeoutException:
        log("tools", f"search_reddit: Timeout for '{query}'")
        return ToolResult(
            data={},
            confidence=0.0,
            status=ToolStatus.FAILED,
            tool_name="search_reddit",
            ticker=query,
            error_message="Request timed out",
        )
    except Exception as e:
        log("tools", f"search_reddit: Error for '{query}': {e}", level="error")
        return ToolResult(
            data={},
            confidence=0.0,
            status=ToolStatus.FAILED,
            tool_name="search_reddit",
            ticker=query,
            error_message=str(e),
        )
