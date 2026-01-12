"""Yahoo Finance tools using yfinance library."""

import asyncio
from typing import Any

from bullsh.logging import log, log_cache_hit, log_cache_miss
from bullsh.storage.cache import get_cache
from bullsh.tools.base import ToolResult, ToolStatus

# yfinance is more reliable than scraping
try:
    import yfinance as yf

    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


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


async def scrape_yahoo(ticker: str) -> ToolResult:
    """
    Get stock data from Yahoo Finance using yfinance library.

    Much more reliable than web scraping - uses official Yahoo Finance API.
    """
    ticker = ticker.upper()
    cache = get_cache()

    # Check cache first (Yahoo data cached for 1 hour)
    cached = cache.get("yahoo", ticker)
    if cached:
        log_cache_hit("yahoo", ticker)
        return _result_from_cache(cached, "scrape_yahoo", ticker)
    log_cache_miss("yahoo", ticker)

    if not YFINANCE_AVAILABLE:
        return ToolResult(
            data={},
            confidence=0.0,
            status=ToolStatus.FAILED,
            tool_name="scrape_yahoo",
            ticker=ticker,
            error_message="yfinance not installed. Run: pip install yfinance",
        )

    try:
        log("tools", f"scrape_yahoo: Fetching data for {ticker}")

        # Run yfinance in executor since it's synchronous
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, _fetch_yahoo_sync, ticker)

        if not data:
            return ToolResult(
                data={"ticker": ticker},
                confidence=0.0,
                status=ToolStatus.FAILED,
                tool_name="scrape_yahoo",
                ticker=ticker,
                error_message=f"No data found for {ticker}",
            )

        # Calculate confidence based on how much data we got
        key_fields = ["price", "pe_ratio", "market_cap", "52w_high", "52w_low"]
        filled = sum(1 for k in key_fields if data.get(k) is not None)
        confidence = filled / len(key_fields)

        log("tools", f"scrape_yahoo: Got {filled}/{len(key_fields)} key fields for {ticker}")

        # Cache successful results
        cache.set(
            "yahoo",
            ticker,
            {
                "data": data,
                "confidence": confidence,
                "status": "success",
                "source_url": f"https://finance.yahoo.com/quote/{ticker}",
            },
        )

        return ToolResult(
            data=data,
            confidence=confidence,
            status=ToolStatus.SUCCESS,
            source_url=f"https://finance.yahoo.com/quote/{ticker}",
            tool_name="scrape_yahoo",
            ticker=ticker,
        )

    except Exception as e:
        log("tools", f"scrape_yahoo: Error for {ticker}: {e}", level="error")
        return ToolResult(
            data={},
            confidence=0.0,
            status=ToolStatus.FAILED,
            tool_name="scrape_yahoo",
            ticker=ticker,
            error_message=str(e),
        )


def _fetch_yahoo_sync(ticker: str) -> dict[str, Any]:
    """Synchronous Yahoo Finance fetch using yfinance."""
    try:
        stock = yf.Ticker(ticker)

        # yfinance 0.2.x changed how info works - try fast_info first
        try:
            fast = stock.fast_info
            log(
                "tools",
                f"scrape_yahoo: fast_info available, price={getattr(fast, 'last_price', None)}",
            )
        except Exception as e:
            log("tools", f"scrape_yahoo: fast_info failed: {e}", level="warning")
            fast = None

        info = stock.info
        log("tools", f"scrape_yahoo: yfinance returned {len(info) if info else 0} fields")

        if not info:
            log(
                "tools", f"scrape_yahoo: yfinance returned empty info for {ticker}", level="warning"
            )
            return {}

        # Check for common error indicators
        if info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
            log(
                "tools",
                f"scrape_yahoo: No price data for {ticker}. Keys: {list(info.keys())[:10]}",
                level="warning",
            )
            # Try to return what we have anyway
            if not any(info.get(k) for k in ["marketCap", "sector", "industry"]):
                return {}

        # Try to get price from fast_info if regular info doesn't have it
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if not price and fast:
            try:
                price = getattr(fast, "last_price", None) or getattr(fast, "previous_close", None)
                log("tools", f"scrape_yahoo: Using fast_info price: {price}")
            except Exception:
                pass

        # Get market cap from fast_info if not in info
        market_cap = info.get("marketCap")
        if not market_cap and fast:
            try:
                market_cap = getattr(fast, "market_cap", None)
            except Exception:
                pass

        return {
            "ticker": ticker,
            "price": price,
            "previous_close": info.get("previousClose")
            or (getattr(fast, "previous_close", None) if fast else None),
            "change": info.get("regularMarketChange"),
            "change_percent": info.get("regularMarketChangePercent"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "eps": info.get("trailingEps"),
            "52w_high": info.get("fiftyTwoWeekHigh")
            or (getattr(fast, "year_high", None) if fast else None),
            "52w_low": info.get("fiftyTwoWeekLow")
            or (getattr(fast, "year_low", None) if fast else None),
            "market_cap": market_cap,
            "volume": info.get("volume"),
            "avg_volume": info.get("averageVolume"),
            "dividend_yield": info.get("dividendYield"),
            "beta": info.get("beta"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "recommendation": info.get("recommendationKey"),
            "target_mean_price": info.get("targetMeanPrice"),
            "target_high_price": info.get("targetHighPrice"),
            "target_low_price": info.get("targetLowPrice"),
            "number_of_analysts": info.get("numberOfAnalystOpinions"),
        }
    except Exception as e:
        log("tools", f"scrape_yahoo: Exception in _fetch_yahoo_sync: {e}", level="error")
        raise


async def compute_ratios(ticker: str) -> ToolResult:
    """
    Compute key financial ratios from Yahoo Finance data.

    Returns P/E, EV/EBITDA, revenue growth, margins.
    """
    # First get Yahoo data
    yahoo_result = await scrape_yahoo(ticker)

    if yahoo_result.status == ToolStatus.FAILED:
        return ToolResult(
            data={},
            confidence=0.0,
            status=ToolStatus.FAILED,
            tool_name="compute_ratios",
            ticker=ticker,
            error_message=f"Failed to fetch Yahoo data: {yahoo_result.error_message}",
        )

    ratios: dict[str, Any] = {"ticker": ticker.upper()}
    confidence_factors = 0

    # Extract available ratios from Yahoo data
    if "pe_ratio" in yahoo_result.data:
        try:
            pe = yahoo_result.data["pe_ratio"]
            if pe and pe != "N/A":
                ratios["pe_ratio"] = pe
                confidence_factors += 1
        except Exception:
            pass

    # Note: Full ratio computation would require more data sources
    # For now, we return what we can extract from Yahoo

    ratios["note"] = "Additional ratios require 10-K data - use sec_fetch for detailed financials"

    return ToolResult(
        data=ratios,
        confidence=yahoo_result.confidence * 0.5,  # Reduce confidence since we're limited
        status=ToolStatus.PARTIAL,
        source_url=f"https://finance.yahoo.com/quote/{ticker}",
        tool_name="compute_ratios",
        ticker=ticker.upper(),
    )
