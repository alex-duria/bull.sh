"""Data fetchers for factor analysis.

Fetches price history (yfinance) and Fama-French factor returns.
All data is cached using existing cache infrastructure.
"""

import asyncio
import io
import zipfile
from datetime import datetime
from typing import Any

import pandas as pd

from bullsh.logging import log, log_cache_hit, log_cache_miss
from bullsh.storage.cache import get_cache
from bullsh.tools.base import ToolResult, ToolStatus

# yfinance for price history
try:
    import yfinance as yf

    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# For HTTP requests
try:
    import httpx  # noqa: F401 - used conditionally below

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


# Fama-French data URL
FAMA_FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"


async def fetch_price_history(
    ticker: str,
    period: str = "3y",
) -> ToolResult:
    """
    Fetch historical price data using yfinance.

    Args:
        ticker: Stock ticker symbol
        period: Data period (e.g., "3y", "5y", "max")

    Returns:
        ToolResult with data containing:
        - closes: list of (date_str, close_price) tuples
        - opens, highs, lows, volumes: similar lists
        - returns: daily returns as list

    Cache TTL: 24 hours (end-of-day refresh)
    """
    ticker = ticker.upper()
    cache = get_cache()

    # Check cache first
    cache_key_params = {"period": period}
    cached = cache.get("price_history", ticker, **cache_key_params)
    if cached:
        log_cache_hit("price_history", ticker)
        return ToolResult(
            data=cached.get("data", {}),
            confidence=cached.get("confidence", 1.0),
            status=ToolStatus(cached.get("status", "success")),
            tool_name="fetch_price_history",
            ticker=ticker,
            cached=True,
        )
    log_cache_miss("price_history", ticker)

    if not YFINANCE_AVAILABLE:
        return ToolResult(
            data={},
            confidence=0.0,
            status=ToolStatus.FAILED,
            tool_name="fetch_price_history",
            ticker=ticker,
            error_message="yfinance not installed. Run: pip install yfinance",
        )

    try:
        log("tools", f"fetch_price_history: Fetching {period} history for {ticker}")

        # Run yfinance in executor since it's synchronous
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(
            None,
            _fetch_price_history_sync,
            ticker,
            period,
        )

        if not data or not data.get("closes"):
            return ToolResult(
                data={"ticker": ticker},
                confidence=0.0,
                status=ToolStatus.FAILED,
                tool_name="fetch_price_history",
                ticker=ticker,
                error_message=f"No price history found for {ticker}",
            )

        # Calculate confidence based on data completeness
        expected_days = {"1y": 252, "2y": 504, "3y": 756, "5y": 1260}.get(period, 500)
        actual_days = len(data.get("closes", []))
        confidence = min(1.0, actual_days / expected_days)

        log("tools", f"fetch_price_history: Got {actual_days} days for {ticker}")

        # Cache with 24-hour TTL
        cache.set(
            "price_history",
            ticker,
            {
                "data": data,
                "confidence": confidence,
                "status": "success",
            },
            ttl_hours=24,
            **cache_key_params,
        )

        return ToolResult(
            data=data,
            confidence=confidence,
            status=ToolStatus.SUCCESS,
            tool_name="fetch_price_history",
            ticker=ticker,
        )

    except Exception as e:
        log("tools", f"fetch_price_history: Error for {ticker}: {e}", level="error")
        return ToolResult(
            data={},
            confidence=0.0,
            status=ToolStatus.FAILED,
            tool_name="fetch_price_history",
            ticker=ticker,
            error_message=str(e),
        )


def _fetch_price_history_sync(ticker: str, period: str) -> dict[str, Any]:
    """Synchronous price history fetch using yfinance."""
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)

    if hist.empty:
        return {}

    # Convert to serializable format
    # Dates as ISO strings, prices as floats
    closes = [(dt.strftime("%Y-%m-%d"), float(row["Close"])) for dt, row in hist.iterrows()]

    opens = [(dt.strftime("%Y-%m-%d"), float(row["Open"])) for dt, row in hist.iterrows()]

    highs = [(dt.strftime("%Y-%m-%d"), float(row["High"])) for dt, row in hist.iterrows()]

    lows = [(dt.strftime("%Y-%m-%d"), float(row["Low"])) for dt, row in hist.iterrows()]

    volumes = [(dt.strftime("%Y-%m-%d"), int(row["Volume"])) for dt, row in hist.iterrows()]

    # Calculate daily returns
    hist["Return"] = hist["Close"].pct_change()
    returns = [
        (dt.strftime("%Y-%m-%d"), float(row["Return"]))
        for dt, row in hist.iterrows()
        if pd.notna(row["Return"])
    ]

    return {
        "ticker": ticker,
        "period": period,
        "closes": closes,
        "opens": opens,
        "highs": highs,
        "lows": lows,
        "volumes": volumes,
        "returns": returns,
        "start_date": closes[0][0] if closes else None,
        "end_date": closes[-1][0] if closes else None,
        "trading_days": len(closes),
    }


async def fetch_fama_french() -> ToolResult:
    """
    Fetch Fama-French daily factor returns from Ken French Data Library.

    Returns:
        ToolResult with data containing:
        - factors: dict mapping date_str -> {Mkt-RF, SMB, HML, RF}
        - start_date, end_date: date range
        - factor_names: list of available factors

    Cache TTL: 7 days (data updates monthly)
    """
    cache = get_cache()

    # Check cache first
    cached = cache.get("fama_french", "factors_daily")
    if cached:
        log_cache_hit("fama_french", "factors_daily")
        return ToolResult(
            data=cached.get("data", {}),
            confidence=cached.get("confidence", 1.0),
            status=ToolStatus(cached.get("status", "success")),
            tool_name="fetch_fama_french",
            ticker="FF",
            cached=True,
        )
    log_cache_miss("fama_french", "factors_daily")

    if not HTTPX_AVAILABLE:
        return ToolResult(
            data={},
            confidence=0.0,
            status=ToolStatus.FAILED,
            tool_name="fetch_fama_french",
            ticker="FF",
            error_message="httpx not installed. Run: pip install httpx",
        )

    try:
        log("tools", "fetch_fama_french: Fetching from Ken French Data Library")

        # Fetch and parse in executor
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, _fetch_fama_french_sync)

        if not data or not data.get("factors"):
            return ToolResult(
                data={},
                confidence=0.0,
                status=ToolStatus.FAILED,
                tool_name="fetch_fama_french",
                ticker="FF",
                error_message="Failed to parse Fama-French data",
            )

        log("tools", f"fetch_fama_french: Got {len(data.get('factors', {}))} days of factor data")

        # Cache with 7-day TTL
        cache.set(
            "fama_french",
            "factors_daily",
            {
                "data": data,
                "confidence": 1.0,
                "status": "success",
            },
            ttl_hours=24 * 7,
        )

        return ToolResult(
            data=data,
            confidence=1.0,
            status=ToolStatus.SUCCESS,
            tool_name="fetch_fama_french",
            ticker="FF",
        )

    except Exception as e:
        log("tools", f"fetch_fama_french: Error: {e}", level="error")
        return ToolResult(
            data={},
            confidence=0.0,
            status=ToolStatus.FAILED,
            tool_name="fetch_fama_french",
            ticker="FF",
            error_message=str(e),
        )


def _fetch_fama_french_sync() -> dict[str, Any]:
    """Synchronous Fama-French data fetch."""
    import httpx

    # Download ZIP file
    response = httpx.get(FAMA_FRENCH_URL, follow_redirects=True, timeout=30)
    response.raise_for_status()

    # Extract CSV from ZIP
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        # Find the CSV file in the archive
        csv_files = [f for f in zf.namelist() if f.endswith(".CSV")]
        if not csv_files:
            return {}

        csv_content = zf.read(csv_files[0]).decode("utf-8")

    # Parse CSV
    # Format: Date, Mkt-RF, SMB, HML, RF
    # First few rows are header text, data starts after blank line
    lines = csv_content.strip().split("\n")

    # Find start of daily data
    data_start = 0
    for i, line in enumerate(lines):
        if line.strip() and line[0].isdigit():
            data_start = i
            break

    # Find end of daily data (before annual section)
    data_end = len(lines)
    for i in range(data_start, len(lines)):
        line = lines[i].strip()
        if not line or "Annual" in line or len(line.split(",")[0]) == 4:
            # Hit annual data or blank line
            data_end = i
            break

    # Parse data rows
    factors = {}
    for line in lines[data_start:data_end]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            try:
                # Date format: YYYYMMDD
                date_str = parts[0]
                if len(date_str) == 8:
                    date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

                    factors[date_formatted] = {
                        "Mkt-RF": float(parts[1]) / 100,  # Convert from percent
                        "SMB": float(parts[2]) / 100,
                        "HML": float(parts[3]) / 100,
                        "RF": float(parts[4]) / 100,
                    }
            except (ValueError, IndexError):
                continue

    if not factors:
        return {}

    dates = sorted(factors.keys())
    return {
        "factors": factors,
        "start_date": dates[0],
        "end_date": dates[-1],
        "factor_names": ["Mkt-RF", "SMB", "HML", "RF"],
        "trading_days": len(factors),
    }


async def fetch_all_factor_data(
    primary_ticker: str,
    peers: list[str],
    benchmark: str = "^GSPC",
    period: str = "3y",
) -> dict[str, Any]:
    """
    Fetch all data needed for factor analysis in parallel.

    This is the main entry point for Stage 4 data fetching.

    Args:
        primary_ticker: Primary stock to analyze
        peers: List of peer tickers
        benchmark: Benchmark index (default S&P 500)
        period: Historical data period

    Returns:
        Dict with:
        - yahoo_data: {ticker: yahoo data dict}
        - price_history: {ticker: price history dict}
        - fama_french: Fama-French factor data
        - benchmark_history: Benchmark price history
    """
    from bullsh.tools.yahoo import scrape_yahoo

    all_tickers = [primary_ticker] + peers

    # Create all fetch tasks
    yahoo_tasks = [scrape_yahoo(t) for t in all_tickers]
    history_tasks = [fetch_price_history(t, period) for t in all_tickers]
    benchmark_task = fetch_price_history(benchmark, period)
    ff_task = fetch_fama_french()

    # Execute all in parallel
    all_results = await asyncio.gather(
        *yahoo_tasks,
        *history_tasks,
        benchmark_task,
        ff_task,
        return_exceptions=True,
    )

    # Unpack results
    n_tickers = len(all_tickers)
    yahoo_results = all_results[:n_tickers]
    history_results = all_results[n_tickers : 2 * n_tickers]
    benchmark_result = all_results[2 * n_tickers]
    ff_result = all_results[2 * n_tickers + 1]

    # Organize into dicts
    yahoo_data = {}
    for ticker, result in zip(all_tickers, yahoo_results):
        if isinstance(result, ToolResult) and result.status == ToolStatus.SUCCESS:
            yahoo_data[ticker] = result.data
        elif isinstance(result, Exception):
            log(
                "tools",
                f"fetch_all_factor_data: Yahoo error for {ticker}: {result}",
                level="warning",
            )
            yahoo_data[ticker] = {}
        else:
            yahoo_data[ticker] = {}

    price_history = {}
    for ticker, result in zip(all_tickers, history_results):
        if isinstance(result, ToolResult) and result.status == ToolStatus.SUCCESS:
            price_history[ticker] = result.data
        elif isinstance(result, Exception):
            log(
                "tools",
                f"fetch_all_factor_data: History error for {ticker}: {result}",
                level="warning",
            )
            price_history[ticker] = {}
        else:
            price_history[ticker] = {}

    # Benchmark
    benchmark_history = {}
    if isinstance(benchmark_result, ToolResult) and benchmark_result.status == ToolStatus.SUCCESS:
        benchmark_history = benchmark_result.data

    # Fama-French
    fama_french = {}
    if isinstance(ff_result, ToolResult) and ff_result.status == ToolStatus.SUCCESS:
        fama_french = ff_result.data

    return {
        "yahoo_data": yahoo_data,
        "price_history": price_history,
        "benchmark_history": benchmark_history,
        "fama_french": fama_french,
        "fetch_timestamp": datetime.now().isoformat(),
    }


def validate_data_completeness(
    data: dict[str, Any],
    required_tickers: list[str],
) -> tuple[bool, list[str]]:
    """
    Validate that fetched data is complete enough for analysis.

    Returns:
        (is_valid, list of issues)
    """
    issues = []

    yahoo_data = data.get("yahoo_data", {})
    price_history = data.get("price_history", {})

    for ticker in required_tickers:
        # Check Yahoo data
        if ticker not in yahoo_data or not yahoo_data[ticker]:
            issues.append(f"Missing Yahoo data for {ticker}")
        elif not yahoo_data[ticker].get("market_cap"):
            issues.append(f"Missing market cap for {ticker}")

        # Check price history
        if ticker not in price_history or not price_history[ticker]:
            issues.append(f"Missing price history for {ticker}")
        elif len(price_history[ticker].get("closes", [])) < 252:
            issues.append(f"Less than 1 year of price history for {ticker}")

    # Check Fama-French (optional but warn)
    if not data.get("fama_french", {}).get("factors"):
        issues.append("Fama-French factor data unavailable (regression will be skipped)")

    is_valid = len([i for i in issues if "Missing" in i]) == 0
    return is_valid, issues
