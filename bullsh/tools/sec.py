"""SEC EDGAR tools using edgartools."""

import asyncio
import contextlib
import io
import sys
from typing import Any

from bullsh.config import get_config
from bullsh.logging import log, log_cache_hit, log_cache_miss
from bullsh.storage.cache import get_cache
from bullsh.tools.base import ToolResult, ToolStatus

# Flag to control RAG auto-indexing
_rag_enabled = True

# edgartools imports - these will fail if not installed
try:
    from edgar import Company, set_identity

    EDGAR_AVAILABLE = True
except ImportError:
    EDGAR_AVAILABLE = False


@contextlib.contextmanager
def _suppress_stdout():
    """Suppress stdout/stderr from noisy libraries."""
    # Capture and discard stdout/stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def _ensure_edgar() -> None:
    """Ensure edgartools is available and identity is set."""
    if not EDGAR_AVAILABLE:
        raise RuntimeError("edgartools not installed. Run: pip install edgartools")

    config = get_config()
    set_identity(config.edgar_identity)


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


async def sec_search(ticker: str, fuzzy: bool = True) -> ToolResult:
    """
    Search SEC EDGAR for available filings.

    Args:
        ticker: Stock ticker symbol
        fuzzy: If True, search by company name if ticker not found

    Returns:
        ToolResult with list of available filings
    """
    ticker = ticker.upper()
    cache = get_cache()

    # Check cache first
    cached = cache.get("sec", ticker, action="search")
    if cached:
        log_cache_hit("sec", f"{ticker}/search")
        return _result_from_cache(cached, "sec_search", ticker)
    log_cache_miss("sec", f"{ticker}/search")

    _ensure_edgar()

    try:
        # Run in executor since edgartools is sync
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            _sec_search_sync,
            ticker,
            fuzzy,
        )

        # Cache successful results
        if result.status == ToolStatus.SUCCESS:
            cache.set(
                "sec",
                ticker,
                {
                    "data": result.data,
                    "confidence": result.confidence,
                    "status": result.status.value,
                    "source_url": result.source_url,
                },
                action="search",
            )

        return result
    except Exception as e:
        return ToolResult(
            data={},
            confidence=0.0,
            status=ToolStatus.FAILED,
            tool_name="sec_search",
            ticker=ticker,
            error_message=str(e),
        )


def _sec_search_sync(ticker: str, fuzzy: bool) -> ToolResult:
    """Synchronous SEC search."""
    try:
        company = Company(ticker)
    except Exception:
        if fuzzy:
            # TODO: Implement fuzzy search by company name
            pass
        return ToolResult(
            data={"error": f"Company not found: {ticker}"},
            confidence=0.0,
            status=ToolStatus.FAILED,
            tool_name="sec_search",
            ticker=ticker,
            error_message=f"Company not found: {ticker}",
        )

    # Get recent filings
    try:
        filings_10k = list(company.get_filings(form="10-K").head(5))
        filings_10q = list(company.get_filings(form="10-Q").head(5))
        filings_8k = list(company.get_filings(form="8-K").head(5))
    except Exception as e:
        return ToolResult(
            data={},
            confidence=0.0,
            status=ToolStatus.FAILED,
            tool_name="sec_search",
            ticker=ticker,
            error_message=f"Failed to fetch filings: {e}",
        )

    filings_data = {
        "company_name": company.name,
        "cik": company.cik,
        "ticker": ticker.upper(),
        "filings": {
            "10-K": [
                {
                    "filing_date": str(f.filing_date),
                    "accession_number": f.accession_number,
                }
                for f in filings_10k
            ],
            "10-Q": [
                {
                    "filing_date": str(f.filing_date),
                    "accession_number": f.accession_number,
                }
                for f in filings_10q
            ],
            "8-K": [
                {
                    "filing_date": str(f.filing_date),
                    "accession_number": f.accession_number,
                }
                for f in filings_8k
            ],
        },
    }

    has_10k = len(filings_10k) > 0

    return ToolResult(
        data=filings_data,
        confidence=1.0 if has_10k else 0.5,
        status=ToolStatus.SUCCESS if has_10k else ToolStatus.PARTIAL,
        source_url=f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=10-K",
        tool_name="sec_search",
        ticker=ticker.upper(),
    )


async def sec_fetch(
    ticker: str,
    filing_type: str,
    year: int | None = None,
    section: str | None = None,
) -> ToolResult:
    """
    Fetch and parse a SEC filing.

    Args:
        ticker: Stock ticker symbol
        filing_type: "10-K" or "10-Q"
        year: Optional year, defaults to latest
        section: Optional specific section to extract

    Returns:
        ToolResult with filing content organized by section
    """
    ticker = ticker.upper()
    cache = get_cache()

    # Build cache key with all parameters
    cache_params = {"type": filing_type}
    if year:
        cache_params["year"] = year
    if section:
        cache_params["section"] = section

    # Check cache first
    cached = cache.get("sec", ticker, action="fetch", **cache_params)
    if cached:
        log_cache_hit("sec", f"{ticker}/fetch/{filing_type}")
        return _result_from_cache(cached, "sec_fetch", ticker)
    log_cache_miss("sec", f"{ticker}/fetch/{filing_type}")

    _ensure_edgar()

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            _sec_fetch_sync,
            ticker,
            filing_type,
            year,
            section,
        )

        # Cache successful results
        if result.status in (ToolStatus.SUCCESS, ToolStatus.PARTIAL):
            # Don't cache full_text - it's huge and only needed for RAG indexing
            cache_data = {k: v for k, v in result.data.items() if k != "full_text"}
            cache.set(
                "sec",
                ticker,
                {
                    "data": cache_data,
                    "confidence": result.confidence,
                    "status": result.status.value,
                    "source_url": result.source_url,
                },
                action="fetch",
                **cache_params,
            )

            # Auto-index into vector database for RAG (uses full_text)
            if _rag_enabled and ("full_text" in result.data or "text" in result.data):
                await _auto_index_for_rag(result.data, ticker, filing_type)

            # Strip full_text from result before returning to agent (already indexed, don't bloat context)
            if "full_text" in result.data:
                del result.data["full_text"]

        return result
    except Exception as e:
        return ToolResult(
            data={},
            confidence=0.0,
            status=ToolStatus.FAILED,
            tool_name="sec_fetch",
            ticker=ticker,
            error_message=str(e),
        )


def _sec_fetch_sync(
    ticker: str,
    filing_type: str,
    year: int | None,
    section: str | None,
) -> ToolResult:
    """Synchronous SEC fetch - uses filing.text() like tenk does."""
    config = get_config()
    max_chars = config.tool_result_max_chars  # Use configurable limit

    try:
        log("tools", f"sec_fetch: Looking up {ticker} {filing_type}")
        company = Company(ticker)
        filings = company.get_filings(form=filing_type)

        if year:
            # Find filing from specific year
            filing = None
            for f in filings:
                if f.filing_date.year == year:
                    filing = f
                    break
            if not filing:
                return ToolResult(
                    data={},
                    confidence=0.0,
                    status=ToolStatus.FAILED,
                    tool_name="sec_fetch",
                    ticker=ticker,
                    error_message=f"No {filing_type} found for {ticker} in {year}",
                )
        else:
            # Get latest
            filing = filings.latest(1)
            if not filing:
                return ToolResult(
                    data={},
                    confidence=0.0,
                    status=ToolStatus.FAILED,
                    tool_name="sec_fetch",
                    ticker=ticker,
                    error_message=f"No {filing_type} found for {ticker}",
                )
            # latest() returns a Filings object, get first
            if hasattr(filing, "__iter__"):
                filing = list(filing)[0]

        log("tools", f"sec_fetch: Found filing dated {filing.filing_date}")

    except Exception as e:
        log("tools", f"sec_fetch: Failed to find filing: {e}", level="error")
        return ToolResult(
            data={},
            confidence=0.0,
            status=ToolStatus.FAILED,
            tool_name="sec_fetch",
            ticker=ticker,
            error_message=f"Failed to fetch filing: {e}",
        )

    # Extract text using filing.text() - simple approach like tenk
    try:
        log("tools", "sec_fetch: Extracting text via filing.text()")

        with _suppress_stdout():
            # Use .text() method directly - this is what tenk does
            # It handles all the HTML/XML parsing internally
            if hasattr(filing, "text"):
                if callable(filing.text):
                    full_text = filing.text()
                else:
                    full_text = filing.text
            else:
                # Fallback: try to get document object
                doc = filing.obj() if hasattr(filing, "obj") else filing
                if hasattr(doc, "text"):
                    full_text = doc.text() if callable(doc.text) else doc.text
                else:
                    full_text = str(doc)

        log("tools", f"sec_fetch: Got {len(full_text)} chars of text")

        # Build URL for citations
        filing_url = getattr(filing, "url", None)
        if not filing_url:
            filing_url = f"https://www.sec.gov/Archives/edgar/data/{company.cik}/{filing.accession_number.replace('-', '')}"

        result_data = {
            "ticker": ticker.upper(),
            "filing_type": filing_type,
            "filing_date": str(filing.filing_date),
            "accession_number": filing.accession_number,
            "url": filing_url,
            "text": _truncate(full_text, max_chars),
            "full_text": full_text,  # Keep full text for RAG indexing
            "full_length": len(full_text),
        }

        log(
            "tools",
            f"sec_fetch: Success - {len(full_text)} chars, truncated to {min(len(full_text), max_chars)}",
        )

        return ToolResult(
            data=result_data,
            confidence=1.0,
            status=ToolStatus.SUCCESS,
            source_url=filing_url,
            tool_name="sec_fetch",
            ticker=ticker.upper(),
        )

    except Exception as e:
        log("tools", f"sec_fetch: Exception extracting text: {e}", level="error")
        return ToolResult(
            data={},
            confidence=0.0,
            status=ToolStatus.FAILED,
            tool_name="sec_fetch",
            ticker=ticker,
            error_message=f"Failed to extract filing text: {e}",
        )


def _truncate(text: str, max_chars: int) -> str:
    """Truncate text to max chars, adding indicator if truncated."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n[...truncated, {len(text) - max_chars} more characters...]"


async def _auto_index_for_rag(data: dict[str, Any], ticker: str, filing_type: str) -> None:
    """Auto-index a fetched filing into the vector database for RAG search."""
    try:
        from bullsh.tools.rag import rag_index

        # Extract year from filing date
        filing_date = data.get("filing_date", "")
        year = int(filing_date[:4]) if filing_date and len(filing_date) >= 4 else 0

        # IMPORTANT: Use full_text for RAG indexing, not the truncated text
        # The truncated text only includes the first N chars (for prompt context)
        # but RAG needs the entire filing to find Risk Factors, MD&A, etc.
        text = data.get("full_text") or data.get("text", "")
        if not text:
            return

        # Index the filing
        result = await rag_index(
            ticker=ticker,
            form=filing_type,
            year=year,
            text=text,
            url=data.get("url"),
            filing_date=filing_date,
        )

        if result.status == ToolStatus.SUCCESS:
            log(
                "tools",
                f"Auto-indexed {ticker} {filing_type} {year} for RAG ({result.data.get('chunks', 0)} chunks)",
            )
        elif result.status == ToolStatus.CACHED:
            log("tools", f"Filing {ticker} {filing_type} {year} already indexed for RAG")

    except ImportError:
        # RAG dependencies not installed - silently skip
        log("tools", "RAG dependencies not installed, skipping auto-index")
    except Exception as e:
        # Don't fail the main request if RAG indexing fails
        log("tools", f"Failed to auto-index for RAG: {e}", level="warning")
