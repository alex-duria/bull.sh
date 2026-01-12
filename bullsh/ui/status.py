"""Pretty status indicators for CLI output."""

from typing import Any

# Tool descriptions for user-friendly display
TOOL_DESCRIPTIONS = {
    "sec_search": "Searching SEC filings",
    "sec_fetch": "Fetching SEC filing",
    "rag_search": "Searching indexed documents",
    "rag_list": "Listing indexed filings",
    "scrape_yahoo": "Getting market data",
    "search_stocktwits": "Checking social sentiment",
    "search_reddit": "Searching Reddit",
    "search_news": "Searching news",
    "web_search": "Searching the web",
    "compute_ratios": "Computing ratios",
    "generate_excel": "Generating Excel spreadsheet",
    "save_thesis": "Saving thesis",
}

# Status symbols
SYMBOLS = {
    "pending": "○",
    "running": "◐",
    "success": "✓",
    "partial": "◑",
    "failed": "✗",
    "cached": "⚡",
}

# ANSI color codes (Rich will handle these, but fallback for plain text)
COLORS = {
    "dim": "\033[2m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "cyan": "\033[36m",
    "reset": "\033[0m",
}


def format_tool_start(tool_name: str, params: dict[str, Any] | None = None) -> str:
    """Format tool start message."""
    desc = TOOL_DESCRIPTIONS.get(tool_name, tool_name)

    # Add context from params
    context = ""
    if params:
        if "ticker" in params:
            context = f" ({params['ticker'].upper()})"
        elif "symbol" in params:
            context = f" ({params['symbol'].upper()})"
        elif "query" in params:
            query = params["query"]
            if len(query) > 30:
                query = query[:27] + "..."
            context = f' "{query}"'

    return f"  {SYMBOLS['running']} {desc}{context}..."


def format_tool_result(
    tool_name: str,
    status: str,
    confidence: float = 1.0,
    cached: bool = False,
    error: str | None = None,
) -> str:
    """Format tool result message."""
    if cached:
        symbol = SYMBOLS["cached"]
        status_text = "cached"
    elif status == "success":
        symbol = SYMBOLS["success"]
        status_text = "done"
    elif status == "partial":
        symbol = SYMBOLS["partial"]
        status_text = f"partial ({confidence:.0%})"
    else:
        symbol = SYMBOLS["failed"]
        # Show truncated error message if available
        if error:
            error_short = error[:50] + "..." if len(error) > 50 else error
            status_text = f"failed: {error_short}"
        else:
            status_text = "failed"

    return f" {symbol} {status_text}"


def format_tool_line(
    tool_name: str,
    params: dict[str, Any] | None,
    status: str,
    confidence: float = 1.0,
    cached: bool = False,
) -> str:
    """Format complete tool execution line (start + result)."""
    desc = TOOL_DESCRIPTIONS.get(tool_name, tool_name)

    # Add context from params
    context = ""
    if params:
        if "ticker" in params:
            context = f" ({params['ticker'].upper()})"
        elif "symbol" in params:
            context = f" ({params['symbol'].upper()})"

    # Status indicator
    if cached:
        symbol = SYMBOLS["cached"]
        suffix = "cached"
    elif status == "success":
        symbol = SYMBOLS["success"]
        suffix = ""
    elif status == "partial":
        symbol = SYMBOLS["partial"]
        suffix = f"({confidence:.0%})"
    else:
        symbol = SYMBOLS["failed"]
        suffix = "failed"

    result = f"  {symbol} {desc}{context}"
    if suffix:
        result += f" [{suffix}]"
    return result


def format_indexing_status(ticker: str, chunks: int) -> str:
    """Format RAG indexing status."""
    return f"  {SYMBOLS['success']} Indexed {ticker} ({chunks} chunks)"


def format_model_loading() -> str:
    """Format model loading status."""
    return f"  {SYMBOLS['running']} Loading embedding model..."


def format_model_loaded() -> str:
    """Format model loaded status."""
    return f"  {SYMBOLS['success']} Embedding model ready"
