"""Logging module - file-based debug logging like Claude Code."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Global debug state
_debug_enabled = False
_debug_categories: set[str] = set()
_excluded_categories: set[str] = set()
_log_file: Path | None = None
_file_handler: logging.FileHandler | None = None

# Categories for filtering
CATEGORIES = {
    "tools",  # Tool calls and results
    "api",  # Claude API requests/responses
    "cache",  # Cache hits/misses
    "session",  # Session management
    "config",  # Configuration loading
    "orchestrator",  # Agent orchestrator
}


def setup_logging(
    logs_dir: Path,
    debug: bool = False,
    debug_filter: str | None = None,
) -> Path:
    """
    Set up file-based logging.

    Args:
        logs_dir: Directory for log files (~/.bullsh/logs/)
        debug: Enable debug logging
        debug_filter: Optional category filter (e.g., "tools,api" or "!cache")

    Returns:
        Path to the log file
    """
    global _debug_enabled, _debug_categories, _excluded_categories, _log_file, _file_handler

    _debug_enabled = debug

    # Parse category filter
    if debug_filter:
        for cat in debug_filter.split(","):
            cat = cat.strip()
            if cat.startswith("!"):
                _excluded_categories.add(cat[1:])
            else:
                _debug_categories.add(cat)

    # If no specific categories, include all
    if not _debug_categories:
        _debug_categories = CATEGORIES.copy()

    # Remove excluded
    _debug_categories -= _excluded_categories

    # Create logs directory
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create logs directory: {e}", file=sys.stderr)
        # Fall back to temp directory
        import tempfile

        logs_dir = Path(tempfile.gettempdir()) / "bullsh_logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_file = logs_dir / f"debug_{timestamp}.log"

    try:
        # Set up file handler
        _file_handler = logging.FileHandler(str(_log_file), encoding="utf-8")
        _file_handler.setLevel(logging.DEBUG)
        _file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
            )
        )

        # Configure root logger to only use file handler (not console)
        root_logger = logging.getLogger("bullsh")
        root_logger.setLevel(logging.DEBUG if debug else logging.INFO)
        root_logger.addHandler(_file_handler)

        # Don't propagate to root (prevents console output)
        root_logger.propagate = False

        # Log startup - write directly to ensure it works
        if debug:
            with open(_log_file, "a", encoding="utf-8") as f:
                f.write(f"=== Debug logging started at {datetime.now().isoformat()} ===\n")
                f.write(f"Categories: {_debug_categories}\n")
                f.write(f"Log file: {_log_file}\n\n")

    except Exception as e:
        print(f"Warning: Could not set up file logging: {e}", file=sys.stderr)

    return _log_file


def log(category: str, message: str, level: str = "debug", **kwargs: Any) -> None:
    """
    Log a message to file (not console).

    Args:
        category: Log category (tools, api, cache, etc.)
        message: Log message
        level: Log level (debug, info, warning, error)
        **kwargs: Additional context to log
    """
    if not _debug_enabled:
        return

    if category not in _debug_categories:
        return

    logger = logging.getLogger(f"bullsh.{category}")

    # Format message with kwargs
    if kwargs:
        extras = " | ".join(f"{k}={v}" for k, v in kwargs.items())
        message = f"{message} | {extras}"

    log_func = getattr(logger, level, logger.debug)
    log_func(message)


def log_tool_call(tool_name: str, params: dict[str, Any]) -> None:
    """Log a tool call."""
    log("tools", f"CALL {tool_name}", params=params)


def log_tool_result(
    tool_name: str, status: str, confidence: float, error: str | None = None
) -> None:
    """Log a tool result."""
    if error:
        log("tools", f"RESULT {tool_name}", level="warning", status=status, error=error)
    else:
        log("tools", f"RESULT {tool_name}", status=status, confidence=f"{confidence:.0%}")


def log_api_call(model: str, input_tokens: int, output_tokens: int, cached: int = 0) -> None:
    """Log a Claude API call."""
    log(
        "api",
        "Claude API call",
        model=model,
        input=input_tokens,
        output=output_tokens,
        cached=cached,
    )


def log_cache_hit(source: str, key: str) -> None:
    """Log a cache hit."""
    log("cache", f"HIT {source}", key=key)


def log_cache_miss(source: str, key: str) -> None:
    """Log a cache miss."""
    log("cache", f"MISS {source}", key=key)


def get_log_file() -> Path | None:
    """Get the current log file path."""
    return _log_file


def is_debug_enabled() -> bool:
    """Check if debug logging is enabled."""
    return _debug_enabled
