"""
Tengu FIRM tools - congressional trades and other "smart money" data.

Adds market-intelligence surfaces that Financial Datasets / Yahoo don't cover:
U.S. congressional trades (STOCK Act), and (easily extended) 13F institutional
positioning and insider clusters.

Opt-in: only active when TENGU_API_KEY is set, so default behavior is unchanged.
Free key: https://tengu.co/api
"""

import os
from typing import Any

import httpx

from bullsh.logging import log
from bullsh.tools.base import ToolResult, ToolStatus

# Tengu FIRM API base URL
FIRM_API_BASE = os.getenv("TENGU_FIRM_BASE", "https://firm.tengu.co")


async def _fetch(path: str, params: dict[str, Any], api_key: str) -> Any:
    """GET a FIRM endpoint with the X-API-Key header."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            f"{FIRM_API_BASE}{path}",
            params=params,
            headers={"X-API-Key": api_key, "accept": "application/json"},
        )
        response.raise_for_status()
        return response.json()


def _extract_list(payload: Any) -> list[Any]:
    """Pull the records list out of a FIRM response envelope."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "results", "trades", "items", "rows", "records"):
            val = payload.get(key)
            if isinstance(val, list):
                return val
            if isinstance(val, dict):
                for key2 in ("data", "results", "trades", "items", "rows"):
                    if isinstance(val.get(key2), list):
                        return val[key2]
    return []


async def get_congress_trades(ticker: str | None = None, limit: int = 50) -> ToolResult:
    """
    Get recent U.S. congressional trades (STOCK Act disclosures) from Tengu FIRM.

    Args:
        ticker: Optional ticker to filter by (e.g., NVDA).
        limit: Max disclosures to return (default: 50).

    Returns:
        ToolResult with the congressional trade disclosures.
    """
    log("tengu", f"Fetching congressional trades{f' for {ticker}' if ticker else ''}")

    api_key = os.getenv("TENGU_API_KEY")
    if not api_key:
        return ToolResult(
            data={
                "error": "Tengu API key not configured",
                "help": "Set TENGU_API_KEY in .env to enable congressional trade data. Free key: https://tengu.co/api",
            },
            confidence=0.0,
            status=ToolStatus.FAILED,
            tool_name="get_congress_trades",
            ticker=(ticker or "").upper(),
            error_message="API key not configured",
        )

    params: dict[str, Any] = {"limit": limit}
    if ticker:
        params["ticker"] = ticker.upper()

    try:
        payload = await _fetch("/api/v3/intel/congress", params, api_key)
        trades = _extract_list(payload)
        return ToolResult(
            data={
                "count": len(trades),
                "trades": trades[:limit],
                "ticker": (ticker or "ALL").upper(),
            },
            confidence=0.85 if trades else 0.4,
            status=ToolStatus.SUCCESS if trades else ToolStatus.PARTIAL,
            source_url="https://firm.tengu.co/api/v3/intel/congress",
            tool_name="get_congress_trades",
            ticker=(ticker or "").upper(),
        )
    except httpx.HTTPStatusError as e:
        code = e.response.status_code
        status = ToolStatus.RATE_LIMITED if code == 429 else ToolStatus.FAILED
        return ToolResult(
            data={"error": f"Tengu FIRM HTTP {code}"},
            confidence=0.0,
            status=status,
            tool_name="get_congress_trades",
            ticker=(ticker or "").upper(),
            error_message=f"HTTP {code}",
        )
    except Exception as e:  # noqa: BLE001 - surface any fetch error as a tool failure
        return ToolResult(
            data={"error": str(e)},
            confidence=0.0,
            status=ToolStatus.FAILED,
            tool_name="get_congress_trades",
            ticker=(ticker or "").upper(),
            error_message=str(e),
        )
