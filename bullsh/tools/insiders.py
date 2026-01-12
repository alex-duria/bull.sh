"""
Insider Transactions Tool - fetches insider buy/sell activity.

Uses Financial Datasets API for insider transaction data.
"""

import os
from typing import Any

import httpx

from bullsh.logging import log
from bullsh.tools.base import ToolResult, ToolStatus

# Financial Datasets API base URL
FD_API_BASE = "https://api.financialdatasets.ai"


async def get_insider_transactions(
    ticker: str,
    limit: int = 50,
) -> ToolResult:
    """
    Get recent insider transactions for a company.

    Fetches buy/sell/gift transactions from company insiders
    (executives, directors, major shareholders).

    Args:
        ticker: Stock ticker symbol
        limit: Maximum number of transactions to return (default: 50)

    Returns:
        ToolResult with insider transaction data
    """
    log("insiders", f"Fetching insider transactions for {ticker}")

    ticker = ticker.upper()
    api_key = os.getenv("FINANCIAL_DATASETS_API_KEY")

    if not api_key:
        log("insiders", "FINANCIAL_DATASETS_API_KEY not configured", level="warning")
        return ToolResult(
            data={
                "error": "Financial Datasets API key not configured",
                "help": "Set FINANCIAL_DATASETS_API_KEY in .env to enable insider transaction data",
            },
            confidence=0.0,
            status=ToolStatus.FAILED,
            tool_name="get_insider_transactions",
            ticker=ticker,
            error_message="API key not configured",
        )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            url = f"{FD_API_BASE}/financials/insider-transactions"
            params = {
                "ticker": ticker,
                "limit": limit,
            }

            response = await client.get(
                url,
                params=params,
                headers={"X-API-KEY": api_key},
            )
            response.raise_for_status()
            data = response.json()

            transactions = data.get("insider_transactions", [])

            if not transactions:
                return ToolResult(
                    data={
                        "ticker": ticker,
                        "transactions": [],
                        "summary": "No recent insider transactions found",
                    },
                    confidence=0.8,
                    status=ToolStatus.SUCCESS,
                    tool_name="get_insider_transactions",
                    ticker=ticker,
                )

            # Process and summarize transactions
            summary = _summarize_transactions(transactions)

            return ToolResult(
                data={
                    "ticker": ticker,
                    "transactions": transactions,
                    "summary": summary,
                },
                confidence=0.95,
                status=ToolStatus.SUCCESS,
                tool_name="get_insider_transactions",
                ticker=ticker,
                source_url=f"https://financialdatasets.ai/stocks/{ticker}",
            )

    except httpx.HTTPStatusError as e:
        log("insiders", f"API error: {e}", level="error")
        return ToolResult(
            data={"error": str(e)},
            confidence=0.0,
            status=ToolStatus.FAILED,
            tool_name="get_insider_transactions",
            ticker=ticker,
            error_message=f"API error: {e.response.status_code}",
        )
    except Exception as e:
        log("insiders", f"Error fetching insider transactions: {e}", level="error")
        return ToolResult(
            data={"error": str(e)},
            confidence=0.0,
            status=ToolStatus.FAILED,
            tool_name="get_insider_transactions",
            ticker=ticker,
            error_message=str(e),
        )


def _summarize_transactions(transactions: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize insider transactions."""
    if not transactions:
        return {"total": 0}

    # Count by type
    buys = []
    sells = []
    other = []

    total_buy_value = 0.0
    total_sell_value = 0.0

    for tx in transactions:
        tx_type = tx.get("transaction_type", "").lower()
        value = tx.get("value", 0) or 0

        if "purchase" in tx_type or "buy" in tx_type:
            buys.append(tx)
            total_buy_value += value
        elif "sale" in tx_type or "sell" in tx_type:
            sells.append(tx)
            total_sell_value += value
        else:
            other.append(tx)

    # Find notable insiders
    notable_buys = []
    notable_sells = []

    for tx in buys[:5]:  # Top 5 buys
        notable_buys.append(
            {
                "insider": tx.get("insider_name", "Unknown"),
                "title": tx.get("insider_title", ""),
                "shares": tx.get("shares", 0),
                "value": tx.get("value", 0),
                "date": tx.get("transaction_date", ""),
            }
        )

    for tx in sells[:5]:  # Top 5 sells
        notable_sells.append(
            {
                "insider": tx.get("insider_name", "Unknown"),
                "title": tx.get("insider_title", ""),
                "shares": tx.get("shares", 0),
                "value": tx.get("value", 0),
                "date": tx.get("transaction_date", ""),
            }
        )

    # Determine overall sentiment
    if total_buy_value > total_sell_value * 1.5:
        sentiment = "bullish"
        sentiment_note = "Insiders are net buyers"
    elif total_sell_value > total_buy_value * 1.5:
        sentiment = "bearish"
        sentiment_note = "Insiders are net sellers"
    else:
        sentiment = "neutral"
        sentiment_note = "Mixed insider activity"

    return {
        "total_transactions": len(transactions),
        "buys": {
            "count": len(buys),
            "total_value": total_buy_value,
            "notable": notable_buys,
        },
        "sells": {
            "count": len(sells),
            "total_value": total_sell_value,
            "notable": notable_sells,
        },
        "other_count": len(other),
        "sentiment": sentiment,
        "sentiment_note": sentiment_note,
        "net_value": total_buy_value - total_sell_value,
    }


# Tool definition for Claude
INSIDERS_TOOL_DEFINITION = {
    "name": "get_insider_transactions",
    "description": """Get recent insider buy/sell transactions for a company.

Shows recent transactions by executives, directors, and major shareholders.
Useful for understanding insider sentiment - are insiders buying or selling?

Returns:
- List of transactions with insider name, title, transaction type, shares, value
- Summary with buy/sell counts and totals
- Overall sentiment indicator (bullish/bearish/neutral)

Requires FINANCIAL_DATASETS_API_KEY to be configured.""",
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Stock ticker symbol (e.g., NVDA, AAPL)",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum transactions to return (default: 50)",
            },
        },
        "required": ["ticker"],
    },
}
