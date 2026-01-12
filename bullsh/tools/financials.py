"""
Unified Financials Tool - fetches financial statements from best available source.

Priority:
1. Financial Datasets API (if key configured)
2. Compute from SEC filings
3. Yahoo Finance fallback

Financial Datasets API: https://financialdatasets.ai/
"""

import os
from typing import Any, Literal

import httpx

from bullsh.logging import log
from bullsh.tools.base import ToolResult, ToolStatus

# Financial Datasets API base URL
FD_API_BASE = "https://api.financialdatasets.ai"


class FinancialDatasetsClient:
    """Client for Financial Datasets API."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("FINANCIAL_DATASETS_API_KEY")
        self.client = httpx.AsyncClient(timeout=30.0)

    @property
    def is_configured(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key)

    async def get_income_statement(
        self,
        ticker: str,
        period: Literal["annual", "quarterly"] = "annual",
        limit: int = 10,
    ) -> dict[str, Any] | None:
        """Fetch income statement data."""
        if not self.is_configured:
            return None

        url = f"{FD_API_BASE}/financials/income-statements"
        params = {
            "ticker": ticker.upper(),
            "period": period,
            "limit": limit,
        }

        try:
            response = await self.client.get(
                url,
                params=params,
                headers={"X-API-KEY": self.api_key},
            )
            response.raise_for_status()
            data = response.json()
            return data.get("income_statements", [])
        except Exception as e:
            log("financials", f"FD API income statement error: {e}", level="warning")
            return None

    async def get_balance_sheet(
        self,
        ticker: str,
        period: Literal["annual", "quarterly"] = "annual",
        limit: int = 10,
    ) -> dict[str, Any] | None:
        """Fetch balance sheet data."""
        if not self.is_configured:
            return None

        url = f"{FD_API_BASE}/financials/balance-sheets"
        params = {
            "ticker": ticker.upper(),
            "period": period,
            "limit": limit,
        }

        try:
            response = await self.client.get(
                url,
                params=params,
                headers={"X-API-KEY": self.api_key},
            )
            response.raise_for_status()
            data = response.json()
            return data.get("balance_sheets", [])
        except Exception as e:
            log("financials", f"FD API balance sheet error: {e}", level="warning")
            return None

    async def get_cash_flow_statement(
        self,
        ticker: str,
        period: Literal["annual", "quarterly"] = "annual",
        limit: int = 10,
    ) -> dict[str, Any] | None:
        """Fetch cash flow statement data."""
        if not self.is_configured:
            return None

        url = f"{FD_API_BASE}/financials/cash-flow-statements"
        params = {
            "ticker": ticker.upper(),
            "period": period,
            "limit": limit,
        }

        try:
            response = await self.client.get(
                url,
                params=params,
                headers={"X-API-KEY": self.api_key},
            )
            response.raise_for_status()
            data = response.json()
            return data.get("cash_flow_statements", [])
        except Exception as e:
            log("financials", f"FD API cash flow error: {e}", level="warning")
            return None

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


async def get_financials(
    ticker: str,
    statement_type: Literal["income", "balance", "cashflow", "all"] = "all",
    period: Literal["annual", "quarterly"] = "annual",
    years: int = 3,
) -> ToolResult:
    """
    Get financial statements from best available source.

    Priority:
    1. Financial Datasets API (if configured)
    2. SEC EDGAR (compute from filings)
    3. Yahoo Finance (fallback)

    Args:
        ticker: Stock ticker symbol
        statement_type: Type of statement (income, balance, cashflow, or all)
        period: Annual or quarterly data
        years: Number of years of data

    Returns:
        ToolResult with financial data
    """
    log("financials", f"Fetching financials for {ticker}, type={statement_type}, period={period}")

    ticker = ticker.upper()
    limit = years * (4 if period == "quarterly" else 1)

    # Try Financial Datasets API first
    fd_client = FinancialDatasetsClient()

    if fd_client.is_configured:
        log("financials", "Using Financial Datasets API")
        result = await _fetch_from_fd_api(fd_client, ticker, statement_type, period, limit)
        await fd_client.close()

        if result:
            return ToolResult(
                data=result,
                confidence=0.95,
                status=ToolStatus.SUCCESS,
                tool_name="get_financials",
                ticker=ticker,
                source_url=f"https://financialdatasets.ai/stocks/{ticker}",
            )

    # Try Yahoo Finance as fallback
    log("financials", "Falling back to Yahoo Finance")
    try:
        result = await _fetch_from_yahoo(ticker, statement_type)
        if result:
            return ToolResult(
                data=result,
                confidence=0.75,  # Lower confidence for Yahoo
                status=ToolStatus.SUCCESS,
                tool_name="get_financials",
                ticker=ticker,
                source_url=f"https://finance.yahoo.com/quote/{ticker}",
            )
    except Exception as e:
        log("financials", f"Yahoo fallback error: {e}", level="warning")

    # All sources failed
    return ToolResult(
        data={"error": "Could not fetch financial data from any source"},
        confidence=0.0,
        status=ToolStatus.FAILED,
        tool_name="get_financials",
        ticker=ticker,
        error_message="Financial data unavailable from all sources",
    )


async def _fetch_from_fd_api(
    client: FinancialDatasetsClient,
    ticker: str,
    statement_type: str,
    period: str,
    limit: int,
) -> dict[str, Any] | None:
    """Fetch data from Financial Datasets API."""
    result: dict[str, Any] = {
        "ticker": ticker,
        "period": period,
        "source": "financial_datasets",
    }

    if statement_type in ("income", "all"):
        income = await client.get_income_statement(ticker, period, limit)
        if income:
            result["income_statements"] = income
            # Extract key metrics from latest
            if income:
                latest = income[0]
                result["revenue"] = latest.get("revenue")
                result["gross_profit"] = latest.get("gross_profit")
                result["operating_income"] = latest.get("operating_income")
                result["net_income"] = latest.get("net_income")

    if statement_type in ("balance", "all"):
        balance = await client.get_balance_sheet(ticker, period, limit)
        if balance:
            result["balance_sheets"] = balance
            if balance:
                latest = balance[0]
                result["total_assets"] = latest.get("total_assets")
                result["total_liabilities"] = latest.get("total_liabilities")
                result["shareholders_equity"] = latest.get("shareholders_equity")
                result["cash_and_equivalents"] = latest.get("cash_and_cash_equivalents")

    if statement_type in ("cashflow", "all"):
        cashflow = await client.get_cash_flow_statement(ticker, period, limit)
        if cashflow:
            result["cash_flow_statements"] = cashflow
            if cashflow:
                latest = cashflow[0]
                result["operating_cash_flow"] = latest.get("net_cash_from_operating_activities")
                result["investing_cash_flow"] = latest.get("net_cash_from_investing_activities")
                result["financing_cash_flow"] = latest.get("net_cash_from_financing_activities")
                result["free_cash_flow"] = latest.get("free_cash_flow")

    # Calculate key ratios if we have the data
    if result.get("net_income") and result.get("total_assets"):
        try:
            result["roa"] = result["net_income"] / result["total_assets"]
        except (TypeError, ZeroDivisionError):
            pass

    if result.get("net_income") and result.get("shareholders_equity"):
        try:
            result["roe"] = result["net_income"] / result["shareholders_equity"]
        except (TypeError, ZeroDivisionError):
            pass

    # Check if we got any data
    has_data = any(
        k in result for k in ["income_statements", "balance_sheets", "cash_flow_statements"]
    )

    return result if has_data else None


async def _fetch_from_yahoo(
    ticker: str,
    statement_type: str,
) -> dict[str, Any] | None:
    """Fetch data from Yahoo Finance as fallback."""
    try:
        import yfinance as yf

        stock = yf.Ticker(ticker)

        result: dict[str, Any] = {
            "ticker": ticker,
            "source": "yahoo_finance",
        }

        if statement_type in ("income", "all"):
            try:
                income = stock.income_stmt
                if income is not None and not income.empty:
                    # Convert to dict format
                    result["income_statement_yahoo"] = income.to_dict()
                    # Get latest values
                    if "Total Revenue" in income.index:
                        result["revenue"] = float(income.loc["Total Revenue"].iloc[0])
                    if "Net Income" in income.index:
                        result["net_income"] = float(income.loc["Net Income"].iloc[0])
            except Exception as e:
                log("financials", f"Yahoo income statement error: {e}", level="warning")

        if statement_type in ("balance", "all"):
            try:
                balance = stock.balance_sheet
                if balance is not None and not balance.empty:
                    result["balance_sheet_yahoo"] = balance.to_dict()
                    if "Total Assets" in balance.index:
                        result["total_assets"] = float(balance.loc["Total Assets"].iloc[0])
                    if "Total Liabilities Net Minority Interest" in balance.index:
                        result["total_liabilities"] = float(
                            balance.loc["Total Liabilities Net Minority Interest"].iloc[0]
                        )
            except Exception as e:
                log("financials", f"Yahoo balance sheet error: {e}", level="warning")

        if statement_type in ("cashflow", "all"):
            try:
                cashflow = stock.cashflow
                if cashflow is not None and not cashflow.empty:
                    result["cash_flow_yahoo"] = cashflow.to_dict()
                    if "Operating Cash Flow" in cashflow.index:
                        result["operating_cash_flow"] = float(
                            cashflow.loc["Operating Cash Flow"].iloc[0]
                        )
                    if "Free Cash Flow" in cashflow.index:
                        result["free_cash_flow"] = float(cashflow.loc["Free Cash Flow"].iloc[0])
            except Exception as e:
                log("financials", f"Yahoo cash flow error: {e}", level="warning")

        # Check if we got any data
        has_data = any(
            k in result
            for k in [
                "income_statement_yahoo",
                "balance_sheet_yahoo",
                "cash_flow_yahoo",
                "revenue",
                "net_income",
            ]
        )

        return result if has_data else None

    except ImportError:
        log("financials", "yfinance not installed", level="warning")
        return None
    except Exception as e:
        log("financials", f"Yahoo Finance error: {e}", level="warning")
        return None


# Tool definition for Claude
FINANCIALS_TOOL_DEFINITION = {
    "name": "get_financials",
    "description": """Get structured financial statements (income, balance sheet, cash flow) from best available source.

Uses Financial Datasets API if configured, falls back to Yahoo Finance.

Returns:
- Income statement: revenue, gross_profit, operating_income, net_income
- Balance sheet: total_assets, total_liabilities, shareholders_equity, cash
- Cash flow: operating_cash_flow, investing_cash_flow, financing_cash_flow, free_cash_flow
- Computed ratios: ROA, ROE (when data available)""",
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Stock ticker symbol (e.g., NVDA, AAPL)",
            },
            "statement_type": {
                "type": "string",
                "enum": ["income", "balance", "cashflow", "all"],
                "description": "Type of statement to fetch (default: all)",
            },
            "period": {
                "type": "string",
                "enum": ["annual", "quarterly"],
                "description": "Data period (default: annual)",
            },
            "years": {
                "type": "integer",
                "description": "Number of years of data (default: 3)",
            },
        },
        "required": ["ticker"],
    },
}
