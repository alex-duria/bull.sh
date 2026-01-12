"""Base classes for tools."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ToolStatus(Enum):
    """Status of a tool execution."""

    SUCCESS = "success"
    PARTIAL = "partial"  # Some data retrieved, but incomplete
    FAILED = "failed"
    RATE_LIMITED = "rate_limited"
    CACHED = "cached"


@dataclass
class ToolResult:
    """
    Result from a tool execution.

    All tools return this structure so the agent can assess data quality.
    """

    data: dict[str, Any]
    confidence: float  # 0.0 to 1.0
    status: ToolStatus
    source_url: str | None = None
    cached: bool = False
    fetched_at: datetime = field(default_factory=datetime.now)
    error_message: str | None = None

    # Metadata for provenance tracking
    tool_name: str = ""
    ticker: str = ""

    def to_prompt_text(self) -> str:
        """Format the result for inclusion in a prompt."""
        if self.status == ToolStatus.FAILED:
            return f"[{self.tool_name}] Failed: {self.error_message or 'Unknown error'}"

        if self.status == ToolStatus.RATE_LIMITED:
            return f"[{self.tool_name}] Rate limited, please try again later"

        confidence_str = f"(confidence: {self.confidence:.0%})"
        cached_str = " [cached]" if self.cached else ""

        lines = [f"[{self.tool_name}] {confidence_str}{cached_str}"]

        for key, value in self.data.items():
            if isinstance(value, str) and len(value) > 500:
                # Truncate long values for prompt
                lines.append(f"  {key}: {value[:500]}...")
            else:
                lines.append(f"  {key}: {value}")

        return "\n".join(lines)

    def to_provenance_dict(self) -> dict[str, Any]:
        """Format for thesis YAML frontmatter."""
        return {
            "type": self.tool_name,
            "ticker": self.ticker,
            "fetched": self.fetched_at.isoformat(),
            "cached": self.cached,
            "confidence": self.confidence,
            "source_url": self.source_url,
            "status": self.status.value,
        }


@dataclass
class ToolDefinition:
    """
    Definition of a tool for Claude's tool_use.

    This maps to the JSON schema format Claude expects.
    """

    name: str
    description: str
    parameters: dict[str, Any]

    def to_claude_schema(self) -> dict[str, Any]:
        """Convert to Claude API tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.parameters.get("properties", {}),
                "required": self.parameters.get("required", []),
            },
        }


# Tool definitions for Claude
SEC_SEARCH_TOOL = ToolDefinition(
    name="sec_search",
    description="Search SEC EDGAR for available filings. Returns list of recent 10-K, 10-Q, 8-K filings with dates. Use this first to verify company has required filings.",
    parameters={
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Stock ticker symbol (e.g., NVDA, AAPL)",
            },
            "fuzzy": {
                "type": "boolean",
                "description": "If true, search by company name if ticker not found",
                "default": True,
            },
        },
        "required": ["ticker"],
    },
)

SEC_FETCH_TOOL = ToolDefinition(
    name="sec_fetch",
    description="Fetch and parse a SEC filing. Returns the full text organized by section (Business, Risk Factors, MD&A, Financials). Respects user's verbosity setting.",
    parameters={
        "properties": {
            "ticker": {"type": "string", "description": "Stock ticker symbol"},
            "filing_type": {
                "type": "string",
                "enum": ["10-K", "10-Q"],
                "description": "Type of filing to fetch",
            },
            "year": {
                "type": "integer",
                "description": "Optional: specific year, defaults to latest",
            },
            "section": {
                "type": "string",
                "description": "Optional: specific section (business, risk_factors, mda, financials)",
            },
        },
        "required": ["ticker", "filing_type"],
    },
)

SEARCH_STOCKTWITS_TOOL = ToolDefinition(
    name="search_stocktwits",
    description="Search StockTwits for stock discussions and sentiment. Primary social sentiment source.",
    parameters={
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Stock ticker symbol",
            },
        },
        "required": ["symbol"],
    },
)

SEARCH_REDDIT_TOOL = ToolDefinition(
    name="search_reddit",
    description="Search Reddit for discussions about a stock. Fallback if StockTwits unavailable.",
    parameters={
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query (usually ticker or company name)",
            },
            "subreddits": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional: specific subreddits to search",
            },
        },
        "required": ["query"],
    },
)

SCRAPE_YAHOO_TOOL = ToolDefinition(
    name="scrape_yahoo",
    description="Scrape Yahoo Finance for analyst ratings, price targets, and key statistics. Returns data with confidence score.",
    parameters={
        "properties": {
            "ticker": {"type": "string", "description": "Stock ticker symbol"},
        },
        "required": ["ticker"],
    },
)

SEARCH_NEWS_TOOL = ToolDefinition(
    name="search_news",
    description="Search recent financial news articles about a company via DuckDuckGo.",
    parameters={
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "days_back": {
                "type": "integer",
                "description": "How many days back to search",
                "default": 30,
            },
        },
        "required": ["query"],
    },
)

COMPUTE_RATIOS_TOOL = ToolDefinition(
    name="compute_ratios",
    description="Calculate key financial ratios from available data. Returns P/E, EV/EBITDA, revenue growth, margins with confidence scores.",
    parameters={
        "properties": {
            "ticker": {"type": "string", "description": "Stock ticker symbol"},
        },
        "required": ["ticker"],
    },
)

SAVE_THESIS_TOOL = ToolDefinition(
    name="save_thesis",
    description="Save current research as a thesis document with full provenance metadata.",
    parameters={
        "properties": {
            "ticker": {"type": "string", "description": "Stock ticker symbol"},
            "content": {
                "type": "string",
                "description": "Markdown content of the thesis",
            },
            "filename": {
                "type": "string",
                "description": "Optional custom filename",
            },
        },
        "required": ["ticker", "content"],
    },
)

RAG_SEARCH_TOOL = ToolDefinition(
    name="rag_search",
    description="""PRIMARY tool for questions about SEC filing content. Searches indexed 10-K and 10-Q filings.

ALWAYS use this FIRST when users ask about: risks, revenue, strategy, competition, management, segments, guidance, or any filing content.

Query tips for better results:
- Use SEC section names: "Item 1A Risk Factors", "Item 7 MD&A", "Item 1 Business"
- If results aren't relevant, VARY YOUR QUERY - try synonyms and alternative phrases
- Examples: "risk factors" → "principal risks", "revenue" → "net sales", "competition" → "competitive landscape"

Do NOT use web_search for information in filings - it's already indexed here!""",
    parameters={
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query - use SEC section names like 'Item 1A Risk Factors' or vary terms for better results",
            },
            "ticker": {
                "type": "string",
                "description": "Optional: filter to specific company",
            },
            "form": {
                "type": "string",
                "enum": ["10-K", "10-Q"],
                "description": "Optional: filter to specific form type",
            },
            "year": {
                "type": "integer",
                "description": "Optional: filter to specific year",
            },
            "k": {
                "type": "integer",
                "description": "Number of results to return (default: 5)",
                "default": 5,
            },
        },
        "required": ["query"],
    },
)

RAG_LIST_TOOL = ToolDefinition(
    name="rag_list",
    description="List all SEC filings indexed in the vector database. Shows which filings are available for semantic search.",
    parameters={
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Optional: filter to specific company",
            },
        },
        "required": [],
    },
)

WEB_SEARCH_TOOL = ToolDefinition(
    name="web_search",
    description="""General web search for current information not found in SEC filings.

Use this to:
- Get current stock prices, market cap, and real-time data when Yahoo Finance fails
- Find recent company announcements and press releases
- Research competitors, market trends, and industry analysis
- Get current analyst ratings and price targets
- Fill gaps when filing data is outdated or a tool returns no data
- Verify or supplement information from other sources

Always use web search when other tools fail or return incomplete data.""",
    parameters={
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query - be specific (e.g., 'NVDA stock price January 2026', 'Tesla revenue Q3 2025')",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum results to return (default: 10)",
                "default": 10,
            },
        },
        "required": ["query"],
    },
)

GENERATE_EXCEL_TOOL = ToolDefinition(
    name="generate_excel",
    description="""Generate Excel spreadsheet with financial metrics, ratios, and comparison tables.

Use this after gathering market data to create an exportable financial model. The spreadsheet includes:
- Key Metrics sheet: Price, P/E, market cap, 52-week range, volume, sector
- Financial Ratios sheet: Valuation ratios, analyst price targets
- Comparison sheet: Side-by-side comparison (when multiple tickers)
- Valuation Analysis: Price vs targets, 52-week range position

Best used after scrape_yahoo has been called for the ticker(s).""",
    parameters={
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Primary stock ticker symbol",
            },
            "include_ratios": {
                "type": "boolean",
                "description": "Include financial ratios sheet (default: true)",
                "default": True,
            },
            "compare_tickers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Additional tickers to include for comparison",
            },
        },
        "required": ["ticker"],
    },
)

CALCULATE_FACTORS_TOOL = ToolDefinition(
    name="calculate_factors",
    description="""Calculate multi-factor exposures for a stock using cross-sectional z-scores.

IMPORTANT: You MUST call this tool when users ask about factor exposures, factor analysis, or factor tilts.
Do NOT theorize about factors - always compute real z-scores using this tool.

Computes z-scores for each factor by comparing the primary ticker against its peer group:
- Value: P/E ratio, P/B ratio, EV/EBITDA (lower is better)
- Momentum: 12-month return, 52-week high proximity (higher is better)
- Quality: ROE, debt/equity inverse, earnings stability (higher is better)
- Growth: Revenue growth, earnings growth (higher is better)
- Size: Market cap (lower z-score = smaller cap)
- Volatility: Realized vol, beta (lower is better)

Returns factor z-scores, composite score, and peer comparison. Use this for any factor-based analysis.""",
    parameters={
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Primary stock ticker to analyze",
            },
            "peers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "2-6 peer tickers for cross-sectional comparison",
            },
            "factors": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["value", "momentum", "quality", "growth", "size", "volatility"],
                },
                "description": "Factors to calculate (default: all)",
            },
        },
        "required": ["ticker", "peers"],
    },
)

RUN_FACTOR_REGRESSION_TOOL = ToolDefinition(
    name="run_factor_regression",
    description="""Run Fama-French factor regression to decompose stock returns.

Performs OLS regression of stock excess returns against Fama-French factors:
- Mkt-RF: Market risk premium
- SMB: Small minus big (size factor)
- HML: High minus low (value factor)
- RF: Risk-free rate

Returns factor betas, R-squared, alpha, and variance decomposition showing what percentage
of the stock's risk comes from each factor. Requires 3+ years of price history.

Use this for understanding systematic risk exposure and factor attribution.""",
    parameters={
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Stock ticker to analyze",
            },
            "window_months": {
                "type": "integer",
                "description": "Rolling window in months (default: 36)",
                "default": 36,
            },
        },
        "required": ["ticker"],
    },
)

GET_FINANCIALS_TOOL = ToolDefinition(
    name="get_financials",
    description="""Get structured financial statements (income, balance sheet, cash flow) from best available source.

Uses Financial Datasets API if configured, falls back to Yahoo Finance.

Returns:
- Income statement: revenue, gross_profit, operating_income, net_income
- Balance sheet: total_assets, total_liabilities, shareholders_equity, cash
- Cash flow: operating_cash_flow, investing_cash_flow, financing_cash_flow, free_cash_flow
- Computed ratios: ROA, ROE (when data available)""",
    parameters={
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
)

GET_INSIDER_TRANSACTIONS_TOOL = ToolDefinition(
    name="get_insider_transactions",
    description="""Get recent insider buy/sell transactions for a company.

Shows recent transactions by executives, directors, and major shareholders.
Useful for understanding insider sentiment - are insiders buying or selling?

Returns:
- List of transactions with insider name, title, transaction type, shares, value
- Summary with buy/sell counts and totals
- Overall sentiment indicator (bullish/bearish/neutral)

Requires FINANCIAL_DATASETS_API_KEY to be configured.""",
    parameters={
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
)

# All available tools
ALL_TOOLS = [
    SEC_SEARCH_TOOL,
    SEC_FETCH_TOOL,
    RAG_SEARCH_TOOL,
    RAG_LIST_TOOL,
    SEARCH_STOCKTWITS_TOOL,
    SEARCH_REDDIT_TOOL,
    SCRAPE_YAHOO_TOOL,
    SEARCH_NEWS_TOOL,
    WEB_SEARCH_TOOL,
    COMPUTE_RATIOS_TOOL,
    GENERATE_EXCEL_TOOL,
    CALCULATE_FACTORS_TOOL,
    RUN_FACTOR_REGRESSION_TOOL,
    SAVE_THESIS_TOOL,
    GET_FINANCIALS_TOOL,
    GET_INSIDER_TRANSACTIONS_TOOL,
]


def get_tools_for_claude() -> list[dict[str, Any]]:
    """Get all tool definitions in Claude API format."""
    return [tool.to_claude_schema() for tool in ALL_TOOLS]
