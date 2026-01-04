"""Piotroski F-Score computation logic."""

from dataclasses import dataclass
from typing import Any

from bullsh.frameworks.base import Framework, load_framework


@dataclass
class FinancialData:
    """Financial data extracted from SEC filings for F-Score calculation."""
    # Current year
    net_income: float | None = None
    total_assets: float | None = None
    total_assets_prior: float | None = None
    operating_cash_flow: float | None = None
    long_term_debt: float | None = None
    long_term_debt_prior: float | None = None
    current_assets: float | None = None
    current_liabilities: float | None = None
    current_assets_prior: float | None = None
    current_liabilities_prior: float | None = None
    shares_outstanding: float | None = None
    shares_outstanding_prior: float | None = None
    gross_profit: float | None = None
    revenue: float | None = None
    gross_profit_prior: float | None = None
    revenue_prior: float | None = None

    # Computed values
    @property
    def roa(self) -> float | None:
        """Return on Assets = Net Income / Total Assets"""
        if self.net_income is not None and self.total_assets and self.total_assets > 0:
            return self.net_income / self.total_assets
        return None

    @property
    def roa_prior(self) -> float | None:
        """Prior year ROA (needs prior net income, which we approximate)"""
        # Note: Full implementation would need prior year net income
        return None

    @property
    def current_ratio(self) -> float | None:
        """Current Ratio = Current Assets / Current Liabilities"""
        if self.current_assets and self.current_liabilities and self.current_liabilities > 0:
            return self.current_assets / self.current_liabilities
        return None

    @property
    def current_ratio_prior(self) -> float | None:
        """Prior year current ratio"""
        if self.current_assets_prior and self.current_liabilities_prior and self.current_liabilities_prior > 0:
            return self.current_assets_prior / self.current_liabilities_prior
        return None

    @property
    def gross_margin(self) -> float | None:
        """Gross Margin = Gross Profit / Revenue"""
        if self.gross_profit is not None and self.revenue and self.revenue > 0:
            return self.gross_profit / self.revenue
        return None

    @property
    def gross_margin_prior(self) -> float | None:
        """Prior year gross margin"""
        if self.gross_profit_prior is not None and self.revenue_prior and self.revenue_prior > 0:
            return self.gross_profit_prior / self.revenue_prior
        return None

    @property
    def asset_turnover(self) -> float | None:
        """Asset Turnover = Revenue / Total Assets"""
        if self.revenue and self.total_assets and self.total_assets > 0:
            return self.revenue / self.total_assets
        return None

    @property
    def asset_turnover_prior(self) -> float | None:
        """Prior year asset turnover"""
        if self.revenue_prior and self.total_assets_prior and self.total_assets_prior > 0:
            return self.revenue_prior / self.total_assets_prior
        return None


@dataclass
class FScoreResult:
    """Result of F-Score calculation."""
    score: int
    max_score: int = 9
    signals: dict[str, bool | None] = None
    details: dict[str, str] = None

    def __post_init__(self):
        if self.signals is None:
            self.signals = {}
        if self.details is None:
            self.details = {}

    @property
    def rating(self) -> str:
        """Human-readable rating."""
        if self.score >= 7:
            return "Strong"
        elif self.score >= 4:
            return "Neutral"
        else:
            return "Weak"

    def to_table_data(self) -> list[dict[str, Any]]:
        """Format for display as table."""
        rows = []
        for signal_id, passed in self.signals.items():
            rows.append({
                "signal": signal_id.replace("_", " ").title(),
                "result": "✓ Pass" if passed else "✗ Fail" if passed is False else "? Unknown",
                "detail": self.details.get(signal_id, ""),
            })
        return rows


def compute_fscore(data: FinancialData) -> FScoreResult:
    """
    Compute Piotroski F-Score from financial data.

    The F-Score consists of 9 binary signals:

    PROFITABILITY (4 points):
    1. ROA > 0
    2. Operating Cash Flow > 0
    3. ROA increasing (vs prior year)
    4. Cash Flow > Net Income (accruals)

    LEVERAGE & LIQUIDITY (3 points):
    5. Long-term debt decreasing
    6. Current ratio increasing
    7. No new shares issued

    EFFICIENCY (2 points):
    8. Gross margin increasing
    9. Asset turnover increasing
    """
    signals: dict[str, bool | None] = {}
    details: dict[str, str] = {}
    score = 0

    # 1. ROA > 0
    roa = data.roa
    if roa is not None:
        signals["roa_positive"] = roa > 0
        details["roa_positive"] = f"ROA = {roa:.2%}"
        if roa > 0:
            score += 1
    else:
        signals["roa_positive"] = None
        details["roa_positive"] = "Could not calculate ROA"

    # 2. Operating Cash Flow > 0
    if data.operating_cash_flow is not None:
        signals["cfo_positive"] = data.operating_cash_flow > 0
        details["cfo_positive"] = f"Operating CF = ${data.operating_cash_flow:,.0f}"
        if data.operating_cash_flow > 0:
            score += 1
    else:
        signals["cfo_positive"] = None
        details["cfo_positive"] = "Operating cash flow not available"

    # 3. ROA increasing
    # Note: Would need prior year data for full calculation
    signals["roa_increasing"] = None
    details["roa_increasing"] = "Prior year comparison not available"

    # 4. Cash Flow > Net Income (quality of earnings)
    if data.operating_cash_flow is not None and data.net_income is not None:
        signals["cfo_gt_ni"] = data.operating_cash_flow > data.net_income
        details["cfo_gt_ni"] = f"CF ${data.operating_cash_flow:,.0f} vs NI ${data.net_income:,.0f}"
        if data.operating_cash_flow > data.net_income:
            score += 1
    else:
        signals["cfo_gt_ni"] = None
        details["cfo_gt_ni"] = "Cannot compare CF to Net Income"

    # 5. Long-term debt decreasing
    if data.long_term_debt is not None and data.long_term_debt_prior is not None:
        signals["debt_decreasing"] = data.long_term_debt < data.long_term_debt_prior
        change = data.long_term_debt - data.long_term_debt_prior
        details["debt_decreasing"] = f"LT Debt changed by ${change:,.0f}"
        if data.long_term_debt < data.long_term_debt_prior:
            score += 1
    else:
        signals["debt_decreasing"] = None
        details["debt_decreasing"] = "Prior year debt not available"

    # 6. Current ratio increasing
    cr = data.current_ratio
    cr_prior = data.current_ratio_prior
    if cr is not None and cr_prior is not None:
        signals["current_ratio_increasing"] = cr > cr_prior
        details["current_ratio_increasing"] = f"Current ratio: {cr:.2f} vs prior {cr_prior:.2f}"
        if cr > cr_prior:
            score += 1
    else:
        signals["current_ratio_increasing"] = None
        details["current_ratio_increasing"] = "Cannot compare current ratios"

    # 7. No share dilution
    if data.shares_outstanding is not None and data.shares_outstanding_prior is not None:
        signals["no_dilution"] = data.shares_outstanding <= data.shares_outstanding_prior
        change_pct = ((data.shares_outstanding / data.shares_outstanding_prior) - 1) * 100
        details["no_dilution"] = f"Shares changed {change_pct:+.1f}%"
        if data.shares_outstanding <= data.shares_outstanding_prior:
            score += 1
    else:
        signals["no_dilution"] = None
        details["no_dilution"] = "Prior year shares not available"

    # 8. Gross margin increasing
    gm = data.gross_margin
    gm_prior = data.gross_margin_prior
    if gm is not None and gm_prior is not None:
        signals["gross_margin_increasing"] = gm > gm_prior
        details["gross_margin_increasing"] = f"Gross margin: {gm:.1%} vs prior {gm_prior:.1%}"
        if gm > gm_prior:
            score += 1
    else:
        signals["gross_margin_increasing"] = None
        details["gross_margin_increasing"] = "Cannot compare gross margins"

    # 9. Asset turnover increasing
    at = data.asset_turnover
    at_prior = data.asset_turnover_prior
    if at is not None and at_prior is not None:
        signals["asset_turnover_increasing"] = at > at_prior
        details["asset_turnover_increasing"] = f"Asset turnover: {at:.2f}x vs prior {at_prior:.2f}x"
        if at > at_prior:
            score += 1
    else:
        signals["asset_turnover_increasing"] = None
        details["asset_turnover_increasing"] = "Cannot compare asset turnover"

    return FScoreResult(score=score, signals=signals, details=details)


def extract_financial_data_from_filing(filing_text: str) -> FinancialData:
    """
    Extract financial data from SEC filing text.

    This is a simplified extraction - production would use XBRL or
    more sophisticated parsing.
    """
    import re

    data = FinancialData()

    # Helper to find numbers after keywords
    def find_number(text: str, keywords: list[str], scale: float = 1.0) -> float | None:
        for keyword in keywords:
            pattern = rf"{keyword}[^\d]*\$?\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion|M|B)?"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    num_str = match.group(1).replace(",", "")
                    num = float(num_str)

                    # Check for scale indicators
                    context = text[match.start():match.end() + 20].lower()
                    if "billion" in context or "b" in context.split()[-1:]:
                        num *= 1_000_000_000
                    elif "million" in context or "m" in context.split()[-1:]:
                        num *= 1_000_000

                    return num * scale
                except ValueError:
                    continue
        return None

    # Try to extract key figures
    # Note: This is very basic - real implementation would be more robust
    data.net_income = find_number(filing_text, ["net income", "net earnings"])
    data.total_assets = find_number(filing_text, ["total assets"])
    data.operating_cash_flow = find_number(
        filing_text,
        ["operating activities", "cash from operations", "operating cash flow"]
    )
    data.revenue = find_number(filing_text, ["total revenue", "net revenue", "total net revenue"])
    data.gross_profit = find_number(filing_text, ["gross profit", "gross margin"])

    return data


def apply_fscore_to_framework(framework: Framework, fscore: FScoreResult) -> Framework:
    """Apply F-Score results to framework criteria."""
    for criterion in framework.criteria:
        if criterion.id in fscore.signals:
            result = fscore.signals[criterion.id]
            criterion.checked = result is not None
            criterion.result = result

    return framework
