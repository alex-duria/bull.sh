"""Valuation framework for price target generation."""

from dataclasses import dataclass, field
from typing import Any

from bullsh.frameworks.base import Framework, FrameworkType, Criterion


@dataclass
class ValuationInputs:
    """Data required for valuation calculations."""
    # Current market data
    current_price: float | None = None
    shares_outstanding: float | None = None
    market_cap: float | None = None

    # Earnings-based
    eps_ttm: float | None = None
    eps_forward: float | None = None
    pe_ratio: float | None = None
    forward_pe: float | None = None

    # Revenue-based
    revenue_ttm: float | None = None
    price_to_sales: float | None = None

    # Enterprise value
    enterprise_value: float | None = None
    ebitda: float | None = None
    ev_to_ebitda: float | None = None

    # Growth
    revenue_growth: float | None = None
    earnings_growth: float | None = None

    # Sector comparables
    sector_pe: float | None = None
    sector_ps: float | None = None
    sector_ev_ebitda: float | None = None

    # Analyst targets
    target_low: float | None = None
    target_mean: float | None = None
    target_high: float | None = None


@dataclass
class PriceTarget:
    """A single price target from a valuation method."""
    method: str
    price: float
    upside_pct: float
    confidence: str  # "high", "medium", "low"
    rationale: str


@dataclass
class ValuationResult:
    """Complete valuation analysis result."""
    ticker: str
    current_price: float

    # Individual method results
    targets: list[PriceTarget] = field(default_factory=list)

    # Composite targets
    bear_case: float | None = None
    base_case: float | None = None
    bull_case: float | None = None

    # Summary
    average_target: float | None = None
    median_target: float | None = None
    average_upside: float | None = None

    def __post_init__(self):
        if self.targets:
            prices = [t.price for t in self.targets]
            self.average_target = sum(prices) / len(prices)
            self.median_target = sorted(prices)[len(prices) // 2]
            self.average_upside = ((self.average_target / self.current_price) - 1) * 100

            # Set bear/base/bull from range
            if len(prices) >= 3:
                sorted_prices = sorted(prices)
                self.bear_case = sorted_prices[0]
                self.bull_case = sorted_prices[-1]
                self.base_case = self.median_target

    def to_summary(self) -> str:
        """Generate summary text."""
        lines = [
            f"## Valuation Summary: {self.ticker}",
            f"**Current Price:** ${self.current_price:.2f}",
            "",
        ]

        if self.bear_case and self.base_case and self.bull_case:
            bear_upside = ((self.bear_case / self.current_price) - 1) * 100
            base_upside = ((self.base_case / self.current_price) - 1) * 100
            bull_upside = ((self.bull_case / self.current_price) - 1) * 100

            lines.extend([
                "### Price Target Range",
                f"| Case | Target | Upside |",
                f"|------|--------|--------|",
                f"| Bear | ${self.bear_case:.2f} | {bear_upside:+.1f}% |",
                f"| Base | ${self.base_case:.2f} | {base_upside:+.1f}% |",
                f"| Bull | ${self.bull_case:.2f} | {bull_upside:+.1f}% |",
                "",
            ])

        if self.targets:
            lines.extend([
                "### Valuation Methods",
                "| Method | Target | Upside | Confidence |",
                "|--------|--------|--------|------------|",
            ])
            for t in self.targets:
                lines.append(f"| {t.method} | ${t.price:.2f} | {t.upside_pct:+.1f}% | {t.confidence} |")
            lines.append("")

        if self.average_upside:
            if self.average_upside > 20:
                verdict = "**Potentially Undervalued** - Multiple methods suggest upside"
            elif self.average_upside < -20:
                verdict = "**Potentially Overvalued** - Multiple methods suggest downside"
            else:
                verdict = "**Fairly Valued** - Near consensus estimates"
            lines.append(f"### Verdict: {verdict}")

        return "\n".join(lines)

    def to_table_data(self) -> list[dict[str, Any]]:
        """Format for display as table."""
        return [
            {
                "method": t.method,
                "target": f"${t.price:.2f}",
                "upside": f"{t.upside_pct:+.1f}%",
                "confidence": t.confidence,
                "rationale": t.rationale,
            }
            for t in self.targets
        ]


def compute_valuation(inputs: ValuationInputs, ticker: str) -> ValuationResult:
    """
    Compute valuation using multiple methods.

    Methods used:
    1. P/E Multiple (sector comparison)
    2. Forward P/E Multiple
    3. Price/Sales Multiple
    4. EV/EBITDA Multiple
    5. Analyst Consensus
    6. Growth-Adjusted (PEG-based)
    """
    if not inputs.current_price:
        return ValuationResult(ticker=ticker, current_price=0)

    result = ValuationResult(ticker=ticker, current_price=inputs.current_price)

    # 1. P/E Multiple vs Sector
    if inputs.eps_ttm and inputs.sector_pe and inputs.eps_ttm > 0:
        implied_price = inputs.eps_ttm * inputs.sector_pe
        upside = ((implied_price / inputs.current_price) - 1) * 100

        # Confidence based on how reasonable the multiple is
        confidence = "medium"
        if 10 <= inputs.sector_pe <= 30:
            confidence = "high"
        elif inputs.sector_pe > 50:
            confidence = "low"

        result.targets.append(PriceTarget(
            method="P/E (Sector Avg)",
            price=implied_price,
            upside_pct=upside,
            confidence=confidence,
            rationale=f"EPS ${inputs.eps_ttm:.2f} × Sector P/E {inputs.sector_pe:.1f}x",
        ))

    # 2. Forward P/E
    if inputs.eps_forward and inputs.sector_pe and inputs.eps_forward > 0:
        # Apply slight discount to forward estimates
        implied_price = inputs.eps_forward * inputs.sector_pe * 0.95
        upside = ((implied_price / inputs.current_price) - 1) * 100

        result.targets.append(PriceTarget(
            method="Forward P/E",
            price=implied_price,
            upside_pct=upside,
            confidence="medium",
            rationale=f"Forward EPS ${inputs.eps_forward:.2f} × Sector P/E (5% haircut)",
        ))

    # 3. Price/Sales
    if inputs.revenue_ttm and inputs.shares_outstanding and inputs.sector_ps:
        revenue_per_share = inputs.revenue_ttm / inputs.shares_outstanding
        implied_price = revenue_per_share * inputs.sector_ps
        upside = ((implied_price / inputs.current_price) - 1) * 100

        result.targets.append(PriceTarget(
            method="Price/Sales",
            price=implied_price,
            upside_pct=upside,
            confidence="low",  # P/S less reliable
            rationale=f"Revenue/share ${revenue_per_share:.2f} × Sector P/S {inputs.sector_ps:.1f}x",
        ))

    # 4. EV/EBITDA
    if inputs.ebitda and inputs.sector_ev_ebitda and inputs.shares_outstanding:
        implied_ev = inputs.ebitda * inputs.sector_ev_ebitda
        # Approximate: EV = Market Cap (ignoring debt/cash for simplicity)
        implied_market_cap = implied_ev
        implied_price = implied_market_cap / inputs.shares_outstanding
        upside = ((implied_price / inputs.current_price) - 1) * 100

        result.targets.append(PriceTarget(
            method="EV/EBITDA",
            price=implied_price,
            upside_pct=upside,
            confidence="medium",
            rationale=f"EBITDA ${inputs.ebitda/1e9:.1f}B × Sector {inputs.sector_ev_ebitda:.1f}x",
        ))

    # 5. Analyst Consensus
    if inputs.target_mean:
        upside = ((inputs.target_mean / inputs.current_price) - 1) * 100
        result.targets.append(PriceTarget(
            method="Analyst Consensus",
            price=inputs.target_mean,
            upside_pct=upside,
            confidence="high",
            rationale=f"Mean of analyst targets (range ${inputs.target_low:.0f}-${inputs.target_high:.0f})",
        ))

    # 6. Growth-Adjusted (PEG-like)
    if inputs.eps_ttm and inputs.earnings_growth and inputs.earnings_growth > 0:
        # Fair P/E = Growth Rate (simplified PEG = 1)
        fair_pe = min(inputs.earnings_growth * 100, 40)  # Cap at 40x
        implied_price = inputs.eps_ttm * fair_pe
        upside = ((implied_price / inputs.current_price) - 1) * 100

        result.targets.append(PriceTarget(
            method="Growth-Adjusted",
            price=implied_price,
            upside_pct=upside,
            confidence="medium",
            rationale=f"PEG=1 implies P/E of {fair_pe:.0f}x for {inputs.earnings_growth*100:.0f}% growth",
        ))

    # Recalculate summary after adding targets
    result.__post_init__()

    return result


def extract_valuation_inputs_from_yahoo(yahoo_data: dict[str, Any]) -> ValuationInputs:
    """Extract valuation inputs from Yahoo Finance data."""
    return ValuationInputs(
        current_price=yahoo_data.get("price"),
        shares_outstanding=yahoo_data.get("shares_outstanding"),
        market_cap=yahoo_data.get("market_cap"),
        eps_ttm=yahoo_data.get("eps"),
        forward_pe=yahoo_data.get("forward_pe"),
        pe_ratio=yahoo_data.get("pe_ratio"),
        target_low=yahoo_data.get("target_low_price"),
        target_mean=yahoo_data.get("target_mean_price"),
        target_high=yahoo_data.get("target_high_price"),
        # These would need additional data sources
        sector_pe=yahoo_data.get("sector_pe", 20),  # Default fallback
        sector_ps=yahoo_data.get("sector_ps", 3),
        sector_ev_ebitda=yahoo_data.get("sector_ev_ebitda", 12),
    )


# Framework definition
VALUATION_FRAMEWORK = Framework(
    name="valuation",
    display_name="Valuation Analysis",
    description="Multi-method price target generation with bear/base/bull cases",
    framework_type=FrameworkType.QUANTITATIVE,
    scoring_enabled=False,
    author="builtin",
    criteria=[
        Criterion(
            id="pe_valuation",
            name="P/E Multiple Valuation",
            question="What is the implied price based on sector P/E multiple?",
            source="yahoo",
            scoring="price",
        ),
        Criterion(
            id="forward_pe",
            name="Forward P/E Valuation",
            question="What is the implied price based on forward earnings?",
            source="yahoo",
            scoring="price",
        ),
        Criterion(
            id="ev_ebitda",
            name="EV/EBITDA Valuation",
            question="What is the implied price based on enterprise value multiples?",
            source="yahoo",
            scoring="price",
        ),
        Criterion(
            id="analyst_targets",
            name="Analyst Price Targets",
            question="What do sell-side analysts expect?",
            source="yahoo",
            scoring="price",
        ),
        Criterion(
            id="growth_adjusted",
            name="Growth-Adjusted Valuation",
            question="What is fair value given the growth rate (PEG-based)?",
            source="yahoo",
            scoring="price",
        ),
        Criterion(
            id="price_target_range",
            name="Price Target Range",
            question="What are the bear, base, and bull case targets?",
            source="synthesis",
            scoring="range",
        ),
    ],
)
