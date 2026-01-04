# Multi-Factor Stock Analysis — Technical Specification

## Overview

An educational, agentic CLI experience within bullsh that guides users through building multi-factor stock analysis models. The professor-led interactive session teaches factor investing concepts while producing a professional Excel workbook suitable for equity research presentations and portfolio analysis.

**Philosophy**: The journey is as important as the destination. Users learn by doing — making real choices about factors, weights, and peers while the agent explains the "why" behind each step.

---

## Part 1: User Experience Flow

### Entry Point

Accessed via the existing framework system:
```
/framework factors
```

This enters an interactive factor analysis session within the REPL. The agent (professor persona) guides the user through a structured learning experience.

### Session Stages

```
Stage 1: Ticker Selection & Validation
    └── User provides primary ticker
    └── Agent validates (US-only for v1, sufficient data history)
    └── Agent explains what data we'll need and why

Stage 2: Peer Selection
    └── User provides peer tickers OR requests suggestions
    └── Agent validates each peer
    └── Agent explains importance of peer selection for cross-sectional analysis

Stage 3: Factor Selection & Weighting
    └── Agent presents the 6 core factors with explanations
    └── User selects which factors to include
    └── User sets weights (or accepts equal-weight default)
    └── Agent explains tradeoffs of each choice

Stage 4: Data Fetching
    └── Subagents fetch data in parallel (price, fundamentals, benchmark)
    └── Agent narrates what's being retrieved and why each piece matters
    └── ~30-60 second wait with educational content

Stage 5: Factor Calculation Walkthrough
    └── Agent calculates each selected factor
    └── Shows the actual math: "Your value z-score is -0.8, calculated as (15.2 - 22.1) / 8.6"
    └── Explains what each score means in context

Stage 6: Risk Decomposition
    └── Agent decomposes variance by factor
    └── Explains idiosyncratic vs. systematic risk
    └── Shows correlation matrix with interpretation

Stage 7: Scenario Analysis
    └── Agent walks through pre-built scenarios
    └── Guides user through building a custom scenario
    └── Shows sensitivity of their stock to each scenario

Stage 8: Excel Generation & Summary
    └── Agent generates the workbook
    └── Provides verbal summary of key findings
    └── Explains how to use the interactive Excel features
```

### Interaction Model

**Choice Presentation**: Numbered menus
```
Which factors would you like to include?

[1] Value      - Price multiples vs. peers (P/E, P/B, EV/EBITDA)
[2] Momentum   - Recent price performance (12-1 month return)
[3] Quality    - Profitability and stability (ROE, leverage, earnings consistency)
[4] Growth     - Revenue and earnings growth rates
[5] Size       - Market capitalization (log-scaled)
[6] Volatility - Price volatility and market beta

[A] All factors (recommended for learning)

Enter your choices (e.g., "1,2,3" or "A"):
```

**Pacing**: Professor pauses at key learning moments but doesn't over-explain mechanical steps. Mix of thorough concept explanations and concise execution updates.

**Backtracking**: Limited — user can go back one stage (e.g., from Stage 5 back to Stage 4 to add a peer), but cannot restart from the beginning without explicit `/factors reset`.

**Failure Handling**: When data is missing or calculations fail, agent asks user how to proceed (pedagogical decision-making).

---

## Part 2: Agent Persona & Pedagogy

### Professor Persona

- **Tone**: Formal but approachable, uses proper financial terminology
- **Teaching style**: Explains concepts before executing, connects theory to practice
- **Error handling**: Frames failures as learning opportunities ("This is common with young companies — the quality factor requires 5 years of data to measure earnings stability.")

### Transparency Principles

1. **Show the math**: Display actual calculations, not just results
   ```
   Value z-score: -0.8
   Calculated as: (15.2 - 22.1) / 8.6
   Where: 15.2 = NVDA's P/E, 22.1 = peer median, 8.6 = peer std dev
   ```

2. **Explain implications**: What does a -0.8 z-score mean for investment decisions?

3. **Connect to practice**: "On a trading desk, this would flag NVDA as expensive relative to semis..."

---

## Part 3: Data Architecture

### Data Sources (Priority Order)

1. **Yahoo Finance** (primary) — already integrated in bullsh
   - Price history (daily adjusted close, volume)
   - Fundamental data (market cap, P/E, revenue, etc.)
   - No rate limits, scraping-based

2. **AlphaVantage** (optional) — user provides API key
   - Cleaner fundamental data
   - Free tier: 5 calls/min, 500/day
   - Used only if key configured and user requests

3. **Ken French Data Library** (factor returns)
   - Fetch fresh from web: `mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html`
   - User manages local caching
   - CSV format, straightforward parsing

### Data Fetching Strategy

**Parallel Subagent Fetching**: At Stage 4, spawn subagents to fetch data concurrently:
- Subagent 1: Primary ticker price + fundamentals
- Subagent 2-N: Peer ticker price + fundamentals (one per peer)
- Subagent N+1: Benchmark (S&P 500) price history
- Subagent N+2: Fama-French factor returns

Agent provides educational narration during the ~30-60 second fetch window.

### Caching

Use existing `storage/cache.py` infrastructure:
- Price data: TTL = 1 day (EOD refresh)
- Fundamental data: TTL = 1 day
- Fama-French returns: TTL = 7 days

---

## Part 4: Factor Methodology

### The Six Factors

| Factor | Components | Calculation | Sign Convention |
|--------|------------|-------------|-----------------|
| **Value** | P/E, P/B, EV/EBITDA | Average z-score of inverted ratios | Higher = cheaper = better |
| **Momentum** | 12-1 month return, 52W high proximity | Average z-score | Higher = stronger momentum |
| **Quality** | ROE, Debt/Equity (inv), Earnings stability | Average z-score | Higher = higher quality |
| **Growth** | YoY revenue growth, YoY earnings growth | Average z-score | Higher = faster growth |
| **Size** | ln(Market Cap) | Z-score | Higher = larger cap |
| **Volatility** | 60-day realized vol, Beta | Average z-score | Higher = MORE volatile (negative exposure = low-vol) |

### Cross-Sectional Normalization

For each factor:
1. Calculate raw value for primary ticker and all peers
2. Compute z-score: `(value - peer_mean) / peer_std`
3. Winsorize at 2.5th and 97.5th percentiles before z-scoring

**Universe**: Peer group only for v1 (5-10 stocks). Simpler, more relevant, statistically sufficient for educational purposes.

### Edge Cases

**Negative Earnings**: Assign floor percentile rank (5th percentile) for P/E calculation. Note limitation to user.

**Missing Data**: Exclude that factor for that stock, warn user, ask how to proceed.

**Short History**: If peer IPO'd < 3 years ago, include with shorter history and flag limitation. Ask user to confirm.

**Non-US Tickers**: Reject with explanation (v1 US-only).

### Factor Exposure via Regression (Optional Enhancement)

If Fama-French data available:
1. Calculate excess returns (stock return - risk-free rate)
2. Run OLS regression against factor returns
3. Report betas with standard errors and t-stats
4. Use rolling 36-month window, start chart at month 36 (cleaner)

---

## Part 5: Risk Decomposition

### Variance Decomposition

Decompose total return variance:
```
Total Variance = Σ(βᵢ² × σᵢ²) + Σ(βᵢ × βⱼ × σᵢⱼ) + σ²ε
```

Report as percentages:
```
Market beta:     42%
Momentum:        18%
Value:           12%
Idiosyncratic:   28%
```

### Correlation Analysis

- Primary stock vs. each peer
- Primary stock vs. benchmark
- Primary stock vs. each factor

---

## Part 6: Scenario Analysis

### Pre-Built Scenarios (Hardcoded)

| Scenario | Value | Momentum | Quality | Growth | Size | Volatility |
|----------|-------|----------|---------|--------|------|------------|
| Rate Shock (+100bps) | +3% | -2% | +1% | -5% | 0% | +2% |
| Risk-Off / Flight to Quality | -2% | -8% | +4% | -3% | -1% | +5% |
| Economic Recession | -4% | -3% | +3% | -6% | -2% | +4% |
| Cyclical Rotation | +2% | +3% | -1% | +2% | +1% | +1% |

### Scenario Calculation

```
Expected Return = Σ(Factor Exposure × Scenario Factor Return)
```

### Custom Scenario (Guided)

Agent walks user through:
1. "In your scenario, what happens to interest rates?"
2. "How does that affect growth stocks?"
3. "What about defensive/quality names?"

Translates narrative answers into factor return assumptions.

---

## Part 7: Excel Output Specification

### Filename Convention

- In-progress: `{TICKER}_factor_analysis_DRAFT_{YYYYMMDD_HHMMSS}.xlsx`
- Complete: `{TICKER}_factor_analysis_{YYYYMMDD_HHMMSS}.xlsx`

### Tab Structure (9 Tabs)

#### Tab 1: Executive Summary
- Company info: name, ticker, sector, market cap, price, 52W range
- Factor profile table: score and peer rank for each factor
- Horizontal bar chart for factor visualization (no radar — native Excel)
- One-sentence interpretation
- Risk decomposition pie chart
- Key scenario outcomes table

#### Tab 2: Factor Exposures — Detail
- Per-factor table:
  - Raw characteristic value
  - Peer median
  - Z-score
  - Percentile rank
  - Regression beta (if available) with std error, t-stat
- Written interpretation section (agent-generated)

#### Tab 3: Peer Comparison
- Matrix: stocks (rows) × factors (columns)
- Cells: z-scores with conditional formatting (green/red)
- Columns: price, market cap, each factor z-score, composite score, rank
- Sorted by composite score

#### Tab 4: Risk Decomposition
- Variance decomposition table + pie chart
- Idiosyncratic risk interpretation
- Correlation matrix: primary vs. peers vs. benchmark

#### Tab 5: Historical Factor Exposures
- Rolling 36-month charts for each factor exposure
- Table noting significant changes over time

#### Tab 6: Scenario Analysis
- Pre-built scenarios table with factor assumptions and calculated returns
- Waterfall chart showing factor contributions
- Custom scenario input section with Excel formulas

#### Tab 7: Fundamental Data
- Income statement summary (3-5 years)
- Balance sheet summary
- Key ratios with historical context
- Valuation multiples

#### Tab 8: Price Data and Returns
- Monthly returns for full period (daily truncated for size)
- Cumulative return chart: primary vs. peers vs. benchmark

#### Tab 9: Methodology Notes
- Factor calculation methodology
- Data sources
- Normalization approach
- Glossary of terms

### Styling Standards

- **Colors**: Navy headers (#4472C4), white background, green positive, red negative
- **Formatting**: Percentages to 1 decimal, currency with commas, ratios to 2 decimals
- **Charts**: Clear titles, axis labels, legends
- **Print-ready**: Page breaks, headers/footers, print areas configured
- **Named ranges**: Key inputs for formula readability

---

## Part 8: Technical Implementation

### Integration Points

#### Framework System
Extend `/framework` command to accept `factors`:
```python
# In ui/repl.py, extend framework handling
if framework_name == "factors":
    return self._start_factor_session()
```

#### Session Management
Extend existing `Session` class via `metadata` field:
```python
session.metadata = {
    "type": "factors",
    "stage": 3,  # Current stage (1-8)
    "primary_ticker": "NVDA",
    "peers": ["AMD", "INTC", "QCOM"],
    "selected_factors": ["value", "momentum", "quality"],
    "weights": {"value": 0.33, "momentum": 0.33, "quality": 0.34},
    "computed_data": {...},  # Cached calculations
    "draft_excel_path": "/path/to/draft.xlsx"
}
```

#### Excel Generation
Extend `tools/excel.py` with mode parameter:
```python
def generate_excel(
    ticker: str,
    include_ratios: bool = True,
    compare_tickers: list[str] | None = None,
    mode: str = "standard"  # "standard" | "factors"
) -> ToolResult:
```

Or create `generate_factor_excel()` function in same module.

### New Modules

```
src/bullsh/
├── factors/
│   ├── __init__.py
│   ├── session.py      # Factor session state machine
│   ├── calculator.py   # Factor calculation logic
│   ├── scenarios.py    # Scenario definitions and calculations
│   ├── regression.py   # Time-series factor regression
│   └── prompts.py      # Professor persona prompts
```

### Subagent Architecture

For Stage 4 (data fetching), spawn parallel subagents:
```python
# In agent/orchestrator.py
async def fetch_factor_data(tickers: list[str], benchmark: str):
    tasks = [
        self.spawn_subagent("data_fetch", ticker=t)
        for t in tickers
    ]
    tasks.append(self.spawn_subagent("data_fetch", ticker=benchmark))
    tasks.append(self.spawn_subagent("ff_fetch"))  # Fama-French

    results = await asyncio.gather(*tasks, return_exceptions=True)
    return self.merge_results(results)
```

### Error Handling

- **Invalid ticker**: Reject with explanation, ask for alternative
- **Non-US ticker**: Reject with "US-only for v1" explanation
- **Missing data**: Warn and exclude, ask user to confirm
- **Rate limits (AlphaVantage)**: Fall back to Yahoo, explain tradeoff
- **Network failure**: Fail if no cache, retry with backoff, ask user

---

## Part 9: Constraints & Boundaries

### V1 Scope

**In Scope**:
- Single primary ticker with 1-6 peers
- US equities only
- 6 core factors (value, momentum, quality, growth, size, volatility)
- 4 pre-built scenarios + 1 custom
- 9-tab Excel output
- Session persistence and resume
- Educational professor persona

**Out of Scope (Stretch/V2)**:
- Automatic peer selection
- Portfolio mode (multiple primary tickers)
- Non-US equities
- Custom factor definitions
- PDF report generation (beyond existing `/export pdf`)
- Backtesting
- Real-time data

### Performance Targets

- Data fetch: < 60 seconds for 6 tickers
- Factor calculation: < 5 seconds
- Excel generation: < 10 seconds
- Session resume: < 2 seconds

### Dependencies

**Required**:
- openpyxl (existing)
- numpy (for calculations)
- pandas (for time-series handling)

**Optional**:
- scipy (for regression, can fall back to numpy)
- AlphaVantage API key (enhanced data)

---

## Part 10: Success Criteria

1. **Educational Value**: A finance student can complete a session and explain factor investing concepts afterward

2. **Professional Output**: The Excel file can be dropped into a stock pitch deck without modification

3. **Usability**: Clear progression, no confusion about next steps, graceful error handling

4. **Reliability**: Completes successfully with Yahoo Finance alone (no API keys required)

5. **Performance**: Total session time < 10 minutes for engaged user

6. **Extensibility**: Architecture supports adding new factors and scenarios in future versions

---

## Appendix A: Sample Session Transcript

```
> /framework factors

Professor: Welcome to the Multi-Factor Analysis module. I'll guide you through
building a comprehensive factor profile for a stock of your choice.

Factor investing is a systematic approach to understanding what drives a stock's
returns. Rather than treating each stock as unique, we identify common
characteristics — factors — that explain return patterns across many stocks.

Let's begin. What stock would you like to analyze?

> NVDA

Professor: Excellent choice. NVIDIA Corporation (NVDA) is a $1.2T semiconductor
company. I've validated that we have sufficient data history for analysis.

Before we calculate NVDA's factor exposures, we need context. Factor scores are
relative — a P/E of 35 means nothing until we compare it to peers.

Who are NVDA's peers for this analysis? Enter 2-6 ticker symbols, or type
"suggest" for recommendations.

> AMD, INTC, QCOM, AVGO

Professor: Good peer set. Let me validate each:
  ✓ AMD  - Advanced Micro Devices ($195B) - Semiconductors
  ✓ INTC - Intel Corporation ($108B) - Semiconductors
  ✓ QCOM - Qualcomm ($167B) - Semiconductors
  ✓ AVGO - Broadcom ($690B) - Semiconductors

All validated. We have a focused semiconductor peer group with market caps
ranging from $108B to $1.2T.

Now, which factors would you like to include in your analysis?

[1] Value      - Price multiples vs. peers (P/E, P/B, EV/EBITDA)
[2] Momentum   - Recent price performance (12-1 month return)
[3] Quality    - Profitability and stability (ROE, leverage, earnings consistency)
[4] Growth     - Revenue and earnings growth rates
[5] Size       - Market capitalization (log-scaled)
[6] Volatility - Price volatility and market beta

[A] All factors (recommended for learning)

> A

Professor: All six factors selected. For weighting, equal-weight is standard
practice — each factor contributes equally to the composite score.

Would you like to:
[1] Use equal weights (16.7% each) — recommended
[2] Set custom weights

> 1

Professor: Equal weights confirmed. Now I'll gather the data we need. This
involves fetching 3 years of price history and current fundamentals for 5
companies plus the S&P 500 benchmark.

[Fetching NVDA price and fundamentals...]
[Fetching AMD price and fundamentals...]
[Fetching INTC price and fundamentals...]
[Fetching QCOM price and fundamentals...]
[Fetching AVGO price and fundamentals...]
[Fetching ^GSPC benchmark...]
[Fetching Fama-French factor returns...]

While we wait — a key insight: The data we're gathering serves two purposes.
Current fundamentals (P/E, ROE, etc.) give us point-in-time factor
characteristics. Price history lets us calculate momentum, volatility, and
run time-series factor regressions.

All data retrieved. Let's calculate your factors.

─────────────────────────────────────────────────────────────
FACTOR CALCULATIONS FOR NVDA
─────────────────────────────────────────────────────────────

VALUE FACTOR
├── P/E Ratio:     65.2 (peer median: 28.4)
├── P/B Ratio:     42.1 (peer median: 8.2)
├── EV/EBITDA:     48.3 (peer median: 18.7)
├── Value Z-Score: -1.42
└── Interpretation: NVDA trades at a significant premium to semiconductor peers

Calculation detail:
  P/E z-score   = (65.2 - 28.4) / 22.1 = +1.67  (inverted: -1.67)
  P/B z-score   = (42.1 - 8.2) / 18.4  = +1.84  (inverted: -1.84)
  EV/EBITDA z   = (48.3 - 18.7) / 16.2 = +1.83  (inverted: -1.83)
  Composite     = average(-1.67, -1.84, -1.83) = -1.78
  Winsorized    = -1.42 (capped at 2.5th percentile)

...

[Session continues through all stages]
```

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| Z-score | Standard deviations from the mean; (value - mean) / std |
| Factor exposure | A stock's sensitivity to a systematic factor |
| Idiosyncratic risk | Stock-specific risk not explained by factors |
| Cross-sectional | Comparing across stocks at a point in time |
| Time-series | Analyzing one stock's behavior over time |
| Winsorization | Capping extreme values to reduce outlier impact |
| Excess return | Return above the risk-free rate |

