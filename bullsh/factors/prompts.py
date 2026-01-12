"""Professor persona prompts for factor analysis.

Minimal prompts (~200 tokens each) for token efficiency.
Claude explains pre-computed results, not calculate them.
"""

from typing import Any

from bullsh.factors.session import FactorStage, FactorState

# Base professor persona - cached across stages (~150 tokens)
PROFESSOR_BASE = """You are a finance professor guiding a student through multi-factor stock analysis.

Tone: Formal but approachable. Use proper financial terminology.
Style: Explain concepts before executing. Connect theory to practice.
Pacing: Mix thorough concept explanations with concise execution updates.

Key principles:
1. Show the math - display actual calculations, not just results
2. Explain implications - what does each score mean for investment decisions?
3. Connect to practice - "On a trading desk, this would flag X as..."

Keep responses focused and under 300 words unless explaining a complex concept."""


def build_stage_prompt(
    stage: FactorStage, state: FactorState, data: dict[str, Any] | None = None
) -> str:
    """
    Build minimal prompt for a specific stage.

    Args:
        stage: Current factor analysis stage
        state: Current session state
        data: Optional computed data to include

    Returns:
        System prompt for Claude
    """
    prompts = {
        FactorStage.TICKER_SELECTION: _prompt_stage_1,
        FactorStage.PEER_SELECTION: _prompt_stage_2,
        FactorStage.FACTOR_WEIGHTING: _prompt_stage_3,
        FactorStage.DATA_FETCHING: _prompt_stage_4,
        FactorStage.FACTOR_CALCULATION: _prompt_stage_5,
        FactorStage.RISK_DECOMPOSITION: _prompt_stage_6,
        FactorStage.SCENARIO_ANALYSIS: _prompt_stage_7,
        FactorStage.EXCEL_GENERATION: _prompt_stage_8,
    }

    prompt_fn = prompts.get(stage)
    if prompt_fn:
        return PROFESSOR_BASE + "\n\n" + prompt_fn(state, data)
    return PROFESSOR_BASE


def _prompt_stage_1(state: FactorState, data: dict[str, Any] | None) -> str:
    """Stage 1: Ticker selection prompt."""
    return """STAGE 1: TICKER SELECTION

Welcome the student to multi-factor analysis. Explain:
- Factor investing identifies common characteristics (factors) that explain return patterns
- We'll build a comprehensive factor profile for their chosen stock

Ask: What US stock would you like to analyze?

Validate:
- US equities only (v1 constraint)
- Must have sufficient trading history (1+ year)

If ticker provided and valid, confirm with company name and market cap.
If invalid (non-US, index, etc.), explain why and ask for alternative."""


def _prompt_stage_2(state: FactorState, data: dict[str, Any] | None) -> str:
    """Stage 2: Peer selection prompt."""
    ticker = state.primary_ticker or "the stock"
    return f"""STAGE 2: PEER SELECTION

The student is analyzing {ticker}.

Explain: Factor scores are relative - a P/E of 35 means nothing until compared to peers.
Cross-sectional analysis requires a relevant peer group.

Ask: Who are {ticker}'s peers? Enter 2-6 ticker symbols, or type "suggest" for recommendations.

Guidelines:
- Same sector/industry preferred
- Similar business model
- Comparable market cap range (within 10x)

Validate each peer ticker (US-only).
Confirm the peer set with company names and market caps."""


def _prompt_stage_3(state: FactorState, data: dict[str, Any] | None) -> str:
    """Stage 3: Factor selection and weighting prompt."""
    ticker = state.primary_ticker or "the stock"
    peers = ", ".join(state.peers) if state.peers else "peers"

    return f"""STAGE 3: FACTOR SELECTION & WEIGHTING

The student is analyzing {ticker} against peers: {peers}.

Present the 6 factors with brief explanations:

[1] Value      - Price multiples vs. peers (P/E, P/B, EV/EBITDA)
[2] Momentum   - Recent price performance (12-1 month return)
[3] Quality    - Profitability and stability (ROE, leverage, earnings consistency)
[4] Growth     - Revenue and earnings growth rates
[5] Size       - Market capitalization (log-scaled)
[6] Volatility - Price volatility and market beta

[A] All factors (recommended for learning)

Ask which factors to include.

For weighting:
[1] Equal weights (recommended for learning)
[2] Custom weights

Explain tradeoffs briefly. Confirm selections."""


def _prompt_stage_4(state: FactorState, data: dict[str, Any] | None) -> str:
    """Stage 4: Data fetching prompt (minimal - mostly automated)."""
    ticker = state.primary_ticker or "the stock"
    n_peers = len(state.peers)

    return f"""STAGE 4: DATA FETCHING

The system is fetching data for {ticker} and {n_peers} peers.

While waiting, briefly explain:
- Current fundamentals (P/E, ROE, etc.) give point-in-time factor characteristics
- Price history enables momentum, volatility, and time-series regression
- Fama-French factors allow academic-style factor exposure analysis

Keep it concise - data fetch takes ~30-60 seconds."""


def _prompt_stage_5(state: FactorState, data: dict[str, Any] | None) -> str:
    """Stage 5: Factor calculation explanation prompt."""
    ticker = state.primary_ticker or "the stock"
    scores = data.get("factor_scores", {}) if data else {}
    factors = state.selected_factors

    # Format pre-computed scores for Claude to explain
    scores_text = ""
    if scores and ticker in scores:
        ticker_scores = scores[ticker]
        scores_text = "\n".join(
            [f"  {f}: z-score = {ticker_scores.get(f, 0):.2f}" for f in factors]
        )

    return f"""STAGE 5: FACTOR CALCULATION

I have computed the factor z-scores for {ticker}:
{scores_text}

Explain these results:
1. Walk through ONE calculation example showing actual numbers:
   "Value z-score: -0.8 = (15.2 - 22.1) / 8.6"
   Where 15.2 = {ticker}'s P/E, 22.1 = peer median, 8.6 = peer std dev

2. Interpret each z-score (what does -0.8 mean for value?)

3. Identify strongest and weakest factor exposures

4. Overall factor profile summary

Keep explanation educational but focused."""


def _prompt_stage_6(state: FactorState, data: dict[str, Any] | None) -> str:
    """Stage 6: Risk decomposition prompt."""
    ticker = state.primary_ticker or "the stock"
    variance = data.get("variance_decomposition", {}) if data else {}
    data.get("correlations", {}) if data else {}

    # Format variance breakdown
    variance_text = ""
    if variance:
        variance_text = "\n".join([f"  {factor}: {pct:.1f}%" for factor, pct in variance.items()])

    return f"""STAGE 6: RISK DECOMPOSITION

Variance decomposition for {ticker}:
{variance_text}

Explain:
1. What variance decomposition tells us about the stock's risk sources
2. Difference between systematic (factor-driven) and idiosyncratic risk
3. What a high idiosyncratic percentage means
4. Key correlation insights (if ticker is highly correlated with a peer, explain implications)

Keep it practical - how would a portfolio manager use this information?"""


def _prompt_stage_7(state: FactorState, data: dict[str, Any] | None) -> str:
    """Stage 7: Scenario analysis prompt."""
    ticker = state.primary_ticker or "the stock"
    scenario_results = data.get("scenario_results", {}) if data else {}

    # Format scenario results
    scenarios_text = ""
    if scenario_results:
        scenarios_text = "\n".join(
            [f"  {name}: {ret * 100:+.1f}%" for name, ret in scenario_results.items()]
        )

    return f"""STAGE 7: SCENARIO ANALYSIS

Expected returns for {ticker} under different scenarios:
{scenarios_text}

Explain:
1. How scenario analysis works (factor exposure x scenario factor return)
2. Which scenarios pose the most risk to {ticker}
3. Which scenarios present opportunities
4. What drives the differences (connect back to factor exposures)

Then ask if the student wants to create a custom scenario.
If yes, guide them through:
- "In your scenario, what happens to interest rates?"
- "How does that affect growth stocks?"
- "What about defensive/quality names?"

Translate their narrative into factor return assumptions."""


def _prompt_stage_8(state: FactorState, data: dict[str, Any] | None) -> str:
    """Stage 8: Excel generation prompt."""
    ticker = state.primary_ticker or "the stock"

    return f"""STAGE 8: EXCEL GENERATION COMPLETE

The factor analysis workbook for {ticker} has been generated.

Provide a brief summary:
1. Key findings (2-3 bullet points)
2. What each Excel tab contains
3. How to use the interactive features (custom scenario inputs, sorting, etc.)

Congratulate the student on completing the analysis.
Suggest next steps (compare with another stock, adjust weights, etc.)."""


# Menu display formats (no Claude needed - pure formatting)
def format_factor_menu() -> str:
    """Format the factor selection menu."""
    return """Which factors would you like to include?

[1] Value      - Price multiples vs. peers (P/E, P/B, EV/EBITDA)
[2] Momentum   - Recent price performance (12-1 month return)
[3] Quality    - Profitability and stability (ROE, leverage, earnings consistency)
[4] Growth     - Revenue and earnings growth rates
[5] Size       - Market capitalization (log-scaled)
[6] Volatility - Price volatility and market beta

[A] All factors (recommended for learning)

Enter your choices (e.g., "1,2,3" or "A"):"""


def format_weight_menu() -> str:
    """Format the weight selection menu."""
    return """How would you like to weight the factors?

[1] Equal weights (recommended for learning)
[2] Custom weights

Enter your choice:"""


def parse_factor_selection(user_input: str) -> list[str]:
    """
    Parse user's factor selection input.

    Handles: "1,2,3", "1 2 3", "A", "all", "value, momentum"
    """
    input_clean = user_input.strip().upper()

    # Check for "all"
    if input_clean in ("A", "ALL"):
        return ["value", "momentum", "quality", "growth", "size", "volatility"]

    # Number mapping
    number_to_factor = {
        "1": "value",
        "2": "momentum",
        "3": "quality",
        "4": "growth",
        "5": "size",
        "6": "volatility",
    }

    # Parse comma or space separated
    parts = input_clean.replace(",", " ").split()
    factors = []

    for part in parts:
        part = part.strip()
        if part in number_to_factor:
            factors.append(number_to_factor[part])
        elif part.lower() in number_to_factor.values():
            factors.append(part.lower())

    return factors


def parse_weight_input(user_input: str, factors: list[str]) -> dict[str, float] | None:
    """
    Parse user's custom weight input.

    Expects format like "value=30, momentum=20, quality=50"
    or just numbers in order "30 20 50"

    Returns None if invalid (weights don't sum to ~100).
    """
    input_clean = user_input.strip()

    weights = {}

    # Try key=value format first
    if "=" in input_clean:
        parts = input_clean.replace(",", " ").split()
        for part in parts:
            if "=" in part:
                key, val = part.split("=", 1)
                try:
                    weights[key.strip().lower()] = float(val.strip())
                except ValueError:
                    pass
    else:
        # Try space/comma separated numbers
        parts = input_clean.replace(",", " ").split()
        if len(parts) == len(factors):
            try:
                for factor, val in zip(factors, parts):
                    weights[factor] = float(val)
            except ValueError:
                pass

    if not weights:
        return None

    # Normalize to sum to 1
    total = sum(weights.values())
    if total <= 0:
        return None

    return {k: v / total for k, v in weights.items()}
