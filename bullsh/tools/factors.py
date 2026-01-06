"""Factor calculation tool executors.

These wrap the pure Python factor calculations to expose them as agent tools.
"""

from datetime import datetime

import numpy as np

from bullsh.tools.base import ToolResult, ToolStatus
from bullsh.factors.fetcher import fetch_all_factor_data
from bullsh.factors.calculator import calculate_factor_scores, calculate_composite_score, FACTORS
from bullsh.factors.regression import (
    run_factor_regression,
    calculate_variance_decomposition,
    prepare_fama_french_data,
    prepare_stock_returns,
    RegressionResult,
)


async def calculate_factors(
    ticker: str,
    peers: list[str],
    factors: list[str] | None = None,
) -> ToolResult:
    """
    Calculate multi-factor exposures for a stock.

    This is the guardrail function - forces real calculations instead of
    letting the agent theorize about factors.

    Args:
        ticker: Primary stock ticker
        peers: 2-6 peer tickers for cross-sectional comparison
        factors: Which factors to calculate (default: all)

    Returns:
        ToolResult with factor z-scores and composite score
    """
    try:
        # Validate inputs
        ticker = ticker.upper().strip()
        peers = [p.upper().strip() for p in peers if p.strip()]

        if len(peers) < 2:
            return ToolResult(
                data={"error": "Need at least 2 peers for cross-sectional analysis"},
                confidence=0.0,
                status=ToolStatus.FAILED,
                tool_name="calculate_factors",
                ticker=ticker,
                error_message="Insufficient peers: need 2-6 peers for z-score calculation",
            )

        if len(peers) > 6:
            peers = peers[:6]  # Cap at 6

        # Default to all factors
        if not factors:
            factors = list(FACTORS.keys())
        else:
            factors = [f.lower().strip() for f in factors if f.lower().strip() in FACTORS]
            if not factors:
                factors = list(FACTORS.keys())

        # Fetch data for all tickers
        all_data = await fetch_all_factor_data(ticker, peers)

        yahoo_data = all_data.get("yahoo_data", {})
        price_history = all_data.get("price_history")

        if not yahoo_data.get(ticker):
            return ToolResult(
                data={"error": f"No market data found for {ticker}"},
                confidence=0.0,
                status=ToolStatus.FAILED,
                tool_name="calculate_factors",
                ticker=ticker,
                error_message=f"Could not fetch data for {ticker}",
            )

        # Calculate factor scores (pure Python - zero tokens!)
        profiles = calculate_factor_scores(
            ticker,
            peers,
            yahoo_data,
            factors,
            price_history,
        )

        if ticker not in profiles:
            return ToolResult(
                data={"error": "Factor calculation failed"},
                confidence=0.0,
                status=ToolStatus.FAILED,
                tool_name="calculate_factors",
                ticker=ticker,
                error_message="Factor score calculation returned no results",
            )

        primary_profile = profiles[ticker]

        # Calculate composite score (equal weights)
        weights = {f: 1.0 / len(factors) for f in factors}
        composite = calculate_composite_score(primary_profile.scores, weights)
        primary_profile.composite_score = composite

        # Format results for the agent
        factor_scores = {}
        for factor_name, score in primary_profile.scores.items():
            # Get first component value as representative raw value
            raw_val = None
            if score.components:
                first_component = next(iter(score.components.values()), None)
                raw_val = round(first_component, 2) if first_component is not None else None

            factor_scores[factor_name] = {
                "z_score": round(score.z_score, 2),
                "percentile": round(score.percentile, 1),  # Already 0-100
                "components": {k: round(v, 2) if v else None for k, v in score.components.items()},
                "component_z_scores": {k: round(v, 2) for k, v in score.component_z_scores.items()},
                "interpretation": _interpret_z_score(score.z_score, factor_name),
            }

        # Peer comparison
        peer_scores = {}
        for peer_ticker, profile in profiles.items():
            if peer_ticker != ticker:
                peer_composite = calculate_composite_score(profile.scores, weights)
                peer_scores[peer_ticker] = {
                    "composite": round(peer_composite, 2),
                    "scores": {f: round(s.z_score, 2) for f, s in profile.scores.items()},
                }

        result_data = {
            "ticker": ticker,
            "peers_used": peers,
            "factors_calculated": factors,
            "factor_scores": factor_scores,
            "composite_score": round(composite, 2),
            "composite_interpretation": _interpret_composite(composite),
            "peer_comparison": peer_scores,
            "methodology": "Cross-sectional z-scores vs peer group, winsorized at 2.5/97.5 percentiles",
        }

        return ToolResult(
            data=result_data,
            confidence=0.85,  # High confidence since this is computed
            status=ToolStatus.SUCCESS,
            tool_name="calculate_factors",
            ticker=ticker,
            fetched_at=datetime.now(),
        )

    except Exception as e:
        return ToolResult(
            data={"error": str(e)},
            confidence=0.0,
            status=ToolStatus.FAILED,
            tool_name="calculate_factors",
            ticker=ticker,
            error_message=f"Factor calculation error: {e}",
        )


async def run_factor_regression_tool(
    ticker: str,
    window_months: int = 36,
) -> ToolResult:
    """
    Run Fama-French factor regression for a stock.

    Decomposes stock returns into market, size, and value factor exposures.

    Args:
        ticker: Stock ticker
        window_months: Rolling window for regression (default 36)

    Returns:
        ToolResult with factor betas, R-squared, and variance decomposition
    """
    try:
        ticker = ticker.upper().strip()

        # Fetch price history and Fama-French data
        all_data = await fetch_all_factor_data(ticker, [])

        price_history = all_data.get("price_history", {})
        ff_data = all_data.get("fama_french")

        if not price_history.get(ticker):
            return ToolResult(
                data={"error": f"No price history for {ticker}"},
                confidence=0.0,
                status=ToolStatus.FAILED,
                tool_name="run_factor_regression",
                ticker=ticker,
                error_message=f"Could not fetch price history for {ticker}",
            )

        if not ff_data or not ff_data.get("factors"):
            return ToolResult(
                data={"error": "Could not fetch Fama-French factor returns"},
                confidence=0.0,
                status=ToolStatus.FAILED,
                tool_name="run_factor_regression",
                ticker=ticker,
                error_message="Fama-French data unavailable",
            )

        # Prepare data - returns (factor_df, rf_series)
        stock_returns = prepare_stock_returns(price_history[ticker])
        factor_returns, rf_returns = prepare_fama_french_data(ff_data, price_history[ticker])

        if stock_returns is None or len(stock_returns) < window_months:
            return ToolResult(
                data={"error": f"Insufficient price history (need {window_months}+ months)"},
                confidence=0.0,
                status=ToolStatus.FAILED,
                tool_name="run_factor_regression",
                ticker=ticker,
                error_message=f"Need at least {window_months} months of data",
            )

        if factor_returns.empty:
            return ToolResult(
                data={"error": "Could not prepare factor returns data"},
                confidence=0.0,
                status=ToolStatus.FAILED,
                tool_name="run_factor_regression",
                ticker=ticker,
                error_message="Factor data preparation failed",
            )

        # Run regression - returns RegressionResult dataclass or None
        reg_result = run_factor_regression(stock_returns, factor_returns, rf_returns)

        if reg_result is None:
            return ToolResult(
                data={"error": "Regression failed - insufficient overlapping data"},
                confidence=0.0,
                status=ToolStatus.FAILED,
                tool_name="run_factor_regression",
                ticker=ticker,
                error_message="Could not run regression - need 36+ months of aligned data",
            )

        # Calculate factor variances for decomposition
        factor_variances = {}
        for col in factor_returns.columns:
            factor_variances[col] = float(factor_returns[col].var())

        var_decomp = calculate_variance_decomposition(reg_result, factor_variances)

        result_data = {
            "ticker": ticker,
            "regression_window": f"{reg_result.n_observations} months",
            "factor_betas": {
                "market_beta": round(reg_result.betas.get("Mkt-RF", 0), 3),
                "size_beta": round(reg_result.betas.get("SMB", 0), 3),
                "value_beta": round(reg_result.betas.get("HML", 0), 3),
            },
            "t_statistics": {
                "market_t": round(reg_result.t_stats.get("Mkt-RF", 0), 2),
                "size_t": round(reg_result.t_stats.get("SMB", 0), 2),
                "value_t": round(reg_result.t_stats.get("HML", 0), 2),
            },
            "alpha": round(reg_result.alpha * 12 * 100, 2),  # Annualized %
            "alpha_t_stat": round(reg_result.alpha_t_stat, 2),
            "r_squared": round(reg_result.r_squared * 100, 1),
            "adj_r_squared": round(reg_result.adj_r_squared * 100, 1),
            "idiosyncratic_vol": round(reg_result.residual_std * np.sqrt(12) * 100, 1),  # Annualized %
            "variance_decomposition": {
                k: round(v, 1) for k, v in var_decomp.items()
            },
            "interpretation": _interpret_regression_result(reg_result),
        }

        return ToolResult(
            data=result_data,
            confidence=0.9,
            status=ToolStatus.SUCCESS,
            tool_name="run_factor_regression",
            ticker=ticker,
            fetched_at=datetime.now(),
        )

    except Exception as e:
        return ToolResult(
            data={"error": str(e)},
            confidence=0.0,
            status=ToolStatus.FAILED,
            tool_name="run_factor_regression",
            ticker=ticker,
            error_message=f"Regression error: {e}",
        )


def _interpret_z_score(z: float, factor: str) -> str:
    """Interpret a factor z-score."""
    if abs(z) < 0.5:
        return "neutral"
    elif z >= 1.5:
        return f"strong {factor} tilt"
    elif z >= 0.5:
        return f"moderate {factor} tilt"
    elif z <= -1.5:
        return f"strong anti-{factor} tilt"
    else:
        return f"moderate anti-{factor} tilt"


def _interpret_composite(score: float) -> str:
    """Interpret composite factor score."""
    if score >= 1.0:
        return "strongly factor-positive vs peers"
    elif score >= 0.5:
        return "moderately factor-positive vs peers"
    elif score >= -0.5:
        return "neutral factor exposure vs peers"
    elif score >= -1.0:
        return "moderately factor-negative vs peers"
    else:
        return "strongly factor-negative vs peers"


def _interpret_regression_result(result: RegressionResult) -> str:
    """Interpret factor regression results from RegressionResult dataclass."""
    market_beta = result.betas.get("Mkt-RF", 1.0)
    smb = result.betas.get("SMB", 0)
    hml = result.betas.get("HML", 0)
    r_sq = result.r_squared

    parts = []

    if market_beta > 1.2:
        parts.append("high market sensitivity (aggressive)")
    elif market_beta < 0.8:
        parts.append("low market sensitivity (defensive)")
    else:
        parts.append("market-neutral beta")

    if smb > 0.3:
        parts.append("small-cap tilt")
    elif smb < -0.3:
        parts.append("large-cap tilt")

    if hml > 0.3:
        parts.append("value tilt")
    elif hml < -0.3:
        parts.append("growth tilt")

    if r_sq > 0.7:
        parts.append("high factor explanatory power")
    elif r_sq < 0.3:
        parts.append("significant idiosyncratic risk")

    return "; ".join(parts) if parts else "balanced factor exposure"
