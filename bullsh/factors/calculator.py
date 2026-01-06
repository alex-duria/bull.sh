"""Pure Python factor calculations - NO Claude API calls.

All factor math is deterministic and computed locally for token efficiency.
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

# Factor definitions: factor_name -> list of component metrics
FACTORS = {
    "value": ["pe_ratio", "pb_ratio", "ev_ebitda"],
    "momentum": ["return_12m_1m", "high_52w_proximity"],
    "quality": ["roe", "debt_equity_inv", "earnings_stability"],
    "growth": ["revenue_growth_yoy", "earnings_growth_yoy"],
    "size": ["ln_market_cap"],
    "volatility": ["realized_vol_60d", "beta"],
}

# Sign conventions: True = higher is better, False = higher is worse
# For value metrics, we invert (lower P/E = better value = higher score)
FACTOR_SIGN = {
    "value": False,       # Lower multiples = better value
    "momentum": True,     # Higher momentum = better
    "quality": True,      # Higher quality = better
    "growth": True,       # Higher growth = better
    "size": True,         # Larger = higher score (neutral interpretation)
    "volatility": True,   # Higher vol = higher score (user interprets)
}


@dataclass
class FactorScore:
    """Score for a single factor."""
    factor: str
    z_score: float
    percentile: float
    components: dict[str, float]  # raw component values
    component_z_scores: dict[str, float]


@dataclass
class TickerFactorProfile:
    """Complete factor profile for a ticker."""
    ticker: str
    scores: dict[str, FactorScore]
    composite_score: float
    composite_rank: int  # rank among peer group


def calculate_z_score(value: float, peer_values: list[float]) -> float:
    """
    Calculate z-score for a value relative to peer group.

    Z-score = (value - mean) / std_dev

    Returns 0 if std_dev is 0 (all values identical).
    """
    if not peer_values:
        return 0.0

    arr = np.array(peer_values)
    mean = np.mean(arr)
    std = np.std(arr)

    if std == 0:
        return 0.0

    return float((value - mean) / std)


def winsorize(values: list[float], lower_pct: float = 2.5, upper_pct: float = 97.5) -> list[float]:
    """
    Winsorize values by capping at percentile thresholds.

    This reduces the impact of outliers on z-score calculations.
    Values below lower_pct percentile are set to that threshold.
    Values above upper_pct percentile are set to that threshold.
    """
    if not values:
        return values

    arr = np.array(values)
    lower_bound = np.percentile(arr, lower_pct)
    upper_bound = np.percentile(arr, upper_pct)

    return [
        max(lower_bound, min(upper_bound, v))
        for v in values
    ]


def percentile_rank(value: float, all_values: list[float]) -> float:
    """
    Calculate percentile rank (0-100) of value within distribution.
    """
    if not all_values:
        return 50.0

    arr = np.array(all_values)
    return float(np.sum(arr <= value) / len(arr) * 100)


def extract_value_metrics(yahoo_data: dict[str, Any]) -> dict[str, float | None]:
    """Extract value factor components from Yahoo data."""
    return {
        "pe_ratio": yahoo_data.get("pe_ratio"),
        "pb_ratio": yahoo_data.get("pb_ratio"),  # may need to compute
        "ev_ebitda": yahoo_data.get("ev_ebitda"),  # may need to compute
    }


def extract_momentum_metrics(
    yahoo_data: dict[str, Any],
    price_history: dict[str, Any] | None = None
) -> dict[str, float | None]:
    """
    Extract momentum factor components.

    - return_12m_1m: 12-month return minus 1-month return (classic momentum)
    - high_52w_proximity: current price / 52-week high
    """
    metrics: dict[str, float | None] = {
        "return_12m_1m": None,
        "high_52w_proximity": None,
    }

    # 52-week high proximity
    price = yahoo_data.get("price")
    high_52w = yahoo_data.get("52w_high")
    if price and high_52w and high_52w > 0:
        metrics["high_52w_proximity"] = price / high_52w

    # Calculate momentum from price history if available
    if price_history:
        returns = calculate_returns_from_history(price_history)
        if returns:
            ret_12m = returns.get("return_12m", 0)
            ret_1m = returns.get("return_1m", 0)
            metrics["return_12m_1m"] = ret_12m - ret_1m

    return metrics


def extract_quality_metrics(yahoo_data: dict[str, Any]) -> dict[str, float | None]:
    """
    Extract quality factor components.

    - roe: Return on equity
    - debt_equity_inv: Inverted debt/equity (lower debt = higher quality)
    - earnings_stability: Would need historical data
    """
    return {
        "roe": yahoo_data.get("roe"),
        "debt_equity_inv": None,  # computed below if debt_equity available
        "earnings_stability": None,  # requires historical earnings
    }


def extract_growth_metrics(yahoo_data: dict[str, Any]) -> dict[str, float | None]:
    """Extract growth factor components."""
    return {
        "revenue_growth_yoy": yahoo_data.get("revenue_growth"),
        "earnings_growth_yoy": yahoo_data.get("earnings_growth"),
    }


def extract_size_metrics(yahoo_data: dict[str, Any]) -> dict[str, float | None]:
    """Extract size factor components (log market cap)."""
    market_cap = yahoo_data.get("market_cap")
    ln_market_cap = None
    if market_cap and market_cap > 0:
        ln_market_cap = math.log(market_cap)

    return {"ln_market_cap": ln_market_cap}


def extract_volatility_metrics(
    yahoo_data: dict[str, Any],
    price_history: dict[str, Any] | None = None
) -> dict[str, float | None]:
    """
    Extract volatility factor components.

    - realized_vol_60d: 60-day realized volatility (annualized)
    - beta: Market beta from Yahoo
    """
    metrics: dict[str, float | None] = {
        "realized_vol_60d": None,
        "beta": yahoo_data.get("beta"),
    }

    if price_history:
        vol = calculate_realized_volatility(price_history, window=60)
        metrics["realized_vol_60d"] = vol

    return metrics


def calculate_returns_from_history(price_history: dict[str, Any]) -> dict[str, float]:
    """
    Calculate various return periods from price history.

    Expects price_history to have 'closes' as list of (date, close) tuples
    sorted by date ascending.
    """
    closes = price_history.get("closes", [])
    if len(closes) < 2:
        return {}

    # Sort by date if not already
    closes = sorted(closes, key=lambda x: x[0])

    current_price = closes[-1][1]
    returns = {}

    # Approximate trading days
    trading_days_1m = 21
    trading_days_12m = 252

    if len(closes) > trading_days_1m:
        price_1m_ago = closes[-trading_days_1m][1]
        if price_1m_ago > 0:
            returns["return_1m"] = (current_price - price_1m_ago) / price_1m_ago

    if len(closes) > trading_days_12m:
        price_12m_ago = closes[-trading_days_12m][1]
        if price_12m_ago > 0:
            returns["return_12m"] = (current_price - price_12m_ago) / price_12m_ago

    return returns


def calculate_realized_volatility(
    price_history: dict[str, Any],
    window: int = 60
) -> float | None:
    """
    Calculate annualized realized volatility from daily returns.

    Uses the last `window` trading days.
    """
    closes = price_history.get("closes", [])
    if len(closes) < window + 1:
        return None

    # Get last window+1 closes to compute window daily returns
    recent_closes = [c[1] for c in sorted(closes, key=lambda x: x[0])[-(window + 1):]]

    # Daily returns
    daily_returns = []
    for i in range(1, len(recent_closes)):
        if recent_closes[i - 1] > 0:
            ret = (recent_closes[i] - recent_closes[i - 1]) / recent_closes[i - 1]
            daily_returns.append(ret)

    if len(daily_returns) < window // 2:
        return None

    # Annualized volatility: daily std * sqrt(252)
    std = np.std(daily_returns)
    return float(std * np.sqrt(252))


def calculate_single_factor_score(
    ticker: str,
    factor: str,
    ticker_data: dict[str, Any],
    peer_data: dict[str, dict[str, Any]],
    price_histories: dict[str, dict[str, Any]] | None = None,
) -> FactorScore:
    """
    Calculate z-score for a single factor.

    1. Extract component metrics for ticker and all peers
    2. Winsorize each component across the peer group
    3. Calculate z-score for each component
    4. Average component z-scores (inverted for value metrics)
    5. Return composite factor z-score
    """
    # Get extractors for this factor
    extractors = {
        "value": extract_value_metrics,
        "momentum": lambda d: extract_momentum_metrics(
            d, price_histories.get(d.get("ticker", "")) if price_histories else None
        ),
        "quality": extract_quality_metrics,
        "growth": extract_growth_metrics,
        "size": extract_size_metrics,
        "volatility": lambda d: extract_volatility_metrics(
            d, price_histories.get(d.get("ticker", "")) if price_histories else None
        ),
    }

    extractor = extractors.get(factor)
    if not extractor:
        return FactorScore(
            factor=factor,
            z_score=0.0,
            percentile=50.0,
            components={},
            component_z_scores={},
        )

    # Extract metrics for ticker and peers
    ticker_metrics = extractor(ticker_data)
    peer_metrics_list = [extractor(pd) for pd in peer_data.values()]
    all_tickers = [ticker] + list(peer_data.keys())
    all_metrics = [ticker_metrics] + peer_metrics_list

    # Get component names
    components = FACTORS.get(factor, [])

    # Calculate z-score for each component
    component_z_scores: dict[str, float] = {}
    valid_z_scores: list[float] = []

    for component in components:
        # Gather values for this component across all tickers
        all_values = []
        ticker_value = None

        for i, metrics in enumerate(all_metrics):
            val = metrics.get(component)
            if val is not None and not math.isnan(val):
                all_values.append(val)
                if i == 0:  # ticker is first
                    ticker_value = val

        if ticker_value is None or len(all_values) < 2:
            continue

        # Winsorize
        winsorized = winsorize(all_values)
        ticker_idx = 0  # ticker value is first in all_values if it exists
        ticker_winsorized = winsorized[ticker_idx] if ticker_idx < len(winsorized) else ticker_value

        # Z-score
        z = calculate_z_score(ticker_winsorized, winsorized)

        # Invert for value factor (lower = better)
        if not FACTOR_SIGN.get(factor, True):
            z = -z

        component_z_scores[component] = z
        valid_z_scores.append(z)

    # Average component z-scores
    avg_z = np.mean(valid_z_scores) if valid_z_scores else 0.0

    # Calculate percentile
    # For this we need all tickers' composite scores, so return raw for now
    percentile = 50.0 + (avg_z * 15.87)  # Approximate from z-score
    percentile = max(0.0, min(100.0, percentile))

    return FactorScore(
        factor=factor,
        z_score=float(avg_z),
        percentile=percentile,
        components=ticker_metrics,
        component_z_scores=component_z_scores,
    )


def calculate_factor_scores(
    primary_ticker: str,
    peers: list[str],
    all_data: dict[str, dict[str, Any]],
    selected_factors: list[str],
    price_histories: dict[str, dict[str, Any]] | None = None,
) -> dict[str, TickerFactorProfile]:
    """
    Calculate factor scores for all tickers (primary + peers).

    Returns dict mapping ticker -> TickerFactorProfile.
    """
    all_tickers = [primary_ticker] + peers
    profiles: dict[str, TickerFactorProfile] = {}

    # Calculate scores for each ticker
    for ticker in all_tickers:
        ticker_data = all_data.get(ticker, {})
        peer_data = {t: all_data.get(t, {}) for t in all_tickers if t != ticker}

        scores: dict[str, FactorScore] = {}
        for factor in selected_factors:
            score = calculate_single_factor_score(
                ticker=ticker,
                factor=factor,
                ticker_data=ticker_data,
                peer_data=peer_data,
                price_histories=price_histories,
            )
            scores[factor] = score

        profiles[ticker] = TickerFactorProfile(
            ticker=ticker,
            scores=scores,
            composite_score=0.0,  # Calculated below
            composite_rank=0,
        )

    return profiles


def calculate_composite_score(
    scores: dict[str, FactorScore],
    weights: dict[str, float],
) -> float:
    """
    Calculate weighted composite score from individual factor scores.

    Composite = sum(weight_i * z_score_i)
    """
    total = 0.0
    weight_sum = 0.0

    for factor, score in scores.items():
        weight = weights.get(factor, 0.0)
        total += weight * score.z_score
        weight_sum += weight

    if weight_sum == 0:
        return 0.0

    return total / weight_sum


def rank_by_composite(profiles: dict[str, TickerFactorProfile]) -> None:
    """
    Rank tickers by composite score (in-place).
    Higher composite = rank 1.
    """
    # Sort by composite score descending
    sorted_tickers = sorted(
        profiles.keys(),
        key=lambda t: profiles[t].composite_score,
        reverse=True,
    )

    for rank, ticker in enumerate(sorted_tickers, start=1):
        profiles[ticker].composite_rank = rank


def format_factor_calculation(
    ticker: str,
    factor: str,
    score: FactorScore,
    peer_median: float,
    peer_std: float,
) -> str:
    """
    Format factor calculation for professor explanation.

    Returns string showing the math:
    "Z-score: -0.8 = (15.2 - 22.1) / 8.6"
    """
    # Get first non-None component for illustration
    for component, z in score.component_z_scores.items():
        raw_val = score.components.get(component)
        if raw_val is not None:
            return (
                f"{factor.title()} z-score: {score.z_score:.2f}\n"
                f"  {component}: {raw_val:.2f} (peer median: {peer_median:.2f}, std: {peer_std:.2f})\n"
                f"  Calculated as: ({raw_val:.2f} - {peer_median:.2f}) / {peer_std:.2f} = {z:.2f}"
            )

    return f"{factor.title()} z-score: {score.z_score:.2f}"
