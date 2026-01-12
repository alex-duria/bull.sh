"""Time-series factor regression using OLS.

Calculates factor betas, standard errors, and t-statistics.
Pure Python/numpy implementation - no Claude API calls.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class RegressionResult:
    """Result of a single factor regression."""

    alpha: float  # Intercept (abnormal return)
    alpha_t_stat: float  # t-statistic for alpha
    betas: dict[str, float]  # Factor betas
    t_stats: dict[str, float]  # t-statistics for betas
    std_errors: dict[str, float]  # Standard errors for betas
    r_squared: float  # R-squared
    adj_r_squared: float  # Adjusted R-squared
    n_observations: int  # Number of observations
    residual_std: float  # Standard deviation of residuals (idiosyncratic vol)


def run_factor_regression(
    stock_returns: pd.Series,
    factor_returns: pd.DataFrame,
    rf_returns: pd.Series | None = None,
) -> RegressionResult | None:
    """
    Run OLS regression of stock excess returns on factor returns.

    Model: R_stock - R_f = alpha + beta1*F1 + beta2*F2 + ... + epsilon

    Args:
        stock_returns: Series of stock returns indexed by date
        factor_returns: DataFrame of factor returns (Mkt-RF, SMB, HML) indexed by date
        rf_returns: Series of risk-free rate (optional, for excess return calculation)

    Returns:
        RegressionResult or None if insufficient data
    """
    # Align dates
    common_dates = stock_returns.index.intersection(factor_returns.index)
    if len(common_dates) < 36:  # Minimum 3 years of monthly data
        return None

    y = stock_returns.loc[common_dates].values

    # Subtract risk-free rate if provided
    if rf_returns is not None:
        rf_common = rf_returns.reindex(common_dates).fillna(0).values
        y = y - rf_common

    X = factor_returns.loc[common_dates].values
    factor_names = factor_returns.columns.tolist()

    # Add constant (intercept)
    n = len(y)
    k = X.shape[1]
    X_with_const = np.column_stack([np.ones(n), X])

    # OLS: beta = (X'X)^-1 X'y
    try:
        XtX = X_with_const.T @ X_with_const
        XtX_inv = np.linalg.inv(XtX)
        Xty = X_with_const.T @ y
        coefficients = XtX_inv @ Xty
    except np.linalg.LinAlgError:
        return None

    # Extract alpha and betas
    alpha = coefficients[0]
    betas = {name: coefficients[i + 1] for i, name in enumerate(factor_names)}

    # Residuals
    y_hat = X_with_const @ coefficients
    residuals = y - y_hat
    residual_std = np.std(residuals, ddof=k + 1)

    # R-squared
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)

    # Standard errors: se = sqrt(diag((X'X)^-1) * s^2)
    mse = ss_res / (n - k - 1)
    var_coef = np.diag(XtX_inv) * mse
    std_errors_all = np.sqrt(var_coef)

    alpha_se = std_errors_all[0]
    alpha_t_stat = alpha / alpha_se if alpha_se > 0 else 0

    std_errors = {name: std_errors_all[i + 1] for i, name in enumerate(factor_names)}
    t_stats = {
        name: betas[name] / std_errors[name] if std_errors[name] > 0 else 0 for name in factor_names
    }

    return RegressionResult(
        alpha=float(alpha),
        alpha_t_stat=float(alpha_t_stat),
        betas=betas,
        t_stats=t_stats,
        std_errors=std_errors,
        r_squared=float(r_squared),
        adj_r_squared=float(adj_r_squared),
        n_observations=n,
        residual_std=float(residual_std),
    )


def run_rolling_regression(
    stock_returns: pd.Series,
    factor_returns: pd.DataFrame,
    rf_returns: pd.Series | None = None,
    window: int = 36,  # 36 months = 3 years
) -> dict[str, list[tuple[str, float]]]:
    """
    Run rolling window regressions to get time-varying factor exposures.

    Args:
        stock_returns: Monthly stock returns
        factor_returns: Monthly factor returns
        rf_returns: Monthly risk-free rate
        window: Rolling window size in months

    Returns:
        Dict mapping factor_name -> list of (date, beta) tuples
    """
    # Align dates
    common_dates = stock_returns.index.intersection(factor_returns.index)
    if rf_returns is not None:
        common_dates = common_dates.intersection(rf_returns.index)

    common_dates = sorted(common_dates)
    if len(common_dates) < window:
        return {}

    factor_names = factor_returns.columns.tolist()
    rolling_betas: dict[str, list[tuple[str, float]]] = {name: [] for name in factor_names}
    rolling_betas["alpha"] = []

    for i in range(window, len(common_dates) + 1):
        window_dates = common_dates[i - window : i]
        end_date = window_dates[-1]

        window_stock = stock_returns.loc[window_dates]
        window_factors = factor_returns.loc[window_dates]
        window_rf = rf_returns.loc[window_dates] if rf_returns is not None else None

        result = run_factor_regression(window_stock, window_factors, window_rf)
        if result:
            date_str = (
                end_date.strftime("%Y-%m-%d") if hasattr(end_date, "strftime") else str(end_date)
            )
            rolling_betas["alpha"].append((date_str, result.alpha))
            for name in factor_names:
                rolling_betas[name].append((date_str, result.betas.get(name, 0)))

    return rolling_betas


def calculate_variance_decomposition(
    regression_result: RegressionResult,
    factor_variances: dict[str, float],
    factor_covariances: dict[tuple[str, str], float] | None = None,
) -> dict[str, float]:
    """
    Decompose stock return variance by factor contribution.

    Total Variance = sum(beta_i^2 * var(F_i)) + sum(beta_i * beta_j * cov(F_i, F_j)) + var(epsilon)

    Args:
        regression_result: OLS regression output
        factor_variances: Variance of each factor
        factor_covariances: Covariances between factors (optional)

    Returns:
        Dict mapping factor_name -> percentage of variance explained
    """
    betas = regression_result.betas
    factor_names = list(betas.keys())

    # Systematic variance from each factor
    systematic_by_factor = {}
    total_systematic = 0.0

    for name in factor_names:
        beta = betas.get(name, 0)
        var = factor_variances.get(name, 0)
        contribution = (beta**2) * var
        systematic_by_factor[name] = contribution
        total_systematic += contribution

    # Add covariance terms if provided
    if factor_covariances:
        for i, name_i in enumerate(factor_names):
            for name_j in factor_names[i + 1 :]:
                beta_i = betas.get(name_i, 0)
                beta_j = betas.get(name_j, 0)
                cov = factor_covariances.get((name_i, name_j), 0)
                cov_contribution = 2 * beta_i * beta_j * cov
                total_systematic += cov_contribution
                # Attribute evenly to both factors
                systematic_by_factor[name_i] = (
                    systematic_by_factor.get(name_i, 0) + cov_contribution / 2
                )
                systematic_by_factor[name_j] = (
                    systematic_by_factor.get(name_j, 0) + cov_contribution / 2
                )

    # Idiosyncratic variance
    idio_var = regression_result.residual_std**2

    # Total variance
    total_variance = total_systematic + idio_var

    if total_variance == 0:
        return {"idiosyncratic": 100.0}

    # Convert to percentages
    decomposition = {
        name: (var / total_variance) * 100 for name, var in systematic_by_factor.items()
    }
    decomposition["idiosyncratic"] = (idio_var / total_variance) * 100

    return decomposition


def calculate_correlations(
    price_histories: dict[str, dict[str, Any]],
    benchmark_history: dict[str, Any] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Calculate correlation matrix between stocks and benchmark.

    Args:
        price_histories: Dict mapping ticker -> price history data
        benchmark_history: Benchmark price history (optional)

    Returns:
        Correlation matrix as nested dict
    """
    # Build returns DataFrame
    returns_data = {}

    for ticker, history in price_histories.items():
        returns = history.get("returns", [])
        if returns:
            returns_data[ticker] = pd.Series({r[0]: r[1] for r in returns})

    if benchmark_history:
        benchmark_returns = benchmark_history.get("returns", [])
        if benchmark_returns:
            returns_data["Benchmark"] = pd.Series({r[0]: r[1] for r in benchmark_returns})

    if len(returns_data) < 2:
        return {}

    # Create DataFrame
    df = pd.DataFrame(returns_data)

    # Calculate correlation matrix
    corr_matrix = df.corr()

    # Convert to nested dict
    result = {}
    for col in corr_matrix.columns:
        result[col] = {row: float(corr_matrix.loc[row, col]) for row in corr_matrix.index}

    return result


def prepare_fama_french_data(
    ff_data: dict[str, Any],
    stock_history: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare Fama-French data aligned with stock returns.

    Converts daily to monthly and aligns dates.

    Args:
        ff_data: Fama-French data from fetcher
        stock_history: Stock price history

    Returns:
        (factor_returns DataFrame, risk_free Series)
    """
    # Parse factor data
    factors = ff_data.get("factors", {})
    if not factors:
        return pd.DataFrame(), pd.Series()

    # Create daily factor DataFrame
    dates = []
    mkt_rf = []
    smb = []
    hml = []
    rf = []

    for date_str, values in factors.items():
        dates.append(pd.to_datetime(date_str))
        mkt_rf.append(values.get("Mkt-RF", 0))
        smb.append(values.get("SMB", 0))
        hml.append(values.get("HML", 0))
        rf.append(values.get("RF", 0))

    factor_df = pd.DataFrame(
        {
            "Mkt-RF": mkt_rf,
            "SMB": smb,
            "HML": hml,
        },
        index=pd.DatetimeIndex(dates),
    )
    factor_df = factor_df.sort_index()

    rf_series = pd.Series(rf, index=pd.DatetimeIndex(dates)).sort_index()

    # Resample to monthly (sum daily returns)
    factor_monthly = factor_df.resample("M").sum()
    rf_monthly = rf_series.resample("M").sum()

    return factor_monthly, rf_monthly


def prepare_stock_returns(
    stock_history: dict[str, Any],
) -> pd.Series:
    """
    Prepare stock returns as monthly Series.

    Args:
        stock_history: Stock price history from fetcher

    Returns:
        Monthly returns Series
    """
    closes = stock_history.get("closes", [])
    if len(closes) < 2:
        return pd.Series()

    # Create daily close Series
    dates = [pd.to_datetime(c[0]) for c in closes]
    prices = [c[1] for c in closes]
    close_series = pd.Series(prices, index=pd.DatetimeIndex(dates)).sort_index()

    # Resample to monthly (last price of month)
    monthly_close = close_series.resample("M").last()

    # Calculate returns
    monthly_returns = monthly_close.pct_change().dropna()

    return monthly_returns


def format_regression_result(
    ticker: str,
    result: RegressionResult,
) -> str:
    """
    Format regression result for display.

    Returns multi-line string with key statistics.
    """
    lines = [
        f"Factor Regression for {ticker}",
        "=" * 40,
        f"Alpha (annualized): {result.alpha * 12 * 100:.2f}% (t={result.alpha_t_stat:.2f})",
        "",
        "Factor Betas:",
    ]

    for factor, beta in result.betas.items():
        t_stat = result.t_stats.get(factor, 0)
        sig = "*" if abs(t_stat) > 1.96 else ""
        lines.append(f"  {factor}: {beta:.3f} (t={t_stat:.2f}){sig}")

    lines.extend(
        [
            "",
            f"R-squared: {result.r_squared:.1%}",
            f"Adjusted R-squared: {result.adj_r_squared:.1%}",
            f"Observations: {result.n_observations}",
            f"Residual Std (monthly): {result.residual_std:.2%}",
            f"Idiosyncratic Vol (annualized): {result.residual_std * np.sqrt(12):.1%}",
        ]
    )

    return "\n".join(lines)
