"""Multi-factor stock analysis module.

Provides educational factor investing analysis with professor-guided sessions.
Factor calculations are pure Python (no LLM calls) for token efficiency.
"""

from bullsh.factors.calculator import (
    FACTORS,
    calculate_composite_score,
    calculate_factor_scores,
    calculate_z_score,
    winsorize,
)
from bullsh.factors.excel_factors import (
    generate_factor_excel,
)
from bullsh.factors.fetcher import (
    fetch_all_factor_data,
    fetch_fama_french,
    fetch_price_history,
)
from bullsh.factors.prompts import (
    build_stage_prompt,
    format_factor_menu,
    parse_factor_selection,
)
from bullsh.factors.regression import (
    calculate_correlations,
    calculate_variance_decomposition,
    run_factor_regression,
    run_rolling_regression,
)
from bullsh.factors.scenarios import (
    SCENARIOS,
    calculate_all_scenario_returns,
    calculate_scenario_return,
)
from bullsh.factors.session import (
    FactorSession,
    FactorStage,
    FactorState,
    validate_peer_set,
    validate_us_ticker,
)

__all__ = [
    # Calculator
    "FACTORS",
    "calculate_z_score",
    "winsorize",
    "calculate_factor_scores",
    "calculate_composite_score",
    # Scenarios
    "SCENARIOS",
    "calculate_scenario_return",
    "calculate_all_scenario_returns",
    # Session
    "FactorStage",
    "FactorState",
    "FactorSession",
    "validate_us_ticker",
    "validate_peer_set",
    # Fetcher
    "fetch_price_history",
    "fetch_fama_french",
    "fetch_all_factor_data",
    # Regression
    "run_factor_regression",
    "run_rolling_regression",
    "calculate_variance_decomposition",
    "calculate_correlations",
    # Prompts
    "build_stage_prompt",
    "format_factor_menu",
    "parse_factor_selection",
    # Excel
    "generate_factor_excel",
]
