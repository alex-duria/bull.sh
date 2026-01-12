"""Pre-built scenarios and scenario return calculations.

All calculations are pure Python - no LLM calls needed.
"""

from dataclasses import dataclass


@dataclass
class Scenario:
    """A market scenario with expected factor returns."""

    name: str
    display_name: str
    description: str
    factor_returns: dict[str, float]  # factor -> expected return


# Pre-built scenarios from spec
# Factor returns represent expected contribution to stock returns
# based on factor exposure under each scenario
SCENARIOS: dict[str, Scenario] = {
    "rate_shock": Scenario(
        name="rate_shock",
        display_name="Rate Shock (+100bps)",
        description="Interest rates rise 100 basis points. Value outperforms as growth discounting increases. Momentum reverses on rate-sensitive names.",
        factor_returns={
            "value": 0.03,  # Value stocks benefit
            "momentum": -0.02,  # Momentum reverses
            "quality": 0.01,  # Quality modest benefit
            "growth": -0.05,  # Growth hurt by higher discount rates
            "size": 0.0,  # Neutral
            "volatility": 0.02,  # Vol increases
        },
    ),
    "risk_off": Scenario(
        name="risk_off",
        display_name="Risk-Off / Flight to Quality",
        description="Market stress drives capital to defensive positions. Quality and low-volatility outperform. Momentum crashes as crowded trades unwind.",
        factor_returns={
            "value": -0.02,  # Value underperforms
            "momentum": -0.08,  # Momentum crashes
            "quality": 0.04,  # Flight to quality
            "growth": -0.03,  # Growth sells off
            "size": -0.01,  # Small caps hurt
            "volatility": 0.05,  # High vol names spike
        },
    ),
    "recession": Scenario(
        name="recession",
        display_name="Economic Recession",
        description="Broad economic contraction. Earnings fall across sectors. Quality and defensive positioning dominate.",
        factor_returns={
            "value": -0.04,  # Value traps emerge
            "momentum": -0.03,  # Momentum struggles
            "quality": 0.03,  # Quality resilience
            "growth": -0.06,  # Growth expectations cut
            "size": -0.02,  # Small caps vulnerable
            "volatility": 0.04,  # Vol premium expands
        },
    ),
    "cyclical_rotation": Scenario(
        name="cyclical_rotation",
        display_name="Cyclical Rotation",
        description="Economic optimism drives rotation into cyclicals. Value and momentum align as beaten-down sectors rally.",
        factor_returns={
            "value": 0.02,  # Value rally
            "momentum": 0.03,  # Momentum benefits
            "quality": -0.01,  # Quality lags
            "growth": 0.02,  # Growth participates
            "size": 0.01,  # Small caps benefit
            "volatility": 0.01,  # Vol compresses slightly
        },
    ),
}


def calculate_scenario_return(
    factor_exposures: dict[str, float],
    scenario: Scenario | dict[str, float],
) -> float:
    """
    Calculate expected return under a scenario.

    Expected Return = sum(factor_exposure_i * scenario_factor_return_i)

    Args:
        factor_exposures: dict mapping factor name -> exposure (z-score)
        scenario: Either a Scenario object or dict of factor returns

    Returns:
        Expected return as decimal (0.05 = 5%)
    """
    if isinstance(scenario, Scenario):
        factor_returns = scenario.factor_returns
    else:
        factor_returns = scenario

    total_return = 0.0
    for factor, exposure in factor_exposures.items():
        scenario_return = factor_returns.get(factor, 0.0)
        total_return += exposure * scenario_return

    return total_return


def calculate_all_scenario_returns(
    factor_exposures: dict[str, float],
) -> dict[str, float]:
    """
    Calculate expected returns under all pre-built scenarios.

    Returns dict mapping scenario_name -> expected return.
    """
    return {
        name: calculate_scenario_return(factor_exposures, scenario)
        for name, scenario in SCENARIOS.items()
    }


def calculate_scenario_contribution(
    factor_exposures: dict[str, float],
    scenario: Scenario,
) -> dict[str, float]:
    """
    Break down scenario return by factor contribution.

    Useful for waterfall charts showing which factors drive returns.

    Returns dict mapping factor -> contribution to return.
    """
    contributions = {}
    for factor, exposure in factor_exposures.items():
        scenario_return = scenario.factor_returns.get(factor, 0.0)
        contributions[factor] = exposure * scenario_return
    return contributions


@dataclass
class CustomScenario:
    """User-defined custom scenario."""

    name: str
    factor_returns: dict[str, float]
    narrative: str  # User's description of the scenario


def create_custom_scenario(
    name: str,
    narrative: str,
    factor_returns: dict[str, float],
) -> CustomScenario:
    """
    Create a custom scenario from user inputs.

    The professor guides the user through defining factor returns
    based on their narrative description.
    """
    # Validate all factors have returns
    required_factors = ["value", "momentum", "quality", "growth", "size", "volatility"]
    for factor in required_factors:
        if factor not in factor_returns:
            factor_returns[factor] = 0.0

    return CustomScenario(
        name=name,
        factor_returns=factor_returns,
        narrative=narrative,
    )


def format_scenario_results(
    ticker: str,
    factor_exposures: dict[str, float],
    scenario_returns: dict[str, float],
) -> str:
    """
    Format scenario results for display.

    Returns formatted string showing expected returns under each scenario.
    """
    lines = [f"Scenario Analysis for {ticker}", "=" * 40]

    for scenario_name, expected_return in scenario_returns.items():
        scenario = SCENARIOS.get(scenario_name)
        if scenario:
            pct = expected_return * 100
            sign = "+" if pct >= 0 else ""
            lines.append(f"\n{scenario.display_name}")
            lines.append(f"  Expected Return: {sign}{pct:.1f}%")
            lines.append(f"  {scenario.description[:80]}...")

    return "\n".join(lines)


def get_scenario_sensitivity(
    factor_exposures: dict[str, float],
) -> dict[str, str]:
    """
    Identify which scenarios the stock is most sensitive to.

    Returns dict mapping scenario -> sensitivity assessment.
    """
    returns = calculate_all_scenario_returns(factor_exposures)

    sensitivity = {}
    for name, ret in returns.items():
        if abs(ret) > 0.05:
            sensitivity[name] = "HIGH"
        elif abs(ret) > 0.02:
            sensitivity[name] = "MODERATE"
        else:
            sensitivity[name] = "LOW"

    return sensitivity


def identify_risk_scenarios(
    factor_exposures: dict[str, float],
    threshold: float = -0.03,
) -> list[tuple[str, float]]:
    """
    Identify scenarios where expected return is below threshold.

    Returns list of (scenario_name, expected_return) sorted by severity.
    """
    returns = calculate_all_scenario_returns(factor_exposures)

    risks = [(name, ret) for name, ret in returns.items() if ret < threshold]

    return sorted(risks, key=lambda x: x[1])


def identify_opportunity_scenarios(
    factor_exposures: dict[str, float],
    threshold: float = 0.03,
) -> list[tuple[str, float]]:
    """
    Identify scenarios where expected return exceeds threshold.

    Returns list of (scenario_name, expected_return) sorted by opportunity.
    """
    returns = calculate_all_scenario_returns(factor_exposures)

    opportunities = [(name, ret) for name, ret in returns.items() if ret > threshold]

    return sorted(opportunities, key=lambda x: x[1], reverse=True)
