"""Factor analysis session state machine.

Manages 8-stage progression through factor analysis.
State persists in Session.metadata for resume support.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from bullsh.storage.sessions import Session, get_session_manager


class FactorStage(Enum):
    """The 8 stages of factor analysis plus completion."""
    TICKER_SELECTION = 1
    PEER_SELECTION = 2
    FACTOR_WEIGHTING = 3
    DATA_FETCHING = 4
    FACTOR_CALCULATION = 5
    RISK_DECOMPOSITION = 6
    SCENARIO_ANALYSIS = 7
    EXCEL_GENERATION = 8
    COMPLETE = 9

    @property
    def display_name(self) -> str:
        """Human-readable stage name."""
        names = {
            1: "Ticker Selection",
            2: "Peer Selection",
            3: "Factor Selection & Weighting",
            4: "Data Fetching",
            5: "Factor Calculation",
            6: "Risk Decomposition",
            7: "Scenario Analysis",
            8: "Excel Generation",
            9: "Complete",
        }
        return names.get(self.value, "Unknown")

    @property
    def requires_claude(self) -> bool:
        """Whether this stage requires Claude API calls."""
        # Stages 1-3: User input with professor guidance
        # Stage 4: Pure data fetch (no Claude)
        # Stages 5-7: Explanation of computed results
        # Stage 8: Pure Excel generation (no Claude)
        return self.value in {1, 2, 3, 5, 6, 7}


# Default equal weights for all 6 factors
DEFAULT_WEIGHTS = {
    "value": 1/6,
    "momentum": 1/6,
    "quality": 1/6,
    "growth": 1/6,
    "size": 1/6,
    "volatility": 1/6,
}

# All available factors
ALL_FACTORS = ["value", "momentum", "quality", "growth", "size", "volatility"]


@dataclass
class FactorState:
    """
    State for a factor analysis session.

    Persisted in Session.metadata['factors'].
    Designed to be small (~2KB JSON) - only computed results stored.
    """
    # Current stage
    stage: FactorStage = FactorStage.TICKER_SELECTION

    # User selections (Stages 1-3)
    primary_ticker: str | None = None
    peers: list[str] = field(default_factory=list)
    selected_factors: list[str] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)

    # Computed results (Stages 5-7) - NOT raw data
    factor_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    # Format: {ticker: {factor: z_score}}

    regression_betas: dict[str, dict[str, float]] = field(default_factory=dict)
    # Format: {ticker: {factor: beta}}

    variance_decomposition: dict[str, float] = field(default_factory=dict)
    # Format: {factor: variance_pct}

    correlation_matrix: dict[str, dict[str, float]] = field(default_factory=dict)
    # Format: {ticker1: {ticker2: correlation}}

    scenario_results: dict[str, float] = field(default_factory=dict)
    # Format: {scenario_name: expected_return}

    custom_scenario: dict[str, Any] | None = None
    # User-defined custom scenario

    # Output (Stage 8)
    draft_excel_path: str | None = None
    final_excel_path: str | None = None

    # Metadata
    data_fetched_at: str | None = None
    completed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON storage."""
        return {
            "stage": self.stage.value,
            "primary_ticker": self.primary_ticker,
            "peers": self.peers,
            "selected_factors": self.selected_factors,
            "weights": self.weights,
            "factor_scores": self.factor_scores,
            "regression_betas": self.regression_betas,
            "variance_decomposition": self.variance_decomposition,
            "correlation_matrix": self.correlation_matrix,
            "scenario_results": self.scenario_results,
            "custom_scenario": self.custom_scenario,
            "draft_excel_path": self.draft_excel_path,
            "final_excel_path": self.final_excel_path,
            "data_fetched_at": self.data_fetched_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FactorState":
        """Deserialize from dict."""
        state = cls()
        state.stage = FactorStage(data.get("stage", 1))
        state.primary_ticker = data.get("primary_ticker")
        state.peers = data.get("peers", [])
        state.selected_factors = data.get("selected_factors", [])
        state.weights = data.get("weights", {})
        state.factor_scores = data.get("factor_scores", {})
        state.regression_betas = data.get("regression_betas", {})
        state.variance_decomposition = data.get("variance_decomposition", {})
        state.correlation_matrix = data.get("correlation_matrix", {})
        state.scenario_results = data.get("scenario_results", {})
        state.custom_scenario = data.get("custom_scenario")
        state.draft_excel_path = data.get("draft_excel_path")
        state.final_excel_path = data.get("final_excel_path")
        state.data_fetched_at = data.get("data_fetched_at")
        state.completed_at = data.get("completed_at")
        return state


class FactorSession:
    """
    Manages factor analysis session lifecycle.

    Wraps Session and provides stage-based progression with persistence.
    """

    def __init__(self, session: Session):
        """
        Initialize factor session from existing Session.

        If session already has factor state, loads it.
        Otherwise, creates fresh state.
        """
        self.session = session
        self._state: FactorState | None = None

    @property
    def state(self) -> FactorState:
        """Get current factor state (lazy loaded)."""
        if self._state is None:
            if "factors" in self.session.metadata:
                self._state = FactorState.from_dict(self.session.metadata["factors"])
            else:
                self._state = FactorState()
        return self._state

    @property
    def stage(self) -> FactorStage:
        """Current stage shortcut."""
        return self.state.stage

    @property
    def is_complete(self) -> bool:
        """Whether analysis is complete."""
        return self.state.stage == FactorStage.COMPLETE

    def save(self) -> None:
        """Persist state to session metadata."""
        self.session.metadata["factors"] = self.state.to_dict()
        get_session_manager().save(self.session)

    def advance_stage(self) -> FactorStage:
        """
        Move to next stage.

        Returns new stage.
        """
        current = self.state.stage.value
        if current < FactorStage.COMPLETE.value:
            self.state.stage = FactorStage(current + 1)
            self.save()
        return self.state.stage

    def go_back(self) -> bool:
        """
        Go back one stage.

        Returns True if successful, False if already at stage 1.
        """
        current = self.state.stage.value
        if current > 1:
            self.state.stage = FactorStage(current - 1)
            self.save()
            return True
        return False

    def reset(self) -> None:
        """Reset to stage 1, clearing all data."""
        self._state = FactorState()
        self.save()

    def set_ticker(self, ticker: str) -> None:
        """Set primary ticker (Stage 1)."""
        self.state.primary_ticker = ticker.upper()
        self.save()

    def set_peers(self, peers: list[str]) -> None:
        """Set peer tickers (Stage 2)."""
        self.state.peers = [p.upper() for p in peers]
        self.save()

    def set_factors(self, factors: list[str], weights: dict[str, float] | None = None) -> None:
        """
        Set selected factors and weights (Stage 3).

        If weights not provided, uses equal weights.
        """
        self.state.selected_factors = factors
        if weights:
            self.state.weights = weights
        else:
            # Equal weight
            n = len(factors)
            self.state.weights = {f: 1/n for f in factors}
        self.save()

    def set_factor_scores(self, scores: dict[str, dict[str, float]]) -> None:
        """Store computed factor scores (Stage 5)."""
        self.state.factor_scores = scores
        self.save()

    def set_regression_betas(self, betas: dict[str, dict[str, float]]) -> None:
        """Store regression betas (Stage 5)."""
        self.state.regression_betas = betas
        self.save()

    def set_variance_decomposition(self, decomposition: dict[str, float]) -> None:
        """Store variance decomposition (Stage 6)."""
        self.state.variance_decomposition = decomposition
        self.save()

    def set_correlation_matrix(self, correlations: dict[str, dict[str, float]]) -> None:
        """Store correlation matrix (Stage 6)."""
        self.state.correlation_matrix = correlations
        self.save()

    def set_scenario_results(self, results: dict[str, float]) -> None:
        """Store scenario analysis results (Stage 7)."""
        self.state.scenario_results = results
        self.save()

    def set_custom_scenario(self, scenario: dict[str, Any]) -> None:
        """Store custom scenario (Stage 7)."""
        self.state.custom_scenario = scenario
        self.save()

    def set_excel_path(self, path: str, final: bool = False) -> None:
        """Set Excel output path (Stage 8)."""
        if final:
            self.state.final_excel_path = path
        else:
            self.state.draft_excel_path = path
        self.save()

    def mark_complete(self) -> None:
        """Mark analysis as complete."""
        from datetime import datetime
        self.state.stage = FactorStage.COMPLETE
        self.state.completed_at = datetime.now().isoformat()
        self.save()

    def get_all_tickers(self) -> list[str]:
        """Get all tickers (primary + peers)."""
        tickers = []
        if self.state.primary_ticker:
            tickers.append(self.state.primary_ticker)
        tickers.extend(self.state.peers)
        return tickers

    def get_progress(self) -> tuple[int, int]:
        """Get progress as (current_stage, total_stages)."""
        return (self.state.stage.value, 8)

    def get_progress_bar(self) -> str:
        """Get visual progress bar."""
        current, total = self.get_progress()
        filled = current - 1  # Stages completed
        return "=" * filled + ">" + "-" * (total - filled - 1)

    def get_status_display(self) -> str:
        """Get status display string."""
        current, total = self.get_progress()
        bar = self.get_progress_bar()
        stage_name = self.state.stage.display_name
        ticker = self.state.primary_ticker or "Not selected"
        return f"[{bar}] Stage {current}/{total}: {stage_name} | Ticker: {ticker}"

    def can_advance(self) -> tuple[bool, str]:
        """
        Check if current stage requirements are met to advance.

        Returns (can_advance, reason_if_not).
        """
        stage = self.state.stage

        if stage == FactorStage.TICKER_SELECTION:
            if not self.state.primary_ticker:
                return False, "Primary ticker not set"
            return True, ""

        elif stage == FactorStage.PEER_SELECTION:
            if len(self.state.peers) < 2:
                return False, "At least 2 peers required"
            if len(self.state.peers) > 6:
                return False, "Maximum 6 peers allowed"
            return True, ""

        elif stage == FactorStage.FACTOR_WEIGHTING:
            if not self.state.selected_factors:
                return False, "No factors selected"
            if not self.state.weights:
                return False, "Weights not set"
            return True, ""

        elif stage == FactorStage.DATA_FETCHING:
            if not self.state.data_fetched_at:
                return False, "Data not yet fetched"
            return True, ""

        elif stage == FactorStage.FACTOR_CALCULATION:
            if not self.state.factor_scores:
                return False, "Factor scores not calculated"
            return True, ""

        elif stage == FactorStage.RISK_DECOMPOSITION:
            if not self.state.variance_decomposition:
                return False, "Variance decomposition not complete"
            return True, ""

        elif stage == FactorStage.SCENARIO_ANALYSIS:
            if not self.state.scenario_results:
                return False, "Scenario analysis not complete"
            return True, ""

        elif stage == FactorStage.EXCEL_GENERATION:
            if not self.state.final_excel_path:
                return False, "Excel not generated"
            return True, ""

        return True, ""


def validate_us_ticker(ticker: str) -> tuple[bool, str]:
    """
    Validate that a ticker is US-only (v1 constraint).

    Returns (is_valid, message).
    """
    ticker = ticker.upper().strip()

    # Basic format check
    if not ticker or len(ticker) > 5:
        return False, f"Invalid ticker format: {ticker}"

    # Check for non-US patterns
    if "." in ticker:
        # ADRs and foreign tickers often have dots
        return False, f"Non-US ticker format: {ticker} (v1 supports US equities only)"

    if ticker.startswith("^"):
        # Index, not a stock
        return False, f"'{ticker}' is an index, not a stock"

    return True, f"{ticker} validated"


def validate_peer_set(
    primary: str,
    peers: list[str],
) -> tuple[bool, list[str]]:
    """
    Validate a peer set.

    Returns (is_valid, list of issues).
    """
    issues = []

    if len(peers) < 2:
        issues.append("Minimum 2 peers required for cross-sectional analysis")

    if len(peers) > 6:
        issues.append("Maximum 6 peers allowed (v1 constraint)")

    if primary.upper() in [p.upper() for p in peers]:
        issues.append(f"Primary ticker {primary} cannot be in peer list")

    # Check for duplicates
    upper_peers = [p.upper() for p in peers]
    if len(upper_peers) != len(set(upper_peers)):
        issues.append("Duplicate tickers in peer list")

    # Validate each peer
    for peer in peers:
        is_valid, msg = validate_us_ticker(peer)
        if not is_valid:
            issues.append(msg)

    return len(issues) == 0, issues
