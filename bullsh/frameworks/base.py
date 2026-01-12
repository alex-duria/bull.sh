"""Framework base class and loader."""

import tomllib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from bullsh.config import get_config


class FrameworkType(Enum):
    """Type of analysis framework."""

    QUANTITATIVE = "quantitative"  # Score-based (Piotroski)
    QUALITATIVE = "qualitative"  # Analysis-based (Porter)
    OUTPUT = "output"  # Output format (Pitch)


@dataclass
class Criterion:
    """A single criterion in a framework."""

    id: str
    name: str
    question: str
    source: str  # Which tool to use (sec, yahoo, etc.)
    scoring: str | None = None  # How to score (binary, scale, etc.)

    # Runtime state
    checked: bool = False
    result: Any = None
    user_override: str | None = None
    user_note: str | None = None


@dataclass
class Framework:
    """An analysis framework."""

    name: str
    display_name: str
    description: str
    framework_type: FrameworkType
    criteria: list[Criterion] = field(default_factory=list)

    # Scoring config
    scoring_enabled: bool = False
    pass_threshold: int | None = None

    # Metadata
    author: str = "builtin"
    version: str = "1.0"

    def get_progress(self) -> tuple[int, int]:
        """Return (completed, total) criteria count."""
        completed = sum(1 for c in self.criteria if c.checked)
        return completed, len(self.criteria)

    def get_progress_bar(self) -> str:
        """Return visual progress bar."""
        completed, total = self.get_progress()
        if total == 0:
            return "░░░░░"

        filled = int((completed / total) * 5)
        return "█" * filled + "░" * (5 - filled)

    def get_unchecked_criteria(self) -> list[Criterion]:
        """Return criteria not yet checked."""
        return [c for c in self.criteria if not c.checked]

    def mark_checked(self, criterion_id: str, result: Any) -> None:
        """Mark a criterion as checked with result."""
        for c in self.criteria:
            if c.id == criterion_id:
                c.checked = True
                c.result = result
                break

    def set_user_override(self, criterion_id: str, override: str, note: str | None = None) -> None:
        """Allow user to override a criterion assessment."""
        for c in self.criteria:
            if c.id == criterion_id:
                c.user_override = override
                c.user_note = note
                break

    def calculate_score(self) -> int | None:
        """Calculate total score if scoring is enabled."""
        if not self.scoring_enabled:
            return None

        score = 0
        for c in self.criteria:
            if c.checked and c.result:
                # Binary scoring: 1 if passed, 0 if not
                if isinstance(c.result, bool):
                    score += 1 if c.result else 0
                elif isinstance(c.result, (int, float)):
                    score += int(c.result)

        return score

    def to_checklist_display(self) -> str:
        """Format as checklist for display."""
        lines = [
            f"**{self.display_name}** [{self.get_progress_bar()}] {self.get_progress()[0]}/{self.get_progress()[1]}\n"
        ]

        for c in self.criteria:
            if c.checked:
                mark = "✓" if c.result else "✗"
                override = f" (User: {c.user_override})" if c.user_override else ""
                lines.append(f"  [{mark}] {c.name}{override}")
            else:
                lines.append(f"  [ ] {c.name}")

        if self.scoring_enabled:
            score = self.calculate_score()
            if score is not None:
                lines.append(f"\n  **Score: {score}/{len(self.criteria)}**")

        return "\n".join(lines)


# Built-in framework definitions
PIOTROSKI_FRAMEWORK = Framework(
    name="piotroski",
    display_name="Piotroski F-Score",
    description="9-point financial health scoring system",
    framework_type=FrameworkType.QUANTITATIVE,
    scoring_enabled=True,
    pass_threshold=7,
    author="builtin",
    criteria=[
        # Profitability (4 points)
        Criterion(
            id="roa_positive",
            name="Positive ROA",
            question="Is Return on Assets > 0?",
            source="sec",
            scoring="binary",
        ),
        Criterion(
            id="cfo_positive",
            name="Positive Operating Cash Flow",
            question="Is Operating Cash Flow > 0?",
            source="sec",
            scoring="binary",
        ),
        Criterion(
            id="roa_increasing",
            name="ROA Increasing",
            question="Is ROA higher than prior year?",
            source="sec",
            scoring="binary",
        ),
        Criterion(
            id="cfo_gt_ni",
            name="Cash Flow > Net Income",
            question="Is Operating Cash Flow > Net Income (quality of earnings)?",
            source="sec",
            scoring="binary",
        ),
        # Leverage & Liquidity (3 points)
        Criterion(
            id="debt_decreasing",
            name="Long-term Debt Decreasing",
            question="Is long-term debt lower than prior year?",
            source="sec",
            scoring="binary",
        ),
        Criterion(
            id="current_ratio_increasing",
            name="Current Ratio Increasing",
            question="Is current ratio higher than prior year?",
            source="sec",
            scoring="binary",
        ),
        Criterion(
            id="no_dilution",
            name="No Share Dilution",
            question="Were no new shares issued (shares outstanding ≤ prior year)?",
            source="sec",
            scoring="binary",
        ),
        # Efficiency (2 points)
        Criterion(
            id="gross_margin_increasing",
            name="Gross Margin Increasing",
            question="Is gross margin higher than prior year?",
            source="sec",
            scoring="binary",
        ),
        Criterion(
            id="asset_turnover_increasing",
            name="Asset Turnover Increasing",
            question="Is asset turnover (revenue/assets) higher than prior year?",
            source="sec",
            scoring="binary",
        ),
    ],
)


PORTER_FRAMEWORK = Framework(
    name="porter",
    display_name="Porter's Five Forces",
    description="Competitive positioning and moat analysis",
    framework_type=FrameworkType.QUALITATIVE,
    scoring_enabled=False,
    author="builtin",
    criteria=[
        Criterion(
            id="threat_new_entrants",
            name="Threat of New Entrants",
            question="How difficult is it for new competitors to enter this market?",
            source="sec",
            scoring="scale",  # LOW/MODERATE/HIGH
        ),
        Criterion(
            id="supplier_power",
            name="Supplier Power",
            question="How much bargaining power do suppliers have?",
            source="sec",
            scoring="scale",
        ),
        Criterion(
            id="buyer_power",
            name="Buyer Power",
            question="How much bargaining power do buyers/customers have?",
            source="sec",
            scoring="scale",
        ),
        Criterion(
            id="threat_substitutes",
            name="Threat of Substitutes",
            question="How easily can customers switch to alternative products/services?",
            source="sec",
            scoring="scale",
        ),
        Criterion(
            id="competitive_rivalry",
            name="Competitive Rivalry",
            question="How intense is competition in this industry?",
            source="sec",
            scoring="scale",
        ),
    ],
)


PITCH_FRAMEWORK = Framework(
    name="pitch",
    display_name="Hedge Fund Stock Pitch",
    description="Professional thesis output format",
    framework_type=FrameworkType.OUTPUT,
    scoring_enabled=False,
    author="builtin",
    criteria=[
        Criterion(
            id="thesis",
            name="Investment Thesis",
            question="What is the core investment thesis (1-2 sentences)?",
            source="synthesis",
        ),
        Criterion(
            id="catalysts",
            name="Key Catalysts",
            question="What events will unlock value?",
            source="synthesis",
        ),
        Criterion(
            id="valuation",
            name="Valuation",
            question="Why is the stock mispriced?",
            source="yahoo",
        ),
        Criterion(
            id="risks",
            name="Risk Factors",
            question="What could invalidate the thesis?",
            source="sec",
        ),
    ],
)


# Import valuation framework
from bullsh.frameworks.valuation import VALUATION_FRAMEWORK

# Multi-factor analysis framework (interactive session, handled separately in repl.py)
FACTORS_FRAMEWORK = Framework(
    name="factors",
    display_name="Multi-Factor Analysis",
    description="Cross-sectional factor scoring with Fama-French regression",
    framework_type=FrameworkType.QUANTITATIVE,
    scoring_enabled=False,  # Uses its own scoring system
    author="builtin",
    criteria=[],  # Interactive session, no predefined criteria
)

# Registry of built-in frameworks
BUILTIN_FRAMEWORKS: dict[str, Framework] = {
    "piotroski": PIOTROSKI_FRAMEWORK,
    "porter": PORTER_FRAMEWORK,
    "pitch": PITCH_FRAMEWORK,
    "valuation": VALUATION_FRAMEWORK,
    "factors": FACTORS_FRAMEWORK,
}


def load_framework(name: str) -> Framework:
    """
    Load a framework by name.

    Args:
        name: Framework name ("piotroski", "porter", "custom:myframework")

    Returns:
        Framework instance (copy to allow runtime state)
    """
    import copy

    # Check for custom framework
    if name.startswith("custom:"):
        custom_name = name[7:]
        return _load_custom_framework(custom_name)

    # Check built-in frameworks
    if name in BUILTIN_FRAMEWORKS:
        # Return a deep copy so each session has its own state
        return copy.deepcopy(BUILTIN_FRAMEWORKS[name])

    raise ValueError(f"Unknown framework: {name}")


def _load_custom_framework(name: str) -> Framework:
    """Load a custom framework from TOML file."""
    config = get_config()
    framework_path = config.custom_frameworks_dir / f"{name}.toml"

    if not framework_path.exists():
        raise ValueError(f"Custom framework not found: {name}")

    with open(framework_path, "rb") as f:
        data = tomllib.load(f)

    # Parse TOML into Framework
    meta = data.get("meta", {})
    criteria_data = data.get("criteria", {}).get("items", [])
    scoring = data.get("scoring", {})

    criteria = [
        Criterion(
            id=c.get("id", f"criterion_{i}"),
            name=c.get("name", c.get("question", "")[:30]),
            question=c.get("question", ""),
            source=c.get("source", "sec"),
            scoring=c.get("scoring"),
        )
        for i, c in enumerate(criteria_data)
    ]

    return Framework(
        name=f"custom:{name}",
        display_name=meta.get("name", name),
        description=meta.get("description", "Custom framework"),
        framework_type=FrameworkType.QUANTITATIVE
        if scoring.get("enabled")
        else FrameworkType.QUALITATIVE,
        criteria=criteria,
        scoring_enabled=scoring.get("enabled", False),
        pass_threshold=scoring.get("pass_threshold"),
        author=meta.get("author", "user"),
    )


def list_frameworks() -> list[dict[str, str]]:
    """List all available frameworks."""
    frameworks = []

    # Built-in
    for name, fw in BUILTIN_FRAMEWORKS.items():
        frameworks.append(
            {
                "name": name,
                "display_name": fw.display_name,
                "description": fw.description,
                "type": fw.framework_type.value,
                "builtin": True,
            }
        )

    # Custom
    config = get_config()
    if config.custom_frameworks_dir.exists():
        for path in config.custom_frameworks_dir.glob("*.toml"):
            try:
                fw = _load_custom_framework(path.stem)
                frameworks.append(
                    {
                        "name": f"custom:{path.stem}",
                        "display_name": fw.display_name,
                        "description": fw.description,
                        "type": fw.framework_type.value,
                        "builtin": False,
                    }
                )
            except Exception:
                pass  # Skip invalid frameworks

    return frameworks
