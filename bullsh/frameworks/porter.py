"""Porter's Five Forces analysis logic."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ForceStrength(Enum):
    """Strength rating for each force."""

    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    UNKNOWN = "UNKNOWN"


@dataclass
class ForceAnalysis:
    """Analysis of a single competitive force."""

    force_id: str
    name: str
    strength: ForceStrength
    evidence: list[str] = field(default_factory=list)
    user_override: ForceStrength | None = None
    user_note: str | None = None

    @property
    def effective_strength(self) -> ForceStrength:
        """Return user override if set, otherwise computed strength."""
        return self.user_override if self.user_override else self.strength

    def to_display(self) -> str:
        """Format for display."""
        strength = self.effective_strength.value
        override_note = ""
        if self.user_override:
            override_note = f" (User assessment: was {self.strength.value})"

        lines = [f"**{self.name}: {strength}**{override_note}"]
        for ev in self.evidence[:3]:  # Limit evidence shown
            lines.append(f"  • {ev}")

        if self.user_note:
            lines.append(f"  📝 User note: {self.user_note}")

        return "\n".join(lines)


@dataclass
class FivesForcesResult:
    """Result of Five Forces analysis."""

    ticker: str
    forces: dict[str, ForceAnalysis] = field(default_factory=dict)

    @property
    def overall_attractiveness(self) -> str:
        """Assess overall industry attractiveness."""
        if not self.forces:
            return "Unknown"

        low_count = sum(
            1 for f in self.forces.values() if f.effective_strength == ForceStrength.LOW
        )
        high_count = sum(
            1 for f in self.forces.values() if f.effective_strength == ForceStrength.HIGH
        )

        if low_count >= 4:
            return "Highly Attractive (strong moat)"
        elif low_count >= 3:
            return "Attractive (good competitive position)"
        elif high_count >= 3:
            return "Unattractive (intense competition)"
        else:
            return "Mixed (selective advantages)"

    def to_summary(self) -> str:
        """Generate summary text."""
        lines = [f"## Porter's Five Forces Analysis: {self.ticker}\n"]
        lines.append(f"**Overall Industry Attractiveness**: {self.overall_attractiveness}\n")

        for force in self.forces.values():
            lines.append(force.to_display())
            lines.append("")

        return "\n".join(lines)


# Keywords for extracting evidence from 10-K
FORCE_KEYWORDS = {
    "threat_new_entrants": {
        "high": ["low barriers", "easy entry", "new competitors", "fragmented market"],
        "low": [
            "high barriers",
            "regulatory approval",
            "significant capital",
            "proprietary",
            "patents",
            "licenses required",
        ],
        "evidence_sections": ["business", "competition", "risk_factors"],
    },
    "supplier_power": {
        "high": [
            "sole supplier",
            "single source",
            "supplier concentration",
            "few suppliers",
            "dependent on",
        ],
        "low": ["multiple suppliers", "commodity", "readily available", "diverse supplier base"],
        "evidence_sections": ["business", "risk_factors", "supply"],
    },
    "buyer_power": {
        "high": [
            "customer concentration",
            "top customers",
            "major customers",
            "price sensitive",
            "commoditized",
        ],
        "low": ["diversified customer", "switching costs", "loyal customers", "differentiated"],
        "evidence_sections": ["business", "risk_factors", "customers"],
    },
    "threat_substitutes": {
        "high": ["alternative", "substitute products", "competing technologies", "disruption"],
        "low": ["no substitute", "unique", "switching costs", "network effects", "ecosystem"],
        "evidence_sections": ["business", "competition", "risk_factors"],
    },
    "competitive_rivalry": {
        "high": [
            "intense competition",
            "price competition",
            "many competitors",
            "low growth",
            "commoditized",
        ],
        "low": [
            "market leader",
            "limited competition",
            "differentiated",
            "growing market",
            "duopoly",
            "oligopoly",
        ],
        "evidence_sections": ["business", "competition"],
    },
}


def analyze_five_forces(
    ticker: str,
    filing_sections: dict[str, str],
) -> FivesForcesResult:
    """
    Analyze Porter's Five Forces from SEC filing text.

    Args:
        ticker: Stock ticker
        filing_sections: Dict of section name -> text content

    Returns:
        FivesForcesResult with analysis for each force
    """
    result = FivesForcesResult(ticker=ticker)

    force_names = {
        "threat_new_entrants": "Threat of New Entrants",
        "supplier_power": "Supplier Power",
        "buyer_power": "Buyer Power",
        "threat_substitutes": "Threat of Substitutes",
        "competitive_rivalry": "Competitive Rivalry",
    }

    for force_id, force_name in force_names.items():
        analysis = _analyze_single_force(
            force_id,
            force_name,
            filing_sections,
            FORCE_KEYWORDS.get(force_id, {}),
        )
        result.forces[force_id] = analysis

    return result


def _analyze_single_force(
    force_id: str,
    force_name: str,
    sections: dict[str, str],
    keywords: dict[str, Any],
) -> ForceAnalysis:
    """Analyze a single competitive force."""
    high_keywords = keywords.get("high", [])
    low_keywords = keywords.get("low", [])
    evidence_sections = keywords.get("evidence_sections", ["business", "risk_factors"])

    high_evidence = []
    low_evidence = []

    # Combine relevant sections
    combined_text = ""
    for section_name in evidence_sections:
        for key, text in sections.items():
            if section_name.lower() in key.lower():
                combined_text += " " + text

    combined_lower = combined_text.lower()

    # Find evidence for high force
    for kw in high_keywords:
        if kw.lower() in combined_lower:
            # Extract context around keyword
            idx = combined_lower.find(kw.lower())
            start = max(0, idx - 50)
            end = min(len(combined_text), idx + len(kw) + 100)
            context = combined_text[start:end].strip()
            # Clean up
            context = " ".join(context.split())
            if len(context) > 20:
                high_evidence.append(f'"{context}..."')

    # Find evidence for low force
    for kw in low_keywords:
        if kw.lower() in combined_lower:
            idx = combined_lower.find(kw.lower())
            start = max(0, idx - 50)
            end = min(len(combined_text), idx + len(kw) + 100)
            context = combined_text[start:end].strip()
            context = " ".join(context.split())
            if len(context) > 20:
                low_evidence.append(f'"{context}..."')

    # Determine strength based on evidence
    if len(high_evidence) > len(low_evidence) + 1:
        strength = ForceStrength.HIGH
        evidence = high_evidence[:3]
    elif len(low_evidence) > len(high_evidence) + 1:
        strength = ForceStrength.LOW
        evidence = low_evidence[:3]
    elif high_evidence or low_evidence:
        strength = ForceStrength.MODERATE
        evidence = (high_evidence + low_evidence)[:3]
    else:
        strength = ForceStrength.UNKNOWN
        evidence = ["No clear evidence found in filing"]

    return ForceAnalysis(
        force_id=force_id,
        name=force_name,
        strength=strength,
        evidence=evidence,
    )


def get_porter_system_prompt_addition() -> str:
    """Get additional system prompt for Porter's Five Forces analysis."""
    return """
When analyzing Porter's Five Forces, extract specific evidence from the 10-K for each force:

1. **Threat of New Entrants**: Look for barriers to entry
   - Capital requirements
   - Regulatory barriers
   - Proprietary technology/patents
   - Brand loyalty
   - Economies of scale

2. **Supplier Power**: Assess supplier dynamics
   - Number of suppliers
   - Switching costs
   - Supplier concentration
   - Importance of volume to supplier

3. **Buyer Power**: Evaluate customer leverage
   - Customer concentration (% of revenue from top customers)
   - Switching costs for customers
   - Price sensitivity
   - Availability of alternatives

4. **Threat of Substitutes**: Consider alternatives
   - Availability of substitute products
   - Price-performance of substitutes
   - Switching costs
   - Customer propensity to switch

5. **Competitive Rivalry**: Measure competition intensity
   - Number and size of competitors
   - Industry growth rate
   - Product differentiation
   - Exit barriers

Rate each force as LOW, MODERATE, or HIGH with specific evidence from the filing.
The user may challenge your assessments - accept their perspective and note the reasoning.
"""
