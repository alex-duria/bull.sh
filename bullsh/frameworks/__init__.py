"""Frameworks module - analysis frameworks and custom framework support."""

from bullsh.frameworks.base import (
    BUILTIN_FRAMEWORKS,
    Criterion,
    Framework,
    FrameworkType,
    list_frameworks,
    load_framework,
)
from bullsh.frameworks.piotroski import (
    FinancialData,
    FScoreResult,
    compute_fscore,
    extract_financial_data_from_filing,
)
from bullsh.frameworks.porter import (
    FivesForcesResult,
    ForceAnalysis,
    ForceStrength,
    analyze_five_forces,
)

__all__ = [
    # Base
    "Criterion",
    "Framework",
    "FrameworkType",
    "BUILTIN_FRAMEWORKS",
    "list_frameworks",
    "load_framework",
    # Piotroski
    "FinancialData",
    "FScoreResult",
    "compute_fscore",
    "extract_financial_data_from_filing",
    # Porter
    "ForceAnalysis",
    "ForceStrength",
    "FivesForcesResult",
    "analyze_five_forces",
]
