"""Thesis export tools."""

from datetime import datetime
from pathlib import Path
from typing import Any

import tomli_w

from bullsh.config import get_config
from bullsh.tools.base import ToolResult, ToolStatus


async def save_thesis(
    ticker: str,
    content: str,
    filename: str | None = None,
) -> ToolResult:
    """
    Save thesis to a markdown file with YAML frontmatter.

    Args:
        ticker: Stock ticker symbol
        content: Markdown content of the thesis
        filename: Optional custom filename

    Returns:
        ToolResult with path to saved file
    """
    config = get_config()
    theses_dir = config.theses_dir
    theses_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename if not provided
    if not filename:
        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = f"{ticker.upper()}_thesis_{date_str}.md"

    # Ensure .md extension
    if not filename.endswith(".md"):
        filename += ".md"

    filepath = theses_dir / filename

    # Build frontmatter
    frontmatter = {
        "ticker": ticker.upper(),
        "generated": datetime.now().isoformat(),
        "model": config.model,
    }

    # Format as YAML-like frontmatter
    frontmatter_lines = ["---"]
    frontmatter_lines.append(f"ticker: {frontmatter['ticker']}")
    frontmatter_lines.append(f"generated: {frontmatter['generated']}")
    frontmatter_lines.append(f"model: {frontmatter['model']}")
    frontmatter_lines.append("---")

    full_content = "\n".join(frontmatter_lines) + "\n\n" + content

    try:
        filepath.write_text(full_content, encoding="utf-8")

        return ToolResult(
            data={
                "filepath": str(filepath),
                "filename": filename,
                "size_bytes": len(full_content.encode("utf-8")),
            },
            confidence=1.0,
            status=ToolStatus.SUCCESS,
            tool_name="save_thesis",
            ticker=ticker.upper(),
        )

    except Exception as e:
        return ToolResult(
            data={},
            confidence=0.0,
            status=ToolStatus.FAILED,
            tool_name="save_thesis",
            ticker=ticker,
            error_message=str(e),
        )


def format_thesis_hedge_fund_pitch(
    ticker: str,
    thesis_summary: str,
    catalysts: list[str],
    valuation_notes: str,
    risk_factors: list[str],
    key_metrics: dict[str, Any] | None = None,
) -> str:
    """
    Format research into Hedge Fund Stock Pitch structure.

    Args:
        ticker: Stock ticker
        thesis_summary: 1-2 sentence investment thesis
        catalysts: List of key catalysts
        valuation_notes: Valuation analysis
        risk_factors: List of key risks

    Returns:
        Formatted markdown content
    """
    lines = [f"# {ticker.upper()} Investment Thesis", ""]

    # Key metrics table if provided
    if key_metrics:
        lines.append("## Key Metrics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for metric, value in key_metrics.items():
            lines.append(f"| {metric} | {value} |")
        lines.append("")

    # Investment thesis
    lines.append("## Investment Thesis")
    lines.append("")
    lines.append(thesis_summary)
    lines.append("")

    # Catalysts
    lines.append("## Key Catalysts")
    lines.append("")
    for catalyst in catalysts:
        lines.append(f"- {catalyst}")
    lines.append("")

    # Valuation
    lines.append("## Valuation")
    lines.append("")
    lines.append(valuation_notes)
    lines.append("")

    # Risk factors
    lines.append("## Risk Factors")
    lines.append("")
    for risk in risk_factors:
        lines.append(f"- {risk}")
    lines.append("")

    return "\n".join(lines)
