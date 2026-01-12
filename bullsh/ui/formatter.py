"""Rich output formatter for agent responses - clean, readable terminal output."""

import re

from rich.console import Console
from rich.text import Text

from bullsh.ui.theme import COLORS

console = Console()


def format_agent_response(content: str) -> Text:
    """
    Format agent response for beautiful terminal display.

    Transforms markdown into a clean, readable format with:
    - Underlined section headers
    - Numbered/bulleted lists with proper indentation
    - Source citations styled
    - Good whitespace and wrapping
    """
    # Clean up any tool status remnants
    content = re.sub(r"^[◐◑◒◓✓✗].*$", "", content, flags=re.MULTILINE)
    content = re.sub(r"^\[Data gathered.*\]$", "", content, flags=re.MULTILINE)

    lines = content.split("\n")
    output_parts = []
    current_list_num = 0
    in_list = False
    last_was_empty = False

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip empty lines but preserve single spacing
        if not stripped:
            if not last_was_empty and output_parts:
                output_parts.append(Text(""))
                last_was_empty = True
            i += 1
            continue

        last_was_empty = False

        # H1: # Header -> Big underlined header
        if stripped.startswith("# ") and not stripped.startswith("##"):
            header_text = stripped[2:].strip()
            output_parts.append(_render_h1(header_text))
            current_list_num = 0
            in_list = False
            i += 1
            continue

        # H2: ## Header -> Underlined section header
        if stripped.startswith("## "):
            header_text = stripped[3:].strip()
            output_parts.append(_render_h2(header_text))
            current_list_num = 0
            in_list = False
            i += 1
            continue

        # H3: ### Header -> Bold with subtle underline
        if stripped.startswith("### "):
            header_text = stripped[4:].strip()
            output_parts.append(_render_h3(header_text))
            current_list_num = 0
            in_list = False
            i += 1
            continue

        # H4+: #### Header -> Just bold
        if stripped.startswith("####"):
            header_text = stripped.lstrip("#").strip()
            text = Text()
            text.append(f"\n  {header_text}", style="bold")
            output_parts.append(text)
            i += 1
            continue

        # Bold line that acts as a header: **Something**
        bold_header_match = re.match(r"^\*\*([^*]+)\*\*:?\s*$", stripped)
        if bold_header_match:
            header_text = bold_header_match.group(1)
            output_parts.append(_render_h3(header_text))
            current_list_num = 0
            in_list = False
            i += 1
            continue

        # Numbered list: 1. Item or 1) Item
        num_match = re.match(r"^(\d+)[.\)]\s+(.+)$", stripped)
        if num_match:
            num = int(num_match.group(1))
            item_text = num_match.group(2)

            # Collect continuation lines (indented text that's part of same item)
            j = i + 1
            while (
                j < len(lines)
                and lines[j].startswith("   ")
                and not re.match(r"^\s*\d+[.\)]\s", lines[j])
            ):
                item_text += " " + lines[j].strip()
                j += 1

            output_parts.append(_render_list_item(num, item_text, numbered=True))
            in_list = True
            i = j
            continue

        # Bullet list: - Item or * Item or • Item
        bullet_match = re.match(r"^[-*•]\s+(.+)$", stripped)
        if bullet_match:
            item_text = bullet_match.group(1)

            # Collect continuation lines
            j = i + 1
            while (
                j < len(lines)
                and lines[j].startswith("   ")
                and not re.match(r"^\s*[-*•]\s", lines[j])
            ):
                item_text += " " + lines[j].strip()
                j += 1

            current_list_num += 1
            output_parts.append(_render_list_item(current_list_num, item_text, numbered=False))
            in_list = True
            i = j
            continue

        # Table detection (markdown tables)
        if "|" in stripped and stripped.startswith("|"):
            # Collect all table lines
            table_lines = [stripped]
            j = i + 1
            while j < len(lines) and "|" in lines[j]:
                table_lines.append(lines[j].strip())
                j += 1

            output_parts.append(_render_table(table_lines))
            i = j
            continue

        # Horizontal rule: --- or ***
        if stripped in ("---", "***", "___") or re.match(r"^[-*_]{3,}$", stripped):
            rule = Text()
            rule.append("  " + "─" * 50, style=COLORS["muted"])
            output_parts.append(rule)
            i += 1
            continue

        # Regular paragraph - reset list counter
        if in_list:
            current_list_num = 0
            in_list = False

        # Process inline formatting and render paragraph
        output_parts.append(_render_paragraph(stripped))
        i += 1

    # Combine all parts
    final = Text()
    for part in output_parts:
        if isinstance(part, Text):
            final.append(part)
            final.append("\n")

    return final


def _render_h1(text: str) -> Text:
    """Render H1 header - big and prominent."""
    output = Text()
    output.append("\n")
    output.append(f"  {text}", style=f"bold {COLORS['primary']}")
    output.append("\n")
    output.append("  " + "═" * min(len(text) + 4, 60), style=COLORS["primary"])
    output.append("\n")
    return output


def _render_h2(text: str) -> Text:
    """Render H2 header - underlined section header."""
    output = Text()
    output.append("\n")
    # Clean up any markdown bold markers
    clean_text = text.replace("**", "")
    output.append(f"  {clean_text}", style="underline bold")
    output.append("\n")
    return output


def _render_h3(text: str) -> Text:
    """Render H3 header - subtle section."""
    output = Text()
    output.append("\n")
    clean_text = text.replace("**", "")
    output.append(f"  {clean_text}", style=f"bold {COLORS['secondary']}")
    output.append("\n")
    return output


def _render_list_item(num: int, text: str, numbered: bool = True) -> Text:
    """Render a list item with proper formatting."""
    output = Text()

    # Format the item text with inline styles
    formatted_text = _process_inline_formatting(text)

    if numbered:
        output.append(f"  {num} ", style=f"bold {COLORS['accent']}")
    else:
        output.append("  • ", style=COLORS["muted"])

    # Word wrap the text
    output.append(formatted_text)

    return output


def _render_paragraph(text: str) -> Text:
    """Render a regular paragraph with inline formatting."""
    output = Text()
    output.append("  ")
    output.append(_process_inline_formatting(text))
    return output


def _process_inline_formatting(text: str) -> Text:
    """Process inline markdown formatting: **bold**, *italic*, `code`, (source)."""
    output = Text()

    # Pattern to match various inline elements
    # We'll process character by character with state tracking
    i = 0
    while i < len(text):
        # Check for **bold**
        if text[i : i + 2] == "**":
            end = text.find("**", i + 2)
            if end != -1:
                output.append(text[i + 2 : end], style="bold")
                i = end + 2
                continue

        # Check for *italic* (but not **)
        if text[i] == "*" and (i + 1 >= len(text) or text[i + 1] != "*"):
            end = text.find("*", i + 1)
            if end != -1 and text[end - 1 : end + 1] != "**":
                output.append(text[i + 1 : end], style="italic")
                i = end + 1
                continue

        # Check for `code`
        if text[i] == "`":
            end = text.find("`", i + 1)
            if end != -1:
                output.append(text[i + 1 : end], style=f"bold {COLORS['secondary']}")
                i = end + 1
                continue

        # Check for source citations (SEC filing), (Yahoo Finance), etc.
        if text[i] == "(":
            end = text.find(")", i + 1)
            if end != -1:
                citation = text[i + 1 : end]
                # Check if it looks like a source citation
                source_keywords = [
                    "SEC",
                    "filing",
                    "Yahoo",
                    "source",
                    "analyst",
                    "10-K",
                    "10-Q",
                    "report",
                ]
                if any(kw.lower() in citation.lower() for kw in source_keywords):
                    output.append("(", style=COLORS["muted"])
                    output.append(citation, style=f"italic {COLORS['secondary']} underline")
                    output.append(")", style=COLORS["muted"])
                    i = end + 1
                    continue

        # Check for percentages and highlight them
        pct_match = re.match(r"([+-]?\d+\.?\d*%)", text[i:])
        if pct_match:
            pct = pct_match.group(1)
            if pct.startswith("+") or (pct[0].isdigit() and not pct.startswith("-")):
                output.append(pct, style=f"bold {COLORS['bull']}")
            else:
                output.append(pct, style=f"bold {COLORS['bear']}")
            i += len(pct)
            continue

        # Check for dollar amounts
        dollar_match = re.match(r"(\$[\d,]+\.?\d*[BMK]?)", text[i:])
        if dollar_match:
            amount = dollar_match.group(1)
            output.append(amount, style="bold white")
            i += len(amount)
            continue

        # Regular character
        output.append(text[i])
        i += 1

    return output


def _render_table(lines: list[str]) -> Text:
    """Render a markdown table."""
    output = Text()
    output.append("\n")

    # Parse table
    rows = []
    for line in lines:
        # Skip separator lines (|---|---|)
        if re.match(r"^\|[\s\-:]+\|$", line.replace("|", "| ").replace("  ", " ")):
            continue

        cells = [c.strip() for c in line.split("|")[1:-1]]  # Remove empty first/last from split
        if cells:
            rows.append(cells)

    if not rows:
        return output

    # Calculate column widths
    num_cols = max(len(row) for row in rows)
    col_widths = [0] * num_cols
    for row in rows:
        for i, cell in enumerate(row):
            if i < num_cols:
                col_widths[i] = max(col_widths[i], len(cell))

    # Render header
    if rows:
        header = rows[0]
        header_line = "  "
        for i, cell in enumerate(header):
            if i < num_cols:
                header_line += cell.ljust(col_widths[i] + 2)
        output.append(header_line, style="bold underline")
        output.append("\n")

        # Render data rows
        for row in rows[1:]:
            for i, cell in enumerate(row):
                if i < num_cols:
                    # Apply formatting to cell content
                    formatted_cell = str(cell).ljust(col_widths[i] + 2)

                    # Color code percentages and values
                    if "+" in cell and "%" in cell:
                        output.append(formatted_cell, style=COLORS["bull"])
                    elif "-" in cell and "%" in cell:
                        output.append(formatted_cell, style=COLORS["bear"])
                    elif cell.lower() in ("high", "strong", "buy"):
                        output.append(formatted_cell, style=COLORS["bull"])
                    elif cell.lower() in ("low", "weak", "sell"):
                        output.append(formatted_cell, style=COLORS["bear"])
                    else:
                        output.append(formatted_cell)
            output.append("\n")

    return output


def format_stock_summary(data: dict) -> Text:
    """Format stock data as a beautiful summary card."""
    output = Text()

    ticker = data.get("ticker", "???")
    name = data.get("name", data.get("company_name", ""))
    price = data.get("price", 0)
    change = data.get("change_percent", 0)

    # Header
    output.append("\n")
    output.append(f"  {ticker}", style=f"bold {COLORS['primary']}")
    if name:
        output.append(f"  {name}", style="dim")
    output.append("\n")

    # Price line
    output.append(f"  ${price:,.2f}", style="bold white")
    if change >= 0:
        output.append(f"  ▲ +{change:.2f}%", style=f"bold {COLORS['bull']}")
    else:
        output.append(f"  ▼ {change:.2f}%", style=f"bold {COLORS['bear']}")
    output.append("\n")

    output.append("  " + "─" * 40, style=COLORS["muted"])
    output.append("\n")

    # Key metrics in two columns
    metrics = [
        ("P/E", data.get("pe_ratio")),
        ("Market Cap", data.get("market_cap")),
        ("52W High", data.get("52w_high")),
        ("52W Low", data.get("52w_low")),
        ("Volume", data.get("volume")),
        ("Avg Volume", data.get("avg_volume")),
    ]

    for label, value in metrics:
        if value is not None:
            output.append(f"  {label}: ", style="dim")
            if isinstance(value, (int, float)):
                if value >= 1_000_000_000:
                    output.append(f"${value / 1e9:.1f}B", style="white")
                elif value >= 1_000_000:
                    output.append(f"${value / 1e6:.1f}M", style="white")
                elif label in ("P/E",):
                    output.append(f"{value:.1f}", style="white")
                elif label in ("52W High", "52W Low"):
                    output.append(f"${value:.2f}", style="white")
                else:
                    output.append(f"{value:,}", style="white")
            else:
                output.append(str(value), style="white")
            output.append("\n")

    return output


def format_analysis_section(title: str, items: list[str], source: str = "SEC filing") -> Text:
    """Format an analysis section like Top Strengths or Top Risks."""
    output = Text()

    # Section header with underline
    output.append("\n")
    output.append(f"  {title}", style="underline bold")
    output.append("\n\n")

    # Numbered items
    for i, item in enumerate(items, 1):
        output.append(f"  {i} ", style=f"bold {COLORS['accent']}")
        output.append(_process_inline_formatting(item))
        output.append(" (", style=COLORS["muted"])
        output.append(source, style=f"italic {COLORS['secondary']} underline")
        output.append(")", style=COLORS["muted"])
        output.append("\n")

    return output
