"""Terminal theme and styling for bullsh."""

from rich.theme import Theme

# Color palette - finance terminal aesthetic (dark mode)
COLORS = {
    # Primary colors
    "bull": "#00ff88",  # Bright green - positive/bullish
    "bear": "#ff4466",  # Red - negative/bearish
    "neutral": "#ffaa00",  # Amber/gold - neutral/caution
    # UI colors
    "primary": "#00ff88",  # Main accent (green)
    "secondary": "#00d4ff",  # Cyan - secondary accent
    "accent": "#aa88ff",  # Purple - highlights
    # Text colors
    "text": "#e0e0e0",  # Main text
    "text_muted": "#666666",  # Dimmed text
    "text_bright": "#ffffff",  # Bright white
    # Backgrounds
    "bg": "#0a0a0a",  # Near black
    "bg_panel": "#1a1a1a",  # Panel background
    "bg_highlight": "#2a2a2a",  # Highlighted row
    # Semantic
    "success": "#00ff88",
    "error": "#ff4466",
    "warning": "#ffaa00",
    "info": "#00d4ff",
    # Data source colors
    "sec": "#00d4ff",  # SEC filings
    "yahoo": "#aa88ff",  # Yahoo Finance
    "social": "#ff88aa",  # Social media
    "news": "#ffaa00",  # News
}


# Rich theme for console
RICH_THEME = Theme(
    {
        # Base styles
        "info": f"{COLORS['info']}",
        "warning": f"{COLORS['warning']}",
        "error": f"bold {COLORS['error']}",
        "success": f"{COLORS['success']}",
        # Custom styles
        "bull": f"bold {COLORS['bull']}",
        "bear": f"bold {COLORS['bear']}",
        "ticker": f"bold {COLORS['primary']}",
        "price": f"{COLORS['text_bright']}",
        "percent_up": f"bold {COLORS['bull']}",
        "percent_down": f"bold {COLORS['bear']}",
        # UI elements
        "header": f"bold {COLORS['primary']}",
        "subheader": f"italic {COLORS['secondary']}",
        "muted": f"{COLORS['text_muted']}",
        "highlight": f"bold {COLORS['accent']}",
        # Tool status
        "tool.name": f"bold {COLORS['secondary']}",
        "tool.success": f"{COLORS['success']}",
        "tool.fail": f"{COLORS['error']}",
        "tool.cached": f"italic {COLORS['text_muted']}",
        # Framework
        "framework.name": f"bold {COLORS['accent']}",
        "framework.score": f"bold {COLORS['primary']}",
        # Sources
        "source.sec": f"{COLORS['sec']}",
        "source.yahoo": f"{COLORS['yahoo']}",
        "source.social": f"{COLORS['social']}",
        "source.news": f"{COLORS['news']}",
    }
)


# prompt_toolkit style for input
PROMPT_STYLE = {
    "prompt": f"bold {COLORS['primary']}",
    "completion-menu": f"bg:{COLORS['bg_panel']} {COLORS['text']}",
    "completion-menu.completion": f"bg:{COLORS['bg_panel']} {COLORS['text']}",
    "completion-menu.completion.current": f"bg:{COLORS['bg_highlight']} {COLORS['text_bright']} bold",
    "completion-menu.meta": f"bg:{COLORS['bg_panel']} {COLORS['text_muted']} italic",
    "completion-menu.meta.current": f"bg:{COLORS['bg_highlight']} {COLORS['secondary']} italic",
}


def format_percent(value: float, with_sign: bool = True) -> str:
    """Format a percentage with color coding."""
    if value > 0:
        sign = "+" if with_sign else ""
        return f"[percent_up]{sign}{value:.2f}%[/percent_up]"
    elif value < 0:
        return f"[percent_down]{value:.2f}%[/percent_down]"
    else:
        return f"[muted]{value:.2f}%[/muted]"


def format_price(value: float, currency: str = "$") -> str:
    """Format a price value."""
    return f"[price]{currency}{value:,.2f}[/price]"


def format_ticker(symbol: str) -> str:
    """Format a ticker symbol."""
    return f"[ticker]{symbol.upper()}[/ticker]"


def format_change(value: float, label: str = "") -> str:
    """Format a change value with arrow."""
    if value > 0:
        arrow = "▲"
        style = "bull"
    elif value < 0:
        arrow = "▼"
        style = "bear"
    else:
        arrow = "─"
        style = "muted"

    prefix = f"{label} " if label else ""
    return f"[{style}]{prefix}{arrow} {abs(value):.2f}%[/{style}]"


def format_score(score: int, max_score: int, threshold: int | None = None) -> str:
    """Format a framework score."""
    if threshold:
        if score >= threshold:
            style = "bull"
        elif score >= threshold - 2:
            style = "warning"
        else:
            style = "bear"
    else:
        style = "framework.score"

    return f"[{style}]{score}/{max_score}[/{style}]"


# Unicode box-drawing characters for tables/panels
BOX = {
    "tl": "╭",  # top-left
    "tr": "╮",  # top-right
    "bl": "╰",  # bottom-left
    "br": "╯",  # bottom-right
    "h": "─",  # horizontal
    "v": "│",  # vertical
    "cross": "┼",
    "t_down": "┬",
    "t_up": "┴",
    "t_right": "├",
    "t_left": "┤",
}


# Progress bar characters
PROGRESS = {
    "filled": "█",
    "empty": "░",
    "partial": ["▏", "▎", "▍", "▌", "▋", "▊", "▉"],
}


def progress_bar(value: float, width: int = 10, filled_style: str = "bull") -> str:
    """Create a colored progress bar."""
    filled = int(value * width)
    empty = width - filled

    bar = f"[{filled_style}]{PROGRESS['filled'] * filled}[/{filled_style}]"
    bar += f"[muted]{PROGRESS['empty'] * empty}[/muted]"
    return bar
