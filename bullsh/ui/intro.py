"""Animated intro sequence for bullsh terminal UI."""

import asyncio
import math
import random
from collections.abc import Callable

from rich.align import Align
from rich.console import Console
from rich.text import Text

console = Console()

# Color theme - finance terminal aesthetic
THEME = {
    "primary": "#00ff88",  # Bright green (bull)
    "secondary": "#00d4ff",  # Cyan
    "accent": "#ffaa00",  # Gold/amber
    "bearish": "#ff4444",  # Red (bear)
    "muted": "#666666",  # Gray
    "dim": "#333333",  # Dark gray
    "bg": "#0a0a0a",  # Near black
    "grid": "#1a1a1a",  # Grid lines
}

# ASCII Art Logo - pixel/block style
LOGO_LARGE = """
    ██████╗  ██╗   ██╗ ██╗      ██╗          ███████╗ ██╗  ██╗
    ██╔══██╗ ██║   ██║ ██║      ██║          ██╔════╝ ██║  ██║
    ██████╔╝ ██║   ██║ ██║      ██║          ███████╗ ███████║
    ██╔══██╗ ██║   ██║ ██║      ██║          ╚════██║ ██╔══██║
    ██████╔╝ ╚██████╔╝ ███████╗ ███████╗ ██╗ ███████║ ██║  ██║
    ╚═════╝   ╚═════╝  ╚══════╝ ╚══════╝ ╚═╝ ╚══════╝ ╚═╝  ╚═╝
"""

LOGO_SMALL = """
██████╗ ██╗   ██╗██╗     ██╗     ███████╗██╗  ██╗
██╔══██╗██║   ██║██║     ██║     ██╔════╝██║  ██║
██████╔╝██║   ██║██║     ██║     ███████╗███████║
██╔══██╗██║   ██║██║     ██║     ╚════██║██╔══██║
██████╔╝╚██████╔╝███████╗███████╗███████║██║  ██║
╚═════╝  ╚═════╝ ╚══════╝╚══════╝╚══════╝╚═╝  ╚═╝
"""

TAGLINE = "Investment Research Agent"
SUBTITLE = "SEC filings • Market data • AI analysis"
CREDIT = "Made with ❤  by Alexander Duria"


def generate_market_data(num_candles: int = 100, volatility: float = 0.025) -> dict:
    """Generate realistic market data for animation."""
    candles = []
    price = 150.0 + random.uniform(-30, 30)

    # Generate with some trending behavior
    trend = random.choice([-1, 1]) * random.uniform(0.001, 0.003)

    for i in range(num_candles):
        # Add some momentum and mean reversion
        if random.random() < 0.1:
            trend = -trend  # Occasional trend reversal

        change = price * (random.gauss(trend, volatility))
        open_price = price
        close_price = price + change

        # Realistic wicks
        wick_mult = random.uniform(0.3, 1.5)
        high = max(open_price, close_price) + abs(change) * wick_mult * random.uniform(0.5, 1.5)
        low = min(open_price, close_price) - abs(change) * wick_mult * random.uniform(0.5, 1.5)

        # Volume with some correlation to price movement
        base_volume = random.uniform(0.3, 1.0)
        volume = base_volume * (1 + abs(change) / price * 10)

        candles.append(
            {
                "open": open_price,
                "high": high,
                "low": low,
                "close": close_price,
                "volume": volume,
                "bullish": close_price >= open_price,
            }
        )
        price = close_price

    return {
        "candles": candles,
        "ticker": random.choice(["BULL.SH", "$BULL", "NVDA", "AAPL", "TSLA", "MSFT"]),
    }


def render_full_chart(
    data: dict,
    width: int,
    height: int,
    visible_candles: int,
    offset: int = 0,
    show_volume: bool = True,
    show_grid: bool = True,
    show_price_axis: bool = True,
    show_ticker: bool = True,
    glow_intensity: float = 0.0,
) -> Text:
    """
    Render a full-terminal candlestick chart with all the bells and whistles.
    """
    candles = data["candles"]
    ticker = data["ticker"]

    # Calculate visible range
    start_idx = max(0, min(offset, len(candles) - visible_candles))
    end_idx = min(start_idx + visible_candles, len(candles))
    visible = candles[start_idx:end_idx]

    if not visible:
        return Text("")

    # Reserve space for components
    price_axis_width = 10 if show_price_axis else 0
    volume_height = 4 if show_volume else 0
    chart_width = width - price_axis_width - 2
    chart_height = height - volume_height - 3  # Leave room for header

    # Calculate price range for visible candles
    all_highs = [c["high"] for c in visible]
    all_lows = [c["low"] for c in visible]
    price_max = max(all_highs) * 1.01
    price_min = min(all_lows) * 0.99
    price_range = price_max - price_min

    if price_range == 0:
        price_range = 1

    # Current price (last close)
    current_price = visible[-1]["close"]
    price_change = ((current_price / visible[0]["open"]) - 1) * 100

    # Initialize grid
    grid = [[" " for _ in range(width)] for _ in range(height)]
    colors = [[None for _ in range(width)] for _ in range(height)]

    # Helper to convert price to row
    def price_to_row(price: float) -> int:
        normalized = (price - price_min) / price_range
        return chart_height - 1 - int(normalized * (chart_height - 1))

    # Draw background grid
    if show_grid:
        for row in range(1, chart_height + 1):
            for col in range(chart_width):
                if row % 4 == 0:
                    grid[row][col] = "─"
                    colors[row][col] = THEME["dim"]
                elif col % 8 == 0:
                    grid[row][col] = "│"
                    colors[row][col] = THEME["dim"]

    # Calculate candle width and spacing
    candle_total_width = max(1, chart_width // len(visible))
    candle_body_width = max(1, candle_total_width - 1)

    # Draw candles
    for i, candle in enumerate(visible):
        col_start = i * candle_total_width
        col_center = col_start + candle_body_width // 2

        if col_center >= chart_width:
            break

        body_top = price_to_row(max(candle["open"], candle["close"])) + 1
        body_bottom = price_to_row(min(candle["open"], candle["close"])) + 1
        wick_top = price_to_row(candle["high"]) + 1
        wick_bottom = price_to_row(candle["low"]) + 1

        # Ensure at least 1 row for body
        if body_top == body_bottom:
            body_bottom = body_top + 1

        color = THEME["primary"] if candle["bullish"] else THEME["bearish"]

        # Draw wick (thin line)
        for row in range(max(1, wick_top), min(body_top, chart_height + 1)):
            if 0 <= col_center < chart_width:
                grid[row][col_center] = "│"
                colors[row][col_center] = color

        for row in range(max(1, body_bottom), min(wick_bottom + 1, chart_height + 1)):
            if 0 <= col_center < chart_width:
                grid[row][col_center] = "│"
                colors[row][col_center] = color

        # Draw body (thick block)
        body_char = "█" if candle["bullish"] else "▓"
        for row in range(max(1, body_top), min(body_bottom + 1, chart_height + 1)):
            for c in range(col_start, min(col_start + candle_body_width, chart_width)):
                if 0 <= c < chart_width:
                    grid[row][c] = body_char
                    colors[row][c] = color

    # Draw volume bars at bottom
    if show_volume:
        max_volume = max(c["volume"] for c in visible)
        vol_start_row = chart_height + 2

        for i, candle in enumerate(visible):
            col_start = i * candle_total_width
            vol_height = int((candle["volume"] / max_volume) * (volume_height - 1))
            color = THEME["primary"] if candle["bullish"] else THEME["bearish"]

            for v in range(vol_height):
                row = vol_start_row + (volume_height - 1 - v)
                for c in range(col_start, min(col_start + candle_body_width, chart_width)):
                    if 0 <= row < height and 0 <= c < chart_width:
                        grid[row][c] = "▄"
                        colors[row][c] = color

    # Draw price axis
    if show_price_axis:
        axis_col = chart_width + 1

        # Price levels
        for i in range(5):
            row = 1 + int((chart_height - 1) * i / 4)
            price_at_row = price_max - (price_range * i / 4)
            price_str = f"${price_at_row:,.0f}"

            if row < height:
                grid[row][axis_col] = "├"
                colors[row][axis_col] = THEME["muted"]
                for j, ch in enumerate(price_str):
                    if axis_col + 1 + j < width:
                        grid[row][axis_col + 1 + j] = ch
                        colors[row][axis_col + 1 + j] = THEME["muted"]

    # Build text output
    text = Text()

    # Header with ticker and price
    if show_ticker:
        header = Text()
        header.append(f"  {ticker}", style=f"bold {THEME['primary']}")
        header.append(f"  ${current_price:,.2f}", style="bold white")

        if price_change >= 0:
            header.append(f"  ▲ +{price_change:.2f}%", style=f"bold {THEME['primary']}")
        else:
            header.append(f"  ▼ {price_change:.2f}%", style=f"bold {THEME['bearish']}")

        # Add some spacing and decorative elements
        header.append("  " + "─" * (width - 35), style=THEME["dim"])
        header.append("  LIVE", style=f"bold {THEME['accent']} blink")

        text.append(header)
        text.append("\n")

    # Render grid
    for row_idx, row in enumerate(grid):
        for col_idx, char in enumerate(row):
            color = colors[row_idx][col_idx]
            if color:
                # Add glow effect to green candles
                if glow_intensity > 0 and color == THEME["primary"]:
                    text.append(char, style=f"bold {color}")
                else:
                    text.append(char, style=color)
            else:
                text.append(char)
        text.append("\n")

    return text


def render_logo(style: str = "large") -> Text:
    """Render the bullsh logo with colors."""
    logo = LOGO_LARGE if style == "large" else LOGO_SMALL
    text = Text()

    for char in logo:
        if char in "█▀▄╔╗╚╝║═":
            text.append(char, style=f"bold {THEME['primary']}")
        else:
            text.append(char)

    return text


def render_intro_frame(
    data: dict,
    frame: int,
    total_frames: int,
    width: int,
    height: int,
) -> Text:
    """Render a single frame of the epic intro animation."""
    output = Text()
    progress = frame / total_frames

    # Phase 1: Chart builds up (0-45%)
    # Phase 2: Chart scrolls/animates (45-65%)
    # Phase 3: Logo appears (65-100%) - clean cut, no grey fade

    # Chart phase (0-65%)
    if progress < 0.65:
        # Calculate how many candles to show
        chart_progress = min(1.0, progress / 0.4)
        total_candles = len(data["candles"])
        visible_count = max(3, int(total_candles * chart_progress))

        # Scrolling effect after initial build
        if progress > 0.45:
            scroll_progress = (progress - 0.45) / 0.2
            offset = int(scroll_progress * (total_candles - visible_count))
        else:
            offset = 0

        # Glow pulse effect
        glow = abs(math.sin(frame * 0.3)) * 0.5

        chart = render_full_chart(
            data,
            width=width,
            height=height - 2,
            visible_candles=min(visible_count, width // 2),
            offset=offset,
            show_volume=progress > 0.15,
            show_grid=progress > 0.08,
            show_price_axis=progress > 0.12,
            show_ticker=progress > 0.05,
            glow_intensity=glow,
        )
        output.append(chart)

        # Bottom bar with animated elements
        if progress > 0.25:
            bar = Text()
            bar.append("  ")

            # Scrolling ticker tape effect
            tape_items = [
                "AAPL +2.3%",
                "NVDA +5.1%",
                "TSLA -1.2%",
                "MSFT +0.8%",
                "GOOGL +1.5%",
                "AMZN +3.2%",
            ]
            tape_offset = int(frame * 2) % (len(tape_items) * 15)
            tape_str = "  •  ".join(tape_items * 3)
            visible_tape = tape_str[tape_offset : tape_offset + width - 4]

            for item in visible_tape.split("  •  "):
                item = item.strip()
                if "+" in item:
                    bar.append(item + "  •  ", style=THEME["primary"])
                elif "-" in item:
                    bar.append(item + "  •  ", style=THEME["bearish"])
                else:
                    bar.append(item, style=THEME["muted"])

            output.append(bar)

    # Logo phase (65-100%) - clean cut transition
    else:
        logo_progress = (progress - 0.65) / 0.35

        # Vertical centering
        output.append("\n" * (height // 4))

        # Logo always shows once we're in this phase
        logo = render_logo("large" if width > 80 else "small")
        for line in str(logo).split("\n"):
            padding = " " * max(0, (width - len(line)) // 2)
            output.append(padding)
            for char in line:
                if char in "█▀▄╔╗╚╝║═":
                    output.append(char, style=f"bold {THEME['primary']}")
                else:
                    output.append(char)
            output.append("\n")

        output.append("\n")

        # Tagline (appears at 20% of logo phase)
        if logo_progress > 0.2:
            padding = " " * max(0, (width - len(TAGLINE)) // 2)
            output.append(padding)
            output.append(TAGLINE, style=f"italic {THEME['secondary']}")
            output.append("\n")

        # Subtitle (appears at 40%)
        if logo_progress > 0.4:
            padding = " " * max(0, (width - len(SUBTITLE)) // 2)
            output.append(padding)
            output.append(SUBTITLE, style="dim")
            output.append("\n")

        # Separator (appears at 55%)
        if logo_progress > 0.55:
            sep = "─" * 40
            padding = " " * max(0, (width - 40) // 2)
            output.append(padding)
            output.append(sep, style=THEME["muted"])
            output.append("\n")

        # Credit (appears at 70%)
        if logo_progress > 0.70:
            credit_text = "Made with ❤  by Alexander Duria"
            padding = " " * max(0, (width - len(credit_text)) // 2)
            output.append(padding)
            output.append("Made with ", style="dim")
            output.append("❤", style=THEME["bearish"])
            output.append("  by Alexander Duria", style="dim")
            output.append("\n")

    return output


def render_final_welcome(width: int, height: int) -> Text:
    """
    Render the final welcome screen that matches _show_welcome() from repl.py.

    This ensures a smooth transition from animation to interactive state.
    """
    output = Text()

    # Calculate padding to vertically center content
    content_height = 20  # Approximate height of welcome content
    top_padding = max(0, (height - content_height) // 3)
    output.append("\n" * top_padding)

    # Logo
    logo = render_logo("large" if width > 80 else "small")
    for line in str(logo).split("\n"):
        padding = " " * max(0, (width - len(line)) // 2)
        output.append(padding)
        for char in line:
            if char in "█▀▄╔╗╚╝║═":
                output.append(char, style=f"bold {THEME['primary']}")
            else:
                output.append(char)
        output.append("\n")

    output.append("\n")

    # Tagline
    padding = " " * max(0, (width - len(TAGLINE)) // 2)
    output.append(padding)
    output.append(TAGLINE, style=f"italic {THEME['secondary']}")
    output.append("\n")

    # Subtitle
    subtitle = "SEC filings • Market data • AI analysis"
    padding = " " * max(0, (width - len(subtitle)) // 2)
    output.append(padding)
    output.append(subtitle, style="dim")
    output.append("\n")

    # Separator
    sep = "─" * 40
    padding = " " * max(0, (width - 40) // 2)
    output.append(padding)
    output.append(sep, style=THEME["muted"])
    output.append("\n")

    # Credit
    credit_text = "Made with ❤  by Alexander Duria"
    padding = " " * max(0, (width - len(credit_text)) // 2)
    output.append(padding)
    output.append("Made with ", style="dim")
    output.append("❤", style=THEME["bearish"])
    output.append("  by Alexander Duria", style="dim")
    output.append("\n\n")

    # Quick start hints (matching _show_welcome in repl.py)
    hints = [
        "",
        "Quick Start:",
        "  research TICKER    Research a company      e.g. research NVDA",
        "  compare T1 T2      Compare companies       e.g. compare AAPL MSFT",
        "  /framework NAME   Use analysis framework  piotroski, porter, valuation",
        "",
        "Tab for suggestions • /help for all commands • Ctrl+C to exit",
        "",
    ]

    for hint in hints:
        if hint.startswith("Quick Start"):
            output.append(hint, style="bold")
        elif (
            hint.startswith("  research")
            or hint.startswith("  compare")
            or hint.startswith("  /framework")
        ):
            # Parse and color the hint
            parts = hint.split("    ")
            if len(parts) >= 2:
                output.append(f"  {parts[0].strip()}", style="cyan")
                output.append("    ")
                rest = "    ".join(parts[1:])
                if "e.g." in rest:
                    desc, example = rest.split("e.g.")
                    output.append(desc.strip(), style="white")
                    output.append("  e.g. ", style="dim")
                    output.append(example.strip(), style="dim")
                else:
                    output.append(rest.strip(), style="white")
            else:
                output.append(hint)
        elif hint.startswith("Tab for"):
            output.append(hint, style="dim")
        else:
            output.append(hint)
        output.append("\n")

    # Ready indicator
    output.append("✓ Ready\n", style=f"bold {THEME['primary']}")

    return output


async def play_intro_animation(
    duration: float = 4.0,
    skip_callback: Callable[[], bool] | None = None,
) -> bool:
    """
    Play the epic animated intro sequence.

    Args:
        duration: Total animation duration in seconds
        skip_callback: Optional callback that returns True to skip animation

    Returns:
        True if animation completed (welcome screen is displayed)
    """
    import os
    import sys

    # Enable ANSI escape codes on Windows
    if sys.platform == "win32":
        os.system("")  # Enables ANSI escape sequences in Windows terminal

    # Get terminal size
    term_width = console.width or 80
    term_height = console.height or 24

    # Generate market data
    data = generate_market_data(num_candles=80, volatility=0.03)

    # Animation parameters - lower FPS for recording-friendly animation
    fps = 12
    total_frames = int(duration * fps)
    frame_delay = 1.0 / fps

    # ANSI escape codes for flicker-free rendering
    CURSOR_HOME = "\033[H"  # Move cursor to top-left
    CLEAR_SCREEN = "\033[2J"  # Clear entire screen
    HIDE_CURSOR = "\033[?25l"  # Hide cursor
    SHOW_CURSOR = "\033[?25h"  # Show cursor

    skipped = False

    # Clear screen and hide cursor for clean animation
    sys.stdout.write(CLEAR_SCREEN + CURSOR_HOME + HIDE_CURSOR)
    sys.stdout.flush()

    try:
        for frame in range(total_frames):
            # Check for skip
            if skip_callback and skip_callback():
                skipped = True
                break

            # Render frame to string
            content = render_intro_frame(
                data,
                frame,
                total_frames,
                term_width,
                term_height,
            )

            # Move cursor home and overwrite (no clear = no flicker)
            sys.stdout.write(CURSOR_HOME)

            # Render frame content line by line with padding
            lines = []
            with console.capture() as capture:
                console.print(content, end="")
            rendered = capture.get()

            # Pad each line to full width to overwrite previous content
            for line in rendered.split("\n"):
                # Strip ANSI codes to get true length, then pad
                padded = line + " " * max(0, term_width - len(line.encode("utf-8").decode("utf-8")))
                lines.append(padded)

            # Ensure we fill the screen height
            while len(lines) < term_height:
                lines.append(" " * term_width)

            sys.stdout.write("\n".join(lines[:term_height]))
            sys.stdout.flush()

            await asyncio.sleep(frame_delay)

        # Render final welcome state if not skipped
        if not skipped:
            await asyncio.sleep(0.1)
            final_content = render_final_welcome(term_width, term_height)

            sys.stdout.write(CURSOR_HOME)

            with console.capture() as capture:
                console.print(final_content, end="")
            rendered = capture.get()

            lines = []
            for line in rendered.split("\n"):
                padded = line + " " * max(0, term_width - len(line))
                lines.append(padded)
            while len(lines) < term_height:
                lines.append(" " * term_width)

            sys.stdout.write("\n".join(lines[:term_height]))
            sys.stdout.flush()

            await asyncio.sleep(0.1)

    finally:
        # Always show cursor again
        sys.stdout.write(SHOW_CURSOR)
        sys.stdout.flush()

    return not skipped


def show_static_banner(compact: bool = False) -> None:
    """Show the static banner (for --no-animation mode)."""
    console.clear()

    if compact:
        logo = render_logo("small")
    else:
        logo = render_logo("large")

    console.print()
    console.print(Align.center(logo))
    console.print()
    console.print(Align.center(Text(TAGLINE, style=f"italic {THEME['secondary']}")))
    console.print(Align.center(Text(SUBTITLE, style="dim")))
    console.print(Align.center(Text("─" * 40, style=f"{THEME['muted']}")))
    console.print()

    # Credit with heart
    credit = Text()
    credit.append("Made with ", style="dim")
    credit.append("❤", style=THEME["bearish"])
    credit.append("  by Alexander Duria", style="dim")
    console.print(Align.center(credit))
    console.print()


def show_ready_prompt() -> None:
    """Show the ready state after intro."""
    console.print(
        f"  [bold {THEME['primary']}]✓[/] Ready, ask anything",
    )


# Quick test
if __name__ == "__main__":
    import sys

    if "--static" in sys.argv:
        show_static_banner()
        show_ready_prompt()
    else:
        # Animation now shows final welcome, no need to show static banner after
        showed_welcome = asyncio.run(play_intro_animation(duration=4.0))
        if not showed_welcome:
            show_static_banner(compact=True)
            show_ready_prompt()
