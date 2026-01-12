"""
Suggestion and Tip engines for proactive agent communication.

Provides context-aware suggestions and tips after agent responses.
"""

from dataclasses import dataclass, field


@dataclass
class Suggestion:
    """A single suggestion with display text and command to execute."""

    label: str
    command: str
    needs_input: bool = False  # If True, command has placeholder needing user input


@dataclass
class SuggestionContext:
    """Context for generating suggestions."""

    action: str  # "research", "debate", "compare", "framework", "conversation"
    tickers: list[str] = field(default_factory=list)
    framework: str | None = None
    message_count: int = 0
    last_response_topics: list[str] = field(default_factory=list)


class SuggestionEngine:
    """Generate context-aware suggestions for next steps."""

    def get_suggestions(self, context: SuggestionContext) -> list[Suggestion]:
        """Generate suggestions based on current context."""
        action = context.action
        tickers = context.tickers

        if action == "research" and tickers:
            return self._research_suggestions(tickers[0])
        elif action == "debate" and tickers:
            return self._debate_suggestions(tickers[0])
        elif action == "compare" and len(tickers) >= 2:
            return self._compare_suggestions(tickers)
        elif action == "framework" and tickers:
            return self._framework_suggestions(tickers[0])
        elif action == "conversation" and tickers:
            return self._conversation_suggestions(tickers, context.message_count)
        else:
            return self._default_suggestions()

    def _research_suggestions(self, ticker: str) -> list[Suggestion]:
        """Suggestions after researching a single ticker."""
        return [
            Suggestion(
                label=f"Compare {ticker} with a peer",
                command=f"/compare {ticker} ",
                needs_input=True,
            ),
            Suggestion(
                label="Run Piotroski F-Score",
                command="/framework piotroski",
            ),
            Suggestion(
                label="Generate investment thesis",
                command=f"/thesis {ticker}",
            ),
            Suggestion(
                label="Export to Excel",
                command=f"/excel {ticker}",
            ),
        ]

    def _debate_suggestions(self, ticker: str) -> list[Suggestion]:
        """Suggestions after a bull vs bear debate."""
        return [
            Suggestion(
                label="Explore bull arguments deeper",
                command="Tell me more about the bull case arguments",
            ),
            Suggestion(
                label="Explore bear arguments deeper",
                command="Tell me more about the bear case arguments",
            ),
            Suggestion(
                label="Export debate summary",
                command="/export",
            ),
            Suggestion(
                label="Run valuation analysis",
                command="/framework valuation",
            ),
        ]

    def _compare_suggestions(self, tickers: list[str]) -> list[Suggestion]:
        """Suggestions after comparing multiple tickers."""
        suggestions = [
            Suggestion(
                label=f"Deep dive on {tickers[0]}",
                command=f"/research {tickers[0]}",
            ),
            Suggestion(
                label=f"Deep dive on {tickers[1]}",
                command=f"/research {tickers[1]}",
            ),
            Suggestion(
                label="Run valuation on both",
                command="/framework valuation",
            ),
            Suggestion(
                label="Export comparison",
                command=f"/excel compare {' '.join(tickers)}",
            ),
        ]
        return suggestions

    def _framework_suggestions(self, ticker: str) -> list[Suggestion]:
        """Suggestions after a framework analysis."""
        return [
            Suggestion(
                label="Try another framework",
                command="/framework ",
                needs_input=True,
            ),
            Suggestion(
                label="Generate thesis",
                command=f"/thesis {ticker}",
            ),
            Suggestion(
                label="Export analysis",
                command="/export",
            ),
            Suggestion(
                label=f"Compare {ticker} with peers",
                command=f"/compare {ticker} ",
                needs_input=True,
            ),
        ]

    def _conversation_suggestions(self, tickers: list[str], message_count: int) -> list[Suggestion]:
        """Suggestions after a conversational response."""
        suggestions = []

        if tickers:
            ticker = tickers[-1]  # Most recent ticker
            suggestions.append(
                Suggestion(
                    label="Go deeper on this topic",
                    command="Tell me more about this",
                )
            )
            suggestions.append(
                Suggestion(
                    label=f"Run analysis framework on {ticker}",
                    command="/framework ",
                    needs_input=True,
                )
            )

        suggestions.append(
            Suggestion(
                label="Research another ticker",
                command="/research ",
                needs_input=True,
            )
        )

        if message_count > 5:
            suggestions.append(
                Suggestion(
                    label="Save session",
                    command="/save",
                )
            )
        else:
            suggestions.append(
                Suggestion(
                    label="Export current research",
                    command="/export",
                )
            )

        return suggestions

    def _default_suggestions(self) -> list[Suggestion]:
        """Default suggestions when no specific context."""
        return [
            Suggestion(
                label="Research a company",
                command="/research ",
                needs_input=True,
            ),
            Suggestion(
                label="Compare companies",
                command="/compare ",
                needs_input=True,
            ),
            Suggestion(
                label="Run a debate",
                command="/debate ",
                needs_input=True,
            ),
            Suggestion(
                label="View available frameworks",
                command="/frameworks",
            ),
        ]


class TipEngine:
    """Context-aware tip suggestions. Tracks shown tips to avoid repetition."""

    def __init__(self):
        self.tips_shown: set[str] = set()

    def get_tip(self, context: SuggestionContext) -> str | None:
        """Get a relevant tip, or None if no tip applies."""
        action = context.action
        tickers = context.tickers
        framework = context.framework
        message_count = context.message_count

        # First research tip
        if action == "research" and "first_research" not in self.tips_shown:
            self.tips_shown.add("first_research")
            return "Use /framework piotroski for quantitative health scoring"

        # After debate tip
        if action == "debate" and "debate_export" not in self.tips_shown:
            self.tips_shown.add("debate_export")
            return "Use /export to save this debate summary"

        # Multiple tickers tip
        if len(tickers) >= 2 and "compare_tip" not in self.tips_shown:
            self.tips_shown.add("compare_tip")
            t1, t2 = tickers[:2]
            return f"Use /compare {t1} {t2} for side-by-side analysis"

        # After framework tip
        if action == "framework" and framework and "thesis_tip" not in self.tips_shown:
            self.tips_shown.add("thesis_tip")
            return "Use /thesis to generate a full investment thesis"

        # Long session tip
        if message_count >= 10 and "save_tip" not in self.tips_shown:
            self.tips_shown.add("save_tip")
            return "Use /save to preserve your research session"

        # Compare tip after framework
        if (
            action == "framework"
            and len(tickers) == 1
            and "compare_after_framework" not in self.tips_shown
        ):
            self.tips_shown.add("compare_after_framework")
            return f"Use /compare {tickers[0]} <PEER> to see how it stacks up"

        return None

    def reset(self):
        """Reset shown tips (e.g., for new session)."""
        self.tips_shown.clear()


def format_suggestions(suggestions: list[Suggestion]) -> str:
    """Format suggestions as a numbered menu using Rich markup."""
    if not suggestions:
        return ""

    lines = ["\n[dim]Next steps:[/dim]"]
    for i, s in enumerate(suggestions, 1):
        hint = " [dim]...[/dim]" if s.needs_input else ""
        lines.append(f"  [cyan][{i}][/cyan] {s.label}{hint}")

    return "\n".join(lines)


def format_tip(tip: str) -> str:
    """Format a tip with styling using Rich markup."""
    return f"\n[dim italic]Tip: {tip}[/dim italic]"


class SuggestionState:
    """Track current suggestions for quick numeric execution."""

    def __init__(self):
        self.current_suggestions: list[Suggestion] = []

    def set_suggestions(self, suggestions: list[Suggestion]):
        """Store current suggestions for potential execution."""
        self.current_suggestions = suggestions

    def get_command(self, index: int) -> str | None:
        """Get command for a 1-based suggestion index."""
        if 1 <= index <= len(self.current_suggestions):
            return self.current_suggestions[index - 1].command
        return None

    def needs_input(self, index: int) -> bool:
        """Check if suggestion needs additional user input."""
        if 1 <= index <= len(self.current_suggestions):
            return self.current_suggestions[index - 1].needs_input
        return False

    def clear(self):
        """Clear current suggestions."""
        self.current_suggestions = []


def parse_numeric_input(user_input: str) -> int | None:
    """Parse numeric input like '1', '[1]', '2', '[2]', etc."""
    stripped = user_input.strip()

    # Handle [1], [2], etc.
    if stripped.startswith("[") and stripped.endswith("]"):
        inner = stripped[1:-1]
        if inner.isdigit():
            return int(inner)

    # Handle plain 1, 2, etc.
    if stripped.isdigit() and len(stripped) == 1:
        return int(stripped)

    return None
