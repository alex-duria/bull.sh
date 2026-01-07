"""Debate coordinator for bull vs. bear adversarial debates."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator

from bullsh.agent.bull import BullAgent
from bullsh.agent.bear import BearAgent
from bullsh.agent.moderator import ModeratorAgent, SynthesisResult
from bullsh.agent.base import AgentResult
from bullsh.config import Config, get_config
from bullsh.logging import log
from bullsh.tools import sec

# Stale data threshold: 1 year (365 days)
STALE_THRESHOLD_DAYS = 365


class DebatePhase(Enum):
    """Phases of a debate."""
    INIT = "init"
    RESEARCH = "research"
    OPENING = "opening"
    REBUTTAL = "rebuttal"
    REBUTTAL_2 = "rebuttal_2"  # Deep mode only
    SYNTHESIS = "synthesis"
    COMPLETE = "complete"
    FAILED = "failed"


class DebateRefused(Exception):
    """Raised when debate cannot proceed (e.g., insufficient data)."""
    pass


@dataclass
class DebateState:
    """
    Persisted state for debate resume and session saving.

    Checkpointed after each phase completes.
    """
    ticker: str
    phase: DebatePhase = DebatePhase.INIT
    deep_mode: bool = False
    framework: str | None = None
    framework_context: str | None = None

    # Phase outputs
    bull_research: dict[str, Any] | None = None
    bear_research: dict[str, Any] | None = None
    bull_opening: str | None = None
    bear_opening: str | None = None
    bull_rebuttals: list[str] = field(default_factory=list)
    bear_rebuttals: list[str] = field(default_factory=list)
    user_hints: list[tuple[str, str]] = field(default_factory=list)  # (target, hint)

    # Synthesis
    synthesis: str | None = None
    synthesis_result: SynthesisResult | None = None

    # Metadata
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    tokens_used: int = 0
    error: str | None = None


class DebateCoordinator:
    """
    Orchestrates a bull vs. bear debate through 4 phases.

    Phase 1: RESEARCH (parallel)
        - Bull and Bear agents gather data simultaneously
        - Cache ensures both see identical data

    Phase 2: OPENING (parallel)
        - Both agents present 3-5 strongest points
        - Streamed to user in real-time

    Phase 3: REBUTTAL (sequential)
        - Bull rebuts Bear's weakest points
        - Bear rebuts Bull's weakest points
        - [Deep mode: second round of rebuttals]

    Phase 4: SYNTHESIS
        - Moderator identifies contentions
        - Produces conviction score with reasoning
        - Surfaces thesis-breaker
    """

    def __init__(
        self,
        config: Config | None = None,
        ticker: str | None = None,
        deep_mode: bool = False,
        framework: str | None = None,
        framework_context: str | None = None,
    ):
        self.config = config or get_config()
        self.ticker = ticker
        self.deep_mode = deep_mode
        self.framework = framework
        self.framework_context = framework_context

        # Initialize state
        self.state = DebateState(
            ticker=ticker or "",
            deep_mode=deep_mode,
            framework=framework,
            framework_context=framework_context,
        )

        # Agents (initialized lazily)
        self._bull_agent: BullAgent | None = None
        self._bear_agent: BearAgent | None = None
        self._moderator: ModeratorAgent | None = None

        # Hint queue for coaching
        self._pending_hints: list[tuple[str, str]] = []

    @property
    def bull_agent(self) -> BullAgent:
        if self._bull_agent is None:
            self._bull_agent = BullAgent(
                config=self.config,
                ticker=self.ticker,
                framework_context=self.framework_context,
            )
        return self._bull_agent

    @property
    def bear_agent(self) -> BearAgent:
        if self._bear_agent is None:
            self._bear_agent = BearAgent(
                config=self.config,
                ticker=self.ticker,
                framework_context=self.framework_context,
            )
        return self._bear_agent

    @property
    def moderator(self) -> ModeratorAgent:
        if self._moderator is None:
            self._moderator = ModeratorAgent(
                config=self.config,
                ticker=self.ticker,
            )
        return self._moderator

    def queue_hint(self, target: str, hint: str) -> None:
        """
        Queue a user hint for the next phase.

        Args:
            target: "bull" or "bear"
            hint: The hint text
        """
        target = target.lower()
        if target in ("bull", "bear"):
            self._pending_hints.append((target, hint))
            self.state.user_hints.append((target, hint))
            log("debate", f"Hint queued for {target}: {hint[:50]}...")

    def _apply_pending_hints(self) -> None:
        """Apply queued hints to agents."""
        for target, hint in self._pending_hints:
            if target == "bull":
                self.bull_agent.add_user_hint(hint)
            elif target == "bear":
                self.bear_agent.add_user_hint(hint)
        self._pending_hints = []

    def _detect_stale_data(self, research_results: dict[str, Any] | None) -> tuple[bool, str | None]:
        """
        Check if research data is older than STALE_THRESHOLD_DAYS.

        Args:
            research_results: Research dict with tool_results

        Returns:
            (is_stale, most_recent_filing_date_str)
        """
        if not research_results:
            return False, None

        tool_results = research_results.get("tool_results", [])

        # Find most recent filing date from SEC tool results
        most_recent_date: datetime | None = None

        for result in tool_results:
            tool_name = result.get("tool_name", "")
            if tool_name not in ("sec_fetch", "sec_search"):
                continue

            data = result.get("data", {})

            # Try multiple date field names
            for field_name in ("filed_date", "filing_date", "filed", "date"):
                date_str = data.get(field_name)
                if date_str:
                    try:
                        # Parse date (handle various formats)
                        date_str = str(date_str)[:10]  # Get YYYY-MM-DD part
                        parsed_date = datetime.strptime(date_str, "%Y-%m-%d")
                        if most_recent_date is None or parsed_date > most_recent_date:
                            most_recent_date = parsed_date
                    except ValueError:
                        pass

        if most_recent_date:
            age_days = (datetime.now() - most_recent_date).days
            is_stale = age_days > STALE_THRESHOLD_DAYS
            return is_stale, most_recent_date.strftime("%Y-%m-%d")

        return False, None

    async def _fetch_fresh_data(self) -> str:
        """
        Fetch recent news via web search to supplement stale SEC data.

        Returns:
            String with recent news headlines/snippets
        """
        from bullsh.tools.news import web_search

        current_year = datetime.now().year

        queries = [
            f"{self.ticker} stock earnings {current_year}",
            f"{self.ticker} latest news analysis {current_year}",
        ]

        all_results = []
        for query in queries:
            try:
                result = await web_search(query)
                if result.status.value == "success":
                    # Extract headlines/snippets from search results
                    results_data = result.data.get("results", [])
                    for item in results_data[:3]:  # Top 3 per query
                        title = item.get("title", "")
                        snippet = item.get("snippet", item.get("description", ""))
                        if title:
                            all_results.append(f"- {title}: {snippet[:200]}")
            except Exception as e:
                log("debate", f"Web search failed for '{query}': {e}", level="warning")

        return "\n".join(all_results) if all_results else ""

    async def validate_ticker(self) -> None:
        """
        Validate that we have enough data to debate.

        Raises:
            DebateRefused: If insufficient data
        """
        log("debate", f"Validating ticker: {self.ticker}")

        result = await sec.sec_search(self.ticker, fuzzy=True)

        if result.status.value == "failed":
            raise DebateRefused(
                f"Cannot debate {self.ticker}: Unable to search SEC filings. "
                f"Error: {result.error_message}"
            )

        # Filings are nested inside result.data["filings"]
        filings = result.data.get("filings", {})
        has_10k = bool(filings.get("10-K"))
        has_10q = bool(filings.get("10-Q"))

        if not has_10k and not has_10q:
            raise DebateRefused(
                f"Cannot debate {self.ticker}: No 10-K or 10-Q filings found. "
                "Debates require at least one SEC filing. "
                "Try /research for preliminary analysis of companies without filings."
            )

        log("debate", f"Ticker validated: {self.ticker} (10-K: {has_10k}, 10-Q: {has_10q})")

    async def run(self) -> AsyncIterator[str]:
        """
        Execute the full debate, yielding output chunks.

        Yields:
            Streamed output text for display
            Special markers: <<PHASE_PAUSE:phase_name>> for UI pause points
        """
        try:
            # Validate ticker first
            await self.validate_ticker()

            # Phase 1: Research
            async for chunk in self._run_research():
                yield chunk
            yield "<<PHASE_PAUSE:research>>"

            # Phase 2: Opening arguments
            async for chunk in self._run_opening():
                yield chunk
            yield "<<PHASE_PAUSE:opening>>"

            # Phase 3: Rebuttals
            async for chunk in self._run_rebuttals():
                yield chunk
            yield "<<PHASE_PAUSE:rebuttals>>"

            # Phase 4: Synthesis
            async for chunk in self._run_synthesis():
                yield chunk
            # No pause after synthesis - debate is complete

            self.state.phase = DebatePhase.COMPLETE
            self.state.completed_at = datetime.now()

        except DebateRefused as e:
            self.state.phase = DebatePhase.FAILED
            self.state.error = str(e)
            yield f"\n{e}\n"

        except Exception as e:
            self.state.phase = DebatePhase.FAILED
            self.state.error = str(e)
            log("debate", f"Debate failed: {e}", level="error")
            yield f"\nDebate failed: {e}\n"

    async def _run_research(self) -> AsyncIterator[str]:
        """Phase 1: Parallel research."""
        self.state.phase = DebatePhase.RESEARCH
        yield f"\n**Phase 1: Research**\n"

        # Research tasks for both agents
        bull_task = f"Research {self.ticker} to build a BULL case. Focus on strengths, growth, and competitive advantages."
        bear_task = f"Research {self.ticker} to build a BEAR case. Focus on risks, valuation, and competitive threats."

        yield f"  Gathering data for {self.ticker}...\n"

        # Run research in parallel
        bull_result, bear_result = await asyncio.gather(
            self.bull_agent.run(bull_task),
            self.bear_agent.run(bear_task),
            return_exceptions=True,
        )

        # Handle exceptions
        if isinstance(bull_result, Exception):
            log("debate", f"Bull research failed: {bull_result}", level="error")
            bull_result = AgentResult(
                content="Research failed",
                ticker=self.ticker,
                success=False,
                error=str(bull_result),
            )

        if isinstance(bear_result, Exception):
            log("debate", f"Bear research failed: {bear_result}", level="error")
            bear_result = AgentResult(
                content="Research failed",
                ticker=self.ticker,
                success=False,
                error=str(bear_result),
            )

        # Store results
        self.state.bull_research = {
            "content": bull_result.content,
            "tool_results": bull_result.tool_results,
            "tokens": bull_result.tokens_used,
        }
        self.state.bear_research = {
            "content": bear_result.content,
            "tool_results": bear_result.tool_results,
            "tokens": bear_result.tokens_used,
        }
        self.state.tokens_used += bull_result.tokens_used + bear_result.tokens_used

        yield f"  Research complete ({bull_result.tokens_used + bear_result.tokens_used:,} tokens)\n"

        # Check if data is stale (> 1 year old)
        is_stale, date_str = self._detect_stale_data(self.state.bull_research)
        if is_stale:
            yield f"\n  ⚠️ SEC filing data may be outdated (most recent: {date_str})\n"
            yield "  Fetching recent news to supplement...\n"

            fresh_data = await self._fetch_fresh_data()
            if fresh_data:
                self.state.bull_research["fresh_context"] = fresh_data
                self.state.bear_research["fresh_context"] = fresh_data
                yield f"  ✓ Fresh context added ({len(fresh_data)} chars)\n"
            else:
                yield "  ⚠️ No fresh data found\n"

        yield "\n"

    async def _run_opening(self) -> AsyncIterator[str]:
        """Phase 2: Opening arguments."""
        self.state.phase = DebatePhase.OPENING

        # Prepare research summaries
        bull_research_summary = self._format_research_summary(self.state.bull_research)
        bear_research_summary = self._format_research_summary(self.state.bear_research)

        yield "**Phase 2: Opening Arguments**\n\n"

        # Run openings in parallel
        bull_opening_task = self.bull_agent.run_opening(bull_research_summary)
        bear_opening_task = self.bear_agent.run_opening(bear_research_summary)

        bull_result, bear_result = await asyncio.gather(
            bull_opening_task,
            bear_opening_task,
            return_exceptions=True,
        )

        # Handle bull result
        if isinstance(bull_result, Exception):
            bull_result = AgentResult(content="Opening failed", success=False, error=str(bull_result))

        self.state.bull_opening = bull_result.content
        self.state.tokens_used += bull_result.tokens_used

        yield "🐂 **BULL CASE**\n"
        yield "────────────────\n"
        yield bull_result.content
        yield "\n\n"

        # Handle bear result
        if isinstance(bear_result, Exception):
            bear_result = AgentResult(content="Opening failed", success=False, error=str(bear_result))

        self.state.bear_opening = bear_result.content
        self.state.tokens_used += bear_result.tokens_used

        yield "🐻 **BEAR CASE**\n"
        yield "────────────────\n"
        yield bear_result.content
        yield "\n\n"

    async def _run_rebuttals(self) -> AsyncIterator[str]:
        """Phase 3: Rebuttals (sequential)."""
        self.state.phase = DebatePhase.REBUTTAL

        yield "**Phase 3: Rebuttals**\n\n"

        # Apply any pending hints
        self._apply_pending_hints()

        # Bull rebuts Bear
        yield "🐂 **BULL REBUTTAL**\n"
        yield "────────────────\n"

        bull_rebuttal = await self.bull_agent.run_rebuttal(
            self.state.bull_opening or "",
            self.state.bear_opening or "",
        )
        self.state.bull_rebuttals.append(bull_rebuttal.content)
        self.state.tokens_used += bull_rebuttal.tokens_used

        yield bull_rebuttal.content
        yield "\n\n"

        # Bear rebuts Bull
        yield "🐻 **BEAR REBUTTAL**\n"
        yield "────────────────\n"

        bear_rebuttal = await self.bear_agent.run_rebuttal(
            self.state.bear_opening or "",
            self.state.bull_opening or "",
        )
        self.state.bear_rebuttals.append(bear_rebuttal.content)
        self.state.tokens_used += bear_rebuttal.tokens_used

        yield bear_rebuttal.content
        yield "\n\n"

        # Deep mode: second round
        if self.deep_mode:
            self.state.phase = DebatePhase.REBUTTAL_2

            # Apply any hints queued during first rebuttal
            self._apply_pending_hints()

            yield "**Round 2 Rebuttals**\n\n"

            # Bull second rebuttal
            yield "🐂 **BULL REBUTTAL (Round 2)**\n"
            yield "────────────────\n"

            bull_rebuttal_2 = await self.bull_agent.run_rebuttal(
                self.state.bull_opening + "\n\n" + self.state.bull_rebuttals[0],
                self.state.bear_opening + "\n\n" + self.state.bear_rebuttals[0],
            )
            self.state.bull_rebuttals.append(bull_rebuttal_2.content)
            self.state.tokens_used += bull_rebuttal_2.tokens_used

            yield bull_rebuttal_2.content
            yield "\n\n"

            # Bear second rebuttal
            yield "🐻 **BEAR REBUTTAL (Round 2)**\n"
            yield "────────────────\n"

            bear_rebuttal_2 = await self.bear_agent.run_rebuttal(
                self.state.bear_opening + "\n\n" + self.state.bear_rebuttals[0],
                self.state.bull_opening + "\n\n" + self.state.bull_rebuttals[0],
            )
            self.state.bear_rebuttals.append(bear_rebuttal_2.content)
            self.state.tokens_used += bear_rebuttal_2.tokens_used

            yield bear_rebuttal_2.content
            yield "\n\n"

    async def _run_synthesis(self) -> AsyncIterator[str]:
        """Phase 4: Moderator synthesis."""
        self.state.phase = DebatePhase.SYNTHESIS

        yield "**Phase 4: Moderator Synthesis**\n\n"
        yield "⚖️  **SYNTHESIS**\n"
        yield "────────────────\n"

        result = await self.moderator.synthesize(
            bull_opening=self.state.bull_opening or "",
            bear_opening=self.state.bear_opening or "",
            bull_rebuttals=self.state.bull_rebuttals,
            bear_rebuttals=self.state.bear_rebuttals,
        )

        self.state.synthesis = result.content
        self.state.tokens_used += result.tokens_used

        # Parse synthesis for structured data
        if result.content:
            self.state.synthesis_result = self.moderator.parse_synthesis(result.content)

        yield result.content
        yield "\n"

    def _format_research_summary(self, research: dict[str, Any] | None) -> str:
        """Format research results as context for opening arguments."""
        if not research:
            return "No research data available."

        lines = []

        # Add content summary
        if research.get("content"):
            lines.append(research["content"][:2000])

        # Add key tool results
        for result in research.get("tool_results", [])[:5]:
            confidence = result.get("confidence", 0)
            if confidence > 0.5:
                tool_name = result.get("tool_name", "unknown")
                data = result.get("data", {})
                data_str = str(data)[:500]
                lines.append(f"\n[{tool_name}] (confidence: {confidence:.0%}):\n{data_str}")

        # Append fresh context if available (supplements stale SEC data)
        if research.get("fresh_context"):
            lines.append("\n\n[RECENT NEWS - supplements outdated SEC filings]:")
            lines.append(research["fresh_context"])

        return "\n".join(lines)

    def get_debate_summary(self) -> dict[str, Any]:
        """Get a summary of the debate for export."""
        return {
            "ticker": self.ticker,
            "mode": "deep" if self.deep_mode else "quick",
            "framework": self.framework,
            "phase": self.state.phase.value,
            "bull_opening": self.state.bull_opening,
            "bear_opening": self.state.bear_opening,
            "bull_rebuttals": self.state.bull_rebuttals,
            "bear_rebuttals": self.state.bear_rebuttals,
            "synthesis": self.state.synthesis,
            "conviction": self.state.synthesis_result.conviction if self.state.synthesis_result else None,
            "conviction_direction": self.state.synthesis_result.conviction_direction if self.state.synthesis_result else None,
            "thesis_breaker": self.state.synthesis_result.thesis_breaker if self.state.synthesis_result else None,
            "tokens_used": self.state.tokens_used,
            "started_at": self.state.started_at.isoformat(),
            "completed_at": self.state.completed_at.isoformat() if self.state.completed_at else None,
        }
