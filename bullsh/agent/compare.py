"""Compare subagent for parallel multi-company analysis."""

import asyncio
from typing import Any

from bullsh.agent.base import AgentResult, SubAgent
from bullsh.agent.research import ResearchAgent
from bullsh.config import Config
from bullsh.logging import log


class CompareAgent(SubAgent):
    """
    Subagent for comparing multiple companies in parallel.

    Spawns ResearchAgent instances for each ticker concurrently,
    then synthesizes results into a comparative analysis.
    """

    def __init__(
        self,
        config: Config | None = None,
        max_iterations: int = 5,
        research_iterations: int = 8,
    ):
        super().__init__(config, max_iterations)
        self.research_iterations = research_iterations

    @property
    def system_prompt(self) -> str:
        return """You are a comparison analyst synthesizing research on multiple companies.

You have received research summaries for each company. Your task is to:
1. Compare the companies across key metrics
2. Identify relative strengths and weaknesses
3. Provide a clear ranking or recommendation

Format your comparison as:
1. **Executive Summary** - 2-3 sentences on the overall picture
2. **Metrics Comparison** - Table comparing key numbers
3. **Strengths & Weaknesses** - Bullet points for each company
4. **Verdict** - Your comparative assessment

Be objective and data-driven. Cite specific numbers from the research."""

    @property
    def tools(self) -> list[dict[str, Any]]:
        """Comparison agent doesn't need tools - it synthesizes existing research."""
        return []

    async def run(
        self,
        task: str,
        tickers: list[str] | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """
        Compare multiple companies in parallel.

        Args:
            task: Comparison task description
            tickers: List of stock tickers to compare

        Returns:
            AgentResult with comparative analysis
        """
        self.tool_results = []

        if not tickers or len(tickers) < 2:
            return AgentResult(
                content="Comparison requires at least 2 tickers.",
                tool_results=[],
                success=False,
                error="Insufficient tickers for comparison",
            )

        tickers = [t.upper() for t in tickers]
        log("agent", f"CompareAgent starting: comparing {', '.join(tickers)}")

        total_tokens = 0

        try:
            # Phase 1: Parallel research on all companies
            research_results = await self._research_companies(tickers, task)

            # Collect tokens from research
            for result in research_results:
                total_tokens += result.tokens_used
                self.tool_results.extend(result.tool_results)

            # Check for failures
            successful = [r for r in research_results if r.success]
            if len(successful) < 2:
                failed_tickers = [r.ticker or "unknown" for r in research_results if not r.success]
                return AgentResult(
                    content=f"Could not gather enough data. Failed: {', '.join(failed_tickers)}",
                    tool_results=self.tool_results,
                    success=False,
                    error="Insufficient research data",
                    tokens_used=total_tokens,
                )

            # Phase 2: Synthesize comparison
            comparison = await self._synthesize_comparison(task, research_results, total_tokens)

            return comparison

        except Exception as e:
            log("agent", f"CompareAgent error: {e}", level="error")
            return AgentResult(
                content="",
                tool_results=self.tool_results,
                success=False,
                error=str(e),
                tokens_used=total_tokens,
            )

    async def _research_companies(
        self,
        tickers: list[str],
        task: str,
    ) -> list[AgentResult]:
        """
        Run research on multiple companies in parallel.

        Args:
            tickers: List of tickers to research
            task: Original comparison task for context

        Returns:
            List of AgentResult from each ResearchAgent
        """
        log("agent", f"Starting parallel research for {len(tickers)} companies")

        # Create research tasks
        async def research_one(ticker: str) -> AgentResult:
            agent = ResearchAgent(
                config=self.config,
                max_iterations=self.research_iterations,
                include_sec=True,
                include_market=True,
                include_social=False,  # Skip social for comparison speed
            )

            research_task = f"""Research {ticker} for comparison analysis.

Context: {task}

Focus on gathering:
- Company overview and business model
- Key financial metrics (revenue, earnings, margins)
- Valuation metrics (P/E, P/S, market cap)
- Growth rates and trends
- Recent developments or concerns

Be efficient - gather key data quickly for comparison."""

            return await agent.run(research_task, ticker=ticker)

        # Execute all research in parallel
        results = await asyncio.gather(
            *[research_one(ticker) for ticker in tickers],
            return_exceptions=True,
        )

        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                log("agent", f"Research failed for {tickers[i]}: {result}", level="error")
                processed_results.append(
                    AgentResult(
                        content="",
                        tool_results=[],
                        ticker=tickers[i],
                        success=False,
                        error=str(result),
                    )
                )
            else:
                processed_results.append(result)

        log(
            "agent",
            f"Parallel research complete: {len([r for r in processed_results if r.success])}/{len(tickers)} successful",
        )
        return processed_results

    async def _synthesize_comparison(
        self,
        task: str,
        research_results: list[AgentResult],
        tokens_so_far: int,
    ) -> AgentResult:
        """
        Synthesize research results into a comparative analysis.

        Args:
            task: Original comparison task
            research_results: Results from parallel research
            tokens_so_far: Tokens used in research phase

        Returns:
            AgentResult with synthesized comparison
        """
        log("agent", "Synthesizing comparison from research results")

        # Build context from research
        context_parts = []
        tickers = []

        for result in research_results:
            if result.success and result.content:
                tickers.append(result.ticker)
                context_parts.append(result.to_context())

        research_context = "\n\n---\n\n".join(context_parts)

        # Create synthesis prompt
        synthesis_prompt = f"""Compare these companies based on the research data:

**Original Question:** {task}

**Research Data:**

{research_context}

---

Now provide a comprehensive comparison of {", ".join(tickers)}.
Focus on answering the user's question with specific data from the research."""

        messages = [{"role": "user", "content": synthesis_prompt}]
        total_tokens = tokens_so_far

        try:
            for iteration in range(self.max_iterations):
                log("agent", f"CompareAgent synthesis iteration {iteration + 1}")

                response = await self._call_claude(messages)
                total_tokens += response.usage.input_tokens + response.usage.output_tokens

                # Extract text content
                text_content = ""
                for block in response.content:
                    if block.type == "text":
                        text_content += block.text

                # CompareAgent doesn't use tools in synthesis phase
                log("agent", "CompareAgent synthesis complete")
                return AgentResult(
                    content=text_content,
                    tool_results=self.tool_results,
                    ticker=None,  # Multiple tickers
                    success=True,
                    tokens_used=total_tokens,
                )

            # Shouldn't reach here, but handle gracefully
            return AgentResult(
                content="Comparison synthesis incomplete.",
                tool_results=self.tool_results,
                success=False,
                error="Max iterations reached",
                tokens_used=total_tokens,
            )

        except Exception as e:
            log("agent", f"Synthesis error: {e}", level="error")
            return AgentResult(
                content="",
                tool_results=self.tool_results,
                success=False,
                error=str(e),
                tokens_used=total_tokens,
            )
