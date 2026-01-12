"""Research subagent for single-company deep dives."""

from typing import Any

from bullsh.agent.base import AgentResult, SubAgent
from bullsh.config import Config
from bullsh.logging import log
from bullsh.tools.base import get_tools_for_claude


class ResearchAgent(SubAgent):
    """
    Subagent specialized for researching a single company.

    Runs with bounded iterations and focused context.
    Used by CompareAgent to research companies in parallel.
    """

    def __init__(
        self,
        config: Config | None = None,
        max_iterations: int = 10,
        include_sec: bool = True,
        include_market: bool = True,
        include_social: bool = False,
    ):
        super().__init__(config, max_iterations)
        self.include_sec = include_sec
        self.include_market = include_market
        self.include_social = include_social

    @property
    def system_prompt(self) -> str:
        return """You are a focused research agent analyzing a single company.

Your task is to gather key data efficiently:
1. Check SEC filings availability
2. Fetch latest 10-K for fundamentals
3. Get current market data (price, P/E, market cap)
4. Search for relevant information in indexed filings

IMPORTANT:
- Be efficient - don't make unnecessary tool calls
- Focus on KEY metrics: revenue, earnings, margins, growth
- Summarize findings concisely
- After 3-4 tool calls, synthesize what you have
- Your output will be combined with other company research

Output format:
- Start with a 1-sentence company description
- List key metrics in a compact format
- Note any significant findings or concerns
- Keep total output under 500 words"""

    @property
    def tools(self) -> list[dict[str, Any]]:
        """Return subset of tools for research."""
        all_tools = get_tools_for_claude()
        # Filter to research-relevant tools
        research_tools = [
            "sec_search",
            "sec_fetch",
            "rag_search",
            "scrape_yahoo",
            "compute_ratios",
        ]
        if self.include_social:
            research_tools.extend(["search_stocktwits", "search_news"])

        return [t for t in all_tools if t["name"] in research_tools]

    async def run(self, task: str, ticker: str | None = None, **kwargs: Any) -> AgentResult:
        """
        Research a company and return structured findings.

        Args:
            task: Research task description
            ticker: Stock ticker to research

        Returns:
            AgentResult with research findings
        """
        self.tool_results = []  # Reset for new run
        ticker = ticker.upper() if ticker else None

        log("agent", f"ResearchAgent starting: {ticker or task}")

        messages = [{"role": "user", "content": task}]
        total_tokens = 0

        try:
            for iteration in range(self.max_iterations):
                log("agent", f"ResearchAgent iteration {iteration + 1}/{self.max_iterations}")

                response = await self._call_claude(messages)
                total_tokens += response.usage.input_tokens + response.usage.output_tokens

                # Extract text content
                text_content = ""
                tool_uses = []

                for block in response.content:
                    if block.type == "text":
                        text_content += block.text
                    elif block.type == "tool_use":
                        tool_uses.append(block)

                # No more tool calls - we're done
                if not tool_uses:
                    log(
                        "agent",
                        f"ResearchAgent completed with {len(self.tool_results)} tool results",
                    )
                    return AgentResult(
                        content=text_content,
                        tool_results=self.tool_results,
                        ticker=ticker,
                        success=True,
                        tokens_used=total_tokens,
                    )

                # Execute tool calls
                tool_results = []
                for tool_use in tool_uses:
                    result = await self._execute_tool(tool_use.name, tool_use.input)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": self._format_tool_result(result),
                        }
                    )

                # Update messages for next iteration
                messages.append(
                    {
                        "role": "assistant",
                        "content": response.content,
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": tool_results,
                    }
                )

            # Max iterations reached
            log("agent", f"ResearchAgent hit max iterations for {ticker}")
            return AgentResult(
                content=text_content or "Research incomplete - max iterations reached.",
                tool_results=self.tool_results,
                ticker=ticker,
                success=True,  # Partial success
                tokens_used=total_tokens,
            )

        except Exception as e:
            log("agent", f"ResearchAgent error: {e}", level="error")
            return AgentResult(
                content="",
                tool_results=self.tool_results,
                ticker=ticker,
                success=False,
                error=str(e),
                tokens_used=total_tokens,
            )
