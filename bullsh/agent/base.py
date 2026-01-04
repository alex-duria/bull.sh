"""Base classes for subagent architecture."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from anthropic import AsyncAnthropic

from bullsh.config import Config, get_config
from bullsh.logging import log


@dataclass
class AgentResult:
    """Result from a subagent execution."""
    content: str
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    ticker: str | None = None
    success: bool = True
    error: str | None = None
    tokens_used: int = 0

    def to_context(self) -> str:
        """Convert to context string for parent agent."""
        if not self.success:
            return f"[{self.ticker or 'Agent'} Error]: {self.error}"

        lines = []
        if self.ticker:
            lines.append(f"[{self.ticker} Research Summary]")
        lines.append(self.content)

        # Include key tool results
        for result in self.tool_results[:5]:  # Limit context size
            if result.get("confidence", 0) > 0.5:
                tool_name = result.get("tool_name", "unknown")
                data = result.get("data", {})
                if data:
                    lines.append(f"\n[{tool_name}]: {str(data)[:500]}")

        return "\n".join(lines)


class SubAgent(ABC):
    """
    Base class for specialized subagents.

    Subagents run with their own context window, bounded iterations,
    and specific system prompts for their task.
    """

    def __init__(
        self,
        config: Config | None = None,
        max_iterations: int = 10,
    ):
        self.config = config or get_config()
        self.max_iterations = max_iterations
        self.client = AsyncAnthropic(api_key=self.config.anthropic_api_key)
        self.tool_results: list[dict[str, Any]] = []

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System prompt for this agent type."""
        pass

    @property
    @abstractmethod
    def tools(self) -> list[dict[str, Any]]:
        """Tools available to this agent."""
        pass

    @abstractmethod
    async def run(self, task: str, **kwargs: Any) -> AgentResult:
        """Execute the agent's task."""
        pass

    async def _call_claude(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
    ) -> Any:
        """Make a Claude API call."""
        try:
            response = await self.client.messages.create(
                model=self.config.model,
                max_tokens=4096,
                system=system or self.system_prompt,
                messages=messages,
                tools=self.tools if self.tools else None,
            )
            return response
        except Exception as e:
            log("agent", f"SubAgent API error: {e}", level="error")
            raise

    async def _execute_tool(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool and return the result."""
        # Import tools lazily
        from bullsh.tools import sec, yahoo, social, news, rag
        from bullsh.tools.base import ToolStatus

        try:
            match name:
                case "sec_search":
                    result = await sec.sec_search(
                        params["ticker"],
                        params.get("fuzzy", True),
                    )
                case "sec_fetch":
                    result = await sec.sec_fetch(
                        params["ticker"],
                        params["filing_type"],
                        params.get("year"),
                        params.get("section"),
                    )
                case "scrape_yahoo":
                    result = await yahoo.scrape_yahoo(params["ticker"])
                case "search_news":
                    result = await news.search_news(
                        params["query"],
                        params.get("days_back", 30),
                    )
                case "web_search":
                    result = await news.web_search(
                        params["query"],
                        params.get("max_results", 10),
                    )
                case "search_stocktwits":
                    result = await social.search_stocktwits(params["symbol"])
                case "search_reddit":
                    result = await social.search_reddit(
                        params["query"],
                        params.get("subreddits"),
                    )
                case "rag_search":
                    result = await rag.rag_search(
                        params["query"],
                        params.get("ticker"),
                        params.get("form"),
                        params.get("year"),
                        params.get("k", 5),
                    )
                case "compute_ratios":
                    result = await yahoo.compute_ratios(params["ticker"])
                case _:
                    return {
                        "tool_name": name,
                        "status": "failed",
                        "error": f"Unknown tool: {name}",
                    }

            # Store result for context
            result_dict = {
                "tool_name": name,
                "status": result.status.value,
                "confidence": result.confidence,
                "data": result.data,
                "error": result.error_message,
            }
            self.tool_results.append(result_dict)

            return result_dict

        except Exception as e:
            log("agent", f"SubAgent tool error {name}: {e}", level="error")
            return {
                "tool_name": name,
                "status": "failed",
                "error": str(e),
            }

    def _format_tool_result(self, result: dict[str, Any]) -> str:
        """Format tool result for Claude."""
        if result.get("status") == "failed":
            return f"Error: {result.get('error', 'Unknown error')}"

        data = result.get("data", {})
        # Truncate large results
        data_str = str(data)
        if len(data_str) > 4000:
            data_str = data_str[:4000] + "... [truncated]"
        return data_str
