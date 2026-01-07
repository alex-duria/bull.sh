"""Bear agent for adversarial debate."""

from typing import Any

from bullsh.agent.base import SubAgent, AgentResult
from bullsh.config import Config
from bullsh.tools.base import get_tools_for_claude
from bullsh.logging import log


class BearAgent(SubAgent):
    """
    Subagent that argues the bear case for a stock.

    Focuses on valuation concerns, competitive threats, and
    reasons to be cautious. Can concede valid bull points
    but must advocate for the bearish thesis.
    """

    def __init__(
        self,
        config: Config | None = None,
        max_iterations: int = 5,
        ticker: str | None = None,
        framework_context: str | None = None,
    ):
        super().__init__(config, max_iterations)
        self.ticker = ticker
        self.framework_context = framework_context
        self._opponent_opening: str | None = None
        self._user_hints: list[str] = []

    def set_opponent_opening(self, opening: str) -> None:
        """Set the bull's opening argument for rebuttal phase."""
        self._opponent_opening = opening

    def add_user_hint(self, hint: str) -> None:
        """Add a user hint to be incorporated."""
        self._user_hints.append(hint)

    @property
    def system_prompt(self) -> str:
        base_prompt = f"""You are an investment analyst arguing the BEAR CASE for {self.ticker or 'the company'}.
Your job is to find the strongest reasons to be cautious.

Focus on:
- Valuation concerns and downside scenarios
- Competitive threats and disruption risk
- Execution risks and management issues
- Macro/regulatory headwinds
- Why bulls are wrong

RULES:
- You MAY concede valid points ("I acknowledge X, but...")
- You MUST cite sources inline: "P/E of 70x (Yahoo Finance)"
- You MUST present 3-5 points even if the case is weak
- Be rigorous but skeptical
- Keep your argument focused and under 400 words"""

        if self.framework_context:
            base_prompt += f"""

FRAMEWORK CONTEXT:
{self.framework_context}

This framework data is available to both you and the bull. You may reference it
to support your argument. If framework findings contradict your position,
you MUST reconcile (explain why despite the data, your thesis holds)."""

        return base_prompt

    @property
    def rebuttal_prompt(self) -> str:
        """System prompt for rebuttal phase."""
        base = f"""You are an investment analyst arguing the BEAR CASE for {self.ticker}.

The bull has presented their case. Your job is to rebut their weakest points.

RULES:
- DIRECTLY QUOTE each bull point you're addressing
- Then explain why it's overstated or incorrect
- You MAY concede strong points ("I acknowledge X, but...")
- Focus on the 2-3 weakest bull arguments
- Keep your rebuttal under 300 words"""

        if self._user_hints:
            hints_text = "\n".join(f"- {h}" for h in self._user_hints)
            base += f"""

USER HINTS (incorporate these points):
{hints_text}"""

        return base

    @property
    def tools(self) -> list[dict[str, Any]]:
        """Return tools for research phase."""
        all_tools = get_tools_for_claude()
        research_tools = [
            "sec_search", "sec_fetch", "rag_search",
            "scrape_yahoo", "compute_ratios", "search_news", "web_search",
        ]
        return [t for t in all_tools if t["name"] in research_tools]

    async def run(self, task: str, **kwargs: Any) -> AgentResult:
        """
        Execute bear agent's task.

        Args:
            task: The task description (research or opening or rebuttal)

        Returns:
            AgentResult with bear's argument
        """
        self.tool_results = []
        log("agent", f"BearAgent starting: {self.ticker}")

        messages = [{"role": "user", "content": task}]
        total_tokens = 0

        try:
            for iteration in range(self.max_iterations):
                log("agent", f"BearAgent iteration {iteration + 1}/{self.max_iterations}")

                response = await self._call_claude(messages)
                total_tokens += response.usage.input_tokens + response.usage.output_tokens

                # Extract text and tool uses
                text_content = ""
                tool_uses = []

                for block in response.content:
                    if block.type == "text":
                        text_content += block.text
                    elif block.type == "tool_use":
                        tool_uses.append(block)

                # No more tool calls - we're done
                if not tool_uses:
                    log("agent", f"BearAgent completed with {len(self.tool_results)} tool results")
                    return AgentResult(
                        content=text_content,
                        tool_results=self.tool_results,
                        ticker=self.ticker,
                        success=True,
                        tokens_used=total_tokens,
                    )

                # Execute tool calls
                tool_results = []
                for tool_use in tool_uses:
                    result = await self._execute_tool(tool_use.name, tool_use.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": self._format_tool_result(result),
                    })

                # Update messages for next iteration
                messages.append({
                    "role": "assistant",
                    "content": response.content,
                })
                messages.append({
                    "role": "user",
                    "content": tool_results,
                })

            # Max iterations reached
            log("agent", f"BearAgent hit max iterations for {self.ticker}")
            return AgentResult(
                content=text_content or "Bear case incomplete - max iterations reached.",
                tool_results=self.tool_results,
                ticker=self.ticker,
                success=True,
                tokens_used=total_tokens,
            )

        except Exception as e:
            log("agent", f"BearAgent error: {e}", level="error")
            return AgentResult(
                content="",
                tool_results=self.tool_results,
                ticker=self.ticker,
                success=False,
                error=str(e),
                tokens_used=total_tokens,
            )

    async def run_opening(self, research_summary: str) -> AgentResult:
        """
        Generate opening argument based on research.

        Args:
            research_summary: Summary of tool results from research phase

        Returns:
            AgentResult with bear's opening argument
        """
        task = f"""Based on this research data, present your BEAR CASE for {self.ticker}.

RESEARCH DATA:
{research_summary}

INSTRUCTIONS:
- Present exactly 3-5 of your strongest bearish points
- Cite sources inline (e.g., "P/E of 70x (Yahoo Finance)")
- Format each point as a bullet starting with the key concern
- Do NOT request more data - use only the research provided above
- Keep total response under 400 words"""

        # Opening phase: NO TOOLS - must use provided research only
        self.tool_results = []
        messages = [{"role": "user", "content": task}]

        try:
            # Direct API call WITHOUT tools to prevent tool-use attempts
            response = await self.client.messages.create(
                model=self.config.model,
                max_tokens=2048,
                system=self.system_prompt,
                messages=messages,
                # NO tools parameter - forces text-only response
            )
            text_content = "".join(
                block.text for block in response.content if block.type == "text"
            )
            tokens_used = response.usage.input_tokens + response.usage.output_tokens

            return AgentResult(
                content=text_content,
                tool_results=[],
                ticker=self.ticker,
                success=True,
                tokens_used=tokens_used,
            )

        except Exception as e:
            log("agent", f"BearAgent opening error: {e}", level="error")
            return AgentResult(
                content="",
                ticker=self.ticker,
                success=False,
                error=str(e),
            )

    async def run_rebuttal(self, own_opening: str, opponent_opening: str) -> AgentResult:
        """
        Generate rebuttal to bull's opening.

        Args:
            own_opening: Bear's opening argument
            opponent_opening: Bull's opening argument to rebut

        Returns:
            AgentResult with bear's rebuttal
        """
        task = f"""YOUR OPENING ARGUMENT:
{own_opening}

BULL'S OPENING ARGUMENT:
{opponent_opening}

INSTRUCTIONS:
- Rebut the bull's 2-3 weakest points
- DIRECTLY QUOTE each bull point before countering (e.g., 'Bull claimed "X". However...')
- You MAY concede strong points ("I acknowledge X, but...")
- Do NOT request more data - argue based on what's presented
- Keep response under 300 words"""

        # Rebuttal phase: NO TOOLS
        self.tool_results = []
        messages = [{"role": "user", "content": task}]

        try:
            # Direct API call WITHOUT tools
            response = await self.client.messages.create(
                model=self.config.model,
                max_tokens=1500,
                system=self.rebuttal_prompt,
                messages=messages,
                # NO tools parameter
            )
            text_content = "".join(
                block.text for block in response.content if block.type == "text"
            )
            tokens_used = response.usage.input_tokens + response.usage.output_tokens

            # Clear hints after use
            self._user_hints = []

            return AgentResult(
                content=text_content,
                tool_results=[],
                ticker=self.ticker,
                success=True,
                tokens_used=tokens_used,
            )

        except Exception as e:
            log("agent", f"BearAgent rebuttal error: {e}", level="error")
            return AgentResult(
                content="",
                ticker=self.ticker,
                success=False,
                error=str(e),
            )
