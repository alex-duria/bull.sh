"""Agent orchestrator - lightweight dispatcher for subagents."""

import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from anthropic import Anthropic

# Stampede integration
from bullsh.agent.stampede import StampedeLoop
from bullsh.config import Config
from bullsh.logging import log, log_api_call, log_tool_call, log_tool_result
from bullsh.tools.base import ToolResult, ToolStatus, get_tools_for_claude


class TokenLimitExceeded(Exception):
    """Raised when token/cost limit is exceeded."""

    pass


@dataclass
class TokenUsage:
    """Track token usage for cost control."""

    input_tokens: int = 0
    output_tokens: int = 0
    api_calls: int = 0
    cache_read_tokens: int = 0  # Tokens read from cache (90% discount)
    cache_creation_tokens: int = 0  # Tokens written to cache

    def add(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_read: int = 0,
        cache_creation: int = 0,
    ) -> None:
        """Add tokens from an API response."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cache_read_tokens += cache_read
        self.cache_creation_tokens += cache_creation
        self.api_calls += 1

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def estimated_cost(self, cost_per_1k_input: float, cost_per_1k_output: float) -> float:
        """Calculate estimated cost in USD, accounting for cache discounts."""
        # Cache reads are 90% cheaper, cache creation is 25% more expensive
        regular_input = self.input_tokens - self.cache_read_tokens - self.cache_creation_tokens
        cache_read_cost = (self.cache_read_tokens / 1000) * cost_per_1k_input * 0.1
        cache_create_cost = (self.cache_creation_tokens / 1000) * cost_per_1k_input * 1.25
        regular_input_cost = (regular_input / 1000) * cost_per_1k_input
        output_cost = (self.output_tokens / 1000) * cost_per_1k_output
        return cache_read_cost + cache_create_cost + regular_input_cost + output_cost

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate as percentage."""
        if self.input_tokens == 0:
            return 0.0
        return (self.cache_read_tokens / self.input_tokens) * 100

    def __str__(self) -> str:
        cache_info = ""
        if self.cache_read_tokens > 0:
            cache_info = f", {self.cache_hit_rate:.0f}% cached"
        return f"{self.total_tokens:,} tokens ({self.input_tokens:,} in, {self.output_tokens:,} out{cache_info})"


@dataclass
class AgentMessage:
    """A message in the conversation."""

    role: str  # "user" or "assistant"
    content: str | list[dict[str, Any]]


@dataclass
class Orchestrator:
    """
    Lightweight orchestrator that dispatches to subagents.

    Responsibilities:
    - Parse user intent
    - Select appropriate subagent (research, compare, thesis)
    - Pass selective context to subagents
    - Weave results into unified response
    - Handle graceful degradation

    With use_stampede=True, uses the new Plan→Execute→Reflect loop
    for research queries.
    """

    config: Config
    verbose: bool = False
    history: list[AgentMessage] = field(default_factory=list)
    _client: Anthropic | None = field(default=None, repr=False)

    # Session for artifact tracking (optional, set by REPL)
    session: Any = field(default=None, repr=False)

    # Token tracking
    session_usage: TokenUsage = field(default_factory=TokenUsage)
    turn_usage: TokenUsage = field(default_factory=TokenUsage)
    _warned_session: bool = field(default=False, repr=False)
    _warned_turn: bool = field(default=False, repr=False)

    # Stampede integration
    use_stampede: bool = True  # Enable new Plan→Execute→Reflect loop
    _stampede: StampedeLoop | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._client = Anthropic(api_key=self.config.anthropic_api_key)
        if self.use_stampede:
            self._stampede = StampedeLoop(config=self.config)

    @property
    def stampede(self) -> StampedeLoop:
        """Get or create the StampedeLoop instance."""
        if self._stampede is None:
            self._stampede = StampedeLoop(config=self.config)
        return self._stampede

    @property
    def client(self) -> Anthropic:
        if self._client is None:
            self._client = Anthropic(api_key=self.config.anthropic_api_key)
        return self._client

    async def chat(
        self,
        user_message: str,
        framework: str | None = None,
        system_override: str | None = None,
    ) -> AsyncIterator[str]:
        """
        Process a user message and yield response chunks.

        This is the main entry point for the REPL.

        Args:
            user_message: The user's message
            framework: Optional framework name for guided analysis
            system_override: Optional complete system prompt override
                             (bypasses normal system prompt building)
        """
        # Reset per-turn tracking
        self.turn_usage = TokenUsage()
        self._warned_turn = False

        # Check session limit before starting
        if self.session_usage.total_tokens >= self.config.max_tokens_per_session:
            raise TokenLimitExceeded(
                f"Session token limit reached ({self.session_usage.total_tokens:,} / "
                f"{self.config.max_tokens_per_session:,}). "
                f"Estimated cost: ${self.get_session_cost():.2f}. "
                "Start a new session to continue."
            )

        self.history.append(AgentMessage(role="user", content=user_message))

        # Check for comparison request - dispatch to parallel subagent
        # (Skip if using system_override - factor session handles its own flow)
        if not system_override:
            comparison_tickers = self._detect_comparison(user_message)
            if comparison_tickers and len(comparison_tickers) >= 2:
                log("orchestrator", f"Detected comparison request for {comparison_tickers}")
                async for chunk in self._run_comparison(user_message, comparison_tickers):
                    yield chunk
                return

        # Use Stampede for research queries (when enabled and no override)
        if self.use_stampede and not system_override:
            log("orchestrator", "Using Stampede loop")
            async for chunk in self._run_stampede(user_message, framework):
                yield chunk
            return

        # Fallback to legacy flow (system_override or stampede disabled)
        # Build system prompt (or use override)
        system_prompt = system_override or self._build_system_prompt(framework)

        # Convert history to Claude format
        messages = self._history_to_messages()

        # Call Claude with streaming
        async for chunk in self._stream_response(system_prompt, messages, framework):
            yield chunk

    def get_session_cost(self) -> float:
        """Get estimated session cost in USD."""
        return self.session_usage.estimated_cost(
            self.config.cost_per_1k_input,
            self.config.cost_per_1k_output,
        )

    def get_usage_summary(self) -> str:
        """Get a human-readable usage summary."""
        cost = self.get_session_cost()
        return (
            f"Session: {self.session_usage} | "
            f"Est. cost: ${cost:.4f} | "
            f"API calls: {self.session_usage.api_calls}"
        )

    async def _stream_response(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        framework: str | None,
    ) -> AsyncIterator[str]:
        """Stream response from Claude, handling tool calls."""
        accumulated_response = ""
        # Allow enough iterations for comprehensive research:
        # Yahoo (1) + SEC search (1) + SEC fetch (2-3) + Social (1) + Framework analysis (2-3) + Synthesis
        max_iterations = 15

        for iteration in range(max_iterations):  # noqa: B007 - iteration used after loop
            # Check turn limit before making another API call
            if self.turn_usage.total_tokens >= self.config.max_tokens_per_turn:
                yield f"\n\n⚠️  Turn token limit reached ({self.turn_usage.total_tokens:,} tokens). Response truncated.\n"
                break

            # Build system prompt with optional caching
            if self.config.enable_prompt_caching:
                system_content = [
                    {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}
                ]
            else:
                system_content = system_prompt

            # Create message with streaming
            with self.client.messages.stream(
                model=self.config.model,
                max_tokens=4096,
                system=system_content,
                tools=get_tools_for_claude(),
                messages=messages,
                extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
                if self.config.enable_prompt_caching
                else {},
            ) as stream:
                current_text = ""

                for event in stream:
                    if hasattr(event, "type"):
                        if event.type == "content_block_delta" and hasattr(event.delta, "text"):
                            current_text += event.delta.text
                            yield event.delta.text
                        elif (
                            event.type == "content_block_start"
                            and hasattr(event.content_block, "type")
                            and event.content_block.type == "tool_use"
                        ):
                            # Tool call starting - handled in response processing below
                            pass

                # Get final message
                response = stream.get_final_message()

            # Track token usage
            if hasattr(response, "usage"):
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                # Capture cache metrics if available
                cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
                cache_creation = getattr(response.usage, "cache_creation_input_tokens", 0) or 0

                self.turn_usage.add(input_tokens, output_tokens, cache_read, cache_creation)
                self.session_usage.add(input_tokens, output_tokens, cache_read, cache_creation)

                # Log API call
                log_api_call(self.config.model, input_tokens, output_tokens, cache_read)

                # Check for warnings
                warning = self._check_usage_warnings()
                if warning:
                    yield warning

            # Check if we need to handle tool calls
            if response.stop_reason == "tool_use":
                # Process tool calls
                from bullsh.ui.status import format_tool_result, format_tool_start

                tool_results = []

                for block in response.content:
                    if block.type == "tool_use":
                        # Show tool call status with pretty formatting
                        yield "\n" + format_tool_start(block.name, block.input)

                        # Log tool call
                        log_tool_call(block.name, block.input)

                        # Execute tool
                        result = await self._execute_tool(block.name, block.input)

                        # Log tool result
                        log_tool_result(
                            block.name,
                            result.status.value,
                            result.confidence,
                            result.error_message,
                        )

                        # Show result status
                        yield (
                            format_tool_result(
                                block.name,
                                result.status.value,
                                result.confidence,
                                result.cached,
                                result.error_message,
                            )
                            + "\n"
                        )

                        # Capture artifact if this tool generated a file
                        if self.session is not None:
                            from bullsh.storage.artifacts import extract_artifact_from_result

                            artifact = extract_artifact_from_result(result, self.session)
                            if artifact:
                                log("orchestrator", f"Registered artifact: {artifact.filename}")

                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result.to_prompt_text(),
                            }
                        )

                # Add assistant response and tool results to messages
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

                accumulated_response += current_text
                continue

            # No more tool calls - we're done
            accumulated_response += current_text
            break

        # If we exhausted max_iterations, add a note
        if iteration >= max_iterations - 1:
            yield "\n\n⚠️ Maximum tool iterations reached. Please ask a follow-up question for more details.\n"

        # Safeguard: if we did tool calls but got no text response, the model didn't synthesize
        if not accumulated_response.strip() and iteration > 0:
            fallback_msg = "[Data gathered but synthesis incomplete. Please ask: 'summarize the data' or 'what did you find?']"
            yield f"\n\n{fallback_msg}\n"
            accumulated_response = fallback_msg

        # Update history with final response (only if non-empty to avoid API errors)
        if accumulated_response.strip():
            self.history.append(AgentMessage(role="assistant", content=accumulated_response))

    def _check_usage_warnings(self) -> str | None:
        """Check token usage and return warning message if needed."""
        warnings = []

        # Check session warning threshold
        session_pct = self.session_usage.total_tokens / self.config.max_tokens_per_session
        if session_pct >= self.config.warn_at_token_pct and not self._warned_session:
            self._warned_session = True
            cost = self.get_session_cost()
            warnings.append(
                f"\n⚠️  Session at {session_pct:.0%} of token limit "
                f"({self.session_usage.total_tokens:,} / {self.config.max_tokens_per_session:,}). "
                f"Est. cost: ${cost:.2f}"
            )

        # Check turn warning threshold
        turn_pct = self.turn_usage.total_tokens / self.config.max_tokens_per_turn
        if turn_pct >= self.config.warn_at_token_pct and not self._warned_turn:
            self._warned_turn = True
            warnings.append(
                f"\n⚠️  Turn at {turn_pct:.0%} of limit "
                f"({self.turn_usage.total_tokens:,} / {self.config.max_tokens_per_turn:,})"
            )

        return "\n".join(warnings) if warnings else None

    async def _execute_tool(self, name: str, params: dict[str, Any]) -> ToolResult:
        """Execute a tool and return the result."""
        # Import tools lazily to avoid circular imports
        from bullsh.tools import excel, news, rag, sec, social, yahoo
        from bullsh.tools import thesis as thesis_tool

        match name:
            case "sec_search":
                return await sec.sec_search(
                    params["ticker"],
                    params.get("fuzzy", True),
                )
            case "sec_fetch":
                return await sec.sec_fetch(
                    params["ticker"],
                    params["filing_type"],
                    params.get("year"),
                    params.get("section"),
                )
            case "rag_search":
                return await rag.rag_search(
                    params["query"],
                    params.get("ticker"),
                    params.get("form"),
                    params.get("year"),
                    params.get("k", 5),
                )
            case "rag_list":
                return await rag.rag_list(params.get("ticker"))
            case "search_stocktwits":
                return await social.search_stocktwits(params["symbol"])
            case "search_reddit":
                return await social.search_reddit(
                    params["query"],
                    params.get("subreddits"),
                )
            case "scrape_yahoo":
                return await yahoo.scrape_yahoo(params["ticker"])
            case "search_news":
                return await news.search_news(
                    params["query"],
                    params.get("days_back", 30),
                )
            case "web_search":
                return await news.web_search(
                    params["query"],
                    params.get("max_results", 10),
                )
            case "compute_ratios":
                return await yahoo.compute_ratios(params["ticker"])
            case "generate_excel":
                return await excel.generate_excel(
                    params["ticker"],
                    params.get("include_ratios", True),
                    params.get("compare_tickers"),
                )
            case "save_thesis":
                return await thesis_tool.save_thesis(
                    params["ticker"],
                    params["content"],
                    params.get("filename"),
                )
            case "calculate_factors":
                from bullsh.tools import factors as factors_tool

                return await factors_tool.calculate_factors(
                    params["ticker"],
                    params.get("peers", []),
                    params.get("factors"),
                )
            case "run_factor_regression":
                from bullsh.tools import factors as factors_tool

                return await factors_tool.run_factor_regression_tool(
                    params["ticker"],
                    params.get("window_months", 36),
                )
            case "get_financials":
                from bullsh.tools import financials

                return await financials.get_financials(
                    params["ticker"],
                    params.get("statement_type", "all"),
                    params.get("period", "annual"),
                    params.get("years", 3),
                )
            case "get_insider_transactions":
                from bullsh.tools import insiders

                return await insiders.get_insider_transactions(
                    params["ticker"],
                    params.get("limit", 50),
                )
            case _:
                return ToolResult(
                    data={},
                    confidence=0.0,
                    status=ToolStatus.FAILED,
                    tool_name=name,
                    error_message=f"Unknown tool: {name}",
                )

    def _build_system_prompt(self, framework: str | None) -> str:
        """Build the system prompt, optionally with framework guidance."""
        base_prompt = """You are an investment research agent. Your job is to help users research companies and build investment theses.

You have access to these tools:
- sec_search: Find what SEC filings are available for a company
- sec_fetch: Download and read 10-K or 10-Q filings (auto-indexes for RAG search)
- rag_search: Semantic search over indexed SEC filings - use this to find specific information
- rag_list: List what filings are indexed and available for RAG search
- search_stocktwits: Search StockTwits for sentiment (primary social source)
- search_reddit: Search Reddit for discussions (fallback if StockTwits fails)
- scrape_yahoo: Get analyst ratings and price targets
- search_news: Search recent financial news
- web_search: **IMPORTANT** General web search - use this to fill gaps when other tools fail or data is outdated
- compute_ratios: Calculate key financial ratios (P/E, EV/EBITDA, growth rates)
- calculate_factors: **REQUIRED for factor analysis** Compute real factor z-scores (value, momentum, quality, growth, size, volatility)
- run_factor_regression: Run Fama-French regression to decompose returns into factor exposures
- save_thesis: Save research to a thesis document

**RAG SEARCH - USE THIS FIRST FOR FILING QUESTIONS:**
When users ask questions about content in SEC filings (risks, revenue, strategy, competition, etc.):
1. ALWAYS use rag_search FIRST - this searches the indexed 10-K and 10-Q filings
2. Use SEC section names for better results: "Item 1A Risk Factors", "Item 7 MD&A", "Item 1 Business"
3. If results aren't relevant, VARY YOUR QUERIES - this is RAG, try different terms:
   - "risk factors" → "principal risks" → "Item 1A"
   - "revenue growth" → "net sales increase" → "total revenue year over year"
   - "competition" → "competitive landscape" → "market position"
4. Do NOT use web_search for information that's in the filings - it's already indexed!
5. web_search is for CURRENT data (stock price, news, analyst estimates) NOT filing content

**FACTOR ANALYSIS GUARDRAIL:**
When users ask about factor exposures, factor tilts, or multi-factor analysis, you MUST call calculate_factors.
Do NOT theorize about factors or describe them conceptually - compute the actual z-scores.
The tool does pure Python math and returns real numbers. Always use it.

**EFFICIENT TOOL USAGE:**
- Make MULTIPLE tool calls in parallel when possible to gather data faster
- Example: Call scrape_yahoo AND sec_search AND search_stocktwits in ONE response
- Don't wait for one tool to complete before calling unrelated tools
- After 2-3 rounds of data gathering, SYNTHESIZE - don't keep fetching

When researching a company:
1. PARALLEL: scrape_yahoo + sec_search + search_stocktwits (get market data, filings list, and sentiment together)
2. THEN: sec_fetch for the most recent 10-K or 10-Q (this auto-indexes for RAG)
3. THEN: rag_search for specific questions about the filing content
4. FINALLY: Synthesize findings into analysis - don't keep calling tools

**CRITICAL: Filling Data Gaps**
- Use web_search for CURRENT data: stock prices, recent news, analyst estimates
- Do NOT use web_search for filing content - use rag_search instead
- If rag_search returns no results, check if the filing is indexed with rag_list first

**CRITICAL: ALWAYS COMPLETE YOUR RESPONSE**
- Every response MUST end with analysis and conclusions, not just tool calls
- After gathering data, you MUST synthesize and present findings to the user
- If a user sends a short or frustrated follow-up, they're telling you that you failed to complete your previous response - use the data you already have to provide the analysis they're waiting for
- Cached data means you already have the information - USE IT, don't re-fetch
- Your primary job is ANALYSIS and INSIGHT, not data collection. Tools are means to an end.

Data quality:
- Each tool returns a confidence score (0-1). Be skeptical of low-confidence data.
- If a source fails once, try web_search. If that fails too, move on.
- Indicate when data is cached vs fresh.

Presentation:
- Display ratios in a summary table when appropriate
- Be direct and analytical. Cite specific numbers.
- Flag risks prominently. Never recommend buying or selling.
- ALWAYS end with a clear summary or conclusion

The user is building an investment thesis through conversation.
Remember context from earlier in the conversation."""

        if framework == "piotroski":
            base_prompt += """

FRAMEWORK: Piotroski F-Score
You are applying the Piotroski F-Score framework - a 9-point quantitative assessment.

Calculate and present each signal:
PROFITABILITY (0-4 points):
- ROA > 0 (+1)
- Operating Cash Flow > 0 (+1)
- ROA increasing YoY (+1)
- Cash Flow > Net Income (+1)

LEVERAGE & LIQUIDITY (0-3 points):
- Long-term debt decreasing (+1)
- Current ratio increasing (+1)
- No new shares issued (+1)

EFFICIENCY (0-2 points):
- Gross margin increasing (+1)
- Asset turnover increasing (+1)

Present the score breakdown clearly. Help the user interpret what each signal means for this specific company. Let them explore and challenge your assessments."""

        elif framework == "porter":
            base_prompt += """

FRAMEWORK: Porter's Five Forces
You are applying Porter's Five Forces competitive analysis framework.

Analyze each force by extracting evidence from the 10-K:

1. THREAT OF NEW ENTRANTS
   - Barriers to entry (capital, regulation, IP)
   - Economies of scale
   - Brand loyalty

2. SUPPLIER POWER
   - Supplier concentration
   - Switching costs
   - Forward integration threat

3. BUYER POWER
   - Buyer concentration
   - Price sensitivity
   - Backward integration threat

4. THREAT OF SUBSTITUTES
   - Availability of alternatives
   - Price-performance tradeoffs
   - Switching costs

5. COMPETITIVE RIVALRY
   - Number and size of competitors
   - Industry growth rate
   - Exit barriers

Rate each force (LOW/MODERATE/HIGH) with evidence. Let the user challenge your assessments - they may have insights you don't. Track which forces have been analyzed.

**WHEN COMPLETE:** Present a summary table and offer to export: "Framework complete. Use /export to save as PDF/DOCX, or /excel for spreadsheet.\""""

        elif framework == "valuation":
            base_prompt += """

FRAMEWORK: Valuation Analysis
You are performing a multi-method valuation to generate price targets.

Calculate implied prices using these methods:

1. P/E MULTIPLE VALUATION
   - Get current EPS and sector average P/E
   - Implied Price = EPS × Sector P/E
   - Note: Use TTM earnings, not forward

2. FORWARD P/E VALUATION
   - Get forward EPS estimates
   - Apply sector P/E with 5-10% haircut for uncertainty
   - Implied Price = Forward EPS × Adjusted P/E

3. EV/EBITDA VALUATION
   - Calculate or fetch EBITDA
   - Apply sector EV/EBITDA multiple
   - Convert enterprise value to equity value per share

4. ANALYST CONSENSUS
   - Fetch analyst price targets (low, mean, high)
   - Note number of analysts and recommendation

5. GROWTH-ADJUSTED (PEG)
   - Fair P/E ≈ Growth Rate (PEG = 1)
   - Implied Price = EPS × Growth Rate
   - Cap P/E at 40x for high-growth companies

PRESENT YOUR RESULTS AS:
| Method | Target | Upside | Confidence |
|--------|--------|--------|------------|
| P/E Multiple | $XXX | +XX% | High/Med/Low |
| ... | ... | ... | ... |

THEN PROVIDE:
- **Bear Case:** Lowest reasonable target (conservative assumptions)
- **Base Case:** Median of methods
- **Bull Case:** Highest reasonable target (optimistic assumptions)

END WITH: Clear verdict (Undervalued/Fairly Valued/Overvalued) and key assumptions.

**WHEN COMPLETE:** Offer to export: "Valuation complete. Use /export to save as PDF/DOCX, or /excel for spreadsheet.\""""

        # Add export reminder for all frameworks
        if framework and framework in ("piotroski", "porter", "valuation", "pitch"):
            base_prompt += """

**EXPORT REMINDER:** When you complete the framework analysis, remind the user they can export with:
- /export thesis.pdf - Save as PDF
- /export report.docx - Save as Word document
- /excel - Generate Excel spreadsheet with data"""

        # Inject artifacts section if session has artifacts
        if self.session is not None:
            from bullsh.storage.artifacts import ArtifactRegistry

            registry = ArtifactRegistry(self.session)
            artifacts_section = registry.to_prompt_section()
            if artifacts_section:
                base_prompt += f"\n\n{artifacts_section}"

        return base_prompt

    def _history_to_messages(self) -> list[dict[str, Any]]:
        """Convert internal history to Claude message format with sliding window."""
        # Apply sliding window to prevent unbounded context growth
        max_messages = self.config.max_history_messages
        history_to_use = self.history

        if len(self.history) > max_messages:
            # Keep first message (often contains important context) + last N-1
            history_to_use = [self.history[0]] + list(self.history[-(max_messages - 1) :])

            # Add a note about truncated history
            if len(history_to_use) > 1:
                truncated_count = len(self.history) - max_messages
                history_to_use.insert(
                    1,
                    AgentMessage(
                        role="user",
                        content=f"[Note: {truncated_count} earlier messages omitted to save context]",
                    ),
                )
                history_to_use.insert(
                    2,
                    AgentMessage(
                        role="assistant",
                        content="Understood, I'll continue based on the available context.",
                    ),
                )

        return [{"role": msg.role, "content": msg.content} for msg in history_to_use]

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.history = []

    def _detect_comparison(self, message: str) -> list[str] | None:
        """
        Detect if the user is asking for a comparison and extract tickers.

        Returns list of tickers if comparison detected, None otherwise.
        """
        message_lower = message.lower()

        # Check for comparison keywords
        comparison_keywords = [
            "compare",
            "vs",
            "versus",
            "comparison",
            "which is better",
            "side by side",
            "head to head",
            "stack up",
            "compete",
        ]

        if not any(kw in message_lower for kw in comparison_keywords):
            return None

        # Extract potential tickers (1-5 uppercase letters)
        ticker_pattern = re.compile(r"\b([A-Z]{1,5})\b")
        potential_tickers = ticker_pattern.findall(message)

        # Filter out common words
        stop_words = {
            "I",
            "A",
            "THE",
            "AND",
            "OR",
            "FOR",
            "TO",
            "IN",
            "ON",
            "AT",
            "IS",
            "IT",
            "AS",
            "BY",
            "BE",
            "VS",
            "DO",
            "IF",
            "SO",
            "NO",
            "AN",
            "OF",
            "UP",
            "GO",
            "MY",
            "WE",
            "HE",
            "ME",
            "US",
        }
        tickers = [t for t in potential_tickers if t not in stop_words and len(t) >= 2]

        # Need at least 2 tickers for comparison
        if len(tickers) >= 2:
            return tickers[:5]  # Limit to 5 for performance

        return None

    async def _run_comparison(
        self,
        message: str,
        tickers: list[str],
    ) -> AsyncIterator[str]:
        """
        Run parallel comparison using CompareAgent.

        Args:
            message: Original user message
            tickers: List of tickers to compare

        Yields:
            Response chunks
        """
        from bullsh.agent.compare import CompareAgent

        log("orchestrator", f"Dispatching to CompareAgent for {', '.join(tickers)}")
        yield f"\n🔄 Running parallel comparison for: {', '.join(tickers)}\n"

        try:
            agent = CompareAgent(
                config=self.config,
                max_iterations=3,
                research_iterations=4,
            )

            result = await agent.run(task=message, tickers=tickers)

            # Track token usage from subagent (estimate 60% input, 40% output)
            input_est = int(result.tokens_used * 0.6)
            output_est = result.tokens_used - input_est
            self.turn_usage.add(input_est, output_est)
            self.session_usage.add(input_est, output_est)

            if result.success:
                yield f"\n✅ Parallel research complete ({len(tickers)} companies analyzed)\n\n"
                yield result.content

                # Update history
                self.history.append(AgentMessage(role="assistant", content=result.content))
            else:
                yield f"\n⚠️ Comparison incomplete: {result.error}\n"
                if result.content:
                    yield result.content

        except Exception as e:
            log("orchestrator", f"CompareAgent error: {e}", level="error")
            yield f"\n❌ Comparison failed: {e}\n"
            yield "Falling back to standard analysis...\n"
            # Fall through to standard processing will be handled by caller

    async def _run_stampede(
        self,
        message: str,
        framework: str | None = None,
    ) -> AsyncIterator[str]:
        """
        Run the Stampede Plan→Execute→Reflect loop.

        Args:
            message: User message
            framework: Optional framework name

        Yields:
            Response chunks including progress and final answer
        """
        log("orchestrator", f"Running Stampede for: {message[:50]}...")

        accumulated_response = ""

        try:
            async for chunk in self.stampede.run(message, show_progress=True):
                accumulated_response += chunk
                yield chunk

            # Sync resolved tickers from Stampede to session
            # This ensures company names like "Tesla" are stored as "TSLA"
            if self.session is not None and hasattr(self.session, "tickers"):
                for ticker in self.stampede.prior_tickers:
                    if ticker not in self.session.tickers:
                        self.session.tickers.append(ticker)

            # Update history with final response
            if accumulated_response.strip():
                self.history.append(AgentMessage(role="assistant", content=accumulated_response))

        except Exception as e:
            log("orchestrator", f"Stampede error: {e}", level="error")
            yield f"\n❌ Stampede error: {e}\n"
            yield "Falling back to legacy flow...\n"

            # Fall back to legacy flow
            system_prompt = self._build_system_prompt(framework)
            messages = self._history_to_messages()

            async for chunk in self._stream_response(system_prompt, messages, framework):
                yield chunk
