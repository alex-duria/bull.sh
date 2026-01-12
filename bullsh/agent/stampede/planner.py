"""
Planning phase for Stampede agent architecture.

Creates structured task plans with dependencies based on:
- Understanding of the query
- Framework requirements
- Prior task results (if replanning)
- Guidance from reflection
"""

import json
from typing import Any

from anthropic import AsyncAnthropic

from bullsh.config import Config, get_config
from bullsh.logging import log

from .schemas import (
    InferredDepth,
    Task,
    TaskPlan,
    TaskResult,
    TaskType,
    Understanding,
)

PLANNING_PROMPT = """You are a research planning agent. Create a structured task plan to answer the user's query.

## Query Understanding
Intent: {intent}
Tickers: {tickers}
Depth: {depth}
Metrics focus: {metrics_focus}
Wants sentiment: {wants_sentiment}
Wants factors: {wants_factors}
Export intent: {export_intent}
Timeframe: {timeframe}

## Session Context
Previously researched: {prior_tickers}
Previously used frameworks: {prior_frameworks}

## Task Planning Rules

### Task Types
- **use_tools**: Fetch data using tools (SEC filings, Yahoo, sentiment, etc.)
- **reason**: Analyze/synthesize data without tool calls (pure LLM reasoning)

### Task Count by Depth
- **quick**: 0-1 tasks (or skip planning entirely for simple lookups)
- **standard**: 2-5 tasks
- **deep**: 5-10 tasks

### Framework-Aware Planning
{framework_guidance}

### Dependencies
- Tasks can depend on other tasks using `depends_on`
- Independent tasks can run in parallel
- Example: "Calculate growth rates" depends on "Fetch financials"

### Data Source Priority
1. **Structured Financials**: Financial Datasets API (if available) → SEC EDGAR → Yahoo
2. **Full Filings**: SEC EDGAR (10-K, 10-Q text)
3. **Real-time Data**: Yahoo Finance
4. **Sentiment**: StockTwits, Reddit
5. **Insider Activity**: Financial Datasets API

### RAG-First Policy
For any question about SEC filing content:
- ALWAYS try rag_search FIRST (searches indexed filings)
- Only fetch fresh filings if not indexed

### Session Memory
{session_memory_guidance}

## Prior Results (if replanning)
{prior_results}

## Reflection Guidance (if replanning)
{reflection_guidance}

## User Query
{query}

Create a task plan. Respond with JSON matching the TaskPlan schema:
{{
    "summary": "Brief description of the plan",
    "selected_frameworks": ["framework1", "framework2"],
    "is_simple_query": false,
    "tasks": [
        {{
            "id": "task_1",
            "description": "Brief task description (max 100 chars)",
            "task_type": "use_tools",
            "depends_on": [],
            "rationale": "Why this task is needed"
        }}
    ]
}}

For simple lookups (single data point), set is_simple_query=true and tasks=[]."""


FRAMEWORK_PROMPTS = {
    "piotroski": """
**Piotroski F-Score Framework Selected**
This framework requires 9 signals. Create tasks to gather:
- ROA and operating cash flow (profitability)
- YoY changes in ROA, cash flow, debt, current ratio
- Share issuance history
- Gross margin and asset turnover changes

Suggested tasks:
1. Fetch latest financials (income statement, balance sheet, cash flow)
2. Fetch prior year financials for YoY comparison
3. Calculate profitability signals (4 tasks or combine)
4. Calculate leverage/liquidity signals (3 tasks or combine)
5. Calculate efficiency signals (2 tasks or combine)
6. Synthesize F-Score and interpretation
""",
    "porter": """
**Porter's Five Forces Framework Selected**
Analyze competitive dynamics through 5 forces:
1. Threat of new entrants
2. Supplier power
3. Buyer power
4. Threat of substitutes
5. Competitive rivalry

Suggested tasks:
1. Fetch 10-K Business section (Item 1)
2. RAG search for competitive landscape
3. RAG search for supplier/customer relationships
4. Analyze barriers to entry
5. Synthesize five forces assessment
""",
    "valuation": """
**Valuation Framework Selected**
Calculate price targets using multiple methods:
1. P/E multiple valuation
2. Forward P/E valuation
3. EV/EBITDA valuation
4. Analyst consensus
5. PEG-based fair value

Suggested tasks:
1. Fetch current financials and multiples
2. Fetch analyst estimates
3. Calculate sector average multiples
4. Compute implied prices for each method
5. Synthesize valuation range
""",
}


class PlanningAgent:
    """
    Agent that creates structured task plans.

    Uses Claude to analyze the query understanding and generate
    a plan with tasks and dependencies.
    """

    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        self.client = AsyncAnthropic(api_key=self.config.anthropic_api_key)

    async def create_plan(
        self,
        query: str,
        understanding: Understanding,
        prior_results: dict[str, TaskResult] | None = None,
        guidance: str | None = None,
    ) -> TaskPlan:
        """
        Create a structured task plan.

        Args:
            query: Original user query
            understanding: Parsed understanding of the query
            prior_results: Results from prior iteration (if replanning)
            guidance: Guidance from reflection (if replanning)

        Returns:
            TaskPlan with tasks and dependencies
        """
        log(
            "stampede",
            f"Planning for intent={understanding.intent}, depth={understanding.inferred_depth.value}",
        )

        # Check for simple query shortcut
        if understanding.is_simple_query:
            log("stampede", "Simple query detected, returning 0-task plan")
            return TaskPlan(
                summary=f"Direct answer for: {query[:50]}...",
                selected_frameworks=[],
                tasks=[],
                is_simple_query=True,
            )

        # Build framework guidance
        framework_guidance = self._build_framework_guidance(understanding)

        # Build session memory guidance
        session_memory = self._build_session_memory_guidance(understanding)

        # Build prior results summary
        prior_results_text = self._format_prior_results(prior_results)

        # Build the prompt
        prompt = PLANNING_PROMPT.format(
            intent=understanding.intent,
            tickers=", ".join(understanding.tickers) or "None specified",
            depth=understanding.inferred_depth.value,
            metrics_focus=", ".join(understanding.metrics_focus) or "General",
            wants_sentiment="Yes" if understanding.wants_sentiment else "No",
            wants_factors="Yes" if understanding.wants_factors else "No",
            export_intent=understanding.export_intent.value,
            timeframe=understanding.timeframe or "Not specified",
            prior_tickers=", ".join(understanding.prior_tickers_researched) or "None",
            prior_frameworks=", ".join(understanding.prior_frameworks_used) or "None",
            framework_guidance=framework_guidance,
            session_memory_guidance=session_memory,
            prior_results=prior_results_text,
            reflection_guidance=guidance or "No prior guidance (first iteration).",
            query=query,
        )

        try:
            response = await self.client.messages.create(
                model=self.config.model,
                max_tokens=2048,
                system="You are a research planning assistant. Respond only with valid JSON.",
                messages=[{"role": "user", "content": prompt}],
            )

            # Parse response
            response_text = response.content[0].text
            plan_dict = self._parse_json_response(response_text)

            # Build TaskPlan
            plan = self._build_plan(plan_dict)

            log(
                "stampede",
                f"Created plan: {len(plan.tasks)} tasks, frameworks={plan.selected_frameworks}",
            )

            return plan

        except Exception as e:
            log("stampede", f"Planning error: {e}", level="error")
            # Return a minimal fallback plan
            return self._create_fallback_plan(query, understanding)

    def _build_framework_guidance(self, understanding: Understanding) -> str:
        """Build framework-specific planning guidance."""
        # Check if a specific framework is requested
        intent = understanding.intent.lower()

        if "piotroski" in intent or "f-score" in intent:
            return FRAMEWORK_PROMPTS["piotroski"]
        elif "porter" in intent or "five forces" in intent:
            return FRAMEWORK_PROMPTS["porter"]
        elif "valuation" in intent or "price target" in intent:
            return FRAMEWORK_PROMPTS["valuation"]
        elif understanding.inferred_depth == InferredDepth.DEEP:
            # For deep analysis, suggest comprehensive approach
            return """
**Deep Analysis Mode**
For thorough research, consider:
1. Financial fundamentals (income, balance sheet, cash flow)
2. Competitive analysis
3. Sentiment analysis
4. Factor analysis (if quantitative focus)
5. Synthesis and thesis generation
"""
        else:
            return "No specific framework. Plan based on query requirements."

    def _build_session_memory_guidance(self, understanding: Understanding) -> str:
        """Build guidance based on session memory."""
        if not understanding.prior_tickers_researched:
            return "Fresh session - no prior research to leverage."

        lines = ["Leverage prior research:"]
        for ticker in understanding.prior_tickers_researched:
            lines.append(f"- {ticker} was already researched, skip re-fetching unless stale")

        return "\n".join(lines)

    def _format_prior_results(
        self,
        prior_results: dict[str, TaskResult] | None,
    ) -> str:
        """Format prior task results for the prompt."""
        if not prior_results:
            return "No prior results (first iteration)."

        lines = ["Prior iteration results:"]
        for task_id, result in prior_results.items():
            status = "✓" if result.succeeded else "✗"
            lines.append(f"- {task_id} [{status}]: {list(result.data.keys())[:5]}")

        return "\n".join(lines)

    def _parse_json_response(self, response_text: str) -> dict[str, Any]:
        """Parse JSON from Claude's response."""
        text = response_text.strip()

        # Extract from markdown code block if present
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()

        return json.loads(text)

    def _build_plan(self, data: dict[str, Any]) -> TaskPlan:
        """Build TaskPlan from parsed JSON."""
        tasks = []

        for task_data in data.get("tasks", []):
            # Parse task type
            task_type_str = task_data.get("task_type", "use_tools")
            try:
                task_type = TaskType(task_type_str)
            except ValueError:
                task_type = TaskType.USE_TOOLS

            task = Task(
                id=task_data.get("id", f"task_{len(tasks) + 1}"),
                description=task_data.get("description", "Unnamed task")[:100],
                task_type=task_type,
                depends_on=task_data.get("depends_on", []),
                rationale=task_data.get("rationale"),
            )
            tasks.append(task)

        return TaskPlan(
            summary=data.get("summary", "Research plan"),
            selected_frameworks=data.get("selected_frameworks", []),
            tasks=tasks,
            is_simple_query=data.get("is_simple_query", False),
        )

    def _create_fallback_plan(
        self,
        query: str,
        understanding: Understanding,
    ) -> TaskPlan:
        """Create a fallback plan when planning fails."""
        tasks = []

        # Add basic tasks based on understanding
        if understanding.tickers:
            ticker = understanding.tickers[0]

            tasks.append(
                Task(
                    id="task_1",
                    description=f"Fetch market data for {ticker}",
                    task_type=TaskType.USE_TOOLS,
                    depends_on=[],
                )
            )

            if understanding.inferred_depth != InferredDepth.QUICK:
                tasks.append(
                    Task(
                        id="task_2",
                        description=f"Search SEC filings for {ticker}",
                        task_type=TaskType.USE_TOOLS,
                        depends_on=[],
                    )
                )

                tasks.append(
                    Task(
                        id="task_3",
                        description="Synthesize findings",
                        task_type=TaskType.REASON,
                        depends_on=["task_1", "task_2"],
                    )
                )

        return TaskPlan(
            summary=f"Fallback plan for: {query[:50]}...",
            selected_frameworks=[],
            tasks=tasks,
            is_simple_query=len(tasks) == 0,
        )


def format_plan_for_display(plan: TaskPlan) -> str:
    """Format a TaskPlan for terminal display."""
    if plan.is_simple_query:
        return "[dim]Simple query - direct answer[/dim]"

    lines = [
        "",
        f"[bold]Plan:[/bold] {plan.summary}",
        "━" * 50,
    ]

    if plan.selected_frameworks:
        lines.append(f"[dim]Frameworks:[/dim] {', '.join(plan.selected_frameworks)}")

    lines.append("[dim]Tasks:[/dim]")

    for i, task in enumerate(plan.tasks, 1):
        deps = ""
        if task.depends_on:
            dep_nums = [t.replace("task_", "") for t in task.depends_on]
            deps = f" [dim](depends: {', '.join(dep_nums)})[/dim]"

        type_icon = "🔧" if task.task_type == TaskType.USE_TOOLS else "💭"
        lines.append(f"  [{i}] {type_icon} {task.description}{deps}")

    lines.append("━" * 50)

    return "\n".join(lines)
