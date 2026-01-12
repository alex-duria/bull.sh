"""
Task Executor for Stampede agent architecture.

Executes tasks with:
- Dependency-aware parallelism
- Dynamic tool selection
- RAG-first policy
- Retry on failure
"""

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

from anthropic import AsyncAnthropic

from bullsh.config import Config, get_config
from bullsh.logging import log
from bullsh.tools.base import ToolResult, ToolStatus, get_tools_for_claude

from .schemas import (
    ProgressEvent,
    ProgressEventType,
    Task,
    TaskPlan,
    TaskResult,
    TaskStatus,
    TaskType,
    Understanding,
)

EXECUTOR_SYSTEM_PROMPT = """You are a research execution agent. Execute the given task by calling the appropriate tools.

## Available Tools
You have access to these research tools:
- sec_search: Find SEC filings available for a company
- sec_fetch: Download and parse 10-K or 10-Q filings
- rag_search: Semantic search over indexed SEC filings (USE FIRST for filing content)
- rag_list: List indexed filings
- search_stocktwits: Get StockTwits sentiment
- search_reddit: Get Reddit discussions
- scrape_yahoo: Get analyst ratings, price targets, key stats
- search_news: Search recent financial news
- web_search: General web search for current data
- compute_ratios: Calculate financial ratios
- calculate_factors: Compute factor z-scores
- run_factor_regression: Fama-French regression
- generate_excel: Create Excel export
- get_financials: Get structured financial statements (unified tool)
- get_insider_transactions: Get insider buy/sell activity

## RAG-First Policy
For ANY question about SEC filing content:
1. ALWAYS try rag_search FIRST
2. Only fetch fresh filings if not indexed
3. Use SEC section names: "Item 1A Risk Factors", "Item 7 MD&A"

## Task Context
Understanding: {understanding}
Prior results available: {prior_result_keys}

## Current Task
ID: {task_id}
Description: {task_description}
Type: {task_type}

Execute this task. Call the appropriate tools to gather the required data.
For 'reason' type tasks, analyze the provided data without tool calls."""


class TaskExecutor:
    """
    Executes tasks with dependency-aware parallelism.

    Handles tool calls, retries, and progress reporting.
    """

    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        self.client = AsyncAnthropic(api_key=self.config.anthropic_api_key)

    async def execute_plan(
        self,
        plan: TaskPlan,
        understanding: Understanding,
        abort_event: asyncio.Event | None = None,
    ) -> AsyncIterator[ProgressEvent | TaskResult]:
        """
        Execute all tasks in a plan with dependency-aware parallelism.

        Args:
            plan: The task plan to execute
            understanding: Query understanding for context
            abort_event: Optional event to signal abort

        Yields:
            ProgressEvents and TaskResults as tasks complete
        """
        if plan.is_simple_query or not plan.tasks:
            log("stampede", "No tasks to execute (simple query)")
            return

        task_results: dict[str, TaskResult] = {}

        # Mark all tasks as pending
        for task in plan.tasks:
            task.status = TaskStatus.PENDING

        log("stampede", f"Executing {len(plan.tasks)} tasks")

        while plan.has_pending_tasks():
            # Check for abort
            if abort_event and abort_event.is_set():
                log("stampede", "Execution aborted")
                break

            # Get tasks ready to execute
            ready_tasks = plan.get_ready_tasks()

            if not ready_tasks:
                # No ready tasks but still pending - dependency issue
                log(
                    "stampede",
                    "No ready tasks but pending remain - possible cycle",
                    level="warning",
                )
                break

            # Emit progress for starting tasks
            for task in ready_tasks:
                task.status = TaskStatus.RUNNING
                yield ProgressEvent(
                    event_type=ProgressEventType.TASK_STARTED,
                    message=f"Starting: {task.description}",
                    task_id=task.id,
                )

            # Execute ready tasks in parallel
            results = await asyncio.gather(
                *[
                    self._execute_single_task(task, task_results, understanding)
                    for task in ready_tasks
                ],
                return_exceptions=True,
            )

            # Process results
            for task, result in zip(ready_tasks, results):
                if isinstance(result, Exception):
                    # Task raised an exception
                    task_result = TaskResult(
                        task_id=task.id,
                        status=TaskStatus.FAILED,
                        error=str(result),
                    )
                else:
                    task_result = result

                # Update task status
                task.status = task_result.status
                task_results[task.id] = task_result

                # Emit progress
                if task_result.succeeded:
                    yield ProgressEvent(
                        event_type=ProgressEventType.TASK_COMPLETED,
                        message=f"Completed: {task.description}",
                        task_id=task.id,
                        data={"keys": list(task_result.data.keys())[:5]},
                    )
                else:
                    yield ProgressEvent(
                        event_type=ProgressEventType.TASK_FAILED,
                        message=f"Failed: {task.description} - {task_result.error}",
                        task_id=task.id,
                    )

                # Yield the actual result
                yield task_result

        log("stampede", f"Execution complete: {len(task_results)} tasks")

    async def _execute_single_task(
        self,
        task: Task,
        prior_results: dict[str, TaskResult],
        understanding: Understanding,
    ) -> TaskResult:
        """
        Execute a single task with retry logic.

        Args:
            task: The task to execute
            prior_results: Results from completed tasks
            understanding: Query understanding for context

        Returns:
            TaskResult with data or error
        """
        start_time = datetime.now()

        try:
            if task.task_type == TaskType.REASON:
                result = await self._execute_reasoning_task(task, prior_results, understanding)
            else:
                result = await self._execute_tool_task(task, prior_results, understanding)

            # Set duration
            duration = (datetime.now() - start_time).total_seconds() * 1000
            result.duration_ms = int(duration)
            result.completed_at = datetime.now()

            return result

        except Exception as e:
            log("stampede", f"Task {task.id} error: {e}, retrying...", level="warning")

            # Retry once
            try:
                if task.task_type == TaskType.REASON:
                    result = await self._execute_reasoning_task(task, prior_results, understanding)
                else:
                    result = await self._execute_tool_task(task, prior_results, understanding)

                result.retried = True
                duration = (datetime.now() - start_time).total_seconds() * 1000
                result.duration_ms = int(duration)
                result.completed_at = datetime.now()

                return result

            except Exception as retry_error:
                log("stampede", f"Task {task.id} retry failed: {retry_error}", level="error")
                duration = (datetime.now() - start_time).total_seconds() * 1000

                return TaskResult(
                    task_id=task.id,
                    status=TaskStatus.FAILED,
                    error=f"Failed after retry: {retry_error}",
                    retried=True,
                    duration_ms=int(duration),
                    completed_at=datetime.now(),
                )

    async def _execute_tool_task(
        self,
        task: Task,
        prior_results: dict[str, TaskResult],
        understanding: Understanding,
    ) -> TaskResult:
        """Execute a task that requires tool calls."""
        # Build context from prior results
        prior_data = self._build_prior_context(prior_results)

        # Build the prompt
        prompt = EXECUTOR_SYSTEM_PROMPT.format(
            understanding=self._format_understanding(understanding),
            prior_result_keys=list(prior_results.keys()) or "None",
            task_id=task.id,
            task_description=task.description,
            task_type=task.task_type.value,
        )

        messages = [
            {
                "role": "user",
                "content": f"Execute this task: {task.description}\n\nPrior data:\n{prior_data}",
            }
        ]

        # Call Claude with tools
        all_tool_calls = []
        collected_data: dict[str, Any] = {}
        max_iterations = 5

        for _ in range(max_iterations):
            response = await self.client.messages.create(
                model=self.config.model,
                max_tokens=2048,
                system=prompt,
                tools=get_tools_for_claude(),
                messages=messages,
            )

            # Check for tool use
            if response.stop_reason == "tool_use":
                tool_results = []

                for block in response.content:
                    if block.type == "tool_use":
                        tool_name = block.name
                        tool_input = block.input

                        log("stampede", f"Task {task.id} calling {tool_name}")

                        # Execute tool
                        tool_result = await self._execute_tool(tool_name, tool_input)

                        all_tool_calls.append(
                            {
                                "tool": tool_name,
                                "input": tool_input,
                                "status": tool_result.status.value,
                                "confidence": tool_result.confidence,
                            }
                        )

                        # Collect data
                        if tool_result.status == ToolStatus.SUCCESS:
                            collected_data[tool_name] = tool_result.data

                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": tool_result.to_prompt_text(),
                            }
                        )

                # Add to messages for next iteration
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
                continue

            # No more tool calls - extract final response
            final_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_text += block.text

            if final_text:
                collected_data["analysis"] = final_text

            break

        return TaskResult(
            task_id=task.id,
            status=TaskStatus.COMPLETED,
            data=collected_data,
            tool_calls=all_tool_calls,
        )

    async def _execute_reasoning_task(
        self,
        task: Task,
        prior_results: dict[str, TaskResult],
        understanding: Understanding,
    ) -> TaskResult:
        """Execute a pure reasoning task without tool calls."""
        # Build context from prior results
        prior_data = self._build_prior_context(prior_results)

        prompt = f"""You are analyzing research data.

Task: {task.description}

Available data from prior tasks:
{prior_data}

Provide your analysis. Be specific and cite the data."""

        response = await self.client.messages.create(
            model=self.config.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )

        analysis = ""
        for block in response.content:
            if hasattr(block, "text"):
                analysis += block.text

        return TaskResult(
            task_id=task.id,
            status=TaskStatus.COMPLETED,
            data={"analysis": analysis},
            tool_calls=[],
        )

    def _build_prior_context(self, prior_results: dict[str, TaskResult]) -> str:
        """Build context string from prior task results."""
        if not prior_results:
            return "No prior data available."

        lines = []
        for task_id, result in prior_results.items():
            if result.succeeded:
                lines.append(f"[{task_id}]:")
                for key, value in result.data.items():
                    value_str = str(value)
                    if len(value_str) > 500:
                        value_str = value_str[:500] + "..."
                    lines.append(f"  {key}: {value_str}")
            else:
                lines.append(f"[{task_id}]: FAILED - {result.error}")

        return "\n".join(lines)

    def _format_understanding(self, understanding: Understanding) -> str:
        """Format understanding for prompt context."""
        return (
            f"Intent: {understanding.intent}, "
            f"Tickers: {understanding.tickers}, "
            f"Depth: {understanding.inferred_depth.value}"
        )

    async def _execute_tool(self, name: str, params: dict[str, Any]) -> ToolResult:
        """Execute a tool and return the result."""
        # Import tools lazily
        from bullsh.tools import excel, news, rag, sec, social, yahoo
        from bullsh.tools import thesis as thesis_tool

        try:
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

        except Exception as e:
            log("stampede", f"Tool {name} error: {e}", level="error")
            return ToolResult(
                data={},
                confidence=0.0,
                status=ToolStatus.FAILED,
                tool_name=name,
                error_message=str(e),
            )


def format_task_progress(plan: TaskPlan, results: dict[str, TaskResult]) -> str:
    """Format task progress for terminal display."""
    lines = [
        "",
        "[bold]Execution Progress:[/bold]",
        "━" * 50,
    ]

    for task in plan.tasks:
        result = results.get(task.id)

        if task.status == TaskStatus.COMPLETED:
            icon = "✓"
            style = "[green]"
        elif task.status == TaskStatus.RUNNING:
            icon = "◐"
            style = "[yellow]"
        elif task.status == TaskStatus.FAILED:
            icon = "✗"
            style = "[red]"
        elif task.status == TaskStatus.SKIPPED:
            icon = "○"
            style = "[dim]"
        else:
            icon = "○"
            style = "[dim]"

        duration = ""
        if result and result.duration_ms:
            duration = f" [dim]({result.duration_ms}ms)[/dim]"

        lines.append(f"  {style}{icon}[/] {task.id}: {task.description}{duration}")

    lines.append("━" * 50)

    return "\n".join(lines)
