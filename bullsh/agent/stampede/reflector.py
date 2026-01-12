"""
Reflection phase for Stampede agent architecture.

Evaluates task results and decides if more work is needed.
Philosophy: DEFAULT TO COMPLETE.
"""

import json
from typing import Any

from anthropic import AsyncAnthropic

from bullsh.config import Config, get_config
from bullsh.logging import log

from .schemas import (
    ReflectionResult,
    TaskResult,
    Understanding,
)

REFLECTION_PROMPT = """You are evaluating research completeness. Your job is to decide if we have enough data to answer the user's query.

## CRITICAL: DEFAULT TO COMPLETE

Mark incomplete ONLY if:
- Critical data for the PRIMARY entity is genuinely missing
- You CANNOT answer the CORE question at all
- A required framework signal cannot be computed

"Nice-to-have" enrichment does NOT justify another iteration.
Prefer pragmatic answers over perfect ones.

Remember: We have 14 tools, but more tools ≠ better answer.
If you have the key data, STOP and let synthesis happen.

## Query Context
Original query: {query}
Intent: {intent}
Tickers: {tickers}
Inferred depth: {depth}
Selected frameworks: {frameworks}

## Current Iteration
Iteration: {iteration} of {max_iterations}

## Task Results Summary
{task_results_summary}

## Evaluation Criteria

For "quick" depth queries:
- Just need the specific data point requested
- Almost always complete after 1 iteration

For "standard" depth queries:
- Need basic financial data and context
- Should complete in 1-2 iterations

For "deep" depth queries:
- May need comprehensive data across multiple sources
- But still: 3 iterations maximum in most cases

For framework-specific queries:
- Piotroski: Need enough data to compute most signals (not all 9 required)
- Porter: Need enough to assess most forces (not all 5 required)
- Valuation: Need at least 2-3 valuation methods

## Your Response

Evaluate if we can answer the query with the data we have.

If COMPLETE:
- is_complete: true
- reasoning: Why the data is sufficient
- missing_info: [] (empty)
- guidance: "" (empty)

If INCOMPLETE (use sparingly):
- is_complete: false
- reasoning: What CRITICAL data is missing
- missing_info: ["specific item 1", "specific item 2"]
- guidance: "Focused guidance for next iteration"

Respond with JSON:
{{
    "is_complete": true/false,
    "reasoning": "...",
    "missing_info": [...],
    "guidance": "..."
}}"""


class ReflectionAgent:
    """
    Agent that evaluates task results and decides if more work is needed.

    Philosophy: Default to complete. Only iterate if critical data missing.
    """

    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        self.client = AsyncAnthropic(api_key=self.config.anthropic_api_key)

    async def reflect(
        self,
        query: str,
        understanding: Understanding,
        task_results: dict[str, TaskResult],
        iteration: int,
        max_iterations: int = 5,
        selected_frameworks: list[str] | None = None,
    ) -> ReflectionResult:
        """
        Evaluate if task results are sufficient to answer the query.

        Args:
            query: Original user query
            understanding: Query understanding
            task_results: Results from all tasks
            iteration: Current iteration number
            max_iterations: Maximum allowed iterations
            selected_frameworks: Frameworks being used

        Returns:
            ReflectionResult indicating completeness and guidance
        """
        log("stampede", f"Reflecting on {len(task_results)} task results (iteration {iteration})")

        # Early completion for simple queries
        if understanding.is_simple_query:
            log("stampede", "Simple query - auto-complete")
            return ReflectionResult(
                is_complete=True,
                reasoning="Simple query - direct answer possible",
                missing_info=[],
                guidance="",
            )

        # Force completion on last iteration
        if iteration >= max_iterations:
            log("stampede", f"Max iterations ({max_iterations}) reached - forcing complete")
            return ReflectionResult(
                is_complete=True,
                reasoning=f"Maximum iterations ({max_iterations}) reached. Proceeding with available data.",
                missing_info=[],
                guidance="",
            )

        # Build task results summary
        results_summary = self._summarize_task_results(task_results)

        # Build the prompt
        prompt = REFLECTION_PROMPT.format(
            query=query,
            intent=understanding.intent,
            tickers=", ".join(understanding.tickers) or "None",
            depth=understanding.inferred_depth.value,
            frameworks=", ".join(selected_frameworks or []) or "None",
            iteration=iteration,
            max_iterations=max_iterations,
            task_results_summary=results_summary,
        )

        try:
            response = await self.client.messages.create(
                model=self.config.model,
                max_tokens=1024,
                system="You are a research completeness evaluator. Respond only with valid JSON.",
                messages=[{"role": "user", "content": prompt}],
            )

            # Parse response
            response_text = response.content[0].text
            reflection_dict = self._parse_json_response(response_text)

            reflection = ReflectionResult(
                is_complete=reflection_dict.get("is_complete", True),
                reasoning=reflection_dict.get("reasoning", "Evaluation complete"),
                missing_info=reflection_dict.get("missing_info", []),
                guidance=reflection_dict.get("guidance", ""),
            )

            log(
                "stampede",
                f"Reflection: is_complete={reflection.is_complete}, "
                f"reasoning={reflection.reasoning[:50]}...",
            )

            return reflection

        except Exception as e:
            log("stampede", f"Reflection error: {e}", level="error")
            # On error, default to complete (pragmatic)
            return ReflectionResult(
                is_complete=True,
                reasoning=f"Reflection error ({e}), proceeding with available data",
                missing_info=[],
                guidance="",
            )

    def _summarize_task_results(self, task_results: dict[str, TaskResult]) -> str:
        """Summarize task results for the reflection prompt."""
        if not task_results:
            return "No tasks executed yet."

        lines = []

        # Count successes and failures
        successes = sum(1 for r in task_results.values() if r.succeeded)
        failures = len(task_results) - successes

        lines.append(f"Tasks: {successes} succeeded, {failures} failed")
        lines.append("")

        for task_id, result in task_results.items():
            if result.succeeded:
                # Show what data was collected
                data_keys = list(result.data.keys())
                data_preview = self._preview_data(result.data)
                lines.append(f"✓ {task_id}:")
                lines.append(f"  Data keys: {data_keys}")
                lines.append(f"  Preview: {data_preview}")
            else:
                lines.append(f"✗ {task_id}: FAILED - {result.error}")

            lines.append("")

        return "\n".join(lines)

    def _preview_data(self, data: dict[str, Any], max_length: int = 200) -> str:
        """Create a preview of collected data."""
        if not data:
            return "No data"

        previews = []
        for key, value in data.items():
            if key == "analysis":
                # For analysis, show first sentence
                text = str(value)[:100]
                previews.append(f"{key}: {text}...")
            elif isinstance(value, dict):
                previews.append(f"{key}: {{{len(value)} keys}}")
            elif isinstance(value, list):
                previews.append(f"{key}: [{len(value)} items]")
            else:
                value_str = str(value)[:50]
                previews.append(f"{key}: {value_str}")

        result = "; ".join(previews)
        if len(result) > max_length:
            result = result[:max_length] + "..."

        return result

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


def format_reflection_for_display(reflection: ReflectionResult, iteration: int) -> str:
    """Format reflection result for terminal display."""
    status = "[green]Complete[/green]" if reflection.is_complete else "[yellow]Incomplete[/yellow]"

    lines = [
        "",
        "[bold]Reflection:[/bold]",
        "━" * 50,
        f"Status: {status}",
        f"Reason: {reflection.reasoning}",
    ]

    if not reflection.is_complete:
        if reflection.missing_info:
            lines.append(f"Missing: {', '.join(reflection.missing_info)}")
        if reflection.guidance:
            lines.append(f"Guidance: {reflection.guidance}")
        lines.append(f"[dim]Replanning (iteration {iteration + 1})...[/dim]")

    lines.append("━" * 50)

    return "\n".join(lines)
