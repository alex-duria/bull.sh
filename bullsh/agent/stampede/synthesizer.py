"""
Synthesis phase for Stampede agent architecture.

Generates the final response from all task results.
Streams the response for real-time display.
"""

from collections.abc import AsyncIterator
from typing import Any

from anthropic import AsyncAnthropic

from bullsh.config import Config, get_config
from bullsh.logging import log

from .schemas import (
    ExportIntent,
    SynthesisContext,
    TaskResult,
)

SYNTHESIS_PROMPT = """You are synthesizing research findings into a clear, actionable response.

## Query
{query}

## Understanding
Intent: {intent}
Tickers: {tickers}
Depth: {depth}
Timeframe: {timeframe}
Metrics focus: {metrics_focus}

## Research Data
{task_results_formatted}

## Selected Frameworks
{frameworks}

## Synthesis Guidelines

1. **Lead with the KEY FINDING**
   Start with the most important insight in the first sentence.
   Example: "NVDA trades at 45x forward P/E, a 20% premium to peers, justified by its 90%+ data center GPU market share."

2. **Use SPECIFIC NUMBERS with context**
   Don't just cite numbers - explain what they mean.
   Example: "Revenue grew 122% YoY to $26.9B, driven by data center up 209%."

3. **Match the expected DEPTH**
   - Quick: One paragraph, answer the specific question
   - Standard: 2-3 paragraphs with key metrics and context
   - Deep: Comprehensive analysis with multiple sections

4. **Structure for clarity**
   Use headers and bullet points for complex analysis.
   Make it scannable.

5. **Flag RISKS prominently**
   Never recommend buying or selling.
   Present balanced view.

6. **SOURCES at the end**
   Format: ---
   Sources: [task_1: 10-K filing], [task_3: Yahoo Finance], [task_5: StockTwits sentiment]

7. **Export awareness**
   {export_guidance}

## Response Format

For quick queries: Direct answer in 1-2 paragraphs.

For standard queries:
**[Key Finding]**
[2-3 paragraphs of analysis]

---
Sources: [task_1], [task_2], ...

For deep queries:
## Summary
[Key takeaways]

## [Section 1]
[Detailed analysis]

## [Section 2]
[Detailed analysis]

## Risks & Considerations
[Balanced risk assessment]

---
Sources: [task_1], [task_2], ...

Now synthesize the research data into a response."""


class Synthesizer:
    """
    Synthesizes task results into a coherent response.

    Streams the response for real-time display.
    """

    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        self.client = AsyncAnthropic(api_key=self.config.anthropic_api_key)

    async def synthesize(
        self,
        context: SynthesisContext,
    ) -> AsyncIterator[str]:
        """
        Generate final response from task results.

        Args:
            context: Full synthesis context with query, understanding, and results

        Yields:
            Response text chunks for streaming
        """
        log("stampede", f"Synthesizing {len(context.task_results)} task results")

        # Format task results
        results_formatted = self._format_task_results(context.task_results)

        # Build export guidance
        export_guidance = self._get_export_guidance(context.understanding.export_intent)

        # Build the prompt
        prompt = SYNTHESIS_PROMPT.format(
            query=context.query,
            intent=context.understanding.intent,
            tickers=", ".join(context.understanding.tickers) or "None",
            depth=context.understanding.inferred_depth.value,
            timeframe=context.understanding.timeframe or "Not specified",
            metrics_focus=", ".join(context.understanding.metrics_focus) or "General",
            task_results_formatted=results_formatted,
            frameworks=", ".join(context.selected_frameworks) or "None",
            export_guidance=export_guidance,
        )

        try:
            # Use streaming for real-time display
            async with self.client.messages.stream(
                model=self.config.model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                async for text in stream.text_stream:
                    yield text

        except Exception as e:
            log("stampede", f"Synthesis error: {e}", level="error")
            yield f"\n\nSynthesis error: {e}\n\nRaw data available from tasks."

    async def synthesize_simple(
        self,
        query: str,
        context: SynthesisContext,
    ) -> AsyncIterator[str]:
        """
        Synthesize a simple/quick query without full task infrastructure.

        Used for 0-task queries that can be answered directly.

        Args:
            query: Original user query
            context: Synthesis context (may have empty results)

        Yields:
            Response text chunks
        """
        log("stampede", "Simple synthesis (0-task query)")

        prompt = f"""Answer this investment research question directly and concisely.

Question: {query}

Tickers: {", ".join(context.understanding.tickers) or "None specified"}

Provide a brief, factual answer. If you need specific data you don't have, say so.
Don't make up numbers - be honest about what you know vs don't know."""

        try:
            async with self.client.messages.stream(
                model=self.config.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                async for text in stream.text_stream:
                    yield text

        except Exception as e:
            log("stampede", f"Simple synthesis error: {e}", level="error")
            yield f"I couldn't process that request: {e}"

    def _format_task_results(self, task_results: dict[str, TaskResult]) -> str:
        """Format task results for the synthesis prompt."""
        if not task_results:
            return "No task data available."

        lines = []

        for task_id, result in task_results.items():
            lines.append(f"### {task_id}")

            if result.succeeded:
                for key, value in result.data.items():
                    if key == "analysis":
                        # Include full analysis text
                        lines.append(f"**Analysis:**\n{value}")
                    elif isinstance(value, dict):
                        lines.append(f"**{key}:**")
                        for k, v in value.items():
                            v_str = str(v)
                            if len(v_str) > 200:
                                v_str = v_str[:200] + "..."
                            lines.append(f"  - {k}: {v_str}")
                    elif isinstance(value, list):
                        lines.append(f"**{key}:** {len(value)} items")
                        for item in value[:5]:  # First 5 items
                            item_str = str(item)
                            if len(item_str) > 100:
                                item_str = item_str[:100] + "..."
                            lines.append(f"  - {item_str}")
                        if len(value) > 5:
                            lines.append(f"  - ... and {len(value) - 5} more")
                    else:
                        value_str = str(value)
                        if len(value_str) > 500:
                            value_str = value_str[:500] + "..."
                        lines.append(f"**{key}:** {value_str}")
            else:
                lines.append(f"**Status:** FAILED - {result.error}")

            lines.append("")

        return "\n".join(lines)

    def _get_export_guidance(self, export_intent: ExportIntent) -> str:
        """Get export-specific synthesis guidance."""
        if export_intent == ExportIntent.EXCEL:
            return """
User wants Excel export. Structure data for spreadsheet:
- Use clear tables with headers
- Include numerical data in extractable format
- End with: "Use /excel to export this data to a spreadsheet."
"""
        elif export_intent == ExportIntent.PDF:
            return """
User wants PDF export. Structure for document:
- Use clear sections with headers
- Include executive summary at top
- End with: "Use /export to save as PDF."
"""
        elif export_intent == ExportIntent.DOCX:
            return """
User wants Word document. Structure for document:
- Use clear sections with headers
- Include executive summary
- End with: "Use /export to save as Word document."
"""
        else:
            return "No specific export format requested."


async def synthesize_from_results(
    query: str,
    understanding: Any,
    task_results: dict[str, TaskResult],
    selected_frameworks: list[str] | None = None,
    config: Config | None = None,
) -> AsyncIterator[str]:
    """
    Convenience function to synthesize results.

    Args:
        query: Original user query
        understanding: Query understanding
        task_results: All task results
        selected_frameworks: Frameworks used
        config: Optional config override

    Yields:
        Response text chunks
    """
    from .schemas import SynthesisContext

    context = SynthesisContext(
        query=query,
        understanding=understanding,
        task_results=task_results,
        selected_frameworks=selected_frameworks or [],
    )

    synthesizer = Synthesizer(config)

    async for chunk in synthesizer.synthesize(context):
        yield chunk
