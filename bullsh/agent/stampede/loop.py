"""
Main Stampede Loop - orchestrates the Plan→Execute→Reflect cycle.

This is the main entry point for the Stampede agent architecture.
"""

import asyncio
from collections.abc import AsyncIterator

from bullsh.config import Config, get_config
from bullsh.logging import log

from .executor import TaskExecutor, format_task_progress
from .planner import PlanningAgent, format_plan_for_display
from .reflector import ReflectionAgent, format_reflection_for_display
from .schemas import (
    ProgressEvent,
    ProgressEventType,
    StampedeState,
    SynthesisContext,
    TaskResult,
)
from .synthesizer import Synthesizer
from .understanding import UnderstandingAgent


class StampedeLoop:
    """
    Main Stampede orchestration loop.

    Implements: Understand → Plan → Execute → Reflect → Synthesize

    Features:
    - Confidence-based clarification
    - Simple query shortcuts (0-task)
    - Bounded iterations (max 5)
    - Progress visibility
    - Session memory
    """

    def __init__(
        self,
        config: Config | None = None,
        max_iterations: int = 5,
    ):
        self.config = config or get_config()
        self.max_iterations = max_iterations

        # Initialize agents
        self.understanding_agent = UnderstandingAgent(config)
        self.planning_agent = PlanningAgent(config)
        self.executor = TaskExecutor(config)
        self.reflection_agent = ReflectionAgent(config)
        self.synthesizer = Synthesizer(config)

        # Abort signal
        self.abort_event = asyncio.Event()

        # Session state
        self.prior_tickers: list[str] = []
        self.prior_frameworks: list[str] = []

    async def run(
        self,
        query: str,
        show_progress: bool = True,
    ) -> AsyncIterator[str]:
        """
        Run the full Stampede loop.

        Args:
            query: User's query
            show_progress: Whether to yield progress messages

        Yields:
            Progress messages and final response chunks
        """
        log("stampede", f"Starting loop for: {query[:50]}...")

        # Initialize state
        state = StampedeState(
            query=query,
            max_iterations=self.max_iterations,
        )

        # Phase 1: UNDERSTAND
        if show_progress:
            yield "[dim]Understanding query...[/dim]\n"

        understanding = await self.understanding_agent.understand(
            query,
            prior_tickers=self.prior_tickers,
            prior_frameworks=self.prior_frameworks,
        )
        state.understanding = understanding

        # Check for clarification needed
        if understanding.needs_clarification:
            yield f"\n{understanding.clarification_question}\n"
            return

        # Update session memory
        for ticker in understanding.tickers:
            if ticker not in self.prior_tickers:
                self.prior_tickers.append(ticker)

        # Phase 1.5: Simple query shortcut
        if understanding.is_simple_query:
            if show_progress:
                yield "[dim]Simple query - direct answer[/dim]\n\n"

            context = SynthesisContext(
                query=query,
                understanding=understanding,
                task_results={},
                selected_frameworks=[],
            )

            async for chunk in self.synthesizer.synthesize_simple(query, context):
                yield chunk

            state.is_complete = True
            return

        # Phase 2-4: Plan → Execute → Reflect loop
        task_results: dict[str, TaskResult] = {}
        guidance: str | None = None

        for iteration in range(1, self.max_iterations + 1):
            state.iteration = iteration

            # Check for abort
            if self.abort_event.is_set():
                yield "\n[dim]Aborted by user[/dim]\n"
                return

            # Phase 2: PLAN
            if show_progress:
                yield f"\n[dim]Planning (iteration {iteration}/{self.max_iterations})...[/dim]\n"

            plan = await self.planning_agent.create_plan(
                query=query,
                understanding=understanding,
                prior_results=task_results if iteration > 1 else None,
                guidance=guidance,
            )
            state.add_plan(plan)

            # Update session memory with frameworks
            for fw in plan.selected_frameworks:
                if fw not in self.prior_frameworks:
                    self.prior_frameworks.append(fw)

            # Show plan to user
            if show_progress:
                yield format_plan_for_display(plan)

            # Handle simple query detected during planning
            if plan.is_simple_query:
                if show_progress:
                    yield "\n[dim]Simple query - direct answer[/dim]\n\n"

                context = SynthesisContext(
                    query=query,
                    understanding=understanding,
                    task_results=task_results,
                    selected_frameworks=plan.selected_frameworks,
                )

                async for chunk in self.synthesizer.synthesize_simple(query, context):
                    yield chunk

                state.is_complete = True
                return

            # Phase 3: EXECUTE
            if show_progress:
                yield f"\n[dim]Executing {len(plan.tasks)} tasks...[/dim]\n"

            async for event in self.executor.execute_plan(
                plan=plan,
                understanding=understanding,
                abort_event=self.abort_event,
            ):
                if isinstance(event, ProgressEvent):
                    if show_progress:
                        yield self._format_progress_event(event)
                elif isinstance(event, TaskResult):
                    task_results[event.task_id] = event
                    state.add_task_result(event)

            # Show execution summary
            if show_progress:
                yield format_task_progress(plan, task_results)

            # Phase 4: REFLECT
            if show_progress:
                yield "\n[dim]Reflecting...[/dim]\n"

            reflection = await self.reflection_agent.reflect(
                query=query,
                understanding=understanding,
                task_results=task_results,
                iteration=iteration,
                max_iterations=self.max_iterations,
                selected_frameworks=plan.selected_frameworks,
            )
            state.add_reflection(reflection)

            # Show reflection to user
            if show_progress:
                yield format_reflection_for_display(reflection, iteration)

            if reflection.is_complete:
                break

            # Prepare for next iteration
            guidance = reflection.guidance

        # Phase 5: SYNTHESIZE
        if show_progress:
            yield "\n[dim]Synthesizing answer...[/dim]\n\n"

        context = SynthesisContext(
            query=query,
            understanding=understanding,
            task_results=state.all_task_results,
            selected_frameworks=state.current_plan.selected_frameworks
            if state.current_plan
            else [],
            iteration_count=state.iteration,
        )

        async for chunk in self.synthesizer.synthesize(context):
            yield chunk

        state.is_complete = True
        log("stampede", f"Loop complete after {state.iteration} iteration(s)")

    def abort(self) -> None:
        """Signal abort to stop the loop."""
        self.abort_event.set()

    def reset_abort(self) -> None:
        """Reset the abort signal for a new run."""
        self.abort_event.clear()

    def reset_session(self) -> None:
        """Reset session memory for a new session."""
        self.prior_tickers = []
        self.prior_frameworks = []
        self.reset_abort()

    def _format_progress_event(self, event: ProgressEvent) -> str:
        """Format a progress event for display."""
        icons = {
            ProgressEventType.UNDERSTANDING: "🔍",
            ProgressEventType.PLANNING: "📋",
            ProgressEventType.TASK_STARTED: "▶",
            ProgressEventType.TASK_COMPLETED: "✓",
            ProgressEventType.TASK_FAILED: "✗",
            ProgressEventType.REFLECTING: "💭",
            ProgressEventType.REPLANNING: "🔄",
            ProgressEventType.SYNTHESIZING: "✍",
            ProgressEventType.COMPLETE: "✅",
        }

        icon = icons.get(event.event_type, "•")

        if event.event_type == ProgressEventType.TASK_COMPLETED:
            return f"  [green]{icon}[/green] {event.message}\n"
        elif event.event_type == ProgressEventType.TASK_FAILED:
            return f"  [red]{icon}[/red] {event.message}\n"
        elif event.event_type == ProgressEventType.TASK_STARTED:
            return f"  [yellow]{icon}[/yellow] {event.message}\n"
        else:
            return f"  {icon} {event.message}\n"


async def run_stampede(
    query: str,
    config: Config | None = None,
    show_progress: bool = True,
) -> AsyncIterator[str]:
    """
    Convenience function to run Stampede loop.

    Args:
        query: User's query
        config: Optional config override
        show_progress: Whether to show progress messages

    Yields:
        Response chunks
    """
    loop = StampedeLoop(config)

    async for chunk in loop.run(query, show_progress):
        yield chunk


# CLI entry point for testing
async def _test_stampede():
    """Test the Stampede loop from command line."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m bullsh.agent.stampede.loop 'your query'")
        return

    query = " ".join(sys.argv[1:])
    print(f"Query: {query}\n")

    async for chunk in run_stampede(query):
        print(chunk, end="", flush=True)

    print()


if __name__ == "__main__":
    asyncio.run(_test_stampede())
