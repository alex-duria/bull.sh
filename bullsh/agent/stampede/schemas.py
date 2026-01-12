"""
Pydantic schemas for the Stampede agent architecture.

Defines data models for all phases:
- Understanding: Query comprehension with confidence scoring
- Planning: Task decomposition with dependencies
- Execution: Task results with status tracking
- Reflection: Completeness evaluation with guidance
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# =============================================================================
# Enums
# =============================================================================


class TaskType(str, Enum):
    """Type of task to execute."""

    USE_TOOLS = "use_tools"  # Fetch data with tools
    REASON = "reason"  # LLM analysis/synthesis only


class TaskStatus(str, Enum):
    """Execution status of a task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"  # Skipped due to dependency failure


class InferredDepth(str, Enum):
    """Inferred research depth from query."""

    QUICK = "quick"  # Simple lookup, 0 tasks
    STANDARD = "standard"  # Normal research, 2-5 tasks
    DEEP = "deep"  # Thorough analysis, 5-10 tasks


class ExportIntent(str, Enum):
    """User's export intention."""

    NONE = "none"
    EXCEL = "excel"
    PDF = "pdf"
    DOCX = "docx"


# =============================================================================
# Understanding Phase
# =============================================================================


class Understanding(BaseModel):
    """
    Result of the Understanding phase.

    Extracts intent, entities, and context from user query.
    If confidence < 0.8, should ask clarifying question.
    """

    # Core extraction
    intent: str = Field(
        description="Primary intent: research, valuation, quick_lookup, thesis, etc."
    )
    tickers: list[str] = Field(
        default_factory=list,
        description="Stock ticker symbols extracted from query",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in understanding. If < 0.8, ask clarification.",
    )

    # Depth inference
    inferred_depth: InferredDepth = Field(
        default=InferredDepth.STANDARD,
        description="Inferred research depth from query phrasing",
    )

    # Research focus
    timeframe: str | None = Field(
        default=None,
        description="Timeframe mentioned: 'last 3 years', 'Q3 2024', etc.",
    )
    metrics_focus: list[str] = Field(
        default_factory=list,
        description="Specific metrics to focus on: revenue, margins, growth, etc.",
    )
    wants_sentiment: bool = Field(
        default=False,
        description="User wants social sentiment analysis",
    )
    wants_factors: bool = Field(
        default=False,
        description="User wants factor/quantitative analysis",
    )
    export_intent: ExportIntent = Field(
        default=ExportIntent.NONE,
        description="User's export intention",
    )

    # Session context
    prior_tickers_researched: list[str] = Field(
        default_factory=list,
        description="Tickers already researched in this session",
    )
    prior_frameworks_used: list[str] = Field(
        default_factory=list,
        description="Frameworks already used in this session",
    )

    # Optional: clarification needed
    clarification_question: str | None = Field(
        default=None,
        description="Question to ask user if confidence < 0.8",
    )

    @property
    def needs_clarification(self) -> bool:
        """Check if we should ask for clarification."""
        return self.confidence < 0.8

    @property
    def is_simple_query(self) -> bool:
        """Check if this is a simple lookup (skip task planning)."""
        return self.inferred_depth == InferredDepth.QUICK


# =============================================================================
# Planning Phase
# =============================================================================


class Task(BaseModel):
    """
    A single task in the execution plan.

    Tasks can depend on other tasks and are executed with
    dependency-aware parallelism.
    """

    id: str = Field(description="Unique task ID: task_1, task_2, etc.")
    description: str = Field(
        max_length=100,
        description="Brief description of what this task does",
    )
    task_type: TaskType = Field(description="Type of task execution")
    depends_on: list[str] = Field(
        default_factory=list,
        description="IDs of tasks that must complete first",
    )
    rationale: str | None = Field(
        default=None,
        description="Why this task is needed (for complex tasks)",
    )
    status: TaskStatus = Field(
        default=TaskStatus.PENDING,
        description="Current execution status",
    )


class TaskPlan(BaseModel):
    """
    Complete execution plan from the Planning phase.

    Contains 0-10 tasks with dependencies.
    0 tasks means simple query, go straight to synthesis.
    """

    summary: str = Field(
        description="Brief summary of the research plan",
    )
    selected_frameworks: list[str] = Field(
        default_factory=list,
        description="Frameworks the planner selected: piotroski, porter, valuation",
    )
    tasks: list[Task] = Field(
        default_factory=list,
        max_length=10,
        description="1-10 tasks to execute",
    )
    is_simple_query: bool = Field(
        default=False,
        description="If True and 0 tasks, go straight to synthesis",
    )

    @property
    def task_count(self) -> int:
        """Number of tasks in the plan."""
        return len(self.tasks)

    def get_task(self, task_id: str) -> Task | None:
        """Get a task by ID."""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    def get_ready_tasks(self) -> list[Task]:
        """Get tasks with all dependencies satisfied."""
        completed_ids = {t.id for t in self.tasks if t.status == TaskStatus.COMPLETED}
        return [
            t
            for t in self.tasks
            if t.status == TaskStatus.PENDING and all(dep in completed_ids for dep in t.depends_on)
        ]

    def has_pending_tasks(self) -> bool:
        """Check if there are tasks still pending."""
        return any(t.status == TaskStatus.PENDING for t in self.tasks)


# =============================================================================
# Execution Phase
# =============================================================================


class TaskResult(BaseModel):
    """
    Result from executing a single task.

    Contains the data gathered and execution metadata.
    """

    task_id: str = Field(description="ID of the task that was executed")
    status: TaskStatus = Field(description="Final status of execution")
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Data gathered by the task",
    )
    tool_calls: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Tools called during execution",
    )
    error: str | None = Field(
        default=None,
        description="Error message if task failed",
    )
    retried: bool = Field(
        default=False,
        description="Whether this task was retried after initial failure",
    )
    duration_ms: int = Field(
        default=0,
        description="Execution duration in milliseconds",
    )
    completed_at: datetime | None = Field(
        default=None,
        description="When the task completed",
    )

    @property
    def succeeded(self) -> bool:
        """Check if task completed successfully."""
        return self.status == TaskStatus.COMPLETED

    def to_prompt_text(self) -> str:
        """Format result for inclusion in LLM prompt."""
        if self.status == TaskStatus.FAILED:
            return f"[{self.task_id}] FAILED: {self.error or 'Unknown error'}"

        if self.status == TaskStatus.SKIPPED:
            return f"[{self.task_id}] SKIPPED (dependency failed)"

        lines = [f"[{self.task_id}] Completed"]
        for key, value in self.data.items():
            value_str = str(value)
            if len(value_str) > 1000:
                value_str = value_str[:1000] + "... [truncated]"
            lines.append(f"  {key}: {value_str}")

        return "\n".join(lines)


# =============================================================================
# Reflection Phase
# =============================================================================


class ReflectionResult(BaseModel):
    """
    Result from the Reflection phase.

    Evaluates whether we have enough data to answer the query,
    or if another iteration is needed.
    """

    is_complete: bool = Field(
        description="Whether we have enough data to answer the query",
    )
    reasoning: str = Field(
        description="Explanation of why complete or incomplete",
    )
    missing_info: list[str] = Field(
        default_factory=list,
        description="Specific data that is missing (if incomplete)",
    )
    guidance: str = Field(
        default="",
        description="Guidance for the next planning iteration",
    )

    def to_display_text(self) -> str:
        """Format for user display."""
        status = "Complete" if self.is_complete else "Incomplete"
        lines = [
            f"Status: {status}",
            f"Reason: {self.reasoning}",
        ]
        if not self.is_complete and self.guidance:
            lines.append(f"Guidance: {self.guidance}")
        return "\n".join(lines)


# =============================================================================
# Loop State
# =============================================================================


class StampedeState(BaseModel):
    """
    Complete state of a Stampede execution.

    Tracks all phases across iterations for debugging and resume.
    """

    query: str = Field(description="Original user query")
    understanding: Understanding | None = Field(
        default=None,
        description="Result of Understanding phase",
    )
    iteration: int = Field(
        default=0,
        description="Current iteration number (1-5)",
    )
    max_iterations: int = Field(
        default=5,
        description="Maximum iterations allowed",
    )

    # Per-iteration history
    plans: list[TaskPlan] = Field(
        default_factory=list,
        description="Plan from each iteration",
    )
    all_task_results: dict[str, TaskResult] = Field(
        default_factory=dict,
        description="All task results across iterations",
    )
    reflections: list[ReflectionResult] = Field(
        default_factory=list,
        description="Reflection from each iteration",
    )

    # Final state
    is_complete: bool = Field(
        default=False,
        description="Whether the loop completed successfully",
    )
    final_guidance: str = Field(
        default="",
        description="Final guidance from last reflection",
    )

    @property
    def current_plan(self) -> TaskPlan | None:
        """Get the most recent plan."""
        return self.plans[-1] if self.plans else None

    @property
    def current_reflection(self) -> ReflectionResult | None:
        """Get the most recent reflection."""
        return self.reflections[-1] if self.reflections else None

    def add_plan(self, plan: TaskPlan) -> None:
        """Add a new plan for the current iteration."""
        self.plans.append(plan)

    def add_task_result(self, result: TaskResult) -> None:
        """Add a task result."""
        self.all_task_results[result.task_id] = result

    def add_reflection(self, reflection: ReflectionResult) -> None:
        """Add a reflection result."""
        self.reflections.append(reflection)
        self.is_complete = reflection.is_complete
        self.final_guidance = reflection.guidance


# =============================================================================
# Synthesis Phase
# =============================================================================


class SynthesisContext(BaseModel):
    """
    Context passed to the Synthesis phase.

    Contains everything needed to generate the final response.
    """

    query: str = Field(description="Original user query")
    understanding: Understanding = Field(description="Query understanding")
    task_results: dict[str, TaskResult] = Field(
        description="All task results",
    )
    selected_frameworks: list[str] = Field(
        default_factory=list,
        description="Frameworks used in analysis",
    )
    iteration_count: int = Field(
        default=1,
        description="Number of iterations taken",
    )


# =============================================================================
# Progress Events (for UI)
# =============================================================================


class ProgressEventType(str, Enum):
    """Types of progress events for UI updates."""

    UNDERSTANDING = "understanding"
    PLANNING = "planning"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    REFLECTING = "reflecting"
    REPLANNING = "replanning"
    SYNTHESIZING = "synthesizing"
    COMPLETE = "complete"


class ProgressEvent(BaseModel):
    """
    Progress event for UI updates.

    Emitted during execution to update the terminal UI.
    """

    event_type: ProgressEventType
    message: str = Field(description="Human-readable progress message")
    iteration: int = Field(default=1, description="Current iteration")
    task_id: str | None = Field(default=None, description="Related task ID")
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional event data",
    )
