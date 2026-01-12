"""
Stampede: Bull.sh Next-Gen Agent Architecture.

Plan -> Execute -> Reflect loop with smart tool selection,
framework-aware task planning, and visible progress.
"""

from bullsh.agent.stampede.executor import TaskExecutor
from bullsh.agent.stampede.loop import StampedeLoop, run_stampede
from bullsh.agent.stampede.planner import PlanningAgent
from bullsh.agent.stampede.reflector import ReflectionAgent
from bullsh.agent.stampede.schemas import (
    ExportIntent,
    InferredDepth,
    ProgressEvent,
    ProgressEventType,
    ReflectionResult,
    StampedeState,
    SynthesisContext,
    Task,
    TaskPlan,
    TaskResult,
    TaskStatus,
    TaskType,
    Understanding,
)
from bullsh.agent.stampede.synthesizer import Synthesizer
from bullsh.agent.stampede.understanding import UnderstandingAgent

__all__ = [
    # Enums
    "TaskType",
    "TaskStatus",
    "InferredDepth",
    "ExportIntent",
    "ProgressEventType",
    # Phase schemas
    "Understanding",
    "Task",
    "TaskPlan",
    "TaskResult",
    "ReflectionResult",
    "SynthesisContext",
    # State and events
    "StampedeState",
    "ProgressEvent",
    # Agents
    "UnderstandingAgent",
    "PlanningAgent",
    "TaskExecutor",
    "ReflectionAgent",
    "Synthesizer",
    # Main loop
    "StampedeLoop",
    "run_stampede",
]
