"""Agent module - orchestrator and subagents."""

from bullsh.agent.base import AgentResult, SubAgent
from bullsh.agent.bear import BearAgent
from bullsh.agent.bull import BullAgent
from bullsh.agent.compare import CompareAgent
from bullsh.agent.debate import DebateCoordinator, DebatePhase, DebateRefused, DebateState
from bullsh.agent.moderator import ModeratorAgent, SynthesisResult
from bullsh.agent.orchestrator import Orchestrator, TokenLimitExceeded, TokenUsage
from bullsh.agent.research import ResearchAgent

# Stampede - next-gen agent architecture
from bullsh.agent.stampede import (
    ReflectionResult,
    StampedeLoop,
    Task,
    TaskPlan,
    TaskResult,
    Understanding,
    run_stampede,
)

__all__ = [
    "Orchestrator",
    "TokenLimitExceeded",
    "TokenUsage",
    "SubAgent",
    "AgentResult",
    "ResearchAgent",
    "CompareAgent",
    # Debate agents
    "BullAgent",
    "BearAgent",
    "ModeratorAgent",
    "SynthesisResult",
    "DebateCoordinator",
    "DebatePhase",
    "DebateState",
    "DebateRefused",
    # Stampede
    "StampedeLoop",
    "run_stampede",
    "Understanding",
    "TaskPlan",
    "Task",
    "TaskResult",
    "ReflectionResult",
]
