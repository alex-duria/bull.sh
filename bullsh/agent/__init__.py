"""Agent module - orchestrator and subagents."""

from bullsh.agent.orchestrator import Orchestrator, TokenLimitExceeded, TokenUsage
from bullsh.agent.base import SubAgent, AgentResult
from bullsh.agent.research import ResearchAgent
from bullsh.agent.compare import CompareAgent
from bullsh.agent.bull import BullAgent
from bullsh.agent.bear import BearAgent
from bullsh.agent.moderator import ModeratorAgent, SynthesisResult
from bullsh.agent.debate import DebateCoordinator, DebatePhase, DebateState, DebateRefused

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
]
