"""Session management - save, resume, and list research sessions."""

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from bullsh.config import get_config


@dataclass
class Message:
    """A single message in a conversation."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: str  # ISO format
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        return cls(**data)

    def to_api_format(self) -> dict[str, Any]:
        """Convert to Claude API message format."""
        return {
            "role": self.role,
            "content": self.content,
        }


@dataclass
class Session:
    """A research session with conversation history."""

    id: str
    name: str  # Human-readable name (inferred or user-set)
    tickers: list[str]  # Primary tickers being researched
    framework: str | None  # Framework in use
    created_at: str  # ISO format
    updated_at: str  # ISO format
    messages: list[Message] = field(default_factory=list)
    summary: str | None = None  # Auto-generated summary for long sessions
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "messages": [m.to_dict() for m in self.messages],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        messages = [Message.from_dict(m) for m in data.pop("messages", [])]
        return cls(**data, messages=messages)

    def add_message(
        self,
        role: str,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_results: list[dict[str, Any]] | None = None,
    ) -> Message:
        """Add a message to the session."""
        msg = Message(
            role=role,
            content=content,
            timestamp=datetime.now().isoformat(),
            tool_calls=tool_calls or [],
            tool_results=tool_results or [],
        )
        self.messages.append(msg)
        self.updated_at = datetime.now().isoformat()
        return msg

    def get_conversation_for_api(self, max_messages: int | None = None) -> list[dict[str, Any]]:
        """Get messages formatted for Claude API."""
        messages = self.messages
        if max_messages and len(messages) > max_messages:
            # Keep first message (context) and last N-1 messages
            messages = [messages[0]] + messages[-(max_messages - 1) :]

        return [m.to_api_format() for m in messages]

    def get_display_preview(self, max_length: int = 100) -> str:
        """Get a preview string for session listing."""
        if self.summary:
            return self.summary[:max_length]
        if self.messages:
            first_user_msg = next(
                (m.content for m in self.messages if m.role == "user"),
                "",
            )
            return first_user_msg[:max_length]
        return "Empty session"


class SessionManager:
    """Manage session persistence and retrieval."""

    def __init__(self, sessions_dir: Path | None = None):
        self.sessions_dir = sessions_dir or get_config().sessions_dir
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _session_path(self, session_id: str) -> Path:
        return self.sessions_dir / f"{session_id}.json"

    def _generate_id(self) -> str:
        """Generate a unique session ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"session_{timestamp}"

    def _infer_name(self, messages: list[Message], tickers: list[str]) -> str:
        """Infer a session name from content."""
        # Start with tickers
        ticker_str = ", ".join(tickers[:3]) if tickers else "Research"

        # Try to extract topic from first user message
        if messages:
            first_msg = next(
                (m.content for m in messages if m.role == "user"),
                "",
            )
            # Look for keywords
            keywords = []
            if "compare" in first_msg.lower():
                keywords.append("Comparison")
            if "thesis" in first_msg.lower():
                keywords.append("Thesis")
            if "risk" in first_msg.lower():
                keywords.append("Risk Analysis")
            if "valuation" in first_msg.lower():
                keywords.append("Valuation")

            if keywords:
                return f"{ticker_str} - {keywords[0]}"

        return f"{ticker_str} Research"

    def create(
        self,
        tickers: list[str] | None = None,
        framework: str | None = None,
        name: str | None = None,
    ) -> Session:
        """Create a new session."""
        session_id = self._generate_id()
        now = datetime.now().isoformat()

        session = Session(
            id=session_id,
            name=name or "New Research",
            tickers=tickers or [],
            framework=framework,
            created_at=now,
            updated_at=now,
        )

        return session

    def save(self, session: Session) -> Path:
        """Save a session to disk."""
        # Update name if still default
        if session.name == "New Research" and session.messages:
            session.name = self._infer_name(session.messages, session.tickers)

        # Auto-generate summary for sessions with 5+ messages
        if len(session.messages) >= 5 and not session.summary:
            session.summary = summarize_conversation(session.messages)

        session.updated_at = datetime.now().isoformat()
        path = self._session_path(session.id)

        with open(path, "w") as f:
            json.dump(session.to_dict(), f, indent=2)

        return path

    def load(self, session_id: str) -> Session:
        """Load a session from disk."""
        path = self._session_path(session_id)

        if not path.exists():
            raise ValueError(f"Session not found: {session_id}")

        with open(path) as f:
            data = json.load(f)

        return Session.from_dict(data)

    def delete(self, session_id: str) -> bool:
        """Delete a session."""
        path = self._session_path(session_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_sessions(
        self,
        ticker: str | None = None,
        framework: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        List available sessions with optional filtering.

        Returns list of session metadata (not full messages).
        """
        sessions = []

        for path in self.sessions_dir.glob("*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)

                # Apply filters
                if ticker and ticker.upper() not in [t.upper() for t in data.get("tickers", [])]:
                    continue
                if framework and data.get("framework") != framework:
                    continue

                sessions.append(
                    {
                        "id": data["id"],
                        "name": data["name"],
                        "tickers": data.get("tickers", []),
                        "framework": data.get("framework"),
                        "created_at": data["created_at"],
                        "updated_at": data["updated_at"],
                        "message_count": len(data.get("messages", [])),
                        "summary": data.get("summary", ""),
                        "preview": data.get("summary", "")[:100] or "...",
                    }
                )
            except (json.JSONDecodeError, KeyError):
                continue  # Skip invalid session files

        # Sort by updated_at, most recent first
        sessions.sort(key=lambda x: x["updated_at"], reverse=True)

        return sessions[:limit]

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search sessions by content."""
        query_lower = query.lower()
        results = []

        for path in self.sessions_dir.glob("*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)

                # Search in name, tickers, and messages
                matches = False

                if query_lower in data.get("name", "").lower() or any(
                    query_lower in t.lower() for t in data.get("tickers", [])
                ):
                    matches = True
                else:
                    for msg in data.get("messages", []):
                        if query_lower in msg.get("content", "").lower():
                            matches = True
                            break

                if matches:
                    results.append(
                        {
                            "id": data["id"],
                            "name": data["name"],
                            "tickers": data.get("tickers", []),
                            "updated_at": data["updated_at"],
                        }
                    )

            except (json.JSONDecodeError, KeyError):
                continue

        results.sort(key=lambda x: x["updated_at"], reverse=True)
        return results[:limit]

    def get_recent(self, limit: int = 5) -> list[dict[str, Any]]:
        """Get the most recently updated sessions."""
        return self.list_sessions(limit=limit)

    def rename(self, session_id: str, new_name: str) -> bool:
        """Rename a session."""
        try:
            session = self.load(session_id)
            session.name = new_name
            self.save(session)
            return True
        except ValueError:
            return False

    def add_ticker(self, session_id: str, ticker: str) -> bool:
        """Add a ticker to a session's ticker list."""
        try:
            session = self.load(session_id)
            if ticker.upper() not in [t.upper() for t in session.tickers]:
                session.tickers.append(ticker.upper())
                self.save(session)
            return True
        except ValueError:
            return False

    def set_framework(self, session_id: str, framework: str) -> bool:
        """Update the framework for a session."""
        try:
            session = self.load(session_id)
            session.framework = framework
            self.save(session)
            return True
        except ValueError:
            return False


def summarize_conversation(messages: list[Message], max_length: int = 500) -> str:
    """
    Generate a summary of a conversation for context compression.

    This is a simple heuristic-based summary. In production, you might
    want to use Claude for smarter summarization.
    """
    if not messages:
        return ""

    # Extract key points
    user_queries = []
    key_findings = []
    tickers_mentioned = set()

    # Regex for ticker mentions
    ticker_pattern = re.compile(r"\b[A-Z]{1,5}\b")

    for msg in messages:
        content = msg.content

        # Find tickers
        potential_tickers = ticker_pattern.findall(content)
        for t in potential_tickers:
            if len(t) >= 2 and t not in {
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
            }:
                tickers_mentioned.add(t)

        if msg.role == "user":
            # Capture first sentence of user queries
            first_sentence = content.split(".")[0][:100]
            if first_sentence and first_sentence not in user_queries:
                user_queries.append(first_sentence)

        elif msg.role == "assistant":
            # Look for findings (sentences with key words)
            sentences = content.split(".")
            for sent in sentences[:3]:  # First few sentences
                if any(
                    kw in sent.lower()
                    for kw in ["found", "shows", "indicates", "suggests", "analysis"]
                ):
                    if len(sent) > 20:
                        key_findings.append(sent.strip()[:150])

    # Build summary
    parts = []

    if tickers_mentioned:
        parts.append(f"Tickers: {', '.join(sorted(tickers_mentioned)[:5])}")

    if user_queries:
        parts.append(f"Topics: {'; '.join(user_queries[:3])}")

    if key_findings:
        parts.append(f"Findings: {' '.join(key_findings[:2])}")

    summary = ". ".join(parts)
    return summary[:max_length]


# Global session manager instance
_session_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


def reset_session_manager() -> None:
    """Reset the global session manager (useful for testing)."""
    global _session_manager
    _session_manager = None
