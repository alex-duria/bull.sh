"""Artifact registry - tracks generated files for session continuity.

Solves the problem of the agent "forgetting" it already generated files.
Artifacts are stored in Session.metadata and injected into the system prompt.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from bullsh.tools.base import ToolResult, ToolStatus


@dataclass
class Artifact:
    """A generated file artifact."""

    path: str
    filename: str
    artifact_type: str  # "excel", "thesis", "factor_excel"
    tickers: list[str]
    created_at: str
    tool_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Artifact":
        return cls(**data)

    def exists(self) -> bool:
        """Check if the file still exists."""
        return Path(self.path).exists()

    def to_prompt_line(self) -> str:
        """Format for system prompt injection."""
        ticker_str = ", ".join(self.tickers) if self.tickers else "N/A"
        sheets = self.metadata.get("sheets", [])
        sheets_str = (
            f" ({', '.join(sheets[:3])}{'...' if len(sheets) > 3 else ''})" if sheets else ""
        )

        return f"- {self.artifact_type.upper()}: {self.filename}{sheets_str}\n  Tickers: {ticker_str}\n  Path: {self.path}"


class ArtifactRegistry:
    """
    Registry for tracking generated artifacts within a session.

    Stored in Session.metadata["artifacts"] for persistence.
    """

    ARTIFACT_KEY = "artifacts"

    def __init__(self, session: Any):
        """
        Initialize registry from session metadata.

        Args:
            session: Session object with metadata dict
        """
        self.session = session
        self._load()

    def _load(self) -> None:
        """Load artifacts from session metadata."""
        raw = self.session.metadata.get(self.ARTIFACT_KEY, {})
        self.artifacts: list[Artifact] = []

        for artifact_data in raw.get("items", []):
            try:
                self.artifacts.append(Artifact.from_dict(artifact_data))
            except (TypeError, KeyError):
                pass  # Skip malformed artifacts

    def _save(self) -> None:
        """Save artifacts to session metadata."""
        self.session.metadata[self.ARTIFACT_KEY] = {
            "items": [a.to_dict() for a in self.artifacts],
            "updated_at": datetime.now().isoformat(),
        }

    def register(self, artifact: Artifact) -> None:
        """Register a new artifact."""
        self.artifacts.append(artifact)
        self._save()

    def register_from_tool_result(self, result: ToolResult) -> Artifact | None:
        """
        Extract and register artifact from a tool result.

        Returns the artifact if one was created, None otherwise.
        """
        if result.status != ToolStatus.SUCCESS:
            return None

        # Detect artifact-generating tools
        artifact = None

        if result.tool_name == "generate_excel":
            artifact = Artifact(
                path=result.data.get("path", ""),
                filename=result.data.get("filename", "unknown.xlsx"),
                artifact_type="excel",
                tickers=result.data.get("tickers", [result.ticker] if result.ticker else []),
                created_at=datetime.now().isoformat(),
                tool_name=result.tool_name,
                metadata={
                    "sheets": result.data.get("sheets", []),
                },
            )

        elif result.tool_name == "save_thesis":
            artifact = Artifact(
                path=result.data.get("path", ""),
                filename=result.data.get("filename", "thesis.md"),
                artifact_type="thesis",
                tickers=[result.ticker] if result.ticker else [],
                created_at=datetime.now().isoformat(),
                tool_name=result.tool_name,
                metadata={},
            )

        elif result.tool_name == "generate_factor_excel":
            artifact = Artifact(
                path=result.data.get("path", ""),
                filename=result.data.get("filename", "factor_analysis.xlsx"),
                artifact_type="factor_excel",
                tickers=result.data.get("tickers", [result.ticker] if result.ticker else []),
                created_at=datetime.now().isoformat(),
                tool_name=result.tool_name,
                metadata={
                    "sheets": result.data.get("sheets", []),
                    "factors": result.data.get("factors", []),
                },
            )

        if artifact and artifact.path:
            self.register(artifact)
            return artifact

        return None

    def get_by_type(self, artifact_type: str) -> list[Artifact]:
        """Get all artifacts of a specific type."""
        return [a for a in self.artifacts if a.artifact_type == artifact_type]

    def get_by_ticker(self, ticker: str) -> list[Artifact]:
        """Get all artifacts for a specific ticker."""
        ticker = ticker.upper()
        return [a for a in self.artifacts if ticker in [t.upper() for t in a.tickers]]

    def get_latest(self, artifact_type: str | None = None) -> Artifact | None:
        """Get the most recent artifact, optionally filtered by type."""
        filtered = self.artifacts
        if artifact_type:
            filtered = [a for a in filtered if a.artifact_type == artifact_type]

        if not filtered:
            return None

        return max(filtered, key=lambda a: a.created_at)

    def get_existing(self) -> list[Artifact]:
        """Get only artifacts whose files still exist."""
        return [a for a in self.artifacts if a.exists()]

    def clear(self) -> None:
        """Clear all artifacts."""
        self.artifacts = []
        self._save()

    def to_prompt_section(self) -> str | None:
        """
        Generate system prompt section listing artifacts.

        Returns None if no artifacts exist.
        """
        existing = self.get_existing()
        if not existing:
            return None

        lines = [
            "**SESSION ARTIFACTS:**",
            "You have generated these files in this session. You can reference or update them:",
            "",
        ]

        for artifact in existing:
            lines.append(artifact.to_prompt_line())
            lines.append("")

        lines.append("To update an existing file, use the same tool with the same ticker(s).")

        return "\n".join(lines)


def extract_artifact_from_result(result: ToolResult, session: Any) -> Artifact | None:
    """
    Convenience function to extract and register artifact from tool result.

    Args:
        result: Tool result that may contain artifact info
        session: Session to store artifact in

    Returns:
        Artifact if one was created, None otherwise
    """
    registry = ArtifactRegistry(session)
    return registry.register_from_tool_result(result)
