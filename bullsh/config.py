"""Configuration management - loads .env and config.toml."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# tomli is built into 3.11+, but we support 3.12+ so use tomllib
import tomllib
import tomli_w


# Default values (extracted so they can be referenced before instance creation)
DEFAULT_KEYBINDINGS = {
    "save_session": "ctrl+s",
    "clear_screen": "ctrl+l",
    "show_sources": "ctrl+o",
    "export_thesis": "ctrl+e",
}

DEFAULT_MODEL = "claude-sonnet-4-20250514"
DEFAULT_LOG_LEVEL = "info"
DEFAULT_VERBOSITY = "full"
DEFAULT_VERBOSE_TOOLS = False
DEFAULT_MAX_TOKENS_PER_SESSION = 1_000_000
DEFAULT_MAX_TOKENS_PER_TURN = 150_000
DEFAULT_WARN_AT_TOKEN_PCT = 0.8
DEFAULT_COST_PER_1K_INPUT = 0.003
DEFAULT_COST_PER_1K_OUTPUT = 0.015
DEFAULT_MAX_HISTORY_MESSAGES = 20
DEFAULT_ENABLE_PROMPT_CACHING = True
DEFAULT_TOOL_RESULT_MAX_CHARS = 8000


@dataclass
class Config:
    """Application configuration."""

    # Required
    anthropic_api_key: str
    edgar_identity: str

    # Optional with defaults
    model: str = DEFAULT_MODEL
    log_level: str = DEFAULT_LOG_LEVEL
    verbosity: str = DEFAULT_VERBOSITY  # "summary" or "full"
    verbose_tools: bool = DEFAULT_VERBOSE_TOOLS

    # Paths
    data_dir: Path = field(default_factory=lambda: Path.home() / ".bullsh")

    # Keybindings
    keybindings: dict[str, str] = field(default_factory=lambda: DEFAULT_KEYBINDINGS.copy())

    # Cost controls
    max_tokens_per_session: int = DEFAULT_MAX_TOKENS_PER_SESSION
    max_tokens_per_turn: int = DEFAULT_MAX_TOKENS_PER_TURN
    warn_at_token_pct: float = DEFAULT_WARN_AT_TOKEN_PCT
    cost_per_1k_input: float = DEFAULT_COST_PER_1K_INPUT
    cost_per_1k_output: float = DEFAULT_COST_PER_1K_OUTPUT

    # Token optimization
    max_history_messages: int = DEFAULT_MAX_HISTORY_MESSAGES
    enable_prompt_caching: bool = DEFAULT_ENABLE_PROMPT_CACHING
    tool_result_max_chars: int = DEFAULT_TOOL_RESULT_MAX_CHARS

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / "sessions"

    @property
    def theses_dir(self) -> Path:
        return self.data_dir / "theses"

    @property
    def frameworks_dir(self) -> Path:
        return self.data_dir / "frameworks"

    @property
    def custom_frameworks_dir(self) -> Path:
        return self.frameworks_dir / "custom"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    def ensure_dirs(self) -> None:
        """Create all required directories."""
        for dir_path in [
            self.cache_dir,
            self.sessions_dir,
            self.theses_dir,
            self.custom_frameworks_dir,
            self.logs_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


class ConfigError(Exception):
    """Raised when configuration is invalid or missing."""


def load_config(env_path: Path | None = None) -> Config:
    """
    Load configuration from .env and config.toml.

    Priority (highest to lowest):
    1. Environment variables
    2. config.toml
    3. Defaults
    """
    # Load .env file
    if env_path:
        load_dotenv(env_path)
    else:
        # Try local .env first, then ~/.bullsh/.env
        local_env = Path.cwd() / ".env"
        home_env = Path.home() / ".bullsh" / ".env"

        if local_env.exists():
            load_dotenv(local_env)
        elif home_env.exists():
            load_dotenv(home_env)

    # Load config.toml
    toml_config = _load_toml_config()

    # Get required values
    api_key = os.getenv("ANTHROPIC_API_KEY")
    edgar_identity = os.getenv("EDGAR_IDENTITY")

    if not api_key:
        raise ConfigError(
            "ANTHROPIC_API_KEY not found. Set it in .env or environment.\n"
            "Get your key at: https://console.anthropic.com/"
        )

    if not edgar_identity:
        raise ConfigError(
            "EDGAR_IDENTITY not found. Set it in .env or environment.\n"
            "SEC requires identification: 'Your Name your@email.com'"
        )

    # Get cost control settings
    cost_config = toml_config.get("cost_controls", {})

    # Build config with overrides
    config = Config(
        anthropic_api_key=api_key,
        edgar_identity=edgar_identity,
        model=os.getenv("MODEL", toml_config.get("general", {}).get("default_model", DEFAULT_MODEL)),
        log_level=os.getenv("LOG_LEVEL", toml_config.get("general", {}).get("log_level", DEFAULT_LOG_LEVEL)),
        verbosity=toml_config.get("general", {}).get("verbosity", DEFAULT_VERBOSITY),
        verbose_tools=toml_config.get("display", {}).get("verbose_tools", DEFAULT_VERBOSE_TOOLS),
        keybindings={
            **DEFAULT_KEYBINDINGS,
            **toml_config.get("keybindings", {}),
        },
        max_tokens_per_session=cost_config.get("max_tokens_per_session", DEFAULT_MAX_TOKENS_PER_SESSION),
        max_tokens_per_turn=cost_config.get("max_tokens_per_turn", DEFAULT_MAX_TOKENS_PER_TURN),
        warn_at_token_pct=cost_config.get("warn_at_token_pct", DEFAULT_WARN_AT_TOKEN_PCT),
        cost_per_1k_input=cost_config.get("cost_per_1k_input", DEFAULT_COST_PER_1K_INPUT),
        cost_per_1k_output=cost_config.get("cost_per_1k_output", DEFAULT_COST_PER_1K_OUTPUT),
        max_history_messages=cost_config.get("max_history_messages", DEFAULT_MAX_HISTORY_MESSAGES),
        enable_prompt_caching=cost_config.get("enable_prompt_caching", DEFAULT_ENABLE_PROMPT_CACHING),
        tool_result_max_chars=cost_config.get("tool_result_max_chars", DEFAULT_TOOL_RESULT_MAX_CHARS),
    )

    # Ensure directories exist
    config.ensure_dirs()

    return config


def _load_toml_config() -> dict[str, Any]:
    """Load config.toml if it exists."""
    config_path = Path.home() / ".bullsh" / "config.toml"

    if not config_path.exists():
        return {}

    try:
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"Invalid config.toml: {e}") from e


def save_config_toml(config: Config) -> Path:
    """Save current config to config.toml."""
    config_path = config.data_dir / "config.toml"
    config.data_dir.mkdir(parents=True, exist_ok=True)

    toml_data = {
        "general": {
            "verbosity": config.verbosity,
            "default_model": config.model,
            "log_level": config.log_level,
        },
        "display": {
            "verbose_tools": config.verbose_tools,
        },
        "keybindings": config.keybindings,
    }

    with open(config_path, "wb") as f:
        tomli_w.dump(toml_data, f)

    return config_path


def create_initial_env(data_dir: Path, api_key: str, edgar_identity: str) -> Path:
    """Create initial .env file during first-run setup."""
    env_path = data_dir / ".env"
    data_dir.mkdir(parents=True, exist_ok=True)

    env_content = f"""# bullsh configuration
ANTHROPIC_API_KEY={api_key}
EDGAR_IDENTITY="{edgar_identity}"

# Optional
# MODEL=claude-sonnet-4-20250514
# LOG_LEVEL=info
"""
    env_path.write_text(env_content)
    return env_path


# Global config instance (lazy loaded)
_config: Config | None = None


def get_config() -> Config:
    """Get the global config instance, loading it if necessary."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reset_config() -> None:
    """Reset the global config (useful for testing)."""
    global _config
    _config = None
