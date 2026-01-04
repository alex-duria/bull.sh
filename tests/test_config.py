"""Tests for configuration module."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from bullsh.config import Config, ConfigError, load_config, reset_config


def test_config_dataclass(test_config: Config):
    """Test Config dataclass creation."""
    assert test_config.anthropic_api_key == "test-api-key-12345"
    assert test_config.edgar_identity == "Test User test@example.com"
    assert test_config.model == "claude-sonnet-4-20250514"


def test_config_paths(test_config: Config):
    """Test Config path properties."""
    assert test_config.cache_dir == test_config.data_dir / "cache"
    assert test_config.sessions_dir == test_config.data_dir / "sessions"
    assert test_config.theses_dir == test_config.data_dir / "theses"


def test_config_ensure_dirs(test_config: Config, tmp_path: Path):
    """Test directory creation."""
    test_config.ensure_dirs()

    assert test_config.cache_dir.exists()
    assert test_config.sessions_dir.exists()
    assert test_config.theses_dir.exists()


def test_load_config_missing_api_key():
    """Test that missing API key raises ConfigError."""
    reset_config()
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}, clear=True):
        with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
            load_config()


def test_load_config_missing_edgar_identity():
    """Test that missing EDGAR identity raises ConfigError."""
    reset_config()
    with patch.dict(
        os.environ,
        {"ANTHROPIC_API_KEY": "test-key", "EDGAR_IDENTITY": ""},
        clear=True,
    ):
        with pytest.raises(ConfigError, match="EDGAR_IDENTITY"):
            load_config()
