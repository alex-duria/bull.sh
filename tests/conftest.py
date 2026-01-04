"""Pytest configuration and fixtures."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from bullsh.config import Config, reset_config


@pytest.fixture(autouse=True)
def reset_global_config():
    """Reset global config before each test."""
    reset_config()
    yield
    reset_config()


@pytest.fixture
def mock_env(tmp_path: Path):
    """Mock environment variables for testing."""
    env_vars = {
        "ANTHROPIC_API_KEY": "test-api-key-12345",
        "EDGAR_IDENTITY": "Test User test@example.com",
        "MODEL": "claude-sonnet-4-20250514",
        "LOG_LEVEL": "debug",
    }
    with patch.dict(os.environ, env_vars, clear=False):
        yield env_vars


@pytest.fixture
def test_config(mock_env, tmp_path: Path) -> Config:
    """Create a test configuration."""
    return Config(
        anthropic_api_key=mock_env["ANTHROPIC_API_KEY"],
        edgar_identity=mock_env["EDGAR_IDENTITY"],
        model=mock_env["MODEL"],
        log_level=mock_env["LOG_LEVEL"],
        data_dir=tmp_path / ".bullsh",
    )


@pytest.fixture
def sample_10k_text() -> str:
    """Sample 10-K text for testing."""
    return """
ITEM 1. BUSINESS

Company Overview
We are a leading technology company focused on artificial intelligence
and accelerated computing. Our products include GPUs, data center solutions,
and AI platforms.

Competition
The markets in which we operate are highly competitive. We face competition
from established companies and new entrants.

ITEM 1A. RISK FACTORS

Supply Chain Risks
We rely on third-party manufacturers for production of our products.
Disruptions in supply chain could materially affect our business.

Customer Concentration
A significant portion of our revenue comes from a limited number of customers.
Loss of key customers could adversely affect our results.
"""


@pytest.fixture
def sample_yahoo_html() -> str:
    """Sample Yahoo Finance HTML for testing."""
    return """
<html>
<body>
<div data-field="regularMarketPrice">145.67</div>
<div data-field="regularMarketChange">+2.34</div>
<table>
    <tr><td>Market Cap</td><td>1.5T</td></tr>
    <tr><td>PE Ratio (TTM)</td><td>65.2</td></tr>
    <tr><td>EPS (TTM)</td><td>2.23</td></tr>
</table>
</body>
</html>
"""
