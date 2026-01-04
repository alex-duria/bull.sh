"""Tests for base tool classes."""

from datetime import datetime

import pytest

from bullsh.tools.base import (
    ToolDefinition,
    ToolResult,
    ToolStatus,
    get_tools_for_claude,
)


def test_tool_result_creation():
    """Test ToolResult dataclass."""
    result = ToolResult(
        data={"key": "value"},
        confidence=0.85,
        status=ToolStatus.SUCCESS,
        tool_name="test_tool",
        ticker="NVDA",
    )

    assert result.confidence == 0.85
    assert result.status == ToolStatus.SUCCESS
    assert result.data["key"] == "value"
    assert not result.cached


def test_tool_result_to_prompt_text():
    """Test formatting result for prompts."""
    result = ToolResult(
        data={"price": "145.67", "pe_ratio": "65.2"},
        confidence=0.9,
        status=ToolStatus.SUCCESS,
        tool_name="scrape_yahoo",
        ticker="NVDA",
    )

    text = result.to_prompt_text()

    assert "scrape_yahoo" in text
    assert "90%" in text
    assert "price" in text
    assert "145.67" in text


def test_tool_result_failed():
    """Test failed tool result formatting."""
    result = ToolResult(
        data={},
        confidence=0.0,
        status=ToolStatus.FAILED,
        tool_name="sec_fetch",
        ticker="INVALID",
        error_message="Company not found",
    )

    text = result.to_prompt_text()

    assert "Failed" in text
    assert "Company not found" in text


def test_tool_result_provenance():
    """Test provenance dict for thesis export."""
    result = ToolResult(
        data={},
        confidence=0.8,
        status=ToolStatus.SUCCESS,
        tool_name="sec_fetch",
        ticker="AAPL",
        cached=True,
        source_url="https://sec.gov/...",
    )

    provenance = result.to_provenance_dict()

    assert provenance["type"] == "sec_fetch"
    assert provenance["ticker"] == "AAPL"
    assert provenance["cached"] is True
    assert provenance["confidence"] == 0.8


def test_tool_definition():
    """Test ToolDefinition schema generation."""
    tool = ToolDefinition(
        name="test_tool",
        description="A test tool",
        parameters={
            "properties": {
                "ticker": {"type": "string"},
            },
            "required": ["ticker"],
        },
    )

    schema = tool.to_claude_schema()

    assert schema["name"] == "test_tool"
    assert schema["description"] == "A test tool"
    assert "input_schema" in schema
    assert schema["input_schema"]["properties"]["ticker"]["type"] == "string"


def test_get_tools_for_claude():
    """Test that all tools are returned in Claude format."""
    tools = get_tools_for_claude()

    assert len(tools) > 0
    assert all("name" in tool for tool in tools)
    assert all("description" in tool for tool in tools)
    assert all("input_schema" in tool for tool in tools)

    # Check for expected tools
    tool_names = [tool["name"] for tool in tools]
    assert "sec_search" in tool_names
    assert "sec_fetch" in tool_names
    assert "search_stocktwits" in tool_names
