"""Tests for the Tengu congressional-trades tool (no network / no key required)."""

import asyncio
import os

from bullsh.tools import tengu
from bullsh.tools.base import ALL_TOOLS, ToolStatus, get_tools_for_claude


def test_tool_is_registered():
    names = [t.name for t in ALL_TOOLS]
    assert "get_congress_trades" in names
    assert any(t["name"] == "get_congress_trades" for t in get_tools_for_claude())


def test_requires_api_key():
    os.environ.pop("TENGU_API_KEY", None)
    res = asyncio.run(tengu.get_congress_trades("NVDA"))
    assert res.status == ToolStatus.FAILED
    assert "tengu.co/api" in res.data.get("help", "")


def test_success_with_mocked_fetch():
    os.environ["TENGU_API_KEY"] = "test-key"
    sample = {"data": [
        {"politician": "Rep A", "ticker": "NVDA", "type": "buy"},
        {"politician": "Sen B", "ticker": "NVDA", "type": "buy"},
    ]}

    async def fake_fetch(path, params, api_key):
        assert path == "/api/v3/intel/congress"
        assert api_key == "test-key"
        assert params.get("ticker") == "NVDA"
        return sample

    tengu._fetch = fake_fetch  # bypass network
    res = asyncio.run(tengu.get_congress_trades("NVDA", limit=10))
    assert res.status == ToolStatus.SUCCESS
    assert res.data["count"] == 2
    assert res.data["ticker"] == "NVDA"


if __name__ == "__main__":
    test_tool_is_registered()
    test_requires_api_key()
    test_success_with_mocked_fetch()
    print("ALL TESTS PASSED")
