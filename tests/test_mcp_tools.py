import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from server import list_tools, _health_check, _list_local_tasks


def test_new_mcp_tools_are_registered():
    tools = asyncio.run(list_tools())
    names = {tool.name for tool in tools}
    assert "wechat_health_check" in names
    assert "wechat_get_publish_status" in names
    assert "wechat_list_local_tasks" in names
    assert "wechat_research" in names
    assert "wechat_fetch_url" in names
    assert "wechat_tavily_search" in names


def test_health_check_without_network():
    result = asyncio.run(_health_check({"check_wechat": False}))
    payload = json.loads(result[0].text)
    assert payload["wechat"]["checked"] is False
    assert "env" in payload
    assert "dependencies" in payload


def test_list_local_tasks_shape():
    result = asyncio.run(_list_local_tasks({"limit": 3}))
    payload = json.loads(result[0].text)
    assert payload["status"] == "ok"
    assert payload["count"] <= 3
    assert isinstance(payload["tasks"], list)
