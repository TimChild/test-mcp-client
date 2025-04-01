import asyncio
import time

import pytest
from langchain_mcp_adapters.client import SSEConnection, StdioConnection

from mcp_client import MultiMCPClient
from mcp_client.multi_client import MCPServerConnectionError

print("exiting script")


def test_init_multi_mcp_client():
    multi_mcp_client = MultiMCPClient(connections={})
    assert isinstance(multi_mcp_client, MultiMCPClient)


def _stdio_connection_from_path(path: str) -> StdioConnection:
    return StdioConnection(
        transport="stdio",
        command="uv",
        args=["run", path],
        env=None,
        encoding="utf-8",
        encoding_error_handler="strict",
    )


def _sse_connection_from_path(path: str) -> SSEConnection:
    return SSEConnection(
        transport="sse",
        url=path,
    )


def test_init_with_connections():
    example_server_conn = _stdio_connection_from_path("tests/example_mcp_server.py")
    multi_mcp_client = MultiMCPClient(connections={"example": example_server_conn})
    assert isinstance(multi_mcp_client, MultiMCPClient)


@pytest.fixture(scope="module")
def client() -> MultiMCPClient:
    example_server_conn = _stdio_connection_from_path("tests/example_mcp_server.py")
    return MultiMCPClient(connections={"example": example_server_conn})


@pytest.fixture
def client_with_missing_servers() -> MultiMCPClient:
    conns = {
        "example": _stdio_connection_from_path("tests/example_mcp_server.py"),
        "missing_stdio": _stdio_connection_from_path("non_existent.py"),
        "missing_sse": _sse_connection_from_path("https://missing-server.com"),
    }
    mcp_client = MultiMCPClient(connections=conns)
    mcp_client.set_connection_timeout(0.5)
    return mcp_client


async def test_get_tools(client: MultiMCPClient):
    tools = await client.get_tools()

    assert isinstance(tools, list)
    assert len(tools) > 0


async def test_call_tool(client: MultiMCPClient):
    result = await client.call_tool(server_name="example", tool_name="test-tool")
    assert isinstance(result, dict)
    assert result == {"data": "Hello World!"}


class TestPing:
    async def test_ping(self, client: MultiMCPClient):
        client.set_connection_timeout(0.3)
        t = time.time()
        errors = await asyncio.wait_for(client.ping_servers(), timeout=1)
        assert time.time() - t < 0.8
        assert errors == {}

    async def test_ping_with_errors(self, client_with_missing_servers: MultiMCPClient):
        client_with_missing_servers.set_connection_timeout(0.5)
        t = time.time()
        errors = await asyncio.wait_for(client_with_missing_servers.ping_servers(), timeout=2)
        assert time.time() - t < 1.5
        assert len(errors) == 2
        assert "missing_stdio" in errors
        assert "missing_sse" in errors


async def test_handles_missing_servers(client_with_missing_servers: MultiMCPClient):
    """Should handle missing servers gracefully."""
    multi_mcp_client = client_with_missing_servers
    assert isinstance(multi_mcp_client, MultiMCPClient)

    async with multi_mcp_client:
        tools = await multi_mcp_client.get_tools()
        assert len(tools) > 0, "Should still find the example tools"
        assert tools[0].name == "test-tool"

    assert len(multi_mcp_client.errored_servers) == 2
    assert "missing_stdio" in multi_mcp_client.errored_servers
    assert "missing_sse" in multi_mcp_client.errored_servers

    async with multi_mcp_client:
        response = await multi_mcp_client.call_tool(server_name="example", tool_name="test-tool")
        assert response is not None

        with pytest.raises(MCPServerConnectionError):
            _ = await multi_mcp_client.call_tool(server_name="missing_stdio", tool_name="test-tool")

        with pytest.raises(MCPServerConnectionError):
            _ = await multi_mcp_client.call_tool(server_name="missing_sse", tool_name="test-tool")
