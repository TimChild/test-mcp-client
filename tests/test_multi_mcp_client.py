import pytest
from langchain_mcp_adapters.client import StdioConnection

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


def test_init_with_connections():
    example_server_conn = _stdio_connection_from_path("tests/example_mcp_server.py")
    multi_mcp_client = MultiMCPClient(connections={"example": example_server_conn})
    assert isinstance(multi_mcp_client, MultiMCPClient)


@pytest.fixture(scope="module")
def client() -> MultiMCPClient:
    example_server_conn = _stdio_connection_from_path("tests/example_mcp_server.py")
    return MultiMCPClient(connections={"example": example_server_conn})


async def test_get_tools(client: MultiMCPClient):
    tools = await client.get_tools()

    assert isinstance(tools, list)
    assert len(tools) > 0


async def test_call_tool(client: MultiMCPClient):
    result = await client.call_tool(server_name="example", tool_name="test-tool")
    assert isinstance(result, dict)
    assert result == {"data": "Hello World!"}


async def test_handles_missing_servers():
    """Should handle missing servers gracefully."""
    example_server_conn = _stdio_connection_from_path("tests/example_mcp_server.py")
    missing_server_conn = _stdio_connection_from_path("missing-server.py")
    multi_mcp_client = MultiMCPClient(
        connections={
            "example": example_server_conn,
            "missing": missing_server_conn,
        }
    )
    multi_mcp_client.set_connection_timeout(0.5)
    assert isinstance(multi_mcp_client, MultiMCPClient)

    async with multi_mcp_client:
        tools = await multi_mcp_client.get_tools()
        assert len(tools) > 0, "Should still find the example tools"
        assert tools[0].name == "test-tool"

    assert len(multi_mcp_client.errored_servers) == 1
    assert "missing" in multi_mcp_client.errored_servers

    async with multi_mcp_client:
        response = await multi_mcp_client.call_tool(server_name="example", tool_name="test-tool")
        assert response is not None

        with pytest.raises(MCPServerConnectionError):
            _ = await multi_mcp_client.call_tool(server_name="missing", tool_name="test-tool")
