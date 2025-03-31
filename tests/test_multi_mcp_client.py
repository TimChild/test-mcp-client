import pytest
from langchain_core.messages.tool import ToolMessage
from langchain_mcp_adapters.client import StdioConnection

from mcp_client import MultiMCPClient

print("exiting script")


def test_init_multi_mcp_client():
    multi_mcp_client = MultiMCPClient(connections={})
    assert isinstance(multi_mcp_client, MultiMCPClient)


def test_init_with_connections():
    example_server_conn = StdioConnection(
        transport="stdio",
        command="uv",
        args=["run", "tests/example_mcp_server.py"],
        env=None,
        encoding="utf-8",
        encoding_error_handler="strict",
    )
    multi_mcp_client = MultiMCPClient(connections={"example": example_server_conn})
    assert isinstance(multi_mcp_client, MultiMCPClient)


@pytest.fixture(scope="module")
def client() -> MultiMCPClient:
    example_server_conn = StdioConnection(
        transport="stdio",
        command="uv",
        args=["run", "tests/example_mcp_server.py"],
        env=None,
        encoding="utf-8",
        encoding_error_handler="strict",
    )
    return MultiMCPClient(connections={"example": example_server_conn})


async def test_get_tools(client: MultiMCPClient):
    tools = await client.get_tools()

    assert isinstance(tools, list)
    assert len(tools) > 0


async def test_call_tool(client: MultiMCPClient):
    result = await client.call_tool(server_name="example", tool_name="test-tool")
    assert isinstance(result, dict)
    assert result == {"data": "Hello World!"}
