import asyncio

from langchain_mcp_adapters.client import StdioConnection

from mcp_client import MultiMCPClient


def get_client() -> MultiMCPClient:
    example_server_conn = StdioConnection(
        transport="stdio",
        command="uv",
        args=["run", "tests/eample_mcp_server.py"],
        env=None,
        encoding="utf-8",
        encoding_error_handler="strict",
    )
    return MultiMCPClient(connections={"example": example_server_conn})


async def main(client: MultiMCPClient):
    tools = await client.get_tools()

    assert isinstance(tools, list)
    assert len(tools) > 0


if __name__ == "__main__":
    client = get_client()
    asyncio.run(main(client))
