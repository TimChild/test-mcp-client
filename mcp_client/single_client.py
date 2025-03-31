import asyncio
import logging
from contextlib import AsyncExitStack
from typing import Optional

from anthropic import Anthropic
from anthropic.types import MessageParam, TextBlock, ToolParam
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client

logging.basicConfig(level=logging.INFO)

load_dotenv()  # load environment variables from .env


class MCPClient:
    def __init__(self) -> None:
        self._session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    @property
    def session(self) -> ClientSession:
        if self._session is None:
            raise ValueError("Session not initialized")
        return self._session

    @session.setter
    def session(self, value: ClientSession) -> None:
        self._session = value

    async def connect_to_server(self, server_path_or_url: str) -> None:
        """Connect to an MCP server

        Args:
            server_path_or_url: Path to the server script (.py or .js) or URL of the SSE server
        """
        if server_path_or_url.startswith("http"):
            await self._connect_http_sse(server_path_or_url)
        else:
            await self._connect_stdio(server_path_or_url)

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def _connect_stdio(self, server_script_path: str) -> None:
        logging.info(f"Connecting to server script: {server_script_path}")
        is_python = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(command=command, args=[server_script_path], env=None)
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

    async def _connect_http_sse(self, server_url: str) -> None:
        """Connect to an MCP server

        Args:
            server_url: URL of the server
        """
        logging.info(f"Connecting to SSE server: {server_url}")
        sse_transport = await self.exit_stack.enter_async_context(sse_client(server_url))
        self.sse, self.write = sse_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.sse, self.write)
        )

    async def cleanup(self) -> None:
        """Clean up resources"""
        logging.info("Cleaning up resources")
        await self.exit_stack.aclose()


class Agent:
    def __init__(self, session: ClientSession) -> None:
        self.session = session
        self.anthropic = Anthropic()

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        messages: list[MessageParam] = [{"role": "user", "content": query}]

        logging.info("Calling list_tools")
        response = await self.session.list_tools()
        available_tools: list[ToolParam] = [
            ToolParam(
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": tool.inputSchema,
                }
            )
            for tool in response.tools
        ]

        # Initial Claude API call
        logging.info("Calling Claude API")
        response = self.anthropic.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=1000,
            messages=messages,
            tools=available_tools,
        )

        # Process response and handle tool calls
        final_text = []

        assistant_message_content = []
        for content in response.content:
            if content.type == "text":
                final_text.append(content.text)
                assistant_message_content.append(content)
            elif content.type == "tool_use":
                tool_name = content.name
                tool_args = content.input
                assert tool_args is None or isinstance(tool_args, dict)

                # Execute tool call
                logging.info(f"Calling tool {tool_name}")
                result = await self.session.call_tool(tool_name, tool_args)
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                assistant_message_content.append(content)
                messages.append({"role": "assistant", "content": assistant_message_content})
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": content.id,
                                "content": result.content,
                            }  # pyright: ignore[reportArgumentType]
                        ],
                    }
                )

                # Get next response from Claude
                logging.info("Calling Claude API with tool result")
                response = self.anthropic.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    messages=messages,
                    tools=available_tools,
                )

                assert isinstance(response.content[0], TextBlock)
                final_text.append(response.content[0].text)

        return "\n".join(final_text)

    async def chat_loop(self) -> None:
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == "quit":
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")


async def main() -> None:
    if len(sys.argv) > 1:
        raise ValueError("Usage: python client.py (no additional arguments)")

    client = MCPClient()

    try:
        await client.connect_to_server("http://localhost:9090/sse")

        agent = Agent(client.session)
        await agent.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    import sys

    asyncio.run(main())
