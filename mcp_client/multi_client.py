import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from langchain_core.messages import AIMessage
from langchain_core.messages.tool import ToolCall, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.client import (
    MultiServerMCPClient,
    SSEConnection,
    StdioConnection,
)
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession


class LCClientPatch(MultiServerMCPClient):
    initialize_timeout_s: float = 5

    # added timeout on intiaializing a session
    async def _initialize_session_and_load_tools(
        self, server_name: str, session: ClientSession
    ) -> None:
        """Initialize a session and load tools from it.

        Args:
            server_name: Name to identify this server connection
            session: The ClientSession to initialize
        """
        # Initialize the session
        try:
            await asyncio.wait_for(session.initialize(), timeout=self.initialize_timeout_s)
        except asyncio.TimeoutError:
            raise RuntimeError("Failed to initialize session within timeout")
        self.sessions[server_name] = session

        # Load tools from this server
        server_tools = await load_mcp_tools(session)
        self.server_name_to_tools[server_name] = server_tools


class MultiMCPClient:
    def __init__(self, connections: dict[str, SSEConnection | StdioConnection]) -> None:
        """Initializes an adapter for multiple mcp clients.

        Args:
            connections: A dictionary mapping server names to connection configurations.
                Each configuration can be either a StdioConnection or SSEConnection.
        """
        self.connections = connections
        self.lc_client = LCClientPatch(connections=connections)
        self._context_depth = 0

    async def __aenter__(self) -> "MultiMCPClient":
        """Connects to all servers during context."""
        logging.critical("Entering context")
        if self._context_depth < 0:
            raise RuntimeError("Context manager has already exited")
        if self._context_depth == 0:
            self.lc_client = await self.lc_client.__aenter__()
        self._context_depth += 1
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:  # noqa: ANN001
        """Closes all server connections."""
        print("Exiting context")
        if self._context_depth <= 0:
            raise RuntimeError("Context manager has already exited")
        self._context_depth -= 1
        if self._context_depth == 0:
            await self.lc_client.__aexit__(exc_type, exc_value, traceback)

    async def get_tools(self) -> list[BaseTool]:
        """Get all tools available from all connected servers."""
        # NOTE: lc loads on initial connection, so don't need to await here (in general it would be awaited though)
        async with self:
            return self.lc_client.get_tools()

    async def call_tool(self, server_name: str, tool_name: str, **kwargs) -> Any:  # noqa: ANN401
        """Call a tool on a specific server.

        Returns whatever the tool returns.
        """
        async with self:
            server_tools = self.lc_client.server_name_to_tools[server_name]
            tool = next(t for t in server_tools if t.name == tool_name)
            assert isinstance(tool, StructuredTool)
            assert tool.coroutine is not None
            tool_call = ToolCall(
                name=tool_name,
                args=kwargs,
                id=str(uuid.uuid4()),
            )
            tool_content = await tool.ainvoke(tool_call)
            try:
                return json.loads(tool_content)
            except json.JSONDecodeError:
                return tool_content
            print(type(tool_content))
            return ToolMessage(tool_call_id=tool_call["id"], content=tool_content)
