import asyncio
import json
import logging
import uuid
from typing import Any

from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.client import (
    MultiServerMCPClient,
    SSEConnection,
    StdioConnection,
)
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession


class MCPServerConnectionError(Exception):
    pass


class LCClientPatch(MultiServerMCPClient):
    initialize_timeout_s: float = 5
    errored_servers: dict[str, Exception] = {}

    async def __aenter__(self) -> "LCClientPatch":
        """Connect to all servers during context."""
        await super().__aenter__()
        return self

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
            # raise Exception
            await asyncio.wait_for(session.initialize(), timeout=self.initialize_timeout_s)
            # NOTE: The problem is that this may only get the timeout error.
            #  The actual error ends up only getting caught in the exit stack
            #  but there I can't know which server it was for. (my PR to mcp may help with this)
        except Exception as e:
            logging.error(f"Failed to initialize session for {server_name}: {e}")
            self.errored_servers[server_name] = e
            return

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
        self.lc_client: LCClientPatch = LCClientPatch(connections=connections)
        self._context_depth = 0

    @property
    def errored_servers(self) -> dict[str, Exception]:
        return self.lc_client.errored_servers

    async def __aenter__(self) -> "MultiMCPClient":
        """Connects to all servers during context."""
        if self._context_depth < 0:
            raise RuntimeError("Context manager has already exited")
        if self._context_depth == 0:
            self.lc_client = await self.lc_client.__aenter__()
        self._context_depth += 1
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:  # noqa: ANN001
        """Closes all server connections."""
        if self._context_depth <= 0:
            raise RuntimeError("Context manager has already exited")
        self._context_depth -= 1
        if self._context_depth == 0:
            try:
                await self.lc_client.__aexit__(exc_type, exc_value, traceback)
            except ExceptionGroup as e:
                logging.error(f"Errors closing connections: {e}")

    async def get_tools(self) -> list[BaseTool]:
        """Get all tools available from all connected servers."""
        # NOTE: lc loads on initial connection, so don't need to await here (in general it would be awaited though)
        async with self:
            return self.lc_client.get_tools()

    async def call_tool(self, server_name: str, tool_name: str, **kwargs) -> Any:  # noqa: ANN401
        """Call a tool on a specific server.

        Returns whatever the tool returns.
        """
        if server_name not in self.lc_client.server_name_to_tools:
            if server_name in self.errored_servers:
                raise MCPServerConnectionError(
                    f"Server {server_name} failed to connect {self.errored_servers[server_name]}"
                )
            raise ValueError(f"Server {server_name} not in connected servers")
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

    def set_connection_timeout(self, timeout_s: float) -> None:
        """Set the timeout for initializing a session."""
        self.lc_client.initialize_timeout_s = timeout_s
