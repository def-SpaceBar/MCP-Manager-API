import os
import signal
from fastapi import HTTPException
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
import asyncio
from typing import Optional
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVERS = {}
DEBUG_BOOL = False


class MCPManager:

    def __init__(self, server_name: str, server_config: dict, process):
        self.exit_stack = AsyncExitStack()
        self.name = server_name
        self.args = server_config.get('args', None)
        self.command = server_config.get('command', None)
        self.env_vars = server_config.get('env', None)
        self.tools_description = []
        self.tools_full_data = []
        self.tools_name = []
        self.process = process


    async def kill_session(self):
        self.process.kill()

        # # Try to close the async resources
        # try:
        #     await self.exit_stack.aclose()
        # except asyncio.CancelledError:
        #     # Suppress expected shutdown error
        #     print(f"[INFO] CancelledError while closing exit_stack — safe during shutdown.")
        # except Exception as e:
        #     print(f"[ERROR] While closing exit_stack: {e}")

        # Always remove from registry
        SERVERS.pop(self.name, None)

    async def register_mcp(self):

        try:
            server_params = StdioServerParameters(
                command=self.command,
                args=self.args,
                env=self.env_vars
            )

            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

            await self.session.initialize()

        except ConnectionError as e:
            await self.kill_session()
            raise ConnectionError(f'Could not connect to the server {self.name}')

        if f"{self.name}" not in SERVERS:
            SERVERS[self.name] = self
            response = await self.session.list_tools()
            self.tools_full_data = response.tools
            self.tools_description = [f"Tool Name: {tool.name}\nTool Description: {tool.description}" for tool in
                                      self.tools_full_data]
            self.tools_name = [tool.name for tool in self.tools_full_data]

        else:
            await self.kill_session()
            raise ValueError(f'Duplicate server ({self.name})')

        if DEBUG_BOOL:
            mcp_server = SERVERS["poc_mcp"]
            print(mcp_server)


    async def list_tools(self):
        return self.tools_full_data
