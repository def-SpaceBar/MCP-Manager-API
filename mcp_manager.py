import asyncio
import json
from typing import Dict
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
import asyncio
from typing import Optional
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

_SERVERS = {}


class MCPManager:
    def __init__(self, server_name: str, server_config: dict):
        self.exit_stack = AsyncExitStack()
        self.name = server_name
        self.args = server_config.get('args', None)
        self.command = server_config.get('command', None)
        self.env_vars = server_config.get('env', None)

    async def kill_connection(self):
        await self.exit_stack.aclose()

    async def register_server(self):
        #server_script_path: Path to the server script (.py or .js)

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
            await self.kill_connection()
            raise f'{e}: Could not connect to the server {self.name}'

        if f"{self.name}" not in _SERVERS:
            _SERVERS.setdefault(f"{self.name}", {})
            _SERVERS[f"{self.name}"]["session"] = self.session
            response = await self.session.list_tools()
            tools = response.tools
            print("\nConnected to server with tools:", [{"tool_name": tool.name, "tool_description": tool.description,
                                                         "tool_args": tool.inputSchema.get('properties')} for tool in
                                                        tools])
        else:
            print(f'Duplicate server ({self.name})')
            await self.kill_connection()
            pass