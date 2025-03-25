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


class MCPManager:
    def __init__(self, server_name: str, server_config: dict):
        self.exit_stack = AsyncExitStack()
        self.name = server_name
        self.args = server_config.get('args', None)
        self.command = server_config.get('command', None)
        self.env_vars = server_config.get('env', None)

    async def kill_connection(self):
        await self.exit_stack.aclose()

    async def register_mcp(self):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """

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
            raise ConnectionError(f'Could not connect to the server {self.name}')

        if f"{self.name}" not in SERVERS:
            SERVERS.setdefault(f"{self.name}", {})
            SERVERS[f"{self.name}"]["session"] = self.session
            response = await self.session.list_tools()
            tools = response.tools
            print("\nConnected to server with tools:", [{"tool_name": tool.name, "tool_description": tool.description,
                                                         "tool_args": tool.inputSchema.get('properties')} for tool in
                                                        tools])
        else:
            await self.kill_connection()
            raise ValueError(f'Duplicate server ({self.name})')

        print(SERVERS)

    async def list_tools(self):
        response = await self.session.list_tools()
        tools = response.tools
        return print(f'Tools: {tools}\n'
                     f'obj-type: {type(tools)}')

#       Trying to perform connection with sse ###
#######
# async def connect_sse_server():
#     url = "http://127.0.0.1:5123/sse"
#     async with sse_client(url) as (read_stream, write_stream):
#         # Initialize
#         await write_stream.send(json.dumps({"command": "initialize"}))
#         init_raw = await read_stream.__anext__()
#         init_data = json.loads(init_raw)
#         print("Init:", init_data)
#
#         # List Tools
#         await write_stream.send(json.dumps({"command": "list_tools"}))
#         tools_raw = await read_stream.__anext__()
#         tools_data = json.loads(tools_raw)
#         print("Tools:", tools_data)
#
#         # Execute a tool
#         exec_cmd = {
#             "command": "execute_tool",
#             "tool": "echo_tool",
#             "params": {"text": "Hello SSE!"}
#         }
#         await write_stream.send(json.dumps(exec_cmd))
#         exec_raw = await read_stream.__anext__()
#         exec_data = json.loads(exec_raw)
#         print("Exec result:", exec_data)

#
# async def main():
#     """
#     Prompt user for server scripts in a loop, connect to each, and store sessions in a dict.
#     """
#     print("Enter the path to an MCP server script (.py or .js). Type 'done' to finish.")
#     # await connect_sse_server()
#
#     client = MCPManager("test", {"command": "python", "args": [r"C:\Users\space\Desktop\Python\mcp_agent\poc_mcp.py"]})
#     await client.register_mcp()
#     import time
#     time.sleep(100)
#     # await client.kill_connection()
#
#     # while True:
#     #     path = input("\nServer path: ").strip()
#     #     if path.lower() == "done":
#     #         break
#     #     if not path:
#     #         continue
#     #
#     #     try:
#     #         session = await connect_to_server(path)
#     #         sessions[path] = session
#     #     except Exception as e:
#     #         print(f"Error connecting to {path}: {e}")
#     #
#     # print("\nAll sessions (no cleanup!):", sessions)
#     print("Script ending, but sessions remain active until Python exits.")
#
#
# if __name__ == "__main__":
#     asyncio.run(main())
