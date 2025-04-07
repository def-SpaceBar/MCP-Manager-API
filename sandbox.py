import json
import os
from fastapi import FastAPI
from mcp import Tool

from routers import mcp_actions
from colorama import Fore, Style, init
from dotenv import (load_dotenv)
import subprocess
import socket
import datetime
from manager.mcp_manager import MCPManager, SERVERS

# load_dotenv('.env')
# init(autoreset=True)
# SYSLOG_SERVER_IP = os.getenv('SYSLOG_SERVER_IP')
# error_message = f"{Fore.RED}ERROR:{Style.RESET_ALL}"
# success_message = f"{Fore.GREEN}SUCCESS:{Style.RESET_ALL}"
#
#
# config_path = os.getenv('mcp_config_path')
# if not config_path:
#     raise ValueError(f"{error_message} Please set the 'mcp_config_path' environment variable.")
#
# try:
#     print('Reading MCP Configuration')
#     with open(config_path, 'r') as mcps_config_file:
#         config_data = json.load(mcps_config_file)
# except FileNotFoundError:
#     raise FileNotFoundError(f'{error_message} reading MCP Configuration - check JSON path.')
#
# config_data = config_data.get('mcpServers')
#
# if config_data:
#     print(f'{success_message} Loaded MCP Configuration - {list(config_data.keys())}')
#     for key, value in config_data.items():
#         print(f'Running {key} MCP')
#         command = value.get('command', None)
#         args = value.get('args', None)
#         env_vars = value.get('env', None)
#
#         if command and args is None:
#             print(f'{key} MCP Do not have a Commands & Args configured. Skipping.')
#             pass
#
#         try:
#             if env_vars:
#                 run_mcp = subprocess.Popen(command, *args, env=env_vars,
#                                            stdout=subprocess.PIPE,
#                                            stderr=subprocess.PIPE,
#                                            text=True)
#                 errors = [e for e in run_mcp.stderr]
#             else:
#                 run_mcp = subprocess.Popen(command, *args,
#                                            stdout=subprocess.PIPE,
#                                            stderr=subprocess.PIPE,
#                                            text=True)
#
#                 errors = [e for e in run_mcp.stderr]
#
#             mcp_server = MCPManager(key, value)
#             mcp_server.register_mcp()
#             mcp_server.list_tools()
#         except Exception as e:
#             print(f'{error_message} Failed executing MCP ({key}) Server.')
#             pass


tools = [Tool(name='echo_tool', description='Bamba', inputSchema={'properties': {'text': {'title': 'Text', 'type': 'string'}}, 'required': ['text'], 'title': 'echo_toolArguments', 'type': 'object'})]

for i in tools:
    print(i.name)