import asyncio
import json
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from routers import mcp_actions
from colorama import Fore, Style, init
from dotenv import (load_dotenv)
import subprocess
from manager.mcp_manager import MCPManager, SERVERS, DEBUG_BOOL

load_dotenv('.env')
init(autoreset=True)

error_message = f"{Fore.RED}ERROR:{Style.RESET_ALL}"
warning_message = f"{Fore.YELLOW}WARNING:{Style.RESET_ALL}"
success_message = f"{Fore.GREEN}SUCCESS:{Style.RESET_ALL}"
info_message = f"{Fore.GREEN}INFO:{Style.RESET_ALL}"

config_path = os.getenv('mcp_config_path')

if not config_path:
    raise ValueError(f"{error_message}     Please set the 'mcp_config_path' environment variable.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        print(f'{info_message}     Reading MCP Configuration')
        with open(config_path, 'r') as mcps_config_file:
            config_data = json.load(mcps_config_file)
    except FileNotFoundError:
        raise FileNotFoundError(f'{error_message}     Cant read MCP Configuration - check JSON path.')

    config_data = config_data.get('mcpServers')

    if config_data:
        print(f'{success_message}     Loaded MCP Configuration - {list(config_data.keys())}')
        for key, value in config_data.items():
            print(f'{info_message}     Running {key} MCP')
            command = value.get('command', None)
            args = value.get('args', None)
            env_vars = value.get('env', None)

            if command and args is None:
                print(f'{info_message}     {key} MCP Do not have a Commands & Args configured. Skipping.')
                pass

            command = [command, *args]
            print(warning_message + "     " + "Executing CMD - " + f"{command}")
            try:
                if env_vars:
                    mcp_process = await asyncio.create_subprocess_exec(
                        *command,
                        env=env_vars
                    )
                else:
                    mcp_process = await asyncio.create_subprocess_exec(
                        *command
                    )

                mcp_server = MCPManager(key, value, mcp_process)
                await mcp_server.register_mcp()
                await mcp_server.list_tools()
            except Exception as e:
                print(f'{error_message}     Failed executing MCP ({key}) Server.\n Exception: {e}')
                pass

    yield

    for server, mcp_object in SERVERS.items():
        await mcp_object.kill_session()

    print(SERVERS)


mcp_manager_api = FastAPI(lifespan=lifespan,
                          title="MCP Manager API - @spacebar",
                          version="1.0",
                          description="API for managing MCP servers and tools")
mcp_manager_api.include_router(mcp_actions.router)
