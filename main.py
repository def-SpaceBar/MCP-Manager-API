import json
import os
from fastapi import FastAPI
from routers import mcp_actions
from colorama import Fore, Style, init
from dotenv import (load_dotenv)
import subprocess
import socket
import datetime

load_dotenv('.env')
init(autoreset=True)
SYSLOG_SERVER_IP = os.getenv('SYSLOG_SERVER_IP')


def send_syslog(message: str, port: int = 514, tag: str = "fastapi-mcp"):
    # Format: <PRI>TIMESTAMP HOST TAG: MESSAGE
    priority = 13  # LOG_USER (1) * 8 + LOG_INFO (5) = 13
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%b %d %H:%M:%S")
    hostname = socket.gethostname()
    syslog_msg = f"<{priority}>{timestamp} {hostname} {tag}: {message}"

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(syslog_msg.encode(), (SYSLOG_SERVER_IP, port))
    sock.close()




async def lifespan(app: FastAPI):
    config_path = os.getenv('mcp_config_path')
    if not config_path:
        raise ValueError("Please set the 'mcp_config_path' environment variable.")

    try:
        print('Reading MCP Configuration')
        with open(config_path, 'r') as mcps_config_file:
            config_data = json.load(mcps_config_file)
    except FileNotFoundError:
        raise FileNotFoundError('Error reading MCP Configuration - check JSON path.')

    config_data = config_data.get('mcpServers')

    if config_data:
        print(Fore.GREEN + 'Loaded MCP Configuration.')
        print(f'Identified MCPS: {config_data.keys()}')
        for key, value in config_data.items():
            print(f'Running {key} MCP')
            command = value.get('command', None)
            args = value.get('args', None)
            env_vars = value.get('env', None)

            if command and args is None:
                print(f'{key} MCP Do not have a Commands & Args configured. Skipping.')
                pass

            if args:
                run_mcp = subprocess.Popen(command, *args, text=True)




app = FastAPI(lifespan=lifespan)
app.include_router(mcp_actions.router)
