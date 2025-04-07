from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from manager.mcp_manager import MCPManager, SERVERS

router = APIRouter(prefix="/mcp_actions", tags=["MCP"])



class MCPConfig(BaseModel):
    name: str
    args: list[str]
    command: str
    env: dict


@router.post("/register_mcp")
async def register_mcp(name, config: MCPConfig):
    if name in SERVERS:
        raise HTTPException(status_code=400, detail="Server already registered.")

    mcp = MCPManager(name, config.model_dump())
    try:
        await mcp.register_mcp()
    except ConnectionError as e:
        raise HTTPException(status_code=500, detail=f"{e}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"{e}")

    return {"mcp_name": f"{name}", "status": "connected"}


@router.get("/list_mcps")
async def list_mcps():
    mcps = list(SERVERS.keys())
    if len(mcps) == 0:
        return 'no available servers'
    return {"mcp_list": mcps}


@router.get("/get_tools/{name}")
async def get_mcp_tools(name: str):
    MCP_OBJECT = SERVERS.get(name, None)
    if MCP_OBJECT is None:
        raise HTTPException(status_code=404, detail=f"The MCP server ({name}) does not exist.")
    return {"mcp_name": name, "tools": MCP_OBJECT.tools_full_data}


@router.post("/kill_session/{name}")
async def kill_session(name: str):
    pass


@router.post("/kill_all_sessions")
async def kill_all_session():
    mcp_count = len(list(SERVERS.keys()))
    try:
        for name in list(SERVERS.keys()):
            try:
                await SERVERS[name].kill_session()
            except Exception as e:
                print(f"[ERROR] Failed to kill {name}: {e}")
        return {"status": "success", "current_mcp_count": len(list(SERVERS.keys())), "pre_call_mcp_count": mcp_count}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail={"status": "error", "current_mcp_count": len(list(SERVERS.keys())),
                                                     "pre_call_mcp_count": mcp_count})
