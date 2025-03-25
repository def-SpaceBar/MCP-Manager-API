from typing import Dict
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather", host='127.0.0.1')


@mcp.tool(
    name="echo_tool",
    description="Bamba"
)
async def echo_tool(text: str) -> Dict[str, str]:
    """

    :param text:
    :return:
    """
    return {"echoed_text": text}



if __name__ == "__main__":
    mcp.run(transport="stdio")
