import os
import logging
from dotenv import load_dotenv, find_dotenv

from mcp.server import FastMCP

load_dotenv(find_dotenv())

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

mcp_server = FastMCP(os.environ["CHEMBL_MCP_HOST"],
                     port=os.environ["CHEMBL_MCP_PORT"])

tools = [
]

for tool in tools:
    mcp_server.tool()(tool)
