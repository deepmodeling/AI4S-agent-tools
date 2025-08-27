import sys
import os
from pathlib import Path

import argparse
from mcp.server.fastmcp import FastMCP
from python_version.mcp_server import handle_tool_call

def parse_args():
    """Parse command line arguments for MCP server."""
    parser = argparse.ArgumentParser(description="DPA Calculator MCP Server")
    parser.add_argument('--port', type=int, default=50001, help='Server port (default: 50001)')
    parser.add_argument('--host', default='0.0.0.0', help='Server host (default: 0.0.0.0)')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    try:
        args = parser.parse_args()
    except SystemExit:
        class Args:
            port = 50001
            host = '0.0.0.0'
            log_level = 'INFO'
        args = Args()
    return args

args = parse_args()
mcp = FastMCP("pubchem", host=args.host, port=args.port)

@mcp.tool()
def get_pubchem_data(query: str, format: str = 'JSON', include_3d: bool = False) -> str:
    """
    Get PubChem data for a given query.
    """
    return handle_tool_call("get_pubchem_data", {"query": query, "format": format, "include_3d": include_3d})

@mcp.tool()
def download_structure(cid: str, format: str = 'sdf') -> str:
    """
    Download a structure from PubChem.
    """
    return handle_tool_call("download_structure", {"cid": cid, "format": format})

if __name__ == "__main__":
    # Get transport type from environment variable, default to SSE
    transport_type = os.getenv('MCP_TRANSPORT', 'sse')
    mcp.run(transport=transport_type)



