#!/usr/bin/env python3
"""
ABACUS MCP Server - Bridge between AI models and first principles calculations.
Provides tools for ABACUS computational jobs including input preparation, job submission, and analysis.
"""
import sys
import os
from pathlib import Path
import argparse
from mcp.server.fastmcp import FastMCP

# Import ABACUS-specific modules
sys.path.append(str(Path(__file__).resolve().parent))
from src.abacusagent.main import load_tools
from src.abacusagent.env import set_envs

def parse_args():
    """Parse command line arguments for MCP server."""
    parser = argparse.ArgumentParser(description="ABACUS MCP Server")
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
mcp = FastMCP("ABACUS", host=args.host, port=args.port)


load_tools()

if __name__ == "__main__":
    set_envs()
    # Get transport type from environment variable, default to SSE
    transport_type = os.getenv('MCP_TRANSPORT', 'sse')
    mcp.run(transport=transport_type)
