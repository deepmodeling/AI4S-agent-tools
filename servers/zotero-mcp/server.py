#!/usr/bin/env python3
"""
Zotero MCP Server using the simplified pattern for AI4S-agent-tools.
This server provides integration with Zotero for literature management and semantic search.
"""
import argparse
import os
import sys

# Add src to path to import the actual server implementation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from zotero_mcp.server import mcp

def parse_args():
    """Parse command line arguments for MCP server."""
    parser = argparse.ArgumentParser(description="Zotero MCP Server")
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

if __name__ == "__main__":
    args = parse_args()
    
    # Get transport type from environment variable, default to SSE
    transport_type = os.getenv('MCP_TRANSPORT', 'sse')
    
    # Note: The mcp instance is already configured in src/zotero_mcp/server.py
    # We just need to run it with the appropriate transport settings
    if transport_type == 'stdio':
        mcp.run(transport='stdio')
    else:
        # For SSE and streamable-http, use host and port from args
        mcp.run(transport=transport_type, host=args.host, port=args.port)