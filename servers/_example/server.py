#!/usr/bin/env python3
"""
Example MCP Server using the new simplified pattern.
This demonstrates how to create a new AI4S tool with tools defined at module level.
"""
import argparse
import os

from mcp.server.fastmcp import FastMCP

def parse_args():
    """Parse command line arguments for MCP server."""
    parser = argparse.ArgumentParser(description="MCP Server")
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
mcp = FastMCP("example",host=args.host,port=args.port)

# Define tools at module level
@mcp.tool()
def example_calculation(x: float, y: float) -> dict:
    """
    Perform an example calculation.
    
    Args:
        x: First number
        y: Second number
        
    Returns:
        Dictionary with sum, product, and ratio
    """
    return {
        "sum": x + y,
        "product": x * y,
        "ratio": x / y if y != 0 else "undefined"
    }

@mcp.tool()
def demo_function(message: str) -> str:
    """
    A demo function that echoes the input message.
    
    Args:
        message: The message to echo
        
    Returns:
        The echoed message with a prefix
    """
    return f"Echo: {message}"


if __name__ == "__main__":
    # Get transport type from environment variable, default to SSE
    transport_type = os.getenv('MCP_TRANSPORT', 'sse')
    mcp.run(transport=transport_type)