"""
Utility module for AI4S MCP servers.
Provides decorators and helper functions to simplify server development.
"""
import argparse
import inspect
import logging
from typing import Dict, Optional, Callable, Any, List
from functools import wraps

# Server metadata registry
_server_registry: Dict[str, Dict[str, Any]] = {}


def mcp_server(name: str, description: str, author: str, category: Optional[str] = None):
    """
    Minimal decorator to register MCP server metadata.
    
    Args:
        name: Server name
        description: Server description  
        author: Tool author/contributor
        category: Tool category (e.g., "biology", "chemistry", "materials", "physics", "research")
    
    Usage:
        @mcp_server("PubChem", "Chemical compound data retrieval", author="@deepmodeling", category="chemistry")
        def create_server(host="0.0.0.0", port=50001):
            mcp = FastMCP("pubchem", host=host, port=port)
            # Define tools...
            return mcp
    """
    def decorator(func):
        # Register minimal metadata
        metadata = {
            "name": name,
            "description": description,
            "author": author
        }
        
        # Add category if specified
        if category:
            metadata["category"] = category.lower()
        
        _server_registry[name] = metadata
        
        # Attach metadata to function
        func.__mcp_metadata__ = metadata
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def setup_server(create_server_func: Optional[Callable] = None) -> Any:
    """
    Minimal setup for MCP server with command line argument parsing.
    
    Args:
        create_server_func: Function decorated with @mcp_server (optional, auto-detected)
    
    Returns:
        The configured MCP server instance
    
    Usage:
        if __name__ == "__main__":
            setup_server().run()  # One line!
    """
    # Find the decorated function if not provided
    if create_server_func is None:
        frame = inspect.currentframe()
        if frame and frame.f_back:
            for name, obj in frame.f_back.f_globals.items():
                if callable(obj) and hasattr(obj, '__mcp_metadata__'):
                    create_server_func = obj
                    break
    
    if create_server_func is None or not hasattr(create_server_func, '__mcp_metadata__'):
        raise ValueError("No function decorated with @mcp_server found")
    
    metadata = create_server_func.__mcp_metadata__
    
    # Simple argument parser
    parser = argparse.ArgumentParser(description=metadata['description'])
    parser.add_argument('--port', type=int, required=True, help='Server port')
    parser.add_argument('--host', default='0.0.0.0', help='Server host')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logging.info(f"Starting {metadata['name']} on {args.host}:{args.port}")
    
    # Create and return server
    return create_server_func(host=args.host, port=args.port)


def get_server_registry() -> Dict[str, Dict[str, Any]]:
    """Get the current server registry for tool discovery."""
    return _server_registry.copy()


