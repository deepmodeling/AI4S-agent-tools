#!/usr/bin/env python3
"""
Generate TOOLS.json by scanning all MCP servers in the repository.
This script extracts metadata from servers decorated with @mcp_server
and reads additional information from pyproject.toml files.
"""
import json
import os
import sys
import importlib.util
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional
import tomllib
import logging


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path to import servers
sys.path.insert(0, str(Path(__file__).parent.parent))




def read_pyproject_toml(server_dir: Path) -> Dict[str, Any]:
    """Read pyproject.toml to extract dependencies and other metadata."""
    pyproject_path = server_dir / "pyproject.toml"
    if not pyproject_path.exists():
        return {}
    
    try:
        with open(pyproject_path, 'rb') as f:
            data = tomllib.load(f)
        
        result = {}
        
        # Extract dependencies
        if 'project' in data:
            project = data['project']
            
            if 'dependencies' in project:
                result['dependencies'] = project['dependencies']
            
            if 'description' in project:
                result['description'] = project['description']
            
            if 'name' in project:
                result['package_name'] = project['name']
        
        return result
    except Exception as e:
        logger.debug(f"Failed to read {pyproject_path}: {e}")
        return {}


def extract_tools_from_server(server_dir: Path) -> List[str]:
    """Extract tool names from server implementation by searching all Python files."""
    tools = []
    
    # Recursively search all .py files in the server directory
    for py_file in server_dir.rglob('*.py'):
        # Skip __pycache__, .venv, and other virtual environment directories
        path_parts = py_file.parts
        if any(part in ['__pycache__', '.venv', 'venv', '.env', 'env', 'site-packages'] for part in path_parts):
            continue
            
        try:
            # Read the file content to check for commented lines
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.splitlines()
            
            # Parse the AST
            tree = ast.parse(content)
            
            # Look for any function decorated with @mcp.tool
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and hasattr(node, 'lineno'):
                    # Check if this function has @mcp.tool decorator
                    has_mcp_tool = False
                    decorator_line = None
                    
                    for decorator in node.decorator_list:
                        if hasattr(decorator, 'lineno'):
                            decorator_line = decorator.lineno - 1  # Convert to 0-based
                            
                        # Check for @mcp.tool() pattern
                        if isinstance(decorator, ast.Call):
                            if (isinstance(decorator.func, ast.Attribute) and 
                                decorator.func.attr == 'tool' and
                                isinstance(decorator.func.value, ast.Name) and
                                decorator.func.value.id == 'mcp'):
                                has_mcp_tool = True
                        # Also check for @mcp.tool without parentheses
                        elif isinstance(decorator, ast.Attribute):
                            if (decorator.attr == 'tool' and
                                isinstance(decorator.value, ast.Name) and
                                decorator.value.id == 'mcp'):
                                has_mcp_tool = True
                    
                    # If we found @mcp.tool, check if it's not commented
                    if has_mcp_tool and decorator_line is not None and decorator_line < len(lines):
                        line = lines[decorator_line].strip()
                        # Skip if the line is commented
                        if not line.startswith('#'):
                            tools.append(node.name)
                            logger.debug(f"  Found tool '{node.name}' in {py_file.relative_to(server_dir)}")
                                
        except Exception as e:
            logger.debug(f"Failed to parse {py_file}: {e}")
    
    return list(set(tools))  # Remove duplicates


def scan_server_directory(server_dir: Path) -> Optional[Dict[str, Any]]:
    """Scan a server directory and extract all metadata."""
    # Only read from metadata.json
    metadata_file = server_dir / "metadata.json"
    
    if not metadata_file.exists():
        logger.warning(f"No metadata.json found in {server_dir}")
        return None
    
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        logger.debug(f"  Loaded metadata from {metadata_file}")
    except Exception as e:
        logger.error(f"  Failed to read {metadata_file}: {e}")
        return None
    
    # Add path information
    metadata['path'] = f"servers/{server_dir.name}"
    
    # Extract transport support (default to both)
    transport = metadata.get('transport', ['sse', 'stdio'])
    if isinstance(transport, str):
        transport = [transport]
    metadata['transport'] = transport
    
    # Enhance with pyproject.toml data
    pyproject_data = read_pyproject_toml(server_dir)
    
    # Add description from pyproject if not already set
    if 'description' in pyproject_data and 'description' not in metadata:
        metadata['description'] = pyproject_data['description']
    
    # Set defaults
    metadata.setdefault('name', server_dir.name)
    metadata.setdefault('description', f"{server_dir.name} MCP server")
    metadata.setdefault('author', '@unknown')
    
    # Generate start command based on transport support and server.py existence
    server_py_path = server_dir / "server.py"
    if server_py_path.exists():
        # Generate commands based on transport support
        commands = []
        if 'sse' in metadata['transport']:
            commands.append(f"# SSE mode\ncd {metadata['path']} && python server.py --port <PORT>")
        if 'stdio' in metadata['transport']:
            commands.append(f"# stdio mode\ncd {metadata['path']} && MCP_TRANSPORT=stdio python server.py")
        metadata['start_command'] = '\n'.join(commands) if commands else "See README for details"
    else:
        # For closed-source or external tools without server.py
        metadata['start_command'] = "See README for details"
    
    # Generate install command based on whether server.py exists
    if server_py_path.exists():
        metadata['install_command'] = f"cd {metadata['path']} && uv sync"
    else:
        # For closed-source tools without server.py
        author = metadata.get('author', '@author')
        metadata['install_command'] = f"Contact {author} for access"
    
    return metadata


def load_categories() -> Dict[str, Any]:
    """Load category definitions from config file."""
    config_path = Path(__file__).parent.parent / "config" / "categories.json"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"categories": {}, "default_category": "research"}


def categorize_tool(name: str, description: str, categories_config: Dict[str, Any]) -> str:
    """Automatically categorize a tool based on its name and description."""
    text = (name + " " + description).lower()
    
    # Keyword-based categorization
    category_keywords = {
        "simulation": ["molecule", "atom", "simulation", "dpa", "deepmd", "dynamics", "md"],
        "materials": ["material", "thermoelectric", "property", "crystal", "structure"],
        "chemistry": ["compound", "pubchem", "chemical", "catalysis", "reaction", "adsorption"],
        "research": ["paper", "search", "arxiv", "document", "rag", "literature"],
        "data": ["data", "analysis", "visualization", "plot", "graph"],
        "machine-learning": ["ml", "machine learning", "ai", "model", "neural", "network"]
    }
    
    for category, keywords in category_keywords.items():
        if any(keyword in text for keyword in keywords):
            return category
    
    return categories_config.get("default_category", "research")


def generate_tools_json(root_dir: Path) -> Dict[str, Any]:
    """Generate the complete TOOLS.json structure."""
    servers_dir = root_dir / "servers"
    
    if not servers_dir.exists():
        logger.error(f"Servers directory not found: {servers_dir}")
        return {}
    
    # Load category configuration
    categories_config = load_categories()
    
    tools = []
    
    # Scan each server directory
    for server_dir in sorted(servers_dir.iterdir()):
        if not server_dir.is_dir() or server_dir.name.startswith('_'):
            continue
        
        logger.info(f"Scanning {server_dir.name}...")
        metadata = scan_server_directory(server_dir)
        
        if metadata:
            # Extract available tools from code
            tools_from_code = extract_tools_from_server(server_dir)
            
            # Get tools from metadata.json if present
            tools_from_metadata = metadata.get('tools', [])
            if isinstance(tools_from_metadata, str):
                tools_from_metadata = [tools_from_metadata]
            
            # Combine tools from both sources (union)
            all_tools = list(set(tools_from_code + tools_from_metadata))
            metadata['tools'] = all_tools
            
            # Use category from decorator if available, otherwise auto-categorize
            if 'category' not in metadata:
                metadata['category'] = categorize_tool(
                    metadata.get('name', ''),
                    metadata.get('description', ''),
                    categories_config
                )
            
            # Validate category exists in config
            if metadata['category'] not in categories_config['categories']:
                logger.warning(f"  Unknown category '{metadata['category']}' for {metadata['name']}, using default")
                metadata['category'] = categories_config.get('default_category', 'general')
            
            tools.append(metadata)
            logger.info(f"  Found: {metadata['name']} with {len(all_tools)} tools (category: {metadata['category']})")
            if tools_from_metadata:
                logger.info(f"    Tools from metadata.json: {tools_from_metadata}")
            if tools_from_code:
                logger.info(f"    Tools from code scanning: {tools_from_code}")
    
    # Sort tools by name (case-insensitive)
    tools.sort(key=lambda x: x['name'].lower())
    
    # Sort categories for consistent output
    sorted_categories = dict(sorted(categories_config["categories"].items()))
    
    return {
        "version": "1.0.0",
        "description": "AI4S Agent Tools Registry - A collection of MCP servers for scientific computing",
        "categories": sorted_categories,
        "tools": tools
    }


def main():
    """Main function to generate TOOLS.json."""
    # Get the repository root
    root_dir = Path(__file__).parent.parent
    
    logger.info(f"Generating TOOLS.json for repository: {root_dir}")
    
    # Generate the data
    tools_data = generate_tools_json(root_dir)
    
    if not tools_data.get('tools'):
        logger.error("No tools found!")
        return 1
    
    # Write to file
    output_path = root_dir / "data" / "tools.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tools_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nSuccessfully generated {output_path}")
    logger.info(f"Found {len(tools_data['tools'])} tools")
    
    # Print summary
    print("\nTools Summary:")
    print("-" * 70)
    print(f"{'Name':20} {'Author':20} {'Tools':30}")
    print("-" * 70)
    for tool in tools_data['tools']:
        tools_str = ', '.join(tool.get('tools', [])[:3])
        if len(tool.get('tools', [])) > 3:
            tools_str += '...'
        print(f"{tool['name']:20} {tool.get('author', '@unknown'):20} {tools_str:30}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())