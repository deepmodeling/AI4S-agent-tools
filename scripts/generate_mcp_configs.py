#!/usr/bin/env python3
"""
Generate MCP configuration files for each tool in the servers directory.
Each tool gets its own mcp-config.json file in its directory.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

def load_json_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """Load and parse a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load {file_path}: {e}")
        return None

def check_has_pyproject(tool_dir: Path) -> bool:
    """Check if the tool directory has a pyproject.toml file."""
    return (tool_dir / "pyproject.toml").exists()

def find_server_script(tool_dir: Path) -> Optional[str]:
    """Find the main server script in the tool directory."""
    # Common server script names
    possible_names = ["server.py", "main.py", "app.py", "__main__.py"]
    
    for name in possible_names:
        if (tool_dir / name).exists():
            return name
    
    # Check in src directory
    src_dir = tool_dir / "src"
    if src_dir.exists():
        for name in possible_names:
            for file in src_dir.rglob(name):
                return str(file.relative_to(tool_dir))
    
    return None

def generate_tool_config(tool_dir: Path) -> Optional[Dict[str, Any]]:
    """Generate MCP configuration for a single tool."""
    metadata_path = tool_dir / "metadata.json"
    
    # Load metadata
    metadata = load_json_file(metadata_path)
    if not metadata:
        return None
    
    tool_name = metadata.get("name", tool_dir.name)
    
    # Check transport support
    transport = metadata.get("transport", ["sse", "stdio"])
    if isinstance(transport, str):
        transport = [transport]
    
    # Only generate stdio config if stdio is supported
    if "stdio" not in transport:
        print(f"  Skipping stdio config for {tool_name} (only supports {transport})")
        return None
    
    # Find the server script
    server_script = find_server_script(tool_dir)
    if not server_script:
        print(f"Warning: No server script found for {tool_name}")
        return None
    
    # Check if it uses uv (has pyproject.toml)
    has_pyproject = check_has_pyproject(tool_dir)
    
    # Build the configuration
    config = {
        "mcpServers": {
            tool_name: {}
        }
    }
    
    server_config = config["mcpServers"][tool_name]
    
    if has_pyproject:
        # Use uv to run the server
        server_config["command"] = "uv"
        server_config["args"] = [
            "run",
            "--directory",
            f"servers/{tool_dir.name}",
            "python",
            server_script
        ]
    else:
        # Direct Python execution
        server_config["command"] = "python"
        server_config["args"] = [server_script]
    
    # Add environment variable for transport type (stdio for Claude Desktop compatibility)
    server_config["env"] = {
        "MCP_TRANSPORT": "stdio"
    }
    
    # Add metadata
    server_config["metadata"] = {
        "name": metadata.get("name", tool_name),
        "description": metadata.get("description", ""),
        "author": metadata.get("author", ""),
        "category": metadata.get("category", "general"),
        "transport": transport
    }
    
    return config

def generate_all_configs(servers_dir: Path) -> Dict[str, Path]:
    """Generate MCP configurations for all tools in the servers directory."""
    generated_configs = {}
    
    # Iterate through all subdirectories in servers/ (sorted alphabetically)
    for tool_dir in sorted(servers_dir.iterdir()):
        if not tool_dir.is_dir():
            continue
        
        # Skip special directories
        if tool_dir.name.startswith('.') or tool_dir.name.startswith('_'):
            continue
        
        # Check if it has metadata.json
        if not (tool_dir / "metadata.json").exists():
            print(f"Skipping {tool_dir.name}: no metadata.json found")
            continue
        
        print(f"Processing {tool_dir.name}...")
        
        # Check if mcp-config.json already exists
        config_path = tool_dir / "mcp-config.json"
        if config_path.exists():
            print(f"  ✓ Using existing {config_path}")
            generated_configs[tool_dir.name] = config_path
            continue
        
        # Generate configuration
        config = generate_tool_config(tool_dir)
        if config:
            # Write configuration to the tool directory
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            generated_configs[tool_dir.name] = config_path
            print(f"  ✓ Generated {config_path}")
        else:
            print(f"  ✗ Failed to generate config for {tool_dir.name}")
    
    return generated_configs

def generate_combined_config(servers_dir: Path, generated_configs: Dict[str, Path]):
    """Generate a combined configuration file that references all individual configs."""
    combined = {
        "version": "1.0.0",
        "description": "AI4S Agent Tools - MCP Server Configurations",
        "tools": []
    }
    
    for tool_name, config_path in sorted(generated_configs.items()):
        # Load the individual config to get metadata
        config = load_json_file(config_path)
        if config and "mcpServers" in config:
            for server_name, server_config in config["mcpServers"].items():
                tool_info = {
                    "name": server_name,
                    "config_path": str(config_path.relative_to(servers_dir.parent)),
                }
                if "metadata" in server_config:
                    tool_info.update(server_config["metadata"])
                combined["tools"].append(tool_info)
    
    # Write combined configuration
    combined_path = servers_dir.parent / "all-mcp-configs.json"
    with open(combined_path, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Generated combined configuration: {combined_path}")
    return combined_path

def main():
    """Main entry point."""
    # Get the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    servers_dir = project_root / "servers"
    
    if not servers_dir.exists():
        print(f"Error: servers directory not found at {servers_dir}")
        return 1
    
    print(f"Generating MCP configurations for tools in {servers_dir}\n")
    
    # Generate individual configurations
    generated_configs = generate_all_configs(servers_dir)
    
    if generated_configs:
        print(f"\n✅ Successfully generated {len(generated_configs)} MCP configurations")
    else:
        print("\n⚠️ No configurations were generated")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())