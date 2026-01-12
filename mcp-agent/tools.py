"""工具函数 - 为MCP Agent提供文件生成功能"""

import json
from pathlib import Path
from typing import List


# ========== CodeAgent工具 - 生成代码文件 ==========

def create_server(tool_name: str, tool_functions: str, additional_imports: str = "") -> str:
    """
    生成并写入MCP服务器的server.py文件
    
    该函数会在servers/[tool_name]/目录下创建server.py文件，包含完整的MCP服务器实现。
    
    Args:
        tool_name: 工具名称（同时作为目录名）
        tool_functions: 完整的工具函数代码（每个函数前需包含@mcp.tool()装饰器）
                       模型应生成可直接运行的完整函数定义
        additional_imports: 工具函数所需的额外导入语句（如numpy, pandas等）
        
    Returns:
        str: 操作结果消息，包含文件路径
        
    Example:
        >>> create_server(
        ...     tool_name="molecule_calculator",
        ...     tool_functions='''
        ... @mcp.tool()
        ... def calculate_molecular_weight(smiles: str) -> Dict[str, Any]:
        ...     \"\"\"Calculate molecular weight from SMILES\"\"\"
        ...     from rdkit import Chem
        ...     mol = Chem.MolFromSmiles(smiles)
        ...     return {"weight": Chem.Descriptors.MolWt(mol)}
        ... ''',
        ...     additional_imports="from rdkit import Chem\\nfrom rdkit.Chem import Descriptors"
        ... )
    """
    # 基础导入
    base_imports = """import argparse
import os
from typing import Dict, Any
from mcp.server.fastmcp import FastMCP"""
    
    # 合并导入语句
    if additional_imports:
        imports_section = f"{base_imports}\n{additional_imports}"
    else:
        imports_section = base_imports
    
    content = f'''#!/usr/bin/env python3
"""
MCP Server for {tool_name}
Generated following AI4S-agent-tools CONTRIBUTING.md standards.
"""
{imports_section}

def parse_args():
    """Parse command line arguments for MCP server."""
    parser = argparse.ArgumentParser(description="{tool_name} MCP Server")
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
mcp = FastMCP("{tool_name}", host=args.host, port=args.port)

{tool_functions}

if __name__ == "__main__":
    # Get transport type from environment variable, default to SSE
    transport_type = os.getenv('MCP_TRANSPORT', 'sse')
    mcp.run(transport=transport_type)
'''
    
    # 创建目录并写入文件
    file_path = Path(f"servers/{tool_name}/server.py")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding='utf-8')
    
    return f"✓ server.py已创建: {file_path}"


def create_metadata(tool_name: str, description: str, author: str, category: str, tools: List[str] = []) -> str:
    """
    生成并写入MCP服务器的metadata.json文件
    
    该函数会在servers/[tool_name]/目录下创建metadata.json文件，包含工具的元数据信息。
    
    Args:
        tool_name: 工具名称（同时作为目录名）
        description: 工具的详细描述
        author: GitHub用户名（自动添加@前缀）
        category: 工具类别，可选值:
                 - chemistry: 化学相关工具
                 - physics: 物理相关工具
                 - biology: 生物相关工具
                 - materials: 材料科学工具
                 - simulation: 模拟仿真工具
                 - data: 数据分析工具
                 - machine-learning: 机器学习工具
                 - research: 研究辅助工具
        tools: 工具函数名称列表（应与server.py中@mcp.tool()装饰的函数名一致）
        
    Returns:
        str: 操作结果消息，包含文件路径
        
    Example:
        >>> create_metadata(
        ...     tool_name="molecule_calculator",
        ...     description="Calculate molecular properties",
        ...     author="john_doe",
        ...     category="chemistry",
        ...     tools=["calculate_molecular_weight", "calculate_logp"]
        ... )
    """
    metadata = {
        "name": tool_name,
        "description": description,
        "author": author if author.startswith("@") else f"@{author}",
        "category": category,
        "transport": ["sse", "stdio"]
    }
    if tools:
        metadata["tools"] = tools
    
    content = json.dumps(metadata, indent=2, ensure_ascii=False)
    
    # 创建目录并写入文件
    file_path = Path(f"servers/{tool_name}/metadata.json")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding='utf-8')
    
    return f"✓ metadata.json已创建: {file_path}"


def create_pyproject(tool_name: str, description: str, dependencies: List[str] = []) -> str:
    """
    生成并写入MCP服务器的pyproject.toml文件
    
    该函数会在servers/[tool_name]/目录下创建pyproject.toml文件，用于管理Python项目依赖。
    自动包含fastmcp作为基础依赖。
    
    Args:
        tool_name: 工具名称（同时作为目录名和项目名）
        description: 项目描述
        dependencies: Python包依赖列表，格式如:
                     - ["numpy>=1.20.0", "pandas", "scipy==1.9.0"]
                     - 会自动添加fastmcp>=2.3.0作为基础依赖
        
    Returns:
        str: 操作结果消息，包含文件路径
        
    Example:
        >>> create_pyproject(
        ...     tool_name="molecule_calculator",
        ...     description="Molecular property calculator",
        ...     dependencies=["rdkit", "numpy>=1.20.0", "pandas"]
        ... )
    """
    deps = dependencies or []
    # 确保fastmcp是第一个依赖
    if not any("fastmcp" in dep for dep in deps):
        deps.insert(0, "fastmcp>=2.3.0")
    
    deps_str = ",\n    ".join([f'"{dep}"' for dep in deps])
    
    content = f'''[project]
name = "{tool_name}"
version = "0.1.0"
description = "{description}"
requires-python = ">=3.11"
dependencies = [
    {deps_str}
]

'''
    
    # 创建目录并写入文件
    file_path = Path(f"servers/{tool_name}/pyproject.toml")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding='utf-8')
    
    return f"✓ pyproject.toml已创建: {file_path}"
