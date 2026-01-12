"""
仓库转换工具函数 - 自动化 GitHub 仓库到 MCP 格式的转换
"""

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


def clone_repository(repo_url: str, target_dir: str) -> str:
    """
    克隆 GitHub 仓库到指定目录
    
    Args:
        repo_url: GitHub 仓库 URL
        target_dir: 目标目录名（在 servers/ 下）
        
    Returns:
        str: 克隆结果信息
    """
    servers_path = Path(f"servers/{target_dir}")
    
    # 如果目录已存在，先删除
    if servers_path.exists():
        subprocess.run(["rm", "-rf", str(servers_path)], check=True)
    
    # 克隆仓库
    result = subprocess.run(
        ["git", "clone", repo_url, str(servers_path)],
        capture_output=True, text=True
    )
    
    if result.returncode != 0:
        return f"❌ 克隆失败: {result.stderr}"
    
    return f"✓ 仓库已克隆到: {servers_path}"


def analyze_repository(tool_name: str) -> Dict[str, Any]:
    """
    分析仓库结构，识别 MCP 相关文件
    
    Args:
        tool_name: 工具目录名
        
    Returns:
        Dict: 包含分析结果的字典
    """
    repo_path = Path(f"servers/{tool_name}")
    analysis = {
        "main_file": None,
        "has_mcp": False,
        "tool_functions": [],
        "dependencies": [],
        "has_git": False,
        "description": ""
    }
    
    # 查找主文件（包含 FastMCP 或 mcp 的 Python 文件）
    for py_file in repo_path.glob("*.py"):
        content = py_file.read_text(encoding='utf-8')
        if 'FastMCP' in content or 'mcp.tool' in content:
            analysis["main_file"] = py_file.name
            analysis["has_mcp"] = True
            
            # 提取工具函数名
            tool_pattern = r'@mcp\.tool\(\)\s*\ndef\s+(\w+)'
            analysis["tool_functions"] = re.findall(tool_pattern, content)
            break
    
    # 检查依赖文件
    pyproject = repo_path / "pyproject.toml"
    requirements = repo_path / "requirements.txt"
    
    if pyproject.exists():
        content = pyproject.read_text(encoding='utf-8')
        # 简单提取依赖（实际可能需要 toml 解析器）
        deps_match = re.search(r'dependencies\s*=\s*\[(.*?)\]', content, re.DOTALL)
        if deps_match:
            deps = deps_match.group(1)
            analysis["dependencies"] = [
                dep.strip().strip('"\'') 
                for dep in deps.split(',') 
                if dep.strip()
            ]
    elif requirements.exists():
        analysis["dependencies"] = [
            line.strip() 
            for line in requirements.read_text().splitlines() 
            if line.strip() and not line.startswith('#')
        ]
    
    # 检查是否有 .git 目录
    analysis["has_git"] = (repo_path / ".git").exists()
    
    # 尝试从 README 提取描述
    readme = repo_path / "README.md"
    if readme.exists():
        content = readme.read_text(encoding='utf-8')
        # 取第一个非标题段落作为描述
        lines = content.splitlines()
        for line in lines:
            if line.strip() and not line.startswith('#'):
                analysis["description"] = line.strip()[:100]  # 限制长度
                break
    
    return analysis


def convert_to_mcp_format(tool_name: str, analysis: Dict[str, Any]) -> str:
    """
    将仓库转换为 MCP 标准格式
    
    Args:
        tool_name: 工具目录名
        analysis: 仓库分析结果
        
    Returns:
        str: 转换结果信息
    """
    repo_path = Path(f"servers/{tool_name}")
    messages = []
    
    # 1. 重命名主文件为 server.py
    if analysis["main_file"] and analysis["main_file"] != "server.py":
        old_path = repo_path / analysis["main_file"]
        new_path = repo_path / "server.py"
        old_path.rename(new_path)
        messages.append(f"✓ 重命名 {analysis['main_file']} → server.py")
    
    # 2. 更新 server.py 添加参数解析
    server_path = repo_path / "server.py"
    if server_path.exists():
        content = server_path.read_text(encoding='utf-8')
        
        # 检查是否已有参数解析
        if 'parse_args' not in content:
            # 添加 argparse 导入
            if 'import argparse' not in content:
                # 在其他导入后添加
                import_match = re.search(r'((?:from .+ import .+\n|import .+\n)+)', content)
                if import_match:
                    imports = import_match.group(1)
                    new_imports = imports + "import argparse\n"
                    content = content.replace(imports, new_imports)
            
            # 添加参数解析函数
            parse_args_code = '''
def parse_args():
    """Parse command line arguments for MCP server."""
    parser = argparse.ArgumentParser(description="{} MCP Server")
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
'''.format(tool_name)
            
            # 在 FastMCP 实例化前插入
            mcp_pattern = r'(mcp\s*=\s*FastMCP\()'
            if re.search(mcp_pattern, content):
                content = re.sub(mcp_pattern, parse_args_code + r'\n\1', content)
            
            server_path.write_text(content, encoding='utf-8')
            messages.append("✓ 添加了参数解析功能")
    
    # 3. 清理 Git 历史
    if analysis["has_git"]:
        git_dir = repo_path / ".git"
        subprocess.run(["rm", "-rf", str(git_dir)], check=True)
        messages.append("✓ 清理了 Git 历史")
    
    return "\n".join(messages)


def extract_author_from_repo(repo_url: str) -> str:
    """
    从仓库 URL 提取作者 GitHub 用户名
    
    Args:
        repo_url: GitHub 仓库 URL
        
    Returns:
        str: GitHub 用户名
    """
    # 匹配 github.com/username/repo 模式
    match = re.search(r'github\.com[:/]([^/]+)/', repo_url)
    if match:
        return match.group(1)
    return "unknown"


def auto_categorize(tool_name: str, description: str, tool_functions: List[str]) -> str:
    """
    根据工具名、描述和函数名自动推断类别
    
    Args:
        tool_name: 工具名称
        description: 工具描述
        tool_functions: 工具函数列表
        
    Returns:
        str: 推断的类别
    """
    # 关键词到类别的映射
    category_keywords = {
        "chemistry": ["molecule", "mol", "chem", "reaction", "compound", "rdkit", "pymol", "smiles"],
        "biology": ["protein", "dna", "rna", "sequence", "gene", "cell", "bio"],
        "physics": ["quantum", "physics", "energy", "force", "particle"],
        "materials": ["material", "crystal", "structure", "lattice"],
        "simulation": ["simulate", "dynamics", "md", "monte carlo"],
        "data": ["data", "analysis", "process", "visualiz"],
        "machine-learning": ["ml", "ai", "model", "train", "predict", "neural"],
        "research": ["paper", "arxiv", "pubmed", "literature", "search"],
    }
    
    # 合并所有文本进行匹配
    text = f"{tool_name} {description} {' '.join(tool_functions)}".lower()
    
    # 计算每个类别的匹配分数
    scores = {}
    for category, keywords in category_keywords.items():
        score = sum(1 for keyword in keywords if keyword in text)
        if score > 0:
            scores[category] = score
    
    # 返回得分最高的类别
    if scores:
        return max(scores, key=scores.get)
    return "general"


def create_metadata_from_analysis(
    tool_name: str, 
    repo_url: str,
    analysis: Dict[str, Any],
    override_author: Optional[str] = None,
    override_category: Optional[str] = None
) -> str:
    """
    基于仓库分析结果创建 metadata.json
    
    Args:
        tool_name: 工具名称
        repo_url: 仓库 URL
        analysis: 仓库分析结果
        override_author: 覆盖自动检测的作者
        override_category: 覆盖自动检测的类别
        
    Returns:
        str: 操作结果
    """
    # 自动提取信息
    author = override_author or extract_author_from_repo(repo_url)
    category = override_category or auto_categorize(
        tool_name, 
        analysis["description"], 
        analysis["tool_functions"]
    )
    
    metadata = {
        "name": tool_name,
        "description": analysis["description"] or f"MCP tools converted from {repo_url}",
        "author": f"@{author}" if not author.startswith("@") else author,
        "category": category,
        "transport": ["sse", "stdio"],
        "tools": analysis["tool_functions"]
    }
    
    # 写入文件
    file_path = Path(f"servers/{tool_name}/metadata.json")
    file_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), 
        encoding='utf-8'
    )
    
    return f"✓ metadata.json 已创建: {file_path}"


def convert_repo_to_mcp(
    repo_url: str,
    tool_name: Optional[str] = None,
    author: Optional[str] = None,
    category: Optional[str] = None
) -> Dict[str, Any]:
    """
    完整的仓库转换流程
    
    Args:
        repo_url: GitHub 仓库 URL
        tool_name: 工具名称（可选，默认从 URL 提取）
        author: 作者 GitHub 用户名（可选，默认从 URL 提取）
        category: 工具类别（可选，默认自动推断）
        
    Returns:
        Dict: 包含转换结果的字典
    """
    # 从 URL 提取默认工具名
    if not tool_name:
        match = re.search(r'/([^/]+?)(?:\.git)?$', repo_url)
        tool_name = match.group(1) if match else "unknown_tool"
    
    results = {"status": "success", "messages": []}
    
    try:
        # 1. 克隆仓库
        clone_result = clone_repository(repo_url, tool_name)
        results["messages"].append(clone_result)
        
        # 2. 分析仓库
        analysis = analyze_repository(tool_name)
        results["analysis"] = analysis
        
        if not analysis["has_mcp"]:
            results["messages"].append("⚠️ 警告: 未检测到 MCP 相关代码")
        
        # 3. 转换格式
        convert_result = convert_to_mcp_format(tool_name, analysis)
        if convert_result:
            results["messages"].append(convert_result)
        
        # 4. 创建 metadata.json
        metadata_result = create_metadata_from_analysis(
            tool_name, repo_url, analysis, author, category
        )
        results["messages"].append(metadata_result)
        
        # 5. 添加到 git
        subprocess.run(
            ["git", "add", f"servers/{tool_name}"],
            cwd=".", check=True
        )
        results["messages"].append(f"✓ 已添加到 git 暂存区")
        
    except Exception as e:
        results["status"] = "error"
        results["messages"].append(f"❌ 错误: {str(e)}")
    
    return results