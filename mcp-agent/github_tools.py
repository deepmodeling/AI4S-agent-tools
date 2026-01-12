"""
GitHub Agent 工具函数 - 提供仓库探索、分析和转换功能
"""

import os
import json
import subprocess
import re
from pathlib import Path
from typing import Dict, Any, List, Optional


def explore_repository(repo_path: str) -> Dict[str, Any]:
    """
    探索仓库结构，获取项目概览
    
    该函数会分析指定仓库的整体结构，包括文件类型分布、关键文件位置、
    项目规模等信息，帮助 Agent 理解项目组织方式。
    
    Args:
        repo_path: 仓库路径（相对于 servers/ 目录）
        
    Returns:
        Dict: 包含仓库结构信息的字典
        
    Example:
        >>> explore_repository("pymol-mcp")
        {
            "total_files": 15,
            "python_files": 8,
            "has_readme": True,
            "has_server_py": True,
            "key_directories": ["utils", "tests"],
            "project_type": "mcp_server"
        }
    """
    full_path = Path(f"servers/{repo_path}")
    if not full_path.exists():
        return {"error": f"仓库不存在: {repo_path}"}
    
    result = {
        "repo_name": repo_path,
        "total_files": 0,
        "python_files": 0,
        "config_files": [],
        "key_files": [],
        "directories": [],
        "has_readme": False,
        "has_server_py": False,
        "has_metadata_json": False,
        "has_git": False,
        "file_tree": {}
    }
    
    # 构建文件树并收集统计信息
    def build_tree(path: Path, level: int = 0, max_level: int = 3) -> Dict:
        if level > max_level:
            return {}
        
        tree = {}
        try:
            for item in sorted(path.iterdir()):
                if item.name.startswith('.') and item.name != '.git':
                    continue
                    
                if item.is_dir():
                    if item.name == '.git':
                        result["has_git"] = True
                    else:
                        result["directories"].append(str(item.relative_to(full_path)))
                        tree[item.name + "/"] = build_tree(item, level + 1, max_level)
                else:
                    result["total_files"] += 1
                    tree[item.name] = item.stat().st_size
                    
                    # 统计文件类型
                    if item.suffix == '.py':
                        result["python_files"] += 1
                    elif item.suffix in ['.toml', '.json', '.yml', '.yaml', '.ini']:
                        result["config_files"].append(str(item.relative_to(full_path)))
                    
                    # 检查关键文件
                    if item.name.lower() == 'readme.md':
                        result["has_readme"] = True
                        result["key_files"].append(str(item.relative_to(full_path)))
                    elif item.name == 'server.py':
                        result["has_server_py"] = True
                        result["key_files"].append(str(item.relative_to(full_path)))
                    elif item.name == 'metadata.json':
                        result["has_metadata_json"] = True
                        result["key_files"].append(str(item.relative_to(full_path)))
                    elif item.name in ['main.py', 'setup.py', 'pyproject.toml', 'requirements.txt']:
                        result["key_files"].append(str(item.relative_to(full_path)))
                        
        except PermissionError:
            tree["<权限不足>"] = None
            
        return tree
    
    result["file_tree"] = build_tree(full_path)
    
    # 判断项目类型
    if result["python_files"] > 0:
        if result["has_server_py"] or any('mcp' in f.lower() for f in result["key_files"]):
            result["project_type"] = "mcp_server"
        else:
            result["project_type"] = "python_project"
    else:
        result["project_type"] = "unknown"
    
    return result


def read_file_content(repo_path: str, file_path: str, start_line: Optional[int] = None, 
                     end_line: Optional[int] = None) -> Dict[str, Any]:
    """
    读取仓库中的文件内容
    
    支持读取完整文件或指定行范围，自动识别文件类型并提供语法高亮提示。
    
    Args:
        repo_path: 仓库路径（相对于 servers/ 目录）
        file_path: 文件路径（相对于仓库根目录）
        start_line: 起始行号（从1开始），可选
        end_line: 结束行号（包含），可选
        
    Returns:
        Dict: 包含文件内容和元信息
        
    Example:
        >>> read_file_content("pymol-mcp", "server.py", 1, 50)
        {
            "content": "#!/usr/bin/env python3\\n...",
            "language": "python",
            "total_lines": 150,
            "displayed_lines": "1-50"
        }
    """
    full_file_path = Path(f"servers/{repo_path}/{file_path}")
    
    if not full_file_path.exists():
        return {"error": f"文件不存在: {file_path}"}
    
    if not full_file_path.is_file():
        return {"error": f"不是文件: {file_path}"}
    
    try:
        with open(full_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        
        # 确定要读取的行范围
        if start_line is None:
            start_line = 1
        if end_line is None:
            end_line = total_lines
            
        # 确保行号有效
        start_line = max(1, min(start_line, total_lines))
        end_line = max(start_line, min(end_line, total_lines))
        
        # 提取内容
        content_lines = lines[start_line-1:end_line]
        content = ''.join(content_lines)
        
        # 识别语言类型
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.md': 'markdown',
            '.json': 'json',
            '.toml': 'toml',
            '.yml': 'yaml',
            '.yaml': 'yaml'
        }
        
        suffix = full_file_path.suffix
        language = language_map.get(suffix, 'text')
        
        return {
            "content": content,
            "language": language,
            "total_lines": total_lines,
            "displayed_lines": f"{start_line}-{end_line}",
            "file_size": full_file_path.stat().st_size,
            "file_path": file_path
        }
        
    except Exception as e:
        return {"error": f"读取文件失败: {str(e)}"}


def analyze_mcp_structure(repo_path: str) -> Dict[str, Any]:
    """
    深度分析仓库中的 MCP 相关结构
    
    该函数会识别 MCP 服务器的关键组件，包括工具函数、配置、依赖等。
    
    Args:
        repo_path: 仓库路径（相对于 servers/ 目录）
        
    Returns:
        Dict: MCP 结构分析结果
    """
    result = {
        "has_mcp": False,
        "main_server_file": None,
        "mcp_tools": [],
        "dependencies": [],
        "uses_fastmcp": False,
        "port_config": None,
        "missing_components": []
    }
    
    repo_dir = Path(f"servers/{repo_path}")
    
    # 查找主服务器文件
    for py_file in repo_dir.glob("*.py"):
        try:
            content = py_file.read_text(encoding='utf-8')
            
            # 检查是否包含 MCP 相关代码
            if 'FastMCP' in content or 'mcp' in content.lower():
                result["has_mcp"] = True
                result["main_server_file"] = py_file.name
                
                # 检查是否使用 FastMCP
                if 'FastMCP' in content:
                    result["uses_fastmcp"] = True
                
                # 提取工具函数
                tool_pattern = r'@mcp\.tool\(\)\s*\n\s*def\s+(\w+)'
                tools = re.findall(tool_pattern, content)
                result["mcp_tools"].extend(tools)
                
                # 查找端口配置
                port_pattern = r'port["\']?\s*[:=]\s*(\d+)'
                port_match = re.search(port_pattern, content)
                if port_match:
                    result["port_config"] = int(port_match.group(1))
                    
        except Exception:
            continue
    
    # 分析依赖
    pyproject_path = repo_dir / "pyproject.toml"
    requirements_path = repo_dir / "requirements.txt"
    
    if pyproject_path.exists():
        try:
            content = pyproject_path.read_text()
            deps_pattern = r'dependencies\s*=\s*\[(.*?)\]'
            deps_match = re.search(deps_pattern, content, re.DOTALL)
            if deps_match:
                deps_text = deps_match.group(1)
                deps = re.findall(r'"([^"]+)"', deps_text)
                result["dependencies"] = deps
        except Exception:
            pass
    elif requirements_path.exists():
        try:
            deps = [line.strip() for line in requirements_path.read_text().splitlines() 
                   if line.strip() and not line.startswith('#')]
            result["dependencies"] = deps
        except Exception:
            pass
    
    # 检查缺失的标准组件
    if not (repo_dir / "metadata.json").exists():
        result["missing_components"].append("metadata.json")
    if not result["main_server_file"] == "server.py":
        result["missing_components"].append("标准命名 server.py")
    if not any('fastmcp' in dep.lower() for dep in result["dependencies"]):
        result["missing_components"].append("fastmcp 依赖")
    
    return result


def modify_file_content(repo_path: str, file_path: str, old_content: str, 
                       new_content: str, create_if_not_exists: bool = False) -> Dict[str, Any]:
    """
    修改仓库中的文件内容
    
    Args:
        repo_path: 仓库路径
        file_path: 文件路径
        old_content: 要替换的内容
        new_content: 新内容
        create_if_not_exists: 如果文件不存在是否创建
        
    Returns:
        Dict: 操作结果
    """
    full_file_path = Path(f"servers/{repo_path}/{file_path}")
    
    if not full_file_path.exists():
        if create_if_not_exists and old_content == "":
            # 创建新文件
            full_file_path.parent.mkdir(parents=True, exist_ok=True)
            full_file_path.write_text(new_content, encoding='utf-8')
            return {"success": True, "message": f"创建文件: {file_path}"}
        else:
            return {"error": f"文件不存在: {file_path}"}
    
    try:
        content = full_file_path.read_text(encoding='utf-8')
        
        if old_content not in content:
            return {"error": "未找到要替换的内容"}
        
        modified_content = content.replace(old_content, new_content)
        full_file_path.write_text(modified_content, encoding='utf-8')
        
        return {
            "success": True,
            "message": f"已修改文件: {file_path}",
            "changes": content.count(old_content)
        }
    except Exception as e:
        return {"error": f"修改失败: {str(e)}"}


def create_standard_files(repo_path: str, author: str, description: str, 
                         category: str, tools: List[str]) -> Dict[str, Any]:
    """
    创建标准的 MCP 配置文件
    
    创建 metadata.json 和其他必要的标准文件。
    
    Args:
        repo_path: 仓库路径
        author: 作者 GitHub 用户名
        description: 工具描述
        category: 工具类别
        tools: 工具函数列表
        
    Returns:
        Dict: 创建结果
    """
    repo_dir = Path(f"servers/{repo_path}")
    results = {"created": [], "errors": []}
    
    # 创建 metadata.json
    metadata = {
        "name": repo_path,
        "description": description,
        "author": f"@{author}" if not author.startswith("@") else author,
        "category": category,
        "transport": ["sse", "stdio"],
        "tools": tools
    }
    
    try:
        metadata_path = repo_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), 
                                encoding='utf-8')
        results["created"].append("metadata.json")
    except Exception as e:
        results["errors"].append(f"metadata.json: {str(e)}")
    
    return results


def convert_repository_format(repo_path: str) -> Dict[str, Any]:
    """
    执行仓库格式转换的主要逻辑
    
    包括重命名文件、添加参数解析、清理 Git 历史等。
    
    Args:
        repo_path: 仓库路径
        
    Returns:
        Dict: 转换结果
    """
    repo_dir = Path(f"servers/{repo_path}")
    results = {"actions": [], "errors": []}
    
    # 1. 查找并重命名主服务器文件
    main_file = None
    for py_file in repo_dir.glob("*.py"):
        content = py_file.read_text(encoding='utf-8')
        if 'FastMCP' in content or 'mcp' in content.lower():
            main_file = py_file
            break
    
    if main_file and main_file.name != "server.py":
        try:
            new_path = repo_dir / "server.py"
            main_file.rename(new_path)
            results["actions"].append(f"重命名 {main_file.name} → server.py")
        except Exception as e:
            results["errors"].append(f"重命名失败: {str(e)}")
    
    # 2. 清理 Git 历史
    git_dir = repo_dir / ".git"
    if git_dir.exists():
        try:
            subprocess.run(["rm", "-rf", str(git_dir)], check=True)
            results["actions"].append("清理 Git 历史")
        except Exception as e:
            results["errors"].append(f"清理 Git 失败: {str(e)}")
    
    # 3. 添加到主仓库
    try:
        subprocess.run(["git", "add", str(repo_dir)], check=True)
        results["actions"].append("添加到 Git 暂存区")
    except Exception as e:
        results["errors"].append(f"Git 添加失败: {str(e)}")
    
    return results


def search_in_repository(repo_path: str, search_term: str, 
                        file_type: str = "*") -> List[Dict[str, Any]]:
    """
    在仓库中搜索特定内容
    
    Args:
        repo_path: 仓库路径
        search_term: 搜索词
        file_type: 文件类型过滤，如 "*.py", "*"
        
    Returns:
        List[Dict]: 搜索结果列表
    """
    repo_dir = Path(f"servers/{repo_path}")
    results = []
    
    pattern = file_type if file_type != "*" else "**/*"
    
    for file_path in repo_dir.glob(pattern):
        if file_path.is_file() and not any(part.startswith('.') for part in file_path.parts):
            try:
                content = file_path.read_text(encoding='utf-8')
                lines = content.splitlines()
                
                for line_num, line in enumerate(lines, 1):
                    if search_term.lower() in line.lower():
                        results.append({
                            "file": str(file_path.relative_to(repo_dir)),
                            "line_number": line_num,
                            "line": line.strip()[:100],  # 限制长度
                            "match_count": line.lower().count(search_term.lower())
                        })
            except Exception:
                continue
    
    return results[:50]  # 限制结果数量


def list_directory_tree(repo_path: str, show_hidden: bool = False) -> str:
    """
    以树形结构显示目录内容
    
    Args:
        repo_path: 仓库路径
        show_hidden: 是否显示隐藏文件
        
    Returns:
        str: 树形结构字符串
    """
    repo_dir = Path(f"servers/{repo_path}")
    
    if not repo_dir.exists():
        return f"目录不存在: {repo_path}"
    
    def build_tree_string(path: Path, prefix: str = "", is_last: bool = True) -> List[str]:
        lines = []
        
        # 当前项
        connector = "└── " if is_last else "├── "
        lines.append(prefix + connector + path.name)
        
        # 子项的前缀
        if is_last:
            extension = "    "
        else:
            extension = "│   "
        
        # 获取子项
        if path.is_dir():
            items = list(path.iterdir())
            if not show_hidden:
                items = [item for item in items if not item.name.startswith('.')]
            
            items.sort(key=lambda x: (x.is_file(), x.name.lower()))
            
            for i, item in enumerate(items):
                is_last_item = i == len(items) - 1
                lines.extend(build_tree_string(item, prefix + extension, is_last_item))
        
        return lines
    
    tree_lines = [repo_path + "/"]
    
    items = list(repo_dir.iterdir())
    if not show_hidden:
        items = [item for item in items if not item.name.startswith('.')]
    
    items.sort(key=lambda x: (x.is_file(), x.name.lower()))
    
    for i, item in enumerate(items):
        is_last = i == len(items) - 1
        tree_lines.extend(build_tree_string(item, "", is_last))
    
    return "\n".join(tree_lines)