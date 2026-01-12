# GitHub Project Agent 使用指南

## 概述

GitHub Project Agent 是一个独立的智能助手，专门用于将 GitHub 上的 MCP 工具仓库转换为 AI4S-agent-tools 的标准格式。与原有的 MCP Agent（用于创建新工具）不同，这个 Agent 专注于仓库分析和转换。

## 主要功能

### 1. 深度仓库分析
- 探索目录结构
- 阅读和理解代码文件
- 识别 MCP 组件
- 分析项目依赖

### 2. 智能转换
- 自动识别主服务器文件
- 保持原有功能不变
- 生成标准配置文件
- 清理版本控制历史

### 3. 文件操作
- 读取任意文件内容
- 修改代码文件
- 创建新文件
- 搜索特定内容

## 使用方法

### 1. 基本使用

```python
from mcp_agent.github_agent import create_github_agent

# 创建 Agent 实例
agent = create_github_agent()

# 转换仓库
response = agent.run("请帮我转换 pymol-mcp 这个仓库")
```

### 2. 完整转换流程

```python
# 假设已经克隆了仓库到 servers/pymol-mcp

# 1. 先让 Agent 探索和理解仓库
response = agent.run("""
我刚克隆了 pymol-mcp 仓库到 servers/pymol-mcp。
请先探索这个仓库，理解它的功能和结构。
""")

# 2. Agent 会自动执行：
# - explore_repository() - 获取概览
# - read_file_content() - 读取 README 和主文件
# - analyze_mcp_structure() - 分析 MCP 组件

# 3. 基于分析结果进行转换
response = agent.run("基于你的分析，请将这个仓库转换为标准格式")
```

### 3. 自定义转换

```python
# 可以指定特定参数
response = agent.run("""
请转换 some-tool 仓库，使用以下信息：
- 作者: @myusername
- 类别: chemistry
- 保持原有的端口配置
""")
```

## 工具函数详解

### explore_repository(repo_path)
探索仓库整体结构，返回文件统计、目录列表等信息。

### read_file_content(repo_path, file_path, start_line, end_line)
读取指定文件的内容，支持读取部分行。

### analyze_mcp_structure(repo_path)
深度分析 MCP 相关结构，识别工具函数、依赖等。

### modify_file_content(repo_path, file_path, old_content, new_content)
修改文件内容，支持创建新文件。

### create_standard_files(repo_path, author, description, category, tools)
创建 metadata.json 等标准配置文件。

### convert_repository_format(repo_path)
执行主要的格式转换操作。

### search_in_repository(repo_path, search_term, file_type)
在仓库中搜索特定内容。

### list_directory_tree(repo_path, show_hidden)
以树形结构显示目录内容。

## 典型工作流程

1. **克隆仓库**（手动或使用其他工具）
   ```bash
   git clone https://github.com/user/tool servers/tool-name
   ```

2. **使用 Agent 分析**
   Agent 会自动：
   - 探索目录结构
   - 读取 README 理解功能
   - 分析代码找出 MCP 工具
   - 检查现有配置

3. **执行转换**
   Agent 会：
   - 重命名主文件为 server.py
   - 添加标准参数解析
   - 创建 metadata.json
   - 清理 .git 目录

4. **验证结果**
   Agent 会再次分析确保转换成功

## 与 MCP Agent 的区别

| 特性 | MCP Agent | GitHub Project Agent |
|-----|-----------|---------------------|
| 主要用途 | 创建新的 MCP 工具 | 转换现有 GitHub 仓库 |
| 工作方式 | 通过对话收集需求 | 分析和修改现有代码 |
| 文件操作 | 仅创建新文件 | 读取、修改、搜索文件 |
| 代码理解 | 不需要 | 深度分析代码结构 |

## 最佳实践

1. **先分析再转换**：让 Agent 充分理解项目后再执行转换
2. **保留原始功能**：Agent 会保持核心功能不变
3. **手动验证**：转换后建议手动测试功能是否正常
4. **备份重要数据**：虽然 Agent 很谨慎，但建议先备份

## 错误处理

如果转换失败，Agent 会：
1. 清楚说明问题所在
2. 提供可能的解决方案
3. 保持已完成的部分不变

## 扩展功能

未来计划添加：
- 自动测试转换后的工具
- 批量转换多个仓库
- 生成转换报告
- 集成 GitHub API 直接克隆