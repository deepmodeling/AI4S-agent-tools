# 🔄 MCP Agent 仓库转换功能

## 概述

MCP Agent 现在支持自动将 GitHub 上的 MCP 工具仓库转换为 AI4S-agent-tools 的标准格式。这个功能可以帮助你快速集成外部的 MCP 工具。

## 新增功能

### 1. **自动仓库转换** (`convert_repo_to_mcp`)

一键将 GitHub 仓库转换为标准 MCP 格式：

```python
from mcp_agent.repo_tools import convert_repo_to_mcp

# 基本用法
result = convert_repo_to_mcp("https://github.com/user/pymol-mcp")

# 自定义参数
result = convert_repo_to_mcp(
    repo_url="https://github.com/user/some-mcp-tool",
    tool_name="my_custom_name",  # 自定义工具名
    author="vrtejus",            # 覆盖作者
    category="chemistry"         # 指定类别
)

print("\n".join(result["messages"]))
```

### 2. **仓库分析** (`analyze_repository`)

分析已克隆仓库的结构：

```python
from mcp_agent.repo_tools import analyze_repository

analysis = analyze_repository("pymol-mcp")
print(f"主文件: {analysis['main_file']}")
print(f"工具函数: {analysis['tool_functions']}")
print(f"依赖: {analysis['dependencies']}")
```

### 3. **仓库克隆** (`clone_repository`)

单独克隆仓库：

```python
from mcp_agent.repo_tools import clone_repository

result = clone_repository(
    "https://github.com/user/tool",
    "target_dir"
)
```

## 转换流程详解

1. **克隆仓库**
   - 下载 GitHub 仓库到 `servers/` 目录

2. **分析结构**
   - 查找包含 FastMCP 的主文件
   - 提取 @mcp.tool() 装饰的函数
   - 识别项目依赖

3. **格式转换**
   - 重命名主文件为 `server.py`
   - 添加标准参数解析（--port, --host, --log-level）
   - 保持原有功能不变

4. **生成元数据**
   - 自动从 URL 提取作者
   - 基于内容推断类别
   - 创建 `metadata.json`

5. **清理 Git**
   - 删除 `.git` 目录
   - 移除 submodule 配置

## 使用示例

### 示例 1: 转换 PyMOL MCP

```python
# 转换 pymol-mcp 仓库
result = convert_repo_to_mcp(
    repo_url="https://github.com/edocollins/pymol-mcp",
    category="chemistry"  # 明确指定为化学类工具
)

# 输出结果：
# ✓ 仓库已克隆到: servers/pymol-mcp
# ✓ 重命名 pymol_mcp_server.py → server.py
# ✓ 添加了参数解析功能
# ✓ 清理了 Git 历史
# ✓ metadata.json 已创建
# ✓ 已添加到 git 暂存区
```

### 示例 2: 批量转换多个仓库

```python
repos = [
    ("https://github.com/user/chem-tool", "chemistry"),
    ("https://github.com/user/bio-analyzer", "biology"),
    ("https://github.com/user/data-processor", "data")
]

for repo_url, category in repos:
    print(f"\n转换 {repo_url}...")
    result = convert_repo_to_mcp(repo_url, category=category)
    
    if result["status"] == "success":
        print("✅ 转换成功")
    else:
        print("❌ 转换失败")
```

### 示例 3: 使用 Agent 对话式转换

```python
from mcp_agent.agent import root_agent

# 通过对话让 Agent 帮你转换
response = root_agent.run(
    "我想转换 https://github.com/user/tool 这个仓库到 MCP 格式"
)
```

## 自动类别识别

工具会根据以下关键词自动推断类别：

- **chemistry**: molecule, mol, chem, reaction, pymol, rdkit
- **biology**: protein, dna, sequence, gene, bio
- **physics**: quantum, physics, energy, particle
- **materials**: material, crystal, structure, lattice
- **simulation**: simulate, dynamics, md
- **data**: data, analysis, visualiz
- **machine-learning**: ml, ai, model, train, neural
- **research**: paper, arxiv, literature

## 注意事项

1. **保持原有功能**：转换过程不会修改核心业务逻辑
2. **手动检查**：转换后建议手动检查 server.py 确保功能正常
3. **依赖管理**：可能需要手动调整 pyproject.toml 中的依赖
4. **测试运行**：转换后使用 `uv run python server.py` 测试

## 错误处理

如果转换失败，检查：
- 仓库 URL 是否正确
- 仓库是否包含 MCP 相关代码
- 网络连接是否正常

## 贡献新功能

欢迎贡献更多转换规则！请查看 `repo_tools.py` 了解实现细节。