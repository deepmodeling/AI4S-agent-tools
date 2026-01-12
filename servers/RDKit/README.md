# RDKit Toolkit

## 描述

RDKit Toolkit 是一个为 AI4S-agent-tools 项目提供基础化学信息学功能的核心工具。它基于强大的 RDKit 库，为 AI Agent 提供分子处理、描述符计算和可视化等基础能力。

## 功能特性

### 🔥 核心功能
- **分子验证**: 检查 SMILES 字符串的化学有效性
- **基础描述符计算**: 计算分子量、LogP、TPSA、氢键供体/受体数等
- **分子指纹生成**: 生成 Morgan 指纹用于相似性分析
- **子结构搜索**: 基于 SMARTS 模式的子结构匹配
- **2D 可视化**: 生成 SVG 格式的分子结构图

## 安装

```bash
cd servers/RDKit
uv sync
```

## 使用

### 启动服务器

```bash
# 运行服务器（SSE模式）
python server.py --port 50001

# 运行服务器（stdio模式）
MCP_TRANSPORT=stdio python server.py

# 启用调试日志
python server.py --log-level DEBUG
```

### 使用 uv 运行

```bash
uv run python server.py
```

## 可用工具

### 1. validate_smiles
检查 SMILES 字符串是否化学有效。

**参数:**
- `smiles` (str): 要验证的 SMILES 字符串

**返回:** bool - 如果有效返回 True

### 2. get_basic_properties
计算分子的基础物理化学性质。

**参数:**
- `smiles` (str): 分子的 SMILES 字符串

**返回:** dict - 包含以下性质：
- `molecular_weight`: 分子量
- `logp`: 脂水分配系数
- `tpsa`: 拓扑极性表面积
- `hydrogen_bond_donors`: 氢键供体数
- `hydrogen_bond_acceptors`: 氢键受体数

### 3. generate_fingerprint
为分子生成 Morgan 指纹。

**参数:**
- `smiles` (str): 分子的 SMILES 字符串
- `radius` (int, 可选): 指纹半径，默认为 2
- `n_bits` (int, 可选): 指纹位数，默认为 2048

**返回:** list - 指纹的活跃位列表

### 4. substructure_search
使用 SMARTS 模式进行子结构搜索。

**参数:**
- `smiles` (str): 要搜索的分子的 SMILES 字符串
- `smarts_pattern` (str): SMARTS 搜索模式

**返回:** bool - 如果找到子结构返回 True

### 5. draw_molecule_svg
生成分子的 2D SVG 图像。

**参数:**
- `smiles` (str): 分子的 SMILES 字符串

**返回:** str - SVG 格式的分子结构图

## 使用示例

```python
# 验证 SMILES
result = validate_smiles("CCO")  # 乙醇
# 返回: True

# 计算基础性质
properties = get_basic_properties("CCO")
# 返回: {
#   "molecular_weight": 46.07,
#   "logp": -0.31,
#   "tpsa": 20.23,
#   "hydrogen_bond_donors": 1,
#   "hydrogen_bond_acceptors": 1
# }

# 生成指纹
fingerprint = generate_fingerprint("CCO")
# 返回: [活跃位的索引列表]

# 子结构搜索
has_alcohol = substructure_search("CCO", "[OH]")
# 返回: True (含有羟基)

# 生成分子图像
svg_image = draw_molecule_svg("CCO")
# 返回: SVG 格式的分子结构图
```

## 依赖

- **rdkit-pypi**: 核心化学信息学库
- **fastmcp**: MCP 服务器框架
- **numpy**: 数值计算支持

## 与其他工具的协同

- **与 PubChem 协同**: 处理和验证从 PubChem 获取的分子结构
- **与 ASKCOS 协同**: 对分子进行预处理和标准化，用于逆合成路径规划
- **与 ORCA/ABACUS 协同**: 生成分子的初始结构，作为量子化学计算的输入

## 开发指南

本工具遵循 AI4S-agent-tools 的标准开发模式：

1. **模块化设计**: 每个功能都是独立的工具函数
2. **类型注解**: 所有函数都有完整的类型注解
3. **错误处理**: 适当的异常处理和错误信息
4. **文档完整**: 详细的函数文档和使用示例

## 许可证

本项目遵循 AI4S-agent-tools 项目的许可证。