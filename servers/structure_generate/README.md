# Structure Generate Server

## 主要功能

### 🔧 ASE 结构构建工具

使用 Atomic Simulation Environment (ASE) 库构建各种晶体结构：

- **体相晶体结构** - 构建标准晶体结构（fcc、bcc、hcp、diamond 等）
- **超胞生成** - 从现有结构生成超胞
- **分子结构** - 构建分子结构并放置在晶胞中
- **分子晶胞添加** - 为分子结构添加适当的晶胞，特别适用于ABACUS分子计算
- **表面切片** - 按指定米勒指数生成表面切片
- **表面吸附** - 在表面上添加吸附分子
- **界面结构** - 构建两个材料的界面结构

### 🧬 CALYPSO 结构预测

基于进化算法和粒子群优化的晶体结构预测：

- **智能结构搜索** - 使用进化算法寻找稳定晶体结构
- **多组分系统** - 支持复杂的多元素化合物结构预测
- **自动优化** - 内置结构筛选和优化机制

### 🤖 CrystalFormer 条件生成

基于机器学习的有条件晶体结构生成：

- **属性导向生成** - 根据目标物理性质生成结构
- **多目标优化** - 同时满足多个性质约束
- **空间群约束** - 在指定空间群下生成结构

## 支持的工具

| 工具名称 | 功能描述 | 主要参数 |
|---------|----------|----------|
| `build_bulk_structure` | 构建体相晶体结构 | 元素、晶体结构类型、晶格参数 |
| `make_supercell_structure` | 生成超胞结构 | 输入结构、超胞矩阵 |
| `build_molecule_structure` | 构建分子结构 | 分子名称、晶胞参数、真空层 |
| `add_cell_for_molecules` | 为分子添加晶胞 | 分子路径、晶胞矢量、真空层 |
| `build_surface_slab` | 构建表面切片 | 材料、米勒指数、层数、真空层 |
| `build_surface_adsorbate` | 构建表面吸附结构 | 表面、吸附分子、位置、高度 |
| `build_surface_interface` | 构建界面结构 | 两个材料、堆叠轴、界面距离 |
| `generate_calypso_structures` | CALYPSO 结构预测 | 元素列表、生成数量 |
| `generate_crystalformer_structures` | 条件结构生成 | 目标性质、空间群、样本数 |
