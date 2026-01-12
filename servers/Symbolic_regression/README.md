# 符号回归 MCP 服务器

基于 PySR（Python 符号回归）的模型上下文协议（MCP）服务器，使用遗传编程技术从数据中发现数学表达式。

## 概述

符号回归服务器使用进化算法自动发现最能描述数据关系的数学公式。与传统回归方法不同，符号回归不仅能找到参数，还能发现数学表达式的结构。

## 功能特性

- **自动公式发现**：无需假设特定形式即可找到数学表达式
- **可解释模型**：生成人类可读的数学公式
- **灵活的运算符**：支持各种数学运算（算术、三角函数、指数等）
- **多变量支持**：处理多个输入变量
- **复杂度控制**：平衡准确性与表达式简洁性

## 安装

```bash
cd servers/Symbolic_regression
uv sync
```

## 使用方法

### 启动服务器

```bash
python server.py --port 50001
```

### 工具函数

#### `symbolic_regression`

从 CSV 数据中发现数学表达式。

**CSV 文件格式要求：**
- 文件格式：标准 CSV 格式，逗号分隔
- 列结构：前 n 列为输入变量（特征），最后一列为目标变量（需要预测的值）
- 数据要求：
  - 所有数据必须为数值类型
  - 不能包含缺失值（NaN）
  - 不能包含文本或类别数据

**CSV 示例：**
```csv
x1,x2,x3,y
1.2,3.4,5.6,10.5
2.3,4.5,6.7,15.8
3.4,5.6,7.8,21.2
...
```

**参数：**
- `csv_path` (str, 必需): CSV 文件路径
  
- `unary_operators` (list, 必需): 表达式中使用的一元运算符列表
  - 可用运算符：
    - `"neg"`：取负 (-x)
    - `"square"`：平方 (x²)
    - `"cube"`：立方 (x³)
    - `"cbrt"`：立方根 (∛x)
    - `"inv"`：倒数 (1/x)
    - `"exp"`：指数 (e^x)
    - `"log"`：自然对数 (ln(x))
    - `"sqrt"`：平方根 (√x)
    - `"abs"`：绝对值 (|x|)
    - `"sin"`：正弦
    - `"cos"`：余弦
    - `"tan"`：正切

**返回值：**
包含以下内容的字典：
- `best_result`：找到的最佳表达式及其复杂度和均方误差
- `candidates`：按性能排序的候选表达式列表
- `metadata`：回归运行的相关信息

## 示例

### 示例 1：发现物理定律

如果你有单摆实验数据（摆长和周期测量值）：

```python
# CSV 格式: length,period
# 数据关系：T = 2π√(L/g)

result = symbolic_regression(
    csv_path="pendulum_data.csv",
    unary_operators=["sqrt"]
)
```

### 示例 2：化学动力学

对于反应速率与浓度、温度的数据：

```python
# CSV 格式: concentration,temperature,rate
# 可能发现阿伦尼乌斯型方程

result = symbolic_regression(
    csv_path="reaction_data.csv",
    unary_operators=["exp", "log", "inv"]
)
```

### 示例 3：工程关系

对于材料的应力-应变数据：

```python
# CSV 格式: strain,stress
# 可能发现多项式或幂律关系

result = symbolic_regression(
    csv_path="material_data.csv",
    unary_operators=["square", "cube", "sqrt"]
)
```

## 输出结果

结果保存到：
- `output/best.txt`：包含复杂度、均方误差和表达式的制表符分隔文件
- `results.json`：包含所有候选项和元数据的详细 JSON 输出

## 技术细节

- 使用 PySR（Python 符号回归）后端
- 二元运算符：+、-、*、/
- 最大表达式复杂度：20
- 种群大小：50
- 迭代次数：1000
- 启用多核处理（10 个进程）
- 应用表达式简化和格式化

## 限制

- 仅支持数值数据
- 不能处理缺失值
- 表达式限制在指定的复杂度范围内
- 性能取决于数据质量和噪声水平

## 使用场景

1. **科学发现**：从实验数据中找到定律和关系
2. **工程设计**：推导系统行为的经验公式
3. **数据分析**：创建可解释的预测模型