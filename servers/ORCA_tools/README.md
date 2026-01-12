# ORCA 工具文档

本文档介绍了 `servers/ORCA_tools/server.py` 中提供的 ORCA 计算和结果检索工具。

## 配置Python环境

```bash
conda create -n orca_agent_tools python=3.10
conda activate orca_agent_tools
cd your_path_to_AI4S-agent-tools/servers/ORCA_tools
pip install -e .
python server.py
```

## 运行前环境设置

在运行 `server.py` 之前，请确保已设置以下环境变量。这些变量对于 ORCA 和 OpenMPI 的正确运行至关重要。


```bash
export PATH="/opt/openmpi411/bin:/opt/orca504/orca_5_0_4_linux_x86-64_shared_openmpi411:$PATH"
export LD_LIBRARY_PATH="/opt/openmpi411/lib:/opt/orca504/orca_5_0_4_linux_x86-64_shared_openmpi411:$LD_LIBRARY_PATH"
export OMPI_ALLOW_RUN_AS_ROOT="1"
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1"
export ORCA="/opt/orca504/orca_5_0_4_linux_x86-64_shared_openmpi411/orca"
```

请根据您的实际安装路径调整 `/opt/openmpi411` 和 `/opt/orca504/orca_5_0_4_linux_x86-64_shared_openmpi411`。

ORCA的安装可以参考博文[量子化学程序ORCA的安装方法](http://sobereva.com/451).


## `run_orca_calculation` 函数

### 功能描述
`run_orca_calculation` 函数用于执行 ORCA 量子化学计算。它接收 ORCA 输入字符串和分子 XYZ 坐标，然后在临时工作目录中运行 ORCA 程序，并返回计算结果文件的路径。

### 参数
- `input_str` (str): ORCA 输入字符串，包含必要的计算参数和分子结构。
- `xyz_coordinates` (str): 分子结构的 XYZ 坐标字符串，每行格式为 "元素符号 x y z"。

### 返回值
`OrcaResult` (TypedDict): 一个字典，包含以下字段：
- `output_file` (Path): ORCA 输出文件的路径。
- `gbw_file` (Path): ORCA GBW 文件的路径。
- `mol_file` (Path): 分子结构文件的路径。
- `message` (str): 成功或错误消息。

### 最佳实践

#### Example 1: 运行一个简单的甲醇单点计算

输入：
```python
input_str = """
! BLYP def2-SVP
* xyz 0 1
C                  0.00000000    0.00000000   -0.56221066
H                  0.00000000   -0.92444767   -1.10110537
H                 -0.00000000    0.92444767   -1.10110537
O                  0.00000000    0.00000000    0.69618930
*
"""
xyz_coordinates = """
C 0.00000000 0.00000000 -0.56221066
H 0.00000000 -0.92444767 -1.10110537
H -0.00000000 0.92444767 -1.10110537
O 0.00000000 0.00000000 0.69618930
"""
```

输出：
```
{
    "output_file": "orca_calc_1678886400/calc.out",
    "gbw_file": "orca_calc_1678886400/calc.gbw",
    "mol_file": "orca_calc_1678886400/mol.xyz",
    "message": "ORCA calculation completed successfully."
}
```

#### Example 2: 运行一个包含更多计算参数的乙烯优化计算

输入：
```python
input_str = """
! B3LYP def2-TZVP Opt Freq
%maxcore 2000
%pal nprocs 4 end
* xyz 0 1
C  -0.66950   0.00000   0.00000
C   0.66950   0.00000   0.00000
H  -1.22750   0.92000   0.00000
H  -1.22750  -0.92000   0.00000
H   1.22750   0.92000   0.00000
H   1.22750  -0.92000   0.00000
*
"""
xyz_coordinates = """
C -0.66950 0.00000 0.00000
C 0.66950 0.00000 0.00000
H -1.22750 0.92000 0.00000
H -1.22750 -0.92000 0.00000
H 1.22750 0.92000 0.00000
H 1.22750 -0.92000 0.00000
"""
```

输出：
```
{
    "output_file": "orca_calc_1678886401/calc.out",
    "gbw_file": "orca_calc_1678886401/calc.gbw",
    "mol_file": "orca_calc_1678886401/mol.xyz",
    "message": "ORCA calculation completed successfully."
}
```

## `retrieve_content_from_orca_output` 函数

### 功能描述
`retrieve_content_from_orca_output` 函数用于从 ORCA 输出文件中检索与给定查询相关的内容。它使用向量存储和相似性搜索来查找最相关的文本片段。

### 参数
- `query` (str): 要搜索的查询字符串。
- `orca_outpath` (Path): ORCA 输出文件的路径。

### 返回值
`dict`: 一个字典，包含以下字段：
- `status` (str): 操作状态，"success" 或 "error"。
- `message` (str): 成功或错误消息。
- `retrieved_content` (str, 可选): 如果成功，则为检索到的内容。

### 最佳实践

#### Example 1: 从 ORCA 输出文件中检索总能量

输入：
```python
orca_output_file = Path("path/to/your/orca_calc_1678886400/calc.out") # 替换为实际的ORCA输出文件路径
query = "total energy"
```

输出：
```
{
    "status": "success",
    "message": "Content retrieved successfully.",
    "retrieved_content": "Score: 0.85\nContent:\n... The final electronic energy is -114.34567890 Ha ..."
}
```

#### Example 2: 从 ORCA 输出文件中检索 HOMO-LUMO 间隙

输入：
```python
orca_output_file = Path("path/to/your/orca_calc_1678886401/calc.out") # 替换为实际的ORCA输出文件路径
query = "HOMO-LUMO gap"
```

输出：
```
{
    "status": "success",
    "message": "Content retrieved successfully.",
    "retrieved_content": "Score: 0.92\nContent:\n... HOMO-LUMO Gap: 5.23 eV ..."
}
```

#### Example 3: 检索特定原子上的电荷信息

输入：
```python
orca_output_file = Path("path/to/your/orca_calc_1678886400/calc.out") # 替换为实际的ORCA输出文件路径
query = "Mulliken charges on carbon atom"
```

输出：
```
{
    "status": "success",
    "message": "Content retrieved successfully.",
    "retrieved_content": "Score: 0.78\nContent:\n... Mulliken atomic charges:\n1 C: -0.25 ..."
}
