# NMRglue Toolkit

NMR数据处理和分析工具，基于nmrglue库实现，提供完整的NMR光谱数据读取、处理、分析和可视化功能。

## 功能特性

- **多格式支持**: 支持Bruker、Varian、NMRPipe、Sparky等主流NMR数据格式
- **数据处理**: 相位校正、基线校正、窗函数应用等核心处理功能
- **谱图分析**: 峰识别、谱图信息提取等分析工具
- **可视化**: 生成高质量的NMR谱图
- **内存管理**: 智能数据缓存机制，支持大型数据集处理

## 安装

```bash
# 进入工具目录
cd servers/nmrglue

# 安装依赖
uv sync
```

## 使用方法

### 启动服务器

```bash
# 默认启动 (端口 50001)
python server.py

# 自定义端口和主机
python server.py --host localhost --port 50002

# 启用调试日志
python server.py --log-level DEBUG

# stdio模式 (用于MCP客户端)
MCP_TRANSPORT=stdio python server.py
```

## 可用工具函数

### 1. read_nmr_data
读取NMR数据文件

**参数:**
- `file_path` (str): NMR数据文件路径
- `data_format` (str): 数据格式，可选值："auto", "bruker", "varian", "pipe", "sparky"

**返回:**
包含数据ID、格式信息、数据形状等的字典

### 2. write_nmr_data
写入NMR数据到文件

**参数:**
- `data_id` (str): 数据ID
- `output_path` (str): 输出文件路径
- `output_format` (str): 输出格式，可选值："pipe", "bruker", "varian", "sparky"

### 3. get_spectrum_info
获取NMR谱图的详细信息

**参数:**
- `data_id` (str): 数据ID

**返回:**
包含维度、数据类型、统计信息等的详细信息

### 4. phase_correction
对NMR数据进行相位校正

**参数:**
- `data_id` (str): 数据ID
- `p0` (float): 零阶相位校正 (度)
- `p1` (float): 一阶相位校正 (度)
- `pivot` (int, 可选): 一阶相位校正的支点位置

### 5. baseline_correction
对NMR数据进行基线校正

**参数:**
- `data_id` (str): 数据ID
- `method` (str): 基线校正方法，目前支持"polynomial"
- `order` (int): 多项式阶数

### 6. apply_apodization
对NMR数据应用窗函数

**参数:**
- `data_id` (str): 数据ID
- `function` (str): 窗函数类型，可选值："exponential", "gaussian", "sine_bell"
- `lb` (float): 线宽参数 (Hz)

### 7. peak_picking
在NMR谱图中进行峰识别

**参数:**
- `data_id` (str): 数据ID
- `threshold` (float): 峰识别阈值 (相对于最大值的比例)
- `min_distance` (int): 峰之间的最小距离

**返回:**
包含峰位置和强度信息的列表

### 8. plot_spectrum
绘制NMR谱图

**参数:**
- `data_id` (str): 数据ID
- `x_range` (list, 可选): X轴范围 [min, max]
- `y_range` (list, 可选): Y轴范围 [min, max]

**返回:**
包含base64编码图像的字典

## 使用示例

### 基本工作流程

1. **读取NMR数据**
```python
# 自动检测格式
result = read_nmr_data("/path/to/nmr/data", "auto")
data_id = result["data_id"]
```

2. **查看数据信息**
```python
info = get_spectrum_info(data_id)
print(f"数据维度: {info['dimensions']}")
print(f"数据形状: {info['shape']}")
```

3. **数据处理**
```python
# 应用指数窗函数
apply_apodization(data_id, "exponential", lb=2.0)

# 相位校正
phase_correction(data_id, p0=45.0, p1=0.0)

# 基线校正
baseline_correction(data_id, "polynomial", order=3)
```

4. **谱图分析**
```python
# 峰识别
peaks = peak_picking(data_id, threshold=0.1, min_distance=10)
print(f"识别到 {peaks['num_peaks']} 个峰")
```

5. **可视化**
```python
# 生成谱图
plot_result = plot_spectrum(data_id)
image_data = plot_result["image_base64"]
```

6. **保存结果**
```python
# 保存处理后的数据
write_nmr_data(data_id, "/path/to/output.ft", "pipe")
```

## 支持的数据格式

### 输入格式
- **Bruker**: TopSpin格式的目录结构
- **Varian**: VnmrJ格式的数据文件
- **NMRPipe**: .ft, .fid等格式
- **Sparky**: .ucsf格式

### 输出格式
- **NMRPipe**: 标准的.ft格式
- **Bruker**: TopSpin兼容格式
- **Varian**: VnmrJ兼容格式
- **Sparky**: .ucsf格式

## 技术特性

- **内存效率**: 使用数据缓存机制，避免重复加载大型数据集
- **错误处理**: 完善的异常处理和错误信息反馈
- **类型安全**: 完整的类型注解支持
- **可扩展性**: 模块化设计，易于添加新功能

## 依赖库

- `nmrglue`: NMR数据处理核心库
- `numpy`: 数值计算
- `matplotlib`: 图形绘制
- `fastmcp`: MCP服务器框架

## 注意事项

1. **数据格式**: 确保输入数据格式正确，建议使用"auto"模式进行自动检测
2. **内存使用**: 大型数据集可能占用较多内存，建议及时清理不需要的数据
3. **文件路径**: 使用绝对路径以避免路径相关问题
4. **相位校正**: 仅适用于复数数据，实数数据会返回错误

## 故障排除

### 常见问题

1. **ImportError: nmrglue库未安装**
   ```bash
   pip install nmrglue
   ```

2. **文件读取失败**
   - 检查文件路径是否正确
   - 确认数据格式是否支持
   - 尝试使用"auto"格式进行自动检测

3. **内存不足**
   - 处理大型数据集时，考虑分批处理
   - 及时清理不需要的数据缓存

4. **图像生成失败**
   - 确保matplotlib后端配置正确
   - 检查数据是否包含有效的数值

## 开发信息

- **版本**: 0.1.0
- **作者**: @jiaodu1307
- **分类**: 化学
- **传输方式**: SSE, stdio

## 相关资源

- [nmrglue官方文档](https://nmrglue.readthedocs.io/)
- [AI4S-agent-tools项目](https://github.com/deepmodeling/AI4S-agent-tools)
- [MCP协议文档](https://modelcontextprotocol.io/)