#!/usr/bin/env python3
"""
NMRglue MCP Server
NMR data processing and analysis tool using nmrglue library.
"""
import argparse
import os
import tempfile
import base64
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import io

try:
    import nmrglue as ng
except ImportError:
    raise ImportError("nmrglue library is required. Install with: pip install nmrglue")

from mcp.server.fastmcp import FastMCP

def parse_args():
    """解析MCP服务器的命令行参数。"""
    parser = argparse.ArgumentParser(description="NMRglue MCP 服务器")
    parser.add_argument('--port', type=int, default=50001, help='服务器端口 (默认: 50001)')
    parser.add_argument('--host', default='0.0.0.0', help='服务器主机 (默认: 0.0.0.0)')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别 (默认: INFO)')
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
mcp = FastMCP("NMRglue Toolkit", host=args.host, port=args.port)

# 全局变量存储加载的NMR数据
_nmr_data_cache = {}

@mcp.tool()
def read_nmr_data(file_path: str, data_format: str = "auto") -> Dict[str, Any]:
    """
    读取NMR数据文件。
    
    Args:
        file_path: NMR数据文件路径
        data_format: 数据格式 ("auto", "bruker", "varian", "pipe", "sparky")
        
    Returns:
        包含数据信息和状态的字典
    """
    try:
        if not os.path.exists(file_path):
            return {"status": "error", "message": f"文件不存在: {file_path}"}
        
        # 根据格式读取数据
        if data_format == "auto":
            # 尝试自动检测格式
            if os.path.isdir(file_path):
                # 可能是Bruker格式
                try:
                    dic, data = ng.bruker.read(file_path)
                    data_format = "bruker"
                except:
                    try:
                        dic, data = ng.varian.read(file_path)
                        data_format = "varian"
                    except:
                        return {"status": "error", "message": "无法自动检测数据格式"}
            else:
                # 可能是单个文件
                try:
                    dic, data = ng.pipe.read(file_path)
                    data_format = "pipe"
                except:
                    try:
                        dic, data = ng.sparky.read(file_path)
                        data_format = "sparky"
                    except:
                        return {"status": "error", "message": "无法自动检测数据格式"}
        elif data_format == "bruker":
            dic, data = ng.bruker.read(file_path)
        elif data_format == "varian":
            dic, data = ng.varian.read(file_path)
        elif data_format == "pipe":
            dic, data = ng.pipe.read(file_path)
        elif data_format == "sparky":
            dic, data = ng.sparky.read(file_path)
        else:
            return {"status": "error", "message": f"不支持的数据格式: {data_format}"}
        
        # 存储到缓存
        data_id = f"nmr_data_{len(_nmr_data_cache)}"
        _nmr_data_cache[data_id] = {"dic": dic, "data": data, "format": data_format}
        
        return {
            "status": "success",
            "data_id": data_id,
            "format": data_format,
            "shape": list(data.shape),
            "dtype": str(data.dtype),
            "size_mb": data.nbytes / (1024 * 1024),
            "message": f"成功读取 {data_format} 格式的NMR数据"
        }
        
    except Exception as e:
        return {"status": "error", "message": f"读取NMR数据失败: {str(e)}"}

@mcp.tool()
def write_nmr_data(data_id: str, output_path: str, output_format: str = "pipe") -> Dict[str, Any]:
    """
    写入NMR数据到文件。
    
    Args:
        data_id: 数据ID
        output_path: 输出文件路径
        output_format: 输出格式 ("pipe", "bruker", "varian", "sparky")
        
    Returns:
        操作状态字典
    """
    try:
        if data_id not in _nmr_data_cache:
            return {"status": "error", "message": f"数据ID不存在: {data_id}"}
        
        cached_data = _nmr_data_cache[data_id]
        dic, data = cached_data["dic"], cached_data["data"]
        
        # 根据格式写入数据
        if output_format == "pipe":
            ng.pipe.write(output_path, dic, data)
        elif output_format == "bruker":
            ng.bruker.write(output_path, dic, data)
        elif output_format == "varian":
            ng.varian.write(output_path, dic, data)
        elif output_format == "sparky":
            ng.sparky.write(output_path, dic, data)
        else:
            return {"status": "error", "message": f"不支持的输出格式: {output_format}"}
        
        return {
            "status": "success",
            "output_path": output_path,
            "format": output_format,
            "message": f"成功写入 {output_format} 格式的NMR数据"
        }
        
    except Exception as e:
        return {"status": "error", "message": f"写入NMR数据失败: {str(e)}"}

@mcp.tool()
def get_spectrum_info(data_id: str) -> Dict[str, Any]:
    """
    获取NMR谱图的详细信息。
    
    Args:
        data_id: 数据ID
        
    Returns:
        包含谱图信息的字典
    """
    try:
        if data_id not in _nmr_data_cache:
            return {"status": "error", "message": f"数据ID不存在: {data_id}"}
        
        cached_data = _nmr_data_cache[data_id]
        dic, data = cached_data["dic"], cached_data["data"]
        data_format = cached_data["format"]
        
        info = {
            "status": "success",
            "format": data_format,
            "dimensions": len(data.shape),
            "shape": list(data.shape),
            "dtype": str(data.dtype),
            "size_mb": data.nbytes / (1024 * 1024),
            "is_complex": np.iscomplexobj(data),
            "min_value": float(np.min(np.real(data))),
            "max_value": float(np.max(np.real(data))),
            "mean_value": float(np.mean(np.real(data)))
        }
        
        # 尝试获取更多元数据信息
        try:
            if data_format == "bruker" and "acqus" in dic:
                acqus = dic["acqus"]
                info["acquisition_params"] = {
                    "sw": acqus.get("SW", "Unknown"),
                    "o1": acqus.get("O1", "Unknown"),
                    "bf1": acqus.get("BF1", "Unknown"),
                    "td": acqus.get("TD", "Unknown")
                }
        except:
            pass
        
        return info
        
    except Exception as e:
        return {"status": "error", "message": f"获取谱图信息失败: {str(e)}"}

@mcp.tool()
def phase_correction(data_id: str, p0: float = 0.0, p1: float = 0.0, pivot: Optional[int] = None) -> Dict[str, Any]:
    """
    对NMR数据进行相位校正。
    
    Args:
        data_id: 数据ID
        p0: 零阶相位校正 (度)
        p1: 一阶相位校正 (度)
        pivot: 一阶相位校正的支点位置
        
    Returns:
        操作状态字典
    """
    try:
        if data_id not in _nmr_data_cache:
            return {"status": "error", "message": f"数据ID不存在: {data_id}"}
        
        cached_data = _nmr_data_cache[data_id]
        dic, data = cached_data["dic"], cached_data["data"]
        
        if not np.iscomplexobj(data):
            return {"status": "error", "message": "相位校正需要复数数据"}
        
        # 执行相位校正
        if pivot is None:
            pivot = data.shape[-1] // 2
        
        corrected_data = ng.proc_base.ps(data, p0=p0, p1=p1, inv=False)
        
        # 更新缓存中的数据
        _nmr_data_cache[data_id]["data"] = corrected_data
        
        return {
            "status": "success",
            "p0": p0,
            "p1": p1,
            "pivot": pivot,
            "message": f"成功应用相位校正: p0={p0}°, p1={p1}°"
        }
        
    except Exception as e:
        return {"status": "error", "message": f"相位校正失败: {str(e)}"}

@mcp.tool()
def baseline_correction(data_id: str, method: str = "polynomial", order: int = 3) -> Dict[str, Any]:
    """
    对NMR数据进行基线校正。
    
    Args:
        data_id: 数据ID
        method: 基线校正方法 ("polynomial", "spline")
        order: 多项式阶数或样条节点数
        
    Returns:
        操作状态字典
    """
    try:
        if data_id not in _nmr_data_cache:
            return {"status": "error", "message": f"数据ID不存在: {data_id}"}
        
        cached_data = _nmr_data_cache[data_id]
        dic, data = cached_data["dic"], cached_data["data"]
        
        # 获取实部数据进行基线校正
        real_data = np.real(data)
        
        if method == "polynomial":
            # 简单的多项式基线校正
            x = np.arange(len(real_data))
            coeffs = np.polyfit(x, real_data, order)
            baseline = np.polyval(coeffs, x)
            corrected_data = real_data - baseline
        else:
            return {"status": "error", "message": f"不支持的基线校正方法: {method}"}
        
        # 如果原数据是复数，保持虚部不变
        if np.iscomplexobj(data):
            corrected_data = corrected_data + 1j * np.imag(data)
        
        # 更新缓存中的数据
        _nmr_data_cache[data_id]["data"] = corrected_data
        
        return {
            "status": "success",
            "method": method,
            "order": order,
            "message": f"成功应用{method}基线校正，阶数={order}"
        }
        
    except Exception as e:
        return {"status": "error", "message": f"基线校正失败: {str(e)}"}

@mcp.tool()
def apply_apodization(data_id: str, function: str = "exponential", lb: float = 1.0) -> Dict[str, Any]:
    """
    对NMR数据应用窗函数。
    
    Args:
        data_id: 数据ID
        function: 窗函数类型 ("exponential", "gaussian", "sine_bell")
        lb: 线宽参数 (Hz)
        
    Returns:
        操作状态字典
    """
    try:
        if data_id not in _nmr_data_cache:
            return {"status": "error", "message": f"数据ID不存在: {data_id}"}
        
        cached_data = _nmr_data_cache[data_id]
        dic, data = cached_data["dic"], cached_data["data"]
        
        if function == "exponential":
            # 指数窗函数
            apodized_data = ng.proc_base.em(data, lb=lb)
        elif function == "gaussian":
            # 高斯窗函数
            apodized_data = ng.proc_base.gm(data, g1=0, g2=lb, g3=0.5)
        elif function == "sine_bell":
            # 正弦钟窗函数
            apodized_data = ng.proc_base.sp(data, off=0.5, end=1.0, pow=1.0)
        else:
            return {"status": "error", "message": f"不支持的窗函数: {function}"}
        
        # 更新缓存中的数据
        _nmr_data_cache[data_id]["data"] = apodized_data
        
        return {
            "status": "success",
            "function": function,
            "lb": lb,
            "message": f"成功应用{function}窗函数，lb={lb} Hz"
        }
        
    except Exception as e:
        return {"status": "error", "message": f"窗函数应用失败: {str(e)}"}

@mcp.tool()
def peak_picking(data_id: str, threshold: float = 0.1, min_distance: int = 10) -> Dict[str, Any]:
    """
    在NMR谱图中进行峰识别。
    
    Args:
        data_id: 数据ID
        threshold: 峰识别阈值 (相对于最大值的比例)
        min_distance: 峰之间的最小距离
        
    Returns:
        包含峰位置和强度的字典
    """
    try:
        if data_id not in _nmr_data_cache:
            return {"status": "error", "message": f"数据ID不存在: {data_id}"}
        
        cached_data = _nmr_data_cache[data_id]
        dic, data = cached_data["dic"], cached_data["data"]
        
        # 获取实部数据
        real_data = np.real(data)
        
        # 简单的峰识别算法
        max_val = np.max(real_data)
        threshold_val = threshold * max_val
        
        peaks = []
        for i in range(min_distance, len(real_data) - min_distance):
            if (real_data[i] > threshold_val and 
                real_data[i] > real_data[i-1] and 
                real_data[i] > real_data[i+1]):
                # 检查最小距离
                too_close = False
                for peak_pos, _ in peaks:
                    if abs(i - peak_pos) < min_distance:
                        too_close = True
                        break
                if not too_close:
                    peaks.append((i, real_data[i]))
        
        # 按强度排序
        peaks.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "status": "success",
            "num_peaks": len(peaks),
            "peaks": [
                {"position": int(pos), "intensity": float(intensity)} 
                for pos, intensity in peaks[:50]  # 限制返回前50个峰
            ],
            "threshold": threshold,
            "min_distance": min_distance,
            "message": f"识别到 {len(peaks)} 个峰"
        }
        
    except Exception as e:
        return {"status": "error", "message": f"峰识别失败: {str(e)}"}

@mcp.tool()
def plot_spectrum(data_id: str, x_range: Optional[List[float]] = None, y_range: Optional[List[float]] = None) -> Dict[str, Any]:
    """
    绘制NMR谱图。
    
    Args:
        data_id: 数据ID
        x_range: X轴范围 [min, max]
        y_range: Y轴范围 [min, max]
        
    Returns:
        包含图像base64编码的字典
    """
    try:
        if data_id not in _nmr_data_cache:
            return {"status": "error", "message": f"数据ID不存在: {data_id}"}
        
        cached_data = _nmr_data_cache[data_id]
        dic, data = cached_data["dic"], cached_data["data"]
        
        # 获取实部数据
        real_data = np.real(data)
        
        # 创建图形
        plt.figure(figsize=(12, 6))
        
        # 创建x轴（假设为ppm或Hz）
        x_axis = np.arange(len(real_data))
        
        plt.plot(x_axis, real_data, 'b-', linewidth=0.8)
        plt.xlabel('Data Points')
        plt.ylabel('Intensity')
        plt.title(f'NMR Spectrum ({cached_data["format"]} format)')
        plt.grid(True, alpha=0.3)
        
        # 设置范围
        if x_range:
            plt.xlim(x_range)
        if y_range:
            plt.ylim(y_range)
        
        # 保存图像到内存
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        
        # 转换为base64
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        plt.close()
        
        return {
            "status": "success",
            "image_base64": img_base64,
            "format": "png",
            "message": "成功生成NMR谱图"
        }
        
    except Exception as e:
        return {"status": "error", "message": f"绘制谱图失败: {str(e)}"}

if __name__ == "__main__":
    # 从环境变量获取传输类型，默认为SSE
    transport_type = os.getenv('MCP_TRANSPORT', 'sse')
    mcp.run(transport=transport_type)