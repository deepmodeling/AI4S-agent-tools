#!/usr/bin/env python3
"""
MCP Server for dna_sequence_analyzer
Generated following AI4S-agent-tools CONTRIBUTING.md standards.
"""
import argparse
import os
from typing import Dict, Any
from mcp.server.fastmcp import FastMCP
from Bio.Seq import Seq
from Bio.Data import CodonTable
import numpy as np

def parse_args():
    """Parse command line arguments for MCP server."""
    parser = argparse.ArgumentParser(description="dna_sequence_analyzer MCP Server")
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
mcp = FastMCP("dna_sequence_analyzer", host=args.host, port=args.port)

@mcp.tool()
def analyze_dna_sequence(sequence: str, sequence_name: str = "") -> dict:
    """
    分析DNA序列的基本特征，包括长度、碱基计数、GC含量等。
    
    Args:
        sequence: DNA序列字符串（只包含ATCG字符）
        sequence_name: 可选序列名称，用于结果标识
        
    Returns:
        dict: 包含序列分析结果的字典
    """
    try:
        # 验证序列有效性
        seq_obj = Seq(sequence.upper())
        valid_bases = set("ATCG")
        
        if not all(base in valid_bases for base in seq_obj):
            return {"error": "序列包含非ATCG字符"}
        
        # 计算基本统计
        total_length = len(seq_obj)
        base_counts = {
            "A": seq_obj.count("A"),
            "T": seq_obj.count("T"),
            "C": seq_obj.count("C"),
            "G": seq_obj.count("G")
        }
        
        # 计算GC含量
        gc_count = base_counts["G"] + base_counts["C"]
        gc_content = (gc_count / total_length) * 100 if total_length > 0 else 0
        
        # 计算AT含量
        at_count = base_counts["A"] + base_counts["T"]
        at_content = (at_count / total_length) * 100 if total_length > 0 else 0
        
        # 碱基比例
        base_ratios = {base: count/total_length for base, count in base_counts.items()}
        
        result = {
            "sequence_name": sequence_name if sequence_name else "unnamed_sequence",
            "sequence_length": total_length,
            "base_counts": base_counts,
            "gc_content": round(gc_content, 2),
            "at_content": round(at_content, 2),
            "base_ratios": {k: round(v, 4) for k, v in base_ratios.items()},
            "sequence": str(seq_obj)
        }
        
        return result
        
    except Exception as e:
        return {"error": f"序列分析失败: {str(e)}"}

@mcp.tool()
def reverse_complement(sequence: str) -> dict:
    """
    生成DNA序列的反向互补序列。
    
    Args:
        sequence: DNA序列字符串
        
    Returns:
        dict: 包含原始序列和反向互补序列的信息
    """
    try:
        # 验证序列有效性
        seq_obj = Seq(sequence.upper())
        valid_bases = set("ATCG")
        
        if not all(base in valid_bases for base in seq_obj):
            return {"error": "序列包含非ATCG字符"}
        
        # 生成反向互补序列
        reverse_comp = seq_obj.reverse_complement()
        
        result = {
            "original_sequence": str(seq_obj),
            "reverse_complement": str(reverse_comp),
            "sequence_length": len(seq_obj)
        }
        
        return result
        
    except Exception as e:
        return {"error": f"反向互补序列生成失败: {str(e)}"}

@mcp.tool()
def translate_dna(sequence: str, stop_at_stop_codon: bool = True) -> dict:
    """
    将DNA序列翻译为蛋白质序列，使用标准遗传密码表。
    
    Args:
        sequence: DNA序列字符串
        stop_at_stop_codon: 遇到终止密码子是否停止翻译
        
    Returns:
        dict: 包含翻译结果的信息
    """
    try:
        # 验证序列有效性
        seq_obj = Seq(sequence.upper())
        valid_bases = set("ATCG")
        
        if not all(base in valid_bases for base in seq_obj):
            return {"error": "序列包含非ATCG字符"}
        
        if len(seq_obj) < 3:
            return {"error": "序列长度不足3个碱基，无法翻译"}
        
        # 获取标准遗传密码表
        standard_table = CodonTable.standard_dna_table
        
        # 翻译序列
        protein_seq = seq_obj.translate(table=standard_table, stop_symbol="*", to_stop=stop_at_stop_codon)
        
        # 统计信息
        protein_length = len(protein_seq)
        stop_codons = protein_seq.count("*")
        
        result = {
            "dna_sequence": str(seq_obj),
            "protein_sequence": str(protein_seq),
            "protein_length": protein_length,
            "stop_codons": stop_codons,
            "stop_at_stop_codon": stop_at_stop_codon,
            "genetic_code": "standard"
        }
        
        return result
        
    except Exception as e:
        return {"error": f"序列翻译失败: {str(e)}"}

if __name__ == "__main__":
    # Get transport type from environment variable, default to SSE
    transport_type = os.getenv('MCP_TRANSPORT', 'sse')
    mcp.run(transport=transport_type)
