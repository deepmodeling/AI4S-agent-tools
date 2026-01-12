#!/usr/bin/env python3
"""
RDKit MCP 服务器
提供基础的化学信息学功能，包括分子处理、描述符计算和可视化。
"""
import argparse
import os
from mcp.server.fastmcp import FastMCP
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Draw
import numpy as np

def parse_args():
    """解析MCP服务器的命令行参数。"""
    parser = argparse.ArgumentParser(description="RDKit MCP 服务器")
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
mcp = FastMCP("RDKit Toolkit", host=args.host, port=args.port)

# --- 辅助函数 ---
def smiles_to_mol(smiles: str):
    """安全地将SMILES转换为RDKit Mol对象。"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"无效的SMILES字符串: {smiles}")
    return mol

# 定义工具函数
@mcp.tool()
def validate_smiles(smiles: str) -> bool:
    """
    检查SMILES字符串是否化学有效。
    
    Args:
        smiles: 要验证的SMILES字符串
        
    Returns:
        如果SMILES有效返回True，否则返回False
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    return mol is not None

@mcp.tool()
def get_basic_properties(smiles: str) -> dict:
    """
    计算给定SMILES的基础分子性质。
    性质包括：分子量、LogP、TPSA、氢键供体数、氢键受体数。
    
    Args:
        smiles: 分子的SMILES字符串
        
    Returns:
        包含计算性质的字典
    """
    mol = smiles_to_mol(smiles)
    properties = {
        "molecular_weight": Descriptors.MolWt(mol),
        "logp": Descriptors.MolLogP(mol),
        "tpsa": Descriptors.TPSA(mol),
        "hydrogen_bond_donors": Descriptors.NumHDonors(mol),
        "hydrogen_bond_acceptors": Descriptors.NumHAcceptors(mol),
    }
    return properties

@mcp.tool()
def generate_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> list:
    """
    为分子生成Morgan指纹。
    
    Args:
        smiles: 分子的SMILES字符串
        radius: Morgan指纹的半径
        n_bits: 指纹向量的位数
        
    Returns:
        表示指纹的整数列表
    """
    mol = smiles_to_mol(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return list(fp.GetOnBits())

@mcp.tool()
def substructure_search(smiles: str, smarts_pattern: str) -> bool:
    """
    使用SMARTS模式对分子进行子结构搜索。
    
    Args:
        smiles: 要搜索的分子的SMILES字符串
        smarts_pattern: 要搜索的模式的SMARTS字符串
        
    Returns:
        如果找到子结构返回True，否则返回False
    """
    mol = smiles_to_mol(smiles)
    pattern = Chem.MolFromSmarts(smarts_pattern)
    if pattern is None:
        raise ValueError(f"无效的SMARTS模式: {smarts_pattern}")
    return mol.HasSubstructMatch(pattern)

@mcp.tool()
def draw_molecule_svg(smiles: str) -> str:
    """
    生成分子的2D SVG图像。
    
    Args:
        smiles: 分子的SMILES字符串
        
    Returns:
        包含SVG图像数据的字符串
    """
    mol = smiles_to_mol(smiles)
    svg = Draw.MolToSVG(mol)
    return svg

if __name__ == "__main__":
    # 从环境变量获取传输类型，默认为SSE
    transport_type = os.getenv('MCP_TRANSPORT', 'sse')
    mcp.run(transport=transport_type)