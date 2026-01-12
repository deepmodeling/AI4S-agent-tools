#!/usr/bin/env python3
"""
ASKCOS MCP Server for computer-aided synthesis planning.
Provides retrosynthesis planning, synthesis route analysis, and reaction prediction tools.
"""
import argparse
import os
import json
import logging
from typing import Dict, List, Optional, Any

import requests
from mcp.server.fastmcp import FastMCP

def parse_args():
    """Parse command line arguments for MCP server."""
    parser = argparse.ArgumentParser(description="ASKCOS MCP Server")
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

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

args = parse_args()
mcp = FastMCP("ASKCOS", host=args.host, port=args.port)

# 从环境变量获取 ASKCOS API URL，使用正确的默认端口9100
ASKCOS_API_URL = os.environ.get("ASKCOS_API_URL", "http://localhost:9100")
logger.info(f"ASKCOS API URL: {ASKCOS_API_URL}")

@mcp.tool()
def retrosynthesis_planning(
    smiles: str, 
    max_steps: int = 5, 
    top_n: int = 1,
    template_set: str = "reaxys",
    max_branching: int = 25
) -> Dict[str, Any]:
    """执行逆合成规划，为给定的目标分子生成合成路线。"""
    logger.info(f"执行逆合成规划: {smiles}")
    
    try:
        # 使用正确的ASKCOS v2 API格式
        payload = {
            "smiles": smiles,
            "max_depth": max_steps,
            "max_branching": max_branching,
            "expansion_time": 60,
            "max_ppg": top_n,
            "template_count": 1000,
            "max_cum_prob": 0.999,
            "chemical_property_logic": "none",
            "max_chemprop_c": 0,
            "max_chemprop_n": 0,
            "max_chemprop_o": 0,
            "max_chemprop_h": 0,
            "chemical_popularity_logic": "none",
            "min_chempop_reactants": 5,
            "min_chempop_products": 5,
            "filter_threshold": 0.1,
            "return_first": "true"
        }
        
        response = requests.post(
            f"{ASKCOS_API_URL}/api/tree-search/mcts/call-sync-without-token",
            json=payload,
            timeout=300
        )
        response.raise_for_status()
        
        result = response.json()
        logger.info(f"逆合成规划成功完成: {smiles}")
        
        return {
            "success": True,
            "smiles": smiles,
            "trees": result.get("trees", []),
            "num_trees": len(result.get("trees", [])),
            "parameters": payload
        }
        
    except requests.exceptions.RequestException as e:
        error_msg = f"逆合成规划请求失败: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "smiles": smiles
        }

@mcp.tool()
def reaction_prediction(
    reactants_smiles: str, 
    reagents_smiles: Optional[str] = None, 
    solvent_smiles: Optional[str] = None,
    temperature: Optional[float] = None
) -> Dict[str, Any]:
    """预测给定反应物和试剂的反应产物。"""
    logger.info(f"执行反应预测: {reactants_smiles}")
    
    try:
        # 使用正确的ASKCOS v2 forward prediction API格式
        payload = {
            "reactants": reactants_smiles,
            "reagents": reagents_smiles or "",
            "solvent": solvent_smiles or "",
            "top_k": 10,
            "threshold": 0.1
        }
        
        if temperature is not None:
            payload["temperature"] = temperature
        
        response = requests.post(
            f"{ASKCOS_API_URL}/api/forward/call-sync-without-token",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        
        result = response.json()
        logger.info(f"反应预测成功完成: {reactants_smiles}")
        
        return {
            "success": True,
            "reactants": reactants_smiles,
            "products": result.get("products", []),
            "scores": result.get("scores", []),
            "conditions": {
                "reagents": reagents_smiles,
                "solvent": solvent_smiles,
                "temperature": temperature
            }
        }
        
    except requests.exceptions.RequestException as e:
        error_msg = f"反应预测请求失败: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "reactants": reactants_smiles
        }

@mcp.tool()
def single_step_retrosynthesis(
    smiles: str,
    num_templates: int = 1000,
    max_cum_prob: float = 0.995,
    top_k: int = 10
) -> Dict[str, Any]:
    """执行单步逆合成分析。"""
    logger.info(f"执行单步逆合成: {smiles}")
    
    try:
        payload = {
            "smiles": smiles,
            "num_templates": num_templates,
            "max_cum_prob": max_cum_prob,
            "top_k": top_k
        }
        
        response = requests.post(
            f"{ASKCOS_API_URL}/api/retro/call-sync-without-token",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        
        result = response.json()
        logger.info(f"单步逆合成成功完成: {smiles}")
        
        return {
            "success": True,
            "smiles": smiles,
            "precursors": result.get("precursors", []),
            "scores": result.get("scores", []),
            "parameters": payload
        }
        
    except requests.exceptions.RequestException as e:
        error_msg = f"单步逆合成请求失败: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "smiles": smiles
        }

@mcp.tool()
def synthesis_route_analysis(route_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    分析给定的合成路线，评估其可行性和复杂度。
    注意：此功能需要ASKCOS服务器支持路线分析API。
    
    Args:
        route_json: 合成路线的JSON表示
        
    Returns:
        包含路线分析结果的字典
    """
    logger.info("执行合成路线分析")
    
    try:
        # 注意：ASKCOS v2可能没有专门的路线分析API端点
        # 这里提供一个基本的路线复杂度分析
        def analyze_route_complexity(route):
            """简单的路线复杂度分析"""
            if not isinstance(route, dict):
                return {"complexity_score": 0, "feasibility_score": 0}
            
            # 计算路线深度和分支数
            def count_steps(node, depth=0):
                if not isinstance(node, dict) or 'children' not in node:
                    return depth
                max_depth = depth
                for child in node.get('children', []):
                    child_depth = count_steps(child, depth + 1)
                    max_depth = max(max_depth, child_depth)
                return max_depth
            
            depth = count_steps(route)
            complexity_score = min(100, depth * 10)  # 简单的复杂度评分
            feasibility_score = max(0, 100 - complexity_score)  # 可行性与复杂度成反比
            
            return {
                "complexity_score": complexity_score,
                "feasibility_score": feasibility_score,
                "route_depth": depth
            }
        
        analysis_result = analyze_route_complexity(route_json)
        logger.info("合成路线分析成功完成")
        
        return {
            "success": True,
            "analysis": analysis_result,
            "route_complexity": analysis_result.get("complexity_score", "未知"),
            "feasibility": analysis_result.get("feasibility_score", "未知"),
            "note": "此分析基于本地算法，如需更详细分析请使用ASKCOS Web界面"
        }
        
    except Exception as e:
        error_msg = f"合成路线分析过程中发生错误: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg
        }


if __name__ == "__main__":
    # Get transport type from environment variable, default to SSE
    transport_type = os.getenv('MCP_TRANSPORT', 'sse')
    mcp.run(transport=transport_type)