"""
MCP server interface for Composition DART optimization.

Strict interface adapter: validates tool input into OptimizationRequest,
delegates to endpoints.run_optimization, returns result. No business logic.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any, Dict, List, Optional

from dp.agent.server import CalculationMCPServer

sys.path.insert(0, "/mcp_server/comp-dart-gitlab")
from comp_dart.api.endpoints import run_optimization
from comp_dart.api.schemas import OptimizationRequest


def parse_args() -> Any:
    """Parse command line arguments for MCP server."""
    parser = argparse.ArgumentParser(description="Composition DART MCP Server")
    parser.add_argument("--port", type=int, default=50001, help="Server port (default: 50001)")
    parser.add_argument("--host", default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    try:
        return parser.parse_args()
    except SystemExit:

        class Args:
            port = 50001
            host = "0.0.0.0"
            log_level = "INFO"

        return Args()


args = parse_args()
mcp = CalculationMCPServer("DPACalculatorServer", host=args.host, port=args.port)


@mcp.tool()
def run_dart_ga(request: OptimizationRequest) -> Dict:
    """
    Run genetic algorithm for composition optimization.

    Accepts elements, targets (list of target config dicts), structure_config (dict),
    optional constraints (list of constraint dicts), and GA parameters. Input is
    validated into OptimizationRequest and delegated to the optimization backend.

    Returns a dict with best_individual, pred_{name}_mean/std per target, and best_score.
    """
    
    return run_optimization(request)


if __name__ == "__main__":
    logging.info("Starting Composition DART MCP Server...")
    mcp.run(transport="streamable-http")