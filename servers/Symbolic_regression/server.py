#!/usr/bin/env python3
"""
Symbolic Regression MCP Server using PySR.
This server provides symbolic regression capabilities to discover mathematical expressions from data.
"""
import argparse
import os
from mcp.server.fastmcp import FastMCP
from src.symbolic_regression import run_symbolic_pysr


def parse_args():
    """Parse command line arguments for MCP server."""
    parser = argparse.ArgumentParser(description="Symbolic Regression MCP Server")
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
mcp = FastMCP("symbolic-regression", host=args.host, port=args.port)


@mcp.tool()
def symbolic_regression(csv_path: str, unary_operators: list) -> dict:
    """
    Discover mathematical expressions from data using symbolic regression.
    
    This tool uses genetic programming to automatically find mathematical formulas that best 
    describe the relationships in your data. It's particularly useful for:
    - Discovering scientific laws from experimental data
    - Finding interpretable models for complex phenomena
    - Reverse engineering mathematical relationships
    - Creating explainable predictive models
    
    The tool assumes your CSV has multiple input columns and the last column is the target
    variable you want to predict.
    
    Args:
        csv_path: Path to CSV file containing your data. Format requirements:
                  - Multiple columns for input variables (features)
                  - Last column is the target variable to predict
                  - No missing values (NaN)
                  - Numeric data only
                  
        unary_operators: Optional list of unary operators to use in expressions.
                        Default: ["neg", "square", "cube", "exp", "log", "sqrt", "abs", "sin", "cos"]
                        Available operators:
                        - "neg": Negation (-x)
                        - "square": Square (x²)
                        - "cube": Cube (x³)
                        - "cbrt": Cube root (∛x)
                        - "inv": Inverse (1/x)
                        - "exp": Exponential (e^x)
                        - "log": Natural logarithm (ln(x))
                        - "sqrt": Square root (√x)
                        - "abs": Absolute value (|x|)
                        - "sin": Sine
                        - "cos": Cosine
                        - "tan": Tangent
    
    Returns:
        Dictionary containing:
        - best_result: The best expression found with its complexity and mean squared error
        - candidates: List of candidate expressions sorted by performance
        - metadata: Information about the regression run including timestamp and parameters
        
    Example:
        # Discover formula for pendulum period from experimental data
        result = symbolic_regression(
            csv_path="pendulum_data.csv",
            unary_operators=["sqrt"]
        )
        # Might discover: T = 2π√(L/g)
        
        # Find relationship between pressure, volume, and temperature
        result = symbolic_regression(
            csv_path="gas_law_data.csv",
            unary_operators=["inv"]
        )
        # Might discover: P = nRT/V
        
    Notes:
        - Results are saved to 'output/best.txt' and 'output/results.json'
        - The tool uses binary operators: +, -, *, /
        - Maximum expression complexity is limited to prevent overfitting
        - Multiple candidate expressions are returned, ordered by accuracy
    """
    return run_symbolic_pysr(csv_path, unary_operators)


if __name__ == "__main__":
    # Get transport type from environment variable, default to SSE
    transport_type = os.getenv('MCP_TRANSPORT', 'sse')
    mcp.run(transport=transport_type)