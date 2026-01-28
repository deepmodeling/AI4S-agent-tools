"""
MCP server interface for Composition DART optimization.

Tool run_dart_ga uses explicit parameters; targets, structure_config, constraints
are Pydantic types. File-related fields (output, targets[].model_path) use Path
so the SDK can resolve OSS links.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Literal, Optional, Union

sys.path.append("/mcp_server/comp-dart-gitlab")

from dp.agent.server import CalculationMCPServer

from comp_dart.api.schemas import (
    RunDartGAArgs,
    TargetConfigInArgs,
    StructureConfigInArgs,
    ConstraintConfig,
)
from comp_dart.api.conv import run_dart_ga_args_to_legacy
from comp_dart.api.endpoints import run_optimization


def parse_args():
    """Parse command line arguments for MCP server."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=50001)
    parser.add_argument("--host", default="0.0.0.0")
    return parser.parse_args()


args = parse_args()
mcp = CalculationMCPServer("DPACalculatorServer", host=args.host, port=args.port)


@mcp.tool()
def run_dart_ga(
    elements: List[str],
    targets: List[TargetConfigInArgs],
    structure_config: StructureConfigInArgs,
    population_size: int = 10,
    generations: int = 10,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.1,
    selection_mode: Literal["roulette", "tournament"] = "roulette",
    output: Union[Path, str] = Path("ga_run.log"),
    constraints: Optional[List[ConstraintConfig]] = None,
) -> dict:
    """
    Run genetic algorithm for composition optimization.

    Args:
        elements: Element symbols (e.g. ['Fe','Ni','Co','V']).
        population_size: GA population size.
        generations: GA generations.
        crossover_rate: Crossover rate.
        mutation_rate: Mutation rate.
        selection_mode: 'roulette' or 'tournament'.
        output: Output file path (Path for SDK/OSS).
        targets: List of target configs (name, type, model_path/data_source, weight_mean, weight_std, normalization).
        structure_config: mode, template_path, supercell.
        constraints: Optional list of {target, condition}.
    """
    output_path = Path(output) if isinstance(output, str) else output
    args_model = RunDartGAArgs(
        elements=elements,
        population_size=population_size,
        generations=generations,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        selection_mode=selection_mode,
        output=output_path,
        targets=targets,
        structure_config=structure_config,
        constraints=constraints,
    )
    problem, algorithm, structure, model_files, template_path, output_path = (
        run_dart_ga_args_to_legacy(args_model)
    )
    return run_optimization(
        problem=problem,
        algorithm=algorithm,
        structure=structure,
        model_files=model_files,
        template_file=template_path,
        output_file=output_path,
    )


if __name__ == "__main__":
    logging.info("Starting DART Server...")
    mcp.run(transport="streamable-http")
