"""
MCP server interface for Composition DART optimization.

Exposes a single tool run_dart_ga for running a genetic algorithm over composition
space. All tool parameters use explicit types; targets, structure_config, and
constraints are Pydantic models. The output path is str; targets[].model_path and
structure_config.template_path accept Path or str so the SDK can resolve OSS
links to local paths.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Literal, Optional

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
    output: str = "ga_run.log",
    init_population: Optional[List[List[float]]] = None,
    constraints: Optional[List[ConstraintConfig]] = None,
) -> dict:
    """
    Run genetic algorithm for composition optimization.

    Optimizes chemical compositions over a given set of elements and target
    properties (surrogate models or linear mixture). Uses structure config for
    template/supercell and optional composition constraints. Returns a result
    dict with optimization outcomes and metadata.

    Args:
        elements (List[str]): Element symbols defining the composition space. At
            least two required (e.g. ['Fe', 'Ni', 'Co', 'V']). Order determines
            composition order in outputs.
        targets (List[TargetConfigInArgs]): List of target configurations. Each
            item: name (str), type (Literal['surrogate','linear_mixture']), and
            either model_path (Path | str for surrogate) or data_source
            (Literal['density','atomic_mass','custom'] for linear_mixture).
            Optional fields:
            - weight_mean (float, default 1.0): Weight for the predicted mean
              in the fitness function. The GA MAXIMIZES fitness = sum(weight *
              value). Therefore: positive weight_mean → MAXIMIZE the property;
              negative weight_mean → MINIMIZE the property. For example, to
              minimize density, set weight_mean=-1.0.
            - weight_std (float, default 0.0): Weight for prediction
              uncertainty. Negative → penalize uncertainty (prefer confident
              predictions); 0.0 → ignore uncertainty.
            - normalization (NormalizationInArgs): Optional z-score
              normalization config.
            Path/str for model_path allows SDK to resolve OSS links.
        structure_config (StructureConfigInArgs): Structure generation config.
            mode: Literal['template','auto']; template_path: Path | str |
            Literal['fcc','bcc','hcp']; supercell: Optional[List[int]], default
            [5,5,5]. Path/str for template_path allows SDK to resolve OSS links.
        population_size (int): Number of individuals per generation (default
            10, >=2).
        generations (int): Number of GA generations (default 10, >=1).
        crossover_rate (float): Crossover probability in [0,1] (default 0.8).
        mutation_rate (float): Mutation probability in [0,1] (default 0.1).
        selection_mode (Literal['roulette','tournament']): Parent selection mode
            (default 'roulette').
        output (str): Output log file path string (default 'ga_run.log').
            This value is converted to pathlib.Path in the args conversion layer.
        init_population (Optional[List[List[float]]]): Optional initial guess
            compositions to seed the GA population. Each inner list is one
            composition whose values correspond to 'elements' in order and
            should sum to 1.0. Example for 4 elements:
            [[0.25, 0.25, 0.25, 0.25]]. If fewer compositions are provided
            than population_size, remaining slots are filled randomly. Use
            this when the user supplies a starting composition or you already
            know a promising region of the search space.
        constraints (Optional[List[ConstraintConfig]]): Optional list of
            composition constraints. Each item has two fields:
            - target (str | List[str]): Element(s) to constrain. A single
              element symbol (e.g. 'Fe') constrains that element's mole
              fraction; a list (e.g. ['Fe','Ni']) constrains the sum of those
              elements' mole fractions.
            - condition (str): Expression in format 'operator value', e.g.
              '<0.5', '>=0.1', '=0.3'. Supported operators: >=, <=, >, <, =.
              The value represents a mole fraction and MUST be in [0, 1].
              Values > 1 (e.g. 30 meaning 30 %) are NOT auto-normalized;
              pass 0.3 instead of 30. Out-of-range values will silently
              produce meaningless constraints.

    Returns:
        dict: Optimization result with keys including run metadata, best
        composition(s), fitness values, and any algorithm-specific outputs
        from run_optimization.
    """
    args_model = RunDartGAArgs(
        elements=elements,
        population_size=population_size,
        generations=generations,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        selection_mode=selection_mode,
        output=output,
        targets=targets,
        structure_config=structure_config,
        init_population=init_population,
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
